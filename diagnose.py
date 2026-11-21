"""Why does the AI Hub-trained model collapse on own photos? Perturb the in-distribution val set one factor
at a time (object scale, background color, resolution, brightness) and see which one reproduces the drop.

usage: python diagnose.py runs/<name>/best.pt
"""
import sys
import torch
import torchvision
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2 as T

from train import CLASSES, MEAN_U8, ROOT, gpu_norm, load_index, split

ck = torch.load(sys.argv[1], map_location="cuda", weights_only=False)
model = getattr(torchvision.models, ck["arch"])(num_classes=4)
model.load_state_dict(ck["model"])
model = model.cuda().to(memory_format=torch.channels_last).eval()
_, val = split(load_index(ROOT), 0.1)
SRC = torch.tensor([0 if g.startswith("140_") else 1 for _, _, g in val])  # 0 = 140 (outdoor), 1 = 71385 (indoor sorter)


def make_tf(frac=1.0, bg=MEAN_U8, lowres=0, bright=1.0, size=224, pre=None):
    """frac: object crop's long side as a fraction of the square canvas side. bg: canvas color.
    lowres: shrink to this size then back up (0 = off). bright: pixel value multiplier."""
    def f(im):
        if pre: im = pre(im)
        w, h = im.size
        s = round(max(w, h) / frac)
        c = Image.new("RGB", (s, s), bg)
        c.paste(im, ((s - w) // 2, (s - h) // 2))
        c = c.resize((size, size), Image.BILINEAR, reducing_gap=3.0)
        if lowres: c = c.resize((lowres, lowres), Image.BILINEAR, reducing_gap=3.0).resize((size, size), Image.BILINEAR)
        x = T.functional.pil_to_tensor(c)
        return (x.float() * bright).clamp(0, 255).byte() if bright != 1.0 else x
    return f


class DS(Dataset):
    def __init__(self, items, tf, exif=False): self.items, self.tf, self.exif = items, tf, exif
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        im = Image.open(self.items[i][0])
        if self.exif: im = ImageOps.exif_transpose(im)
        return self.tf(im.convert("RGB")), self.items[i][1]


def run(items, tf, exif=False):
    P, Y = [], []
    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
        for x, y in DataLoader(DS(items, tf, exif), 256, num_workers=16):
            P.append(model(gpu_norm(x, "cuda")).argmax(1).cpu()); Y.append(y)
    return torch.cat(P), torch.cat(Y)


def macro(p, y, mask=None):
    if mask is not None: p, y = p[mask], y[mask]
    return sum((p[y == k] == k).float().mean().item() for k in range(4)) / 4


def row(name, **kw):
    p, y = run(val, make_tf(**kw))
    per = " ".join(f"{c} {(p[y == k] == k).float().mean():.2f}" for k, c in enumerate(CLASSES))
    print(f"  {name:44s} macro {macro(p, y):.3f} | 140 {macro(p, y, SRC == 0):.3f}  71385 {macro(p, y, SRC == 1):.3f} | {per}"
          f" | pred share " + " ".join(f"{(p == k).float().mean():.2f}" for k in range(4)), flush=True)


print(f"val n={len(val)} (140: {(SRC == 0).sum().item()}, 71385: {(SRC == 1).sum().item()})")
print("[A] object scale: crop pasted on a mean-color canvas (mean color = 0 after normalize)")
for f in (1.0, 0.7, 0.5, 0.35, 0.25): row(f"object long side = {f:.0%} of frame", frac=f)
print("[B] background color (object at 100% = only letterbox bars change; at 50% = surroundings change)")
for name, bg in (("black", (0, 0, 0)), ("dark gray 60", (60,) * 3), ("light gray 190", (190,) * 3), ("white", (255,) * 3), ("wood brown", (150, 110, 70)), ("grass green", (70, 120, 50))):
    row(f"100% frame, {name} bars", bg=bg); row(f" 50% frame, {name} background", frac=0.5, bg=bg)
print("[C] resolution: shrink to N px, enlarge back to 224")
for r in (112, 64, 48, 32): row(f"detail of a {r}px image", lowres=r)
print("[D] brightness: pixel values x k")
for b in (0.5, 1.5, 2.0, 3.0): row(f"brightness x{b}", bright=b)
print("[E] combined: 50% frame + white background + brightness x1.5")
row("50%, white, x1.5", frac=0.5, bg=(255,) * 3, bright=1.5)

# [F] how bright / how big are the images themselves? train vs own photos
import random
def lum(path, exif=False):
    im = Image.open(path); im = ImageOps.exif_transpose(im) if exif else im
    w, h = im.size; im = im.convert("L"); im.thumbnail((64, 64))
    px = torch.tensor(list(im.getdata()), dtype=torch.float)
    return px.mean().item(), min(w, h), max(w, h)
tr, _ = split(load_index(ROOT), 0.1)
samp = random.Random(0).sample(tr, 4000)
test = [(str(p), k) for k, c in enumerate(CLASSES) for p in sorted((ROOT / "test" / c).iterdir()) if p.suffix != ".csv"]
def q(xs): xs = sorted(xs); n = len(xs); return f"p10 {xs[n // 10]:.0f}  median {xs[n // 2]:.0f}  p90 {xs[9 * n // 10]:.0f}"
print("[F] mean image brightness (0-255)")
for name, sel in (("train 140 (outdoor)", [x for x in samp if x[2].startswith("140_")]), ("train 71385 (indoor sorter)", [x for x in samp if x[2].startswith("71385_")])):
    print(f"  {name:28s} {q([lum(p)[0] for p, *_ in sel])}  (n={len(sel)})")
tl = [lum(p, True) for p, _ in test]
print(f"  {'own photos (test)':28s} {q([t[0] for t in tl])}  (n={len(tl)})")
print(f"  own photos original short side: {q([t[1] for t in tl])}")

# [G] own photos: accuracy by brightness and by original resolution, eval preprocessing = center 70% + 320
def c70(im):
    w, h = im.size; return im.crop((int(w * .15), int(h * .15), int(w * .85), int(h * .85)))
p, y = run(test, make_tf(size=320, pre=c70), exif=True)
print(f"[G] own photos, center 70% + 320: macro {macro(p, y):.3f}")
L = torch.tensor([t[0] for t in tl]); S = torch.tensor([float(t[1]) for t in tl])
for name, v in (("brightness", L), ("original short side px", S)):
    qs = torch.quantile(v, torch.tensor([1 / 3, 2 / 3]))
    for lab, m in (("low third", v <= qs[0]), ("mid third", (v > qs[0]) & (v <= qs[1])), ("high third", v > qs[1])):
        print(f"  {name:24s} {lab:10s} ({v[m].min():.0f}-{v[m].max():.0f}): n={int(m.sum())} acc {(p[m] == y[m]).float().mean():.3f}")
# does low resolution hurt own photos? shrink every photo to N px short side first
for r in (112, 224, 448):
    def pre(im, r=r):
        im = c70(im); s = r / min(im.size)
        return im.resize((max(1, round(im.width * s)), max(1, round(im.height * s))), Image.BILINEAR) if s < 1 else im
    p2, _ = run(test, make_tf(size=320, pre=pre), exif=True)
    print(f"  own photos first shrunk to <= {r}px short side: macro {macro(p2, y):.3f}")
