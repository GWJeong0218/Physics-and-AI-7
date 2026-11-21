"""Train a ResNet from scratch on the 4-class waste data (vinyl, glass, plastic, can).

Data: <root>/{vinyl,glass,plastic,can} + manifest.csv from prepare_data.py (AI Hub only), split into train/val by `group`
so repeat shots of one object never land on both sides. Test set = own photos (not used here).

examples:
  python train.py --subset 20000 --epochs 30 --name r18_sub     # quick hyperparameter run
  python train.py --epochs 100 --name r18_full                  # full run
  python train.py --resume runs/r18_full/last.pt                # settings come from the run's config.json
"""
import argparse
import csv
import hashlib
import json
import math
import os
import random
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
import torchvision
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2 as T

CLASSES = ("vinyl", "glass", "plastic", "can")
ROOT = Path(__file__).resolve().parent
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def src(group):
    """Data source of a group: "140" (incl. the later-added Training split, prefix 140tr) or "71385"."""
    p = group.split("_")[0]
    return "140" if p == "140tr" else p


def load_index(root):
    """[(path, class index, group)] from <root>/manifest.csv (written by prepare_data.py)."""
    with open(root / "manifest.csv", encoding="utf-8-sig") as f:
        return [(str(root / r["path"]), CLASSES.index(r["class"]), r["group"]) for r in csv.DictReader(f)]


def session(group):
    """71385 names are 71385_<batch>_<date>_<frame>; batch+date is one capture session. None for other sources."""
    p = group.split("_")
    return "_".join(p[:3]) if p[0] == "71385" else None


def capture_val_sessions(items, val_frac):
    """Whole 71385 capture sessions held out for val. Sessions are taken in hash order until val holds
    val_frac of the 71385 crops and at least val_frac/2 of each class (sessions are class-skewed)."""
    per = {}
    for _, y, g in items:
        s = session(g)
        if s:
            per.setdefault(s, [0] * len(CLASSES))[y] += 1
    total = [sum(c[k] for c in per.values()) for k in range(len(CLASSES))]
    chosen, got = set(), [0] * len(CLASSES)
    for s in sorted(per, key=lambda s: hashlib.md5(s.encode()).hexdigest()):
        if sum(got) >= val_frac * sum(total) and all(g >= val_frac / 2 * t for g, t in zip(got, total)):
            break
        chosen.add(s)
        got = [a + b for a, b in zip(got, per[s])]
    return chosen


def split(items, val_frac, mode="object"):
    """Deterministic group split: same group -> same side, independent of seed and item order.
    mode "capture": 71385 is split by whole capture session instead of single image; 140 stays per object."""
    held = capture_val_sessions(items, val_frac) if mode == "capture" else None
    def is_val(g):
        if g.startswith("140tr_"):  # 140 Training split, added later: train only, so val stays identical
            return False
        if held is not None and session(g):
            return session(g) in held
        return int(hashlib.md5(g.encode()).hexdigest(), 16) % 10_000 < val_frac * 10_000
    tr = [x for x in items if not is_val(x[2])]
    va = [x for x in items if is_val(x[2])]
    assert not {x[2] for x in tr} & {x[2] for x in va}, "group leaked across train/val"
    return tr, va


class Images(Dataset):
    def __init__(self, items, tf, size):
        self.items, self.tf, self.size = items, tf, size

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, y, _ = self.items[i]
        im = Image.open(path)
        im.draft("RGB", (self.size * 2, self.size * 2))  # JPEG: decode at reduced scale, ~3x faster
        return self.tf(im.convert("RGB")), y


MEAN_U8 = tuple(round(m * 255) for m in MEAN)


def pad_square(im):
    """Letterbox to a square with the mean color (0 after normalize), so elongated objects
    (bottles: 27% of glass has aspect > 2) keep their neck and base instead of being center-cropped."""
    w, h = im.size
    if w == h:
        return im
    s = max(w, h)
    bg = Image.new("RGB", (s, s), MEAN_U8)
    bg.paste(im, ((s - w) // 2, (s - h) // 2))
    return bg


class RandomCanvas:
    """Strong-aug replacement for pad_square: paste the object crop at a random spot on a random-color
    square canvas up to `max_scale` times larger. AI Hub crops are tight boxes on dark sorter / outdoor
    ground; own photos show the object smaller, off-center, on bright backgrounds."""

    def __init__(self, max_scale=2.0, p=0.7):
        self.max_scale, self.p = max_scale, p

    def __call__(self, im):
        im.thumbnail((320, 320))  # bounds canvas size (and CPU cost); output is 224 anyway
        w, h = im.size
        s = max(w, h)
        if torch.rand(1).item() < self.p:
            s = int(s * (1 + torch.rand(1).item() * (self.max_scale - 1)))
        bg = Image.new("RGB", (s, s), tuple(torch.randint(0, 256, (3,)).tolist()))
        bg.paste(im, (int(torch.randint(0, s - w + 1, (1,))), int(torch.randint(0, s - h + 1, (1,)))))
        return bg


def save_atomic(obj, path):
    """Write to a temp file then rename, so a crash mid-save never corrupts the checkpoint."""
    tmp = Path(str(path) + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def transforms(size, aug="base"):
    """CPU side stays uint8; float conversion + normalize run on GPU (gpu_norm). Profiling showed
    float+Normalize cost ~5x TrivialAugment on CPU, and uint8 batches are 4x smaller to transfer."""
    mean_u8 = list(MEAN_U8)  # erased patch = mean color = 0 after normalize, as before
    train = T.Compose([pad_square, T.RandomResizedCrop(size, scale=(0.35, 1.0), antialias=True),
                       T.RandomHorizontalFlip(), T.TrivialAugmentWide(), T.ToImage(),
                       T.RandomErasing(0.25, value=mean_u8)])
    if aug == "strong":
        # crop scale >= 0.6 of the canvas so the object is rarely cropped out entirely
        train = T.Compose([RandomCanvas(), T.RandomResizedCrop(size, scale=(0.6, 1.0), antialias=True),
                           T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
                           T.ColorJitter(0.4, 0.4, 0.4, 0.05), T.RandomGrayscale(0.1), T.TrivialAugmentWide(),
                           T.ToImage(), T.RandomApply([T.GaussianBlur(5, (0.1, 2.0))], p=0.2),
                           T.RandomErasing(0.25, value=mean_u8)])
    if aug == "light":  # base without TrivialAugment and erasing
        train = T.Compose([pad_square, T.RandomResizedCrop(size, scale=(0.35, 1.0), antialias=True),
                           T.RandomHorizontalFlip(), T.ToImage()])
    if aug == "none":  # identical to validation preprocessing
        train = T.Compose([pad_square, T.Resize((size, size), antialias=True), T.ToImage()])
    val = T.Compose([pad_square, T.Resize((size, size), antialias=True), T.ToImage()])
    if aug == "squash":  # no padding, no augmentation: stretch the whole image to size x size (train and val)
        train = val = T.Compose([T.Resize((size, size), antialias=True), T.ToImage()])
    if aug == "squash_base":  # base augmentation on a stretched (unpadded) image; val is stretched too
        big = int(size * 1.5)  # stretch to a larger square first so the random crop is not upsampled
        train = T.Compose([T.Resize((big, big), antialias=True), T.RandomResizedCrop(size, scale=(0.35, 1.0), antialias=True),
                           T.RandomHorizontalFlip(), T.TrivialAugmentWide(), T.ToImage(),
                           T.RandomErasing(0.25, value=mean_u8)])
        val = T.Compose([T.Resize((size, size), antialias=True), T.ToImage()])
    return train, val


def gpu_norm(x, dev):
    """uint8 NCHW batch -> normalized float on GPU, channels_last."""
    mean = torch.tensor(MEAN, device=dev).view(1, 3, 1, 1) * 255
    std = torch.tensor(STD, device=dev).view(1, 3, 1, 1) * 255
    x = x.to(dev, non_blocking=True).float()
    return ((x - mean) / std).contiguous(memory_format=torch.channels_last)


@torch.no_grad()
def evaluate(model, loader, dev):
    """Loader must not shuffle: predictions are matched to loader.dataset.items by position for per-source scores."""
    model.eval()
    P, Y = [], []
    for x, y in loader:
        with torch.autocast("cuda", torch.bfloat16):
            P.append(model(gpu_norm(x, dev)).argmax(1).cpu())
        Y.append(y)
    p, y = torch.cat(P), torch.cat(Y)
    n = len(CLASSES)
    conf = torch.bincount(y * n + p, minlength=n ** 2).view(n, -1)
    per_class = (conf.diag() / conf.sum(1).clamp(min=1)).tolist()
    out = {"acc": conf.diag().sum().item() / conf.sum().item(), "macro_acc": sum(per_class) / len(per_class),
           **{f"acc_{c}": a for c, a in zip(CLASSES, per_class)}}
    srcs = [src(g) for _, _, g in loader.dataset.items]
    for s in sorted(set(srcs)):  # macro accuracy per data source (140 = outdoor, 71385 = indoor sorter)
        m = torch.tensor([v == s for v in srcs])
        accs = [(p[m & (y == k)] == k).float().mean().item() for k in range(n) if (m & (y == k)).any()]
        out[f"val_{s}_macro"] = sum(accs) / len(accs)
    out["val_glass_as_plastic"] = conf[CLASSES.index("glass"), CLASSES.index("plastic")].item()
    return out, conf


def load_test(test_dir):
    """Own photos as fixed uint8 batches, for tracking test accuracy per epoch (never for model selection).
    c70: central 70% + letterbox to 320 (eval.py default). sq: whole photo stretched to 224."""
    xs = {"c70": [], "sq": []}
    ys = []
    for k, c in enumerate(CLASSES):
        for f in sorted((Path(test_dir) / c).iterdir()):
            if f.suffix.lower() not in (".jpg", ".png", ".webp"):
                continue
            im = ImageOps.exif_transpose(Image.open(f)).convert("RGB")
            w, h = im.size
            crop = pad_square(im.crop((int(w * .15), int(h * .15), int(w * .85), int(h * .85))))
            xs["c70"].append(T.functional.pil_to_tensor(crop.resize((320, 320), Image.BILINEAR, reducing_gap=3.0)))
            xs["sq"].append(T.functional.pil_to_tensor(im.resize((224, 224), Image.BILINEAR, reducing_gap=3.0)))
            ys.append(k)
    return {k: torch.stack(v) for k, v in xs.items()}, torch.tensor(ys)


@torch.no_grad()
def test_scores(model, xs, y, dev):
    out = {}
    for name, x in xs.items():
        with torch.autocast("cuda", torch.bfloat16):
            p = torch.cat([model(gpu_norm(x[i:i + 128], dev)).argmax(1).cpu() for i in range(0, len(x), 128)])
        per = [(p[y == k] == k).float().mean().item() for k in range(len(CLASSES))]
        out[f"test_{name}_acc"] = (p == y).float().mean().item()
        out[f"test_{name}_macro"] = sum(per) / len(per)
        out[f"test_{name}_glass_as_plastic"] = ((y == CLASSES.index("glass")) & (p == CLASSES.index("plastic"))).sum().item()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="resnet18", choices=["resnet18", "resnet34", "resnet50"])
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--warmup", type=int, default=5, help="warmup epochs")
    ap.add_argument("--aug", default="base", choices=["squash", "squash_base", "none", "light", "base", "strong"],
                    help="squash: stretch to square, no padding | squash_base: stretch + base augmentation | none: letterbox + resize only | light: crop+flip | base: +TrivialAugment, erasing | "
                         "strong: +random canvas/background, color jitter, vflip, grayscale, blur")
    ap.add_argument("--dropout", type=float, default=0.0, help="dropout on the pooled features, before the final layer")
    ap.add_argument("--smoothing", type=float, default=0.1)
    ap.add_argument("--no-class-weight", action="store_true", help="plain CE instead of inverse-frequency weights")
    ap.add_argument("--subset", type=int, default=0, help="random train subset size, 0 = all")
    ap.add_argument("--skip-140tr", action="store_true",
                    help="leave out the dataset 140 Training split (source 140tr), as in the earlier experiments")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--split", default="object", choices=["object", "capture"],
                    help="capture: hold out whole 71385 capture sessions (batch+date) for val; 140 stays per object")
    ap.add_argument("--train-source", default="all", choices=["all", "140", "71385"],
                    help="train on one AI Hub dataset only; val is unchanged so runs stay comparable")
    ap.add_argument("--source-balance", action="store_true",
                    help="sample training images 50:50 from 140 and 71385 (same number of samples per epoch)")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=0, help="stop each epoch after N steps (smoke test)")
    ap.add_argument("--test-dir", default="", help="log accuracy on this folder every epoch (monitoring only)")
    ap.add_argument("--name", default="run")
    ap.add_argument("--resume", default="")
    a = ap.parse_args()
    if a.resume:
        # a resumed run must keep its split, schedule and model; only runtime knobs may change
        cfg = json.loads((Path(a.resume).parent / "config.json").read_text())
        for k, v in cfg.items():
            if k in vars(a) and k not in ("resume", "workers"):
                if getattr(a, k) != v:
                    print(f"resume: using saved {k}={v} (ignoring {getattr(a, k)})")
                setattr(a, k, v)

    torch.manual_seed(a.seed)
    random.seed(a.seed)
    dev = "cuda"
    out = ROOT / "runs" / a.name
    out.mkdir(parents=True, exist_ok=True)

    tr, va = split(load_index(ROOT), a.val_frac, a.split)
    if a.skip_140tr:
        tr = [x for x in tr if not x[2].startswith("140tr_")]
    if a.train_source != "all":
        tr = [x for x in tr if src(x[2]) == a.train_source]
    if a.subset:
        tr = random.Random(a.seed).sample(tr, min(a.subset, len(tr)))
    counts = Counter(y for _, y, _ in tr)
    print(f"train {len(tr)} {dict((CLASSES[k], v) for k, v in sorted(counts.items()))} | val {len(va)}")

    tf_tr, tf_va = transforms(a.size, a.aug)
    kw = dict(num_workers=a.workers, pin_memory=True, persistent_workers=a.workers > 0)
    sampler = None
    if a.source_balance:
        n_src = Counter(src(g) for _, _, g in tr)
        wts = [0.5 / n_src[src(g)] for _, _, g in tr]  # each source gets half of every epoch
        sampler = torch.utils.data.WeightedRandomSampler(wts, num_samples=len(tr), replacement=True)
        print(f"source-balance: {dict(n_src)} resampled 50:50, {len(tr)} samples/epoch")
    dl_tr = DataLoader(Images(tr, tf_tr, a.size), a.batch, shuffle=sampler is None, sampler=sampler, drop_last=True, **kw)
    dl_va = DataLoader(Images(va, tf_va, a.size), a.batch * 2, **kw)

    model = getattr(torchvision.models, a.arch)(weights=None, num_classes=len(CLASSES))  # scratch: no pretrained
    if a.dropout:
        # after global pooling, before fc. No parameters, so checkpoints still load into a plain torchvision model
        # (eval.py), where dropout is simply absent - same as dropout in eval mode.
        model.avgpool = nn.Sequential(model.avgpool, nn.Dropout(a.dropout))
    model = model.to(dev).to(memory_format=torch.channels_last)
    w = None
    if not a.no_class_weight:
        w = torch.tensor([len(tr) / (len(CLASSES) * counts[i]) for i in range(len(CLASSES))], device=dev)
    crit = nn.CrossEntropyLoss(weight=w, label_smoothing=a.smoothing)
    # no weight decay on BN and bias
    decay = [p for n, p in model.named_parameters() if p.ndim > 1]
    no_decay = [p for n, p in model.named_parameters() if p.ndim <= 1]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": a.wd}, {"params": no_decay, "weight_decay": 0}], lr=a.lr)
    steps_per_epoch = min(len(dl_tr), a.max_steps or len(dl_tr))
    total, warm = a.epochs * steps_per_epoch, a.warmup * steps_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))

    start, best = 0, -1.0
    if a.resume:
        ck = torch.load(a.resume, map_location=dev)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        start, best = ck["epoch"] + 1, ck["best"]
        print(f"resumed at epoch {start}, best macro_acc {best:.4f}")
    (out / "config.json").write_text(json.dumps({**vars(a), "classes": CLASSES, "n_train": len(tr), "n_val": len(va)}, indent=2))

    test = load_test(a.test_dir) if a.test_dir else None
    log = open(out / "metrics.csv", "a", newline="")
    writer = None
    for ep in range(start, a.epochs):
        model.train()
        t0, loss_sum, n = time.time(), 0.0, 0
        for i, (x, y) in enumerate(dl_tr):
            if i >= steps_per_epoch:
                break
            x = gpu_norm(x, dev)
            y = y.to(dev, non_blocking=True)
            with torch.autocast("cuda", torch.bfloat16):
                loss = crit(model(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            loss_sum += loss.item(); n += 1
        assert math.isfinite(loss_sum), "loss diverged (NaN/inf)"
        train_time = time.time() - t0
        m, conf = evaluate(model, dl_va, dev)
        row = {"epoch": ep, "train_loss": loss_sum / n, "lr": sched.get_last_lr()[0],
               "img_per_s": n * a.batch / train_time, **m}
        if test:  # model is in eval mode after evaluate(); best.pt is still chosen by val only
            row.update(test_scores(model, *test, dev))
        if writer is None:
            writer = csv.DictWriter(log, fieldnames=list(row))
            if log.tell() == 0:
                writer.writeheader()
        writer.writerow(row); log.flush()
        print(f"ep {ep} loss {row['train_loss']:.4f} val acc {m['acc']:.4f} macro {m['macro_acc']:.4f} "
              f"| {row['img_per_s']:.0f} img/s, {train_time:.0f}s", flush=True)
        if m["macro_acc"] > best:
            best = m["macro_acc"]
            save_atomic({"model": model.state_dict(), "arch": a.arch, "classes": CLASSES, "size": a.size,
                         "epoch": ep, "val": m, "confusion": conf.tolist()}, out / "best.pt")
        save_atomic({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                     "epoch": ep, "best": best}, out / "last.pt")
    print(f"done. best val macro_acc {best:.4f} -> {out / 'best.pt'}")


if __name__ == "__main__":
    main()
