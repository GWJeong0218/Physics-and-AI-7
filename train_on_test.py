"""Train ResNet-18 from scratch on the own-photo `test` folder only, stratified 8/1/1 split, heavy augmentation.

usage: python train_on_test.py [--seeds 0 1 2 3 4] [--epochs 300] [--ref runs/<name>/best.pt]
Each seed is a different random 8/1/1 split. Best epoch is picked on the val part, then scored once on the
held-out test part. With --ref, an AI Hub-trained ResNet-18 checkpoint is scored on the same held-out images.
Caveats: ~46 held-out images per split (one image = ~2 points), and photos of the same object can land on
both sides of a split (no object ids), which inflates the score.
"""
import argparse
import json
import math
import random
from pathlib import Path

import torch
import torch.nn as nn
import torchvision
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2 as T

from train import CLASSES, MEAN_U8, ROOT, gpu_norm, pad_square, save_atomic

SIZE = 224


class Mem(Dataset):
    def __init__(self, items, tf):
        self.items, self.tf = items, tf

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        im, y = self.items[i]
        return self.tf(im), y


def predict(model, items, tf):
    out = []
    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
        for i in range(0, len(items), 64):
            x = torch.stack([tf(im) for im, _ in items[i:i + 64]])
            out.append(model(gpu_norm(x, "cuda")).argmax(1).cpu())
    return torch.cat(out)


def scores(pred, items):
    y = torch.tensor([t for _, t in items])
    per = [(pred[y == k] == k).float().mean().item() for k in range(len(CLASSES))]
    return {"acc": (pred == y).float().mean().item(), "macro_acc": sum(per) / len(per),
            **{f"acc_{c}": v for c, v in zip(CLASSES, per)}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--ref", default="", help="AI Hub-trained ResNet-18 checkpoint to score on the same held-out images")
    a = ap.parse_args()

    # whole photos, EXIF rotation applied, shrunk once (model input is 224)
    data = []
    for k, c in enumerate(CLASSES):
        for p in sorted((ROOT / "test" / c).iterdir()):
            if p.suffix.lower() in (".jpg", ".png", ".webp"):
                im = ImageOps.exif_transpose(Image.open(p)).convert("RGB")
                im.thumbnail((448, 448))
                data.append((im, k))

    heavy = T.Compose([
        pad_square, T.RandomResizedCrop(SIZE, scale=(0.25, 1.0), antialias=True),
        T.RandomHorizontalFlip(), T.RandomVerticalFlip(), T.RandomRotation(30, fill=MEAN_U8),
        T.ColorJitter(0.4, 0.4, 0.4, 0.05), T.RandomGrayscale(0.1), T.TrivialAugmentWide(),
        T.ToImage(), T.RandomApply([T.GaussianBlur(5, (0.1, 2.0))], p=0.2),
        T.RandomErasing(0.25, value=list(MEAN_U8))])
    plain = T.Compose([pad_square, T.Resize((SIZE, SIZE), antialias=True), T.ToImage()])
    mix = T.RandomChoice([T.MixUp(num_classes=len(CLASSES)), T.CutMix(num_classes=len(CLASSES))])

    def center70_320(im):  # the preprocessing eval.py uses for the AI Hub model
        w, h = im.size
        im = im.crop((int(w * .15), int(h * .15), int(w * .85), int(h * .85)))
        return T.Compose([pad_square, T.Resize((320, 320), antialias=True), T.ToImage()])(im)

    ref = None
    if a.ref:
        ref = torchvision.models.resnet18(num_classes=len(CLASSES))
        ref.load_state_dict(torch.load(a.ref, map_location="cuda", weights_only=False)["model"])
        ref = ref.cuda().to(memory_format=torch.channels_last).eval()

    results = []
    for seed in a.seeds:
        rng = random.Random(seed)
        tr, va, te = [], [], []
        for k in range(len(CLASSES)):  # stratified 8/1/1
            idx = [i for i, (_, y) in enumerate(data) if y == k]
            rng.shuffle(idx)
            n_va = n_te = round(len(idx) * 0.1)
            te += idx[:n_te]; va += idx[n_te:n_te + n_va]; tr += idx[n_te + n_va:]
        assert not (set(tr) & set(va) or set(tr) & set(te) or set(va) & set(te))
        tr, va, te = ([data[i] for i in s] for s in (tr, va, te))

        torch.manual_seed(seed)
        model = torchvision.models.resnet18(weights=None, num_classes=len(CLASSES))  # scratch
        model = model.cuda().to(memory_format=torch.channels_last)
        dl = DataLoader(Mem(tr, heavy), a.batch, shuffle=True, drop_last=True, num_workers=a.workers,
                        persistent_workers=a.workers > 0)
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
        total, warm = a.epochs * len(dl), 10 * len(dl)
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
        crit = nn.CrossEntropyLoss(label_smoothing=0.1)
        out = ROOT / "runs" / f"r18_testonly_s{seed}"
        out.mkdir(parents=True, exist_ok=True)

        best, best_ep = -1.0, -1
        for ep in range(a.epochs):
            model.train()
            loss_sum = 0.0
            for x, y in dl:
                x, y = gpu_norm(x, "cuda"), y.cuda(non_blocking=True)
                if torch.rand(1).item() < 0.5:  # MixUp or CutMix on half the batches (needs float images)
                    x, y = mix(x, y)
                with torch.autocast("cuda", torch.bfloat16):
                    loss = crit(model(x), y)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                sched.step()
                loss_sum += loss.item()
            assert math.isfinite(loss_sum), "loss diverged"
            model.eval()
            v = scores(predict(model, va, plain), va)["macro_acc"]
            if v > best:  # ties keep the earlier epoch
                best, best_ep = v, ep
                save_atomic({"model": model.state_dict(), "arch": "resnet18", "classes": CLASSES, "size": SIZE,
                             "epoch": ep}, out / "best.pt")
            if ep % 25 == 0 or ep == a.epochs - 1:
                print(f"seed {seed} ep {ep} loss {loss_sum / len(dl):.3f} val macro {v:.3f} (best {best:.3f} @ {best_ep})", flush=True)

        model.load_state_dict(torch.load(out / "best.pt", map_location="cuda", weights_only=False)["model"])
        model.eval()
        r = {"seed": seed, "n": [len(tr), len(va), len(te)], "best_epoch": best_ep, "val_macro": best,
             "train_fit": scores(predict(model, tr, plain), tr)["acc"], "test": scores(predict(model, te, plain), te)}
        if ref is not None:
            r["ref_same_test_whole224"] = scores(predict(ref, te, plain), te)
            r["ref_same_test_center70_320"] = scores(predict(ref, te, center70_320), te)
        results.append(r)
        print(json.dumps(r), flush=True)
        (out / "result.json").write_text(json.dumps(r, indent=1))

    def mean_sd(f):
        xs = [f(r) for r in results]
        m = sum(xs) / len(xs)
        return f"{m:.3f} +- {(sum((x - m) ** 2 for x in xs) / max(1, len(xs) - 1)) ** 0.5:.3f} (min {min(xs):.3f}, max {max(xs):.3f})"
    print("\nheld-out test over", len(results), "splits")
    print("  trained on own photos, acc      :", mean_sd(lambda r: r["test"]["acc"]))
    print("  trained on own photos, macro    :", mean_sd(lambda r: r["test"]["macro_acc"]))
    for c in CLASSES:
        print(f"    {c:8s}:", mean_sd(lambda r: r["test"][f"acc_{c}"]))
    if ref is not None:
        print("  ref (AI Hub), whole 224, acc    :", mean_sd(lambda r: r["ref_same_test_whole224"]["acc"]))
        print("  ref (AI Hub), 70%+320, acc      :", mean_sd(lambda r: r["ref_same_test_center70_320"]["acc"]))


if __name__ == "__main__":
    main()
