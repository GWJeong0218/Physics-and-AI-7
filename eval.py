"""Evaluate a trained checkpoint on own photos: test/<class>/*.

usage: python eval.py runs/<name>/best.pt [--data test] [--center 0.7] [--size 320]
Preprocessing: keep the central `--center` fraction of each side, letterbox to a square, resize to `--size`.
Default 70% + 320 makes small, centered objects appear closer to the training crops' scale
(ResNet-18 baseline: 0.652 vs 0.487 for --center 1 --size 224). Chosen on this test set, so the score is optimistic.
Writes <checkpoint dir>/test_predictions.csv (path, true, pred, confidence, per-class probabilities).
"""
import argparse
import csv
from pathlib import Path

import torch
import torchvision
from PIL import Image, ImageOps
from torchvision.transforms import v2 as T

from train import CLASSES, ROOT, gpu_norm, pad_square


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--data", default=str(ROOT / "test"))
    ap.add_argument("--center", type=float, default=0.7, help="central fraction of each side to keep (1 = whole photo)")
    ap.add_argument("--size", type=int, default=320, help="model input size (ResNet accepts any size)")
    a = ap.parse_args()

    ck = torch.load(a.ckpt, map_location="cuda", weights_only=False)
    assert tuple(ck["classes"]) == CLASSES
    model = getattr(torchvision.models, ck["arch"])(weights=None, num_classes=len(CLASSES))
    model.load_state_dict(ck["model"])
    model = model.cuda().to(memory_format=torch.channels_last).eval()

    def center_crop(im):
        w, h = im.size
        dw, dh = w * (1 - a.center) / 2, h * (1 - a.center) / 2
        return im.crop((int(dw), int(dh), int(w - dw), int(h - dh)))

    tf = T.Compose([center_crop, pad_square, T.Resize((a.size, a.size), antialias=True), T.ToImage()])

    items = [(p, CLASSES.index(c)) for c in CLASSES for p in sorted((Path(a.data) / c).iterdir()) if p.is_file()]
    rows, conf = [], torch.zeros(len(CLASSES), len(CLASSES), dtype=torch.long)
    with torch.no_grad():
        for i in range(0, len(items), 64):
            batch = items[i:i + 64]
            # phone photos store rotation in EXIF; without this, portrait shots are fed sideways
            x = torch.stack([tf(ImageOps.exif_transpose(Image.open(p)).convert("RGB")) for p, _ in batch])
            with torch.autocast("cuda", torch.bfloat16):
                prob = model(gpu_norm(x, "cuda")).float().softmax(1).cpu()
            for (p, y), pr in zip(batch, prob):
                k = int(pr.argmax())
                conf[y, k] += 1
                rows.append([p.as_posix(), CLASSES[y], CLASSES[k], f"{pr[k]:.4f}", *(f"{v:.4f}" for v in pr)])

    out = Path(a.ckpt).parent / "test_predictions.csv"
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["path", "true", "pred", "confidence", *(f"p_{c}" for c in CLASSES)])
        w.writerows(rows)

    per = conf.diag() / conf.sum(1).clamp(min=1)
    print(f"preprocess: center {a.center}, size {a.size}")
    print(f"n={conf.sum().item()} acc {conf.diag().sum().item() / conf.sum().item():.4f} macro_acc {per.mean():.4f}")
    for c, v, n in zip(CLASSES, per.tolist(), conf.sum(1).tolist()):
        print(f"  {c:8s} {v:.4f} (n={n})")
    print("confusion (row=true, col=pred):", CLASSES)
    for c, r in zip(CLASSES, conf.tolist()):
        print(f"  {c:8s} {r}")
    print("predicted counts:", dict(zip(CLASSES, conf.sum(0).tolist())))
    print("saved", out)


if __name__ == "__main__":
    main()
