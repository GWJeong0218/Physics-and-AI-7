"""Fix the split of our own photos: test/ -> test300/ (75 per class, held out) + test_train/ (the rest, for training).

Photos are moved in runs of 5 neighbouring numbers, because neighbours often show the same object or place;
this keeps near-duplicates on one side. The choice is seeded, so every run gives the same split.
Photos for training get their EXIF rotation applied and are shrunk to 640px on the long side.

usage: python split_test.py        (writes test300/, test_train/ and test_split.json next to test/)
"""
import json
import random
import shutil
from pathlib import Path

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent
CLASSES = ("vinyl", "glass", "plastic", "can")
PER_CLASS, RUN = 75, 5

lists = {}
for c in CLASSES:
    fs = sorted((ROOT / "test" / c).iterdir())
    runs = [fs[i:i + RUN] for i in range(0, len(fs) - len(fs) % RUN, RUN)]   # a short last run goes to training
    keep = set(random.Random(f"a{c}").sample(range(len(runs)), PER_CLASS // RUN))
    test = [f for i in sorted(keep) for f in runs[i]]
    train = [f for f in fs if f not in test]
    assert len(test) == PER_CLASS and len(test) + len(train) == len(fs)
    for d in ("test300", "test_train"):
        (ROOT / d / c).mkdir(parents=True, exist_ok=True)
    for f in test:
        shutil.copy(f, ROOT / "test300" / c / f.name)
    for f in train:
        im = ImageOps.exif_transpose(Image.open(f)).convert("RGB")
        im.thumbnail((640, 640))
        im.save(ROOT / "test_train" / c / (f.stem + ".jpg"), quality=92)
    lists[c] = {"test": [f.name for f in test], "train": [f.name for f in train]}
    print(c, "test", len(test), "train", len(train))
(ROOT / "test_split.json").write_text(json.dumps(lists, indent=1))
