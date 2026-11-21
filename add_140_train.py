"""Add the AI Hub dataset 140 *Training* split to the 4-class training data.

Reads the downloaded zips under _aihub_raw/140 directly, crops every labeled vinyl/glass/plastic/can object by
its box (+5% margin), shrinks the short side to 256, and writes
  <root>/<class>/140tr_<image stem>_<object index>.jpg
  <root>/manifest_full.csv = manifest.csv + the new rows (source "140tr", group "140tr_<object folder>")
train.py keeps every "140tr" group in train, so the validation set stays identical to earlier runs.

usage: python add_140_train.py            (reruns skip files that already exist)
"""
import csv
import io
import json
import os
import zipfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "_aihub_raw" / "140"
CLASSES = ("vinyl", "glass", "plastic", "can")
MAP = {"비닐류": "vinyl", "유리병류": "glass", "페트병류": "plastic", "플라스틱류": "plastic", "캔류": "can"}
MARGIN, MIN_SIDE, SHORT = 0.05, 32, 256
LABELS = {}  # image stem -> (group, [(class, detail, box)]); filled before the pool forks


def stem(name):
    return os.path.splitext(name.replace("\\", "/").rsplit("/", 1)[-1])[0]


def load_labels(zip_path):
    out = {}
    with zipfile.ZipFile(zip_path) as z:
        for n in z.namelist():
            if not n.lower().endswith(".json"):
                continue
            d = json.loads(z.read(n).decode("utf-8-sig"))
            objs = []
            for b in d.get("Bounding", []):
                cls = MAP.get(b.get("CLASS"))
                if not cls:
                    continue
                if b.get("Drawing") == "POLYGON":
                    pts = [tuple(map(float, next(iter(p.values())).split(","))) for p in b["PolygonPoint"]]
                    xs, ys = zip(*pts)
                    box = (min(xs), min(ys), max(xs), max(ys))
                else:
                    box = tuple(float(b[k]) for k in ("x1", "y1", "x2", "y2"))
                objs.append((cls, b.get("DETAILS", ""), box))
            if objs:
                s = stem(d["FILE NAME"])
                out[s] = (s.rsplit("_", 1)[0], objs)
    return out


def work(job):
    zip_path, names = job
    rows, skip = [], Counter()
    with zipfile.ZipFile(zip_path) as z:
        for n in names:
            s = stem(n)
            if s not in LABELS:
                skip["no_label"] += 1
                continue
            group, objs = LABELS[s]
            img = None
            for i, (cls, detail, (x1, y1, x2, y2)) in enumerate(objs):
                dst = ROOT / cls / f"140tr_{s}_{i}.jpg"
                if not dst.exists():
                    if img is None:
                        try:
                            img = Image.open(io.BytesIO(z.read(n)))
                            img.load()
                            img = img.convert("RGB")
                        except Exception:
                            skip["bad_image"] += 1
                            break
                    W, H = img.size
                    x1, x2 = sorted((x1, x2))
                    y1, y2 = sorted((y1, y2))
                    mw, mh = (x2 - x1) * MARGIN, (y2 - y1) * MARGIN
                    box = (max(0, int(x1 - mw)), max(0, int(y1 - mh)), min(W, int(x2 + mw)), min(H, int(y2 + mh)))
                    if box[2] - box[0] < MIN_SIDE or box[3] - box[1] < MIN_SIDE:
                        skip["too_small"] += 1
                        continue
                    c = img.crop(box)
                    sc = SHORT / min(c.size)
                    if sc < 1:
                        c = c.resize((round(c.width * sc), round(c.height * sc)), Image.LANCZOS)
                    tmp = dst.with_suffix(".tmp")
                    c.save(tmp, "JPEG", quality=95)
                    os.replace(tmp, dst)  # never leave a half-written image behind
                rows.append({"path": f"{cls}/{dst.name}", "class": cls, "source": "140tr",
                             "group": f"140tr_{group}", "detail": detail})
    return rows, skip


def main():
    global LABELS
    lab = next(RAW.rglob("1.Training/라벨링데이터/라벨링데이터.zip"))
    LABELS = load_labels(lab)
    print(f"labels: {len(LABELS)} images with target objects", flush=True)
    jobs = []
    for zp in sorted(RAW.rglob("1.Training/원천데이터/**/*.zip")):
        with zipfile.ZipFile(zp) as z:
            names = [n for n in z.namelist() if n.lower().endswith((".jpg", ".jpeg", ".png"))]
        jobs += [(str(zp), names[i:i + 400]) for i in range(0, len(names), 400)]
    print(f"{len(jobs)} chunks", flush=True)
    rows, skip = [], Counter()
    with ProcessPoolExecutor(max(1, (os.cpu_count() or 2) - 8)) as ex:  # fork: workers inherit LABELS
        for k, (r, s) in enumerate(ex.map(work, jobs)):
            rows += r
            skip += s
            if k % 200 == 0:
                print(f"chunk {k}/{len(jobs)}: {len(rows)} crops", flush=True)

    old = list(csv.DictReader(open(ROOT / "manifest.csv", encoding="utf-8-sig")))
    assert not {r["path"] for r in old} & {r["path"] for r in rows}, "new file names collide with existing ones"
    assert len({r["path"] for r in rows}) == len(rows), "duplicate new file names"
    shared = {r["group"][len("140_"):] for r in old if r["source"] == "140"} & {r["group"][len("140tr_"):] for r in rows}
    with open(ROOT / "manifest_full.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["path", "class", "source", "group", "detail"])
        w.writeheader()
        w.writerows(old + rows)
    missing = [r["path"] for r in rows if not (ROOT / r["path"]).exists()]
    assert not missing, f"{len(missing)} manifest rows have no file"
    by = Counter(r["class"] for r in rows)
    print("new crops:", len(rows), dict(by), "| groups:", len({r["group"] for r in rows}), "| skipped:", dict(skip))
    print("object folders shared with the existing 140 data (possible train/val overlap):", len(shared))
    print("manifest_full.csv rows:", len(old) + len(rows))


if __name__ == "__main__":
    main()
