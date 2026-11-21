"""Crop AI Hub waste images (datasets 140, 71385) into 4 classes: vinyl, glass, plastic, can.

Reads images straight from the downloaded zips (no full extraction), crops every labeled
object of a target class by its bounding box, and writes:
  <out>/<class>/<source>_<image stem>_<object index>.jpg
  <out>/manifest.csv   (path, class, source, split, group, detail, image)

Split train/test by the `group` column, not by file: dataset 140 shoots each object 4-5 times
(group = object folder), so a per-file split leaks near-duplicates into the test set.

usage: python -I build_4class.py <_aihub_raw dir> <out dir>
"""
import csv
import io
import json
import os
import sys
import zipfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image

CLASSES = ("vinyl", "glass", "plastic", "can")
MARGIN = 0.05  # context around each box, as a fraction of box size
MIN_SIDE = 32  # smaller crops are too blurry to classify

# dataset 140: Bounding[].CLASS names
MAP_140 = {"비닐류": "vinyl", "유리병류": "glass", "페트병류": "plastic", "플라스틱류": "plastic", "캔류": "can"}
# dataset 71385: class_name codes c_<group>_...; 3 can, 4 glass, 5 PET, 6 plastic, 7 vinyl
MAP_71385 = {"3": "can", "4": "glass", "5": "plastic", "6": "plastic", "7": "vinyl"}


def stem(name):
    return os.path.splitext(name.replace("\\", "/").rsplit("/", 1)[-1])[0]


def labels_140(zip_path):
    """image stem -> (group, [(class, detail, (x1, y1, x2, y2))])"""
    out = {}
    with zipfile.ZipFile(zip_path) as z:
        for n in z.namelist():
            if not n.lower().endswith(".json"):
                continue
            d = json.loads(z.read(n).decode("utf-8-sig"))
            objs = []
            for b in d.get("Bounding", []):
                cls = MAP_140.get(b.get("CLASS"))
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


def labels_71385(zip_path):
    out = {}
    with zipfile.ZipFile(zip_path) as z:
        for n in z.namelist():
            if not n.lower().endswith(".json"):
                continue
            d = json.loads(z.read(n).decode("utf-8-sig"))
            objs = []
            for o in d.get("objects", []):
                code = o.get("class_name", "")
                cls = MAP_71385.get(code.split("_")[1] if code.count("_") else "")
                if not cls or o.get("annotation_type") != "box":
                    continue
                c = o["annotation"]["coord"]
                objs.append((cls, code, (c["x"], c["y"], c["x"] + c["width"], c["y"] + c["height"])))
            if objs:
                s = stem(d["Image"])
                out[s] = (s, objs)  # no object id in this dataset; frames may still be near-duplicates
    return out


def crop_zip(job):
    """Crop every labeled object in one source zip. Returns (manifest rows, Counter of skips)."""
    src_zip, source, split, labels, out_dir = job
    rows, skip = [], Counter()
    with zipfile.ZipFile(src_zip) as z:
        for n in z.namelist():
            if not n.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            s = stem(n)
            if s not in labels:
                skip["no_label"] += 1
                continue
            group, objs = labels[s]
            todo = [(i, o) for i, o in enumerate(objs)
                    if not (out_dir / o[0] / f"{source}_{s}_{i}.jpg").exists()]
            if not todo:
                rows += [_row(out_dir, o, source, split, group, s, i) for i, o in enumerate(objs)]
                continue
            try:
                img = Image.open(io.BytesIO(z.read(n)))
                img.load()
            except Exception:
                skip["bad_image"] += 1
                continue
            img = img.convert("RGB")
            W, H = img.size
            for i, (cls, detail, (x1, y1, x2, y2)) in enumerate(objs):
                x1, x2 = sorted((x1, x2))
                y1, y2 = sorted((y1, y2))
                mw, mh = (x2 - x1) * MARGIN, (y2 - y1) * MARGIN
                box = (max(0, int(x1 - mw)), max(0, int(y1 - mh)), min(W, int(x2 + mw)), min(H, int(y2 + mh)))
                if box[2] - box[0] < MIN_SIDE or box[3] - box[1] < MIN_SIDE:
                    skip["too_small"] += 1
                    continue
                dst = out_dir / cls / f"{source}_{s}_{i}.jpg"
                if not dst.exists():
                    img.crop(box).save(dst, quality=95)
                rows.append(_row(out_dir, (cls, detail, None), source, split, group, s, i))
    return rows, skip


def _row(out_dir, obj, source, split, group, s, i):
    return {"path": f"{obj[0]}/{source}_{s}_{i}.jpg", "class": obj[0], "source": source,
            "split": split, "group": f"{source}_{group}", "detail": obj[1], "image": s}


def jobs(raw, out_dir):
    # dataset 140: one label zip per split, many source zips
    for split_dir in (raw / "140").rglob("*.Training"), (raw / "140").rglob("*.Validation"):
        for d in split_dir:
            lab = next(d.rglob("라벨링데이터.zip"), None)
            if not lab:
                continue
            labels = labels_140(lab)
            split = "train" if "Training" in d.name else "val"
            for z in d.rglob("*.zip"):
                if z != lab:
                    yield z, "140", split, labels, out_dir
    # dataset 71385: label zip TL_x / VL_x pairs with source zip TS_x / VS_x
    for lab in sorted((raw / "71385").rglob("[TV]L_*.zip")):
        src = next(lab.parents[1].rglob(lab.name[0] + "S_" + lab.name[3:]), None)
        if src:
            yield src, "71385", "train" if lab.name[0] == "T" else "val", labels_71385(lab), out_dir


def main(raw, out_dir):
    raw, out_dir = Path(raw), Path(out_dir)
    for c in CLASSES:
        (out_dir / c).mkdir(parents=True, exist_ok=True)
    rows, skip = [], Counter()
    with ProcessPoolExecutor(max(1, (os.cpu_count() or 2) - 1)) as ex:
        for z, (r, s) in zip_names(ex, jobs(raw, out_dir)):
            rows += r
            skip += s
            print(f"{z}: {len(r)} crops", flush=True)
    with open(out_dir / "manifest.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    by = Counter((r["source"], r["split"], r["class"]) for r in rows)
    for k in sorted(by):
        print(*k, by[k])
    print("skipped:", dict(skip))
    on_disk = sum(len(os.listdir(out_dir / c)) for c in CLASSES)
    assert on_disk == len(rows), f"manifest rows {len(rows)} != files on disk {on_disk}"
    print(f"total {len(rows)} crops, manifest matches files on disk")


def zip_names(ex, job_iter):
    jl = list(job_iter)
    for j, res in zip(jl, ex.map(crop_zip, jl)):
        yield Path(j[0]).name, res


if __name__ == "__main__":
    main(*sys.argv[1:3])
