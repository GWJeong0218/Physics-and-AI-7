"""Build the 4-class training data (vinyl, glass, plastic, can) from two AI Hub datasets.

  python prepare_data.py fetch     # download the needed zips into _aihub_raw/   (about 135 GB)
  python prepare_data.py build     # crop every labeled object, shrink to 256 px, write manifest.csv

Datasets (both need a download approval on aihub.or.kr and an API key):
  140    생활 폐기물 이미지              outdoor photos; categories 비닐, 유리병, 캔류, 페트병, 플라스틱류
  71385  생활폐기물 데이터 활용·환류     the "실내형 분류기" (indoor sorter) part

API key: environment variable AIHUB_APIKEY, or a file `.aihub_key` next to this script.

Output of `build`:
  <root>/<class>/<source>_<image>_<object index>.jpg     short side 256 px
  <root>/manifest.csv                                    path, class, source, group, detail
`source` is 140 (dataset 140 Validation split), 140tr (dataset 140 Training split) or 71385.
`group` ties together photos that must stay on the same side of a train/val split (see train.py).
"""
import argparse
import csv
import io
import json
import os
import re
import shutil
import subprocess
import time
import zipfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

from PIL import Image

CLASSES = ("vinyl", "glass", "plastic", "can")
MARGIN = 0.05   # context kept around each box, as a fraction of the box size
MIN_SIDE = 32   # smaller crops are too blurry to classify
SHORT = 256     # stored short side; training uses 224

# AI Hub file keys (from `https://api.aihub.or.kr/info/<dataset>.do`)
KEYS = {
    "140-val": ("140", [47732,                                                        # labels
                        47679, 47680, 47681, 47682, 47683, 47684, 47685,              # 비닐
                        47690, 47691, 47692, 47693, 47694, 47695, 47696, 47608,       # 유리병
                        47569, 47570, 47571, 47572, 47573, 47574,                     # 캔류
                        47575, 47576, 47577,                                          # 페트병
                        47578, 47579, 47580, 47581, 47582, 47583]),                   # 플라스틱류
    "140-train": ("140", [47869,
                          47820, 47821, 47822, 47823, 47824, 47825, 47826,
                          47831, 47832, 47833, 47744, 47745, 47746, 47747, 47748,
                          47709, 47710, 47711, 47712, 47713, 47714, 47715,
                          47716, 47717, 47718,
                          47719, 47720, 47721, 47722, 47723, 47724]),
    "71385": ("71385", [*range(514421, 514431), *range(514445, 514455),               # Training images, labels
                        *range(514465, 514475), *range(514485, 514495)]),             # Validation images, labels
}
# dataset 140: label CLASS -> class
MAP_140 = {"비닐류": "vinyl", "유리병류": "glass", "페트병류": "plastic", "플라스틱류": "plastic", "캔류": "can"}
# dataset 71385: class_name codes c_<n>_...; 3 can, 4 glass, 5 PET, 6 plastic, 7 vinyl.
# Codes for "contaminated" or "multi-material packaging" variants share the same leading number.
MAP_71385 = {"3": "can", "4": "glass", "5": "plastic", "6": "plastic", "7": "vinyl"}

LABELS = {}  # per worker: source tag -> {image stem: (group, [(class, detail, box)])}


# ---------------------------------------------------------------- fetch

def valid_zip(p):
    """Opens, has members, and the members' compressed bytes fit in the file (a truncated zip fails here)."""
    try:
        with zipfile.ZipFile(p) as z:
            infos = z.infolist()
            return len(infos) > 0 and sum(i.compress_size for i in infos) <= os.path.getsize(p)
    except Exception:
        return False


def fetch_one(raw, api, dataset, key):
    """AI Hub drops connections mid-file and cannot resume, so each file is fetched whole into its own
    folder, retried from scratch, and accepted only if every merged zip validates."""
    d = raw / dataset / f"k_{key}"
    if (d / ".ok").exists():
        return True
    for attempt in range(1, 9):
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)
        # the key goes in through stdin (curl -K -), not argv, so it does not show up in the process list
        r = subprocess.run(["curl", "-sS", "-L", "--fail", "-o", "dl.tar", "-K", "-",
                            f"https://api.aihub.or.kr/down/0.6/{dataset}.do?fileSn={key}"],
                           input=f'header = "apikey:{api}"\n', cwd=d, capture_output=True, text=True)
        if r.returncode == 0 and subprocess.run(["tar", "-xf", "dl.tar"], cwd=d, capture_output=True).returncode == 0:
            (d / "dl.tar").unlink()
            groups = {}  # the tar holds each zip as numbered parts: <name>.zip.part0, .part1, ...
            for part in d.rglob("*.part*"):
                m = re.match(r"(.*)\.part(\d+)$", str(part))
                if m:
                    groups.setdefault(m[1], []).append((int(m[2]), part))
            for out, parts in groups.items():
                with open(out, "wb") as f:
                    for _, part in sorted(parts):
                        with open(part, "rb") as src:
                            shutil.copyfileobj(src, f, 16 << 20)
                for _, part in parts:
                    part.unlink()
            zips = list(d.rglob("*.zip"))
            if zips and all(valid_zip(z) for z in zips):
                (d / ".ok").touch()
                print(f"[{dataset}/{key}] ok ({sum(os.path.getsize(z) for z in zips) / 1e9:.2f} GB, attempt {attempt})", flush=True)
                return True
            print(f"[{dataset}/{key}] attempt {attempt}: downloaded but zip invalid", flush=True)
        else:
            print(f"[{dataset}/{key}] attempt {attempt}: curl exit {r.returncode} {r.stderr.strip()[:90]}", flush=True)
        time.sleep(30)
    print(f"[{dataset}/{key}] FAILED", flush=True)
    return False


def fetch(root, parts, keys):
    key_file = root / ".aihub_key"
    api = os.environ.get("AIHUB_APIKEY") or (key_file.read_text().strip() if key_file.exists() else "")
    if not api:
        raise SystemExit("set AIHUB_APIKEY or create .aihub_key")
    todo = [tuple(k.split(":")) for k in keys] if keys else [(KEYS[p][0], k) for p in parts for k in KEYS[p][1]]
    with ThreadPoolExecutor(3) as ex:
        ok = list(ex.map(lambda t: fetch_one(root / "_aihub_raw", api, t[0], int(t[1])), todo))
    print(f"fetched {sum(ok)}/{len(ok)} files")
    if not all(ok):
        raise SystemExit(1)


# ---------------------------------------------------------------- build

def stem(name):
    return os.path.splitext(name.replace("\\", "/").rsplit("/", 1)[-1])[0]


def labels_140(zip_path):
    """image stem -> (group, [(class, detail, (x1, y1, x2, y2))]). One object is photographed 4-5 times as
    <object>_0.jpg ... <object>_4.jpg, so the object name is the group."""
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
    """Same shape as labels_140. There is no object id; the image name (<batch>_<date>_<frame>) is the group."""
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
                out[s] = (s, objs)
    return out


def _init(labels):
    global LABELS
    LABELS = labels


def crop_chunk(job):
    """Crop every labeled object in a chunk of images from one source zip. Existing outputs are kept."""
    root, tag, zip_path, names = job
    labels = LABELS[tag]
    rows, skip = [], Counter()
    with zipfile.ZipFile(zip_path) as z:
        for n in names:
            s = stem(n)
            if s not in labels:
                skip["no_target_object"] += 1
                continue
            group, objs = labels[s]
            img = None
            for i, (cls, detail, (x1, y1, x2, y2)) in enumerate(objs):
                dst = Path(root) / cls / f"{tag}_{s}_{i}.jpg"
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
                rows.append({"path": f"{cls}/{dst.name}", "class": cls, "source": tag, "group": f"{tag}_{group}", "detail": detail})
    return rows, skip


def build(root):
    raw = root / "_aihub_raw"
    labels, sources = {}, []  # sources: (tag, zip path)
    for lab in sorted((raw / "140").rglob("라벨링데이터.zip")):
        tag = "140tr" if "1.Training" in lab.as_posix() else "140"
        labels.setdefault(tag, {}).update(labels_140(lab))
    for z in sorted((raw / "140").rglob("*.zip")):
        if z.name != "라벨링데이터.zip":
            sources.append(("140tr" if "1.Training" in z.as_posix() else "140", z))
    for lab in sorted((raw / "71385").rglob("[TV]L_*.zip")):   # labels TL_x / VL_x pair with images TS_x / VS_x
        labels.setdefault("71385", {}).update(labels_71385(lab))
    sources += [("71385", z) for z in sorted((raw / "71385").rglob("[TV]S_*.zip"))]
    print({t: len(v) for t, v in labels.items()}, "labeled images;", len(sources), "image zips", flush=True)

    jobs = []
    for tag, zp in sources:
        if tag not in labels:
            continue
        with zipfile.ZipFile(zp) as z:
            names = [n for n in z.namelist() if n.lower().endswith((".jpg", ".jpeg", ".png"))]
        jobs += [(str(root), tag, str(zp), names[i:i + 400]) for i in range(0, len(names), 400)]
    for c in CLASSES:
        (root / c).mkdir(exist_ok=True)
    rows, skip = [], Counter()
    with ProcessPoolExecutor(max(1, (os.cpu_count() or 2) - 1), initializer=_init, initargs=(labels,)) as ex:
        for k, (r, s) in enumerate(ex.map(crop_chunk, jobs)):
            rows += r
            skip += s
            if k % 200 == 0:
                print(f"chunk {k}/{len(jobs)}: {len(rows)} crops", flush=True)

    # Keep rows of an existing manifest whose zips are no longer under _aihub_raw (raw data may be deleted
    # after a build), so adding a dataset later extends the manifest instead of replacing it.
    merged = {}
    mf = root / "manifest.csv"
    if mf.exists():
        with open(mf, encoding="utf-8-sig") as f:
            merged = {r["path"]: r for r in csv.DictReader(f)}
    merged.update({r["path"]: r for r in rows})
    assert len({r["path"] for r in rows}) == len(rows), "duplicate output names"
    missing = [p for p in merged if not (root / p).exists()]
    assert not missing, f"{len(missing)} manifest rows have no image file, e.g. {missing[:3]}"
    tmp = mf.with_suffix(".tmp")
    with open(tmp, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["path", "class", "source", "group", "detail"])
        w.writeheader()
        w.writerows(merged.values())
    os.replace(tmp, mf)
    by = Counter((r["source"], r["class"]) for r in merged.values())
    for k in sorted(by):
        print(*k, by[k])
    print(f"this run: {len(rows)} crops, skipped {dict(skip)} | manifest.csv: {len(merged)} rows")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["fetch", "build"])
    ap.add_argument("--root", default=str(Path(__file__).resolve().parent), help="project folder (default: this script's folder)")
    ap.add_argument("--parts", nargs="+", default=list(KEYS), choices=list(KEYS), help="fetch: which parts to download")
    ap.add_argument("--keys", nargs="+", default=[], metavar="DATASET:FILEKEY", help="fetch: only these files, e.g. 140:47680")
    a = ap.parse_args()
    root = Path(a.root).resolve()
    if a.command == "fetch":
        fetch(root, a.parts, a.keys)
    else:
        build(root)


if __name__ == "__main__":
    main()
