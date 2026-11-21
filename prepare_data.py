"""Copy _aihub_4class into 4 class folders at the project root, shorter side resized to 256.

Output: <root>/{vinyl,glass,plastic,can}/*.jpg and <root>/manifest.csv (path, class, source, group, detail).
train.py splits train/val by `group`; `detail` keeps the item type (e.g. 통조림캔 for steel cans).
Smaller images are not upscaled. Reruns skip files that already exist.
"""
import csv
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent
CLASSES = ("vinyl", "glass", "plastic", "can")
SHORT = 256


def resize(job):
    src, dst = job
    if os.path.exists(dst):
        return True
    try:
        im = Image.open(src)
        im.draft("RGB", (SHORT, SHORT))  # JPEG: decode at reduced scale when the image is large
        im = im.convert("RGB")
        s = SHORT / min(im.size)
        if s < 1:
            im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
        im.save(dst, quality=95)
        return True
    except Exception as e:
        print("FAIL", src, e)
        return False


def main():
    rows, jobs = [], []
    with open(ROOT / "_aihub_4class" / "manifest.csv", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            name = Path(r["path"]).name
            jobs.append((str(ROOT / "_aihub_4class" / r["path"]), str(ROOT / r["class"] / name)))
            rows.append({"path": f"{r['class']}/{name}", "class": r["class"], "source": r["source"],
                         "group": r["group"], "detail": r["detail"]})
    assert len({r["path"] for r in rows}) == len(rows), "output name collision"
    for c in CLASSES:
        (ROOT / c).mkdir(exist_ok=True)
    with ProcessPoolExecutor(max(1, (os.cpu_count() or 2) - 1)) as ex:
        ok = list(ex.map(resize, jobs, chunksize=256))
    rows = [r for r, good in zip(rows, ok) if good]

    with open(ROOT / "manifest.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    on_disk = sum(len(os.listdir(ROOT / c)) for c in CLASSES)
    assert on_disk == len(rows), f"manifest {len(rows)} != files on disk {on_disk}"
    print(f"{len(rows)} images, failed {ok.count(False)}; manifest matches files on disk")


if __name__ == "__main__":
    main()
