"""Download (or repair) the AI Hub dataset 140 Training split for the 5 needed categories.

AI Hub's download server drops connections mid-file and does not support resuming, so every file is fetched
whole into its own folder (_aihub_raw/140/k_<filekey>/), retried from scratch on failure, and accepted only
if the merged zip opens and its members fit inside the file. Files that already exist and validate anywhere
under _aihub_raw/140 are kept. API key: environment variable AIHUB_APIKEY, or a file `.aihub_key` next to this script.

usage: python fetch_140_train.py [--then-process]     (--then-process runs add_140_train.py when all 31 files are valid)
"""
import os
import re
import shutil
import subprocess
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "_aihub_raw" / "140"
FILES = {  # filekey -> (category folder, zip name) under 1.Training/원천데이터; None = the label zip
    47869: None,
    47831: ("유리병", "기타"), 47832: ("유리병", "기타술병"), 47833: ("유리병", "맥주병"), 47744: ("유리병", "물병"),
    47745: ("유리병", "박카스병"), 47746: ("유리병", "소주병"), 47747: ("유리병", "음료수병"), 47748: ("유리병", "주방용기"),
    47820: ("비닐", "과자봉지"), 47821: ("비닐", "기타"), 47822: ("비닐", "리필용기"), 47823: ("비닐", "봉투"),
    47824: ("비닐", "에어캡"), 47825: ("비닐", "일회용덮개"), 47826: ("비닐", "포장제"),
    47709: ("캔류", "기타"), 47710: ("캔류", "맥주캔"), 47711: ("캔류", "스팸류"), 47712: ("캔류", "음료수캔"),
    47713: ("캔류", "참기름캔"), 47714: ("캔류", "커피캔"), 47715: ("캔류", "통조림캔"),
    47716: ("페트병", "기타"), 47717: ("페트병", "일회용음료수잔"), 47718: ("페트병", "페트병"),
    47719: ("플라스틱류", "기타"), 47720: ("플라스틱류", "대용량플라스틱통"), 47721: ("플라스틱류", "밀폐용기"),
    47722: ("플라스틱류", "바구니"), 47723: ("플라스틱류", "욕실용품"), 47724: ("플라스틱류", "장남감"),
}


def rel(key):
    f = FILES[key]
    return "1.Training/라벨링데이터/라벨링데이터.zip" if f is None else f"1.Training/원천데이터/{f[0]}/{f[1]}/{f[1]}.zip"


def valid_zip(p):
    """Opens, has members, and the members' compressed bytes fit in the file (a truncated zip fails here)."""
    try:
        with zipfile.ZipFile(p) as z:
            infos = z.infolist()
            return len(infos) > 0 and sum(i.compress_size for i in infos) <= os.path.getsize(p)
    except Exception:
        return False


def find_valid(key):
    good = None
    for c in RAW.rglob(rel(key)):
        if good is None and valid_zip(c):
            good = c
        elif not valid_zip(c):
            c.unlink()  # truncated or merged from incomplete parts
    return good


def fetch(key):
    d = RAW / f"k_{key}"
    api = os.environ.get("AIHUB_APIKEY") or (ROOT / ".aihub_key").read_text().strip()
    for attempt in range(1, 9):
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)
        # the key goes in through stdin (curl -K -), not argv, so other users of the machine cannot see it in `ps`
        r = subprocess.run(["curl", "-sS", "-L", "--fail", "-o", "dl.tar", "-K", "-",
                            f"https://api.aihub.or.kr/down/0.6/140.do?fileSn={key}"],
                           input=f'header = "apikey:{api}"\n', cwd=d, capture_output=True, text=True)
        if r.returncode == 0 and subprocess.run(["tar", "-xf", "dl.tar"], cwd=d, capture_output=True).returncode == 0:
            (d / "dl.tar").unlink()
            groups = {}
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
            got = find_valid(key)
            if got:
                print(f"[{key}] ok after {attempt} attempt(s): {got.relative_to(RAW)} ({os.path.getsize(got) / 1e9:.2f} GB)", flush=True)
                return True
            print(f"[{key}] attempt {attempt}: downloaded but zip invalid", flush=True)
        else:
            print(f"[{key}] attempt {attempt}: curl exit {r.returncode} {r.stderr.strip()[:90]}", flush=True)
        time.sleep(30)
    shutil.rmtree(d, ignore_errors=True)
    print(f"[{key}] FAILED after 8 attempts", flush=True)
    return False


def main():
    for junk in list(RAW.rglob("dl_*.tar")) + list(RAW.rglob("*.part*")):
        junk.unlink()  # leftovers of dropped transfers
    todo = [k for k in FILES if not find_valid(k)]
    print(f"valid already: {len(FILES) - len(todo)}/{len(FILES)}; to fetch: {todo}", flush=True)
    with ThreadPoolExecutor(3) as ex:
        ok = list(ex.map(fetch, sorted(todo, key=lambda k: k not in (47718, 47826, 47721, 47712))))  # big files first
    missing = [k for k in FILES if not find_valid(k)]
    total = sum(os.path.getsize(find_valid(k)) for k in FILES if k not in missing)
    print(f"DONE: {len(FILES) - len(missing)}/{len(FILES)} valid, {total / 1e9:.1f} GB; missing: {missing}", flush=True)
    if not missing and "--then-process" in sys.argv:
        sys.exit(subprocess.run([sys.executable, str(ROOT / "add_140_train.py")]).returncode)
    sys.exit(1 if missing else 0)


if __name__ == "__main__":
    main()
