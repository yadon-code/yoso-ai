"""公式の番組表(B)・競走成績(K)を data_full/raw に集める。

    python src/v3/collect_raw.py 2022-01-01 2026-11-25

- まず seed/raw.zip（2024-01-01〜2026-11-09）があれば広げる（通信なし）。
- 足りない日だけ公式から取る。download.py と同じ間隔（1ファイルごとに1秒待つ）で、同時は2つまで（本番より控えめ）。
- すでにある日は飛ばすので、止まっても続きから。
"""
import datetime as dt
import io
import sys
import time
import urllib.error
import urllib.request
import zipfile

import lhafile

from paths import RAW, REPO

BASE = "https://www1.mbrace.or.jp/od2/{kind}/{ym}/{kind_l}{ymd}.lzh"
WAIT_SEC = 1.0


def unpack_seed():
    seed = REPO / "seed" / "raw.zip"
    if not seed.exists():
        return 0
    n = 0
    with zipfile.ZipFile(seed) as z:
        for name in z.namelist():
            if not name.upper().endswith(".TXT"):
                continue
            out = RAW.parent / name  # raw/B/B240101.TXT
            if out.exists():
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(z.read(name))
            n += 1
    return n


def fetch(kind, day):
    out = RAW / kind / f"{kind}{day:%y%m%d}.TXT"
    if out.exists():
        return "skip"
    url = BASE.format(kind=kind, kind_l=kind.lower(), ym=f"{day:%Y%m}", ymd=f"{day:%y%m%d}")
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            body = r.read()
    except urllib.error.HTTPError as e:
        time.sleep(WAIT_SEC)
        return "none" if e.code == 404 else f"error:{e.code}"
    except Exception as e:
        time.sleep(WAIT_SEC * 3)
        return f"error:{e}"
    arc = lhafile.Lhafile(io.BytesIO(body))
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    tmp.write_bytes(arc.read(arc.namelist()[0]))
    tmp.rename(out)
    time.sleep(WAIT_SEC)
    return "ok"


def main(start, end):
    print("seed から広げた:", unpack_seed(), flush=True)
    d, last = dt.date.fromisoformat(start), dt.date.fromisoformat(end)
    while d <= last:
        res = [fetch(k, d) for k in ("B", "K")]
        if res != ["skip", "skip"]:
            print(d, *res, flush=True)
        d += dt.timedelta(days=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
