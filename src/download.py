"""公式サイト（BOAT RACE）から番組表(B)と競走成績(K)をダウンロードして解凍する。

使い方:
    python src/download.py 2024-01-01 2024-12-31

保存先:
    data/raw/B/B240101.TXT  (番組表＝出走表)
    data/raw/K/K240101.TXT  (競走成績＝結果)

すでにあるファイルは飛ばすので、途中で止めても続きから再開できます。
サーバーに負担をかけないよう、1ファイルごとに少し待ちます。
"""
import datetime as dt
import io
import sys
import time
from concurrent.futures import ThreadPoolExecutor
import urllib.error
import urllib.request
from pathlib import Path

import lhafile

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
BASE = "https://www1.mbrace.or.jp/od2/{kind}/{ym}/{kind_l}{ymd}.lzh"
WAIT_SEC = 1.0


def fetch(kind: str, day: dt.date) -> str:
    out_dir = RAW / kind
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"{kind}{day:%y%m%d}.TXT"
    out = out_dir / name
    if out.exists():
        return "skip"
    url = BASE.format(kind=kind, kind_l=kind.lower(), ym=f"{day:%Y%m}", ymd=f"{day:%y%m%d}")
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            body = r.read()
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return "none"  # その日は開催なし、またはまだ公開されていない
        raise
    arc = lhafile.Lhafile(io.BytesIO(body))
    inner = arc.namelist()[0]
    out.write_bytes(arc.read(inner))
    time.sleep(WAIT_SEC)
    return "ok"


WORKERS = 4  # 同時にダウンロードする数（サーバーの応答が遅いので少しだけ並列にする）


def fetch_day(d: dt.date) -> None:
    res = []
    for k in ("B", "K"):
        try:
            res.append(fetch(k, d))
        except Exception as e:  # 一時的なエラーは記録して先へ進む（再実行すれば取り直せる）
            res.append(f"error:{e}")
    print(d, *res, flush=True)


def main(start: str, end: str) -> None:
    d = dt.date.fromisoformat(start)
    last = dt.date.fromisoformat(end)
    days = []
    while d <= last:
        days.append(d)
        d += dt.timedelta(days=1)
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        list(ex.map(fetch_day, days))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
