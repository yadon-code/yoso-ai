"""公式の「レーサー期別成績（ファン手帳）」を集めて広げる（半年ごと、1ファイル数MB）。

    python src/v3/collect_kibetsu.py

保存: data_full/kibetsu/fanYYMM.TXT（すでにあれば飛ばす。1ファイルごとに3秒待つ）
"""
import io
import re
import time
import urllib.request

import lhafile

from fetchutil import UA
from paths import KIBETSU

PAGE = "https://www.boatrace.jp/owpc/pc/extra/data/download.html"
BASE = "https://www.boatrace.jp"


def main():
    KIBETSU.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(PAGE, headers={"User-Agent": UA})
    html = urllib.request.urlopen(req, timeout=30).read().decode("utf-8", "replace")
    links = sorted(set(re.findall(r'href="(/static_extra/pc_static/download/data/kibetsu/fan(\d{4})\.lzh)"', html)))
    for href, yymm in links:
        if int(yymm[:2]) < 18:
            continue
        out = KIBETSU / f"fan{yymm}.TXT"
        if out.exists():
            continue
        try:
            body = urllib.request.urlopen(urllib.request.Request(BASE + href, headers={"User-Agent": UA}), timeout=60).read()
            arc = lhafile.Lhafile(io.BytesIO(body))
            out.write_bytes(arc.read(arc.namelist()[0]))
            print("期別成績", yymm, "ok", flush=True)
        except Exception as e:
            print("期別成績", yymm, "失敗", e, flush=True)
        time.sleep(3)


if __name__ == "__main__":
    main()
