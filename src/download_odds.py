"""公式サイトのレースページから、3連単の全120通りの締切時オッズを集める。

使い方:
    python src/download_odds.py 2026-05-01 2026-11-07

保存先:
    data/odds/odds_20260501.csv  (1日1ファイル。列: date, venue_code, race_no, combo, odds)

どのレースがあったかは data/csv/races.csv（結果）から調べるので、先に parse.py を実行しておく。
すでにある日のファイルは飛ばすので、途中で止めても続きから再開できます。
"""
import csv
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "odds"
URL = "https://www.boatrace.jp/owpc/pc/race/odds3t?rno={rno}&jcd={jcd}&hd={hd}"
RE_ODDS = re.compile(r'class="oddsPoint[^"]*">([^<]*)<')
WORKERS = 8
WAIT_SEC = 0.5


def combos_in_page_order():
    """ページの表は「1着の艇番」が列、行は(2着,3着)の組。左上から右へ順に並ぶ。"""
    order = []
    for row in range(20):
        for first in range(1, 7):
            others = [b for b in range(1, 7) if b != first]
            second = others[row // 4]
            rest = [b for b in others if b != second]
            order.append(f"{first}-{second}-{rest[row % 4]}")
    return order


ORDER = combos_in_page_order()


def fetch_race(date: str, jcd: str, rno: int):
    url = URL.format(rno=rno, jcd=jcd, hd=date.replace("-", ""))
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                html = r.read().decode("utf-8", "replace")
            break
        except Exception:
            time.sleep(3 * (attempt + 1))
    else:
        return []
    time.sleep(WAIT_SEC)
    vals = RE_ODDS.findall(html)
    if len(vals) != 120:
        return []  # 中止・不成立などでオッズがないレース
    rows = []
    for combo, v in zip(ORDER, vals):
        try:
            odds = float(v.replace(",", ""))
        except ValueError:
            odds = None  # 欠場がらみで発売なし
        rows.append({"date": date, "venue_code": jcd, "race_no": rno, "combo": combo, "odds": odds})
    return rows


def main(start: str, end: str):
    races = pd.read_csv(ROOT / "data" / "csv" / "races.csv", dtype={"venue_code": str})
    races = races[(races.date >= start) & (races.date <= end)]
    OUT.mkdir(parents=True, exist_ok=True)
    for date, day in races.groupby("date"):
        out = OUT / f"odds_{date.replace('-', '')}.csv"
        if out.exists():
            continue
        keys = list(day[["venue_code", "race_no"]].itertuples(index=False))
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            results = list(ex.map(lambda k: fetch_race(date, k[0], int(k[1])), keys))
        rows = [r for res in results for r in res]
        tmp = out.with_suffix(".tmp")
        with tmp.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["date", "venue_code", "race_no", "combo", "odds"])
            w.writeheader()
            w.writerows(rows)
        tmp.rename(out)
        got = sum(1 for r in results if r)
        print(f"{date}: {got}/{len(keys)} レース", flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
