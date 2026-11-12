"""過去の SG・G1 のレースだけ、3連単の締切オッズと直前情報を公式サイトから集める（2019年1月〜）。

使い方:
    python src/collect_grade_history.py 2019-01-01 2024-12-31

先に src/download_grade.py で data/csv/grades.csv を作っておく。
保存先（毎日の全レース分のファイルとは名前を分ける。読み込み側は odds_*.csv / bi_*.csv でまとめて読める）:
    data/odds/odds_g_YYYYMMDD.csv        3連単120通りの締切オッズ
    data/beforeinfo/bi_g_YYYYMMDD.csv    直前情報（展示タイム・チルト・スタート展示など）
すでにある日は飛ばすので、途中で止めても続きから再開できる。公式サイトに残っているのは2019年1月ごろから。
"""
import csv
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import download_beforeinfo as dbi  # noqa: E402
import download_odds as dod  # noqa: E402
from grade import BIG  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
WORKERS = 12


def write(path, rows, fields):
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    tmp.rename(path)


def main(start, end):
    g = pd.read_csv(ROOT / "data" / "csv" / "grades.csv", dtype={"jcd": str})
    g = g[g.grade.isin(BIG) & (g.date >= start) & (g.date <= end)]
    for d in ["odds", "beforeinfo"]:
        (ROOT / "data" / d).mkdir(parents=True, exist_ok=True)
    for date, day in sorted(g.groupby("date"), reverse=True):
        ymd = date.replace("-", "")
        p_o = ROOT / "data" / "odds" / f"odds_g_{ymd}.csv"
        p_b = ROOT / "data" / "beforeinfo" / f"bi_g_{ymd}.csv"
        if p_o.exists() and p_b.exists():
            continue
        keys = [(jcd, rno) for jcd in sorted(set(day.jcd)) for rno in range(1, 13)]

        def one(k):
            return dod.fetch_race(date, k[0], k[1]), dbi.fetch_race(date, k[0], k[1])

        with ThreadPoolExecutor(WORKERS) as ex:
            res = list(ex.map(one, keys))
        write(p_o, [r for o, _ in res for r in o], ["date", "venue_code", "race_no", "combo", "odds"])
        write(p_b, [r for _, b in res for r in b], dbi.FIELDS)
        n_o = sum(1 for o, _ in res if any(r["odds"] for r in o))
        print(f"{date} {','.join(sorted(set(day.jcd)))}: オッズ {n_o}/{len(keys)}  直前 {sum(1 for _, b in res if b)}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
