"""開催ごとのグレード（SG/G1/G2/G3/一般）を data_full/tables/grades.csv に集める（src/download_grade.py の読み方を使う）。
1日1ページ。全体で2秒に1回まで（本番の8並列より控えめ）。すでにある日は飛ばす。

    python src/v3/collect_grade.py 2022-01-01 2026-11-25
"""
import paths  # noqa: F401  src/ の本番のモジュールを読めるようにする（先に読む）
import sys
import time

import download_grade as dg
from paths import TABLES

dg.OUT = TABLES / "grades.csv"
dg.WORKERS = 1
_orig = dg.fetch_day


def slow(d):
    r = _orig(d)
    time.sleep(2)
    return r


dg.fetch_day = slow

if __name__ == "__main__":
    dg.main()
