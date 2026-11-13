"""開催ごとのグレード（SG / G1 / G2 / G3 / 一般）を公式サイトの「本日のレース」ページから集める。

使い方:
    python src/download_grade.py 2024-01-01 2026-11-12

出力: data/csv/grades.csv（date, jcd, grade） 1行 = 1日 × 1場
既にある日付は取り直さない（追記）。1日あたり1ページだけなので軽い。
"""
import csv
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path

import urllib.request

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "csv" / "grades.csv"
URL = "https://www.boatrace.jp/owpc/pc/race/index?hd={hd}"
WORKERS = 8

# 公式ページのクラス名 → グレード（a/b は表示用の色違いなので同じ扱い）
GRADE_MAP = {"SG": "SG", "PG1": "G1", "G1": "G1", "G2": "G2", "G3": "G3", "ippan": "一般"}


def norm_grade(cls: str) -> str:
    base = re.sub(r"[ab]$", "", cls)
    return GRADE_MAP.get(base, base)


def fetch_day(d: date):
    hd = d.strftime("%Y%m%d")
    for attempt in range(4):
        try:
            req = urllib.request.Request(URL.format(hd=hd), headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as r:
                text = r.read().decode("utf-8", "replace")
            break
        except Exception:
            time.sleep(2 * (attempt + 1))
    else:
        return d, None
    rows = []
    for tb in re.findall(r"<tbody>(.*?)</tbody>", text, re.S):
        j = re.search(r"jcd=(\d\d)", tb)
        g = re.search(r'class="[^"]*\bis-(SG[ab]?|PG1[ab]?|G[123][ab]?|ippan)\b', tb)
        if j and g:
            rows.append((d.isoformat(), j.group(1), norm_grade(g.group(1))))
    return d, rows


def main():
    start = date.fromisoformat(sys.argv[1])
    end = date.fromisoformat(sys.argv[2]) if len(sys.argv) > 2 else date.today()
    done = set()
    if OUT.exists():
        with OUT.open(encoding="utf-8") as f:
            done = {r["date"] for r in csv.DictReader(f)}
    days = [start + timedelta(i) for i in range((end - start).days + 1)]
    days = [d for d in days if d.isoformat() not in done]
    new = not OUT.exists()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("a", newline="", encoding="utf-8") as f, ThreadPoolExecutor(WORKERS) as ex:
        w = csv.writer(f)
        if new:
            w.writerow(["date", "jcd", "grade"])
        for i, (d, rows) in enumerate(ex.map(fetch_day, days)):
            if rows is None:
                print("取得失敗:", d, file=sys.stderr)
                continue
            w.writerows(rows)
            if i % 50 == 0:
                print(d, len(rows), "場", flush=True)


if __name__ == "__main__":
    main()
