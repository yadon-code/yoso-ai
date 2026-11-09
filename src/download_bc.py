"""公式映像サイト「BOATCAST」のデータファイルから、
3連単の確定オッズ と オリジナル展示（一周・まわり足・直線）を集める。
（公式サイトのページより速く取れる）

使い方:
    python src/download_bc.py 2025-01-01 2026-11-08

保存先（1日1ファイル、新しい日から順に取る。すでにある日は飛ばす）:
    data/odds/odds_YYYYMMDD.csv      列: date, venue_code, race_no, combo, odds
    data/oriten/oriten_YYYYMMDD.csv  列: date, venue_code, race_no, boat, lap, turn, straight, lap_kind
        lap      = 一周タイム（場によっては半周ラップ）
        turn     = まわり足タイム
        straight = 直線タイム
        ※場によって測っていない項目は空欄。測り方も場ごとに違うので、同じレースの中で比べて使う
"""
import csv
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
BASE = "https://race.boatcast.jp/txt/{jo}/{name}_{ymd}_{jo}_{rno:02d}.txt"
WORKERS = 12
NAME_MAP = {"一周": "lap", "半周ラップ": "lap", "まわり足": "turn", "直線": "straight"}


def get(url):
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=20) as r:
                t = r.read().decode("utf-8", "replace")
            return t if t.startswith("data=") else None
        except Exception:
            time.sleep(2 * (attempt + 1))
    return None


def parse_odds(text):
    """6行（1着の艇番ごと）×20個（2着・3着の組み合わせ順）"""
    lines = [l.split("\t") for l in text.strip().split("\n")[2:]]
    if len(lines) != 6:
        return None
    out = {}
    for i, row in enumerate(lines):
        first = i + 1
        others = [b for b in range(1, 7) if b != first]
        combos = [(s, t) for s in others for t in others if t != s]
        for k, (s, t) in enumerate(combos):
            try:
                v = float(row[1 + k])
            except (ValueError, IndexError):
                v = None
            out[f"{first}-{s}-{t}"] = v if v and v > 0 else None
    return out


def parse_oriten(text):
    lines = text.strip().split("\n")
    if len(lines) < 4:
        return None
    names = [n.replace("　", "").strip() for n in lines[2].split("\t")]
    keys = [NAME_MAP.get(n) for n in names]
    rows = []
    for l in lines[3:9]:
        c = l.split("\t")
        if not c[0].strip().isdigit():
            continue
        r = {"boat": int(c[0]), "lap": None, "turn": None, "straight": None,
             "lap_kind": names[0] if names else ""}
        for k, v in zip(keys, c[2:]):
            if k:
                try:
                    r[k] = float(v)
                except ValueError:
                    pass
        rows.append(r)
    return rows


def fetch_race(date, jo, rno):
    ymd = date.replace("-", "")
    o = get(BASE.format(jo=jo, name="bc_kakutei_od3", ymd=ymd, rno=rno))
    t = get(BASE.format(jo=jo, name="bc_oriten", ymd=ymd, rno=rno))
    odds = parse_odds(o) if o else None
    ori = parse_oriten(t) if t else None
    key = {"date": date, "venue_code": jo, "race_no": rno}
    odds_rows = [{**key, "combo": c, "odds": v} for c, v in odds.items()] if odds else []
    ori_rows = [{**key, **r} for r in ori] if ori else []
    return odds_rows, ori_rows


def write(path, rows, fields):
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    tmp.rename(path)


def main(start, end):
    races = pd.read_csv(ROOT / "data" / "csv" / "races.csv", dtype={"venue_code": str})
    races = races[(races.date >= start) & (races.date <= end)]
    (ROOT / "data" / "odds").mkdir(parents=True, exist_ok=True)
    (ROOT / "data" / "oriten").mkdir(parents=True, exist_ok=True)
    for date, day in sorted(races.groupby("date"), reverse=True):
        d = date.replace("-", "")
        p_odds = ROOT / "data" / "odds" / f"odds_{d}.csv"
        p_ori = ROOT / "data" / "oriten" / f"oriten_{d}.csv"
        if p_odds.exists() and p_ori.exists():
            continue
        keys = list(day[["venue_code", "race_no"]].itertuples(index=False))
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            res = list(ex.map(lambda k: fetch_race(date, k[0], int(k[1])), keys))
        if not p_odds.exists():
            write(p_odds, [r for o, _ in res for r in o], ["date", "venue_code", "race_no", "combo", "odds"])
        write(p_ori, [r for _, t in res for r in t],
              ["date", "venue_code", "race_no", "boat", "lap", "turn", "straight", "lap_kind"])
        n_o = sum(1 for o, _ in res if o)
        n_t = sum(1 for _, t in res if t)
        print(f"{date}: オッズ {n_o}/{len(keys)}  オリジナル展示 {n_t}/{len(keys)}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
