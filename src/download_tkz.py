"""直前情報（展示タイム・体重・調整重量・チルト・スタート展示）を BOATCAST から集める。
公式サイトのページ（download_beforeinfo.py、1レース約10秒）より15倍ほど速い。

使い方:
    python src/download_tkz.py 2025-01-01 2026-11-08

保存先: data/beforeinfo/bi_YYYYMMDD.csv（download_beforeinfo.py と同じ列。すでにある日は飛ばす）
公式サイト版との違い:
    - プロペラ交換・部品交換・気温・水温は取れないので空欄
    - 天候・風・波は、レース結果（races.csv）の値を入れる（発走時の値。締切前とほぼ同じ）
    - source 列が "bc"
"""
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from download_bc import WORKERS, get, write  # noqa: E402
from features import WIND_DIRS  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
URL = "https://race.boatcast.jp/hp_txt/{jo}/bc_j_tkz_{ymd}_{jo}_{rno:02d}.txt"
FIELDS = ["date", "venue_code", "race_no", "boat", "weight_day", "exhibit_time_bi", "tilt",
          "propeller_new", "parts_n", "parts", "adjust_weight", "exh_course", "exh_st", "exh_st_f",
          "temp", "weather_bi", "wind_speed_bi", "wind_dir_bi", "water_temp", "wave_bi", "source"]


def num(s):
    try:
        return float(s.replace(" ", ""))
    except (ValueError, AttributeError):
        return None


def parse(text):
    """2〜7行目が1〜6号艇（名前, 展示, 調整あり, 調整重量x10, 体重, ?, チルト）、8行目がスタート展示（進入順に 艇,ST,F）"""
    lines = text.strip("\n").split("\n")
    if len(lines) < 8:
        return None
    rows = []
    for i, l in enumerate(lines[2:8]):
        c = l.split("\t")
        if len(c) < 7:
            return None
        ex = num(c[1])
        adj = num(c[3])
        rows.append({"boat": i + 1, "exhibit_time_bi": ex if ex else None, "weight_day": num(c[4]),
                     "adjust_weight": adj / 10 if adj is not None else None, "tilt": num(c[6])})
    st = [x.strip() for x in lines[8].split("\t")] if len(lines) > 8 else []
    for k in range(0, len(st) - 1, 3):
        if not st[k].isdigit():
            continue
        b = int(st[k])
        if not 1 <= b <= 6:
            continue
        v = num(st[k + 1])
        f = k + 2 < len(st) and st[k + 2] == "F"
        r = rows[b - 1]
        r["exh_course"] = k // 3 + 1
        r["exh_st"] = (-v if f else v) if v is not None else None
        r["exh_st_f"] = int(f)
    return rows


def fetch_race(date, jo, rno):
    t = get(URL.format(jo=jo, ymd=date.replace("-", ""), rno=rno))
    rows = parse(t) if t else None
    return [{"date": date, "venue_code": jo, "race_no": rno, **r} for r in rows] if rows else []


def main(start, end):
    races = pd.read_csv(ROOT / "data" / "csv" / "races.csv", dtype={"venue_code": str})
    races = races[(races.date >= start) & (races.date <= end)]
    out_dir = ROOT / "data" / "beforeinfo"
    out_dir.mkdir(parents=True, exist_ok=True)
    for date, day in sorted(races.groupby("date"), reverse=True):
        path = out_dir / f"bi_{date.replace('-', '')}.csv"
        if path.exists():
            continue
        keys = list(day[["venue_code", "race_no", "weather", "wind_dir", "wind_speed", "wave_cm"]]
                    .itertuples(index=False))
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            res = list(ex.map(lambda k: fetch_race(date, k[0], int(k[1])), keys))
        rows = []
        for k, rs in zip(keys, res):
            wd = WIND_DIRS.index(k.wind_dir) + 1 if k.wind_dir in WIND_DIRS else (17 if k.wind_dir == "無風" else None)
            for r in rs:
                r.update({"weather_bi": k.weather, "wind_speed_bi": k.wind_speed, "wind_dir_bi": wd,
                          "wave_bi": k.wave_cm, "source": "bc"})
                rows.append(r)
        write(path, rows, FIELDS)
        print(f"{date}: {sum(1 for r in res if r)}/{len(keys)} レース", flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
