"""公式サイトの「直前情報」ページから、レース直前にわかる情報を集める。

使い方:
    python src/download_beforeinfo.py 2026-05-01 2026-11-07

1艇ごとに取る情報:
    当日体重、展示タイム、チルト角度、プロペラ交換（新）、部品交換の数と種類、調整重量、
    スタート展示の進入コースとST
レースごとに取る情報:
    気温、天候、風速、風向き、水温、波高

保存先: data/beforeinfo/bi_20260501.csv（1日1ファイル、1行=1艇。レースの情報は各行に付ける）
どのレースがあったかは data/csv/races.csv から調べる。すでにある日は飛ばす。
"""
import csv
import html as htmllib
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "beforeinfo"
URL = "https://www.boatrace.jp/owpc/pc/race/beforeinfo?rno={rno}&jcd={jcd}&hd={hd}"
WORKERS = 8
WAIT_SEC = 0.5
FIELDS = ["date", "venue_code", "race_no", "boat", "weight_day", "exhibit_time_bi", "tilt",
          "propeller_new", "parts_n", "parts", "adjust_weight", "exh_course", "exh_st", "exh_st_f",
          "temp", "weather_bi", "wind_speed_bi", "wind_dir_bi", "water_temp", "wave_bi"]


def num(s):
    m = re.search(r"-?\d*\.?\d+", s or "")
    return float(m.group()) if m else None


def strip(s):
    return htmllib.unescape(re.sub(r"<[^>]+>", "", s)).strip()


def parse(html: str):
    boats = {}
    for tb in re.findall(r'<tbody class="is-fs12 ?">(.*?)</tbody>', html, re.S):
        tds = re.findall(r"<td[^>]*>(.*?)</td>", tb, re.S)
        if len(tds) < 10:
            continue
        boat = int(strip(tds[0]))
        parts = re.findall(r'<span class="label4[^"]*">([^<]+)</span>', tb)
        boats[boat] = {
            "boat": boat,
            "weight_day": num(strip(tds[3])),
            "exhibit_time_bi": num(strip(tds[4])),
            "tilt": num(strip(tds[5])),
            "propeller_new": 1 if "新" in strip(tds[6]) else 0,
            "parts_n": len(parts),
            "parts": "|".join(p.strip() for p in parts),
            # 調整重量は3行目の最初のセル
            "adjust_weight": num(strip(re.findall(r"<tr>(.*?)</tr>", tb, re.S)[2].split("</td>")[0])),
        }
    if len(boats) != 6:
        return []
    # スタート展示: 上から順に 1コース, 2コース, ...
    st_block = html[html.find("スタート展示"):]
    st_items = re.findall(
        r'table1_boatImage1Number[^>]*>(\d)</span>.*?table1_boatImage1Time[^>]*>([^<]*)</span>',
        st_block, re.S)[:6]
    for course, (b, st) in enumerate(st_items, start=1):
        b = int(b)
        if b in boats:
            boats[b]["exh_course"] = course
            boats[b]["exh_st_f"] = 1 if "F" in st else 0
            v = num(st.replace("F", ""))
            boats[b]["exh_st"] = -v if (v is not None and "F" in st) else v
    w = html[html.find("水面気象情報"):]
    data = dict(re.findall(
        r'LabelTitle">([^<]*)</span>\s*<span class="weather1_bodyUnitLabelData">([^<]*)</span>', w))
    titles = re.findall(r'LabelTitle">([^<]*)</span>', w)
    wd = re.search(r'is-windDirection">\s*<p class="weather1_bodyUnitImage is-wind(\d+)"', w)
    race = {
        "temp": num(data.get("気温")),
        "weather_bi": titles[1] if len(titles) > 1 else None,
        "wind_speed_bi": num(data.get("風速")),
        "wind_dir_bi": int(wd.group(1)) if wd else None,  # 1〜16の方角番号（公式サイトの表示）
        "water_temp": num(data.get("水温")),
        "wave_bi": num(data.get("波高")),
    }
    return [{**b, **race} for b in boats.values()]


def fetch_race(date, jcd, rno):
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
    try:
        rows = parse(html)
    except Exception:
        return []
    return [{"date": date, "venue_code": jcd, "race_no": rno, **r} for r in rows]


def main(start, end):
    races = pd.read_csv(ROOT / "data" / "csv" / "races.csv", dtype={"venue_code": str})
    races = races[(races.date >= start) & (races.date <= end)]
    OUT.mkdir(parents=True, exist_ok=True)
    for date, day in sorted(races.groupby("date"), reverse=True):  # 新しい日から
        out = OUT / f"bi_{date.replace('-', '')}.csv"
        if out.exists():
            continue
        keys = list(day[["venue_code", "race_no"]].itertuples(index=False))
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            results = list(ex.map(lambda k: fetch_race(date, k[0], int(k[1])), keys))
        tmp = out.with_suffix(".tmp")
        with tmp.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            for res in results:
                w.writerows(res)
        tmp.rename(out)
        print(f"{date}: {sum(1 for r in results if r)}/{len(keys)} レース", flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
