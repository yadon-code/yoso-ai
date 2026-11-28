"""締切前の3連単オッズを、締切の 15・10・5・2・1 分前に記録する（回収率AI の「確定オッズの予想」を学ぶため）。
本番（live.py）とは別に、Mac で1日動かす。本番の通知・記録には触らない。

    python src/v3/snap_odds.py              # 今日（最終レースの締切まで動く）
    python src/v3/snap_odds.py 2026-11-27

- 出走表（番組表 B）から締切時刻を読む（無ければ公式からその日の B を取る）。
- 1回 = 公式サイトの3連単オッズのページ1枚。1日約150レース×5回 ≒ 750回（本番の取得と同じページ・同じ程度）。
- 保存: data_full/odds_snap/snap_YYYYMMDD.jsonl（1行=1回。odds は backtest.COMBOS の順、発売なしは null）
- 途中で止めてもう一度動かすと、残りの時刻から続ける（取った分は残る）。
"""
import datetime as dt
import json
import sys
import time

import paths  # noqa: F401
import download_odds as dod
import parse
from collect_raw import fetch as fetch_raw
from fetchutil import ERR, RateLimiter, get_text
from paths import DATA, RAW

JST = dt.timezone(dt.timedelta(hours=9))
OFFSETS_MIN = [15, 10, 5, 2, 1]
OUT = DATA / "odds_snap"


def schedule(day):
    p = RAW / "B" / f"B{day:%y%m%d}.TXT"
    if not p.exists():
        print("番組表を取ります:", fetch_raw("B", day), flush=True)
    rows = parse.parse_programs(p) if p.exists() else []
    races = {}
    for r in rows:
        if r.get("deadline"):
            h, m = map(int, r["deadline"].split(":"))
            races[(r["venue_code"], int(r["race_no"]))] = dt.datetime(day.year, day.month, day.day, h, m, tzinfo=JST)
    return races


def main():
    day = dt.date.fromisoformat(sys.argv[1]) if len(sys.argv) > 1 else dt.datetime.now(JST).date()
    races = schedule(day)
    if not races:
        print("この日のレースが見つかりません", day)
        return
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"snap_{day:%Y%m%d}.jsonl"
    done = set()
    if path.exists():
        for l in path.read_text(encoding="utf-8").splitlines():
            r = json.loads(l)
            done.add((r["venue_code"], r["race_no"], r["target_min"]))
    jobs = sorted((dl - dt.timedelta(minutes=m), jo, rno, m, dl) for (jo, rno), dl in races.items() for m in OFFSETS_MIN
                  if (jo, rno, m) not in done)
    lim = RateLimiter(1.0)
    print(f"{day}: {len(races)} レース、記録の予定 {len(jobs)} 回", flush=True)
    for when, jo, rno, m, dl in jobs:
        now = dt.datetime.now(JST)
        if when < now - dt.timedelta(seconds=30):
            continue  # 過ぎた時刻は取らない（あとからでは同じものは取れない）
        if when > now:
            time.sleep((when - now).total_seconds())
        html = get_text(dod.URL.format(rno=rno, jcd=jo, hd=f"{day:%Y%m%d}"), lim, tries=2, timeout=20)
        t = dt.datetime.now(JST)
        odds = None
        if html and html != ERR:
            vals = dod.RE_ODDS.findall(html)
            if len(vals) == 120:
                d = {}
                for c, v in zip(dod.ORDER, vals):
                    try:
                        d[c] = float(v.replace(",", ""))
                    except ValueError:
                        d[c] = None
                odds = [d[c] for c in sorted(d, key=_order)]  # backtest.COMBOS の順
        rec = {"date": day.isoformat(), "venue_code": jo, "race_no": rno, "target_min": m,
               "fetched_at": t.strftime("%Y-%m-%d %H:%M:%S"), "seconds_before": int((dl - t).total_seconds()),
               "ok": odds is not None, "odds": odds}
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


_COMBOS = None


def _order(c):
    global _COMBOS
    if _COMBOS is None:
        from itertools import permutations
        _COMBOS = {f"{a}-{b}-{c}": i for i, (a, b, c) in enumerate(permutations(range(1, 7), 3))}
    return _COMBOS[c]


if __name__ == "__main__":
    main()
