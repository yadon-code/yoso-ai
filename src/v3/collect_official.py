"""BOATCAST に無い日（2024年1〜4月ごろ）だけ、公式サイトのページから 3連単の確定オッズ と 直前情報 を集める。

    python src/v3/collect_official.py 2024-01-01 2024-04-30 [--rate 1]

- data_full/bc に「無し（no_bc）」と記録された日だけが対象（BOATCAST を先に動かす）。
- 公式サイトは1ページが重いので、全体で1秒に1回まで（前の download_odds.py の8並列より控えめ）。
- 保存: data_full/official/YYYY/of_YYYYMMDD.jsonl.gz（1行=1レース。odds3t={組: オッズ}、bi=[艇ごとの直前情報]）
"""
import paths  # noqa: F401  src/ の本番のモジュールを読めるようにする（先に読む）
import argparse
import gzip
import json
import re
from concurrent.futures import ThreadPoolExecutor

import download_beforeinfo as dbi
import download_odds as dod
from collect_bc import path_of as bc_path, read_day
from fetchutil import ERR, RateLimiter, get_text
from paths import OFFICIAL
from racelist import days, races_of


def path_of(day):
    return OFFICIAL / f"{day:%Y}" / f"of_{day:%Y%m%d}.jsonl.gz"


def one(lim, date, jo, rno):
    hd = date.replace("-", "")
    h1 = get_text(dod.URL.format(rno=rno, jcd=jo, hd=hd), lim, timeout=30)
    h2 = get_text(dbi.URL.format(rno=rno, jcd=jo, hd=hd), lim, timeout=30)
    row = {"date": date, "venue_code": jo, "race_no": rno, "odds3t": None, "bi": None}
    if h1 == ERR or h2 == ERR:
        row["err"] = True
    if h1 and h1 != ERR:
        vals = dod.RE_ODDS.findall(h1)
        if len(vals) == 120:
            row["odds3t"] = {c: (float(v.replace(",", "")) if re.match(r"^[\d.,]+$", v) else None)
                             for c, v in zip(dod.ORDER, vals)}
    if h2 and h2 != ERR:
        try:
            row["bi"] = dbi.parse(h2) or None
        except Exception:
            row["bi"] = None
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("start")
    ap.add_argument("end")
    ap.add_argument("--rate", type=float, default=1.0)
    a = ap.parse_args()
    lim = RateLimiter(a.rate)
    for day in days(a.start, a.end):
        bp = bc_path(day)
        if not bp.exists() or not read_day(bp)[0].get("no_bc"):
            continue
        p = path_of(day)
        old = {}
        if p.exists():
            with gzip.open(p, "rt", encoding="utf-8") as f:
                old = {(r["venue_code"], r["race_no"]): r for r in map(json.loads, f)}
            if not any(r.get("err") for r in old.values()):
                continue
        races = races_of(day)
        todo = [k for k in races if k not in old or old[k].get("err")]
        with ThreadPoolExecutor(max_workers=2) as ex:
            got = list(ex.map(lambda k: one(lim, day.isoformat(), k[0], k[1]), todo))
        for k, r in zip(todo, got):
            old[k] = r
        rows = [old[k] for k in races if k in old]
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        with gzip.open(tmp, "wt", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        tmp.rename(p)
        print(day, f"{len(rows)}R オッズ{sum(1 for r in rows if r['odds3t'])} 直前{sum(1 for r in rows if r['bi'])}"
              f" 失敗{sum(1 for r in rows if r.get('err'))}", flush=True)


if __name__ == "__main__":
    main()
