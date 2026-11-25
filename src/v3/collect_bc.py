"""BOATCAST（公式映像サイト）のデータファイルを、1レースずつ生のまま集める（あとで何度でも読み直せるように）。

    python src/v3/collect_bc.py 2024-01-01 2026-11-25 [--rate 4]

1レースで取るもの（5ファイル）:
    od3    3連単の確定オッズ          od2  2連単・2連複の確定オッズ
    od1    3連複・拡連複・単勝・複勝の確定オッズ（生のまま保存）
    oriten オリジナル展示（一周・まわり足・直線）   tkz  直前情報（展示タイム・体重・チルト・スタート展示）
保存: data_full/bc/YYYY/bc_YYYYMMDD.jsonl.gz（1行=1レース）。値は 生テキスト / null（無い）/ "!ERR"（通信失敗）
- 新しい日から順に取る。できた日は飛ばし、"!ERR" が残った日はそこだけ取り直す（止まっても続きから）。
- 全体で1秒あたり --rate 回まで（既定4回。前の download_bc.py の12並列・待ちなしより控えめ）。
- BOATCAST は 2024年4月ごろより前のデータが無い。各日の最初の3レースが全部無ければ、その日は「無し」として記録して先へ。
"""
import argparse
import gzip
import json
from concurrent.futures import ThreadPoolExecutor

from fetchutil import ERR, RateLimiter, get_text
from paths import BC
from racelist import days, races_of

URLS = {
    "od3": "https://race.boatcast.jp/txt/{jo}/bc_kakutei_od3_{ymd}_{jo}_{rno:02d}.txt",
    "od2": "https://race.boatcast.jp/txt/{jo}/bc_kakutei_od2_{ymd}_{jo}_{rno:02d}.txt",
    "od1": "https://race.boatcast.jp/txt/{jo}/bc_kakutei_od1_{ymd}_{jo}_{rno:02d}.txt",
    "oriten": "https://race.boatcast.jp/txt/{jo}/bc_oriten_{ymd}_{jo}_{rno:02d}.txt",
    "tkz": "https://race.boatcast.jp/hp_txt/{jo}/bc_j_tkz_{ymd}_{jo}_{rno:02d}.txt",
}


def path_of(day):
    return BC / f"{day:%Y}" / f"bc_{day:%Y%m%d}.jsonl.gz"


def read_day(p):
    with gzip.open(p, "rt", encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def write_day(p, rows):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.rename(p)


def fetch_field(lim, ymd, jo, rno, k):
    return get_text(URLS[k].format(jo=jo, ymd=ymd, rno=rno), lim, ok_prefix="data=")


def do_day(day, lim, workers):
    p = path_of(day)
    races = races_of(day)
    if not races:
        return None
    ymd = f"{day:%Y%m%d}"
    if p.exists():
        rows = read_day(p)
        if rows and rows[0].get("no_bc"):
            return "skip"
        todo = [(i, k) for i, r in enumerate(rows) for k in URLS if r.get(k) == ERR]
        if not todo:
            return "skip"
    else:
        # BOATCAST にこの日があるか、最初の3レースの3連単オッズで確かめる
        probe = [fetch_field(lim, ymd, jo, rno, "od3") for jo, rno in races[:3]]
        if all(x is None for x in probe):
            write_day(p, [{"date": day.isoformat(), "no_bc": True}])
            return "no_bc"
        rows = [{"date": day.isoformat(), "venue_code": jo, "race_no": rno, **{k: ERR for k in URLS}}
                for jo, rno in races]
        for i, x in enumerate(probe):
            rows[i]["od3"] = x
        todo = [(i, k) for i, r in enumerate(rows) for k in URLS if r[k] == ERR]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        got = list(ex.map(lambda t: fetch_field(lim, ymd, rows[t[0]]["venue_code"], rows[t[0]]["race_no"], t[1]), todo))
    for (i, k), v in zip(todo, got):
        rows[i][k] = v
    write_day(p, rows)
    n_err = sum(1 for r in rows for k in URLS if r.get(k) == ERR)
    have = {k: sum(1 for r in rows if r.get(k) not in (None, ERR)) for k in URLS}
    return f"{len(rows)}R " + " ".join(f"{k}{v}" for k, v in have.items()) + (f" 失敗{n_err}" if n_err else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("start")
    ap.add_argument("end")
    ap.add_argument("--rate", type=float, default=4.0)
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    lim = RateLimiter(a.rate)
    for day in days(a.start, a.end):
        r = do_day(day, lim, a.workers)
        if r and r != "skip":
            print(day, r, flush=True)


if __name__ == "__main__":
    main()
