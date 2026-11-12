"""ウェブアプリ用のデータ（JSON）を作る。

使い方:
    python src/build_site.py            # 今日の分と、日付一覧・累計を作り直す
    python src/build_site.py 2026-11-09

出力:
    data/site/days.json         日付ごとのエア投票の成績と累計
    data/site/YYYYMMDD.json     その日の全レース（締切・予想・買い目・結果）
締切を過ぎたレースは、公式サイトで結果（3連単）を調べて data/paper/results_live.csv に貯める。
"""
import datetime as dt
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from parse import VENUES  # noqa: E402
import sitecrypt  # noqa: E402
from settle_live import load_cache, result, save_cache, settle  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PRED = ROOT / "data" / "predictions"
OUT = ROOT / "data" / "site"
JST = dt.timezone(dt.timedelta(hours=9))


def write_site(name, obj, pw):
    """ウェブアプリ用のファイルを書く。合言葉があれば暗号化した .enc だけを残す。"""
    text = json.dumps(obj, ensure_ascii=False)
    plain, enc = OUT / name, OUT / (name + ".enc")
    if pw:
        enc.write_text(sitecrypt.encrypt(text, pw), encoding="ascii")
        plain.unlink(missing_ok=True)
    else:
        plain.write_text(text, encoding="utf-8")
        enc.unlink(missing_ok=True)


def parse_picks(s):
    """'1-2-3(12.3倍,期待値1.05) ...' → [{combo, odds, ev}]"""
    out = []
    for m in re.finditer(r"(\d-\d-\d)\(([\d.]+)倍,期待値([\d.]+)\)", s if isinstance(s, str) else ""):
        out.append({"combo": m.group(1), "odds": float(m.group(2)), "ev": float(m.group(3))})
    return out


def build_day(day: dt.date, known, now):
    ymd = f"{day:%Y%m%d}"
    iso = day.isoformat()
    races = {}
    sp = PRED / f"schedule_{ymd}.csv"
    if sp.exists():
        for r in pd.read_csv(sp, dtype={"venue_code": str}).itertuples():
            races[(r.venue_code, int(r.race_no))] = {
                "deadline": r.deadline,
                "grade": getattr(r, "grade", "") if isinstance(getattr(r, "grade", ""), str) else "",
                "race_name": getattr(r, "race_name", "") if isinstance(getattr(r, "race_name", ""), str) else ""}
    lp = PRED / f"live_{ymd}.csv"
    preds = pd.read_csv(lp, dtype={"venue_code": str}) if lp.exists() else pd.DataFrame()
    for r in preds.itertuples():
        d = races.setdefault((r.venue_code, int(r.race_no)), {})
        d.update({"predicted_at": r.predicted_at[-5:], "exhibit": bool(r.exhibit),
                  "p_win": [float(x) for x in str(r.p_win).split()],
                  "top5": [{"combo": t.split(":")[0], "prob": float(t.split(":")[1])} for t in str(r.top5).split()],
                  "picks": parse_picks(r.picks)})
    if not races:
        return None
    # 締切を8分以上過ぎたレースの結果を調べる（4つ同時に）
    todo = []
    for (v, rn), d in races.items():
        if (iso, v, rn) in known:
            continue
        dl = d.get("deadline") if isinstance(d.get("deadline"), str) else d.get("predicted_at")
        if isinstance(dl, str):
            t = dt.datetime.combine(day, dt.time(*map(int, dl.split(":"))), JST)
            if now < t + dt.timedelta(minutes=8):
                continue
        elif day >= now.date():
            continue
        todo.append((iso, v, rn))
    with ThreadPoolExecutor(max_workers=4) as ex:
        for k, r in zip(todo, ex.map(lambda k: result(*k), todo)):
            if r:
                known[k] = r
    out = []
    for (v, rn), d in sorted(races.items(), key=lambda x: (x[0][0], x[0][1])):
        res = known.get((iso, v, rn))
        bets = [p for p in d.get("picks", [])] if d.get("exhibit") else []
        back = (res[2] or "").split() if res else []
        for p in bets:
            p["refund"] = any(x in back for x in p["combo"].split("-"))  # 返還（買ったお金が戻る）
            p["hit"] = bool(res and res[0] == p["combo"]) and not p["refund"]
        out.append({"venue_code": v, "venue": VENUES.get(v, v), "race_no": rn, **d,
                    "result": res[0] if res else None, "payout": res[1] if res else None})
    return {"date": iso, "updated": now.strftime("%Y-%m-%d %H:%M"), "races": out}


def main():
    now = dt.datetime.now(JST)
    days = [dt.date.fromisoformat(sys.argv[1])] if len(sys.argv) > 1 else [now.date()]
    known = load_cache()
    OUT.mkdir(parents=True, exist_ok=True)
    pw = sitecrypt.password()
    if pw:  # 合言葉を設定する前に作った、カギなしの過去の日のファイルにもカギをかける
        for f in OUT.glob("2*.json"):
            write_site(f.name, json.loads(f.read_text(encoding="utf-8")), pw)
    for day in days:
        data = build_day(day, known, now)
        if data:
            write_site(f"{day:%Y%m%d}.json", data, pw)
            print(f"{day}: {len(data['races'])} レース")
    save_cache(known)
    # 日付一覧とエア投票の成績
    b = settle()
    rows = []
    stems = sorted({f.name[:8] for f in list(OUT.glob("2*.json")) + list(OUT.glob("2*.json.enc"))})
    for stem in stems:
        iso = f"{stem[:4]}-{stem[4:6]}-{stem[6:]}"
        d = b[b.date == iso] if len(b) else b
        s = d[d.settled] if len(d) else d
        rows.append({"date": iso, "bets": int(len(d)), "settled": int(len(s)),
                     "stake": int(s.stake.sum()) if len(s) else 0,
                     "return": int(s["return"].sum()) if len(s) else 0,
                     "hits": int(s["hit"].sum()) if len(s) else 0})
    tot = {k: sum(r[k] for r in rows) for k in ["bets", "settled", "stake", "return", "hits"]}
    # 🏆 SG・G1 のレースだけの累計
    gtot = {"bets": 0, "settled": 0, "stake": 0, "return": 0, "hits": 0}
    try:
        from grade import BIG
        from settle_live import with_grades
        if len(b):
            bg = with_grades(b)
            bg = bg[bg["grade"].isin(BIG)]
            s = bg[bg.settled]
            gtot = {"bets": int(len(bg)), "settled": int(len(s)), "stake": int(s.stake.sum()),
                    "return": int(s["return"].sum()), "hits": int(s["hit"].sum())}
    except Exception as e:
        print("グレード別の集計に失敗:", e)
    write_site("days.json", {"updated": now.strftime("%Y-%m-%d %H:%M"), "total": tot, "grade_total": gtot,
                             "days": rows[::-1]}, pw)


if __name__ == "__main__":
    main()
