"""エア投票の買い目を、公式サイトの結果ページで当日のうちに照合する。

使い方:
    python src/settle_live.py              # 今日の分を照合して、まとめを表示
    python src/settle_live.py 2026-11-09 --notify   # まとめを通知もする
"""
import argparse
import datetime as dt
import re
import sys
import urllib.request
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from notify import send  # noqa: E402
from parse import VENUES  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "paper" / "results_live.csv"
JST = dt.timezone(dt.timedelta(hours=9))


def result(date, jo, rno):
    """3連単の組番と払戻（100円あたり）。まだ確定していなければ None。"""
    url = f"https://www.boatrace.jp/owpc/pc/race/raceresult?rno={int(rno)}&jcd={jo}&hd={date.replace('-', '')}"
    try:
        h = urllib.request.urlopen(url, timeout=30).read().decode("utf-8", "replace")
    except Exception:
        return None
    i = h.find("3連単")
    if i < 0:
        return None
    t = " ".join(re.sub(r"<[^>]+>", " ", h[i:i + 1500]).replace("&yen;", "").split())
    m = re.search(r"(\d) - (\d) - (\d) ([\d,]+)", t)
    return (f"{m.group(1)}-{m.group(2)}-{m.group(3)}", int(m.group(4).replace(",", ""))) if m else None


def settle():
    """bets.csv の全買い目に settled / return 列をつけて返す（結果はキャッシュに保存）。"""
    led = ROOT / "data" / "paper" / "bets.csv"
    if not led.exists():
        return pd.DataFrame(columns=["date", "venue_code", "race_no", "combo", "stake", "settled", "return"])
    b = pd.read_csv(led, dtype={"venue_code": str})
    cache = pd.read_csv(CACHE, dtype={"venue_code": str}) if CACHE.exists() else \
        pd.DataFrame(columns=["date", "venue_code", "race_no", "trifecta", "payout"])
    known = {(r.date, r.venue_code, int(r.race_no)): (r.trifecta, int(r.payout)) for r in cache.itertuples()}
    for k in set(zip(b.date, b.venue_code, b.race_no.astype(int))):
        if k not in known:
            r = result(*k)
            if r:
                known[k] = r
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"date": k[0], "venue_code": k[1], "race_no": k[2], "trifecta": v[0], "payout": v[1]}
                  for k, v in known.items()]).to_csv(CACHE, index=False)
    keys = list(zip(b.date, b.venue_code, b.race_no.astype(int)))
    b["settled"] = [k in known for k in keys]
    b["result"] = [known[k][0] if k in known else "" for k in keys]
    b["return"] = [known[k][1] * s / 100 if k in known and known[k][0] == c else 0
                   for k, c, s in zip(keys, b.combo, b.stake)]
    return b


def summary(day: str):
    b = settle()
    d = b[b.date == day]
    if d.empty:
        return f"📊 {day} のエア投票: 買い目なし"
    stake, ret = d.stake.sum(), d["return"].sum()
    lines = [f"📊 **{day} のエア投票まとめ**",
             f"買い目 {len(d)}点（{stake:,}円）→ 払戻 {ret:,.0f}円　回収率 {ret / stake:.0%}",
             f"的中 {int((d['return'] > 0).sum())}点"]
    for (v, r), g in d.groupby(["venue_code", "race_no"]):
        hit = g[g["return"] > 0]
        res = g["result"].iloc[0] or "未確定"
        mark = f"⭕ {hit.combo.iloc[0]} 的中 {hit['return'].sum():,.0f}円" if len(hit) else "❌"
        lines.append(f"・{VENUES.get(v, v)} {r}R 結果 {res}  {mark}")
    tot = b[b.settled]
    if len(tot) and tot.date.nunique() > 1:
        lines.append(f"累計: {len(tot)}点 回収率 {tot['return'].sum() / tot.stake.sum():.0%}")
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("date", nargs="?")
    ap.add_argument("--notify", action="store_true")
    a = ap.parse_args()
    day = a.date or dt.datetime.now(JST).date().isoformat()
    text = summary(day)
    print(text)
    if a.notify:
        send(text)
