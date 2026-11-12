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
    """(3連単の組番, 払戻（100円あたり）, 返還になった艇番を空白区切り)。まだ確定していなければ None。
    フライングなどで返還になった艇を含む舟券は、買ったお金がそのまま戻る。"""
    url = f"https://www.boatrace.jp/owpc/pc/race/raceresult?rno={int(rno)}&jcd={jo}&hd={date.replace('-', '')}"
    try:
        h = urllib.request.urlopen(url, timeout=30).read().decode("utf-8", "replace")
    except Exception:
        return None
    i = h.find("3連単")
    if i < 0:
        return None
    t = " ".join(re.sub(r"<[^>]+>", " ", h[i:i + 1500]).replace("&yen;", "").split())
    seg = re.search(r"<th>返還</th>(.*?)</table>", h, re.S)
    refund = " ".join(sorted(set(re.findall(r'numberSet1_number[^>]*>\s*([1-6])\s*<', seg.group(1))))) if seg else ""
    m = re.search(r"(\d) - (\d) - (\d) ([\d,]+)", t)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}", int(m.group(4).replace(",", "")), refund
    if "不成立" in t[:200]:  # 3連単が成立しなかったレースは全額返還
        return "不成立", 0, "1 2 3 4 5 6"
    return None


def load_cache():
    """結果のキャッシュ {(日付, 場, R): (組番, 払戻, 返還艇)}。返還艇が None のものは、返還を調べる前の古い記録。"""
    if not CACHE.exists():
        return {}
    c = pd.read_csv(CACHE, dtype={"venue_code": str, "refund": str}, keep_default_na=False)
    has = "refund" in c.columns
    return {(r.date, r.venue_code, int(r.race_no)): (r.trifecta, int(r.payout), r.refund if has and r.refund != "?" else None)
            for r in c.itertuples()}


def save_cache(known):
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"date": k[0], "venue_code": k[1], "race_no": k[2], "trifecta": v[0], "payout": v[1],
                   "refund": "?" if v[2] is None else v[2]}
                  for k, v in sorted(known.items())],
                 columns=["date", "venue_code", "race_no", "trifecta", "payout", "refund"]).to_csv(CACHE, index=False)


def settle():
    """bets.csv の全買い目に settled / return 列をつけて返す（結果はキャッシュに保存）。"""
    led = ROOT / "data" / "paper" / "bets.csv"
    if not led.exists():
        return pd.DataFrame(columns=["date", "venue_code", "race_no", "combo", "stake", "settled", "return", "refunded", "hit"])
    b = pd.read_csv(led, dtype={"venue_code": str})
    known = load_cache()
    for k in set(zip(b.date, b.venue_code, b.race_no.astype(int))):
        if k not in known or known[k][2] is None:  # まだ結果がない、または返還を調べていない
            r = result(*k)
            if r:
                known[k] = r
    save_cache(known)
    keys = list(zip(b.date, b.venue_code, b.race_no.astype(int)))
    b["settled"] = [k in known for k in keys]
    b["result"] = [known[k][0] if k in known else "" for k in keys]
    # 返還になった艇を含む買い目は、買った金額がそのまま戻る（的中には数えない）
    b["refunded"] = [k in known and any(x in (known[k][2] or "").split() for x in c.split("-"))
                     for k, c in zip(keys, b.combo)]
    b["return"] = [s if rf else (known[k][1] * s / 100 if k in known and known[k][0] == c else 0)
                   for k, c, s, rf in zip(keys, b.combo, b.stake, b.refunded)]
    b["hit"] = (b["return"] > 0) & ~b["refunded"]
    return b


def summary(day: str):
    b = settle()
    d = b[b.date == day]
    if d.empty:
        return f"📊 {day} のエア投票: 買い目なし"
    stake, ret = d.stake.sum(), d["return"].sum()
    lines = [f"📊 **{day} のエア投票まとめ**",
             f"買い目 {len(d)}点（{stake:,}円）→ 払戻 {ret:,.0f}円　回収率 {ret / stake:.0%}",
             f"的中 {int(d['hit'].sum())}点" + (f"　返還 {int(d['refunded'].sum())}点" if d["refunded"].any() else "")]
    for (v, r), g in d.groupby(["venue_code", "race_no"]):
        hit = g[g["hit"]]
        res = g["result"].iloc[0] or "未確定"
        mark = f"⭕ {hit.combo.iloc[0]} 的中 {hit['return'].sum():,.0f}円" if len(hit) else "❌"
        if g["refunded"].any():
            mark += f"（返還 {int(g['refunded'].sum())}点）"
        lines.append(f"・{VENUES.get(v, v)} {r}R 結果 {res}  {mark}")
    tot = b[b.settled]
    if len(tot) and tot.date.nunique() > 1:
        lines.append(f"累計: {len(tot)}点 回収率 {tot['return'].sum() / tot.stake.sum():.0%}")
    lines += grade_lines(b, day)
    return "\n".join(lines)


def with_grades(b):
    """買い目に、そのレースのグレード（SG・G1など。記録を始める前の日は空）をつける。"""
    from grade import schedule_grades
    g = schedule_grades().drop_duplicates(["date", "venue_code", "race_no"])
    g["race_no"] = g["race_no"].astype(int)
    b = b.drop(columns=["grade"], errors="ignore").copy()
    b["race_no"] = b["race_no"].astype(int)
    return b.merge(g[["date", "venue_code", "race_no", "grade"]], on=["date", "venue_code", "race_no"], how="left")


def grade_lines(b, day):
    """🏆 SG・G1 のレースだけの成績（その日と累計）。"""
    from grade import BIG
    try:
        bg = with_grades(b)
    except Exception:
        return []
    bg = bg[bg["grade"].isin(BIG) & bg.settled]
    if bg.empty:
        return []
    d = bg[bg.date == day]
    out = []
    if len(d):
        out.append(f"🏆 SG・G1: {len(d)}点 回収率 {d['return'].sum() / d.stake.sum():.0%}　的中 {int(d['hit'].sum())}点")
    out.append(f"🏆 SG・G1 累計: {len(bg)}点 回収率 {bg['return'].sum() / bg.stake.sum():.0%}　的中 {int(bg['hit'].sum())}点")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("date", nargs="?")
    ap.add_argument("--notify", action="store_true")
    a = ap.parse_args()
    day = a.date or dt.datetime.now(JST).date().isoformat()
    text = summary(day)
    print(text)
    if a.notify:
        send(text, room="📊 毎日のまとめ")
