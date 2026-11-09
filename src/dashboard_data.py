"""進捗ボード用のCSVを作る（クラウド側で実行）。"""
import datetime as dt, sys
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parent.parent
OUT = Path(sys.argv[1])
races = pd.read_csv(ROOT/"data/csv/races.csv", usecols=["date"])
days = sorted(races.date.unique())
def cov(folder, pat, start):
    need = [d for d in days if d >= start]
    have = {p.stem.split("_")[-1] for p in (ROOT/"data"/folder).glob(pat)}
    got = [d for d in need if d.replace("-","") in have]
    return len(got), len(need), (min(got) if got else "")
rows = []
for name, folder, pat in [("3連単オッズ（締切時）","odds","odds_*.csv"),("オリジナル展示","oriten","oriten_*.csv"),("直前情報（展示・展示ST・チルト）","beforeinfo","bi_*.csv")]:
    g, n, m = cov(folder, pat, "2025-01-01")
    rows.append({"item": name, "done_days": g, "total_days": n, "pct": round(100*g/n,1), "oldest": m, "goal_from": "2025-01-01"})
pd.DataFrame(rows).to_csv(OUT/"collection.csv", index=False)
# エア投票
led = ROOT/"data/paper/bets.csv"
b = pd.read_csv(led, dtype={"venue_code": str}) if led.exists() else pd.DataFrame(columns=["date","stake"])
rep = ROOT/"data/paper/report.csv"
live = sorted((ROOT/"data/predictions").glob("live_*.csv"))
sys.path.insert(0, str(ROOT / "src"))
from settle_live import settle  # noqa: E402
if len(b):
    b = settle()
pr = []
for f in live:
    d = pd.read_csv(f)
    day = f.stem.split("_")[1]; day = f"{day[:4]}-{day[4:6]}-{day[6:]}"
    bb = b[b.date == day] if len(b) else b
    pr.append({"date": day, "races_predicted": len(d), "with_exhibit": int(d["exhibit"].sum()), "bets": len(bb), "stake_yen": int(bb["stake"].sum()) if len(bb) else 0,
               "settled": int(bb["settled"].sum()) if len(bb) else 0,
               "hits": int((bb["return"] > 0).sum()) if len(bb) else 0,
               "return_yen": int(bb["return"].sum()) if len(bb) else 0})
pd.DataFrame(pr, columns=["date","races_predicted","with_exhibit","bets","stake_yen","settled","hits","return_yen"]).to_csv(OUT/"paper.csv", index=False)
now = dt.datetime.now(dt.timezone(dt.timedelta(hours=9))).strftime("%Y-%m-%d %H:%M")
print(now)
