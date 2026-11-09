"""要因の分析レポート用の集計（2024年1月〜のデータ）。結果は docs/要因分析.md に使う。"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import KEY

df = pd.read_pickle("/tmp/feat.pkl") if Path("/tmp/feat.pkl").exists() else __import__("features").build()
df = df[df.finish.notna() | True]
df["win"] = (df.finish == 1).astype(int)
df["top2"] = (df.finish <= 2).astype(int)
df["top3"] = (df.finish <= 3).astype(int)
pct = lambda s: (s * 100).round(1)
out = {}

# 1. 場ごとの1号艇1着率
v = df[df.boat == 1].groupby("venue")["win"].agg(["mean", "size"]).sort_values("mean", ascending=False)
out["venue_b1"] = v.assign(mean=pct(v["mean"]))

# 2. 枠ごとの1着率（全国）
out["boat_win"] = df.groupby("boat")[["win", "top2", "top3"]].mean().pipe(pct)

# 3. 風速・波と1号艇
b1 = df[df.boat == 1].copy()
b1["風速"] = pd.cut(b1.wind_speed, [-1, 2, 4, 6, 100], labels=["0-2m", "3-4m", "5-6m", "7m以上"])
b1["波"] = pd.cut(b1.wave_cm, [-1, 2, 5, 10, 1000], labels=["0-2cm", "3-5cm", "6-10cm", "11cm以上"])
out["wind"] = b1.groupby("風速", observed=True)["win"].agg(["mean", "size"]).assign(mean=lambda d: pct(d["mean"]))
out["wave"] = b1.groupby("波", observed=True)["win"].agg(["mean", "size"]).assign(mean=lambda d: pct(d["mean"]))

# 4. 1号艇の級別と、2〜6号艇の1着率
t = df.copy()
t["1号艇の級"] = t.groupby(KEY)["class"].transform(lambda s: s.iloc[0])
out["b1class"] = (t.pivot_table(index="1号艇の級", columns="boat", values="win", aggfunc="mean") * 100).round(1)

# 5. 1号艇と2号艇の勝率差と、2号艇の1着率（差し）・3号艇（まくり）
t["差"] = pd.cut(t["self_minus_b1_win_rate"], [-10, -2, -1, 0, 1, 2, 10],
                labels=["2以上低い", "1〜2低い", "0〜1低い", "0〜1高い", "1〜2高い", "2以上高い"])
out["vs_b1"] = (t[t.boat.isin([2, 3, 4])].pivot_table(index="差", columns="boat", values="win",
                                                     aggfunc="mean", observed=True) * 100).round(1)

# 6. 展示タイム順位と1着率
out["exh"] = df.groupby("exhibit_time_rank")[["win", "top3"]].mean().pipe(pct)

# 7. モーター（場×モーター直近30走の2連率）
df["モーター"] = pd.qcut(df["m_top2_rate"], 5, labels=["下位20%", "20-40%", "40-60%", "60-80%", "上位20%"])
out["motor"] = df.groupby("モーター", observed=True)[["win", "top3"]].mean().pipe(pct)

# 8. F持ち（直近60走でF）と1号艇
b1 = df[df.boat == 1]
out["f"] = b1.groupby(b1.r_f_count.clip(upper=2).fillna(0))[["win"]].agg(["mean", "size"])

# 9. 今節の調子
df["今節平均着"] = pd.cut(df.setsu_avg_finish, [0, 2, 3, 4, 5, 7], labels=["2以下", "2-3", "3-4", "4-5", "5超"])
out["setsu"] = df[df.setsu_n >= 2].groupby("今節平均着", observed=True)[["win", "top3"]].mean().pipe(pct)

# 10. 荒れ具合：場ごとの3連単払戻の中央値
r = pd.read_csv(Path(__file__).resolve().parent.parent / "data/csv/races.csv")
out["payout"] = r.groupby("venue")["trifecta_payout"].median().sort_values()

for k, v in out.items():
    print(f"\n### {k}\n{v.to_string()}")
