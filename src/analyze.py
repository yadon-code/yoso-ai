"""AIの予想が「オッズにない情報」を持っているかを調べる分析。

1. 市場（オッズ）の確率に、AIの確率をどれくらい混ぜると一番正確になるか
   → 混ぜたほうが正確なら、AIはオッズにない情報を持っている
2. 期待値（混ぜた確率×オッズ）の高さごとに、実際の回収率がどうだったか
   → 期待値が高いグループほど回収率が高ければ、AIの「割安」の判断は当たっている
   回収率は運でぶれるので、ぶれ幅（95%の範囲）も出す
"""
import numpy as np
import pandas as pd

from backtest import COMBOS, TAKEOUT, fit_lambdas, harville, load, logloss


def blend(p_model, odds, alpha):
    p_mkt = TAKEOUT / odds
    p_mkt = p_mkt / p_mkt.sum(axis=1, keepdims=True)
    b = np.exp(alpha * np.log(p_model + 1e-12) + (1 - alpha) * np.log(p_mkt))
    return b / b.sum(axis=1, keepdims=True)


def main():
    pw, ow, res = load()
    l2, l3 = fit_lambdas()
    p_model = harville(pw.to_numpy(), l2, l3)
    odds = ow.to_numpy()

    print("\n[1] 市場の確率にAIを混ぜたときの正確さ（対数損失、小さいほど良い）")
    for a in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]:
        print(f"  AIの割合 {a:.1f}: {logloss(blend(p_model, odds, a), res):.4f}")

    a = 0.3
    b = blend(p_model, odds, a)
    ev = b * odds
    hit_col = np.array([COMBOS.index(t) if isinstance(t, str) else -1 for t in res["trifecta"]])
    payout = res["trifecta_payout"].fillna(0).to_numpy()
    rows = []
    for i in range(len(odds)):
        if hit_col[i] < 0:
            continue
        for j in range(120):
            rows.append((ev[i, j], odds[i, j], payout[i] if j == hit_col[i] else 0))
    df = pd.DataFrame(rows, columns=["ev", "odds", "ret"])
    df["ev_bin"] = pd.cut(df["ev"], [0, 0.6, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4, 10])
    df["odds_bin"] = pd.cut(df["odds"], [0, 10, 30, 100, 1e6])
    rng = np.random.default_rng(0)

    def summary(g):
        r = g["ret"].to_numpy() / 100
        boots = [rng.choice(r, len(r)).mean() for _ in range(200)] if len(r) > 0 else [np.nan]
        return pd.Series({"点数": len(r), "回収率": r.mean(),
                          "下限": np.percentile(boots, 2.5), "上限": np.percentile(boots, 97.5)})

    print(f"\n[2] 期待値の高さ別の回収率（AIの割合 {a}、全期間）")
    print(df.groupby("ev_bin", observed=True).apply(summary).round(3).to_string())
    print("\n[3] 期待値1.1以上の買い目を、オッズの高さ別に")
    sub = df[df["ev"] >= 1.1]
    print(sub.groupby("odds_bin", observed=True).apply(summary).round(3).to_string())


if __name__ == "__main__":
    main()
