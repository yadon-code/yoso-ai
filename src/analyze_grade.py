"""グレード（SG / G1 / G2 / G3 / 一般）ごとに、レース数・市場（オッズ）の正確さ・今のAIの成績を比べる。

使い方:
    TRAIN_END=2026-03-31 VALID_END=2026-06-30 python src/analyze_grade.py

必要: data/csv/grades.csv（src/download_grade.py で作る）
"""
import numpy as np
import pandas as pd

import market_model as mm
from backtest import COMBOS, logloss

ROOT = mm.ROOT
KEY = mm.KEY


def load_grades():
    g = pd.read_csv(ROOT / "data" / "csv" / "grades.csv", dtype={"jcd": str})
    g = g.rename(columns={"jcd": "venue_code"}).drop_duplicates(["date", "venue_code"])
    return g.set_index(["date", "venue_code"])["grade"]


def grade_of(index, grades):
    k = pd.MultiIndex.from_arrays([index.get_level_values(0), index.get_level_values(1)])
    return grades.reindex(k).fillna("不明").to_numpy()


def main():
    grades = load_grades()

    # 1) レース数（公式の番組表ベース。2024年〜）
    races = pd.read_csv(ROOT / "data" / "csv" / "races.csv", dtype={"venue_code": str})
    races["grade"] = grade_of(races.set_index(["date", "venue_code"]).index, grades)
    races["year"] = races["date"].str[:4]
    print("■ レース数（グレード × 年）")
    print(pd.crosstab(races["grade"], races["year"], margins=True).to_string())
    big = races[races.grade.isin(["SG", "G1"])]
    print("\nSG・G1 の中のレースの種類（上位）")
    print(big["race_name"].value_counts().head(12).to_string())
    print("\n1号艇の1着率（グレード別）:")
    r1 = races.dropna(subset=["trifecta"]).assign(w1=lambda d: d.trifecta.str[0] == "1")
    print(r1.groupby("grade")["w1"].mean().round(3).to_string())
    print("\n3連単の平均払戻・中央値（グレード別）:")
    print(races.groupby("grade")["trifecta_payout"].agg(["mean", "median"]).round(0).to_string())

    # 2) オッズのある期間で、市場の正確さとAIの成績
    ow, pm, res, df = mm.load()
    dates = ow.index.get_level_values("date")
    gr = grade_of(ow.index, grades)
    big = np.isin(gr, ["SG", "G1"])
    print(f"\n■ オッズのあるレース: {len(ow)}（うち SG・G1 {big.sum()}）")

    model, p = mm.fit_win(df)
    pw = p.reshape(-1, 6)
    mw = df["mw"].to_numpy().reshape(-1, 6)
    win_col = res["trifecta"].str[0].astype(float).to_numpy() - 1
    ok = ~np.isnan(win_col)
    va = np.asarray((dates > mm.TRAIN_END) & (dates <= mm.VALID_END))
    te = np.asarray(dates > mm.VALID_END)
    q = mm.trifecta_probs(pm, pw / mw, 1.5, 0.5, 0.5)
    odds = ow.to_numpy()

    # 市場の偏り: 1号艇の 実際の1着率 − オッズの1着確率
    print("\n1号艇: 実際の1着率 − オッズから見た確率（＋なら市場が過小評価）")
    for name in ["SG", "G1", "G2", "G3", "一般"]:
        s = (gr == name) & ok
        if s.sum():
            print(f"  {name}: {(win_col[s] == 0).mean() - mw[s, 0].mean():+.3f}  ({s.sum()}レース)")

    # 全部の買い目をオッズ帯ごとに買った場合の回収率（市場の歪みの目安）
    hit = np.zeros_like(odds, dtype=bool)
    hc = np.array([COMBOS.index(t) if isinstance(t, str) else -1 for t in res["trifecta"]])
    hit[np.arange(len(hc))[hc >= 0], hc[hc >= 0]] = True
    print("\nオッズ帯ごとに全部買ったときの回収率（市場の歪みの目安）")
    bins = [0, 10, 30, 100, 300, 1e9]
    rows = []
    for name, sel in [("SG・G1", big), ("それ以外", ~big)]:
        o, h = odds[sel], hit[sel]
        r = {"区分": name}
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (o >= lo) & (o < hi)
            r[f"{lo:g}〜{hi:g}倍" if hi < 1e9 else f"{lo:g}倍〜"] = round((o[m] * h[m]).sum() / m.sum(), 3)
        rows.append(r)
    print(pd.DataFrame(rows).to_string(index=False))

    print("\n今の第2段階AI（オッズ＋直前情報で補正）の成績")
    for pname, per in [("調整用", va), ("テスト", te), ("調整+テスト", va | te)]:
        for name, sel in [("SG・G1", big), ("G2・G3", np.isin(gr, ["G2", "G3"])), ("一般", gr == "一般")]:
            s = per & sel
            if s.sum() < 30:
                continue
            s2 = s & ok
            ll_m = -np.log(mw[s2, win_col[s2].astype(int)]).mean()
            ll_p = -np.log(pw[s2, win_col[s2].astype(int)]).mean()
            n, r, h = mm.roi(q[s], odds[s], res[s], 1.0, 100)
            print(f"  [{pname}] {name}: {s.sum()}レース  1着の対数損失 市場{ll_m:.4f}→AI{ll_p:.4f}"
                  f"  3連単 市場{logloss(pm[s], res[s]):.4f}→AI{logloss(q[s], res[s]):.4f}"
                  f"  期待値>1買い {n}点 回収率{r:.0%} 的中{h}")


if __name__ == "__main__":
    main()
