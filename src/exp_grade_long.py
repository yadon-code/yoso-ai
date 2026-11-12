"""グレードレース（SG・G1）向けの工夫が効くかを、2019年からの長いデータで確かめる。

使い方:
    OOF_FILE=oof_predictions_long.csv python src/exp_grade_long.py

1年ずつ「それより前のデータだけで学習 → その年を予想」を繰り返し（ウォークフォワード）、SG・G1 のレースで比べる:
    A 今のAI（第2段階そのまま）
    B Aにグレード・レースの種類を特徴量として足したもの
    C Aの1号艇の確率を、前の年までのSG・G1のズレの分だけ補正したもの（SG と G1 で別々に1つずつ）
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.optimize import minimize_scalar

import market_model as mm
import analyze_grade as ag
from backtest import logloss

GMAP = {"SG": 4, "G1": 3, "G2": 2, "G3": 1}


def rtype(s):
    if "優勝" in s and "準" not in s:
        return 3
    if "準優" in s:
        return 2
    if "ドリーム" in s or "選抜" in s or "特別" in s:
        return 1
    return 0


def fit(df, tr, es, feats):
    m = lgb.LGBMClassifier(n_estimators=3000, learning_rate=0.02, num_leaves=15, min_child_samples=200,
                           subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5, verbose=-1)
    b = lambda d: np.log(d["mw"] / (1 - d["mw"]))
    m.fit(df.loc[tr, feats], df.loc[tr, "win"], init_score=b(df[tr]),
          eval_set=[(df.loc[es, feats], df.loc[es, "win"])], eval_init_score=[b(df[es])],
          categorical_feature=["venue_code_i"], callbacks=[lgb.early_stopping(200, verbose=False)])
    raw = m.predict(df[feats], raw_score=True) + b(df).to_numpy()
    p = 1 / (1 + np.exp(-raw))
    p = p / pd.Series(p).groupby([df[k].to_numpy() for k in mm.KEY]).transform("sum").to_numpy()
    return p.reshape(-1, 6)


def shift(pw, d):
    l = np.log(pw)
    l[:, 0] += d
    e = np.exp(l)
    return e / e.sum(1, keepdims=True)


def main():
    grades = ag.load_grades()
    ow, pm, res, df = mm.load()
    gr = ag.grade_of(ow.index, grades)
    races = pd.read_csv(mm.ROOT / "data" / "csv" / "races.csv", dtype={"venue_code": str}).set_index(mm.KEY)
    rn = races["race_name"].reindex(ow.index).fillna("")
    df["grade_i"] = np.repeat([GMAP.get(g, 0) for g in gr], 6)
    df["rtype"] = np.repeat([rtype(s) for s in rn], 6)
    df["g_boat"] = df["grade_i"] * 10 + df["boat"]
    dates = np.asarray(ow.index.get_level_values("date"))
    big = np.isin(gr, ["SG", "G1"])
    print(f"オッズのあるレース {len(ow)}（SG・G1 {big.sum()}、{dates[big].min()}〜）")
    print(pd.crosstab(pd.Series(dates[big]).str[:4], gr[big]).to_string())
    wc = res["trifecta"].str[0].astype(float).to_numpy() - 1
    ok = ~np.isnan(wc)
    odds = ow.to_numpy()
    mw = df["mw"].to_numpy().reshape(-1, 6)
    row_dates = df["date"].to_numpy()
    base = list(mm.FEATURES)
    plus = base + ["grade_i", "rtype", "g_boat"]
    allq = {k: [] for k in "MABC"}
    sel_all = []
    for year in range(2021, 2027):
        y0, y1 = f"{year}-01-01", f"{year + 1}-01-01"
        es0 = f"{year - 1}-11-01"
        tr = row_dates < es0
        es = (row_dates >= es0) & (row_dates < y0)
        te_r = (dates >= y0) & (dates < y1)
        if not te_r.any() or tr.sum() < 6000:
            continue
        pa = fit(df, tr, es, base)
        pb = fit(df, tr, es, plus)
        # C: 前の年までのSG・G1で、Aの1号艇のズレ（学習に使っていない検証用期間 es と、その前年の結果）を推定
        pc = pa.copy()
        for g in ["SG", "G1"]:
            past = (gr == g) & (dates < y0) & (dates >= f"{year - 1}-01-01") & ok
            if past.sum() < 50:
                continue
            d = minimize_scalar(lambda d: -np.log(shift(pa[past], d)[np.arange(past.sum()), wc[past].astype(int)]).mean(),
                                bounds=(-1, 1), method="bounded").x
            s = (gr == g) & te_r
            pc[s] = shift(pa[s], d)
            print(f"  {year} {g}: 1号艇の補正 {d:+.3f}（前年{past.sum()}レースから）")
        s = te_r & big
        qs = {"M": pm[s]}
        for k, p in [("A", pa), ("B", pb), ("C", pc)]:
            qs[k] = mm.trifecta_probs(pm[s], p[s] / mw[s], 1.5, 0.5, 0.5)
        line = [f"{year}年 SG・G1 {s.sum()}R 3連単LL"] + [f"{k}{logloss(q, res[s]):.4f}" for k, q in qs.items()]
        for k in "ABC":
            n, r, h = mm.roi(qs[k], odds[s], res[s], 1.0, 300)
            line.append(f"{k}:{n}点{r:.0%}的中{h}")
        print(" ".join(line), flush=True)
        for k in qs:
            allq[k].append(qs[k])
        sel_all.append(np.where(s)[0])
    idx = np.concatenate(sel_all)
    r = res.iloc[idx]
    print("\n■ 全テスト年の合計（SG・G1）", len(idx), "レース")
    for k in "MABC":
        q = np.concatenate(allq[k])
        out = [f"{k} 3連単LL {logloss(q, r):.4f}"]
        if k != "M":
            for th, mo in [(1.0, 100), (1.0, 300), (1.1, 300), (1.2, 300)]:
                n, rr, h = mm.roi(q, odds[idx], r, th, mo)
                out.append(f"EV>{th}/{mo}倍: {n}点 {rr:.0%} 的中{h}")
        print(" | ".join(out), flush=True)


if __name__ == "__main__":
    main()
