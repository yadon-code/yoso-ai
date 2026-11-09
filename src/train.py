"""各艇の「1着・2着・3着になる確率」を出すモデルを学習する（LightGBM）。

期間の分け方（未来のデータで過去を当てる「カンニング」を防ぐため、時間で区切る）:
    学習:   2024-01-01 〜 2026-04-30
    調整用: 2026-05-01 〜 2026-07-31  （買い方のルールを決めるのに使う）
    テスト: 2026-08-01 〜            （最後に一度だけ成績を見る）

出力:
    models/win_model.txt          学習したモデル
    data/pred/predictions.csv     調整用・テスト期間の各艇の1着確率
"""
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from features import FEATURES, KEY, build

ROOT = Path(__file__).resolve().parent.parent
TRAIN_END = "2026-04-30"
VALID_END = "2026-07-31"


def normalize(df: pd.DataFrame, raw: np.ndarray) -> np.ndarray:
    """1レース6艇の確率の合計が1になるようにそろえる。"""
    s = pd.Series(raw, index=df.index)
    return (s / s.groupby([df[k] for k in KEY]).transform("sum")).to_numpy()


def main():
    df = build()
    tr = df[df.date <= TRAIN_END]
    va = df[(df.date > TRAIN_END) & (df.date <= VALID_END)]
    model = lgb.LGBMClassifier(
        n_estimators=2000, learning_rate=0.03, num_leaves=63, min_child_samples=100,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1,
    )
    # 着順を「1着・2着・3着・4着以下」の4つに分けて、それぞれの確率を出す
    # （公式データにまれに着順「00」があるので、1未満は4着以下扱いにする）
    lab = lambda d: d["finish"].where(d["finish"] >= 1).fillna(6).clip(upper=4).astype(int) - 1
    model.fit(
        tr[FEATURES], lab(tr),
        eval_set=[(va[FEATURES], lab(va))],
        categorical_feature=["venue_code_i"],
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )
    print("木の数:", model.best_iteration_)

    out = df[df.date > TRAIN_END].copy()
    proba = model.predict_proba(out[FEATURES])
    out["p_win"] = normalize(out, proba[:, 0])
    out["p_2nd"] = normalize(out, proba[:, 1])
    out["p_3rd"] = normalize(out, proba[:, 2])
    out["period"] = np.where(out.date <= VALID_END, "valid", "test")

    # 1着的中率（各レースで確率が一番高い艇が勝ったか）
    for name, g in out.groupby("period"):
        top = g.loc[g.groupby(KEY)["p_win"].idxmax()]
        print(f"{name}: 本命の1着的中率 {top['win'].mean():.1%}  ({len(top)}レース)")
        one = g[g.boat == 1]
        print(f"   参考: いつも1号艇を本命にした場合 {one['win'].mean():.1%}")
        w = g[g.win == 1]
        print(f"   1着確率の対数損失 {-np.log(w['p_win']).mean():.4f}（小さいほど良い）")

    (ROOT / "models").mkdir(exist_ok=True)
    model.booster_.save_model(str(ROOT / "models" / "win_model.txt"))
    (ROOT / "data" / "pred").mkdir(parents=True, exist_ok=True)
    out[KEY + ["boat", "finish", "p_win", "p_2nd", "p_3rd", "period"]].to_csv(
        ROOT / "data" / "pred" / "predictions.csv", index=False)

    imp = pd.Series(model.booster_.feature_importance("gain"), index=FEATURES)
    imp = (imp / imp.sum()).sort_values(ascending=False)
    imp.round(4).to_csv(ROOT / "data" / "pred" / "feature_importance.csv", header=["importance"])
    print("効いている特徴量 TOP15:\n", imp.head(15).round(3))


if __name__ == "__main__" and len(__import__("sys").argv) == 1:
    main()


def oof():
    """時期をずらしながら学習し直して、2025年以降の各レースを「それより前のデータだけで学習したAI」で予想する。
    第2段階のモデルの学習に使う（未来のデータが混ざらないようにするため）。
    出力: data/pred/oof_predictions.csv"""
    df = build()
    lab = lambda d: d["finish"].where(d["finish"] >= 1).fillna(6).clip(upper=4).astype(int) - 1
    starts = pd.date_range("2025-01-01", "2027-01-01", freq="QS").strftime("%Y-%m-%d").tolist()
    outs = []
    for s, e in zip(starts[:-1], starts[1:]):
        es_start = (pd.Timestamp(s) - pd.DateOffset(months=2)).strftime("%Y-%m-%d")
        tr = df[df.date < es_start]
        es = df[(df.date >= es_start) & (df.date < s)]
        te = df[(df.date >= s) & (df.date < e)]
        part = ROOT / "data" / "pred" / "oof" / f"oof_{s}.csv"
        if te.empty:
            continue
        if part.exists():  # 途中で止まっても続きから
            outs.append(pd.read_csv(part, dtype={"venue_code": str}))
            continue
        m = lgb.LGBMClassifier(n_estimators=2000, learning_rate=0.03, num_leaves=63, min_child_samples=100,
                               subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1)
        m.fit(tr[FEATURES], lab(tr), eval_set=[(es[FEATURES], lab(es))],
              categorical_feature=["venue_code_i"], callbacks=[lgb.early_stopping(100, verbose=False)])
        pr = m.predict_proba(te[FEATURES])
        o = te[KEY + ["boat", "finish"]].copy()
        o["p_win"] = normalize(te, pr[:, 0])
        o["p_2nd"] = normalize(te, pr[:, 1])
        o["p_3rd"] = normalize(te, pr[:, 2])
        w = o[o.finish == 1]
        print(f"{s}〜{e}: {len(te)//6}レース  1着の対数損失 {-np.log(w['p_win']).mean():.4f}", flush=True)
        part.parent.mkdir(parents=True, exist_ok=True)
        o.to_csv(part, index=False)
        outs.append(o)
    pd.concat(outs).to_csv(ROOT / "data" / "pred" / "oof_predictions.csv", index=False)


if __name__ == "__main__" and len(__import__("sys").argv) > 1 and __import__("sys").argv[1] == "oof":
    oof()
