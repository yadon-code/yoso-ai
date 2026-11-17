"""各艇の「1着・2着・3着になる確率」を出すモデルを学習する（LightGBM）。

期間の分け方（未来のデータで過去を当てる「カンニング」を防ぐため、時間で区切る）:

    引数なし（これまでと同じ分け方）:
        学習:   2024-01-01 〜 2026-04-30
        調整用: 2026-05-01 〜 2026-07-31  （買い方のルールを決めるのに使う）
        テスト: 2026-08-01 〜            （最後に一度だけ成績を見る）

    `--holdout-days 30`（週1回の自動学習 `src/auto_train.py` が使う形）:
        学習:   最初 〜 (最終日 - 30日 - 30日)
        調整用: (最終日 - 30日 - 30日) 〜 (最終日 - 30日)   ← 木の数を決めるためだけに使う
        評価:   (最終日 - 30日) 〜 最終日                  ← 学習にも木の数を決めるのにも使わない
        こうすると、データが増えるたびに「直近を残して学習し直す」ができる。

出力:
    models/win_model.txt          学習したモデル
    data/pred/predictions.csv     調整用・テスト（--holdout-days のときは評価期間）の各艇の1着確率
"""
import argparse
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from features import FEATURES, KEY, build

ROOT = Path(__file__).resolve().parent.parent
TRAIN_END = "2026-04-30"   # 引数なしで動かしたときの学習の終わり（これまでと同じ）
VALID_END = "2026-07-31"   # 引数なしで動かしたときの調整用の終わり

# モデルの設定（学習のやり方を変えないため、`--holdout-days` でも同じものを使う）
PARAMS = dict(
    n_estimators=2000, learning_rate=0.03, num_leaves=63, min_child_samples=100,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1,
)


def normalize(df: pd.DataFrame, raw: np.ndarray) -> np.ndarray:
    """1レース6艇の確率の合計が1になるようにそろえる。"""
    s = pd.Series(raw, index=df.index)
    return (s / s.groupby([df[k] for k in KEY]).transform("sum")).to_numpy()


def label(d: pd.DataFrame) -> pd.Series:
    """着順を「1着・2着・3着・4着以下」の4つに分ける
    （公式データにまれに着順「00」があるので、1未満は4着以下扱いにする）。"""
    return d["finish"].where(d["finish"] >= 1).fillna(6).clip(upper=4).astype(int) - 1


def split(df: pd.DataFrame, holdout_days: int | None = None):
    """(学習, 木の数を決める用, 評価用) の3つに、日付で分ける。
    holdout_days を渡すと、直近その日数を評価用に残し、その前の同じ日数を木の数を決める用にする。"""
    if holdout_days is None:  # これまでと同じ分け方
        tr = df[df.date <= TRAIN_END]
        va = df[(df.date > TRAIN_END) & (df.date <= VALID_END)]
        te = df[df.date > VALID_END]
        return tr, va, te
    n = int(holdout_days)
    last = pd.Timestamp(df.date.max())
    cut = (last - pd.Timedelta(days=n - 1)).strftime("%Y-%m-%d")          # 評価用はここから最終日まで
    es_start = (pd.Timestamp(cut) - pd.Timedelta(days=n)).strftime("%Y-%m-%d")
    tr = df[df.date < es_start]
    va = df[(df.date >= es_start) & (df.date < cut)]
    te = df[df.date >= cut]
    return tr, va, te


def fit(tr: pd.DataFrame, es: pd.DataFrame | None):
    """学習する。es を渡すと、そこでの当てはまりが悪くなり始めたところで止める。"""
    model = lgb.LGBMClassifier(**PARAMS)
    if es is None or len(es) == 0:
        model.fit(tr[FEATURES], label(tr), categorical_feature=["venue_code_i"])
    else:
        model.fit(
            tr[FEATURES], label(tr),
            eval_set=[(es[FEATURES], label(es))],
            categorical_feature=["venue_code_i"],
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )
    return model


def add_probs(out: pd.DataFrame, proba: np.ndarray) -> pd.DataFrame:
    """4つの確率 (n,4) から、p_win・p_2nd・p_3rd の列を作る（レースごとに合計1）。"""
    out = out.copy()
    out["p_win"] = normalize(out, proba[:, 0])
    out["p_2nd"] = normalize(out, proba[:, 1])
    out["p_3rd"] = normalize(out, proba[:, 2])
    return out


def win_logloss(out: pd.DataFrame) -> float:
    """1着になった艇の1着確率の対数損失（小さいほど良い）。自動学習の判定に使う。"""
    w = out[out.win == 1]
    return float(-np.log(w["p_win"].clip(lower=1e-12)).mean())


def win_auc(out: pd.DataFrame) -> float:
    """1着を当てる力（AUC、大きいほど良い）。判定の参考として記録する。"""
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(out["win"], out["p_win"]))


def main(holdout_days: int | None = None):
    df = build()
    tr, va, te = split(df, holdout_days)
    model = fit(tr, va)

    print("木の数:", getattr(model, "best_iteration_", None))

    out = pd.concat([va, te]) if holdout_days is None else te
    out = add_probs(out, model.predict_proba(out[FEATURES]))
    if holdout_days is None:
        out["period"] = np.where(out.date <= VALID_END, "valid", "test")
    else:
        out["period"] = "holdout"

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


def oof():
    """時期をずらしながら学習し直して、2025年以降の各レースを「それより前のデータだけで学習したAI」で予想する。
    第2段階のモデルの学習に使う（未来のデータが混ざらないようにするため）。
    出力: data/pred/oof_predictions.csv"""
    df = build()
    lab = label
    import os  # OOF_START で開始を変えられる（例: 2020-01-01。グレードレースの過去分の検証用）
    starts = pd.date_range(os.environ.get("OOF_START", "2025-01-01"), "2027-01-01", freq="QS").strftime("%Y-%m-%d").tolist()
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
        m = lgb.LGBMClassifier(**PARAMS)
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
    pd.concat(outs).to_csv(ROOT / "data" / "pred" / os.environ.get("OOF_FILE", "oof_predictions.csv"), index=False)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "oof":
        oof()
    else:
        ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
        ap.add_argument("--holdout-days", type=int, default=None,
                        help="直近この日数を評価用に残して学習する（省略すると今までと同じ期間で学習する）")
        a = ap.parse_args()
        main(a.holdout_days)
