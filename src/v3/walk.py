"""v3 の検証の仕組み: 時期をずらしながら「その前まで」で学習し、次の期間を予想する（ウォークフォワード）。

決まり:
- 学習: TRAIN_START 〜 (テスト期間の始め − ES_MONTHS か月)。木の数を決める用（ES）: その後の ES_MONTHS か月。テスト: その次の期間。
- **金庫の期間（HOLDOUT_START 以降）は、--open-holdout を付けたときしか予想しない。** 開けたら data_full/work/holdout_opened.log に記録する。
  設定はすべて金庫を開ける前に決める（開けたあとに設定を変えたら、その結果は「金庫」とは呼ばない）。
- 予想は 1レース120通りの確率で保存する（data_full/work/<name>/pred_*.parquet）。成績の計算は evaluate.py。

    python src/v3/walk.py --name hit_nomkt                 # オッズなし
    python src/v3/walk.py --name hit_mkt --market          # 確定オッズあり
    python src/v3/walk.py --name hit_nomkt --step 3        # 3か月ごと（速い・試し用）
"""
import argparse
import datetime as dt
import json

import numpy as np
import pandas as pd

import paths  # noqa: F401
from feats import FEATURES, PRE_EXHIBIT_DROP
from model import COMBOS, TrifectaModel, market_probs
from paths import KEY, TABLES, WORK

TRAIN_START = "2022-07-01"   # 2022年前半は「過去の成績」がたまる準備期間
DEV_START = "2025-01-01"     # ここから予想して成績を測る
HOLDOUT_START = "2026-09-01"  # 金庫（最後の約3か月）。決めるまで開けない
ES_MONTHS = 1


def load(market=False):
    df = pd.read_parquet(TABLES / "feats.parquet")
    df = df[df.date >= TRAIN_START]
    ok = df.groupby(KEY)["fin"].transform(lambda s: (s == 1).sum() == 1)
    df = df[ok].copy()
    if market:
        o = pd.read_parquet(TABLES / "odds3t.parquet").drop_duplicates(KEY)
        cols = ["o_" + c for c in COMBOS]
        q = market_probs(o[cols].to_numpy(dtype=float))
        good = (np.isfinite(o[cols].to_numpy(dtype=float)).sum(1) >= 100)
        o = o[KEY][good].copy()
        o["__q120"] = list(q[good])
        df = df.merge(o, on=KEY, how="inner")
    return df.sort_values(KEY + ["boat"]).reset_index(drop=True)


def folds(step_months, end, open_holdout=False):
    out = []
    s = pd.Timestamp(DEV_START)
    stop = pd.Timestamp(end if open_holdout else HOLDOUT_START)
    while s < stop:
        e = min(s + pd.DateOffset(months=step_months), stop)
        es = s - pd.DateOffset(months=ES_MONTHS)
        out.append((es.strftime("%Y-%m-%d"), s.strftime("%Y-%m-%d"), e.strftime("%Y-%m-%d")))
        s = e
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--market", action="store_true", help="確定オッズから出した確率を材料に足す")
    ap.add_argument("--offset", action="store_true", help="市場の確率を出発点にして、ずれだけを学ぶ（--market も入る）")
    ap.add_argument("--pre-exhibit", action="store_true", help="展示の材料を使わない（朝の予想用）")
    ap.add_argument("--step", type=int, default=1, help="何か月ごとに学習し直すか")
    ap.add_argument("--open-holdout", action="store_true")
    ap.add_argument("--drop", default="", help="使わない特徴量（カンマ区切り）")
    a = ap.parse_args()
    feats = [f for f in FEATURES if f not in set(a.drop.split(","))]
    if a.pre_exhibit:
        feats = [f for f in feats if f not in PRE_EXHIBIT_DROP and not any(p in f for p in PRE_EXHIBIT_DROP)]
    a.market = a.market or a.offset
    df = load(a.market)
    out = WORK / a.name
    out.mkdir(parents=True, exist_ok=True)
    end = (pd.Timestamp(df.date.max()) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    if a.open_holdout:
        with (WORK / "holdout_opened.log").open("a", encoding="utf-8") as f:
            f.write(f"{dt.datetime.now():%F %T} {a.name} を金庫の期間（{HOLDOUT_START}〜）で予想\n")
    meta = {"features": feats, "market": a.market, "offset": a.offset, "step": a.step, "folds": []}
    for es, s, e in folds(a.step, end, a.open_holdout):
        p = out / f"pred_{s}.parquet"
        if p.exists():
            continue
        tr = df[df.date < es]
        ev = df[(df.date >= es) & (df.date < s)]
        te = df[(df.date >= s) & (df.date < e)]
        if te.empty:
            continue
        m = TrifectaModel(feats, use_market=a.market, offset=a.offset).fit(tr, ev)
        keys, q = m.predict(te)
        res = pd.DataFrame(q.astype(np.float32), columns=COMBOS)
        res = pd.concat([keys, res], axis=1)
        res.to_parquet(p, index=False)
        meta["folds"].append({"es": es, "start": s, "end": e, "n_train_races": len(tr) // 6, "n_test": len(te) // 6,
                              "iters": m.best_iter})
        print(f"{s}〜{e}: 学習 {len(tr) // 6} レース / テスト {len(te) // 6}  木の数 {m.best_iter}", flush=True)
        if s == folds(a.step, end, a.open_holdout)[-1][1]:
            imp = m.importance(1, 25)
            imp.to_csv(out / "importance_stage1.csv")
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
