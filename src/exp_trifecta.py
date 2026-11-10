"""実験: 3連単120通りを直接学習するモデル（2着・3着も専用に見る）。本番には影響しない。

オッズの3連単確率を出発点（init_score）にして、1着・2着・3着に置いた艇それぞれの
直前情報・第1段階AIとのズレ・市場の2着/3着確率から「ズレ」を学ぶ。1レース120行。

使い方:  TRAIN_END=2026-03-31 VALID_END=2026-06-30 python src/exp_trifecta.py
"""
import lightgbm as lgb
import numpy as np

import experiments as ex
import market_model as mm
from backtest import IDX

BOAT_FEATS = ["log_mw", "log_ratio_ai", "exh_st_diff", "exh_st_rank", "exhibit_time_bi_diff", "exhibit_time_bi_rank",
              "lap_diff", "turn_diff", "straight_diff", "tilt_diff", "exh_course", "weight_day_diff"]


def build(pm, df):
    n = len(pm)
    F = np.stack([df[c].to_numpy(dtype=np.float32).reshape(n, 6) for c in BOAT_FEATS], axis=2)  # (n,6,k)
    # 市場が見ている「2着に来る確率」「3着に来る確率」（艇ごと）
    m2 = np.stack([pm[:, IDX[:, 1] == i].sum(axis=1) for i in range(6)], axis=1)
    m3 = np.stack([pm[:, IDX[:, 2] == i].sum(axis=1) for i in range(6)], axis=1)
    cols, names = [], []
    logpm = np.log(pm).astype(np.float32)
    cols.append(logpm.ravel()); names.append("log_pm")
    cols.append(np.argsort(np.argsort(-pm, axis=1), axis=1).astype(np.float32).ravel()); names.append("pm_rank")
    for pos, tag in enumerate(["a", "b", "c"]):
        ix = IDX[:, pos]
        cols.append(np.tile(ix + 1, n).astype(np.float32)); names.append(f"{tag}_boat")
        for k, c in enumerate(BOAT_FEATS):
            cols.append(F[:, ix, k].ravel()); names.append(f"{tag}_{c}")
    cols.append(np.log(m2[:, IDX[:, 1]]).astype(np.float32).ravel()); names.append("b_log_m2")
    cols.append(np.log(m3[:, IDX[:, 2]]).astype(np.float32).ravel()); names.append("c_log_m3")
    venue = df["venue_code_i"].to_numpy().reshape(n, 6)[:, 0]
    cols.append(np.repeat(venue, 120).astype(np.float32)); names.append("venue")
    X = np.column_stack(cols)
    init = (logpm - np.log1p(-pm).astype(np.float32)).ravel()
    return X, init, names


def main():
    ow, pm, res, df = mm.load()
    dates = np.asarray(ow.index.get_level_values("date"))
    tr, va, te = dates <= mm.TRAIN_END, (dates > mm.TRAIN_END) & (dates <= mm.VALID_END), dates > mm.VALID_END
    hit = ex.hit_index(res)
    ok = hit >= 0
    X, init, names = build(pm, df)
    y = np.zeros((len(pm), 120), dtype=np.int8)
    y[np.arange(len(pm))[ok], hit[ok]] = 1
    y = y.ravel()
    row = lambda sel: np.repeat(sel & ok, 120)
    dtr = lgb.Dataset(X[row(tr)], y[row(tr)], init_score=init[row(tr)], feature_name=names, categorical_feature=["venue"])
    dva = lgb.Dataset(X[row(va)], y[row(va)], init_score=init[row(va)], reference=dtr)
    params = dict(objective="binary", learning_rate=0.03, num_leaves=31, min_data_in_leaf=2000, feature_fraction=0.7,
                  bagging_fraction=0.5, bagging_freq=1, lambda_l2=20, verbose=-1, num_threads=4)
    bst = lgb.train(params, dtr, 1500, valid_sets=[dva], callbacks=[lgb.early_stopping(60, verbose=False)])
    print("木の数:", bst.best_iteration)
    raw = bst.predict(X, raw_score=True, num_iteration=bst.best_iteration) + init
    q = (1 / (1 + np.exp(-raw))).reshape(-1, 120)
    q /= q.sum(axis=1, keepdims=True)
    base = ex.ll_rows(pm, hit)
    l = ex.ll_rows(q, hit)
    for name, sel in [("調整", va), ("テスト", te)]:
        d = (l - base)[sel]; d = d[~np.isnan(d)]
        print(f"[{name}] 3連単の対数損失 市場 {np.nanmean(base[sel]):.4f} → 直接モデル {np.nanmean(l[sel]):.4f}  差 {d.mean():+.4f} ± {d.std() / np.sqrt(len(d)):.4f}")
    odds = ow.to_numpy(); payout = res["trifecta_payout"].fillna(0).to_numpy()
    for name, sel in [("調整", va), ("テスト", te)]:
        print(f"\n[直接モデルの回収率・{name}]")
        print(ex.roi_table(q, odds, hit, payout, dates, sel).to_string(index=False))
    imp = sorted(zip(bst.feature_importance("gain"), names), reverse=True)[:12]
    tot = sum(bst.feature_importance("gain"))
    print("効いた情報:", ", ".join(f"{n} {g / tot:.0%}" for g, n in imp))
    np.save(mm.ROOT / "data" / "pred" / "exp_trifecta_q.npy", q.astype(np.float32))


if __name__ == "__main__":
    main()
