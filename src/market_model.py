"""「オッズのズレ」を探すモデル（第2段階）。

考え方:
    オッズ（市場）の確率はとても正確。そこで、オッズの確率を出発点にして、
    「直前情報」や「第1段階AIの予想」から、市場が見落としている分だけを補正する。

期間:
    学習:   2025-01-01 〜 2026-05-31
    調整用: 2026-06-01 〜 2026-08-31（補正の強さ・買う基準を決める）
    テスト: 2026-09-01 〜          （最後に一度だけ見る）

使うデータ: data/odds（3連単オッズ）、data/beforeinfo（直前情報。2026年5月〜）、
           data/oriten（オリジナル展示）、data/pred/oof_predictions.csv（第1段階AI、時期をずらして学習したもの）
"""
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from backtest import COMBOS, IDX, TAKEOUT, logloss

ROOT = Path(__file__).resolve().parent.parent
KEY = ["date", "venue_code", "race_no"]
import os
TRAIN_END = os.environ.get("TRAIN_END", "2026-05-31")
VALID_END = os.environ.get("VALID_END", "2026-08-31")
START = os.environ.get("START", "2000-01-01")  # これより前のレースは使わない

FEATURES = [
    "boat", "venue_code_i", "log_mw", "log_ratio_ai", "mw_rank",
    "exh_st", "exh_st_diff", "exh_st_rank", "exh_st_f", "course_change", "exh_course",
    "exhibit_time_bi", "exhibit_time_bi_diff", "exhibit_time_bi_rank",
    "tilt", "tilt_diff", "propeller_new", "parts_n", "adjust_weight", "weight_day_diff",
    "wind_speed_bi", "wind_dir_bi", "wave_bi",
    # オリジナル展示（同じレースの中での差と順位。小さいほど速い）
    "lap_diff", "lap_rank", "turn_diff", "turn_rank", "straight_diff", "straight_rank",
    "lap_half",
]


def read_dir(name, pattern):
    files = sorted((ROOT / "data" / name).glob(pattern))
    return pd.concat([pd.read_csv(f, dtype={"venue_code": str}) for f in files], ignore_index=True)


def make_features(df):
    """1レース6艇ぶんの行（mw=オッズの1着確率, p_win=第1段階AI, 直前情報, オリジナル展示）から特徴量を作る。
    学習（load）と当日の予想（live.py）で同じものを使う。"""
    df = df.copy()
    df["lap_half"] = (df["lap_kind"] == "半周ラップ").astype(float) if "lap_kind" in df else 0.0
    df["venue_code_i"] = df["venue_code"].astype(int)
    df["log_mw"] = np.log(df["mw"])
    df["log_ratio_ai"] = np.log(df["p_win"]) - df["log_mw"]
    df["course_change"] = (df["exh_course"] != df["boat"]).astype(float).where(df["exh_course"].notna())
    g = df.groupby(KEY)
    df["mw_rank"] = g["mw"].rank(ascending=False)
    for c in ["exh_st", "exhibit_time_bi", "lap", "turn", "straight"]:
        df[f"{c}_rank"] = g[c].rank(ascending=True)
    for c in ["exh_st", "exhibit_time_bi", "tilt", "weight_day", "lap", "turn", "straight"]:
        df[f"{c}_diff"] = df[c] - g[c].transform("mean")
    for c in FEATURES:
        if c not in df:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def market_win_probs(odds_vec):
    """3連単オッズ120個（COMBOS順）→ 市場の3連単確率と1着確率。"""
    pm = TAKEOUT / np.asarray(odds_vec, dtype=float)
    pm = pm / pm.sum()
    mw = np.array([pm[IDX[:, 0] == i].sum() for i in range(6)])
    return pm, mw


def load():
    odds = read_dir("odds", "odds_*.csv")
    odds["odds"] = pd.to_numeric(odds["odds"], errors="coerce")  # 中身が空のファイルがあっても数値にそろえる
    ow = odds.pivot_table(index=KEY, columns="combo", values="odds").reindex(columns=COMBOS)
    ow = ow[ow.notna().all(axis=1)]
    ow = ow[ow.index.get_level_values("date") >= START]
    pm = TAKEOUT / ow.to_numpy()
    pm = pm / pm.sum(axis=1, keepdims=True)
    mw = np.stack([pm[:, IDX[:, 0] == i].sum(axis=1) for i in range(6)], axis=1)

    races = pd.read_csv(ROOT / "data" / "csv" / "races.csv", dtype={"venue_code": str}).set_index(KEY)
    res = races.reindex(ow.index)[["trifecta", "trifecta_payout"]]

    boats = pd.DataFrame({
        "date": np.repeat(ow.index.get_level_values(0), 6),
        "venue_code": np.repeat(ow.index.get_level_values(1), 6),
        "race_no": np.repeat(ow.index.get_level_values(2), 6),
        "boat": np.tile(np.arange(1, 7), len(ow)),
        "mw": mw.ravel(),
    })
    pred = pd.read_csv(ROOT / "data" / "pred" / os.environ.get("OOF_FILE", "oof_predictions.csv"),
                       dtype={"venue_code": str})
    bi = read_dir("beforeinfo", "bi_*.csv")
    ori = read_dir("oriten", "oriten_*.csv")
    df = boats.merge(pred[KEY + ["boat", "p_win"]], on=KEY + ["boat"], how="left")
    df = df.merge(bi, on=KEY + ["boat"], how="left")
    df = df.merge(ori, on=KEY + ["boat"], how="left")
    first = res["trifecta"].str[0]
    df["win"] = (df["boat"].astype(str).to_numpy() == np.repeat(first.to_numpy(), 6)).astype(int)
    df = make_features(df)
    return ow, pm, res, df


def fit_win(df):
    tr = df[df.date <= TRAIN_END]
    va = df[(df.date > TRAIN_END) & (df.date <= VALID_END)]
    model = lgb.LGBMClassifier(n_estimators=3000, learning_rate=0.02, num_leaves=15,
                               min_child_samples=200, subsample=0.8, subsample_freq=1,
                               colsample_bytree=0.8, reg_lambda=5, verbose=-1)
    # オッズの確率を出発点にする（init_score）。モデルは「ズレ」だけを学ぶ
    base_tr = np.log(tr["mw"] / (1 - tr["mw"]))
    base_va = np.log(va["mw"] / (1 - va["mw"]))
    model.fit(tr[FEATURES], tr["win"], init_score=base_tr,
              eval_set=[(va[FEATURES], va["win"])], eval_init_score=[base_va],
              categorical_feature=["venue_code_i"],
              callbacks=[lgb.early_stopping(200, verbose=False)])
    raw = model.predict(df[FEATURES], raw_score=True) + np.log(df["mw"] / (1 - df["mw"])).to_numpy()
    p = 1 / (1 + np.exp(-raw))
    p = p / pd.Series(p).groupby([df[k] for k in KEY]).transform("sum").to_numpy()
    return model, p


def trifecta_probs(pm, ratio, a, b, c):
    """市場の3連単確率を、1着確率の補正比率で調整する。"""
    r = np.log(ratio)
    adj = a * r[:, IDX[:, 0]] + b * r[:, IDX[:, 1]] + c * r[:, IDX[:, 2]]
    q = pm * np.exp(adj)
    return q / q.sum(axis=1, keepdims=True)


def roi(q, odds, res, th, max_odds):
    ev = q * odds
    buy = (ev > th) & (odds <= max_odds)
    hit_col = np.array([COMBOS.index(t) if isinstance(t, str) else -1 for t in res["trifecta"]])
    ok = hit_col >= 0
    hit = np.zeros_like(buy)
    hit[np.arange(len(buy))[ok], hit_col[ok]] = True
    win = (buy & hit).any(axis=1)
    paid = res["trifecta_payout"].fillna(0).to_numpy()[win].sum()
    n = buy.sum()
    return n, (paid / (n * 100) if n else np.nan), win.sum()


def main():
    ow, pm, res, df = load()
    dates = ow.index.get_level_values("date")
    print(f"オッズのあるレース: {len(ow)}（{dates.min()}〜{dates.max()}）、"
          f"直前情報あり: {df.groupby(KEY)['exh_st'].count().gt(0).mean():.0%}")
    model, p = fit_win(df)
    pw = p.reshape(-1, 6)
    mw = df["mw"].to_numpy().reshape(-1, 6)
    win_col = res["trifecta"].str[0].astype(float).to_numpy() - 1
    ok = ~np.isnan(win_col)

    def wll(m, sel):
        s = sel & ok
        return -np.log(m[s, win_col[s].astype(int)]).mean()

    for name, sel in [("学習", dates <= TRAIN_END), ("調整用", (dates > TRAIN_END) & (dates <= VALID_END)),
                      ("テスト", dates > VALID_END)]:
        sel = np.asarray(sel)
        print(f"[{name}] 1着確率の対数損失  市場 {wll(mw, sel):.4f} → 補正後 {wll(pw, sel):.4f}")

    va = np.asarray((dates > TRAIN_END) & (dates <= VALID_END))
    te = np.asarray(dates > VALID_END)
    ratio = pw / mw
    best = None
    for a in [0.5, 1.0, 1.5]:
        for b in [0.0, 0.5, 1.0]:
            for c in [0.0, 0.5]:
                ll = logloss(trifecta_probs(pm[va], ratio[va], a, b, c), res[va])
                if best is None or ll < best[0]:
                    best = (ll, a, b, c)
    _, a, b, c = best
    q = trifecta_probs(pm, ratio, a, b, c)
    print(f"3連単の補正 a={a}, b={b}, c={c}")
    for name, sel in [("調整用", va), ("テスト", te)]:
        print(f"[{name}] 3連単の対数損失  市場 {logloss(pm[sel], res[sel]):.4f} → 補正後 {logloss(q[sel], res[sel]):.4f}")

    odds = ow.to_numpy()
    rows = []
    for th in [1.0, 1.05, 1.1, 1.2, 1.3, 1.5]:
        for mo in [20, 50, 100, 300]:
            n, r, h = roi(q[va], odds[va], res[va], th, mo)
            n2, r2, h2 = roi(q[te], odds[te], res[te], th, mo)
            rows.append({"基準": th, "オッズ上限": mo, "調整_点数": n, "調整_回収率": r,
                         "テスト_点数": n2, "テスト_回収率": r2})
    t = pd.DataFrame(rows)
    print(t.round(3).to_string(index=False))
    imp = pd.Series(model.booster_.feature_importance("gain"), index=FEATURES)
    print("補正に効いた情報:\n", (imp / imp.sum()).sort_values(ascending=False).head(10).round(3))


def predict_win(booster, df):
    """保存したモデルで、オッズの1着確率を補正した1着確率を出す（レースごとに合計1）。"""
    raw = booster.predict(df[FEATURES], raw_score=True) + np.log(df["mw"] / (1 - df["mw"])).to_numpy()
    p = 1 / (1 + np.exp(-raw))
    return p / pd.Series(p).groupby([df[k].to_numpy() for k in KEY]).transform("sum").to_numpy()


def save():
    """毎日の予想で使うモデルを作って保存する。
    検証と同じ分け方で木の数を決め、全期間のデータで作り直す。"""
    import json
    ow, pm, res, df = load()
    model, p = fit_win(df)
    n = model.best_iteration_ or model.n_estimators
    final = lgb.LGBMClassifier(n_estimators=n, learning_rate=0.02, num_leaves=15, min_child_samples=200,
                               subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5, verbose=-1)
    final.fit(df[FEATURES], df["win"], init_score=np.log(df["mw"] / (1 - df["mw"])),
              categorical_feature=["venue_code_i"])
    out = ROOT / "models"
    final.booster_.save_model(str(out / "market_model.txt"))
    conf = {"a": 1.5, "b": 0.5, "c": 0.5, "ev_threshold": 1.0, "max_odds": 100,
            "trained_until": str(df.date.max()), "n_trees": int(n)}
    (out / "market_model.json").write_text(json.dumps(conf, ensure_ascii=False, indent=1), encoding="utf-8")
    print("保存:", out / "market_model.txt", conf)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "save":
        save()
    else:
        main()
