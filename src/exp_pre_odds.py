"""「締切5分前のオッズで買って、確定オッズで払い戻される」ときの本当の成績を調べる。

これまでの検証は、確定（締切時）オッズを見て買い目を決めた場合の成績だった。
本番は5分前のオッズしか見えないので、同じレースについて
    A. 5分前オッズで買い目を決め、確定オッズで払い戻し（＝本番と同じ）
    B. 確定オッズで買い目を決め、確定オッズで払い戻し（＝これまでの検証と同じ。実際にはできない）
を並べて比べる。あわせて、買い目のオッズが5分でどれだけ動くかも出す。

使い方（src フォルダで）:
    python exp_pre_odds.py <5分前オッズのフォルダ>       # 既定は data/odds_pre
必要なもの: 対象日の data/odds（確定オッズ）・data/beforeinfo・data/oriten・data/csv・
           data/pred/oof_predictions.csv（第1段階の予想）・models/market_model.txt（本番の第2段階）
対象日は本番モデルの学習に使っていない日（trained_until より後）だけにする。
"""
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from backtest import COMBOS, IDX, TAKEOUT
from experiments import hit_index, ll_rows
from market_model import KEY, make_features, predict_win, read_dir, trifecta_probs

ROOT = Path(__file__).resolve().parent.parent


def load_pre(folder):
    rows = []
    for f in sorted(Path(folder).glob("pre_*.csv")):
        d = pd.read_csv(f, dtype={"venue_code": str, "odds": str})
        rows.append(d)
    d = pd.concat(rows, ignore_index=True).drop_duplicates(KEY, keep="last")
    mat = np.array([np.array(s.split(), dtype=float) if isinstance(s, str) and len(s.split()) == 120
                    else np.full(120, np.nan) for s in d["odds"]])
    ow = pd.DataFrame(mat, columns=COMBOS, index=pd.MultiIndex.from_frame(d[KEY]))
    ow = ow[(ow > 0).all(axis=1)]
    sec = d.set_index(KEY)["seconds_before"].reindex(ow.index)
    return ow, sec


def market(ow):
    pm = TAKEOUT / ow.to_numpy()
    pm = pm / pm.sum(axis=1, keepdims=True)
    mw = np.stack([pm[:, IDX[:, 0] == i].sum(axis=1) for i in range(6)], axis=1)
    return pm, mw


def model_probs(booster, conf, ow, pred, bi, ori):
    """本番と同じ計算で、オッズ ow を見たときの3連単確率を出す。"""
    pm, mw = market(ow)
    boats = pd.DataFrame({
        "date": np.repeat(ow.index.get_level_values(0), 6),
        "venue_code": np.repeat(ow.index.get_level_values(1), 6),
        "race_no": np.repeat(ow.index.get_level_values(2), 6),
        "boat": np.tile(np.arange(1, 7), len(ow)),
        "mw": mw.ravel(),
    })
    df = boats.merge(pred[KEY + ["boat", "p_win"]], on=KEY + ["boat"], how="left")
    df = df.merge(bi, on=KEY + ["boat"], how="left").merge(ori, on=KEY + ["boat"], how="left")
    df = make_features(df)
    pw = predict_win(booster, df).reshape(-1, 6)
    return pm, trifecta_probs(pm, pw / mw, conf["a"], conf["b"], conf["c"])


def bets(q, odds_seen, odds_final, hit, date, th, max_odds, lo=None, n_boot=2000, seed=0):
    """odds_seen で期待値を計算して買い、odds_final で払い戻し。日単位のブートストラップで95%幅。"""
    ev = q * odds_seen
    buy = (ev > th) & (odds_seen <= max_odds)
    if lo is not None:
        buy = (ev > lo) & (ev <= th) & (odds_seen <= max_odds)
    won = np.zeros_like(buy)
    won[np.arange(len(hit)), hit] = True
    pay = np.where(buy & won, odds_final, 0.0)
    n, ret = buy.sum(axis=1), pay.sum(axis=1)
    days = pd.Series(date).to_numpy()
    by = pd.DataFrame({"d": days, "n": n, "r": ret}).groupby("d").sum()
    rng = np.random.default_rng(seed)
    rs = []
    for _ in range(n_boot):
        s = by.iloc[rng.integers(0, len(by), len(by))]
        if s.n.sum():
            rs.append(s.r.sum() / s.n.sum())
    lo95, hi95 = (np.percentile(rs, [2.5, 97.5]) if rs else (np.nan, np.nan))
    return {"点数": int(n.sum()), "的中": int((buy & won).sum()),
            "回収率": ret.sum() / n.sum() if n.sum() else np.nan, "95%下": lo95, "95%上": hi95,
            "買った所のオッズの動き(中央値)": float(np.median((odds_final / odds_seen)[buy])) if buy.any() else np.nan}


def main(folder):
    conf = json.loads((ROOT / "models" / "market_model.json").read_text(encoding="utf-8"))
    booster = lgb.Booster(model_file=str(ROOT / "models" / "market_model.txt"))
    pre, sec = load_pre(folder)
    pre = pre[pre.index.get_level_values("date") > conf["trained_until"]]

    odds = read_dir("odds", "odds_2*.csv")
    odds["odds"] = pd.to_numeric(odds["odds"], errors="coerce")
    odds = odds[odds.date >= pre.index.get_level_values("date").min()]
    fin = odds.pivot_table(index=KEY, columns="combo", values="odds").reindex(columns=COMBOS)
    fin = fin[fin.notna().all(axis=1)]
    races = pd.read_csv(ROOT / "data" / "csv" / "races.csv", dtype={"venue_code": str}).set_index(KEY)

    common = pre.index.intersection(fin.index).intersection(races.index)
    res = races.loc[common, ["trifecta", "trifecta_payout"]]
    ok = res["trifecta"].isin(COMBOS).to_numpy()
    common = common[ok]
    pre, fin, res = pre.loc[common], fin.loc[common], res.loc[common]
    sec = sec.reindex(common)
    date = common.get_level_values("date")
    hit = hit_index(res)
    print(f"対象: {len(common)}レース（{date.min()}〜{date.max()}、{date.nunique()}日）"
          f"  オッズを見た時刻: 締切の{sec.median():.0f}秒前（中央値）")

    pred = pd.read_csv(ROOT / "data" / "pred" / "oof_predictions.csv", dtype={"venue_code": str})
    bi, ori = read_dir("beforeinfo", "bi_2*.csv"), read_dir("oriten", "oriten_2*.csv")
    pm_pre, q_pre = model_probs(booster, conf, pre, pred, bi, ori)
    pm_fin, q_fin = model_probs(booster, conf, fin, pred, bi, ori)

    base = ll_rows(pm_fin, hit)
    print("\n[当たりやすさ] 3連単の対数損失（小さいほど正確）。確定オッズとの差 ± 標準誤差")
    for name, p in [("確定オッズ（市場の最終形）", pm_fin), ("5分前オッズ", pm_pre),
                    ("AI（5分前オッズを見て）", q_pre), ("AI（確定オッズを見て）", q_fin)]:
        l = ll_rows(p, hit)
        d = l - base
        print(f"  {name:<18} {l.mean():.4f}   差 {d.mean():+.4f} ± {d.std(ddof=1) / np.sqrt(len(d)):.4f}")

    o_pre, o_fin = pre.to_numpy(), fin.to_numpy()
    ratio = o_fin / o_pre
    print(f"\n[オッズの動き] 確定÷5分前: 全買い目の中央値 {np.median(ratio):.2f}、"
          f"的中した買い目（勝った組）の中央値 {np.median(ratio[np.arange(len(hit)), hit]):.2f}")
    ev_pre = q_pre * o_pre
    for lo, hi in [(0.8, 0.9), (0.9, 1.0), (1.0, 1.1), (1.1, 1.3), (1.3, 9)]:
        m = (ev_pre > lo) & (ev_pre <= hi) & (o_pre <= conf["max_odds"])
        if not m.any():
            continue
        print(f"   5分前の期待値 {lo}〜{hi}: {m.sum():>6}点  確定÷5分前の中央値 {np.median(ratio[m]):.2f}"
              f"  確定オッズで計算し直した期待値の平均 {(q_fin * o_fin)[m].mean():.2f}")

    rows = []
    for th in [1.0, 1.05, 1.1, 1.2]:
        for mo in [conf["max_odds"]]:
            a = bets(q_pre, o_pre, o_fin, hit, date, th, mo)
            b = bets(q_fin, o_fin, o_fin, hit, date, th, mo)
            rows.append({"基準": th, "買い方": "A 5分前で決める（本番）", **a})
            rows.append({"基準": th, "買い方": "B 確定で決める（従来の検証）", **b})
    for lo, hi in [(0.9, 1.0), (0.8, 0.9)]:
        a = bets(q_pre, o_pre, o_fin, hit, date, hi, conf["max_odds"], lo=lo)
        rows.append({"基準": f"{lo}〜{hi}", "買い方": "A 5分前で決める（参考）", **a})
    print("\n[回収率] 払い戻しはどちらも確定オッズ")
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    allbuy = (o_fin[np.arange(len(hit)), hit]).sum() / (len(hit) * 120)
    print(f"参考: 全120通りを買った場合 {allbuy:.3f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ROOT / "data" / "odds_pre")
