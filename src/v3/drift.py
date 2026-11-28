"""回収率AI の部品: 「締切前のオッズ」から「確定オッズ」がいくらになるかを予想する。

学ぶもの: log(確定オッズ ÷ 締切前のオッズ)（3連単の各組ごと）
材料: 締切前のオッズ（対数・人気順）、締切までの秒数、AI の確率と市場の確率の比（#12: お得に見える所ほど下がる）、
      場・時刻・グレード、そのレースのオッズ全体の形（1番人気の確率など）
データ: data_full/odds_snap（Mac で記録している締切 15/10/5/2/1 分前のオッズ）と、BOATCAST の確定オッズ。

    python src/v3/drift.py report        # たまった量と、今の「下がり方」の表（期待値の帯ごと）
    python src/v3/drift.py train         # 学習（3,000レース以上たまってから）
"""
import json
import sys

import numpy as np
import pandas as pd

import paths  # noqa: F401
from model import COMBOS
from paths import DATA, KEY, TABLES, WORK

SNAP = DATA / "odds_snap"
# 足りないあいだの代わり: #12（2026-11-11〜16、877レース）の「5分前の期待値の帯 → 確定までの下がり方（中央値）」
FALLBACK = [(0.0, 0.8, 0.94), (0.8, 0.9, 0.94), (0.9, 1.0, 0.88), (1.0, 1.1, 0.80), (1.1, 1.3, 0.64), (1.3, 99, 0.54)]


def fallback_factor(ev_pre):
    f = np.ones_like(ev_pre)
    for lo, hi, k in FALLBACK:
        f = np.where((ev_pre >= lo) & (ev_pre < hi), k, f)
    return f


def load_snaps(target_min=5):
    rows = []
    for p in sorted(SNAP.glob("snap_*.jsonl")):
        for l in p.read_text(encoding="utf-8").splitlines():
            r = json.loads(l)
            if r["ok"] and r["target_min"] == target_min:
                rows.append(r)
    if not rows:
        return pd.DataFrame()
    d = pd.DataFrame(rows)
    d = d.sort_values("fetched_at").drop_duplicates(KEY, keep="last")
    return d


def pairs(target_min=5):
    """締切前と確定のオッズの組 → (keys, pre (n,120), final (n,120))。"""
    s = load_snaps(target_min)
    if s.empty:
        return None
    o = pd.read_parquet(TABLES / "odds3t.parquet").drop_duplicates(KEY)
    m = s.merge(o, on=KEY, how="inner")
    pre = np.array([[np.nan if x is None else x for x in v] for v in m["odds"]], dtype=float)
    fin = m[["o_" + c for c in COMBOS]].to_numpy(dtype=float)
    return m[KEY + ["seconds_before"]], pre, fin


def report():
    for t in (15, 10, 5, 2, 1):
        p = pairs(t)
        if p is None:
            print(f"{t}分前: まだ記録なし")
            continue
        keys, pre, fin = p
        r = fin / pre
        ok = np.isfinite(r)
        print(f"{t}分前: {len(keys)} レース  確定÷締切前 の中央値 {np.nanmedian(r[ok]):.3f}")


def features(pre, q_ai, keys):
    pm = np.where(np.isfinite(pre) & (pre > 0), 0.75 / pre, 0)
    pm = pm / pm.sum(1, keepdims=True)
    rank = (-pm).argsort(1).argsort(1)
    n = len(pre)
    X = {
        "log_pre": np.log(pre).ravel(),
        "rank": rank.ravel(),
        "log_ratio_ai": (np.log(q_ai + 1e-9) - np.log(pm + 1e-9)).ravel(),
        "ev_pre": (q_ai * pre).ravel(),
        "top_p": np.repeat(pm.max(1), 120),
        "secs": np.repeat(keys["seconds_before"].to_numpy(dtype=float), 120),
        "venue": np.repeat(keys["venue_code"].astype(int).to_numpy(), 120),
        "first": np.tile(np.array([int(c[0]) for c in COMBOS]), n),
    }
    return pd.DataFrame(X)


def train():
    import lightgbm as lgb
    p = pairs(5)
    if p is None or len(p[0]) < 3000:
        print("まだ足りません（3,000レース以上たまってから）:", 0 if p is None else len(p[0]))
        return
    keys, pre, fin = p
    # AI の確率は、その日を学習に使っていない予想（walk.py の hit_mkt）を使う
    pr = pd.concat([pd.read_parquet(f) for f in sorted((WORK / "hit_mkt_live").glob("pred_*.parquet"))])
    m = keys.merge(pr, on=KEY, how="left")
    q = m[COMBOS].to_numpy(dtype=float)
    ok = np.isfinite(q).all(1)
    X = features(pre[ok], q[ok], keys[ok].reset_index(drop=True))
    y = np.log(fin[ok] / pre[ok]).ravel()
    good = np.isfinite(y) & np.isfinite(X["log_pre"].to_numpy())
    days = np.repeat(keys[ok]["date"].to_numpy(), 120)[good]
    cut = np.sort(np.unique(days))[int(len(np.unique(days)) * 0.8)]
    tr, te = days < cut, days >= cut
    Xg, yg = X[good], y[good]
    b = lgb.train(dict(objective="regression", learning_rate=0.05, num_leaves=31, min_data_in_leaf=500, verbose=-1),
                  lgb.Dataset(Xg[tr], yg[tr]), 2000, valid_sets=[lgb.Dataset(Xg[te], yg[te])],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
    pred = b.predict(Xg[te])
    print(f"確定÷締切前 の予想: テストの誤差（二乗平均）{np.sqrt(np.mean((pred - yg[te]) ** 2)):.4f}  "
          f"何もしない（0）{np.sqrt(np.mean(yg[te] ** 2)):.4f}")
    b.save_model(str(WORK / "drift_model.txt"))


if __name__ == "__main__":
    {"report": report, "train": train}[sys.argv[1]]()
