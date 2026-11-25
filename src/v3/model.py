"""v3 の着順確率モデル（的中AI の本体。回収率AI も同じ確率を使う）。

作り（段階ごとのプラケット＝ルース型）:
    P(a,b,c) = P1(a) × P2(b | 1着=a) × P3(c | 1着=a, 2着=b)
    P1: 6艇から1着を選ぶ確率。P2: 残り5艇から2着。P3: 残り4艇から3着。
    それぞれ LightGBM（二値）で「その艇がその着になる」点数を出し、残った艇の中で合計1にそろえる。
    P2・P3 には「1着（と2着）が誰か」の情報を足す（1号艇が逃げたときと、4号艇がまくったときでは2着の来方が違うため）。

市場（オッズ）あり版: use_market=True で、オッズから出した「1着・2着・3着の確率」を特徴量に足す。
"""
from itertools import permutations

import lightgbm as lgb
import numpy as np
import pandas as pd

from feats import FEATURES
from paths import KEY

PERM = np.array(list(permutations(range(6), 3)))  # (120,3) 0始まり
COMBOS = [f"{a + 1}-{b + 1}-{c + 1}" for a, b, c in PERM]
PARAMS = dict(objective="binary", learning_rate=0.03, num_leaves=63, min_data_in_leaf=200,
              feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, verbose=-1, num_threads=6)
MKT1 = ["mk_logp1"]
MKT2 = ["mk_logp2c"]
MKT3 = ["mk_logp3c"]


def market_probs(odds120: np.ndarray, takeout=0.75):
    """3連単オッズ (n,120) → 市場の3連単確率 (n,120)。発売なし(NaN)は0。"""
    q = np.where(np.isfinite(odds120) & (odds120 > 0), takeout / odds120, 0.0)
    s = q.sum(axis=1, keepdims=True)
    return q / np.where(s > 0, s, 1)


def marginals(q120):
    """3連単の確率 (n,120) → 1着 (n,6)、2着|1着 (n,6,6)、3着|1着,2着 (n,6,6,6)。"""
    n = len(q120)
    t = np.zeros((n, 6, 6, 6))
    t[:, PERM[:, 0], PERM[:, 1], PERM[:, 2]] = q120
    p1 = t.sum(axis=(2, 3))
    p12 = t.sum(axis=3)
    p2c = p12 / np.where(p1[:, :, None] > 0, p1[:, :, None], 1)
    p3c = t / np.where(p12[:, :, :, None] > 0, p12[:, :, :, None], 1)
    return p1, p2c, p3c


def race_arrays(df):
    """1レース6行にそろえた表 → (レース数, 6, 特徴量) の配列と、着順 (レース数,6)。"""
    df = df.sort_values(KEY + ["boat"])
    n = len(df) // 6
    return df, n


def _stack(X1, extra_cols):
    return np.concatenate([X1] + [c[..., None] if c.ndim == X1.ndim - 1 else c for c in extra_cols], axis=-1)


class TrifectaModel:
    def __init__(self, features=None, use_market=False, params=None, rounds=3000):
        self.features = list(features or FEATURES)
        self.use_market = use_market
        self.params = {**PARAMS, **(params or {})}
        self.rounds = rounds
        self.boosters = [None, None, None]
        self.best_iter = [None, None, None]

    # ---------- 学習用の行を作る ----------
    def _base(self, df):
        df = df.sort_values(KEY + ["boat"]).reset_index(drop=True)
        n = len(df) // 6
        X = df[self.features].to_numpy(dtype=np.float32).reshape(n, 6, -1)
        fin = df["fin"].to_numpy(dtype=float).reshape(n, 6)
        mk = None
        if self.use_market:
            mk = marginals(np.stack(df.groupby(KEY, sort=False)["__q120"].first().to_numpy()))
        return df, n, X, fin, mk

    def stage_rows(self, X, fin, mk, stage):
        """学習用: 実際の着順を使って、各段階の行を作る。"""
        n = len(X)
        boats = np.arange(6)
        pos = {k: np.full(n, -1) for k in (1, 2, 3)}
        for k in (1, 2, 3):
            hit = fin == k
            ok = hit.sum(axis=1) == 1
            pos[k][ok] = hit[ok].argmax(axis=1)
        rows, ys, grp = [], [], []
        if stage == 1:
            sel = pos[1] >= 0
            Xs = X[sel]
            extra = [np.broadcast_to(boats, (sel.sum(), 6)).astype(np.float32)]
            if mk is not None:
                extra.append(np.log(mk[0][sel] + 1e-9).astype(np.float32))
            F = _stack(Xs, extra)
            y = (fin[sel] == 1)
            mask = np.ones_like(y, dtype=bool)
        elif stage == 2:
            sel = (pos[1] >= 0) & (pos[2] >= 0)
            a = pos[1][sel]
            Xs = X[sel]
            m = sel.sum()
            win_wr = Xs[np.arange(m), a, self.features.index("nat_win_rate")]  # 1着の艇の勝率
            extra = [np.repeat(a[:, None], 6, 1).astype(np.float32), (boats[None, :] - a[:, None]).astype(np.float32),
                     win_wr[:, None].repeat(6, 1).astype(np.float32)]
            if mk is not None:
                extra.append(np.log(mk[1][sel][np.arange(m), a] + 1e-9).astype(np.float32))
            F = _stack(Xs, extra)
            y = fin[sel] == 2
            mask = boats[None, :] != a[:, None]
        else:
            sel = (pos[1] >= 0) & (pos[2] >= 0) & (pos[3] >= 0)
            a, b = pos[1][sel], pos[2][sel]
            Xs = X[sel]
            m = sel.sum()
            extra = [np.repeat(a[:, None], 6, 1).astype(np.float32), np.repeat(b[:, None], 6, 1).astype(np.float32),
                     (boats[None, :] - a[:, None]).astype(np.float32), (boats[None, :] - b[:, None]).astype(np.float32)]
            if mk is not None:
                extra.append(np.log(mk[2][sel][np.arange(m), a, b] + 1e-9).astype(np.float32))
            F = _stack(Xs, extra)
            y = fin[sel] == 3
            mask = (boats[None, :] != a[:, None]) & (boats[None, :] != b[:, None])
        return F[mask], y[mask].astype(np.float32)

    def fit(self, tr, es):
        _, _, Xt, ft, mkt = self._base(tr)
        _, _, Xe, fe, mke = self._base(es)
        for s in (1, 2, 3):
            Ft, yt = self.stage_rows(Xt, ft, mkt, s)
            Fe, ye = self.stage_rows(Xe, fe, mke, s)
            dt = lgb.Dataset(Ft, yt, free_raw_data=True)
            de = lgb.Dataset(Fe, ye, reference=dt)
            b = lgb.train(self.params, dt, self.rounds, valid_sets=[de],
                          callbacks=[lgb.early_stopping(100, verbose=False)])
            self.boosters[s - 1] = b
            self.best_iter[s - 1] = b.best_iteration
        return self

    # ---------- 予想: 120通りの確率 ----------
    def predict(self, df):
        df, n, X, _, mk = self._base(df.assign(fin=np.nan) if "fin" not in df else df)
        boats = np.arange(6)
        idx = self.features.index("nat_win_rate")

        def score(stage, F):
            b = self.boosters[stage - 1]
            return b.predict(F.reshape(-1, F.shape[-1]), raw_score=True, num_iteration=b.best_iteration).reshape(F.shape[:-1])

        # 1着
        extra = [np.broadcast_to(boats, (n, 6)).astype(np.float32)]
        if mk is not None:
            extra.append(np.log(mk[0] + 1e-9).astype(np.float32))
        s1 = score(1, _stack(X, extra))
        p1 = np.exp(s1 - s1.max(1, keepdims=True))
        p1 /= p1.sum(1, keepdims=True)
        # 2着（1着=a ごとに）
        p2 = np.zeros((n, 6, 6))
        for a in range(6):
            extra = [np.full((n, 6), a, np.float32), (boats[None, :] - a).astype(np.float32).repeat(n, 0),
                     X[:, a, idx][:, None].repeat(6, 1)]
            if mk is not None:
                extra.append(np.log(mk[1][:, a] + 1e-9).astype(np.float32))
            s = score(2, _stack(X, extra))
            s[:, a] = -np.inf
            e = np.exp(s - s.max(1, keepdims=True))
            p2[:, a] = e / e.sum(1, keepdims=True)
        # 3着（1着=a, 2着=b ごとに）
        p3 = np.zeros((n, 6, 6, 6))
        for a in range(6):
            for b in range(6):
                if a == b:
                    continue
                extra = [np.full((n, 6), a, np.float32), np.full((n, 6), b, np.float32),
                         (boats[None, :] - a).astype(np.float32).repeat(n, 0), (boats[None, :] - b).astype(np.float32).repeat(n, 0)]
                if mk is not None:
                    extra.append(np.log(mk[2][:, a, b] + 1e-9).astype(np.float32))
                s = score(3, _stack(X, extra))
                s[:, [a, b]] = -np.inf
                e = np.exp(s - s.max(1, keepdims=True))
                p3[:, a, b] = e / e.sum(1, keepdims=True)
        q = p1[:, PERM[:, 0]] * p2[:, PERM[:, 0], PERM[:, 1]] * p3[:, PERM[:, 0], PERM[:, 1], PERM[:, 2]]
        q /= q.sum(1, keepdims=True)
        keys = df.iloc[::6][KEY].reset_index(drop=True)
        return keys, q

    def importance(self, stage=1, top=20):
        b = self.boosters[stage - 1]
        names = self.features + ["stage_boat", "mk"][: (1 + int(self.use_market))] if stage == 1 else None
        imp = b.feature_importance("gain")
        names = (self.features + [f"x{i}" for i in range(len(imp) - len(self.features))])
        s = pd.Series(imp, index=names)
        return (s / s.sum()).sort_values(ascending=False).head(top)
