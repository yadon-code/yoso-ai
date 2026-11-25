"""v3 の成績を測る（的中AI・回収率AI 共通）。幅は「日ごとに引き直す」ブートストラップの95%。

    python src/v3/evaluate.py hit_nomkt [hit_mkt ...]          # 確率の正確さと「上位N点の的中率」
    python src/v3/evaluate.py hit_mkt --roi                    # 期待値で買ったときの回収率も（確定オッズ・払戻）

比べる相手: 確定オッズ（市場）。
"""
import argparse

import numpy as np
import pandas as pd

import paths  # noqa: F401
from model import COMBOS, PERM, market_probs
from paths import KEY, TABLES, WORK

RNG = np.random.default_rng(0)


def load_pred(name):
    files = sorted((WORK / name).glob("pred_*.parquet"))
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def truth(keys):
    rc = pd.read_parquet(TABLES / "races.parquet")[KEY + ["trifecta", "trifecta_payout"]].drop_duplicates(KEY)
    t = keys.merge(rc, on=KEY, how="left")
    idx = {c: i for i, c in enumerate(COMBOS)}
    t["hit"] = t["trifecta"].map(idx).fillna(-1).astype(int)
    return t


def market(keys):
    o = pd.read_parquet(TABLES / "odds3t.parquet").drop_duplicates(KEY)
    m = keys.merge(o, on=KEY, how="left")
    odds = m[["o_" + c for c in COMBOS]].to_numpy(dtype=float)
    ok = np.isfinite(odds).sum(1) >= 100
    return odds, ok


def boot_mean(x, days, n=1000):
    """日ごとにまとめて引き直した平均の95%幅。"""
    d = pd.DataFrame({"x": x, "d": days}).groupby("d")["x"].agg(["sum", "count"]).to_numpy()
    est = d[:, 0].sum() / d[:, 1].sum()
    bs = []
    for _ in range(n):
        s = d[RNG.integers(0, len(d), len(d))]
        bs.append(s[:, 0].sum() / s[:, 1].sum())
    return est, np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def boot_ratio(num, den, days, n=1000):
    d = pd.DataFrame({"a": num, "b": den, "d": days}).groupby("d")[["a", "b"]].sum().to_numpy()
    est = d[:, 0].sum() / max(d[:, 1].sum(), 1e-9)
    bs = []
    for _ in range(n):
        s = d[RNG.integers(0, len(d), len(d))]
        if s[:, 1].sum() > 0:
            bs.append(s[:, 0].sum() / s[:, 1].sum())
    return est, np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def derived(q):
    """120通り → 1着 (n,6)、2連単 (n,30)、2連複 (n,15)、3連複 (n,20)。"""
    n = len(q)
    t = np.zeros((n, 6, 6, 6))
    t[:, PERM[:, 0], PERM[:, 1], PERM[:, 2]] = q
    win = t.sum(axis=(2, 3))
    ex = t.sum(axis=3)
    return win, ex


def accuracy(name, q, keys, tr, mk_q=None, ok=None):
    if ok is not None and not ok.any():
        return {"モデル": name, "レース数": 0}
    hit = tr["hit"].to_numpy()
    days = keys["date"].to_numpy()
    v = hit >= 0
    if ok is not None:
        v = v & ok
    rows = []
    ll = -np.log(q[np.arange(len(q)), np.clip(hit, 0, None)] + 1e-12)
    win, ex = derived(q)
    a = PERM[np.clip(hit, 0, None)]
    ll_win = -np.log(win[np.arange(len(q)), a[:, 0]] + 1e-12)
    ll_ex = -np.log(ex[np.arange(len(q)), a[:, 0], a[:, 1]] + 1e-12)
    brier = ((win - np.eye(6)[a[:, 0]]) ** 2).sum(1)
    rank = (-q).argsort(1)
    pos = (rank == hit[:, None]).argmax(1)
    win_top = win.argmax(1) == a[:, 0]
    out = {"モデル": name, "レース数": int(v.sum())}
    for lab, x in [("3連単の対数損失", ll), ("1着の対数損失", ll_win), ("2連単の対数損失", ll_ex), ("1着のブライアー", brier),
                   ("本命の1着的中率", win_top.astype(float))] + [(f"3連単上位{k}点の的中率", (pos < k).astype(float)) for k in (1, 3, 6, 10, 20)]:
        e, lo, hi = boot_mean(x[v], days[v])
        out[lab] = f"{e:.4f} [{lo:.4f}, {hi:.4f}]"
    if mk_q is not None:
        llm = -np.log(mk_q[np.arange(len(q)), np.clip(hit, 0, None)] + 1e-12)
        e, lo, hi = boot_mean((ll - llm)[v], days[v])
        out["市場との差（3連単の対数損失、マイナスが良い）"] = f"{e:+.4f} [{lo:+.4f}, {hi:+.4f}]"
        wm, exm = derived(mk_q)
        llwm = -np.log(wm[np.arange(len(q)), a[:, 0]] + 1e-12)
        e, lo, hi = boot_mean((ll_win - llwm)[v], days[v])
        out["市場との差（1着の対数損失）"] = f"{e:+.4f} [{lo:+.4f}, {hi:+.4f}]"
    return out


def roi_table(q, odds, tr, days, ths=(1.0, 1.1, 1.2, 1.3, 1.5), max_odds=(50, 100, 300), adj=1.0):
    """期待値 = 確率 × オッズ × adj（adj<1 は「確定までにオッズが下がる分」の割引）。100円ずつ買う。"""
    hit = tr["hit"].to_numpy()
    pay = tr["trifecta_payout"].fillna(0).to_numpy()
    ev = q * odds * adj
    rows = []
    for mo in max_odds:
        for th in ths:
            buy = (ev > th) & (odds <= mo) & np.isfinite(odds)
            nb = buy.sum(1)
            won = np.where(hit >= 0, buy[np.arange(len(q)), np.clip(hit, 0, None)], False) * pay
            e, lo, hi = boot_ratio(won, nb * 100, days)
            rows.append({"オッズ上限": mo, "基準": th, "点数": int(nb.sum()), "的中": int((won > 0).sum()),
                         "回収率": round(e, 3), "95%下": round(lo, 3), "95%上": round(hi, 3)})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="+")
    ap.add_argument("--roi", action="store_true")
    ap.add_argument("--since", default=None)
    a = ap.parse_args()
    res = []
    for name in a.names:
        p = load_pred(name)
        if a.since:
            p = p[p.date >= a.since]
        keys = p[KEY].reset_index(drop=True)
        q = p[COMBOS].to_numpy(dtype=float)
        tr = truth(keys)
        odds, ok = market(keys)
        mq = market_probs(odds)
        print(f"\n== {name}: {keys.date.min()}〜{keys.date.max()}  {len(keys)}レース（オッズあり {ok.sum()}）")
        res.append(accuracy(name, q, keys, tr))
        if ok.any():
            res.append(accuracy(name + "（オッズのあるレースだけ）", q, keys, tr, mq, ok))
            res.append(accuracy("市場（確定オッズ）", mq, keys, tr, None, ok))
        if a.roi and ok.any():
            print(roi_table(q[ok], odds[ok], tr[ok].reset_index(drop=True), keys.date.to_numpy()[ok]).to_string(index=False))
    print(pd.DataFrame(res).set_index("モデル").T.to_string())


if __name__ == "__main__":
    main()
