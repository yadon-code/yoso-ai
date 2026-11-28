"""回収率AI: 確率 × 予想した確定オッズ = 期待値。期待値が基準を超えたものだけ買う。券種ごとに測る。

    python src/v3/value.py hit_mkt                         # 3連単・単勝・2連単・2連複・3連複で、基準ごとの表（開発期間）
    python src/v3/value.py hit_mkt --pick 2025-01-01 2025-12-31 --confirm 2026-01-01 2026-08-31
        # 前半で一番よかった設定を決め、後半にそのまま当てはめる（どちらも金庫の前）

過去の検証で使えるのは確定オッズだけ。本番は締切前のオッズで買うので、確定までに下がる分を adj で割り引く:
    --adj fallback   #12 の帯ごとの下がり方（drift.FALLBACK）。既定
    --adj 1.0        割り引かない（＝確定オッズで買えた場合。上限の目安）
成績: 回収率と95%幅（日ごとのブートストラップ）、的中数、最大の連敗（買ったレースで数える）、最大の落ち込み（円）、月ごとの回収率。
"""
import argparse
from itertools import combinations, permutations

import numpy as np
import pandas as pd

import paths  # noqa: F401
from drift import fallback_factor
from evaluate import boot_ratio, load_pred
from model import COMBOS, PERM
from paths import KEY, TABLES

C2T = [f"{a}-{b}" for a, b in permutations(range(1, 7), 2)]
C2F = [f"{a}-{b}" for a, b in combinations(range(1, 7), 2)]
C3F = [f"{a}-{b}-{c}" for a, b, c in combinations(range(1, 7), 3)]


def bet_probs(q):
    """3連単120通りの確率 → 券種ごとの確率。"""
    n = len(q)
    t = np.zeros((n, 6, 6, 6))
    t[:, PERM[:, 0], PERM[:, 1], PERM[:, 2]] = q
    win = t.sum(axis=(2, 3))
    ex = t.sum(axis=3)
    out = {"3連単": (q, COMBOS), "単勝": (win, [str(i) for i in range(1, 7)])}
    out["2連単"] = (np.stack([ex[:, a - 1, b - 1] for a, b in permutations(range(1, 7), 2)], 1), C2T)
    out["2連複"] = (np.stack([ex[:, a - 1, b - 1] + ex[:, b - 1, a - 1] for a, b in combinations(range(1, 7), 2)], 1), C2F)
    tri = []
    for a, b, c in combinations(range(1, 7), 3):
        s = 0
        for x, y, z in permutations((a, b, c)):
            s = s + t[:, x - 1, y - 1, z - 1]
        tri.append(s)
    out["3連複"] = (np.stack(tri, 1), C3F)
    return out


def odds_for(keys):
    o3 = pd.read_parquet(TABLES / "odds3t.parquet").drop_duplicates(KEY)
    oo = pd.read_parquet(TABLES / "odds_other.parquet").drop_duplicates(KEY) if (TABLES / "odds_other.parquet").exists() else None
    m3 = keys.merge(o3, on=KEY, how="left")
    out = {"3連単": m3[["o_" + c for c in COMBOS]].to_numpy(dtype=float)}
    if oo is not None:
        mo = keys.merge(oo, on=KEY, how="left")

        def cols(prefix, combos):
            cc = [prefix + c for c in combos]
            return mo.reindex(columns=cc).to_numpy(dtype=float)
        out["単勝"] = cols("tan_", [str(i) for i in range(1, 7)])
        out["2連単"] = cols("t2_", C2T)
        out["2連複"] = cols("f2_", C2F)
        out["3連複"] = cols("f3_", C3F)
    return out


def hits_for(keys, bet, combos):
    pay = pd.read_parquet(TABLES / "payouts.parquet")
    pay = pay[pay.bet == bet]
    idx = {c: i for i, c in enumerate(combos)}
    pay = pay[pay.combo.isin(idx)]
    pay["j"] = pay.combo.map(idx)
    m = keys.reset_index().merge(pay, on=KEY, how="inner")
    P = np.zeros((len(keys), len(combos)))
    P[m["index"].to_numpy(), m["j"].to_numpy()] = m["payout"].to_numpy()
    settled = np.zeros(len(keys), bool)
    settled[m["index"].unique()] = True
    return P, settled  # P[i, j] = 100円あたりの払戻（当たらなければ0）


def simulate(prob, odds, pay, settled, days, th, max_odds, adj):
    ev_pre = prob * odds
    f = fallback_factor(ev_pre) if adj == "fallback" else float(adj)
    ev = ev_pre * f
    buy = (ev > th) & (odds <= max_odds) & np.isfinite(odds) & settled[:, None]
    nb = buy.sum(1)
    ret = (buy * pay).sum(1)
    return nb, ret


def stats(nb, ret, days):
    e, lo, hi = boot_ratio(ret, nb * 100, days)
    bet_races = nb > 0
    won = ret[bet_races] > 0
    streak = best = 0
    for w in won:
        streak = 0 if w else streak + 1
        best = max(best, streak)
    pnl = np.cumsum(ret[bet_races] - nb[bet_races] * 100)
    dd = float((np.maximum.accumulate(np.concatenate([[0], pnl])) - np.concatenate([[0], pnl])).max())
    return {"点数": int(nb.sum()), "的中": int(won.sum()), "回収率": round(e, 3), "95%下": round(lo, 3), "95%上": round(hi, 3),
            "最大連敗": best, "最大の落ち込み(円)": int(dd), "収支(円)": int(ret.sum() - nb.sum() * 100)}


GRID_TH = (1.0, 1.05, 1.1, 1.2, 1.3, 1.5)
GRID_MO = {"3連単": (30, 50, 100, 300), "単勝": (5, 10, 30), "2連単": (20, 50, 100), "2連複": (10, 30, 50), "3連複": (20, 50, 100)}


def table(name, start, end, adj, bets=None):
    p = load_pred(name)
    p = p[(p.date >= start) & (p.date <= end)].reset_index(drop=True)
    keys = p[KEY]
    q = p[COMBOS].to_numpy(dtype=float)
    days = keys["date"].to_numpy()
    probs = bet_probs(q)
    odds = odds_for(keys)
    rows = []
    for bet, (pr, combos) in probs.items():
        if bets and bet not in bets or bet not in odds:
            continue
        pay, settled = hits_for(keys, bet, combos)
        settled &= np.isfinite(odds[bet]).sum(1) > 0
        for mo in GRID_MO[bet]:
            for th in GRID_TH:
                nb, ret = simulate(pr, odds[bet], pay, settled, days, th, mo, adj)
                rows.append({"券種": bet, "オッズ上限": mo, "基準": th, **stats(nb, ret, days)})
    return pd.DataFrame(rows)


def monthly(name, start, end, adj, bet, th, mo):
    p = load_pred(name)
    p = p[(p.date >= start) & (p.date <= end)].reset_index(drop=True)
    keys = p[KEY]
    pr, combos = bet_probs(p[COMBOS].to_numpy(dtype=float))[bet]
    odds = odds_for(keys)[bet]
    pay, settled = hits_for(keys, bet, combos)
    nb, ret = simulate(pr, odds, pay, settled, keys.date.to_numpy(), th, mo, adj)
    d = pd.DataFrame({"月": keys.date.str[:7], "点数": nb, "払戻": ret}).groupby("月").sum()
    d["回収率"] = (d["払戻"] / (d["点数"] * 100)).round(3)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--adj", default="fallback")
    ap.add_argument("--pick", nargs=2, default=["2025-01-01", "2025-12-31"])
    ap.add_argument("--confirm", nargs=2, default=["2026-01-01", "2026-08-31"])
    ap.add_argument("--min-hits", type=int, default=30)
    a = ap.parse_args()
    t1 = table(a.name, *a.pick, a.adj)
    pd.set_option("display.width", 200)
    print(f"[決める期間 {a.pick[0]}〜{a.pick[1]}] 的中{a.min_hits}回以上で回収率の高い順 TOP15")
    cand = t1[t1["的中"] >= a.min_hits].sort_values("95%下", ascending=False)
    print(cand.head(15).to_string(index=False))
    print(f"\n[確かめる期間 {a.confirm[0]}〜{a.confirm[1]}] 上の TOP5 の設定をそのまま当てはめた結果")
    t2 = table(a.name, *a.confirm, a.adj)
    top = cand.head(5)[["券種", "オッズ上限", "基準"]]
    print(top.merge(t2, on=["券種", "オッズ上限", "基準"]).to_string(index=False))
    for _, r in top.head(2).iterrows():
        print(f"\n月ごと（{r['券種']}・上限{r['オッズ上限']}・基準{r['基準']}）")
        print(monthly(a.name, a.pick[0], a.confirm[1], a.adj, r["券種"], r["基準"], r["オッズ上限"]).to_string())


if __name__ == "__main__":
    main()
