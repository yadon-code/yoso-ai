"""作り方の違うAIを同じ条件で比べる実験場（本番には影響しない）。

使い方:
    START=2026-01-10 TRAIN_END=2026-07-31 VALID_END=2026-09-15 python src/experiments.py

比べるもの（3連単120通りの確率の正確さ＝対数損失。小さいほど良い）:
    market   : オッズそのまま（1/オッズ を合計1にしたもの）
    flb      : 人気の偏りを直したオッズ（確率を γ 乗して直す。γ は学習期間で決める）
    v2       : 今の本番と同じ第2段階（1着の補正を a,b,c で3連単に広げる）
    v2+flb   : 上の2つを合わせたもの
回収率は「期待値 > 基準」の買い目を100円ずつ買ったとき。日ごとのブートストラップで幅も出す。
"""
import numpy as np
import pandas as pd

import market_model as mm
from backtest import COMBOS, IDX, logloss

rng = np.random.default_rng(0)


def hit_index(res):
    return np.array([COMBOS.index(t) if isinstance(t, str) and t in COMBOS else -1 for t in res["trifecta"]])


def ll_rows(q, hit):
    ok = hit >= 0
    out = np.full(len(q), np.nan)
    out[ok] = -np.log(q[np.arange(len(q))[ok], hit[ok]] + 1e-12)
    return out


def fit_gamma(pm, hit, sel):
    best = None
    for g in np.arange(0.90, 1.31, 0.01):
        q = pm[sel] ** g
        q /= q.sum(axis=1, keepdims=True)
        v = np.nanmean(ll_rows(q, hit[sel]))
        if best is None or v < best[0]:
            best = (v, g)
    return best[1]


def power(pm, g):
    q = pm ** g
    return q / q.sum(axis=1, keepdims=True)


def roi_table(q, odds, hit, payout, dates, sel, ths=(1.0, 1.05, 1.1, 1.2), max_odds=100, nboot=500):
    rows = []
    ev = q * odds
    for th in ths:
        buy = (ev > th) & (odds <= max_odds) & sel[:, None]
        n_race = buy.sum(axis=1)
        won = np.zeros(len(q))
        ok = hit >= 0
        r = np.arange(len(q))[ok]
        won[r] = buy[r, hit[ok]] * payout[ok] / 100.0
        d = pd.DataFrame({"date": dates, "n": n_race, "ret": won})
        d = d[sel].groupby("date").sum()
        n, ret = d.n.sum(), d.ret.sum()
        if n == 0:
            rows.append({"基準": th, "点数": 0}); continue
        bs = []
        a = d.to_numpy()
        for _ in range(nboot):
            s = a[rng.integers(0, len(a), len(a))]
            if s[:, 0].sum():
                bs.append(s[:, 1].sum() / s[:, 0].sum())
        rows.append({"基準": th, "点数": int(n), "回収率": round(ret / n, 3),
                     "95%下": round(np.percentile(bs, 2.5), 3), "95%上": round(np.percentile(bs, 97.5), 3)})
    return pd.DataFrame(rows)


def main():
    ow, pm, res, df = mm.load()
    dates = np.asarray(ow.index.get_level_values("date"))
    tr, va, te = dates <= mm.TRAIN_END, (dates > mm.TRAIN_END) & (dates <= mm.VALID_END), dates > mm.VALID_END
    hit = hit_index(res)
    payout = res["trifecta_payout"].fillna(0).to_numpy()
    odds = ow.to_numpy()
    print(f"レース数 学習{tr.sum()} 調整{va.sum()} テスト{te.sum()}  ({dates.min()}〜{dates.max()})")

    g = fit_gamma(pm, hit, tr)
    print(f"人気の偏りの補正 γ = {g:.2f}（1より大きい=本命が過小評価されている）")
    model, p = mm.fit_win(df)
    pw, mw = p.reshape(-1, 6), df["mw"].to_numpy().reshape(-1, 6)
    ratio = pw / mw

    def best_abc(base):
        best = None
        for a in [0.5, 1.0, 1.5, 2.0]:
            for b in [0.0, 0.5, 1.0]:
                for c in [0.0, 0.5, 1.0]:
                    v = np.nanmean(ll_rows(mm.trifecta_probs(base[va], ratio[va], a, b, c), hit[va]))
                    if best is None or v < best[0]:
                        best = (v, a, b, c)
        return best[1:]

    cands = {"market": pm, "flb": power(pm, g)}
    abc = best_abc(pm)
    cands["v2"] = mm.trifecta_probs(pm, ratio, *abc)
    abc2 = best_abc(cands["flb"])
    cands["v2+flb"] = mm.trifecta_probs(cands["flb"], ratio, *abc2)
    print("v2 の a,b,c:", abc, " v2+flb:", abc2)
    base = ll_rows(pm, hit)
    print("\n[3連単の対数損失]  調整 / テスト / テストでの市場との差（±標準誤差）")
    for name, q in cands.items():
        l = ll_rows(q, hit)
        d = (l - base)[te]
        d = d[~np.isnan(d)]
        print(f"  {name:8s} {np.nanmean(l[va]):.4f} / {np.nanmean(l[te]):.4f} / {d.mean():+.4f} ± {d.std() / np.sqrt(len(d)):.4f}")
    for name in ["flb", "v2", "v2+flb"]:
        for label, sel in [("調整", va), ("テスト", te)]:
            print(f"\n[{name} の回収率・{label}]")
            print(roi_table(cands[name], odds, hit, payout, dates, sel).to_string(index=False))
    return cands


if __name__ == "__main__":
    main()
