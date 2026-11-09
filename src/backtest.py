"""過去のレースで「割安な買い目だけ買う」作戦を試し、回収率を計算する。

流れ:
1. モデルの1着確率から、3連単120通りそれぞれの確率を計算（ハーヴィル式）
2. オッズから「市場（ほかの人たち）が考えている確率」も出す
3. 2つを混ぜた確率 × オッズ ＝ 期待値。期待値が基準を超えた買い目だけ100円ずつ買う
4. 当たったら払戻金をもらう → 回収率 = 払戻の合計 ÷ 購入額の合計

期間:
    オッズがある期間の前半 = 調整用（混ぜ方や基準を決める）
    後半 = テスト（決めたルールをそのまま当てはめて成績を見る）

※ 使うオッズは「締切時」の最終オッズ。実際に買うときは締切数分前のオッズしか見えないので、
   本番の成績はここより少し悪くなる可能性があります。
"""
from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
KEY = ["date", "venue_code", "race_no"]
COMBOS = [f"{a}-{b}-{c}" for a, b, c in permutations(range(1, 7), 3)]
IDX = np.array([(a - 1, b - 1, c - 1) for a, b, c in permutations(range(1, 7), 3)])
TAKEOUT = 0.75  # 払戻率（控除率25%）


def harville(p: np.ndarray, l2: float = 1.0, l3: float = 1.0) -> np.ndarray:
    """6艇の1着確率 (n,6) → 3連単120通りの確率 (n,120)
    2着・3着は「残った艇の中で p^l2, p^l3 の比率」で決まると考える（l=1なら普通のハーヴィル式）。
    l<1 にすると、強い艇が2着・3着でも勝ちきるとは限らない、という現実に近づく。"""
    a, b, c = IDX[:, 0], IDX[:, 1], IDX[:, 2]
    q2, q3 = p ** l2, p ** l3
    s2, s3 = q2.sum(axis=1, keepdims=True), q3.sum(axis=1, keepdims=True)
    pa = p[:, a]
    pb = q2[:, b] / (s2[:, 0:1].squeeze(1)[:, None] - q2[:, a])
    pc = q3[:, c] / (s3[:, 0:1].squeeze(1)[:, None] - q3[:, a] - q3[:, b])
    return pa * pb * pc


def position_probs(p1, p2, p3):
    """1着・2着・3着それぞれの確率 (n,6) から3連単120通りの確率を作る。"""
    a, b, c = IDX[:, 0], IDX[:, 1], IDX[:, 2]
    q = p1[:, a] * p2[:, b] / (1 - p2[:, a]) * p3[:, c] / (1 - p3[:, a] - p3[:, b])
    return q / q.sum(axis=1, keepdims=True)


def fit_lambdas():
    """調整用期間（5〜7月）の全レースの結果から、2着・3着の補正(l2,l3)を決める。オッズ不要。"""
    pred = pd.read_csv(ROOT / "data" / "pred" / "predictions.csv", dtype={"venue_code": str})
    pred = pred[pred.period == "valid"]
    races = pd.read_csv(ROOT / "data" / "csv" / "races.csv", dtype={"venue_code": str})
    pw = pred.pivot_table(index=KEY, columns="boat", values="p_win").dropna()
    res = races.set_index(KEY).reindex(pw.index)
    best = None
    for l2 in np.arange(0.5, 1.21, 0.1):
        for l3 in np.arange(0.4, 1.21, 0.1):
            ll = logloss(harville(pw.to_numpy(), l2, l3), res)
            if best is None or ll < best[0]:
                best = (ll, round(l2, 2), round(l3, 2))
    print(f"2着・3着の補正: l2={best[1]}, l3={best[2]}（調整用期間の対数損失 {best[0]:.3f}、"
          f"補正なし {logloss(harville(pw.to_numpy()), res):.3f}）")
    return best[1], best[2]


def load():
    pred = pd.read_csv(ROOT / "data" / "pred" / "predictions.csv", dtype={"venue_code": str})
    odds = pd.concat([pd.read_csv(f, dtype={"venue_code": str})
                      for f in sorted((ROOT / "data" / "odds").glob("odds_*.csv"))])
    races = pd.read_csv(ROOT / "data" / "csv" / "races.csv", dtype={"venue_code": str})

    pw = pred.pivot_table(index=KEY, columns="boat", values="p_win")
    ow = odds.pivot_table(index=KEY, columns="combo", values="odds").reindex(columns=COMBOS)
    common = pw.index.intersection(ow.index)
    pw, ow = pw.loc[common], ow.loc[common]
    ok = ow.notna().all(axis=1) & pw.notna().all(axis=1)  # 欠場のあるレースは除く
    pw, ow = pw[ok], ow[ok]
    res = races.set_index(KEY).loc[pw.index, ["trifecta", "trifecta_payout"]]
    return pw, ow, res


def evaluate(p_model, odds, res, alpha, threshold, max_odds):
    """alpha: モデルを信じる割合（0=市場だけ, 1=モデルだけ）"""
    p_mkt = TAKEOUT / odds
    p_mkt = p_mkt / p_mkt.sum(axis=1, keepdims=True)
    blend = np.exp(alpha * np.log(p_model + 1e-12) + (1 - alpha) * np.log(p_mkt))
    blend = blend / blend.sum(axis=1, keepdims=True)
    ev = blend * odds
    buy = (ev > threshold) & (odds <= max_odds)
    hit_col = np.array([COMBOS.index(t) if isinstance(t, str) else -1 for t in res["trifecta"]])
    rows = np.arange(len(hit_col))
    hit = np.zeros_like(buy)
    valid = hit_col >= 0
    hit[rows[valid], hit_col[valid]] = True
    n_bets = buy.sum()
    payout = (buy & hit).any(axis=1) * res["trifecta_payout"].fillna(0).to_numpy()
    paid = payout[(buy & hit).any(axis=1)].sum()
    return {
        "alpha": alpha, "threshold": threshold, "max_odds": max_odds,
        "races_bet": int(buy.any(axis=1).sum()), "bets": int(n_bets),
        "hits": int((buy & hit).sum()),
        "roi": paid / (n_bets * 100) if n_bets else np.nan,
    }


def logloss(prob, res):
    hit_col = np.array([COMBOS.index(t) if isinstance(t, str) else -1 for t in res["trifecta"]])
    m = hit_col >= 0
    return -np.log(prob[np.arange(len(prob))[m], hit_col[m]] + 1e-12).mean()


def main():
    pw, ow, res = load()
    dates = pw.index.get_level_values("date")
    cut = sorted(set(dates))[len(set(dates)) // 2]
    l2, l3 = fit_lambdas()
    p_model = harville(pw.to_numpy(), l2, l3)
    odds = ow.to_numpy()
    p_mkt = TAKEOUT / odds
    p_mkt = p_mkt / p_mkt.sum(axis=1, keepdims=True)

    print(f"オッズがあるレース: {len(pw)}  ({min(dates)} 〜 {max(dates)})")
    print(f"調整用: 〜{cut} 未満 / テスト: {cut} 〜")
    print("\n[当たりやすさの比較] 小さいほど確率が正確（3連単の対数損失）")
    print(f"  モデル: {logloss(p_model, res):.3f}   市場（オッズ）: {logloss(p_mkt, res):.3f}"
          f"   でたらめ(1/120): {np.log(120):.3f}")

    tune, test = dates < cut, dates >= cut
    grid = []
    for alpha in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        for th in [1.0, 1.1, 1.2, 1.3, 1.5, 2.0]:
            for mo in [30, 100, 1000]:
                r = evaluate(p_model[tune], odds[tune], res[tune], alpha, th, mo)
                grid.append(r)
    g = pd.DataFrame(grid)
    g = g[g.bets >= 300]  # 買う数が少なすぎる作戦は運の影響が大きいので除く
    print("\n[調整用期間] 回収率が高かった作戦 TOP10")
    print(g.sort_values("roi", ascending=False).head(10).to_string(index=False))

    best = g.sort_values("roi", ascending=False).iloc[0]
    r = evaluate(p_model[test], odds[test], res[test], best.alpha, best.threshold, best.max_odds)
    print("\n[テスト期間] 調整用で一番よかった作戦をそのまま使った結果")
    print(r)
    base = evaluate(p_model[test], odds[test], res[test], 0.0, 0.0, 1e9)
    print("参考: 全120通りを毎レース買った場合の回収率", round(base["roi"], 3))
    return g, r


if __name__ == "__main__":
    main()
