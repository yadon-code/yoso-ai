"""エア投票（お金をかけずに記録だけする投票）の成績を集計する。

使い方:
    python src/paper.py            # これまでの全記録を、結果と照らし合わせて集計

predict.py がレース前に data/paper/bets.csv に買い目を記録する。
このスクリプトは、結果（data/csv/races.csv）が出たレースについて的中・払戻を計算して、
日ごと・全体の回収率を出す。結果は data/paper/report.csv にも保存する。
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
KEY = ["date", "venue_code", "race_no"]


def main():
    path = ROOT / "data" / "paper" / "bets.csv"
    if not path.exists():
        print("まだエア投票の記録がありません")
        return
    bets = pd.read_csv(path, dtype={"venue_code": str})
    races = pd.read_csv(ROOT / "data" / "csv" / "races.csv", dtype={"venue_code": str})
    m = bets.merge(races[KEY + ["trifecta", "trifecta_payout"]], on=KEY, how="left")
    done = m[m.trifecta.notna()].copy()
    done["hit"] = done.combo == done.trifecta
    done["payout"] = done.hit * done.trifecta_payout * done.stake / 100
    if done.empty:
        print(f"記録 {len(bets)} 点。まだ結果の出たレースがありません")
        return
    day = done.groupby("date").agg(点数=("combo", "size"), 的中=("hit", "sum"),
                                   購入=("stake", "sum"), 払戻=("payout", "sum"))
    day["回収率"] = (day.払戻 / day.購入).round(3)
    print(day.to_string())
    tot = day.sum(numeric_only=True)
    print(f"\n合計: {int(tot.点数)}点 的中{int(tot.的中)} 購入{int(tot.購入)}円 払戻{int(tot.払戻)}円 "
          f"回収率 {tot.払戻 / tot.購入:.1%}")
    print(f"結果待ち: {len(m) - len(done)}点")
    day.to_csv(ROOT / "data" / "paper" / "report.csv")


if __name__ == "__main__":
    main()
