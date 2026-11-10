"""当日（または指定日）のレースを予想して、買い目候補を出す。ステップ3の土台。

使い方:
    python src/predict.py 2026-11-09                 # その日の全レース
    python src/predict.py 2026-11-09 --venue 24      # 大村だけ
    python src/predict.py 2026-11-09 --venue 24 --race 12

流れ:
    1. 前日までのデータを最新にする（download.py → parse.py 相当）
    2. その日の番組表(B)を取る
    3. 各レースの直前情報（展示タイム・気象）と3連単オッズを公式サイトから取る
       ※展示前のレースは展示タイムなしで予想する（精度は下がる）
    4. AIで1着確率 → 3連単確率 → オッズと比べて期待値の高い買い目を出す
出力: data/predictions/YYYYMMDD.csv と画面表示
"""
import argparse
import datetime as dt
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import download  # noqa: E402
import parse  # noqa: E402
from backtest import COMBOS, TAKEOUT, harville  # noqa: E402
from download_beforeinfo import fetch_race as fetch_bi  # noqa: E402
from download_odds import fetch_race as fetch_odds  # noqa: E402
from features import FEATURES, KEY, WIND_DIRS, build  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

# 買い方の設定（検証結果を見て更新する）
SETTINGS = {
    "l2": 0.7, "l3": 0.5,      # 2着・3着の補正
    "alpha": 0.3,              # AIを信じる割合（残りはオッズ）
    "ev_threshold": 1.2,       # 期待値がこれを超えた買い目だけ候補にする
    "max_odds": 100,           # これより高いオッズは買わない
}
# 直前情報ページの風向きアイコン（1〜16）は、方角ではなく「その場の水面図の中での向き」。
# 学習データ（競走成績）は方角なので、レース場ごとにずらして読み替える。
# 2026-08〜11月の約1,400レースで、直前情報のアイコンと競走成績の風向きを突き合わせて決めた（各場で一番多かったずれ）。
BI_WIND_OFFSET = {"01": 11, "02": 9, "03": 5, "04": 4, "05": 0, "06": 12, "07": 14, "08": 0, "09": 1, "10": 11,
                  "11": 12, "12": 12, "13": 15, "14": 10, "15": 2, "16": 12, "17": 14, "18": 2, "19": 14, "20": 15,
                  "21": 8, "22": 7, "23": 13, "24": 6}


def bi_wind_index(venue_code, icon):
    """直前情報の風向きアイコン番号 → WIND_DIRS の番号（学習データと同じ方角）。無風(17)や不明は NaN。"""
    if pd.isna(icon) or not 1 <= int(icon) <= 16 or str(venue_code).zfill(2) not in BI_WIND_OFFSET:
        return np.nan
    return (int(icon) - 1 + BI_WIND_OFFSET[str(venue_code).zfill(2)]) % 16


def update_history(day: dt.date):
    last = day - dt.timedelta(days=1)
    have = sorted((ROOT / "data" / "raw" / "K").glob("K*.TXT"))
    start = parse.file_date(have[-1]) if have else dt.date(2024, 1, 1)
    if start <= last:
        download.main(start.isoformat(), last.isoformat())
        parse.main()


def todays_programs(day: dt.date) -> pd.DataFrame:
    download.fetch("B", day)
    return pd.DataFrame(parse.parse_programs(download.RAW / "B" / f"B{day:%y%m%d}.TXT"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("date")
    ap.add_argument("--venue")
    ap.add_argument("--race", type=int)
    a = ap.parse_args()
    day = dt.date.fromisoformat(a.date)

    update_history(day)
    prog = todays_programs(day)
    if a.venue:
        prog = prog[prog.venue_code == a.venue.zfill(2)]
    if a.race:
        prog = prog[prog.race_no == a.race]
    races = prog[KEY].drop_duplicates()
    print(f"{day} の対象レース: {len(races)}")

    # 直前情報とオッズ
    bi_rows, odds_rows = [], []
    for _, r in races.iterrows():
        bi_rows += fetch_bi(r.date, r.venue_code, int(r.race_no))
        odds_rows += fetch_odds(r.date, r.venue_code, int(r.race_no))
    bi = pd.DataFrame(bi_rows)
    up = prog.copy()
    if len(bi):
        up = up.merge(bi[KEY + ["boat", "exhibit_time_bi", "wind_speed_bi", "wave_bi", "wind_dir_bi"]],
                      on=KEY + ["boat"], how="left")
        up["exhibit_time"] = up["exhibit_time_bi"].where(up["exhibit_time_bi"] > 0)
        up["wind_speed"] = up["wind_speed_bi"]
        up["wave_cm"] = up["wave_bi"]
        up["wind_dir_i"] = [bi_wind_index(v, x) for v, x in zip(up["venue_code"], up["wind_dir_bi"])]
    else:
        up["exhibit_time"] = np.nan
        up["wind_speed"] = up["wave_cm"] = up["wind_dir_i"] = np.nan

    df = build(upcoming=up)
    df = df[df.date == day.isoformat()]
    if a.venue:
        df = df[df.venue_code == a.venue.zfill(2)]
    if a.race:
        df = df[df.race_no == a.race]
    booster = lgb.Booster(model_file=str(ROOT / "models" / "win_model.txt"))
    proba = booster.predict(df[FEATURES])
    df = df.assign(p_raw=proba[:, 0] if proba.ndim == 2 else proba)
    df["p_win"] = df["p_raw"] / df.groupby(KEY)["p_raw"].transform("sum")

    odds = pd.DataFrame(odds_rows)
    out, bets = [], []
    now = dt.datetime.now(dt.timezone(dt.timedelta(hours=9))).strftime("%Y-%m-%d %H:%M")
    for key, g in df.groupby(KEY):
        g = g.sort_values("boat")
        if len(g) != 6:
            continue
        p3 = harville(g["p_win"].to_numpy()[None, :], SETTINGS["l2"], SETTINGS["l3"])[0]
        o = odds[(odds.venue_code == key[1]) & (odds.race_no == key[2])] if len(odds) else odds
        line = {"date": key[0], "venue": g["venue"].iloc[0], "venue_code": key[1], "race_no": key[2],
                "honmei": int(g.loc[g.p_win.idxmax(), "boat"]),
                "p_win": " ".join(f"{p:.2f}" for p in g["p_win"])}
        if len(o) == 120 and o["odds"].notna().all():
            ov = o.set_index("combo").reindex(COMBOS)["odds"].to_numpy()
            pm = TAKEOUT / ov
            pm = pm / pm.sum()
            b = np.exp(SETTINGS["alpha"] * np.log(p3 + 1e-12) + (1 - SETTINGS["alpha"]) * np.log(pm))
            b = b / b.sum()
            ev = b * ov
            pick = np.where((ev > SETTINGS["ev_threshold"]) & (ov <= SETTINGS["max_odds"]))[0]
            for i in pick:
                bets.append({"date": key[0], "venue_code": key[1], "race_no": key[2], "combo": COMBOS[i],
                             "odds_at_pick": ov[i], "ev": round(ev[i], 3), "prob": round(b[i], 5),
                             "picked_at": now, "stake": 100})
            line["picks"] = " ".join(f"{COMBOS[i]}({ov[i]:.1f}倍,期待値{ev[i]:.2f})"
                                     for i in pick[np.argsort(-ev[pick])])
        else:
            top = np.argsort(-p3)[:5]
            line["picks"] = "オッズなし／参考: " + " ".join(f"{COMBOS[i]}({p3[i]:.1%})" for i in top)
        out.append(line)

    res = pd.DataFrame(out)
    path = ROOT / "data" / "predictions" / f"{day:%Y%m%d}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(path, index=False, encoding="utf-8-sig")
    for _, r in res.iterrows():
        print(f"{r.venue} {r.race_no:>2}R 本命{r.honmei}  1着確率[{r.p_win}]  {r.picks or '買い目なし'}")
    print("保存:", path)
    # エア投票の記録（レース前に保存。結果は paper.py で照合する）
    if bets:
        led = ROOT / "data" / "paper" / "bets.csv"
        led.parent.mkdir(parents=True, exist_ok=True)
        new = pd.DataFrame(bets)
        if led.exists():
            old = pd.read_csv(led, dtype={"venue_code": str})
            keys = ["date", "venue_code", "race_no", "combo"]
            old = old.merge(new[keys], on=keys, how="left", indicator=True)
            old = old[old._merge == "left_only"].drop(columns="_merge")  # 同じ買い目は新しいほうで上書き
            new = pd.concat([old, new], ignore_index=True)
        new.to_csv(led, index=False)
        print(f"エア投票に {len(bets)} 点記録: {led}")


if __name__ == "__main__":
    main()
