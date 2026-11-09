"""「勝負駆け」やレースの格（SG・G1など）が着順に効くか、オッズが織り込んでいるかを調べる。

使い方:
    python src/analyze_motivation.py

見るもの:
    1. 開催の格（SG / G1 / G2 / 女子戦 / ルーキー / 一般）
    2. 予選最終日の勝負駆け（準優勝戦に乗れるかどうかのボーダー付近の選手）
    3. 期末の勝負駆け（級別審査期間の最後3週間で、A1・A2ボーダー付近の選手）
それぞれ「実際の1着率」と「オッズから見た1着確率」「第1段階AIの1着確率」を比べる。
実際 − オッズ がプラスなら、オッズはその選手を過小評価している（＝買うと得な可能性）。
"""
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

from backtest import IDX, TAKEOUT, COMBOS

ROOT = Path(__file__).resolve().parent.parent
KEY = ["date", "venue_code", "race_no"]
POINTS = {1: 10, 2: 8, 3: 6, 4: 4, 5: 2, 6: 1}

SG = ["ボートレースクラシック", "ボートレースオールスター", "グランドチャンピオン", "オーシャンカップ",
      "ボートレースメモリアル", "ボートレースダービー", "チャレンジカップ", "グランプリ", "総理大臣杯",
      "笹川賞", "全日本選手権"]
G1 = ["地区選手権", "マスターズチャンピオン", "ヤングダービー", "レディースチャンピオン", "クイーンズクライマックス",
      "BBCトーナメント", "ダイヤモンドカップ", "江戸川大賞", "高松宮記念", "全日本"]
G2 = ["モーターボート大賞", "レディースオールスター", "甲子園", "MB大賞", "秩父宮妃記念杯"]


def grade_of(title):
    t = title.replace(" ", "")
    if any(k in t for k in SG):
        return "SG"
    if any(k in t for k in G1) or (re.search(r"開設\d+周年記念", t) and not t.startswith("BTS")
                                   and "BTS" not in t):
        return "G1"
    if any(k in t for k in G2):
        return "G2"
    if any(k in t for k in ["ヴィーナス", "オールレディース", "レディース", "女子"]):
        return "女子戦"
    if "ルーキー" in t:
        return "ルーキー"
    return "一般"


def titles():
    rows = []
    for f in sorted((ROOT / "data" / "raw" / "B").glob("B*.TXT")):
        L = f.read_bytes().decode("cp932", "replace").splitlines()
        y, m, d = 2000 + int(f.stem[1:3]), int(f.stem[3:5]), int(f.stem[5:7])
        for i, l in enumerate(L):
            g = re.match(r"^(\d\d)BBGN", l)
            if g and i + 5 < len(L):
                head = unicodedata.normalize("NFKC", L[i + 1])
                day = re.search(r"第\s*(\d+)日", head)
                t = unicodedata.normalize("NFKC", L[i + 5]).strip()
                rows.append({"date": f"{y:04d}-{m:02d}-{d:02d}", "venue_code": g.group(1), "title": t,
                             "meet_day": int(day.group(1)) if day else np.nan})
    return pd.DataFrame(rows)


def market_win():
    files = sorted((ROOT / "data" / "odds").glob("odds_*.csv"))
    odds = pd.concat([pd.read_csv(f, dtype={"venue_code": str}) for f in files], ignore_index=True)
    ow = odds.pivot_table(index=KEY, columns="combo", values="odds").reindex(columns=COMBOS)
    ow = ow[ow.notna().all(axis=1)]
    pm = TAKEOUT / ow.to_numpy()
    pm = pm / pm.sum(axis=1, keepdims=True)
    mw = np.stack([pm[:, IDX[:, 0] == i].sum(axis=1) for i in range(6)], axis=1)
    idx = ow.index.to_frame(index=False)
    out = idx.loc[idx.index.repeat(6)].reset_index(drop=True)
    out["boat"] = np.tile(np.arange(1, 7), len(idx))
    out["p_mkt"] = mw.ravel()
    return out


def load():
    prog = pd.read_csv(ROOT / "data" / "csv" / "programs.csv", dtype={"venue_code": str},
                       usecols=KEY + ["boat", "racer_id", "race_name", "class", "nat_win_rate"])
    res = pd.read_csv(ROOT / "data" / "csv" / "results.csv", dtype={"venue_code": str},
                      usecols=KEY + ["boat", "racer_id", "finish"])
    races = pd.read_csv(ROOT / "data" / "csv" / "races.csv", dtype={"venue_code": str},
                        usecols=KEY + ["trifecta_payout"])
    df = prog.merge(res, on=KEY + ["boat", "racer_id"], how="inner").merge(races, on=KEY, how="left")
    df["win"] = (df["finish"] == 1).astype(int)
    df["top3"] = (df["finish"] <= 3).astype(int)
    df["pts"] = df["finish"].map(POINTS).fillna(0)
    df = df.merge(titles(), on=["date", "venue_code"], how="left")
    df["grade"] = df["title"].fillna("").map(grade_of)
    df = df.merge(market_win(), on=KEY + ["boat"], how="left")
    oof = ROOT / "data" / "pred" / "oof_predictions.csv"
    if oof.exists():
        p = pd.read_csv(oof, dtype={"venue_code": str}, usecols=KEY + ["boat", "p_win"])
        df = df.merge(p.rename(columns={"p_win": "p_ai"}), on=KEY + ["boat"], how="left")
    return df


def summarize(g, label):
    """実際の1着率と、オッズ・AIの1着確率の平均（オッズがあるレースだけ）。"""
    m = g[g.p_mkt.notna()]
    out = {"区分": label, "艇数": len(g), "1着率": g.win.mean(), "3着内率": g.top3.mean(),
           "オッズあり艇数": len(m)}
    if len(m):
        out.update({"実際(オッズあり)": m.win.mean(), "オッズの見込み": m.p_mkt.mean(),
                    "ズレ(実際-オッズ)": m.win.mean() - m.p_mkt.mean(),
                    "ズレの誤差幅±": 1.96 * np.sqrt((m.p_mkt * (1 - m.p_mkt)).sum()) / len(m)})
    if "p_ai" in g and g.p_ai.notna().any():
        a = g[g.p_ai.notna()]
        out["AIの見込み"] = a.p_ai.mean()
        out["AIとのズレ"] = a.win.mean() - a.p_ai.mean()
    return out


def grade_table(df):
    rows = []
    for gname in ["SG", "G1", "G2", "女子戦", "ルーキー", "一般"]:
        g = df[df.grade == gname]
        if len(g) == 0:
            continue
        b1 = g[g.boat == 1]
        r = summarize(b1, f"{gname} の1号艇")
        r["レース数"] = b1[KEY].drop_duplicates().shape[0]
        r["3連単配当の中央値"] = g.drop_duplicates(KEY)["trifecta_payout"].median()
        rows.append(r)
    return pd.DataFrame(rows)


def semifinal_border(df):
    """予選最終日（準優勝戦の前日）に、それまでの得点率の順位で選手を分ける。"""
    semi_days = df[df.race_name.str.contains("準優", na=False)][["venue_code", "date", "title"]].drop_duplicates()
    rows = []
    for _, s in semi_days.iterrows():
        meet = df[(df.venue_code == s.venue_code) & (df.title == s.title) & (df.date < s.date)]
        if meet.empty:
            continue
        last = meet.date.max()
        before = meet[meet.date < last]
        if before.empty:
            continue
        st = before.groupby("racer_id")["pts"].agg(["mean", "count"])
        st["rank"] = st["mean"].rank(ascending=False, method="min")
        day = meet[(meet.date == last) & meet.race_name.str.contains("予選", na=False)].copy()
        day = day.merge(st, left_on="racer_id", right_index=True, how="left")
        rows.append(day)
    d = pd.concat(rows, ignore_index=True)
    d["区分"] = pd.cut(d["rank"], [0, 10, 16, 24, 999], labels=["1〜10位（ほぼ当確）", "11〜16位（当落線の上）",
                                                             "17〜24位（勝負駆け）", "25位以下（圏外）"])
    out = [summarize(g, k) for k, g in d.groupby("区分", observed=True)]
    # 同じ枠の中で比べる（枠の有利不利を消す）
    out2 = []
    for (k, b), g in d.groupby(["区分", d.boat.clip(upper=3).map({1: "1号艇", 2: "2号艇", 3: "3〜6号艇"})],
                               observed=True):
        out2.append(summarize(g, f"{k} / {b}"))
    return pd.DataFrame(out), pd.DataFrame(out2)


def period_border(df):
    """級別審査期間（5〜10月・11〜4月）の最後3週間、期間内の勝率順位がボーダー付近の選手。"""
    d = df.copy()
    dt_ = pd.to_datetime(d.date)
    start_year = np.where(dt_.dt.month >= 11, dt_.dt.year, np.where(dt_.dt.month <= 4, dt_.dt.year - 1, dt_.dt.year))
    first_half = (dt_.dt.month >= 5) & (dt_.dt.month <= 10)
    d["period"] = np.where(first_half, start_year.astype(str) + "前", start_year.astype(str) + "後")
    end = np.where(first_half, pd.to_datetime(start_year.astype(str) + "-10-31"),
                   pd.to_datetime((start_year + 1).astype(str) + "-04-30"))
    d["days_left"] = (pd.to_datetime(end) - dt_).dt.days
    rows = []
    for per, g in d.groupby("period"):
        g = g.sort_values("date")
        late = g[g.days_left <= 21]
        if late.empty:
            continue
        cut = late.date.min()
        st = g[g.date < cut].groupby("racer_id")["pts"].agg(["mean", "count"])
        st = st[st["count"] >= 30]
        st["pct"] = st["mean"].rank(ascending=False, pct=True)
        x = late.merge(st, left_on="racer_id", right_index=True, how="inner")
        rows.append(x)
    x = pd.concat(rows, ignore_index=True)
    bins = [0, 0.15, 0.25, 0.35, 0.45, 1.0]
    labels = ["上位15%（A1安泰）", "15〜25%（A1ボーダー）", "25〜35%（A2安泰）", "35〜45%（A2ボーダー）", "45%より下"]
    x["区分"] = pd.cut(x["pct"], bins, labels=labels)
    return pd.DataFrame([summarize(g, k) for k, g in x.groupby("区分", observed=True)])


def main():
    df = load()
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    fmt = lambda t: t.round(3).to_string(index=False)
    print("== 1. 開催の格（1号艇）==")
    print(fmt(grade_table(df)))
    a, b = semifinal_border(df)
    print("\n== 2. 予選最終日の勝負駆け ==")
    print(fmt(a))
    print(fmt(b))
    print("\n== 3. 期末の勝負駆け（最後3週間）==")
    print(fmt(period_border(df)))


if __name__ == "__main__":
    main()
