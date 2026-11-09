"""学習用の表を作る。1行=1艇。レース前にわかる情報だけを特徴量にする。

使う情報:
- 番組表: 艇番、級別、年齢、体重、全国/当地の勝率・2連率、モーター/ボートの2連率、場、距離
- 直前情報: 展示タイム（レース直前の展示航走で計測。締切前に公開される）
- レース内での相対値: 勝率や展示タイムがそのレースの6艇の中で何番目か、平均との差
- 過去の実績: 選手の直近成績、枠番別・コース別の成績、今節の成績、モーターの直近成績、F持ち
- 気象: 風速、風向き、波高
- 艇どうしの関係: 1号艇の強さ、すぐ内側・外側の艇との勝率・スタート・展示の差
答え（目的変数）: 着順
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "data" / "csv"
KEY = ["date", "venue_code", "race_no"]
CLASS_RANK = {"A1": 4, "A2": 3, "B1": 2, "B2": 1}
WIND_DIRS = ["北", "北北東", "北東", "東北東", "東", "東南東", "南東", "南南東",
             "南", "南南西", "南西", "西南西", "西", "西北西", "北西", "北北西"]

FEATURES = [
    "boat", "venue_code_i", "distance", "class_rank", "age", "weight",
    "nat_win_rate", "nat_2rate", "local_win_rate", "local_2rate", "motor_2rate", "boat_2rate",
    "exhibit_time",
    "nat_win_rate_rank", "nat_win_rate_diff", "local_win_rate_diff", "motor_2rate_rank",
    "motor_2rate_diff", "exhibit_time_rank", "exhibit_time_diff", "class_rank_diff",
    "month",
    # 過去の実績（そのレースより前のレースだけから計算）
    "r_avg_st", "r_avg_finish", "r_win_rate", "r_top3_rate",
    "rb_win_rate", "rb_top3_rate", "rb_n", "r_n",
    "r_avg_st_diff", "rb_win_rate_diff",
    # コース別の実績（実際に入ったコースで集計。枠番どおりのコースに入る前提で付ける）
    "rc_win_rate", "rc_top2_rate", "rc_avg_st", "rc_n", "rc_win_rate_diff", "rc_avg_st_diff",
    # 今節（同じ開催）の成績
    "setsu_n", "setsu_avg_finish", "setsu_top2_rate", "setsu_avg_st",
    # モーターの直近成績（場×モーター番号）
    "m_top2_rate", "m_n", "m_top2_rate_diff",
    # フライング持ち（直近60走のF回数）
    "r_f_count",
    # 気象（レース時）
    "wind_speed", "wave_cm", "wind_dir_i",
    # 艇どうしの関係
    "b1_nat_win_rate", "b1_class_rank", "b1_rc_win_rate", "b1_r_f_count", "b1_exhibit_time_diff",
    "inner_win_rate_gap", "inner_st_gap", "inner_exhibit_gap", "outer_win_rate_gap",
    "inner_max_win_rate", "self_minus_b1_win_rate",
    # 展示のクセ（その選手のいつもの展示と比べて今日はどうか）
    "ex_habit", "ex_vs_habit",
]


def add_exhibit_habit(df: pd.DataFrame) -> pd.DataFrame:
    """展示タイムを出走6艇の平均と比べた差が、その選手の普段（直近30走）と比べてどうか。
    展示で流しがちな選手・いつも速い選手のクセを打ち消すため。"""
    df = df.sort_values(KEY + ["boat"]).reset_index(drop=True)
    df["ex_habit"] = df.groupby("racer_id")["exhibit_time_diff"].transform(
        lambda s: s.shift(1).rolling(30, min_periods=3).mean())
    df["ex_vs_habit"] = df["exhibit_time_diff"] - df["ex_habit"]
    return df


def add_relations(df: pd.DataFrame) -> pd.DataFrame:
    """1号艇の強さや、すぐ内側・外側の艇との差を付け足す。"""
    b1 = df[df.boat == 1][KEY + ["nat_win_rate", "class_rank", "rc_win_rate", "r_f_count",
                                 "exhibit_time_diff"]]
    b1 = b1.rename(columns={c: f"b1_{c}" for c in b1.columns if c not in KEY})
    df = df.merge(b1, on=KEY, how="left")
    df = df.sort_values(KEY + ["boat"]).reset_index(drop=True)
    g = df.groupby(KEY)
    inner_wr = g["nat_win_rate"].shift(1)
    outer_wr = g["nat_win_rate"].shift(-1)
    df["inner_win_rate_gap"] = df["nat_win_rate"] - inner_wr
    df["outer_win_rate_gap"] = df["nat_win_rate"] - outer_wr
    df["inner_st_gap"] = df["rc_avg_st"] - g["rc_avg_st"].shift(1)  # マイナスなら内側より速い
    df["inner_exhibit_gap"] = df["exhibit_time"] - g["exhibit_time"].shift(1)
    df["inner_max_win_rate"] = g["nat_win_rate"].transform(lambda s: s.shift(1).cummax())
    df["self_minus_b1_win_rate"] = df["nat_win_rate"] - df["b1_nat_win_rate"]
    return df


def add_history(df: pd.DataFrame, upcoming: pd.DataFrame | None = None) -> pd.DataFrame:
    """選手ごとの直近の成績を、そのレースより前のデータだけで計算して付け足す。
    r_  : 選手の直近60走（どの枠でも）
    rb_ : 選手がその枠番（艇番）で走った直近30走
    """
    r = pd.read_csv(CSV / "results.csv", dtype={"venue_code": str},
                    usecols=KEY + ["boat", "racer_id", "finish", "start_timing", "course", "motor_no"])
    r["future"] = False
    if upcoming is not None:  # これから走るレース（結果はまだない）
        u = upcoming[KEY + ["boat", "racer_id", "motor_no"]].copy()
        u["future"] = True
        r = pd.concat([r, u], ignore_index=True)
    known = ~r["future"]
    r["is_win"] = (r["finish"] == 1).astype(float).where(known)
    r["is_top3"] = (r["finish"] <= 3).astype(float).where(known)
    r["fin"] = r["finish"].fillna(6).where(known)
    r = r.sort_values(["date", "venue_code", "race_no"]).reset_index(drop=True)
    known = ~r["future"]

    def roll(g, cols, n, prefix):
        out = {}
        for c, name in cols:
            out[f"{prefix}{name}"] = g[c].transform(lambda s: s.shift(1).rolling(n, min_periods=1).mean())
        out[f"{prefix}n"] = g.cumcount().clip(upper=n)
        return pd.DataFrame(out)

    g1 = r.groupby("racer_id")
    h1 = roll(g1, [("start_timing", "avg_st"), ("fin", "avg_finish"),
                   ("is_win", "win_rate"), ("is_top3", "top3_rate")], 60, "r_")
    g2 = r.groupby(["racer_id", "boat"])
    h2 = roll(g2, [("is_win", "win_rate"), ("is_top3", "top3_rate")], 30, "rb_")
    r["is_top2"] = (r["finish"] <= 2).astype(float).where(known)
    r["is_f"] = (r["start_timing"] < 0).astype(float).where(known)
    r["st_pos"] = r["start_timing"].where(r["start_timing"] >= 0)
    r["seq"] = r.groupby(KEY).ngroup()  # レースの時間順の番号

    # フライング回数
    h1["r_f_count"] = g1["is_f"].transform(lambda s: s.shift(1).rolling(60, min_periods=1).sum())

    # 今節：同じ場で、前の出走から3日以内なら同じ開催とみなす
    d = pd.to_datetime(r["date"])
    prev_d = d.groupby(r["racer_id"]).shift(1)
    prev_v = r.groupby("racer_id")["venue_code"].shift(1)
    new_meet = (prev_v != r["venue_code"]) | ((d - prev_d).dt.days > 3) | prev_d.isna()
    r["meet"] = new_meet.groupby(r["racer_id"]).cumsum()
    gm = r.groupby(["racer_id", "meet"])
    h3 = pd.DataFrame({
        "setsu_n": gm.cumcount(),
        "setsu_avg_finish": gm["fin"].transform(lambda s: s.shift(1).expanding().mean()),
        "setsu_top2_rate": gm["is_top2"].transform(lambda s: s.shift(1).expanding().mean()),
        "setsu_avg_st": gm["st_pos"].transform(lambda s: s.shift(1).expanding().mean()),
    })

    # モーター（場×モーター番号の直近30走）
    gmo = r.groupby(["venue_code", "motor_no"])
    h4 = pd.DataFrame({
        "m_top2_rate": gmo["is_top2"].transform(lambda s: s.shift(1).rolling(30, min_periods=1).mean()),
        "m_n": gmo.cumcount().clip(upper=30),
    })
    hist = pd.concat([r[KEY + ["boat"]], h1, h2, h3, h4], axis=1)
    df = df.merge(hist, on=KEY + ["boat"], how="left")

    # コース別：各(選手,コース)の「そのレース後までの」直近30走の成績を作り、
    # 対象レースより前で一番新しいものを、枠番=コースとして引く
    rc = r.dropna(subset=["course"]).copy()
    rc["course"] = rc["course"].astype(int)
    gc = rc.groupby(["racer_id", "course"])
    rc["rc_win_rate"] = gc["is_win"].transform(lambda s: s.rolling(30, min_periods=1).mean())
    rc["rc_top2_rate"] = gc["is_top2"].transform(lambda s: s.rolling(30, min_periods=1).mean())
    rc["rc_avg_st"] = gc["st_pos"].transform(lambda s: s.rolling(30, min_periods=1).mean())
    rc["rc_n"] = (gc.cumcount() + 1).clip(upper=30)
    seq_map = r.drop_duplicates(KEY)[KEY + ["seq"]]
    df = df.merge(seq_map, on=KEY, how="left")
    left = df[["seq", "racer_id", "boat"]].rename(columns={"boat": "course"}).reset_index()
    left["seq_before"] = left["seq"] - 1
    right = rc[["seq", "racer_id", "course", "rc_win_rate", "rc_top2_rate", "rc_avg_st", "rc_n"]]
    m = pd.merge_asof(left.sort_values("seq_before"), right.sort_values("seq"),
                      left_on="seq_before", right_on="seq", by=["racer_id", "course"],
                      direction="backward", suffixes=("", "_r"))
    m = m.set_index("index").sort_index()
    for c in ["rc_win_rate", "rc_top2_rate", "rc_avg_st", "rc_n"]:
        df[c] = m[c]
    df["rc_n"] = df["rc_n"].fillna(0)
    g = df.groupby(KEY)
    df["r_avg_st_diff"] = df["r_avg_st"] - g["r_avg_st"].transform("mean")
    df["rb_win_rate_diff"] = df["rb_win_rate"] - g["rb_win_rate"].transform("mean")
    df["rc_win_rate_diff"] = df["rc_win_rate"] - g["rc_win_rate"].transform("mean")
    df["rc_avg_st_diff"] = df["rc_avg_st"] - g["rc_avg_st"].transform("mean")
    df["m_top2_rate_diff"] = df["m_top2_rate"] - g["m_top2_rate"].transform("mean")
    return df


WEATHER = ["wind_speed", "wave_cm", "wind_dir_i"]


def build(upcoming: pd.DataFrame | None = None) -> pd.DataFrame:
    """upcoming: これから走るレースの出走表（programs.csvと同じ列）に、
    exhibit_time（展示タイム）と wind_speed, wave_cm, wind_dir_i（直前の気象）を足した表。
    渡すと、そのレースの特徴量も一緒に作る（予想用）。"""
    p = pd.read_csv(CSV / "programs.csv", dtype={"venue_code": str})
    r = pd.read_csv(CSV / "results.csv", dtype={"venue_code": str},
                    usecols=KEY + ["boat", "finish", "exhibit_time"])
    df = p.merge(r, on=KEY + ["boat"], how="inner")  # 欠場した艇・中止レースは除く
    if upcoming is not None:
        up = upcoming.drop(columns=[c for c in WEATHER if c in upcoming.columns]).copy()
        up["finish"] = float("nan")
        df = pd.concat([df, up[df.columns.intersection(up.columns)]], ignore_index=True)
    # 6艇そろっているレースだけ使う
    n = df.groupby(KEY)["boat"].transform("size")
    df = df[n == 6].copy()

    df["class_rank"] = df["class"].map(CLASS_RANK)
    df["venue_code_i"] = df["venue_code"].astype(int)
    df["month"] = pd.to_datetime(df["date"]).dt.month
    g = df.groupby(KEY)
    for col, asc in [("nat_win_rate", False), ("motor_2rate", False), ("exhibit_time", True)]:
        df[f"{col}_rank"] = g[col].rank(ascending=asc, method="min")
    for col in ["nat_win_rate", "local_win_rate", "motor_2rate", "exhibit_time", "class_rank"]:
        df[f"{col}_diff"] = df[col] - g[col].transform("mean")
    df = add_history(df, upcoming)
    w = pd.read_csv(CSV / "races.csv", dtype={"venue_code": str},
                    usecols=KEY + ["wind_speed", "wave_cm", "wind_dir"])
    w["wind_dir_i"] = w["wind_dir"].map({d: i for i, d in enumerate(WIND_DIRS)})
    w = w.drop(columns="wind_dir")
    if upcoming is not None:
        w = pd.concat([w, upcoming[KEY + WEATHER].drop_duplicates(KEY)], ignore_index=True)
    df = df.merge(w, on=KEY, how="left")
    df = add_relations(df)
    df = add_exhibit_habit(df)
    df["win"] = (df["finish"] == 1).astype(int)
    return df.sort_values(KEY + ["boat"]).reset_index(drop=True)


if __name__ == "__main__":
    d = build()
    print(d.shape)
    print(d[FEATURES].describe().T[["mean", "min", "max"]])
