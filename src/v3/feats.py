"""新しいAI（v3）の特徴量。1行=1艇。**締切5分前に分かることだけ**を使う。

いつ分かるか（as-of）の決まり:
    前日まで   … 番組表（級別・勝率・モーター…）、過去のレースの成績（このレースより前のレースだけ）、期別成績（算出期間の終わり＋45日から）
    当日・前のレース … 同じ日に先に終わったレースの成績（今節の成績）
    展示のあと … 展示タイム・スタート展示・チルト・体重・オリ展（BOATCAST）
    気象       … 競走成績の値（発走時）で代用している ※本当は締切前の値を使うべき。差は小さいが、本番では直前情報の値を使う
    オッズ     … この表には入れない（market.py で別に足す）

    python src/v3/feats.py     # data_full/tables/feats.parquet を作る
"""
import os
import time

import numpy as np
import pandas as pd

import paths  # noqa: F401
from paths import KEY, TABLES

CLASS_RANK = {"A1": 4, "A2": 3, "B1": 2, "B2": 1}
WIND_DIRS = ["北", "北北東", "北東", "東北東", "東", "東南東", "南東", "南南東",
             "南", "南南西", "南西", "西南西", "西", "西北西", "北西", "北北西"]
POINTS = {1: 10, 2: 8, 3: 6, 4: 4, 5: 2, 6: 1}
KIBETSU_LAG_DAYS = 45


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def race_type_flags(name: pd.Series) -> pd.DataFrame:
    n = name.fillna("")
    return pd.DataFrame({
        "rt_fixed": n.str.contains("進入固定").astype(int),
        "rt_yosen": n.str.contains("予選").astype(int),
        "rt_semifinal": n.str.contains("準優").astype(int),
        "rt_final": n.str.contains("優勝戦").astype(int),
        "rt_special": n.str.contains("ドリーム|選抜|特選|記念|特賞|選").astype(int),
        "rt_ippan": n.str.contains("一般").astype(int),
    })


def base_frame():
    p = pd.read_parquet(TABLES / "programs.parquet")
    r = pd.read_parquet(TABLES / "results.parquet")
    rc = pd.read_parquet(TABLES / "races.parquet")
    p = p.drop_duplicates(KEY + ["boat"])
    r = r.drop_duplicates(KEY + ["boat"])
    until = os.environ.get("YOSO_FEATS_UNTIL")  # リークの点検用: この日より後のデータを丸ごと無いことにする
    if until:
        p, r, rc = p[p.date <= until], r[r.date <= until], rc[rc.date <= until]
    df = p.merge(r[KEY + ["boat", "finish", "finish_raw", "course", "start_timing", "exhibit_time"]],
                 on=KEY + ["boat"], how="left")
    n = df.groupby(KEY)["boat"].transform("size")
    df = df[n == 6].copy()
    w = rc[KEY + ["weather", "wind_dir", "wind_speed", "wave_cm", "kimarite", "trifecta"]]
    df = df.merge(w, on=KEY, how="left")
    return df.sort_values(KEY + ["boat"]).reset_index(drop=True)


def add_order(df):
    """レースの時間順。同じ日のレースは締切時刻の順（無ければレース番号）。"""
    dl = pd.to_datetime(df["date"] + " " + df["deadline"].fillna("00:00"), errors="coerce")
    df["t"] = dl.fillna(pd.to_datetime(df["date"]))
    df["hour"] = dl.dt.hour + dl.dt.minute / 60
    return df


def history(df):
    """選手・モーターの過去の成績。そのレースより前に終わったレースだけを使う（shift(1)）。"""
    h = df[KEY + ["boat", "racer_id", "t", "finish", "finish_raw", "course", "start_timing",
                  "exhibit_time", "motor_no"]].copy()
    h = h.sort_values(["racer_id", "t", "race_no"]).reset_index()
    fin = h["finish"].where(h["finish"].between(1, 6))
    done = h["finish_raw"].notna()  # 結果がある（欠場でない）
    h["fin_s"] = fin.fillna(6).where(done)  # 失格・転覆は6着扱い
    h["win"] = (fin == 1).astype(float).where(done)
    h["top2"] = (fin <= 2).astype(float).where(done)
    h["top3"] = (fin <= 3).astype(float).where(done)
    h["pts"] = fin.map(POINTS).fillna(0).where(done)
    h["st"] = h["start_timing"].where(h["start_timing"] >= 0)
    h["is_f"] = (h["finish_raw"].astype(str).str.startswith("F")).astype(float).where(done)
    h["is_l"] = (h["finish_raw"].astype(str).str.startswith("L")).astype(float).where(done)
    h["c_minus_b"] = (h["course"] - h["boat"]).where(done)
    h["waku"] = (h["course"] == h["boat"]).astype(float).where(done & h["course"].notna())
    h["mae"] = (h["course"] < h["boat"]).astype(float).where(done & h["course"].notna())
    g = h.groupby("racer_id", sort=False)
    out = pd.DataFrame(index=h.index)

    def roll(col, n, how="mean"):
        s = g[col].shift(1)
        rr = s.groupby(h["racer_id"]).rolling(n, min_periods=1)
        v = rr.mean() if how == "mean" else rr.sum()
        return v.reset_index(level=0, drop=True)

    for n in (10, 30, 90):
        out[f"r{n}_fin"] = roll("fin_s", n)
        out[f"r{n}_win"] = roll("win", n)
        out[f"r{n}_top2"] = roll("top2", n)
        out[f"r{n}_top3"] = roll("top3", n)
        out[f"r{n}_st"] = roll("st", n)
    out["r90_f"] = roll("is_f", 90, "sum")
    out["r90_l"] = roll("is_l", 90, "sum")
    out["r30_cmb"] = roll("c_minus_b", 30)
    out["r30_waku"] = roll("waku", 30)
    out["r30_mae"] = roll("mae", 30)
    out["r_n"] = g.cumcount().clip(upper=90)
    prev_t = g["t"].shift(1)
    out["days_since"] = (h["t"] - prev_t).dt.total_seconds() / 86400
    # 今節（同じ場で、前の出走から3日以内なら同じ開催）
    new_meet = (g["venue_code"].shift(1) != h["venue_code"]) | (out["days_since"] > 3) | prev_t.isna()
    meet = new_meet.groupby(h["racer_id"]).cumsum()
    gm = h.groupby([h["racer_id"], meet], sort=False)
    out["s_n"] = gm.cumcount()
    for col, name in [("fin_s", "s_fin"), ("pts", "s_pts"), ("st", "s_st"), ("top2", "s_top2")]:
        out[name] = gm[col].transform(lambda s: s.shift(1).expanding().mean())
    # 当地（その場での直近20走）
    gv = h.groupby(["racer_id", "venue_code"], sort=False)
    out["v20_fin"] = gv["fin_s"].transform(lambda s: s.shift(1).rolling(20, min_periods=1).mean())
    out["v20_win"] = gv["win"].transform(lambda s: s.shift(1).rolling(20, min_periods=1).mean())
    out["v_n"] = gv.cumcount().clip(upper=20)
    out.index = h["index"]
    df = df.join(out)

    # モーター（場×モーター番号）。モーターは毎年入れ替わるので、直近の数字だけ見る
    m = df[["venue_code", "motor_no", "t", "race_no", "boat", "finish", "finish_raw", "exhibit_time"]].copy()
    m = m.sort_values(["venue_code", "motor_no", "t", "race_no"])
    fin = m["finish"].where(m["finish"].between(1, 6))
    done = m["finish_raw"].notna()
    m["top2"] = (fin <= 2).astype(float).where(done)
    m["top3"] = (fin <= 3).astype(float).where(done)
    ex_d = df["exhibit_time"] - df.groupby(KEY)["exhibit_time"].transform("mean")
    m["exd"] = ex_d.loc[m.index]
    gmo = m.groupby(["venue_code", "motor_no"], sort=False)
    for col in ["top2", "top3", "exd"]:
        df.loc[m.index, f"m30_{col}"] = gmo[col].transform(lambda s: s.shift(1).rolling(30, min_periods=3).mean())
    df.loc[m.index, "m_n"] = gmo.cumcount().clip(upper=30)
    df.loc[m.index, "m_gap_days"] = (m["t"] - gmo["t"].shift(1)).dt.days
    return df


def course_history(df):
    """コース別の直近40走（実際に入ったコースで集計）を、「枠番＝コース」として引く。"""
    h = df[["racer_id", "t", "race_no", "course", "finish", "finish_raw", "start_timing"]].dropna(subset=["course"]).copy()
    h["course"] = h["course"].astype(int)
    h = h.sort_values(["racer_id", "course", "t", "race_no"])
    fin = h["finish"].where(h["finish"].between(1, 6))
    h["win"] = (fin == 1).astype(float)
    h["top2"] = (fin <= 2).astype(float)
    h["top3"] = (fin <= 3).astype(float)
    h["st"] = h["start_timing"].where(h["start_timing"] >= 0)
    g = h.groupby(["racer_id", "course"], sort=False)
    for c in ["win", "top2", "top3", "st"]:
        h[f"rc_{c}"] = g[c].transform(lambda s: s.rolling(40, min_periods=1).mean())  # そのレースまで（後で1つ前を引く）
    h["rc_n"] = (g.cumcount() + 1).clip(upper=40)
    right = h[["racer_id", "course", "t", "rc_win", "rc_top2", "rc_top3", "rc_st", "rc_n"]].sort_values("t")
    left = df[["racer_id", "boat", "t"]].rename(columns={"boat": "course"}).reset_index().sort_values("t")
    left["course"] = left["course"].astype(int)
    m = pd.merge_asof(left, right, on="t", by=["racer_id", "course"], direction="backward", allow_exact_matches=False)
    m = m.set_index("index").sort_index()
    for c in ["rc_win", "rc_top2", "rc_top3", "rc_st", "rc_n"]:
        df[c] = m[c]
    df["rc_n"] = df["rc_n"].fillna(0)
    return df


def kibetsu(df):
    """期別成績（公表されたあとのものだけ）。枠番＝コースとしてコース別の数字を引く。"""
    path = TABLES / "kibetsu.parquet"
    if not path.exists():
        return df
    k = pd.read_parquet(path)
    k["racer_id"] = pd.to_numeric(k["racer_id"], errors="coerce")
    k = k.dropna(subset=["racer_id"])
    k["avail"] = pd.to_datetime(k["calc_to"], format="%Y%m%d", errors="coerce") + pd.Timedelta(days=KIBETSU_LAG_DAYS)
    k = k.dropna(subset=["avail"]).sort_values("avail").drop_duplicates(["racer_id", "avail"], keep="last")
    cols = {"ability": "k_ability", "win_rate": "k_win_rate", "rate2": "k_rate2", "avg_st": "k_avg_st",
            "starts": "k_starts", "finals": "k_finals", "champs": "k_champs"}
    sub = k[["racer_id", "avail"] + list(cols)].rename(columns=cols)
    for c in range(1, 7):
        ent = k[f"c{c}_entries"].replace(0, np.nan)
        sub[f"kc{c}_n"] = k[f"c{c}_entries"]
        sub[f"kc{c}_win"] = k[f"c{c}_f1"] / ent
        sub[f"kc{c}_rate2"] = k[f"c{c}_rate2"]
        sub[f"kc{c}_st"] = k[f"c{c}_avg_st"]
        sub[f"kc{c}_strank"] = k[f"c{c}_avg_st_rank"]
        sub[f"kc{c}_f"] = k[f"c{c}_F"]
    sub["k_f_total"] = sum(k[f"c{c}_F"].fillna(0) for c in range(1, 7))
    left = df[["racer_id", "t"]].reset_index()
    left["racer_id"] = pd.to_numeric(left["racer_id"], errors="coerce")
    left = left.sort_values("t")
    m = pd.merge_asof(left, sub.sort_values("avail"), left_on="t", right_on="avail", by="racer_id", direction="backward")
    m = m.set_index("index").sort_index()
    for c in ["k_ability", "k_win_rate", "k_rate2", "k_avg_st", "k_starts", "k_finals", "k_champs", "k_f_total"]:
        df[c] = m[c]
    b = df["boat"].astype(int).to_numpy()
    for name in ["n", "win", "rate2", "st", "strank", "f"]:
        arr = np.column_stack([m[f"kc{c}_{name}"].to_numpy(dtype=float) for c in range(1, 7)])
        df[f"kc_{name}"] = arr[np.arange(len(df)), b - 1]
    return df


def exhibition(df):
    """BOATCAST の直前情報（tkz）とオリ展。無い日は空欄（学習と本番で同じ扱い）。"""
    for name, cols in [("tkz", ["exh_st", "exh_st_f", "exh_course", "tilt", "adjust_weight", "weight_day"]),
                       ("oriten", ["lap", "turn", "straight"])]:
        path = TABLES / f"{name}.parquet"
        if not path.exists():
            for c in cols:
                df[c] = np.nan
            continue
        x = pd.read_parquet(path).drop_duplicates(KEY + ["boat"])
        for c in cols:
            if c not in x:
                x[c] = np.nan
        df = df.merge(x[KEY + ["boat"] + cols], on=KEY + ["boat"], how="left")
    df["weight_gap"] = df["weight_day"] - df["weight"]
    df["exh_course_gap"] = df["exh_course"] - df["boat"]
    return df


def relative(df):
    df = df.copy()
    g = df.groupby(KEY)
    rel = ["nat_win_rate", "local_win_rate", "motor_2rate", "class_rank", "exhibit_time", "r30_fin", "r30_st",
           "rc_win", "rc_st", "kc_win", "kc_st", "k_ability", "m30_top2", "m30_exd", "s_pts", "exh_st",
           "lap", "turn", "straight", "tilt", "weight_gap"]
    for c in rel:
        df[f"{c}_d"] = df[c] - g[c].transform("mean")
        df[f"{c}_rk"] = g[c].rank(ascending=c in ("exhibit_time", "r30_fin", "r30_st", "rc_st", "kc_st", "exh_st",
                                                  "lap", "turn", "straight"), method="min")
    df = df.sort_values(KEY + ["boat"]).reset_index(drop=True)
    g = df.groupby(KEY)
    for c in ["nat_win_rate", "rc_st", "exh_st", "exhibit_time", "kc_win"]:
        df[f"{c}_in"] = df[c] - g[c].shift(1)   # すぐ内側との差
        df[f"{c}_out"] = df[c] - g[c].shift(-1)  # すぐ外側との差
    b1 = df[df.boat == 1][KEY + ["nat_win_rate", "class_rank", "kc_win", "rc_st", "r90_f", "exh_st_d", "exhibit_time_d"]]
    b1 = b1.rename(columns={c: f"b1_{c}" for c in b1.columns if c not in KEY})
    df = df.merge(b1, on=KEY, how="left")
    assert (df.groupby(KEY)["boat"].transform("size") == 6).all()
    a = df["nat_win_rate"].to_numpy(dtype=float).reshape(-1, 6)  # 1レース6行（艇番順）にそろっている
    sh = np.full_like(a, np.nan)
    sh[:, 1:] = a[:, :-1]
    df["inner_max_wr"] = np.fmax.accumulate(sh, axis=1).ravel()  # 内側の艇の中で一番高い勝率
    return df


def race_context(df):
    df["class_rank"] = df["class"].map(CLASS_RANK)
    df["venue_i"] = df["venue_code"].astype(int)
    df["month"] = pd.to_datetime(df["date"]).dt.month
    wd = df["wind_dir"].map({d: i for i, d in enumerate(WIND_DIRS)})
    df["wind_sin"] = np.sin(2 * np.pi * wd / 16)
    df["wind_cos"] = np.cos(2 * np.pi * wd / 16)
    df["wind_dir_i"] = wd
    df["weather_i"] = df["weather"].map({"晴": 0, "曇り": 1, "雨": 2, "雪": 3, "霧": 4})
    df = pd.concat([df, race_type_flags(df["race_name"])], axis=1)
    gp = TABLES / "grades.csv"
    if gp.exists():
        gr = pd.read_csv(gp, dtype={"jcd": str}).rename(columns={"jcd": "venue_code"}).drop_duplicates(["date", "venue_code"])
        df = df.merge(gr, on=["date", "venue_code"], how="left")
        df["grade_i"] = df["grade"].map({"SG": 5, "G1": 4, "G2": 3, "G3": 2, "一般": 1})
    else:
        df["grade_i"] = np.nan
    df["lady"] = df["event"].fillna("").str.contains("レディース|女子|オールレディース|ヴィーナス").astype(int)
    df["rookie"] = df["event"].fillna("").str.contains("ルーキー").astype(int)
    df["sex_f"] = np.nan
    return df


FEATURES = [
    # 番組表
    "boat", "venue_i", "distance", "class_rank", "age", "weight", "nat_win_rate", "nat_2rate", "local_win_rate",
    "local_2rate", "motor_2rate", "boat_2rate",
    # レース
    "month", "hour", "meet_day", "grade_i", "lady", "rookie",
    "rt_fixed", "rt_yosen", "rt_semifinal", "rt_final", "rt_special", "rt_ippan",
    # 気象（発走時の値で代用）
    "wind_speed", "wave_cm", "wind_sin", "wind_cos", "weather_i",
    # 選手の過去
    "r10_fin", "r10_win", "r10_top3", "r10_st", "r30_fin", "r30_win", "r30_top2", "r30_top3", "r30_st",
    "r90_fin", "r90_win", "r90_top3", "r90_st", "r90_f", "r90_l", "r30_cmb", "r30_waku", "r30_mae", "r_n", "days_since",
    "s_n", "s_fin", "s_pts", "s_st", "s_top2", "v20_fin", "v20_win", "v_n",
    "rc_win", "rc_top2", "rc_top3", "rc_st", "rc_n",
    # モーター
    "m30_top2", "m30_top3", "m30_exd", "m_n", "m_gap_days",
    # 期別成績
    "k_ability", "k_win_rate", "k_rate2", "k_avg_st", "k_starts", "k_finals", "k_champs", "k_f_total",
    "kc_n", "kc_win", "kc_rate2", "kc_st", "kc_strank", "kc_f",
    # 展示
    "exhibit_time", "exh_st", "exh_st_f", "exh_course", "exh_course_gap", "tilt", "adjust_weight", "weight_gap",
    "lap", "turn", "straight",
]
REL = [c for c in ["nat_win_rate", "local_win_rate", "motor_2rate", "class_rank", "exhibit_time", "r30_fin", "r30_st",
                   "rc_win", "rc_st", "kc_win", "kc_st", "k_ability", "m30_top2", "m30_exd", "s_pts", "exh_st",
                   "lap", "turn", "straight", "tilt", "weight_gap"]]
FEATURES += [f"{c}_d" for c in REL] + [f"{c}_rk" for c in REL]
FEATURES += [f"{c}_{s}" for c in ["nat_win_rate", "rc_st", "exh_st", "exhibit_time", "kc_win"] for s in ("in", "out")]
FEATURES += ["b1_nat_win_rate", "b1_class_rank", "b1_kc_win", "b1_rc_st", "b1_r90_f", "b1_exh_st_d", "b1_exhibit_time_d",
             "inner_max_wr"]
# 展示がまだ無いとき用（朝の予想）
PRE_EXHIBIT_DROP = {"exhibit_time", "exh_st", "exh_st_f", "exh_course", "exh_course_gap", "tilt", "adjust_weight",
                    "weight_gap", "lap", "turn", "straight"}


def build():
    log("読み込み")
    df = base_frame()
    df = add_order(df)
    df = race_context(df)
    log("選手の過去", len(df))
    df = history(df)
    log("コース別")
    df = course_history(df)
    log("期別成績")
    df = kibetsu(df)
    log("展示")
    df = exhibition(df)
    log("レース内の比較")
    df = relative(df)
    df["fin"] = df["finish"].where(df["finish"].between(1, 6))
    keep = KEY + ["racer_id", "t", "fin", "finish_raw", "course", "trifecta", "kimarite", "grade", "race_name"] + FEATURES
    df = df[[c for c in dict.fromkeys(keep) if c in df.columns]]
    for c in FEATURES:
        if c in df and df[c].dtype == "float64":
            df[c] = df[c].astype("float32")
    out = TABLES / os.environ.get("YOSO_FEATS_FILE", "feats.parquet")
    df.to_parquet(out, index=False)
    log("保存", out, df.shape)
    return df


if __name__ == "__main__":
    build()
