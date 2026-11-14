"""展開予想：展示（直前情報・オリジナル展示）と出走表から、レースがどんな決まり方をしそうかを確率で出す。

7つのパターン:
    1逃げ / 2差し / 2まくり / 3まくり / 3まくり差し / 4コース勝ち / その他（5・6コース、抜き、恵まれ など）
答え（決まり手）は公式の競走成績テキスト（data/raw/K）の見出し行の一番右にある。

使い方:
    python src/tenkai.py train            # 学習して models/tenkai_model.txt を作る（数分）
    DATA=/別の/data python src/tenkai.py train   # データの場所を変えるとき

毎日の予想（live.py）からは predict_race() を呼ぶ。失敗しても予想本体は止めない。
"""
import json
import os
import re
import sys
import unicodedata
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "models" / "tenkai_model.txt"
KEY = ["date", "venue_code", "race_no"]
NAMES = ["1逃げ", "2差し", "2まくり", "3まくり", "3まくり差し", "4コース勝ち", "その他"]
BOAT_COLS = ["boat", "exh_st_eff", "exh_st_d", "exhibit_time_bi_d", "lap_d", "turn_d", "straight_d",
             "nat_win_rate", "local_win_rate", "cls", "motor_2rate", "tilt", "weight_day"]
FEATURES = [f"{c}_c{i}" for c in BOAT_COLS for i in range(1, 7)] + ["vc", "wind_speed_bi", "wind_dir_bi", "wave_bi"]
CLASS = {"A1": 4, "A2": 3, "B1": 2, "B2": 1}


# ---------- 特徴量（学習と本番で同じ関数を使う） ----------
def make_features(b: pd.DataFrame) -> pd.DataFrame:
    """1行=1艇の表（直前情報＋オリ展＋出走表の列）→ 1行=1レースの特徴量。
    艇は「展示の進入コース」の順に並べる（1コースの艇、2コースの艇…）。"""
    b = b.copy()
    for c in ["exh_st", "exh_st_f", "exhibit_time_bi", "lap", "turn", "straight", "nat_win_rate", "local_win_rate",
              "motor_2rate", "tilt", "weight_day", "wind_speed_bi", "wind_dir_bi", "wave_bi", "exh_course"]:
        b[c] = pd.to_numeric(b[c], errors="coerce") if c in b else np.nan
    b["exhibit_time_bi"] = b["exhibit_time_bi"].where(b["exhibit_time_bi"] > 0)
    b["cls"] = b["class"].map(CLASS) if "class" in b else np.nan
    b["exh_st_eff"] = np.where(b["exh_st_f"] == 1, -b["exh_st"], b["exh_st"])  # フライングはマイナス
    g = b.groupby(KEY)
    for c in ["exhibit_time_bi", "lap", "turn", "straight"]:
        b[c + "_d"] = b[c] - g[c].transform("mean")
    b["exh_st_d"] = b["exh_st_eff"] - g["exh_st_eff"].transform("mean")
    # 進入コースが1〜6にそろっていないレースは、艇番どおりの進入とみなす
    ok = g["exh_course"].transform(lambda s: sorted(s.dropna().astype(int).tolist()) == [1, 2, 3, 4, 5, 6])
    b["c"] = np.where(ok, b["exh_course"], b["boat"]).astype(int)
    w = b.pivot_table(index=KEY, columns="c", values=BOAT_COLS, aggfunc="first", dropna=False)
    w.columns = [f"{a}_c{int(c)}" for a, c in w.columns]
    w = w.reset_index()
    race = g[["wind_speed_bi", "wind_dir_bi", "wave_bi"]].first().reset_index()
    w = w.merge(race, on=KEY, how="left")
    w["vc"] = w["venue_code"].astype(int)
    for f in FEATURES:
        if f not in w:
            w[f] = np.nan
    return w


def scenario(kimarite: str, win_course: int) -> int:
    if kimarite == "逃げ":
        return 0
    if win_course == 2 and kimarite in ("差し", "まくり"):
        return 1 if kimarite == "差し" else 2
    if win_course == 3 and kimarite in ("まくり", "まくり差し"):
        return 3 if kimarite == "まくり" else 4
    if win_course == 4:
        return 5
    return 6


# ---------- 本番（live.py から） ----------
_booster = None


def load():
    global _booster
    if _booster is None and MODEL.exists():
        import lightgbm as lgb
        _booster = lgb.Booster(model_file=str(MODEL))
    return _booster


def predict_race(prog_g: pd.DataFrame, bi: pd.DataFrame, ori: pd.DataFrame):
    """1レース分。prog_g=出走表6行、bi=直前情報6行、ori=オリ展（無くてもよい）。
    戻り値: (7パターンの確率, ひとことメモ) / モデルが無ければ None"""
    booster = load()
    if booster is None or len(bi) != 6:
        return None
    b = bi.drop(columns=[c for c in ["date", "venue_code", "race_no"] if c in bi]).copy()
    p = prog_g[["boat", "nat_win_rate", "local_win_rate", "class", "motor_2rate"]].copy()
    b = b.merge(p, on="boat", how="left")
    if ori is not None and len(ori):
        b = b.merge(ori[["boat", "lap", "turn", "straight"]], on="boat", how="left")
    b["date"], b["venue_code"], b["race_no"] = str(prog_g["date"].iloc[0]), str(prog_g["venue_code"].iloc[0]), int(prog_g["race_no"].iloc[0])
    x = make_features(b)
    prob = booster.predict(x[FEATURES])[0]
    return prob / prob.sum(), note(b)


def note(b: pd.DataFrame) -> str:
    """展示で目立った艇のひとこと。"""
    out = []
    for col, label in [("exhibit_time_bi", "展示1番時計"), ("lap", "一周1番"), ("turn", "まわり足1番")]:
        if col in b:
            s = pd.to_numeric(b[col], errors="coerce")
            s = s.where(s > 0)
            if s.notna().sum() >= 4:
                out.append(f"{label} {int(b.loc[s.idxmin(), 'boat'])}号艇")
    return "、".join(out)


def text(prob, memo="", top=3) -> str:
    """'1逃げ 48% / 4コース勝ち 18% / 3まくり 11%（展示1番時計 4号艇）'"""
    order = np.argsort(-np.asarray(prob))[:top]
    s = " / ".join(f"{NAMES[i]} {prob[i]:.0%}" for i in order)
    return s + (f"（{memo}）" if memo else "")


def encode(prob) -> str:
    """記録用: 7パターン全部の確率を空白区切りで（NAMESの順）。"""
    return " ".join(f"{x:.3f}" for x in prob)


# ---------- 学習 ----------
def read_kimarite(raw_k: Path, since="24") -> pd.DataFrame:
    rows = []
    for p in sorted(raw_k.glob("K*.TXT")):
        if p.stem[1:3] < since:
            continue
        day = date(2000 + int(p.stem[1:3]), int(p.stem[3:5]), int(p.stem[5:7])).isoformat()
        code, rno = None, None
        for ln in p.read_bytes().splitlines():
            t = unicodedata.normalize("NFKC", ln.decode("cp932", "replace"))
            m = re.match(r"^(\d\d)KBGN", t)
            if m:
                code = m.group(1)
                continue
            h = re.match(r"^\s+(\d{1,2})R\s+\S", t)
            if h and "風" in t and "[払戻金]" not in t:
                rno = int(h.group(1))
                continue
            if rno and t.strip().startswith("着 艇") and "レースタイム" in t:
                k = t.split("レースタイム")[-1].strip()
                if k:
                    rows.append((day, code, rno, k))
                rno = None
    return pd.DataFrame(rows, columns=KEY + ["kimarite"])


def build_training(data: Path) -> pd.DataFrame:
    import glob

    def rd(pat):
        fs = sorted(glob.glob(str(data / pat)))
        return pd.concat([pd.read_csv(f, dtype={"venue_code": str}) for f in fs], ignore_index=True)

    bi = rd("beforeinfo/bi_2*.csv")
    ori = rd("oriten/oriten_*.csv")
    start = bi["date"].min()
    prog = pd.read_csv(data / "csv" / "programs.csv", dtype={"venue_code": str}, encoding="utf-8-sig",
                       usecols=KEY + ["boat", "nat_win_rate", "local_win_rate", "class", "motor_2rate"])
    res = pd.read_csv(data / "csv" / "results.csv", dtype={"venue_code": str}, encoding="utf-8-sig",
                      usecols=KEY + ["boat", "finish", "course"])
    prog, res = prog[prog.date >= start], res[res.date >= start]
    for d in (bi, ori, prog, res):
        d["venue_code"] = d["venue_code"].str.zfill(2)
    b = bi.merge(prog, on=KEY + ["boat"], how="left")
    b = b.merge(ori[KEY + ["boat", "lap", "turn", "straight"]], on=KEY + ["boat"], how="left")
    x = make_features(b)
    win = res[res.finish == 1][KEY + ["course"]].drop_duplicates(KEY).rename(columns={"course": "win_course"})
    kim = read_kimarite(data / "raw" / "K", since=start[2:4])
    x = x.merge(win, on=KEY).merge(kim, on=KEY)
    x["y"] = [scenario(k, c) for k, c in zip(x.kimarite, x.win_course)]
    return x


def train():
    import lightgbm as lgb
    from sklearn.metrics import log_loss
    data = Path(os.environ.get("DATA", ROOT / "data"))
    df = build_training(data)
    print(f"学習データ: {len(df)} レース（{df.date.min()}〜{df.date.max()}）", flush=True)
    params = dict(objective="multiclass", num_class=7, learning_rate=0.03, num_leaves=31, min_data_in_leaf=100,
                  feature_fraction=0.8, verbose=-1, seed=1)
    # 時期で分けて確認（最後の約6か月をテスト）
    cut = (pd.Timestamp(df.date.max()) - pd.DateOffset(months=6)).date().isoformat()
    vcut = (pd.Timestamp(cut) - pd.DateOffset(months=3)).date().isoformat()
    tr, va, te = df[df.date < vcut], df[(df.date >= vcut) & (df.date < cut)], df[df.date >= cut]
    m = lgb.train(params, lgb.Dataset(tr[FEATURES], tr.y, categorical_feature=["vc"]), 3000,
                  valid_sets=[lgb.Dataset(va[FEATURES], va.y)], callbacks=[lgb.early_stopping(100, verbose=False)])
    n = m.best_iteration
    m = lgb.train(params, lgb.Dataset(df[df.date < cut][FEATURES], df[df.date < cut].y, categorical_feature=["vc"]), n)
    p = m.predict(te[FEATURES])
    base = df[df.date < cut].groupby("vc").y.value_counts(normalize=True).unstack().reindex(columns=range(7)).fillna(1e-4)
    pb = base.loc[te.vc].to_numpy()
    pb = pb / pb.sum(1, keepdims=True)
    report = {"test_from": cut, "test_races": int(len(te)), "rounds": int(n),
              "logloss_venue_avg": round(log_loss(te.y, pb, labels=range(7)), 4),
              "logloss_model": round(log_loss(te.y, p, labels=range(7)), 4)}
    print(report, flush=True)
    # 本番用は全期間で学習し直す
    m = lgb.train(params, lgb.Dataset(df[FEATURES], df.y, categorical_feature=["vc"]), n)
    MODEL.parent.mkdir(parents=True, exist_ok=True)
    m.save_model(str(MODEL))
    report["trained_until"] = df.date.max()
    (ROOT / "models" / "tenkai_model.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print("保存:", MODEL)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        train()
    else:
        print(__doc__)
