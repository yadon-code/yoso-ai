"""今の本番（v2＝第2段階）を、取り直したデータで作り直して、前の検証の数字が再現できるか確かめる。

前の数字（docs/第2段階_全期間_20261109.txt、学習 2025-01〜2026-05、調整 2026-06〜08）:
    [調整用] 1着の対数損失 市場 1.1338 → 補正後 1.1272、3連単 市場 3.6596 → 3.6473
    調整用の回収率（期待値>1.0・100倍まで）3,897点 100.8%
**金庫の期間（2026-09-01〜）は使わない**（ここでは 2026-08-31 までのデータだけ読む）。

やること:
    1. data_full/tables から、本番のコードが読む形（data/csv・data/odds・data/beforeinfo・data/oriten）を data_full/v2root に作る
    2. 本番の src/train.py の oof（第1段階の時期ずらし予想）を、そこで動かす
    3. 本番の src/market_model.py の学習と評価を、2026-08-31 までで動かす
本番のファイル（models/ など）は書き換えない（出力先を v2root に向ける）。

    python src/v3/repro_v2.py prepare   # 1
    python src/v3/repro_v2.py oof       # 2（数十分）
    python src/v3/repro_v2.py eval      # 3
"""
import sys

import numpy as np
import pandas as pd

import paths  # noqa: F401
from paths import DATA, KEY, TABLES

V2 = DATA / "v2root"
LAST = "2026-08-31"
COMBOS = [c[2:] for c in pd.read_parquet(TABLES / "odds3t.parquet").columns if c.startswith("o_")] if (TABLES / "odds3t.parquet").exists() else []
WIND_DIRS = ["北", "北北東", "北東", "東北東", "東", "東南東", "南東", "南南東",
             "南", "南南西", "南西", "西南西", "西", "西北西", "北西", "北北西"]


def prepare():
    csv = V2 / "data" / "csv"
    csv.mkdir(parents=True, exist_ok=True)
    pr = pd.read_parquet(TABLES / "programs.parquet")
    pr = pr[(pr.date >= "2024-01-01") & (pr.date <= LAST)]
    pr.drop(columns=["meet_day", "event", "setsu_marks", "hayami"], errors="ignore").to_csv(csv / "programs.csv", index=False)
    rs = pd.read_parquet(TABLES / "results.parquet")
    rs[(rs.date >= "2024-01-01") & (rs.date <= LAST)].to_csv(csv / "results.csv", index=False)
    rc = pd.read_parquet(TABLES / "races.parquet")
    rc = rc[(rc.date >= "2024-01-01") & (rc.date <= LAST)]
    rc.drop(columns=["kimarite"], errors="ignore").to_csv(csv / "races.csv", index=False)
    # オッズ（1日1ファイル、縦長）
    o = pd.read_parquet(TABLES / "odds3t.parquet")
    o = o[(o.date >= "2025-01-01") & (o.date <= LAST) & (o.src == "bc")]
    od = V2 / "data" / "odds"
    od.mkdir(parents=True, exist_ok=True)
    for d, g in o.groupby("date"):
        long = g.melt(id_vars=KEY, value_vars=["o_" + c for c in COMBOS], var_name="combo", value_name="odds")
        long["combo"] = long["combo"].str[2:]
        long.to_csv(od / f"odds_{d.replace('-', '')}.csv", index=False)
    # 直前情報（BOATCAST 版。前の download_tkz.py と同じく、気象は競走成績の値）
    t = pd.read_parquet(TABLES / "tkz.parquet")
    t = t[(t.date >= "2025-01-01") & (t.date <= LAST)]
    w = rc[KEY + ["weather", "wind_dir", "wind_speed", "wave_cm"]].copy()
    w["wind_dir_bi"] = w["wind_dir"].map({d: i + 1 for i, d in enumerate(WIND_DIRS)}).where(w["wind_dir"] != "無風", 17)
    t = t.merge(w.rename(columns={"weather": "weather_bi", "wind_speed": "wind_speed_bi", "wave_cm": "wave_bi"})
                .drop(columns="wind_dir"), on=KEY, how="left")
    for c in ["propeller_new", "parts_n", "parts", "temp", "water_temp"]:
        t[c] = np.nan
    t["source"] = "bc"
    bd = V2 / "data" / "beforeinfo"
    bd.mkdir(parents=True, exist_ok=True)
    for d, g in t.groupby("date"):
        g.to_csv(bd / f"bi_{d.replace('-', '')}.csv", index=False)
    ori = pd.read_parquet(TABLES / "oriten.parquet")
    ori = ori[(ori.date >= "2025-01-01") & (ori.date <= LAST)]
    rd = V2 / "data" / "oriten"
    rd.mkdir(parents=True, exist_ok=True)
    for d, g in ori.groupby("date"):
        g.to_csv(rd / f"oriten_{d.replace('-', '')}.csv", index=False)
    print("用意しました:", V2, f"オッズ {o.shape[0]} レース・直前 {t.shape[0] // 6}・オリ展 {ori.shape[0] // 6}")


def point_to_v2root():
    import features
    import market_model as mm
    import train
    features.CSV = V2 / "data" / "csv"
    train.ROOT = V2
    mm.ROOT = V2
    return features, train, mm


def oof():
    import os
    _, train, _ = point_to_v2root()
    os.environ.setdefault("OOF_START", "2025-01-01")
    train.oof()


def evaluate():
    import os
    os.environ["TRAIN_END"] = "2026-05-31"
    os.environ["VALID_END"] = "2026-08-31"
    os.environ["START"] = "2025-01-01"
    _, _, mm = point_to_v2root()
    mm.TRAIN_END, mm.VALID_END, mm.START = "2026-05-31", "2026-08-31", "2025-01-01"
    from backtest import logloss
    ow, pm, res, df = mm.load()
    dates = np.asarray(ow.index.get_level_values("date"))
    assert dates.max() <= LAST, "金庫の期間のデータが混ざっている"
    model, p = mm.fit_win(df)
    pw = p.reshape(-1, 6)
    mw = df["mw"].to_numpy().reshape(-1, 6)
    win_col = res["trifecta"].str[0].astype(float).to_numpy() - 1
    ok = ~np.isnan(win_col)

    def wll(m, sel):
        s = sel & ok
        return -np.log(m[s, win_col[s].astype(int)]).mean()

    tr = dates <= mm.TRAIN_END
    va = (dates > mm.TRAIN_END) & (dates <= mm.VALID_END)
    print(f"オッズのあるレース {len(ow)}（{dates.min()}〜{dates.max()}）")
    for name, sel in [("学習", tr), ("調整用", va)]:
        print(f"[{name}] 1着の対数損失 市場 {wll(mw, sel):.4f} → 補正後 {wll(pw, sel):.4f}")
    ratio = pw / mw
    best = None
    for a in [0.5, 1.0, 1.5]:
        for b in [0.0, 0.5, 1.0]:
            for c in [0.0, 0.5]:
                ll = logloss(mm.trifecta_probs(pm[va], ratio[va], a, b, c), res[va])
                if best is None or ll < best[0]:
                    best = (ll, a, b, c)
    _, a, b, c = best
    q = mm.trifecta_probs(pm, ratio, a, b, c)
    print(f"3連単の補正 a={a}, b={b}, c={c}")
    print(f"[調整用] 3連単の対数損失 市場 {logloss(pm[va], res[va]):.4f} → 補正後 {logloss(q[va], res[va]):.4f}")
    odds = ow.to_numpy()
    for th in [1.0, 1.05, 1.1]:
        for mo in [50, 100, 300]:
            n, r, h = mm.roi(q[va], odds[va], res[va], th, mo)
            print(f"  基準 {th} 上限 {mo}: 調整_点数 {n}  調整_回収率 {r:.3f}  的中 {h}")


if __name__ == "__main__":
    {"prepare": prepare, "oof": oof, "eval": evaluate}[sys.argv[1]]()
