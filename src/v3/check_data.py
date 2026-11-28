"""集めたデータの抜け・失敗を数える（表を作ったあとに動かす）。

    python src/v3/check_data.py
"""
import gzip
import json

import pandas as pd

import paths  # noqa: F401
from paths import BC, KEY, KIBETSU, RAW, TABLES, DATA


def main():
    rc = pd.read_parquet(TABLES / "races.parquet")[KEY].drop_duplicates()
    rc["month"] = rc.date.str[:7]
    o3 = pd.read_parquet(TABLES / "odds3t.parquet")[KEY + ["src"]].drop_duplicates(KEY)
    oo = pd.read_parquet(TABLES / "odds_other.parquet")
    tk = pd.read_parquet(TABLES / "tkz.parquet")[KEY].drop_duplicates()
    ori = pd.read_parquet(TABLES / "oriten.parquet")[KEY].drop_duplicates()
    bio = pd.read_parquet(TABLES / "bi_official.parquet")[KEY].drop_duplicates() if (TABLES / "bi_official.parquet").exists() else pd.DataFrame(columns=KEY)
    t = rc.copy()
    for name, d in [("3連単オッズ", o3[KEY]), ("直前(BC)", tk), ("直前(公式)", bio), ("オリ展", ori),
                    ("単勝オッズ", oo[oo["tan_1"].notna()][KEY] if "tan_1" in oo else oo[KEY].iloc[:0]),
                    ("2連単オッズ", oo[oo["t2_1-2"].notna()][KEY] if "t2_1-2" in oo else oo[KEY].iloc[:0])]:
        t[name] = t.merge(d.assign(x=1), on=KEY, how="left")["x"].fillna(0).to_numpy()
    t["year"] = t.date.str[:4]
    g = t.groupby("year").agg(レース数=("date", "size"), **{c: (c, "sum") for c in ["3連単オッズ", "直前(BC)", "直前(公式)", "オリ展", "単勝オッズ", "2連単オッズ"]})
    print("== 年ごとのレース数と、取れたレース数")
    print(g.astype(int).to_string())
    m = t[t.date >= "2024-04-01"].groupby("month")[["3連単オッズ"]].mean()
    low = m[m["3連単オッズ"] < 0.97]
    print("\n== 2024-04 以降で、3連単オッズが97%未満しか取れていない月:", "なし" if low.empty else "")
    if not low.empty:
        print(low.round(3).to_string())
    # 通信失敗のまま残ったもの
    err = 0
    nobc = []
    for p in sorted(BC.glob("*/bc_*.jsonl.gz")):
        with gzip.open(p, "rt", encoding="utf-8") as f:
            rows = [json.loads(l) for l in f]
        if rows and rows[0].get("no_bc"):
            nobc.append(rows[0]["date"])
            continue
        err += sum(1 for r in rows for k in ("od3", "od2", "od1", "oriten", "tkz") if r.get(k) == "!ERR")
    print(f"\n== BOATCAST: 通信失敗のまま残ったファイル {err}。BOATCAST に無かった日 {len(nobc)}（{min(nobc) if nobc else '-'}〜{max(nobc) if nobc else '-'}）")
    days = pd.date_range("2022-01-01", t.date.max())
    have_k = {p.stem[1:] for p in (RAW / "K").glob("K*.TXT")}
    miss = [d.strftime("%Y-%m-%d") for d in days if d.strftime("%y%m%d") not in have_k]
    print(f"== 公式の競走成績(K)が無い日: {len(miss)} {miss[:10]}")
    print("== 期別成績:", sorted(p.stem for p in KIBETSU.glob("fan*.TXT")))
    gp = TABLES / "grades.csv"
    if gp.exists():
        gr = pd.read_csv(gp)
        print(f"== グレード: {gr.date.nunique()} 日分（{gr.date.min()}〜{gr.date.max()}）", gr.grade.value_counts().to_dict())
    snaps = sorted((DATA / "odds_snap").glob("snap_*.jsonl"))
    for p in snaps:
        rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()]
        ok = sum(r["ok"] for r in rows)
        print(f"== 締切前オッズ {p.stem}: {len(rows)} 回（取れた {ok}）")


if __name__ == "__main__":
    main()
