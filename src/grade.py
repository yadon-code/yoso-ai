"""グレード（SG / G1 / G2 / G3 / 一般）を扱う小さな道具。

- today_grades(day): その日の各場のグレードを公式サイトから取る {場コード: グレード}
- schedule_grades(): 毎日の出走表メモ（data/predictions/schedule_*.csv）に残したグレードを集める
"""
import datetime as dt
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
BIG = ("SG", "G1")  # 「🏆 グレードレース」として特別に扱うグレード
ROOM = "🏆 グレードレース"


def today_grades(day: dt.date) -> dict:
    """失敗しても予想は止めない（空の辞書を返す）。"""
    try:
        from download_grade import fetch_day
        _, rows = fetch_day(day)
        return {jcd: g for _, jcd, g in (rows or [])}
    except Exception as e:
        print("グレードの取得に失敗:", e, flush=True)
        return {}


def schedule_grades() -> pd.DataFrame:
    """columns: date, venue_code, race_no, grade, race_name（グレードを記録し始めた日の分だけ）"""
    out = []
    for f in sorted((ROOT / "data" / "predictions").glob("schedule_*.csv")):
        d = pd.read_csv(f, dtype={"venue_code": str}, keep_default_na=False)
        if "grade" in d:
            if "race_name" not in d:
                d["race_name"] = ""
            out.append(d[["date", "venue_code", "race_no", "grade", "race_name"]])
    if not out:
        return pd.DataFrame(columns=["date", "venue_code", "race_no", "grade", "race_name"])
    return pd.concat(out, ignore_index=True)


def is_big(g) -> bool:
    return isinstance(g, str) and g in BIG
