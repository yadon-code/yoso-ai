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
    """失敗しても予想は止めない。

    公式ページが取れなかったとき（fetch_day が None・中身が読めず空）は「グレードの取得に失敗」とログに出し、
    同じ日の前の回が残した出走表メモ（data/predictions/schedule_YYYYMMDD.csv）のグレードを使う。
    それも無ければ空の辞書を返す。
    """
    try:
        from download_grade import fetch_day
        _, rows = fetch_day(day)
        if rows:
            return {jcd: g for _, jcd, g in rows}
        print("グレードの取得に失敗:", "公式ページが取れませんでした" if rows is None else "ページからグレードを読めませんでした",
              flush=True)
    except Exception as e:
        print("グレードの取得に失敗:", e, flush=True)
    prev = previous_grades(day)
    print("前の回のグレードを使います:" if prev else "前の回のグレードもありません（グレードなしで続けます）",
          "、".join(f"{k}={v}" for k, v in sorted(prev.items()) if v and v != "一般") if prev else "", flush=True)
    return prev


def previous_grades(day: dt.date) -> dict:
    """同じ日の前の回が残した出走表メモから {場コード: グレード} を返す（無い・読めないときは空）。"""
    f = ROOT / "data" / "predictions" / f"schedule_{day:%Y%m%d}.csv"
    try:
        d = pd.read_csv(f, dtype={"venue_code": str}, keep_default_na=False)
        d = d[d["grade"].astype(str) != ""]
        return dict(zip(d["venue_code"], d["grade"]))
    except Exception:
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
