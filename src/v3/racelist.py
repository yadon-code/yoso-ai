"""その日にあったレースの一覧を、data_full/raw の競走成績(K)から作る。"""
import paths  # noqa: F401  src/ の本番のモジュールを読めるようにする（先に読む）
import datetime as dt

import parse  # src/parse.py（読み方は本番と同じ）
from paths import RAW


def races_of(day: dt.date):
    """[(場コード, レース番号), ...]。K が無い日は []。"""
    p = RAW / "K" / f"K{day:%y%m%d}.TXT"
    if not p.exists():
        return []
    _, races = parse.parse_results(p)
    return sorted({(r["venue_code"], int(r["race_no"])) for r in races})


def days(start: str, end: str, newest_first=True):
    d0, d1 = dt.date.fromisoformat(start), dt.date.fromisoformat(end)
    out = [d0 + dt.timedelta(days=i) for i in range((d1 - d0).days + 1)]
    return out[::-1] if newest_first else out
