"""10分前の速報・5分前の本番（src/live.py の main）を、作り物の時計・取得・予想で通して確かめる。

    python tests/test_live_two_stage.py     （pandas・numpy が要る。lightgbm は使わない）

見ること:
- 速報は買い目があるときだけ通知。5分前は、送れた速報と買い目が変わったときだけ「更新」を通知（同じなら通知しない）。
  速報を送っていない（買い目なし・送信失敗）ときは、5分前の買い目をふつうの通知で送る
- エア投票（bets.csv）は5分前の買い目だけ。速報は early_YYYYMMDD.csv、10分前のオッズは pre10_YYYYMMDD.csv
- SG は5分前の回で、買い目がなくても通知
- 部の終わりの先にレースがあれば、10分前の予想に間に合う時刻に次の回へ渡す
- 公式サイトへの取得は1レースあたり 直前情報2回・オッズ2回（展示が出ていれば取り直しなし）
"""
import datetime as dt
import shutil
import sys
import tempfile
import types
from pathlib import Path

sys.modules.setdefault("lightgbm", types.SimpleNamespace(Booster=lambda **k: None))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import live  # noqa: E402
from backtest import COMBOS  # noqa: E402

JST = live.JST
DAY = dt.date(2026, 11, 26)


class Clock:
    t = dt.datetime.combine(DAY, dt.time(9, 0), JST)


class FakeDatetime(dt.datetime):
    @classmethod
    def now(cls, tz=None):
        return Clock.t


fake_dt = types.SimpleNamespace(datetime=FakeDatetime, date=dt.date, time=dt.time, timedelta=dt.timedelta,
                                timezone=dt.timezone)

# (場コード, レース番号, 締切, 速報の買い目, 5分前の買い目)
RACES = [
    ("01", 1, "10:00", ["1-2-3"], ["1-2-3"]),            # 同じ → 更新なし
    ("01", 2, "10:02", ["1-2-3"], ["1-3-2"]),            # 変わった → 更新
    ("01", 3, "10:04", ["1-2-3"], []),                   # なくなった → 更新（取り消し）
    ("01", 4, "10:06", [], ["2-1-3"]),                   # 速報は買い目なしで送っていない → 5分前はふつうの通知
    ("08", 1, "10:08", [], []),                          # SG・買い目なし → SG の通知だけ
    ("01", 6, "10:10", ["1-2-3"], ["1-2-3"]),            # 速報の送信に失敗 → 同じ買い目でも5分前にふつうの通知
    ("01", 5, "12:40", ["1-2-3"], ["1-2-3"]),            # 部の終わり（12:30）の先 → 次の回へ
]


def run(until="12:30", job_start=None):
    tmp = Path(tempfile.mkdtemp())
    sent, calls = [], {"bi": 0, "odds": 0, "ori": 0}
    stage = {}
    prog = pd.DataFrame([{"date": DAY.isoformat(), "venue_code": v, "race_no": n, "boat": b,
                          "venue": "常滑" if v == "08" else "桐生", "deadline": d, "race_name": "予選"}
                         for v, n, d, *_ in RACES for b in range(1, 7)])

    def fetch_bi(date, jo, rno):
        calls["bi"] += 1
        return [{"boat": b, "exhibit_time_bi": 6.7} for b in range(1, 7)]

    def fetch_odds(date, jo, rno):
        calls["odds"] += 1
        return [{"combo": c, "odds": 10.0} for c in COMBOS]

    def predict_race(booster, g, odds_rows):
        v, n = g.venue_code.iloc[0], int(g.race_no.iloc[0])
        k = (v, n)
        stage[k] = stage.get(k, 0) + 1
        race = next(x for x in RACES if x[0] == v and x[1] == n)
        combos = race[3] if stage[k] == 1 else race[4]
        return np.ones(6) / 6, np.ones(120) / 120, [(c, 10.0, 1.3, 0.13) for c in combos]

    def oriten(*a):
        calls["ori"] += 1
        return pd.DataFrame([{"boat": b, "lap": 37.0} for b in range(1, 7)])

    def sleep(sec):
        Clock.t += dt.timedelta(seconds=sec)

    def send(msg, room=None):
        if "桐生 6R" in msg and live.EARLY_NOTE in msg:
            return False  # 送信に失敗したことにする
        sent.append((Clock.t.strftime("%H:%M"), room, msg))
        return True

    patches = dict(
        dt=fake_dt, ROOT=tmp, prepare=lambda day, notify=True: (prog, prog),
        load_market_model=lambda: None, today_grades=lambda day: {"08": "SG"},
        fetch_bi=fetch_bi, fetch_odds=fetch_odds, predict_race=predict_race, fetch_oriten=oriten,
        refresh=lambda g, bi: g, send=send, summary=lambda d: "まとめ", notify_no_exhibit_skips=lambda d: None,
        save_live_inputs=lambda *a, **k: None,
        tenkai=types.SimpleNamespace(predict_race=lambda *a: None, text=lambda *a: "", encode=lambda x: ""),
        time=types.SimpleNamespace(sleep=sleep),
    )
    old = {k: getattr(live, k) for k in patches}
    for k, v in patches.items():
        setattr(live, k, v)
    Clock.t = dt.datetime.combine(DAY, dt.time(9, 0), JST)
    handoff = tmp / "handoff"
    argv = ["live.py", DAY.isoformat(), "--handoff-file", str(handoff)]
    if until:
        argv += ["--until", until]
    if job_start:
        argv += ["--job-start", str(int(job_start.timestamp()))]
    sys.argv = argv
    try:
        live.main()
    finally:
        for k, v in old.items():
            setattr(live, k, v)
    out = dict(tmp=tmp, sent=sent, calls=calls, handoff=handoff.exists(), end=Clock.t)
    return out


def read(p):
    return pd.read_csv(p, dtype=str, keep_default_na=False) if p.exists() else pd.DataFrame()


def test_two_stage():
    o = run(job_start=dt.datetime.combine(DAY, dt.time(9, 0), JST))
    tmp, sent = o["tmp"], o["sent"]
    race_msgs = [(t, m) for t, room, m in sent if room and room.startswith("🚤")]
    sg_msgs = [(t, m) for t, room, m in sent if room == live.GRADE_ROOM]
    for t, m in race_msgs + sg_msgs:
        print(t, m.splitlines()[0], "/", m.splitlines()[-1])

    def msgs(race_no, venue="桐生"):
        return [(t, m) for t, m in race_msgs if f"{venue} {race_no}R" in m]

    # 1R: 速報（9:50）だけ。5分前は同じなので通知なし
    assert [t for t, _ in msgs(1)] == ["09:50"] and live.EARLY_NOTE in msgs(1)[0][1]
    # 2R: 速報＋更新
    assert [t for t, _ in msgs(2)] == ["09:52", "09:57"] and live.UPDATE_NOTE in msgs(2)[1][1] and "1-3-2" in msgs(2)[1][1]
    # 3R: 速報＋取り消しの更新
    assert [t for t, _ in msgs(3)] == ["09:54", "09:59"] and live.UPDATE_NONE_NOTE in msgs(3)[1][1]
    # 4R: 速報は買い目なしで通知なし → 5分前は「更新」ではなくふつうの買い目の通知
    assert [t for t, _ in msgs(4)] == ["10:01"] and "🔄" not in msgs(4)[0][1] and "⚡" not in msgs(4)[0][1]
    # 6R: 速報の送信に失敗 → 5分前は買い目が同じでも、ふつうの買い目の通知
    assert [t for t, _ in msgs(6)] == ["10:05"] and "🔄" not in msgs(6)[0][1] and "⚡" not in msgs(6)[0][1]
    # SG: 買い目なしでも5分前に1回だけ
    assert [t for t, _ in sg_msgs] == ["10:03"] and not msgs(1, "常滑")
    # 部の終わりの先の 5R は予想しない
    assert not msgs(5)

    bets = read(tmp / "data" / "paper" / "bets.csv")
    assert sorted(zip(bets.race_no, bets.combo)) == [("1", "1-2-3"), ("2", "1-3-2"), ("4", "2-1-3"), ("6", "1-2-3")], bets
    early = read(tmp / "data" / "predictions" / "early_20261126.csv")
    assert list(early.race_no) == ["1", "2", "3", "4", "1", "6"]
    assert list(early.combos) == ["1-2-3", "1-2-3", "1-2-3", "", "", "1-2-3"]
    assert list(early.notified) == ["1", "1", "1", "0", "0", "0"]
    live_rows = read(tmp / "data" / "predictions" / "live_20261126.csv")
    assert len(live_rows) == 6
    pre10 = read(tmp / "data" / "odds_pre" / "pre10_20261126.csv")
    pre5 = read(tmp / "data" / "odds_pre" / "pre_20261126.csv")
    assert len(pre10) == 6 and len(pre5) == 6
    assert set(pre10.seconds_before.astype(int)) == {600} and set(pre5.seconds_before.astype(int)) == {300}
    # 取得回数: 6レース × 2回
    assert o["calls"]["bi"] == 12 and o["calls"]["odds"] == 12, o["calls"]
    # 次の回へ: 5R（12:40 締切）の速報 12:30 の15分前 = 12:15 まで待って渡す
    assert o["handoff"] and o["end"].strftime("%H:%M") == "12:15", o["end"]
    shutil.rmtree(tmp)


def test_restart_keeps_early():
    """回が替わっても、5分前は前の回の速報（early_YYYYMMDD.csv）と比べられる。"""
    o = run(job_start=dt.datetime.combine(DAY, dt.time(9, 0), JST))
    old = live.ROOT
    live.ROOT = o["tmp"]
    try:
        seen, sent = live.load_early(DAY)
    finally:
        live.ROOT = old
    assert seen[("01", 1)] == {"1-2-3"} and seen[("01", 4)] == set() and seen[("08", 1)] == set(), seen
    assert set(sent) == {("01", 1), ("01", 2), ("01", 3)}, sent   # 送れた速報だけ（6R は送信失敗）
    shutil.rmtree(o["tmp"])


def test_night_part_summary():
    """夜の部（部の終わりなし）は最後まで予想して、まとめを送り、次の回は呼ばない。"""
    o = run(until=None)
    assert not o["handoff"]
    assert any(m == "まとめ" for _, _, m in o["sent"])
    assert o["calls"]["bi"] == 14  # 7レース × 2回
    shutil.rmtree(o["tmp"])


if __name__ == "__main__":
    test_two_stage()
    test_restart_keeps_early()
    test_night_part_summary()
    print("OK")
