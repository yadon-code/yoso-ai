"""自動予想の「この回はいつまで動くか」（src/live_plan.py）の確かめ。

    python tests/test_live_plan.py     （pytest でも動く）

GitHub の1回の上限（timeout-minutes 355分）より前に必ず次の回へ渡すこと、
夜中に始まった回が上限まで待ってから渡すこと、1日をつないだときにレースが抜けないことを見る。
"""
import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import live_plan as lp  # noqa: E402

JST = lp.JST
DAY = dt.date(2026, 11, 26)
TIMEOUT_MIN = 355
MINUTES_BEFORE = 5      # live.py と同じ（締切の5分前に予想）
EARLY_MINUTES_BEFORE = 10  # live.py と同じ（締切の10分前に速報。次の回へ渡す時刻はこちらに合わせる）
SAVE_MIN = 10           # 切り上げたあとの記録の保存・キャッシュの保存に見込む時間（多めに）
STARTUP_MIN = 6         # 次の回が呼ばれてから予想を始められるまで（準備が一番長い日で約5分）


def at(hhmm, day=DAY):
    h, m = map(int, hhmm.split(":"))
    return dt.datetime.combine(day, dt.time(h, m), JST)


def until_dt(now):
    u = lp.part_until(now)
    return at(u, now.date()) if u else None


# 1日のレース（8:30〜21:00 の締切。ナイターまで）。3分おきに並べる。
RACES = []
t = at("08:30")
while t <= at("21:00"):
    RACES.append(t)
    t += dt.timedelta(minutes=3)


def run_once(start):
    """live.py の流れをまねる: 1回の実行で予想するレースと、渡す時刻（無ければ None）と終わる時刻を返す。"""
    until = until_dt(start)
    stop = lp.stop_by(start)
    now = start
    done = []
    for d in RACES:
        if now > d:
            continue  # 締切を過ぎたレースは飛ばす（抜け）
        if (until and d > until) or d > stop:
            wake = lp.handoff_wake(now, stop, d - dt.timedelta(minutes=EARLY_MINUTES_BEFORE))
            return done, wake, wake
        now = max(now, d - dt.timedelta(minutes=MINUTES_BEFORE)) + dt.timedelta(seconds=30)  # 1レース30秒
        done.append(d)
    return done, None, now


def chain(first_start):
    """最初の回の開始時刻から、渡しをつないで1日を回す。"""
    start = first_start
    predicted, runs = [], []
    while start is not None and len(runs) < 20:
        done, handoff, end = run_once(start)
        runs.append((start, end))
        predicted += done
        start = handoff + dt.timedelta(minutes=STARTUP_MIN) if handoff else None
    return predicted, runs


def test_part_until():
    assert lp.part_until(at("00:40")) == "12:30"
    assert lp.part_until(at("04:30")) == "12:30"
    assert lp.part_until(at("07:20")) == "12:30"
    assert lp.part_until(at("11:29")) == "12:30"
    assert lp.part_until(at("11:30")) == "17:30"
    assert lp.part_until(at("12:30")) == "17:30"
    assert lp.part_until(at("16:30")) is None
    assert lp.part_until(at("21:56")) is None


def test_stop_by_before_timeout():
    for s in ("00:40", "04:30", "07:20", "12:30", "16:50", "21:56"):
        start = at(s)
        assert lp.stop_by(start) + dt.timedelta(minutes=SAVE_MIN) < start + dt.timedelta(minutes=TIMEOUT_MIN), s


def test_handoff_wake():
    # 夜中 0:40 開始: 最初のレース（8:30 締切）はずっと先 → 上限 6:00 まで待ってから渡す
    start = at("00:40")
    assert lp.handoff_wake(start, lp.stop_by(start), at("08:25")) == at("06:00")
    # 4:30 開始: 上限は 9:50。8:30 のレースは自分で予想するので、渡すのは 9:50 を超えるレースのとき（待たない）
    start = at("04:30")
    assert lp.stop_by(start) == at("09:50")
    assert lp.handoff_wake(at("09:46"), lp.stop_by(start), at("09:46")) == at("09:46")
    # 朝の部の終わり（12:30）: 次のレースがすぐなので待たない
    assert lp.handoff_wake(at("12:26"), at("12:40"), at("12:27")) == at("12:26")
    # 次のレースが少し先なら、予想時刻の15分前まで待つ（上限よりは後にしない）
    assert lp.handoff_wake(at("12:26"), at("14:00"), at("13:25")) == at("13:10")


def test_chain_no_timeout_no_gap():
    for s in ("00:40", "04:30", "07:20", "12:30"):
        predicted, runs = chain(at(s))
        for a, b in runs:
            assert (b - a) <= dt.timedelta(minutes=TIMEOUT_MIN - SAVE_MIN), (s, a, b)
        missed = [d for d in RACES if d not in predicted and d >= at(s) + dt.timedelta(minutes=STARTUP_MIN)]
        # 渡しの間（次の回の準備の数分）に締切が来るレースだけは抜けうる。1回の渡しにつき最大2レース（3分おき・6分の準備）
        assert len(missed) <= 2 * (len(runs) - 1), (s, [m.strftime("%H:%M") for m in missed])
        print(f"{s} 開始: 回の数 {len(runs)}  "
              + " / ".join(f"{a:%H:%M}→{b:%H:%M}" for a, b in runs)
              + f"  予想 {len(predicted)}/{len(RACES)}  抜け {[m.strftime('%H:%M') for m in missed]}")


if __name__ == "__main__":
    test_part_until()
    test_stop_by_before_timeout()
    test_handoff_wake()
    test_chain_no_timeout_no_gap()
    print("OK")
