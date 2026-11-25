"""毎日の自動予想の「この回はいつまで動くか」を決める小さな道具（live.py と live.yml から使う）。

GitHub の1回の実行は timeout-minutes（355分）で強制終了される。強制終了されると「次の回を呼ぶ」が動かず、
次の予約がたまたま来るまで予想が止まってしまう（2026-11-20〜25 に、夜中に始まった回で毎日起きた）。
そこで、どの回も「部の終わり」と「回の開始＋RUN_LIMIT_MIN 分」の早いほうで切り上げ、
レースが残っていれば次の回にバトンを渡す。

重いライブラリを読まないので、テスト（tests/test_live_plan.py）から直接呼べる。
"""
import datetime as dt

JST = dt.timezone(dt.timedelta(hours=9))

# 回の開始からこの分数で必ず切り上げる。timeout-minutes 355 との差（35分）で、
# 最後のレースの取得・記録の保存（push_data.sh final）・キャッシュの保存を済ませる。
RUN_LIMIT_MIN = 320
# 次の回が準備（記録を戻す・出走表を取る）を終えるまでの見込み。
# 次のレースの予想時刻がこれより先なら、上限まで待ってから次を呼ぶ（夜中に回を連打しない）。
HANDOFF_LEAD_MIN = 15

# 部の分け方（始まった時刻で決める。今までの live.yml と同じ）
PARTS = (
    (dt.time(11, 30), dt.time(12, 30)),  # 11:30 より前に始まった回 = 朝の部（12:30 締切まで）
    (dt.time(16, 30), dt.time(17, 30)),  # 16:30 より前に始まった回 = 昼の部（17:30 締切まで）
)


def part_until(now: dt.datetime):
    """始まった時刻から部を決め、その部が受け持つ最後の締切時刻（"HH:MM"）を返す。夜の部は None（最後まで）。"""
    t = now.astimezone(JST).time()
    for start_before, until in PARTS:
        if t < start_before:
            return until.strftime("%H:%M")
    return None


def stop_by(job_start: dt.datetime) -> dt.datetime:
    """この回が必ず切り上げる時刻。"""
    return job_start.astimezone(JST) + dt.timedelta(minutes=RUN_LIMIT_MIN)


def handoff_wake(now: dt.datetime, stop: dt.datetime, next_target: dt.datetime = None) -> dt.datetime:
    """次の回にバトンを渡す前に、いつまで待つか。

    - 次のレースの予想時刻がすぐ（HANDOFF_LEAD_MIN 分以内）なら待たない（すぐ次を呼ぶ）。
    - 次のレースがずっと先（夜中に始まった回など）なら、上限（stop）まで待ってから次を呼ぶ。
      ただし次のレースに間に合うよう、予想時刻の HANDOFF_LEAD_MIN 分前より後には待たない。
    """
    wake = stop
    if next_target is not None:
        wake = min(wake, next_target - dt.timedelta(minutes=HANDOFF_LEAD_MIN))
    return max(now, wake)
