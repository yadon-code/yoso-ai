"""予想や結果をスマホに通知する。

設定: 環境変数 DISCORD_WEBHOOK_URL に、Discordの「ウェブフック」のURLを入れる。
      （.env ファイルに DISCORD_WEBHOOK_URL=... と書いてもよい）
設定がなければ、画面に表示するだけで何も送らない。

使い方（テスト）:
    python src/notify.py "テスト通知です"
"""
import atexit
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _webhook():
    url = os.environ.get("DISCORD_WEBHOOK_URL")
    env = ROOT / ".env"
    if not url and env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.startswith("DISCORD_WEBHOOK_URL="):
                url = line.split("=", 1)[1].strip()
    return url


THREADS = ROOT / "data" / "discord_threads.json"  # フォーラムの部屋の名前 → 部屋のID

# 送れなかった通知の数（この回＝このプログラムが動いている間の合計）。終わるときに GitHub Actions の注釈で出す。
_failures = 0


def failure_count() -> int:
    """この回で送れなかった通知の数（試験・確認用）。"""
    return _failures


def _gh(s) -> str:
    """GitHub Actions の注釈（::warning::）に入れられるように、改行などを置き換える。"""
    return str(s).replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def _report_at_exit():
    if _failures:
        print(f"::warning title=Discord通知の失敗::この回で Discord への送信に {_failures} 件失敗しました"
              "（その通知は Discord に届いていません。送信の失敗で予想は止めていません）", flush=True)


atexit.register(_report_at_exit)


def _post(url, payload):
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json",
                                                          "User-Agent": "yoso-ai"})
    with urllib.request.urlopen(req, timeout=20) as r:
        t = r.read().decode("utf-8") or "{}"
    return json.loads(t)


def send(text: str, room: str = "📢 お知らせ") -> bool:
    """Discordに送る。送れたらTrue。
    通知役（ウェブフック）がフォーラムチャンネルにある場合は、room の名前の部屋（投稿）に送る。
    部屋がなければ作り、IDを data/discord_threads.json に覚えておく。
    普通のチャンネルにある場合は、そのチャンネルにそのまま送る。"""
    global _failures
    url = _webhook()
    if not url:
        print(f"[通知（未設定なので表示だけ）] {room}\n" + text, flush=True)
        return False
    threads = json.loads(THREADS.read_text(encoding="utf-8")) if THREADS.exists() else {}
    ok = True
    err = None
    for i in range(0, len(text), 1900):
        chunk = {"content": text[i:i + 1900]}
        try:
            if room in threads:
                _post(f"{url}?wait=true&thread_id={threads[room]}", chunk)
            else:
                try:  # フォーラムなら新しい部屋を作る
                    res = _post(f"{url}?wait=true", {**chunk, "thread_name": room})
                    tid = res.get("channel_id")
                    if tid:
                        threads[room] = tid
                        THREADS.parent.mkdir(parents=True, exist_ok=True)
                        THREADS.write_text(json.dumps(threads, ensure_ascii=False, indent=1), encoding="utf-8")
                except urllib.error.HTTPError as e:
                    if e.code != 400:
                        raise
                    _post(f"{url}?wait=true", chunk)  # 普通のチャンネル
        except Exception as e:  # 通知の失敗で予想を止めない
            print("通知に失敗:", e, flush=True)
            ok = False
            err = e
    if not ok:  # 1通（長い文の分割ぶんも含めて）につき1件と数え、Actions のログで目立つ注釈にする
        _failures += 1
        print(f"::warning title=Discord通知の失敗::「{_gh(room)}」への送信に失敗（{_gh(err)}）", flush=True)
    return ok


def race_message(venue, race_no, deadline, p_win, picks, tenkai=None):
    """1レース分の買い目通知の文面。picks = [(組番, オッズ, 期待値, 確率), ...]"""
    honmei = max(range(6), key=lambda i: p_win[i]) + 1
    lines = [f"🚤 **{venue} {race_no}R**（締切 {deadline}）",
             "1着確率: " + " / ".join(f"{i + 1}号艇 {p:.0%}" for i, p in enumerate(p_win)),
             f"本命: {honmei}号艇"]
    if tenkai:
        lines.append(f"展開: {tenkai}")
    if picks:
        lines.append("買い目（各100円）:")
        lines += [f"・{c}  {o:.1f}倍  期待値{e:.2f}" for c, o, e, _ in picks]
    return "\n".join(lines)


def grade_message(grade, venue, race_no, race_name, deadline, p_win, p3, picks, tenkai=None):
    """SG・G1のレースの通知。買い目がなくても、1着確率と当たりやすい組を出す。"""
    from backtest import COMBOS
    order = sorted(range(len(p3)), key=lambda i: -p3[i])[:5]
    lines = [f"🏆 **{grade} {venue} {race_no}R {race_name}**（締切 {deadline}）",
             "1着確率: " + " / ".join(f"{i + 1}号艇 {p:.0%}" for i, p in enumerate(p_win)),
             "当たりやすい順: " + " / ".join(f"{COMBOS[i]} {p3[i]:.1%}" for i in order)]
    if tenkai:
        lines.append(f"展開: {tenkai}")
    if picks:
        lines.append("買い目（各100円）:")
        lines += [f"・{c}  {o:.1f}倍  期待値{e:.2f}" for c, o, e, _ in picks]
    else:
        lines.append("買い目: なし（オッズに割安な組がありません）")
    return "\n".join(lines)


if __name__ == "__main__":
    # コマンドとして呼ばれたとき（テスト通知・「止まりました」の知らせなど）は、送れなかったら終了コード1にして
    # Actions の手順が赤くなるようにする。予想の本体（live.py）は send() を直接呼ぶので、ここは影響しない。
    sent = send(" ".join(sys.argv[1:]) or "予想AIからのテスト通知です")
    sys.exit(0 if sent or not _webhook() else 1)
