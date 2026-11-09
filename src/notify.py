"""予想や結果をスマホに通知する。

設定: 環境変数 DISCORD_WEBHOOK_URL に、Discordの「ウェブフック」のURLを入れる。
      （.env ファイルに DISCORD_WEBHOOK_URL=... と書いてもよい）
設定がなければ、画面に表示するだけで何も送らない。

使い方（テスト）:
    python src/notify.py "テスト通知です"
"""
import json
import os
import sys
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


def send(text: str) -> bool:
    """Discordに送る。送れたらTrue。2000文字を超える分は分けて送る。"""
    url = _webhook()
    if not url:
        print("[通知（未設定なので表示だけ）]\n" + text, flush=True)
        return False
    ok = True
    for i in range(0, len(text), 1900):
        body = json.dumps({"content": text[i:i + 1900]}).encode("utf-8")
        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json",
                                                              "User-Agent": "yoso-ai"})
        try:
            urllib.request.urlopen(req, timeout=20).read()
        except Exception as e:  # 通知の失敗で予想を止めない
            print("通知に失敗:", e, flush=True)
            ok = False
    return ok


def race_message(venue, race_no, deadline, p_win, picks):
    """1レース分の買い目通知の文面。picks = [(組番, オッズ, 期待値, 確率), ...]"""
    honmei = max(range(6), key=lambda i: p_win[i]) + 1
    lines = [f"🚤 **{venue} {race_no}R**（締切 {deadline}）",
             "1着確率: " + " / ".join(f"{i + 1}号艇 {p:.0%}" for i, p in enumerate(p_win)),
             f"本命: {honmei}号艇"]
    if picks:
        lines.append("買い目（各100円）:")
        lines += [f"・{c}  {o:.1f}倍  期待値{e:.2f}" for c, o, e, _ in picks]
    return "\n".join(lines)


if __name__ == "__main__":
    send(" ".join(sys.argv[1:]) or "予想AIからのテスト通知です")
