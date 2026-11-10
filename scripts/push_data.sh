#!/bin/bash
# 予想・エア投票・ウェブアプリ用データをリポジトリに保存する（GitHub Actions から呼ぶ）
cd "$(dirname "$0")/.."
python src/build_site.py || true
git add -f data/predictions data/paper data/site 2>/dev/null
[ -d data/odds_pre ] && git add -f data/odds_pre
[ -f data/discord_threads.json ] && git add -f data/discord_threads.json
if ! git diff --cached --quiet; then
  git commit -q -m "予想記録 $(date +%Y-%m-%d\ %H:%M)"
  ok=0
  for i in 1 2 3; do
    if git pull -q --rebase --autostash && git push -q; then ok=1; break; fi
    git rebase --abort 2>/dev/null   # 途中で止まった取り込みを片付けてからやり直す
    sleep 5
  done
  if [ "$ok" = 0 ]; then
    echo "記録の保存（push）に失敗しました"
    # 最後の保存で失敗したときだけ知らせる（10分ごとの保存は次の回でまた試す）
    [ "$1" = "final" ] && python src/notify.py "⚠️ 予想の記録をGitHubに保存できませんでした。次の回でもう一度試します。"
  fi
fi
