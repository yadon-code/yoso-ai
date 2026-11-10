#!/bin/bash
# 予想・エア投票・ウェブアプリ用データをリポジトリに保存する（GitHub Actions から呼ぶ）
cd "$(dirname "$0")/.."
python src/build_site.py || true
git add -f data/predictions data/paper data/site 2>/dev/null
[ -d data/odds_pre ] && git add -f data/odds_pre
[ -f data/discord_threads.json ] && git add -f data/discord_threads.json
if ! git diff --cached --quiet; then
  git commit -q -m "予想記録 $(date +%Y-%m-%d\ %H:%M)"
  for i in 1 2 3; do git pull -q --rebase --autostash && git push -q && break; sleep 5; done
fi
