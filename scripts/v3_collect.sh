#!/bin/zsh
# 新しいAI（v3）のデータを集める（Mac で動かしっぱなし用）。止まっても、もう一度動かせば続きから。
#   公式テキスト・期別成績・グレード（軽い）:
#     nohup caffeinate -i zsh scripts/v3_collect.sh base > ~/Desktop/競艇AI/data_full/logs/collect_base.log 2>&1 &
#   BOATCAST と公式ページ（重い。1日あたり約150レース×5ファイル、全体で1秒4回まで → 約1.5日）:
#     nohup caffeinate -i zsh scripts/v3_collect.sh bc > ~/Desktop/競艇AI/data_full/logs/collect_bc.log 2>&1 &
# 同じ種類を2つ同時に動かさないこと。
# caffeinate -i: 動いている間は Mac が勝手に眠らないようにする（ふたを閉じたときや再起動では止まる → もう一度動かす）。
set -u
cd "$(dirname "$0")/.."
PY="${PY:-$HOME/Desktop/競艇AI/venv/bin/python}"
END="${END:-$(date -v-1d +%F)}"
WHAT="${1:-base}"
echo "== $WHAT 始め $(date '+%F %T')  終わりの日 $END"
if [ "$WHAT" = base ]; then
  "$PY" src/v3/collect_raw.py 2022-01-01 "$END"
  "$PY" src/v3/collect_kibetsu.py
  "$PY" src/v3/collect_grade.py 2022-01-01 "$END"
elif [ "$WHAT" = bc ]; then
  "$PY" src/v3/collect_bc.py 2024-01-01 "$END"
  "$PY" src/v3/collect_official.py 2024-01-01 2024-06-30
fi
zsh scripts/v3_sync.sh
echo "== $WHAT 終わり $(date '+%F %T')"
