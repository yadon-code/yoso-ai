#!/bin/zsh
# 締切前のオッズの記録（src/v3/snap_odds.py）を毎日動かす。Mac で動かしっぱなし用。
#   nohup caffeinate -i zsh scripts/v3_snap.sh >> ~/Desktop/競艇AI/data_full/logs/snap.log 2>&1 &
# 止まったら（再起動・ふたを閉じた）もう一度動かせば、その日の残りから続ける。
cd "$(dirname "$0")/.."
PY="${PY:-$HOME/Desktop/競艇AI/venv/bin/python}"
while true; do
  echo "== $(date '+%F %T') 開始"
  "$PY" src/v3/snap_odds.py
  zsh scripts/v3_sync.sh
  # 次の日の 7:00 まで待つ
  now=$(date +%s); next=$(date -v+1d -v7H -v0M -v0S +%s)
  sleep $(( next - now ))
done
