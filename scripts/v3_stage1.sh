#!/bin/zsh
# 段階1の一連の計算（長い。数時間）。途中で失敗したらそこで止まる。結果は data_full/logs/stage1_*.log
#   nohup caffeinate -i zsh scripts/v3_stage1.sh > ~/Desktop/競艇AI/data_full/logs/stage1.log 2>&1 &
{
set -e
cd "$(dirname "$0")/.."
PY="${PY:-$HOME/Desktop/競艇AI/venv/bin/python}"
L="$HOME/Desktop/競艇AI/data_full/logs"
step() { echo "== $(date '+%F %T') $1"; }
step "リーク点検"; "$PY" src/v3/check_leak.py 2026-06-15 > "$L/stage1_leak.log" 2>&1
step "v2 の再現: 用意"; "$PY" src/v3/repro_v2.py prepare > "$L/stage1_v2.log" 2>&1
step "v2 の再現: 第1段階の時期ずらし"; "$PY" src/v3/repro_v2.py oof >> "$L/stage1_v2.log" 2>&1
step "v2 の再現: 評価（2026-08 まで）"; "$PY" src/v3/repro_v2.py eval >> "$L/stage1_v2.log" 2>&1
step "的中AI（オッズなし）1か月ずらし"; "$PY" src/v3/walk.py --name hit_nomkt > "$L/stage1_hit_nomkt.log" 2>&1
step "的中AI（確定オッズあり）1か月ずらし"; "$PY" src/v3/walk.py --name hit_mkt --market > "$L/stage1_hit_mkt.log" 2>&1
step "成績"; "$PY" src/v3/evaluate.py hit_nomkt hit_mkt > "$L/stage1_eval.log" 2>&1
step "回収率の表（確定オッズで買えた場合・帯ごとの割引あり）"
"$PY" src/v3/value.py hit_mkt --adj 1.0 > "$L/stage1_value_adj1.log" 2>&1
"$PY" src/v3/value.py hit_mkt --adj fallback > "$L/stage1_value_fallback.log" 2>&1
zsh scripts/v3_sync.sh
step "おわり"
exit
}
