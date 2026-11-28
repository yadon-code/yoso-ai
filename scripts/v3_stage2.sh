#!/bin/zsh
# 段階2: 的中AI を「市場の確率を出発点にして、ずれだけを学ぶ」作りにして、1か月ずらしで確かめる（2時間ほど）。
#   nohup caffeinate -i zsh scripts/v3_stage2.sh > ~/Desktop/競艇AI/data_full/logs/stage2.log 2>&1 &
{
set -e
cd "$(dirname "$0")/.."
PY="${PY:-$HOME/Desktop/競艇AI/venv/bin/python}"
L="$HOME/Desktop/競艇AI/data_full/logs"
echo "== $(date '+%F %T') 的中AI（市場を出発点）1か月ずらし"
"$PY" src/v3/walk.py --name hit_off --offset > "$L/stage2_hit_off.log" 2>&1
echo "== $(date '+%F %T') 成績"
"$PY" src/v3/evaluate.py hit_off hit_mkt > "$L/stage2_eval.log" 2>&1
echo "== $(date '+%F %T') 3連単の回収率（確定オッズで買えた場合＝上限の目安）"
"$PY" - > "$L/stage2_value_3t.log" 2>&1 <<'PYEOF'
import sys; sys.path.insert(0, 'src/v3')
import pandas as pd
pd.set_option('display.width', 200)
from value import table, monthly
for a, b in [('2025-01-01', '2025-12-31'), ('2026-01-01', '2026-08-31')]:
    t = table('hit_off', a, b, '1.0', bets=['3連単'])
    print(f'\n[3連単 {a}〜{b}]'); print(t.to_string(index=False))
print(monthly('hit_off', '2025-01-01', '2026-08-31', '1.0', '3連単', 1.0, 100).to_string())
PYEOF
zsh scripts/v3_sync.sh
echo "== $(date '+%F %T') おわり"
exit
}
