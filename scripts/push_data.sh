#!/bin/bash
# 予想・エア投票・ウェブアプリ用データを保存する（GitHub Actions から呼ぶ）。中身は src/records.py の save。
#   bash scripts/push_data.sh         10分ごと: ウェブ用データと記録の金庫の最新を site-data ブランチへ（履歴を残さず上書き）
#   bash scripts/push_data.sh final   回の最後: 上に加えて、金庫を main にも保存する
# 合言葉（SITE_PASSWORD）が無いとき・RECORDS_MODE=legacy のときは、前と同じく元の記録を main にそのまま保存する。
cd "$(dirname "$0")/.."
GD=$(git rev-parse --git-dir)
LOCK="$GD/push_data.lock"
n=0
until mkdir "$LOCK" 2>/dev/null; do   # 10分ごとの保存と最後の保存が重ならないように、順番を待つ
  n=$((n + 1))
  if [ "$n" -ge 120 ]; then
    echo "::error title=記録の保存::前の保存が10分たっても終わらないので、今回の保存はやめます"
    exit 1
  fi
  sleep 5
done
trap 'rmdir "$LOCK" 2>/dev/null' EXIT
python src/build_site.py || true
if ! python src/records.py save "$1"; then
  echo "記録の保存（push）に失敗しました"
  # 最後の保存で失敗したときだけ知らせる（10分ごとの保存は次でまた試す）。
  # 記録を戻せなかった回は「止まりました」の知らせがもう出ているので、重ねて送らない。
  if [ "$1" = "final" ]; then
    [ -f "$GD/records-restore-failed" ] || python src/notify.py "⚠️ 予想の記録をGitHubに保存できませんでした。次の回でもう一度試します。"
    exit 1
  fi
fi
exit 0
