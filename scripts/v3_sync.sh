#!/bin/zsh
# 新しいAI（v3）のデータ data_full を外付けへ写す（消さない写し。外付けにだけある古いファイルは残す）。
SRC="${YOSO_DATA:-$HOME/Desktop/競艇AI/data_full}/"
DST="/Volumes/WD Blue 2TB/AI退避/競艇AI/data_full/"
if [ ! -d "/Volumes/WD Blue 2TB/AI退避" ]; then
  echo "外付けがつながっていないので写しませんでした"; exit 0
fi
mkdir -p "$DST"
rsync -a --exclude '*.tmp' "$SRC" "$DST" && echo "外付けへ写しました: $(date '+%F %T')"
