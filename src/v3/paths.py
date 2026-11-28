"""新しいAI（v3）のデータの置き場。本番（src/live.py など）とは別。

大きなデータは公開リポジトリに入れない。置き場は Mac の ~/Desktop/競艇AI/data_full/（環境変数 YOSO_DATA で変えられる）。
外付けへの写しは scripts/v3_sync.sh。
"""
import os
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent  # src/
REPO = SRC.parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

DATA = Path(os.environ.get("YOSO_DATA", Path.home() / "Desktop" / "競艇AI" / "data_full"))
RAW = DATA / "raw"            # 公式の番組表 B / 競走成績 K（テキスト）
BC = DATA / "bc"              # BOATCAST の生テキスト（1日1ファイル jsonl.gz）
OFFICIAL = DATA / "official"  # 公式サイトのページから取ったもの（BOATCAST が無い 2024-01〜03 など）
KIBETSU = DATA / "kibetsu"    # 期別成績（ファン手帳）
TABLES = DATA / "tables"      # 生データを読みやすい表（parquet）にしたもの
WORK = DATA / "work"          # 検証の途中結果・学習したモデル
LOGS = DATA / "logs"

KEY = ["date", "venue_code", "race_no"]
