# 競艇予想AI

3連単の買い目を予想し、オッズと比べて「割安な買い目」だけを選ぶことで回収率100%超えを目指すプロジェクトです。

## 進め方（計画）
1. データを集めて整える ✓
2. **予想モデルを作って、過去のレースで回収率を検証する** ← いまここ（結果は docs/検証結果_ステップ2.md）
3. 毎日自動で予想して、LINEやDiscordに通知する
4. Webアプリとして公開する

## フォルダの中身
```
予想AI/
├── README.md            このファイル
├── requirements.txt     必要なPythonのライブラリ一覧
├── docs/
│   ├── データ元の比較.md     どこからどんなデータが取れるか
│   └── 検証結果_ステップ2.md  予想モデルの回収率の検証結果
├── src/
│   ├── download.py      公式サイトから番組表・結果をダウンロード
│   ├── parse.py         ダウンロードしたデータをCSV（表）に変換
│   ├── download_odds.py 3連単の締切オッズを集める
│   ├── features.py      AIに渡す情報（特徴量）を作る
│   ├── train.py         1着確率のAIモデルを学習する
│   └── backtest.py      過去のレースで回収率を検証する
├── models/
│   └── win_model.txt    学習済みのAIモデル
└── data/
    └── csv/
        ├── programs.csv 出走表：レース前にわかる情報（1行=1艇）
        ├── results.csv  結果：着順・進入コース・展示タイム・ST（1行=1艇）
        ├── races.csv    レース情報：天候・風・波・3連単の結果と払戻金（1行=1レース）
        └── odds_3t.csv  3連単120通りの締切オッズ（2026/8/1〜11/7）
```

CSVはExcelやNumbersでそのまま開けます（programs.csvは約96万行あるので、開くのに少し時間がかかります）。

## 今あるデータ
- 期間：2024年1月1日〜2026年11月7日（1,042日分）
- 約16万レース、のべ約96万艇
- 公式の元データ（テキスト）は容量が大きいのでMacには置いていません。`src/download.py` でいつでも取り直せます。

## 自分で動かすとき（今は読み飛ばしてOK）
```
pip3 install -r requirements.txt
python3 src/download.py 2024-01-01 2024-12-31   # 期間を指定してダウンロード
python3 src/parse.py                             # CSVを作り直す
python3 src/train.py                             # AIモデルを学習
python3 src/backtest.py                          # 回収率を検証
```

## 毎日の自動予想（GitHub Actions）

`.github/workflows/live.yml` が毎日3回（日本時間 7:50 / 12:20 / 17:20）動き、
各レースの締切5分前に予想して、買い目があれば Discord に通知します。最終レースの後に1日のまとめも届きます。

- 通知先: リポジトリの Secret `DISCORD_WEBHOOK_URL`
- 予想の記録: `data/predictions/live_YYYYMMDD.csv`、エア投票: `data/paper/bets.csv`（毎回自動で保存）
- 手動テスト: Actions → 「毎日の自動予想」→ Run workflow（mode: test でテスト通知だけ）
- 過去データの初期値: `seed/raw.zip`（公式の番組表・競走成績）
