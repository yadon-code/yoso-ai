"""週1回の自動学習。`.github/workflows/train.yml` から呼ぶ（手で動かすときは下の「使い方」）。

やること:
  1. いつものキャッシュ（`data/csv` など）からデータを読む。無ければ何もせず
     「データが無いので学習を見送りました」と記録して正常終了する（止まらない）。
  2. 1着モデル（`models/win_model.txt`）を作り直す。
     - 直近 --holdout-days 日（標準30日）は、学習にも木の数を決めるのにも使わず、評価だけに使う。
       （その前の同じ日数を木の数を決める用＝early stopping に使う。`src/train.py` の split() と同じ分け方）
     - 指標は「1着になった艇の1着確率の対数損失（logloss。小さいほど良い）」。AUC も記録するが判定は logloss。
     - 今の `models/win_model.txt` も同じ評価期間で採点し、次の3つを**全部**満たしたときだけ差し替える:
         ・今より低い（良くなっている）
         ・差が偶然のばらつきより大きい（艇ごとの差の平均 − 2×標準誤差 > 0）
         ・差が 0.001 以上（ごく小さな行き来を防ぐ）
       どれか欠けたら「見送り」。今のモデルはそのまま（上書きしない）。
  3. 差し替えたときだけ、`models/win_model.txt` を書き換え、`docs/モデル更新記録.md` に1行足す。
     commit と push は train.yml の次の手順で行う（このスクリプトは git を触らない）。
  4. Discord の「📢 お知らせ」に1行だけ結果を知らせる。--dry のときは送らない。

第2段階（オッズ補正・`market_model.py`）と展開（`tenkai.py`）は、この回では学習しない:
  必要な `data/odds`・`data/beforeinfo`・`data/oriten`・`data/pred/oof_predictions.csv` が
  Actions のキャッシュ（`data/raw`・`data/csv`・`data/cache`）に入っておらず、クラウドでは作れないため。
  決めたときの指標（ここに書いておく）:
    ・オッズ補正: 評価期間で、設定（`market_model.json` の ev_threshold と max_odds）どおりに選んだ
      買い目の回収率（`market_model.roi`、払戻の合計 ÷ 購入額）。今のモデルの回収率より低ければ差し替えない。
      3連単の対数損失も一緒に見る。
    ・展開: 既存の `tenkai.train()` に合わせて、評価期間の7パターンの対数損失（`sklearn` の log_loss）。
      「場ごとの平均」で当てるより悪ければ差し替えない。
  データがキャッシュに入るようになったら、`market_and_tenkai()` の中に足す。

使い方:
    python src/auto_train.py                  # 学習と評価。良ければ models/ と docs/ を書き換えて知らせる
    python src/auto_train.py --dry            # 学習と評価だけ（保存・通知なし）
    python src/auto_train.py --no-data        # データが無かったときの「見送り」の記録と知らせだけ
    python src/auto_train.py --holdout-days 30 --min-gain 0.001
"""
import argparse
import os
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train  # noqa: E402
from features import FEATURES, KEY, build  # noqa: E402
from notify import send  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "models" / "win_model.txt"
DOC = ROOT / "docs" / "モデル更新記録.md"
JST = timezone(timedelta(hours=9))

Z = 2.0            # 差が偶然のばらつきより大きいと言える線（標準誤差の何倍か）
MIN_GAIN = 0.001   # これより小さい改善では差し替えない（logloss の絶対値）
MIN_RACES = 200    # 評価期間にこれだけレースが無いと判定しない

NEEDED_CSV = ("programs.csv", "results.csv", "races.csv")


# ---------- 知らせる・記録する ----------
def _say(out: dict, text: str):
    """Actions の出力（次の手順で使う）と、実行のまとめに書く。"""
    for k, v in out.items():
        print(f"{k}={v}")
    path = os.environ.get("GITHUB_OUTPUT")
    if path:
        with open(path, "a", encoding="utf-8") as f:
            for k, v in out.items():
                f.write(f"{k}={v}\n")
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as f:
            f.write(f"- {text}\n")


def _notify(text: str, dry: bool) -> bool:
    """Discord に1行。「📢 お知らせ」に送る（--dry のときは送らない）。送れなくても止めない。"""
    if dry:
        print(f"[dry なので通知しません] {text}", flush=True)
        return False
    try:
        return send(text)
    except Exception as e:  # 知らせの失敗で学習の結果を消さない
        print("通知に失敗:", e, flush=True)
        return False


# ---------- データ ----------
def missing_inputs() -> list:
    """学習に要る CSV で、無い（または空の）ものの名前。"""
    out = []
    for name in NEEDED_CSV:
        p = ROOT / "data" / "csv" / name
        if not p.exists() or p.stat().st_size == 0:
            out.append(name)
    return out


def _as_proba(a) -> np.ndarray | None:
    """モデルの出力を「4つの確率」(n,4) にそろえる。
    Booster は生のスコアを返すことがあるので、確率に見えないとき（0未満がある・行の合計が1でない）は
    softmax に通す。想定の形でなければ None（＝判定しない）。"""
    a = np.asarray(a, dtype=float)
    if a.ndim != 2 or a.shape[1] != 4:
        return None
    if (a < 0).any() or not np.allclose(a.sum(axis=1), 1.0, atol=1e-3):
        e = np.exp(a - a.max(axis=1, keepdims=True))
        a = e / e.sum(axis=1, keepdims=True)
    return a if np.isfinite(a).all() else None


def score(d: pd.DataFrame, proba: np.ndarray):
    """評価期間の点数。(logloss, AUC, 「1着になった艇」ごとの log p)。小さいほど logloss が良い。"""
    out = train.add_probs(d, proba)
    logs = np.log(out.loc[out.win == 1, "p_win"].clip(lower=1e-12).to_numpy())
    return float(-logs.mean()), train.win_auc(out), logs


# ---------- 判定 ----------
def judge(ll_old: float, ll_new: float, logs_old: np.ndarray, logs_new: np.ndarray, min_gain: float = MIN_GAIN):
    """差し替えてよいか。戻り値: (差し替える?, 人が読む理由)。

    線: 今より logloss が低い ／ 差が偶然のばらつきより大きい（平均 − 2×標準誤差 > 0）／ 差が min_gain 以上。"""
    gain = ll_old - ll_new  # プラスなら良くなった
    diff = logs_new - logs_old  # 艇ごとの改善（プラスなら良くなった）。同じ艇・同じ順番どうしで比べる
    n = len(diff)
    if not (np.isfinite(ll_old) and np.isfinite(ll_new)) or not np.isfinite(diff).all():
        return False, "点数を計算できなかったので判定しない"
    if n < 2:
        return False, "評価できるレースが足りないので判定しない"
    se = float(diff.std(ddof=1) / np.sqrt(n))
    stat = float(diff.mean() - Z * se)
    if gain <= 0:
        return False, f"今のモデルのほうが良い（logloss {ll_old:.4f} → {ll_new:.4f}）"
    if stat <= 0:
        return False, (f"差が偶然の範囲かもしれない（logloss {ll_old:.4f} → {ll_new:.4f}、"
                       f"差 {gain:+.4f}、ばらつきを引くと {stat:+.4f}）")
    if gain < min_gain:
        return False, f"差が小さすぎる（{gain:+.4f} < {min_gain}）"
    return True, f"logloss {ll_old:.4f} → {ll_new:.4f}、差 {gain:+.4f}"


# ---------- 保存 ----------
PERIOD = "%Y-%m-%d"


def save_model(model, path: Path):
    """学習したモデルを保存し、読み直して使えることを確かめる。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(path))
    import lightgbm as lgb
    booster = lgb.Booster(model_file=str(path))
    if booster is None:
        raise RuntimeError(f"保存したモデルを読み直せませんでした: {path}")


def write_doc(row: str):
    """`docs/モデル更新記録.md` に1行足す（無ければ見出しごと作る）。"""
    header = (
        "# モデル更新記録\n\n"
        "週1回の自動学習（`.github/workflows/train.yml`）で、モデルを差し替えた記録です。1行＝1回。\n\n"
        "- 指標: 1着モデルは「評価期間（直近30日）で、1着になった艇の1着確率の対数損失（logloss。小さいほど良い）」"
        "と AUC。判定は logloss で行う。\n"
        "- 差し替えの線: 今のモデルより logloss が低く、差が偶然のばらつきより大きく"
        "（艇ごとの差の平均 − 2×標準誤差 > 0）、差が 0.001 以上。満たさなければ見送り（今のモデルのまま）。\n"
        "- 学習期間は「学習に使った期間」で、木の数を決めるために使った期間も含む。評価期間は学習に使っていない。\n\n"
        "| 日付 (JST) | モデル | 学習に使った期間 | 評価期間 | 評価レース数 | 前 logloss | 後 logloss | 前 AUC | 後 AUC |\n"
        "|---|---|---|---|---|---|---|---|---|\n"
    )
    DOC.parent.mkdir(parents=True, exist_ok=True)
    text = DOC.read_text(encoding="utf-8") if DOC.exists() else header
    if not text.endswith("\n"):
        text += "\n"
    DOC.write_text(text + row + "\n", encoding="utf-8")


def market_and_tenkai():
    """第2段階（オッズ補正）と展開のモデルは、この回では作らない（理由と指標はファイル冒頭に書いた）。"""
    print("第2段階（オッズ補正）・展開の学習: 見送り"
          "（data/odds・data/beforeinfo・data/oriten・data/pred が Actions のキャッシュに無いため）", flush=True)


# ---------- 本体 ----------
def run(dry: bool, holdout_days: int, min_gain: float) -> int:
    today = datetime.now(JST).strftime(PERIOD)
    n = int(holdout_days)

    def finish(result: str, msg: str, with_market: bool = True) -> int:
        print(msg, flush=True)
        _say({"result": result, "replaced": "1" if result == "replaced" else "0", "reason": msg}, msg)
        _notify(f"学習: {msg}", dry)  # dry のときは _notify が送らない
        if with_market:
            market_and_tenkai()
        return 0

    left = missing_inputs()
    if left:
        return finish("no_data", f"データが無いので学習を見送りました（{', '.join(left)} がありません）",
                      with_market=False)

    df = build()
    tr, va, te = train.split(df, n)
    n_race = int(te[KEY].drop_duplicates().shape[0])
    if len(tr) == 0 or len(va) == 0 or n_race < MIN_RACES:
        return finish("skipped", f"学習に使えるデータが足りないので見送りました"
                                 f"（学習 {len(tr)//6} レース・調整 {len(va)//6} レース・評価 {n_race} レース）",
                      with_market=False)

    period = f"{te.date.min()}〜{te.date.max()}"
    used = f"{tr.date.min()}〜{va.date.max()}"
    print(f"学習 {len(tr)//6} レース（{tr.date.min()}〜{tr.date.max()}）／"
          f"調整 {len(va)//6} レース／評価 {n_race} レース（{period}）", flush=True)

    # 今のモデルを、同じ評価期間で採点する（読めないときは判定せず、今のモデルをそのまま残す）
    import lightgbm as lgb
    if not MODEL.exists():
        return finish("skipped", "今のモデル（models/win_model.txt）が無いので判定できないため見送りました"
                                 "（手で確かめてください）")
    try:
        cur = _as_proba(lgb.Booster(model_file=str(MODEL)).predict(te[FEATURES]))
    except Exception as e:
        return finish("skipped", f"今のモデルを読み込めないので見送りました（{type(e).__name__}）"
                                 "（手で確かめてください）")
    if cur is None:
        return finish("skipped", "今のモデルの出力の形が想定と違うので判定できないため見送りました"
                                 "（手で確かめてください）")
    ll_old, auc_old, logs_old = score(te, cur)
    if not np.isfinite(ll_old):  # 評価期間に1着の行が無いときなど。判定しない（今のモデルのまま）
        return finish("skipped", "評価の点数を計算できなかったので見送りました（今のモデルのまま）")

    # 作り直す
    model = train.fit(tr, va)
    cand = _as_proba(model.predict_proba(te[FEATURES]))
    if cand is None:
        return finish("skipped", "新しいモデルの出力の形が想定と違うので見送りました（今のモデルのまま）")
    ll_new, auc_new, logs_new = score(te, cand)
    if not np.isfinite(ll_new):
        return finish("skipped", "新しいモデルの点数を計算できなかったので見送りました（今のモデルのまま）")
    print(f"評価 {period}: logloss 今 {ll_old:.4f} → 新 {ll_new:.4f}／AUC 今 {auc_old:.4f} → 新 {auc_new:.4f}"
          f"（木の数 {getattr(model, 'best_iteration_', None)}）", flush=True)

    ok, why = judge(ll_old, ll_new, logs_old, logs_new, min_gain)
    row = (f"| {today} | 1着 win_model.txt | {used} | {period} | {n_race} | "
           f"{ll_old:.4f} | {ll_new:.4f} | {auc_old:.4f} | {auc_new:.4f} |")

    if not ok:
        return finish("skipped", f"1着モデルは差し替えずに見送りました（{why}）")

    if dry:
        # 保存の手順が通るかだけ確かめる（models/ と docs/ は書き換えない）
        with tempfile.TemporaryDirectory() as td:
            save_model(model, Path(td) / "win_model.txt")
        return finish("would_replace", f"[dry] 差し替えの条件を満たしました（{why}）。保存と commit はしていません")

    save_model(model, MODEL)
    write_doc(row)
    return finish("replaced", f"1着モデルを差し替えました（{why}／評価 {period}）")


def main() -> int:
    ap = argparse.ArgumentParser(description="週1回の自動学習（1着モデル）")
    ap.add_argument("--holdout-days", type=int, default=30, help="直近この日数を評価用に残す（標準30）")
    ap.add_argument("--min-gain", type=float, default=MIN_GAIN, help=f"差し替えるのに要る logloss の改善幅（標準 {MIN_GAIN}）")
    ap.add_argument("--dry", action="store_true", help="学習と評価だけ（保存・commit・通知をしない）")
    ap.add_argument("--no-data", action="store_true", help="データが無かったとき用: 見送りの記録と知らせだけする")
    a = ap.parse_args()
    if a.no_data:
        msg = "データが無いので学習を見送りました（Actions のキャッシュに data/csv がありませんでした）"
        print(msg, flush=True)
        _say({"result": "no_data", "replaced": "0", "reason": msg}, msg)
        _notify(f"学習: {msg}", a.dry)
        return 0
    return run(a.dry, a.holdout_days, a.min_gain)


if __name__ == "__main__":
    sys.exit(main())
