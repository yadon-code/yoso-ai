"""当日のレースを、締切の少し前に1レースずつ自動で予想し、エア投票に記録する（1日動かしっぱなし）。

使い方:
    python src/live.py                 # 今日
    python src/live.py 2026-11-10      # 日付指定
    python src/live.py --once          # 待たずに、今の時点で締切前のレースを全部1回だけ予想
    python src/live.py --auto --job-start <UNIX秒> --handoff-file <パス>
                                       # 自動予想（live.yml）用: 始まった時刻で部を決め、回の開始から
                                       # live_plan.RUN_LIMIT_MIN 分で切り上げる。レースが残ればパスにファイルを作る

流れ:
    朝: 前日までのデータを更新 → 今日の出走表を取る → 過去の成績から作る特徴量を先に計算しておく（数分）
    各レースの締切 EARLY_MINUTES_BEFORE 分前: そのとき取れる展示・オッズで予想し、買い目を「速報」として通知（記録は別ファイル）
    各レースの締切 MINUTES_BEFORE 分前: 展示タイム・気象・最新オッズを取り直し、展示に関わる特徴量だけ計算し直して予想。
                                     エア投票はこの買い目で記録し、買い目が速報と変わったときだけ「更新」を通知
    結果: data/predictions/live_YYYYMMDD.csv（全レースの予想）と data/paper/bets.csv（エア投票）
"""
import argparse
import datetime as dt
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from backtest import COMBOS, TAKEOUT, harville  # noqa: E402
from download_beforeinfo import fetch_race as fetch_bi  # noqa: E402
from download_beforeinfo import FIELDS as BI_FIELDS  # noqa: E402
from download_odds import fetch_race as fetch_odds  # noqa: E402
from features import FEATURES, KEY, build  # noqa: E402
from notify import grade_message, race_message, send  # noqa: E402
from grade import ROOM as GRADE_ROOM, is_big, today_grades  # noqa: E402
import json  # noqa: E402
import market_model as mm  # noqa: E402
import sitecrypt  # noqa: E402
import tenkai  # noqa: E402
from download_bc import BASE as BC_BASE, get as bc_get, parse_oriten  # noqa: E402
from settle_live import summary  # noqa: E402
from predict import ROOT, SETTINGS, bi_wind_index, todays_programs, update_history  # noqa: E402
import live_plan  # noqa: E402

JST = dt.timezone(dt.timedelta(hours=9))
MINUTES_BEFORE = 5
# 速報（買い目の1回目の通知）を出す時刻。締切のこれだけ前。エア投票の記録には使わない
EARLY_MINUTES_BEFORE = 10
EARLY_NOTE = "⚡ 速報（締切10分前の予想。5分前に買い目が変われば「更新」を送ります）"
UPDATE_NOTE = "🔄 更新（締切5分前の予想で、速報から買い目が変わりました。こちらが最終です）"
UPDATE_NONE_NOTE = "🔄 更新（締切5分前の予想で、買い目がなくなりました。速報の買い目は取り消しです）"

# 展示（直前情報）がまだ出ていないときの取り直し。
# 展示は締切の数分前までに少しずつ更新されるので、最初の取得で出ていなければ待って取り直す。
# 20秒にした理由: 公式サイトは1分ごとではなく随時更新されるが、短すぎると同じ内容を何度も取りに行くことになる。
# 2回までにした理由: 1レースあたり最大40秒の追加で、後ろのレースの予想が締切に間に合わなくなるのを避ける。
# 公式サイトへの追加の取得は、展示が出ていないレースだけ・最大2回（1日132レースなら最悪でも +264回/日、平常時は数回）。
EXHIBIT_RETRIES = 2
EXHIBIT_RETRY_WAIT = 20
EXHIBIT_RETRY_MARGIN = 60  # 締切の60秒前までに次の取得を始められないときは、やり直さない
# 締切までこれ以上あるときは、この時点では取り直さない（--once で何時間も前にまとめて予想するときに、
# まだ出ていない展示を待って何度も取りに行かないため）。
EXHIBIT_RETRY_WITHIN_MIN = 15
# 展示が取れずに「見送り」になったレースがこの数以上の日は、1日1回まとめて知らせる。
# ふだんは0〜2レースなので、5レース（1場分の番組に近い）を超えたら取得の不調かもしれないと知らせる。
NO_EXHIBIT_NOTICE_MIN = 5
PRE_EXHIBIT_NOTE = "※展示前の予想（第2段階のオッズ補正なし）"

# 第2段階（v2）の買う基準の割引。5分前のオッズで計算した期待値は、確定オッズで見ると約15%甘い
# （docs/5分前オッズの検証_20261123.md）。そこで「5分前の期待値 × EV_DISCOUNT > ev_threshold（1.0）」の買い目だけ買う
# （＝5分前の期待値が約1.18を超えたもの）。確定オッズを予想する仕組みができるまでのつなぎ。
# 帯ごとの割引（期待値が高いほど大きく下がる）も考えられるが、点数が少なく過学習しやすいので、単純な一律の値にする。
# 記録・通知に出す期待値は今までどおり5分前の値（割引前）のまま。
EV_DISCOUNT = 0.85
# 買い方の版。予想の記録（live_*.csv）とエア投票（bets.csv）の buy_rule 列に入れて、基準を変える前後の成績を分けて比べられるようにする。
# 列が空の行は、この列を足す前（2026-11-25 まで。v2 は「5分前の期待値 > 1.0」で買っていた）。
BUY_RULE = "2026-11-26_v2期待値x0.85"


def prepare(day, notify=True):
    """重い準備は一度だけ。結果をファイルに保存して、途中で止まっても再利用する。"""
    cache = ROOT / "data" / "cache" / f"live_{day:%Y%m%d}.pkl"
    if cache.exists():
        prog, df = pd.read_pickle(cache)
        return prog, df
    update_history(day, notify=notify)  # 取り直せなかった日があれば「お知らせ」に1日1回まで知らせる
    prog = todays_programs(day)
    up = prog.copy()
    up["exhibit_time"] = np.nan
    for c in ["wind_speed", "wave_cm", "wind_dir_i"]:
        up[c] = np.nan
    df = build(upcoming=up)
    df = df[df.date == day.isoformat()].copy()
    cache.parent.mkdir(parents=True, exist_ok=True)
    pd.to_pickle((prog, df), cache)
    return prog, df


def refresh(g, bi):
    """展示タイムと気象が出たら、それに関わる特徴量だけ計算し直す。"""
    g = g.sort_values("boat").copy()
    b = bi.set_index("boat")
    g["exhibit_time"] = g["boat"].map(b["exhibit_time_bi"]).where(lambda s: s > 0)
    w = bi.iloc[0]
    g["wind_speed"] = w["wind_speed_bi"]
    g["wave_cm"] = w["wave_bi"]
    g["wind_dir_i"] = bi_wind_index(g["venue_code"].iloc[0], w["wind_dir_bi"])
    g["exhibit_time_rank"] = g["exhibit_time"].rank(method="min")
    g["exhibit_time_diff"] = g["exhibit_time"] - g["exhibit_time"].mean()
    g["inner_exhibit_gap"] = g["exhibit_time"] - g["exhibit_time"].shift(1)
    g["b1_exhibit_time_diff"] = g["exhibit_time_diff"].iloc[0]
    g["ex_vs_habit"] = g["exhibit_time_diff"] - g["ex_habit"]
    return g


def has_exhibit(bi):
    """6艇ぶんの展示タイムがそろっているか。"""
    return len(bi) == 6 and "exhibit_time_bi" in bi.columns and bool(bi["exhibit_time_bi"].fillna(0).gt(0).all())


def fetch_beforeinfo(date, jo, rno, deadline_dt, retry=True):
    """直前情報を取る。展示がそろっていなければ、締切に余裕がある範囲で数回やり直す。
    展示が無くても第1段階の予想はできるので、ここで止めない（呼び出し側が has_exhibit で判定する）。
    retry=False（10分前の速報）のときは取り直さない（5分前にもう一度取るので、公式サイトへの取得を増やさない）。"""
    bi = pd.DataFrame(fetch_bi(date, jo, rno))
    if has_exhibit(bi) or not retry:
        return bi
    if deadline_dt - dt.datetime.now(JST) > dt.timedelta(minutes=EXHIBIT_RETRY_WITHIN_MIN):
        return bi  # 締切までまだ遠いので、この時点では取り直さない
    for n in range(EXHIBIT_RETRIES):
        now = dt.datetime.now(JST)
        if now + dt.timedelta(seconds=EXHIBIT_RETRY_WAIT + EXHIBIT_RETRY_MARGIN) > deadline_dt:
            break  # 締切に近すぎるので、これ以上待たない
        time.sleep(EXHIBIT_RETRY_WAIT)
        # 取り直しは1回だけ通信する（fetch_race 内の3回のやり直しは初回に任せ、通信障害のときに何度も叩かない）
        bi = pd.DataFrame(fetch_bi(date, jo, rno, tries=1))
        if has_exhibit(bi):
            print(f"{jo} {int(rno)}R 展示を {n + 2}回目の取得で確認", flush=True)
            break
    return bi


def load_market_model():
    """第2段階（オッズのズレ）モデル。なければ None（第1段階だけで予想する）。"""
    path = ROOT / "models" / "market_model.txt"
    if not path.exists():
        return None
    conf = json.loads((ROOT / "models" / "market_model.json").read_text(encoding="utf-8"))
    return lgb.Booster(model_file=str(path)), conf


def fetch_oriten(date, jo, rno):
    t = bc_get(BC_BASE.format(jo=jo, name="bc_oriten", ymd=date.replace("-", ""), rno=int(rno)))
    rows = parse_oriten(t) if t else None
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["boat", "lap", "turn", "straight", "lap_kind"])


def predict_race_v2(mmodel, key, p_ai, bi, ori, odds_rows):
    """第2段階: オッズを出発点に、直前情報・オリジナル展示・第1段階AIでズレを補正して買い目を出す。"""
    booster, conf = mmodel
    o = pd.DataFrame(odds_rows).set_index("combo").reindex(COMBOS)["odds"].to_numpy(dtype=float)
    pm, mw = mm.market_win_probs(o)
    d = pd.DataFrame({"date": key[0], "venue_code": key[1], "race_no": key[2], "boat": range(1, 7),
                      "mw": mw, "p_win": p_ai})
    if len(bi):
        bi = bi.copy()
        # 学習データの風向きは「方角の番号（北=1…、無風=17）」なので、直前情報のアイコン番号を同じ表し方に直す
        bi["wind_dir_bi"] = [17 if pd.notna(x) and int(x) == 17 else bi_wind_index(key[1], x) + 1 for x in bi["wind_dir_bi"]]
        d = d.merge(bi.drop(columns=["date", "venue_code", "race_no"], errors="ignore"), on="boat", how="left")
    if len(ori):
        d = d.merge(ori, on="boat", how="left")
    d = mm.make_features(d)
    pw = mm.predict_win(booster, d)
    q = mm.trifecta_probs(pm[None, :], (pw / mw)[None, :], conf["a"], conf["b"], conf["c"])[0]
    ev = q * o
    picks = [(COMBOS[i], o[i], ev[i], q[i])
             for i in np.where((ev * EV_DISCOUNT > conf["ev_threshold"]) & (o <= conf["max_odds"]))[0]]
    return pw, q, sorted(picks, key=lambda x: -x[2])


def predict_race(booster, g, odds_rows):
    proba = booster.predict(g[FEATURES])
    p = proba[:, 0] if proba.ndim == 2 else proba
    p = p / p.sum()
    p3 = harville(p[None, :], SETTINGS["l2"], SETTINGS["l3"])[0]
    picks = []
    if len(odds_rows) == 120:
        o = pd.DataFrame(odds_rows).set_index("combo").reindex(COMBOS)["odds"].to_numpy()
        if not np.isnan(o).any():
            pm = TAKEOUT / o
            pm = pm / pm.sum()
            bl = np.exp(SETTINGS["alpha"] * np.log(p3 + 1e-12) + (1 - SETTINGS["alpha"]) * np.log(pm))
            bl = bl / bl.sum()
            ev = bl * o
            for i in np.where((ev > SETTINGS["ev_threshold"]) & (o <= SETTINGS["max_odds"]))[0]:
                picks.append((COMBOS[i], o[i], ev[i], bl[i]))
    return p, p3, sorted(picks, key=lambda x: -x[2])


def append_rows(path, rows):
    """CSVに行を足す。前に作ったファイルと列が違うとき（列を増やしたあとなど）は、全体を書き直してそろえる。"""
    new = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        old = pd.read_csv(path, dtype=str, keep_default_na=False, nrows=0)
        if list(old.columns) != list(new.columns):
            old = pd.read_csv(path, dtype=str, keep_default_na=False)
            pd.concat([old, new.astype(str)], ignore_index=True).fillna("").to_csv(path, index=False, encoding="utf-8")
            return
    new.to_csv(path, mode="a", header=not path.exists(), index=False, encoding="utf-8")


def save_odds_snapshot(day, key, deadline_dt, odds_rows, prefix="pre"):
    """予想した時点（締切の数分前）の3連単オッズ120通りを保存する。あとからは取り直せないデータ。
    1行=1レース。odds は backtest.COMBOS の順に空白区切り。
    prefix: 5分前は pre_YYYYMMDD.csv（今までどおり）、10分前の速報は pre10_YYYYMMDD.csv に分けて置く。"""
    if len(odds_rows) != 120:
        return
    now = dt.datetime.now(JST)
    o = pd.DataFrame(odds_rows).set_index("combo").reindex(COMBOS)["odds"]
    row = {"date": key[0], "venue_code": key[1], "race_no": key[2], "fetched_at": now.strftime("%Y-%m-%d %H:%M:%S"),
           "seconds_before": int((deadline_dt - now).total_seconds()),
           "odds": " ".join("" if pd.isna(x) else f"{x:g}" for x in o)}
    append_rows(ROOT / "data" / "odds_pre" / f"{prefix}_{day:%Y%m%d}.csv", [row])


def save_live_inputs(day, key, bi=None, ori=None, p_ai=None):
    """予想のときに実際に見た材料を残す（第2段階を学び直すため）。data/odds_pre/ に日付入りの名前で置くので、
    5分前オッズと同じく暗号化して金庫に入る。あとから取り直すと「予想時点では空だった展示」が埋まってしまうので、見たままを残す。
      bi_YYYYMMDD.csv  直前情報（download_beforeinfo.FIELDS の列＋保存した時刻 fetched_at）。展示前のレースも見たまま
      ori_YYYYMMDD.csv オリジナル展示（江戸川など公開していない場は行なし）
      pai_YYYYMMDD.csv 第1段階AIの1着確率（1〜6号艇、空白区切り）。あとからは作り直せない
    回が途中で止まると次の回で同じレースがもう一度入るので、学び直しでは fetched_at の新しい行を使う"""
    now = dt.datetime.now(JST).strftime("%Y-%m-%d %H:%M:%S")
    base = {"date": key[0], "venue_code": key[1], "race_no": key[2], "fetched_at": now}
    d = ROOT / "data" / "odds_pre"
    if bi is not None and len(bi):
        rows = [{**base, **{c: r.get(c, "") for c in BI_FIELDS[3:]}} for r in bi.to_dict("records")]
        append_rows(d / f"bi_{day:%Y%m%d}.csv", rows)
    if ori is not None and len(ori):
        rows = [{**base, **{c: r.get(c, "") for c in ["boat", "lap", "turn", "straight", "lap_kind"]}}
                for r in ori.to_dict("records")]
        append_rows(d / f"ori_{day:%Y%m%d}.csv", rows)
    if p_ai is not None:
        append_rows(d / f"pai_{day:%Y%m%d}.csv", [{**base, "p_ai": " ".join(f"{x:.5f}" for x in p_ai)}])


def record(day, key, p, p3, picks, has_exhibit, model="v1", grade="", tk=None):
    now = dt.datetime.now(JST).strftime("%Y-%m-%d %H:%M")
    top = np.argsort(-p3)[:5]
    row = {"date": key[0], "venue_code": key[1], "race_no": key[2], "predicted_at": now,
           "exhibit": has_exhibit, "p_win": " ".join(f"{x:.3f}" for x in p),
           "top5": " ".join(f"{COMBOS[i]}:{p3[i]:.3f}" for i in top),
           "picks": " ".join(f"{c}({o:.1f}倍,期待値{e:.2f})" for c, o, e, _ in picks),
           "model": model, "grade": grade, "buy_rule": BUY_RULE,
           "tenkai": tenkai.encode(tk[0]) if tk else "", "tenkai_note": tk[1] if tk else ""}  # 展開予想（7パターンの確率）
    # v2=第2段階（オッズのズレ補正）、v1=第1段階だけ（第2段階が使えなかったとき）
    append_rows(ROOT / "data" / "predictions" / f"live_{day:%Y%m%d}.csv", [row])
    if picks:  # 展示前の予想も、買い目があればエア投票に記録する（pre_exhibit=1 の印で、あとから分けられる）
        led = ROOT / "data" / "paper" / "bets.csv"
        rows = [{"date": key[0], "venue_code": key[1], "race_no": key[2], "combo": c, "odds_at_pick": o,
                 "ev": round(e, 3), "prob": round(b, 5), "picked_at": now, "stake": 100, "model": model,
                 "grade": grade, "pre_exhibit": 0 if has_exhibit else 1, "buy_rule": BUY_RULE}
                for c, o, e, b in picks]
        append_rows(led, rows)
    return row


def picks_text(picks):
    return " ".join(f"{c}({o:.1f}倍,期待値{e:.2f})" for c, o, e, _ in picks)


def early_path(day):
    """10分前の速報の記録（エア投票には入れない。5分前の買い目とあとで比べる用）。"""
    return ROOT / "data" / "predictions" / f"early_{day:%Y%m%d}.csv"


def record_early(day, key, p, picks, has_exhibit, model, notified=False):
    """notified: 速報を Discord に送れたか（買い目なし・通知なし・送信失敗は 0）。5分前の通知を「更新」にするかの判定に使う。"""
    row = {"date": key[0], "venue_code": key[1], "race_no": key[2],
           "predicted_at": dt.datetime.now(JST).strftime("%Y-%m-%d %H:%M:%S"),
           "exhibit": has_exhibit, "model": model, "p_win": " ".join(f"{x:.3f}" for x in p),
           "picks": picks_text(picks), "combos": " ".join(c for c, *_ in picks), "buy_rule": BUY_RULE,
           "notified": int(bool(notified))}
    append_rows(early_path(day), [row])


def load_early(day):
    """前の回までの速報（回が途中で替わっても比べられるように）。
    返り値 (seen, sent):
      seen = {(場コード, レース番号): {組番,...}}  速報の予想をしたレース（買い目なしも含む）
      sent = {(場コード, レース番号): {組番,...}}  そのうち速報を Discord に送れたレース
    notified 列が無い・空の古い記録（この列を足す前の行）は、買い目があれば送れたものとみなす。"""
    f = early_path(day)
    if not f.exists():
        return {}, {}
    d = pd.read_csv(f, dtype={"venue_code": str, "combos": str, "notified": str}, keep_default_na=False)
    if "notified" not in d:
        d["notified"] = ""
    seen, sent = {}, {}
    for v, r, c, n in zip(d.venue_code, d.race_no, d.combos, d.notified):
        k = (v, int(r))
        seen[k] = set(c.split())
        if n == "1" or (n == "" and c):
            sent[k] = set(c.split())
        else:
            sent.pop(k, None)
    return seen, sent


def no_exhibit_counts(day):
    """その日の予想記録から (展示前だったレース数, そのうち買い目がなくて見送った数) を数える。
    朝・昼の部で記録した分も、夜の部でこのファイルを読めば入る。"""
    path = ROOT / "data" / "predictions" / f"live_{day:%Y%m%d}.csv"
    if not path.exists():
        return 0, 0
    d = pd.read_csv(path, dtype={"venue_code": str}, keep_default_na=False)
    if "exhibit" not in d.columns or "picks" not in d.columns:
        return 0, 0
    ex = d["exhibit"].astype(str).str.strip().str.lower().isin(("true", "1"))
    no_picks = d["picks"].astype(str).str.strip().eq("")
    return int((~ex).sum()), int((~ex & no_picks).sum())


def notify_no_exhibit_skips(day):
    """展示が取れずに見送ったレースが多い日は、1日1回だけまとめて知らせる。
    買い目があったレースは「展示前の予想」として通知済みなので、ここでは数えない。"""
    n_all, n_skip = no_exhibit_counts(day)
    if n_skip < NO_EXHIBIT_NOTICE_MIN:
        return n_skip
    flag = ROOT / "data" / "paper" / f"no_exhibit_notice_sent_{day:%Y%m%d}"
    if flag.exists():  # 予備の回が後から動いても、同じ日の知らせは1回だけ
        return n_skip
    text = f"📝 今日は {n_skip} レースが展示前だったため見送りました。"
    if n_all > n_skip:
        text += (f"（ほかに {n_all - n_skip} レースも展示前でしたが、"
                 "買い目があったので「展示前の予想」として通知しました）")
    send(text)
    flag.parent.mkdir(parents=True, exist_ok=True)
    flag.touch()
    return n_skip


def hand_off(a, stop, next_target=None):
    """この回はここまで。必要なら上限まで待ってから、次の回に任せる印（--handoff-file）を作る。"""
    if stop:
        wake = live_plan.handoff_wake(dt.datetime.now(JST), stop, next_target)
        wait = (wake - dt.datetime.now(JST)).total_seconds()
        if wait > 0:
            print(f"次のレースまで時間があるので {wake:%H:%M} まで待ってから次の回に渡します", flush=True)
            time.sleep(wait)
    if a.handoff_file:
        Path(a.handoff_file).touch()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("date", nargs="?")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--no-notify", action="store_true", help="通知を送らない")
    ap.add_argument("--until", help="この時刻(HH:MM)より後が締切のレースは次の回に任せて終わる（GitHubの1回6時間の制限用）")
    ap.add_argument("--auto", action="store_true", help="始まった時刻で朝・昼・夜の部を決める（--until を自動で付ける）")
    ap.add_argument("--job-start", type=int, help="回の開始時刻（UNIX秒）。ここから live_plan.RUN_LIMIT_MIN 分で必ず切り上げる")
    ap.add_argument("--handoff-file", help="レースが残ったまま切り上げたとき、このファイルを作る（live.yml が見て次の回を呼ぶ）")
    a = ap.parse_args()
    if a.auto and not a.until:
        a.until = live_plan.part_until(dt.datetime.now(JST))
    stop = live_plan.stop_by(dt.datetime.fromtimestamp(a.job_start, JST)) if a.job_start else None
    if stop:
        print(f"この回は {a.until or '最後'} までの締切を受け持ち、遅くとも {stop:%m/%d %H:%M} に次の回へ渡します", flush=True)
    day = dt.date.fromisoformat(a.date) if a.date else dt.datetime.now(JST).date()
    print(f"{day} の準備中...", flush=True)
    prog, df = prepare(day, notify=not a.no_notify)
    booster = lgb.Booster(model_file=str(ROOT / "models" / "win_model.txt"))
    mmodel = load_market_model()
    print("第2段階モデル:", "あり" if mmodel else "なし（第1段階だけで予想）", flush=True)
    sched = prog.drop_duplicates(KEY)[KEY + ["venue", "deadline", "race_name"]].copy()
    sched["race_name"] = sched["race_name"].fillna("")
    grades = today_grades(day)
    sched["grade"] = sched["venue_code"].map(grades).fillna("")
    big = sorted({r.venue for r in sched.itertuples() if is_big(r.grade)})
    print("グレードレース:", "、".join(big) if big else "なし", flush=True)
    sched["deadline_dt"] = sched["deadline"].map(
        lambda t: dt.datetime.combine(day, dt.time(*map(int, t.split(":"))), JST) if isinstance(t, str) else None)
    sched = sched.dropna(subset=["deadline_dt"]).sort_values("deadline_dt")
    sp = ROOT / "data" / "predictions" / f"schedule_{day:%Y%m%d}.csv"
    sp.parent.mkdir(parents=True, exist_ok=True)
    sched[KEY + ["venue", "deadline", "grade", "race_name"]].to_csv(sp, index=False)  # ウェブアプリ用の出走表（締切時刻）
    done_path = ROOT / "data" / "predictions" / f"live_{day:%Y%m%d}.csv"
    done = set()
    if done_path.exists():
        d = pd.read_csv(done_path, dtype={"venue_code": str})
        done = set(zip(d.venue_code, d.race_no))
    print(f"対象 {len(sched)} レース（予想済み {len(done)}）", flush=True)

    until = dt.datetime.combine(day, dt.time(*map(int, a.until.split(":"))), JST) if a.until else None
    early_seen, early_sent = load_early(day)   # 速報の予想をしたレース・そのうち速報を送れたレース → 買い目（組番の集まり）
    ori_cache = {}                 # 速報のときに取れたオリジナル展示（展示後は変わらないので5分前に使い回す）
    v2_errors = 0

    def predict_now(r, early):
        """今の時点の直前情報・オッズで予想する。予想できないときは None。"""
        nonlocal v2_errors
        key = (r.date, r.venue_code, r.race_no)
        g = df[(df.venue_code == r.venue_code) & (df.race_no == r.race_no)]
        if len(g) != 6:
            return None
        bi = fetch_beforeinfo(r.date, r.venue_code, int(r.race_no), r.deadline_dt, retry=not early)
        has_ex = has_exhibit(bi)
        if a.once and not has_ex:
            # 手動の1回予想（--once）では、展示前のレースは今までどおり見送る。
            # その日の残りをすべて v1 で埋めてしまい、通知が大量になるのを避けるため。
            return None
        if has_ex:
            g = refresh(g, bi)
        odds_rows = fetch_odds(r.date, r.venue_code, int(r.race_no))
        try:
            save_odds_snapshot(day, key, r.deadline_dt, odds_rows, prefix="pre10" if early else "pre")
        except Exception as e:  # 保存の失敗で予想を止めない
            print("オッズの保存に失敗:", e, flush=True)
        if dt.datetime.now(JST) > r.deadline_dt - dt.timedelta(seconds=10):
            # 取得に時間がかかって締切が目の前に来たレースは、予想・記録・通知をしない（間に合わない買い目を出さない）
            print(f"{r.venue} {r.race_no:>2}R 締切に間に合わないため飛ばします", flush=True)
            return None
        p, p3, picks = predict_race(booster, g, odds_rows)
        model = "v1"
        p_ai = p
        ori = ori_cache.get(key)
        if mmodel and has_ex and len(odds_rows) == 120 and all(x["odds"] for x in odds_rows):
            try:
                if ori is None:
                    ori = fetch_oriten(r.date, r.venue_code, r.race_no)
                p, p3, picks = predict_race_v2(mmodel, key, p, bi, ori, odds_rows)
                model = "v2"
            except Exception as e:  # 第2段階で失敗したら第1段階の結果を使う
                print("第2段階でエラー:", e, flush=True)
                v2_errors += 1
                if v2_errors == 3 and not a.no_notify:  # 続けて起きているときだけ、1回知らせる
                    send(f"⚠️ 第2段階の予想でエラーが続いています（{e}）。第1段階だけで予想を続けます。")
        tk = None
        if has_ex:
            try:  # 展開予想（失敗しても予想本体は止めない）
                if ori is None:
                    ori = fetch_oriten(r.date, r.venue_code, r.race_no)
                tk = tenkai.predict_race(prog[(prog.venue_code == r.venue_code) & (prog.race_no == r.race_no)], bi, ori)
            except Exception as e:
                print("展開予想でエラー:", e, flush=True)
        if ori is not None and len(ori):
            ori_cache[key] = ori
        return dict(key=key, bi=bi, has_ex=has_ex, p=p, p3=p3, picks=picks, model=model, p_ai=p_ai, ori=ori, tk=tk)

    def early_stage(r):
        """締切10分前: 予想して、買い目があれば「速報」として通知する。エア投票には記録しない。"""
        res = predict_now(r, early=True)
        if res is None:
            return
        picks, has_ex = res["picks"], res["has_ex"]
        k = (r.venue_code, int(r.race_no))
        combos = {c for c, *_ in picks}
        print(f"{r.venue} {r.race_no:>2}R 締切{r.deadline} 展示{'あり' if has_ex else 'なし'} "
              f"速報の予想をしました（{res['model']}・買い目 {len(picks)}点）", flush=True)
        notified = False
        if picks and not a.no_notify:
            note = EARLY_NOTE if has_ex else EARLY_NOTE + "\n" + PRE_EXHIBIT_NOTE
            tk_text = tenkai.text(*res["tk"]) if res["tk"] else None
            notified = bool(send(race_message(r.venue, r.race_no, r.deadline, res["p"], picks, tk_text, note=note),
                                 room=f"🚤 {r.venue}"))
        early_seen[k] = combos
        if notified:
            early_sent[k] = combos
        try:
            record_early(day, res["key"], res["p"], picks, has_ex, res["model"], notified)
        except Exception as e:
            print("速報の記録に失敗:", e, flush=True)

    def final_stage(r):
        """締切5分前: 今までどおり予想してエア投票に記録する。速報を出したレースは、買い目が変わったときだけ通知する。"""
        res = predict_now(r, early=False)
        if res is None:
            return
        key, p, p3, picks, has_ex, model, tk = (res[k] for k in ("key", "p", "p3", "picks", "has_ex", "model", "tk"))
        try:  # 学び直し用の材料を残す（失敗しても予想は止めない）
            save_live_inputs(day, key, res["bi"], res["ori"], res["p_ai"])
        except Exception as e:
            print("材料の保存に失敗:", e, flush=True)
        tk_text = tenkai.text(*tk) if tk else None
        row = record(day, key, p, p3, picks, has_ex, model, r.grade, tk)
        # 「更新」と比べる相手は、実際に送れた速報だけ。速報を送っていない（買い目なし・送信失敗など）ときは
        # 5分前の買い目をふつうの通知として送る
        early = early_sent.get((r.venue_code, int(r.race_no)))
        changed = early is not None and early != {c for c, *_ in picks}
        if sitecrypt.password():  # 実行記録は公開なので、合言葉があるときは予想の中身を出さない
            print(f"{r.venue} {r.race_no:>2}R 締切{r.deadline} 展示{'あり' if has_ex else 'なし'} "
                  f"予想しました（{model}・買い目 {len(picks)}点"
                  f"{'' if early is None else '・速報から変更あり' if changed else '・速報と同じ'}）", flush=True)
        else:
            print(f"{r.venue} {r.race_no:>2}R 締切{r.deadline} 展示{'あり' if has_ex else 'なし'} "
                  f"1着確率[{row['p_win']}]  買い目: {row['picks'] or 'なし'}  展開: {tk_text or '-'}", flush=True)
        if not a.no_notify:
            pre = None if has_ex else PRE_EXHIBIT_NOTE
            if early is None:
                # 速報を送っていないレース（速報が買い目なし・送信失敗・回の替わり目・--once など）は今までどおり: 買い目があれば通知
                # 展示ありは今までどおりの文面。展示前は、既存の文面の末尾に「展示前」と分かる1行を足すだけにする。
                if picks:
                    send(race_message(r.venue, r.race_no, r.deadline, p, picks, tk_text, note=pre), room=f"🚤 {r.venue}")
            elif changed:
                note = (UPDATE_NOTE if picks else UPDATE_NONE_NOTE) + ("\n" + pre if pre else "")
                send(race_message(r.venue, r.race_no, r.deadline, p, picks, tk_text, note=note), room=f"🚤 {r.venue}")
        if is_big(r.grade) and has_ex and not a.no_notify:  # SG・G1は買い目がなくても予想を知らせる（5分前の回だけ）
            send(grade_message(r.grade, r.venue, r.race_no, r.race_name, r.deadline, p, p3, picks, tk_text), room=GRADE_ROOM)

    # この回が受け持つレース（部の終わり・回の時間切れまでに締切が来るもの）と、次の回に渡すレースに分ける
    todo = [r for _, r in sched.iterrows() if (r.venue_code, r.race_no) not in done]
    beyond = [(until and r.deadline_dt > until) or (stop and r.deadline_dt > stop) for r in todo]
    mine = [r for r, x in zip(todo, beyond) if not x]
    later = [r for r, x in zip(todo, beyond) if x]
    # 1レースに2回（10分前の速報・5分前の本番）。時刻の順に並べて、間のレースと交互に進める
    events = []
    for r in mine:
        if not a.once and (r.venue_code, int(r.race_no)) not in early_seen:
            events.append((r.deadline_dt - dt.timedelta(minutes=EARLY_MINUTES_BEFORE), r.deadline_dt, 0, r))
        events.append((r.deadline_dt - dt.timedelta(minutes=MINUTES_BEFORE), r.deadline_dt, 1, r))
    events.sort(key=lambda x: x[:3])
    for target, deadline_dt, final, r in events:
        now = dt.datetime.now(JST)
        if now > deadline_dt:
            continue  # 締切を過ぎたレースは飛ばす
        if not final and now > deadline_dt - dt.timedelta(minutes=MINUTES_BEFORE):
            continue  # 速報の時刻を過ぎて5分前の予想の時刻になっていたら、速報は出さない
        if not a.once and now < target:
            time.sleep((target - now).total_seconds())
        (final_stage if final else early_stage)(r)
    if later:
        r = min(later, key=lambda x: x.deadline_dt)
        if until and r.deadline_dt > until:
            print(f"{a.until} 以降のレースは次の回で予想します", flush=True)
        else:
            # GitHub の制限（1回6時間）で強制終了される前に、次の回に渡す
            print(f"{stop:%H:%M} でこの回の時間切れになるので、締切 {r.deadline} 以降のレースは次の回で予想します", flush=True)
        hand_off(a, stop, r.deadline_dt - dt.timedelta(minutes=EARLY_MINUTES_BEFORE))
        return

    # 最終レースの結果が出るころ（締切から20分後）に、1日のまとめを通知する
    if not a.once and not a.no_notify and len(sched):
        end = sched["deadline_dt"].max() + dt.timedelta(minutes=20)
        now = dt.datetime.now(JST)
        if stop and end > stop and now < end:
            print("1日のまとめは次の回で送ります（この回の時間切れが先に来るため）", flush=True)
            hand_off(a, stop)
            return
        if now < end:
            time.sleep((end - now).total_seconds())
        flag = ROOT / "data" / "paper" / f"summary_sent_{day:%Y%m%d}"
        if not flag.exists():  # 予備の回が後から動いても、まとめは1日1回だけ
            send(summary(day.isoformat()), room="📊 毎日のまとめ")
            flag.parent.mkdir(parents=True, exist_ok=True)
            flag.touch()

    # 展示が取れずに見送ったレースが多かった日は、1日1回だけまとめて知らせる。
    # --until で途中終了する朝・昼の部はここまで来ないので、数がそろう夜の部で出す。
    if not a.once and not a.no_notify:
        notify_no_exhibit_skips(day)


if __name__ == "__main__":
    main()
