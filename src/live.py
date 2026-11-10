"""当日のレースを、締切の少し前に1レースずつ自動で予想し、エア投票に記録する（1日動かしっぱなし）。

使い方:
    python src/live.py                 # 今日
    python src/live.py 2026-11-10      # 日付指定
    python src/live.py --once          # 待たずに、今の時点で締切前のレースを全部1回だけ予想

流れ:
    朝: 前日までのデータを更新 → 今日の出走表を取る → 過去の成績から作る特徴量を先に計算しておく（数分）
    各レースの締切 MINUTES_BEFORE 分前: 展示タイム・気象・最新オッズを取り、展示に関わる特徴量だけ計算し直して予想
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
from download_odds import fetch_race as fetch_odds  # noqa: E402
from features import FEATURES, KEY, build  # noqa: E402
from notify import race_message, send  # noqa: E402
import json  # noqa: E402
import market_model as mm  # noqa: E402
from download_bc import BASE as BC_BASE, get as bc_get, parse_oriten  # noqa: E402
from settle_live import summary  # noqa: E402
from predict import ROOT, SETTINGS, bi_wind_index, todays_programs, update_history  # noqa: E402

JST = dt.timezone(dt.timedelta(hours=9))
MINUTES_BEFORE = 5


def prepare(day):
    """重い準備は一度だけ。結果をファイルに保存して、途中で止まっても再利用する。"""
    cache = ROOT / "data" / "cache" / f"live_{day:%Y%m%d}.pkl"
    if cache.exists():
        prog, df = pd.read_pickle(cache)
        return prog, df
    update_history(day)
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
        d = d.merge(bi.drop(columns=["date", "venue_code", "race_no"], errors="ignore"), on="boat", how="left")
    if len(ori):
        d = d.merge(ori, on="boat", how="left")
    d = mm.make_features(d)
    pw = mm.predict_win(booster, d)
    q = mm.trifecta_probs(pm[None, :], (pw / mw)[None, :], conf["a"], conf["b"], conf["c"])[0]
    ev = q * o
    picks = [(COMBOS[i], o[i], ev[i], q[i])
             for i in np.where((ev > conf["ev_threshold"]) & (o <= conf["max_odds"]))[0]]
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


def save_odds_snapshot(day, key, deadline_dt, odds_rows):
    """予想した時点（締切の数分前）の3連単オッズ120通りを保存する。あとからは取り直せないデータ。
    1行=1レース。odds は backtest.COMBOS の順に空白区切り。"""
    if len(odds_rows) != 120:
        return
    now = dt.datetime.now(JST)
    o = pd.DataFrame(odds_rows).set_index("combo").reindex(COMBOS)["odds"]
    row = {"date": key[0], "venue_code": key[1], "race_no": key[2], "fetched_at": now.strftime("%Y-%m-%d %H:%M:%S"),
           "seconds_before": int((deadline_dt - now).total_seconds()),
           "odds": " ".join("" if pd.isna(x) else f"{x:g}" for x in o)}
    append_rows(ROOT / "data" / "odds_pre" / f"pre_{day:%Y%m%d}.csv", [row])


def record(day, key, p, p3, picks, has_exhibit, model="v1"):
    now = dt.datetime.now(JST).strftime("%Y-%m-%d %H:%M")
    top = np.argsort(-p3)[:5]
    row = {"date": key[0], "venue_code": key[1], "race_no": key[2], "predicted_at": now,
           "exhibit": has_exhibit, "p_win": " ".join(f"{x:.3f}" for x in p),
           "top5": " ".join(f"{COMBOS[i]}:{p3[i]:.3f}" for i in top),
           "picks": " ".join(f"{c}({o:.1f}倍,期待値{e:.2f})" for c, o, e, _ in picks),
           "model": model}  # v2=第2段階（オッズのズレ補正）、v1=第1段階だけ（第2段階が使えなかったとき）
    append_rows(ROOT / "data" / "predictions" / f"live_{day:%Y%m%d}.csv", [row])
    if picks and has_exhibit:  # エア投票は、展示を見たうえでの本番と同じ条件の予想だけ記録する
        led = ROOT / "data" / "paper" / "bets.csv"
        rows = [{"date": key[0], "venue_code": key[1], "race_no": key[2], "combo": c, "odds_at_pick": o,
                 "ev": round(e, 3), "prob": round(b, 5), "picked_at": now, "stake": 100, "model": model}
                for c, o, e, b in picks]
        append_rows(led, rows)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("date", nargs="?")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--no-notify", action="store_true", help="通知を送らない")
    ap.add_argument("--until", help="この時刻(HH:MM)より後が締切のレースは次の回に任せて終わる（GitHubの1回6時間の制限用）")
    a = ap.parse_args()
    day = dt.date.fromisoformat(a.date) if a.date else dt.datetime.now(JST).date()
    print(f"{day} の準備中...", flush=True)
    prog, df = prepare(day)
    booster = lgb.Booster(model_file=str(ROOT / "models" / "win_model.txt"))
    mmodel = load_market_model()
    print("第2段階モデル:", "あり" if mmodel else "なし（第1段階だけで予想）", flush=True)
    sched = prog.drop_duplicates(KEY)[KEY + ["venue", "deadline"]].copy()
    sched["deadline_dt"] = sched["deadline"].map(
        lambda t: dt.datetime.combine(day, dt.time(*map(int, t.split(":"))), JST) if isinstance(t, str) else None)
    sched = sched.dropna(subset=["deadline_dt"]).sort_values("deadline_dt")
    sp = ROOT / "data" / "predictions" / f"schedule_{day:%Y%m%d}.csv"
    sp.parent.mkdir(parents=True, exist_ok=True)
    sched[KEY + ["venue", "deadline"]].to_csv(sp, index=False)  # ウェブアプリ用の出走表（締切時刻）
    done_path = ROOT / "data" / "predictions" / f"live_{day:%Y%m%d}.csv"
    done = set()
    if done_path.exists():
        d = pd.read_csv(done_path, dtype={"venue_code": str})
        done = set(zip(d.venue_code, d.race_no))
    print(f"対象 {len(sched)} レース（予想済み {len(done)}）", flush=True)

    until = dt.datetime.combine(day, dt.time(*map(int, a.until.split(":"))), JST) if a.until else None
    v2_errors = 0
    for _, r in sched.iterrows():
        if (r.venue_code, r.race_no) in done:
            continue
        if until and r.deadline_dt > until:
            print(f"{a.until} 以降のレースは次の回で予想します", flush=True)
            return
        target = r.deadline_dt - dt.timedelta(minutes=MINUTES_BEFORE)
        now = dt.datetime.now(JST)
        if now > r.deadline_dt:
            continue  # 締切を過ぎたレースは飛ばす
        if not a.once and now < target:
            time.sleep((target - now).total_seconds())
        key = (r.date, r.venue_code, r.race_no)
        g = df[(df.venue_code == r.venue_code) & (df.race_no == r.race_no)]
        if len(g) != 6:
            continue
        bi = pd.DataFrame(fetch_bi(r.date, r.venue_code, int(r.race_no)))
        has_ex = len(bi) == 6 and bi["exhibit_time_bi"].fillna(0).gt(0).all()
        if has_ex:
            g = refresh(g, bi)
        odds_rows = fetch_odds(r.date, r.venue_code, int(r.race_no))
        try:
            save_odds_snapshot(day, key, r.deadline_dt, odds_rows)
        except Exception as e:  # 保存の失敗で予想を止めない
            print("オッズの保存に失敗:", e, flush=True)
        p, p3, picks = predict_race(booster, g, odds_rows)
        model = "v1"
        if mmodel and has_ex and len(odds_rows) == 120 and all(x["odds"] for x in odds_rows):
            try:
                ori = fetch_oriten(r.date, r.venue_code, r.race_no)
                p, p3, picks = predict_race_v2(mmodel, key, p, bi, ori, odds_rows)
                model = "v2"
            except Exception as e:  # 第2段階で失敗したら第1段階の結果を使う
                print("第2段階でエラー:", e, flush=True)
                v2_errors += 1
                if v2_errors == 3 and not a.no_notify:  # 続けて起きているときだけ、1回知らせる
                    send(f"⚠️ 第2段階の予想でエラーが続いています（{e}）。第1段階だけで予想を続けます。")
        row = record(day, key, p, p3, picks, has_ex, model)
        print(f"{r.venue} {r.race_no:>2}R 締切{r.deadline} 展示{'あり' if has_ex else 'なし'} "
              f"1着確率[{row['p_win']}]  買い目: {row['picks'] or 'なし'}", flush=True)
        if picks and has_ex and not a.no_notify:
            send(race_message(r.venue, r.race_no, r.deadline, p, picks), room=f"🚤 {r.venue}")

    # 最終レースの結果が出るころ（締切から20分後）に、1日のまとめを通知する
    if not a.once and not a.no_notify and len(sched):
        end = sched["deadline_dt"].max() + dt.timedelta(minutes=20)
        now = dt.datetime.now(JST)
        if now < end:
            time.sleep((end - now).total_seconds())
        send(summary(day.isoformat()), room="📊 毎日のまとめ")


if __name__ == "__main__":
    main()
