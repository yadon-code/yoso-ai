"""data_full の生データを、学習・検証で使う表（parquet）にまとめる。何度でも作り直せる（生データは消さない）。

    python src/v3/build_tables.py            # 全部
    python src/v3/build_tables.py raw bc     # 一部だけ

できる表（data_full/tables/）:
    programs.parquet  出走表（1行=1艇）。締切時刻・今節成績・節の何日目・大会名つき
    results.parquet   成績（1行=1艇）
    races.parquet     レース（1行=1レース）。気象（発走時）・決まり手
    payouts.parquet   全券種の払戻（1行=1つの当たり組。単勝/複勝/2連単/2連複/拡連複/3連単/3連複）
    odds3t.parquet    3連単の確定オッズ（1行=1レース、120列 o_1-2-3 …）  出どころ src=bc/official
    odds_other.parquet 2連単30・2連複15・3連複20・単勝6・複勝(下限/上限)・拡連複(下限/上限)の確定オッズ（BOATCAST の日だけ）
    oriten.parquet    オリジナル展示（1行=1艇）
    tkz.parquet       直前情報（BOATCAST 版。1行=1艇。展示タイム・体重・調整重量・チルト・スタート展示）
    bi_official.parquet 公式ページの直前情報（BOATCAST が無い日）
    kibetsu.parquet   期別成績（1行=1選手×1期）
"""
import paths  # noqa: F401  src/ の本番のモジュールを読めるようにする（先に読む）
import gzip
import json
import re
import sys
import unicodedata
from itertools import combinations, permutations

import numpy as np
import pandas as pd

import parse
from paths import BC, KIBETSU, OFFICIAL, RAW, TABLES

COMBOS3T = [f"{a}-{b}-{c}" for a, b, c in permutations(range(1, 7), 3)]
COMBOS2T = [f"{a}-{b}" for a, b in permutations(range(1, 7), 2)]
COMBOS2F = [f"{a}-{b}" for a, b in combinations(range(1, 7), 2)]
COMBOS3F = [f"{a}-{b}-{c}" for a, b, c in combinations(range(1, 7), 3)]
nfkc = parse.nfkc


# ---------- 番組表・成績・払戻（公式テキスト） ----------
RE_DAYHEAD = re.compile(r"第\s*(\d+)日")
RE_PAY = re.compile(r"^\s*(単勝|複勝|2連単|2連複|拡連複|3連単|3連複)?\s+([1-6](?:-[1-6]){0,2})\s+(\d+)(?:\s+人気\s+(\d+))?(.*)$")
RE_B_TAIL = re.compile(r"^.*?\d+\.\d+\s+\d+\.\d+\s+\d+\.\d+\s+\d+\.\d+\s+\d+\s+\d+\.\d+\s+\d+\s+\d+\.\d+ ?(.*)$")
RE_KIMARITE = re.compile(r"レースタイム\s*(\S+)")  # NFKC で半角カナが全角になる


def parse_b_extra(path):
    """番組表から、締切時刻・大会名・節の何日目・今節成績・早見を取る。"""
    day = parse.file_date(path)
    out = []
    for code, lines in parse.split_venues(path.read_bytes().splitlines(), "B"):
        texts = [nfkc(l.decode("cp932", "replace")) for l in lines]
        m = [RE_DAYHEAD.search(t) for t in texts[:3]]
        meet_day = next((int(x.group(1)) for x in m if x), None)
        head = texts[0] if texts else ""
        race_no = None
        for raw, t in zip(lines, texts):
            h = parse.RE_B_HEAD.match(t)
            if h:
                race_no = int(h.group(1))
                continue
            if race_no and parse.RE_B_ROW.match(raw):
                m2 = RE_B_TAIL.match(raw.decode("cp932", "replace").rstrip())
                tail = m2.group(1) if m2 else ""
                out.append({"date": day.isoformat(), "venue_code": code, "race_no": race_no,
                            "boat": int(t[0]), "meet_day": meet_day, "event": head,
                            "setsu_marks": tail[:12].rstrip(), "hayami": tail[12:].strip()})
    return out


def parse_k_extra(path):
    """競走成績から、全券種の払戻と決まり手を取る。"""
    day = parse.file_date(path)
    pays, kim = [], []
    for code, lines in parse.split_venues(path.read_bytes().splitlines(), "K"):
        race_no, kind = None, None
        for ln in lines:
            t = nfkc(ln.decode("cp932", "replace"))
            h = parse.RE_K_HEAD.match(t)
            if h and "[払戻金]" not in t:
                race_no, kind = int(h.group(1)), None
                continue
            if race_no is None:
                continue
            k = RE_KIMARITE.search(t)
            if k:
                kim.append({"date": day.isoformat(), "venue_code": code, "race_no": race_no, "kimarite": k.group(1)})
                continue
            p = RE_PAY.match(t)
            if p and (p.group(1) or kind):
                kind = p.group(1) or kind
                pays.append({"date": day.isoformat(), "venue_code": code, "race_no": race_no, "bet": kind,
                             "combo": p.group(2), "payout": int(p.group(3)),
                             "popularity": int(p.group(4)) if p.group(4) else None})
                rest = p.group(5).strip()  # 複勝は1行に2つ
                q = re.match(r"^([1-6])\s+(\d+)", rest)
                if q and kind == "複勝":
                    pays.append({"date": day.isoformat(), "venue_code": code, "race_no": race_no, "bet": kind,
                                 "combo": q.group(1), "payout": int(q.group(2)), "popularity": None})
            elif t.strip() == "" or t.lstrip().startswith(("着", "--")):
                kind = None if t.strip() == "" else kind
    return pays, kim


def build_raw():
    progs, ents, races, extra, pays, kims = [], [], [], [], [], []
    for p in sorted((RAW / "B").glob("B*.TXT")):
        progs += parse.parse_programs(p)
        extra += parse_b_extra(p)
    for p in sorted((RAW / "K").glob("K*.TXT")):
        e, r = parse.parse_results(p)
        ents += e
        races += r
        a, k = parse_k_extra(p)
        pays += a
        kims += k
    key = ["date", "venue_code", "race_no", "boat"]
    pr = pd.DataFrame(progs).merge(pd.DataFrame(extra), on=key, how="left")
    pr.to_parquet(TABLES / "programs.parquet", index=False)
    pd.DataFrame(ents).to_parquet(TABLES / "results.parquet", index=False)
    rc = pd.DataFrame(races).merge(pd.DataFrame(kims).drop_duplicates(["date", "venue_code", "race_no"]),
                                   on=["date", "venue_code", "race_no"], how="left")
    rc.to_parquet(TABLES / "races.parquet", index=False)
    pd.DataFrame(pays).to_parquet(TABLES / "payouts.parquet", index=False)
    print(f"programs {len(pr)} / results {len(ents)} / races {len(rc)} / payouts {len(pays)}", flush=True)


# ---------- BOATCAST ----------
def fnum(s):
    try:
        v = float(s)
        return v if v > 0 else np.nan
    except (TypeError, ValueError):
        return np.nan


def body(text):
    return text.split("\n")[2:] if text else []


def parse_od3(t):
    lines = [l.split("\t") for l in body(t) if l.strip()]
    if len(lines) < 6:
        return None
    out = []
    for i in range(6):
        out += [fnum(x) for x in lines[i][1:21]]
    return out if len(out) == 120 else None


def parse_od2(t):
    lines = [l.split("\t") for l in body(t) if l.strip()]
    if len(lines) < 7:
        return None, None
    o2t = []
    for i in range(6):
        o2t += [fnum(x) for x in lines[i][1:6]]
    o2f = [fnum(x) for x in lines[6][:15]]
    return (o2t if len(o2t) == 30 else None), (o2f if len(o2f) == 15 else None)


def ranges(cells):
    v = [c for c in cells if c.strip() and c.strip() != "～"]
    return [fnum(x) for x in v[0::2]], [fnum(x) for x in v[1::2]]


def parse_od1(t):
    lines = [l.split("\t") for l in body(t) if l.strip()]
    if len(lines) < 6:
        return None
    o3f = [fnum(x) for x in lines[0][:20]]
    k_lo, k_hi = ranges(lines[1])
    tan = [fnum(x) for x in lines[2][:6]]
    f_lo, f_hi = ranges(lines[5])
    if len(o3f) != 20 or len(k_lo) != 15 or len(tan) != 6 or len(f_lo) != 6:
        return None
    return {"o3f": o3f, "kaku_lo": k_lo, "kaku_hi": k_hi[:15], "tan": tan, "fuku_lo": f_lo, "fuku_hi": f_hi[:6]}


def parse_tkz(t):
    """2〜7行目が1〜6号艇（名前, 展示, 調整あり, 調整重量x10, 体重, ?, チルト）、8行目がスタート展示（進入順に 艇,ST,F）。
    src/download_tkz.py と同じ読み方。"""
    lines = t.strip("\n").split("\n") if t else []
    if len(lines) < 8:
        return None
    rows = []
    for i, l in enumerate(lines[2:8]):
        c = l.split("\t")
        if len(c) < 7:
            return None
        adj = fnum(c[3]) if c[3].strip() not in ("0", "") else 0.0
        rows.append({"boat": i + 1, "exhibit_time_bi": fnum(c[1]), "weight_day": fnum(c[4]),
                     "adjust_weight": adj / 10 if adj == adj else np.nan, "tilt": float(c[6].replace(" ", "")) if c[6].strip() else np.nan,
                     "exh_course": np.nan, "exh_st": np.nan, "exh_st_f": 0})
    st = [x.strip() for x in lines[8].split("\t")] if len(lines) > 8 else []
    for k in range(0, len(st) - 1, 3):
        if not st[k].isdigit() or not 1 <= int(st[k]) <= 6:
            continue
        r = rows[int(st[k]) - 1]
        f = k + 2 < len(st) and st[k + 2] == "F"
        try:
            v = float(st[k + 1].replace(" ", ""))
        except ValueError:
            v = np.nan
        r["exh_course"] = k // 3 + 1
        r["exh_st"] = -v if f else v
        r["exh_st_f"] = int(f)
    return rows


def parse_oriten(t):
    from download_bc import parse_oriten as po
    return po(t) if t else None


def build_bc():
    o3, oth, ori, tkz = [], [], [], []
    for p in sorted(BC.glob("*/bc_*.jsonl.gz")):
        with gzip.open(p, "rt", encoding="utf-8") as f:
            rows = [json.loads(l) for l in f]
        if not rows or rows[0].get("no_bc"):
            continue
        for r in rows:
            key = {"date": r["date"], "venue_code": r["venue_code"], "race_no": r["race_no"]}
            v = parse_od3(r.get("od3")) if isinstance(r.get("od3"), str) and r["od3"] != "!ERR" else None
            if v:
                o3.append({**key, "src": "bc", **dict(zip(["o_" + c for c in COMBOS3T], v))})
            d = dict(key)
            if isinstance(r.get("od2"), str) and r["od2"] != "!ERR":
                a, b = parse_od2(r["od2"])
                if a:
                    d.update(zip(["t2_" + c for c in COMBOS2T], a))
                if b:
                    d.update(zip(["f2_" + c for c in COMBOS2F], b))
            if isinstance(r.get("od1"), str) and r["od1"] != "!ERR":
                x = parse_od1(r["od1"])
                if x:
                    d.update(zip(["f3_" + c for c in COMBOS3F], x["o3f"]))
                    d.update(zip(["klo_" + c for c in COMBOS2F], x["kaku_lo"]))
                    d.update(zip(["khi_" + c for c in COMBOS2F], x["kaku_hi"]))
                    d.update(zip([f"tan_{i}" for i in range(1, 7)], x["tan"]))
                    d.update(zip([f"flo_{i}" for i in range(1, 7)], x["fuku_lo"]))
                    d.update(zip([f"fhi_{i}" for i in range(1, 7)], x["fuku_hi"]))
            if len(d) > 3:
                oth.append(d)
            if isinstance(r.get("oriten"), str) and r["oriten"] != "!ERR":
                for b in parse_oriten(r["oriten"]) or []:
                    ori.append({**key, **b})
            if isinstance(r.get("tkz"), str) and r["tkz"] != "!ERR":
                for b in parse_tkz(r["tkz"]) or []:
                    tkz.append({**key, **b})
    for p in sorted(OFFICIAL.glob("*/of_*.jsonl.gz")):
        with gzip.open(p, "rt", encoding="utf-8") as f:
            for r in map(json.loads, f):
                if r.get("odds3t"):
                    key = {"date": r["date"], "venue_code": r["venue_code"], "race_no": r["race_no"]}
                    o3.append({**key, "src": "official",
                               **{"o_" + c: (r["odds3t"].get(c) or np.nan) for c in COMBOS3T}})
    pd.DataFrame(o3).to_parquet(TABLES / "odds3t.parquet", index=False)
    pd.DataFrame(oth).to_parquet(TABLES / "odds_other.parquet", index=False)
    pd.DataFrame(ori).to_parquet(TABLES / "oriten.parquet", index=False)
    pd.DataFrame(tkz).to_parquet(TABLES / "tkz.parquet", index=False)
    bi = []
    for p in sorted(OFFICIAL.glob("*/of_*.jsonl.gz")):
        with gzip.open(p, "rt", encoding="utf-8") as f:
            for r in map(json.loads, f):
                for b in r.get("bi") or []:
                    bi.append({"date": r["date"], "venue_code": r["venue_code"], "race_no": r["race_no"], **b})
    pd.DataFrame(bi).to_parquet(TABLES / "bi_official.parquet", index=False)
    print(f"odds3t {len(o3)} / odds_other {len(oth)} / oriten {len(ori)} / tkz {len(tkz)} / bi_official {len(bi)}", flush=True)


# ---------- 期別成績 ----------
def build_kibetsu():
    """ファン手帳の固定長テキスト。1行=1選手。まずは確かめやすい先頭の項目と、コース別の成績を読む。
    桁の位置は公式の「レーサー期別成績 レイアウト」に合わせる（kibetsu_layout）。"""
    from kibetsu_layout import parse_file
    rows = []
    for p in sorted(KIBETSU.glob("fan*.TXT")):
        rows += parse_file(p)
    pd.DataFrame(rows).to_parquet(TABLES / "kibetsu.parquet", index=False)
    print(f"kibetsu {len(rows)}", flush=True)


if __name__ == "__main__":
    TABLES.mkdir(parents=True, exist_ok=True)
    what = sys.argv[1:] or ["raw", "bc", "kibetsu"]
    if "raw" in what:
        build_raw()
    if "bc" in what:
        build_bc()
    if "kibetsu" in what:
        build_kibetsu()
