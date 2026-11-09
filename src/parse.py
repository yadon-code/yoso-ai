"""ダウンロードした番組表(B)と競走成績(K)のテキストを読み、CSVにまとめる。

使い方:
    python src/parse.py

出力 (data/csv/):
    programs.csv  レース前にわかる情報（選手の勝率、モーター、ボートなど）1行=1艇
    results.csv   レース後の結果（着順、進入コース、展示タイム、ST など）1行=1艇
    races.csv     レースごとの情報（天候、風、波、3連単の結果と払戻金）1行=1レース
"""
import csv
import re
import unicodedata
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "csv"

VENUES = {
    "01": "桐生", "02": "戸田", "03": "江戸川", "04": "平和島", "05": "多摩川", "06": "浜名湖",
    "07": "蒲郡", "08": "常滑", "09": "津", "10": "三国", "11": "びわこ", "12": "住之江",
    "13": "尼崎", "14": "鳴門", "15": "丸亀", "16": "児島", "17": "宮島", "18": "徳山",
    "19": "下関", "20": "若松", "21": "芦屋", "22": "福岡", "23": "唐津", "24": "大村",
}


def nfkc(s: str) -> str:
    return unicodedata.normalize("NFKC", s)


def to_float(s: str):
    s = s.strip()
    try:
        return float(s)
    except ValueError:
        return None


def file_date(path: Path) -> date:
    y, m, d = path.stem[1:3], path.stem[3:5], path.stem[5:7]
    return date(2000 + int(y), int(m), int(d))


def split_venues(lines, kind):
    """ファイルを開催場ごとのブロックに分ける。"""
    blocks, cur, code = [], None, None
    for ln in lines:
        m = re.match(rf"^(\d\d){kind}BGN", ln.decode("cp932", "replace"))
        if m:
            code, cur = m.group(1), []
            continue
        if re.match(rf"^\d\d{kind}END", ln.decode("cp932", "replace")):
            blocks.append((code, cur))
            cur = None
            continue
        if cur is not None:
            cur.append(ln)
    return blocks


# ---------- 番組表 (B) ----------
RE_B_HEAD = re.compile(r"^\s*(\d+)R\s+(\S+).*?H(\d+)m")
RE_B_ROW = re.compile(rb"^[1-6] \d{4}")


def parse_programs(path: Path):
    day = file_date(path)
    rows = []
    for code, lines in split_venues(path.read_bytes().splitlines(), "B"):
        race_no = race_name = dist = deadline = None
        for ln in lines:
            text = nfkc(ln.decode("cp932", "replace"))
            h = RE_B_HEAD.match(text)
            if h:
                race_no, race_name, dist = int(h.group(1)), h.group(2), int(h.group(3))
                dl = re.search(r"締切予定(\d+:\d+)", text)
                deadline = dl.group(1) if dl else None
                continue
            if race_no is None or not RE_B_ROW.match(ln):
                continue
            # 先頭24バイトは固定幅（全角は2バイト）
            head = ln[:24]
            # 勝率・2連率などは固定幅（数字がくっつくことがあるので空白区切りにしない）
            widths = [5, 6, 5, 6, 3, 6, 3, 6]
            nums, pos = [], 24
            for w in widths:
                nums.append(ln[pos:pos + w].decode("cp932", "replace"))
                pos += w
            rows.append({
                "date": day.isoformat(), "venue_code": code, "venue": VENUES.get(code, code),
                "race_no": race_no, "race_name": race_name, "distance": dist, "deadline": deadline,
                "boat": int(head[0:1]),
                "racer_id": int(head[2:6]),
                "racer_name": head[6:14].decode("cp932", "replace").replace("　", ""),
                "age": int(head[14:16]),
                "branch": head[16:20].decode("cp932", "replace"),
                "weight": int(head[20:22]),
                "class": head[22:24].decode("cp932", "replace"),
                "nat_win_rate": to_float(nums[0]), "nat_2rate": to_float(nums[1]),
                "local_win_rate": to_float(nums[2]), "local_2rate": to_float(nums[3]),
                "motor_no": int(nums[4]), "motor_2rate": to_float(nums[5]),
                "boat_no": int(nums[6]), "boat_2rate": to_float(nums[7]),
            })
    return rows


# ---------- 競走成績 (K) ----------
RE_K_HEAD = re.compile(
    r"^\s*(\d+)R\s+(.+?)\s+H(\d+)m\s+(\S+)\s+風\s+(\S+)\s+(\d+)m\s+波\s+(\d+)cm"
)
RE_K_ROW = re.compile(
    r"^\s+(\S\S?)\s+([1-6])\s+(\d{4})\s+(.+?)\s+(\d+)\s+(\d+)\s+([\d.]+)\s+(\d)\s+(\S+)\s*(\S*)"
)
RE_K_3T = re.compile(r"^\s*3連単\s+(\d-\d-\d)\s+(\d+)\s+人気\s+(\d+)")


def parse_results(path: Path):
    day = file_date(path)
    entries, races = [], []
    for code, lines in split_venues(path.read_bytes().splitlines(), "K"):
        race = None
        for ln in lines:
            text = nfkc(ln.decode("cp932", "replace"))
            h = RE_K_HEAD.match(text)
            if h and "[払戻金]" not in text:
                race = {
                    "date": day.isoformat(), "venue_code": code, "venue": VENUES.get(code, code),
                    "race_no": int(h.group(1)), "race_name": h.group(2).strip(),
                    "distance": int(h.group(3)), "weather": h.group(4),
                    "wind_dir": h.group(5), "wind_speed": int(h.group(6)),
                    "wave_cm": int(h.group(7)),
                    "trifecta": None, "trifecta_payout": None, "trifecta_popularity": None,
                }
                races.append(race)
                continue
            if race is None:
                continue
            t = RE_K_3T.match(text)
            if t:
                race["trifecta"] = t.group(1)
                race["trifecta_payout"] = int(t.group(2))
                race["trifecta_popularity"] = int(t.group(3))
                continue
            r = RE_K_ROW.match(text)
            if r:
                pos = r.group(1)
                rt = r.group(10)
                entries.append({
                    "date": race["date"], "venue_code": code, "race_no": race["race_no"],
                    "boat": int(r.group(2)), "racer_id": int(r.group(3)),
                    "finish": int(pos) if pos.isdigit() else None,
                    "finish_raw": pos,  # F=フライング, L=出遅れ, K=欠場, S=失格 など
                    "motor_no": int(r.group(5)), "boat_no": int(r.group(6)),
                    "exhibit_time": to_float(r.group(7)),
                    "course": int(r.group(8)),
                    "start_timing": to_float(r.group(9).replace("F", "-").replace("L", "")),
                    "race_time": rt if re.match(r"\d\.\d\d\.\d", rt) else None,
                })
    return entries, races


def write_csv(path: Path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    programs, entries, races = [], [], []
    for p in sorted((RAW / "B").glob("B*.TXT")):
        programs += parse_programs(p)
    for p in sorted((RAW / "K").glob("K*.TXT")):
        e, r = parse_results(p)
        entries += e
        races += r
    write_csv(OUT / "programs.csv", programs)
    write_csv(OUT / "results.csv", entries)
    write_csv(OUT / "races.csv", races)
    print(f"programs: {len(programs)} 行, results: {len(entries)} 行, races: {len(races)} 行")


if __name__ == "__main__":
    main()
