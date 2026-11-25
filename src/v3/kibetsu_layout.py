"""公式「モーターボートファン手帳」（レーサー期別成績）の固定長レイアウト（2014年後期以降）。
出どころ: https://www.boatrace.jp/owpc/pc/extra/data/layout.html （1行=1選手、cp932、416バイト）
"""
import unicodedata

HEAD = [("racer_id", 4, "s"), ("name", 16, "s"), ("kana", 15, "s"), ("branch", 4, "s"), ("class", 2, "s"),
        ("era", 1, "s"), ("birth", 6, "s"), ("sex", 1, "i"), ("age", 2, "i"), ("height", 3, "i"), ("weight", 2, "i"),
        ("blood", 2, "s"), ("win_rate", 4, 100), ("rate2", 4, 10), ("n1", 3, "i"), ("n2", 3, "i"), ("starts", 3, "i"),
        ("finals", 2, "i"), ("champs", 2, "i"), ("avg_st", 3, 100)]
COURSE = [("entries", 3, "i"), ("rate2", 4, 10), ("avg_st", 3, 100), ("avg_st_rank", 3, 100)]
MID = [("class_prev1", 2, "s"), ("class_prev2", 2, "s"), ("class_prev3", 2, "s"), ("ability_prev", 4, 100),
       ("ability", 4, 100), ("year", 4, "i"), ("term", 1, "i"), ("calc_from", 8, "s"), ("calc_to", 8, "s"),
       ("training_no", 3, "i")]
COURSE2 = [(f"f{k}", 3, "i") for k in range(1, 7)] + [(x, 2, "i") for x in ["F", "L0", "L1", "K0", "K1", "S0", "S1", "S2"]]
TAIL = [("nocourse_L0", 2, "i"), ("nocourse_L1", 2, "i"), ("nocourse_K0", 2, "i"), ("nocourse_K1", 2, "i"),
        ("birthplace", 6, "s")]


def layout():
    out = list(HEAD)
    for c in range(1, 7):
        out += [(f"c{c}_{n}", w, t) for n, w, t in COURSE]
    out += MID
    for c in range(1, 7):
        out += [(f"c{c}_{n}", w, t) for n, w, t in COURSE2]
    return out + TAIL


LAYOUT = layout()
WIDTH = sum(w for _, w, _ in LAYOUT)


def conv(b, t):
    s = unicodedata.normalize("NFKC", b.decode("cp932", "replace")).strip()
    if t == "s":
        return s.replace(" ", "")
    if not s:
        return None
    try:
        v = int(s)
    except ValueError:
        return None
    return v if t == "i" else v / t


def parse_file(path):
    rows = []
    for line in path.read_bytes().split(b"\n"):
        line = line.rstrip(b"\r")
        if len(line) < WIDTH - 6:
            continue
        r, pos = {}, 0
        for name, w, t in LAYOUT:
            r[name] = conv(line[pos:pos + w], t)
            pos += w
        r["file"] = path.stem
        rows.append(r)
    return rows
