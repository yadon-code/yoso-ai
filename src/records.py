"""予想・エア投票などの「元の記録」を、合言葉つきの金庫にしまう・戻す・保存する。

元の記録（回をまたいで読み書きし、なくすと困るもの）:
    data/predictions/          予想（live_*.csv）・出走表メモ（schedule_*.csv）
    data/paper/                エア投票（bets.csv）・結果のキャッシュ（results_live.csv）・まとめを送った印（summary_sent_*）
    data/odds_pre/             予想した時点のオッズ（あとからは取り直せない）
    data/discord_threads.json  Discord の部屋のID
ウェブ用データ data/site/ は、記録から作り直せる（build_site.py）。

合言葉（SITE_PASSWORD）があるときの保存先:
    site-data ブランチ    10分ごと。ウェブ用データ(data/site)と金庫(data/vault)の最新だけ。1コミットだけを上書きするので履歴が積もらない
    main の data/vault/  回の最後だけ。日ごとの束を圧縮して暗号化したもの。中身が同じ日は同じファイルのままなので増えない
    Actions のキャッシュ  data/cache/records/ に元の記録の写し（data/cache はもともとキャッシュに入れている）
合言葉がないとき・RECORDS_MODE=legacy のときは、前と同じく元の記録を main にそのまま保存する（site-data にも出す）。

使い方（GitHub Actions から）:
    python src/records.py restore        回の始めに、キャッシュ・site-data・main のうち新しいものから記録を戻す
    python src/records.py save [final]   scripts/push_data.sh から呼ぶ（final は回の最後）
"""
import datetime as dt
import hashlib
import json
import lzma
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sitecrypt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
VAULT = DATA / "vault"
SITE = DATA / "site"
SNAP = DATA / "cache" / "records"          # キャッシュに入る写し
RECORD_DIRS = ("predictions", "paper", "odds_pre")
RECORD_FILES = ("discord_threads.json",)
RECORD_PATHS = [f"data/{x}" for x in RECORD_DIRS + RECORD_FILES]
SPLIT = ("paper/bets.csv", "paper/results_live.csv")   # 全部の日が入った表。日ごとの束に行を分けて入れる
BRANCH = "site-data"
DAY = re.compile(r"(20\d{6})")
JST = dt.timezone(dt.timedelta(hours=9))


class Fail(Exception):
    pass


class Locked(Exception):
    """金庫を開けられない（合言葉が無い・ちがう・壊れている）。"""


def log(msg):
    print(msg, flush=True)


def gh(kind, title, msg):
    print(f"::{kind} title={title}::{msg}", flush=True)


def run(args, env=None, input=None, check=False):
    r = subprocess.run(args, cwd=ROOT, capture_output=True, input=input, env=env)
    if check and r.returncode:
        raise Fail(f"{' '.join(args[:3])} に失敗: {r.stderr.decode('utf-8', 'replace').strip()[:300]}")
    return r


def out(args, **kw):
    return run(args, check=True, **kw).stdout.decode("utf-8").strip()


def git_dir() -> Path:
    return ROOT / out(["git", "rev-parse", "--git-dir"])


def now_stamp():
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def nlines(b: bytes) -> int:
    return b.count(b"\n") + (1 if b and not b.endswith(b"\n") else 0)


def is_record(p: str) -> bool:
    return p.split("/", 1)[0] in RECORD_DIRS or p in RECORD_FILES


def load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


# ---------- 手元のファイル ----------

def read_files(base: Path, subs) -> dict:
    """base の下の subs（フォルダかファイル）を {data からの相対パス: 中身} で返す。"""
    found = {}
    for s in subs:
        p = base / s
        if p.is_file():
            found[s] = p.read_bytes()
        elif p.is_dir():
            for f in sorted(p.rglob("*")):
                if f.is_file() and not f.name.startswith("."):
                    found[f.relative_to(base).as_posix()] = f.read_bytes()
    return found


def local_records() -> dict:
    return read_files(DATA, RECORD_DIRS + RECORD_FILES)


def record_days(files: dict) -> set:
    return set(group(files)) - {"common"}


def site_days(paths) -> set:
    return {m.group(1) for p in paths if p.startswith("site/") and (m := re.match(r"site/(\d{8})\.json", p))}


# ---------- 金庫の束 ----------

def split_rows(b: bytes):
    """見出し1行＋「日付,…」の行を、日ごとに分ける。分けられない形なら None。"""
    lines = b.splitlines(keepends=True)
    if not lines:
        return None
    parts = {}
    for ln in lines[1:]:
        m = re.match(rb"(\d{4})-(\d{2})-(\d{2}),", ln)
        if not m:
            return None
        parts.setdefault(b"".join(m.groups()).decode(), []).append(ln)
    return lines[0], {k: b"".join(v) for k, v in parts.items()}


def join_rows(header: bytes, parts: dict) -> bytes:
    return header + b"".join(parts[k] for k in sorted(parts))


def group(files: dict) -> dict:
    """ファイルを日ごとの束（YYYYMMDD）と、日付のないもの（common）に分ける。"""
    g = {"common": {}}
    for path in sorted(files):
        b = files[path]
        if path in SPLIT:
            sp = split_rows(b)
            if sp and join_rows(*sp) == b:   # 日付の順に並んでいて、つなぎ直すと元どおりになるときだけ分ける
                g["common"][path + "#header"] = sp[0]
                for day, rows in sp[1].items():
                    g.setdefault(day, {})[path + "#rows"] = rows
                continue
        m = DAY.search(path.rsplit("/", 1)[-1])
        g.setdefault(m.group(1) if m else "common", {})[path] = b
    return g


def ungroup(g: dict) -> dict:
    files, rows, headers = {}, {}, {}
    for name, d in g.items():
        for p, b in d.items():
            if p.endswith("#rows"):
                rows.setdefault(p[:-5], {})[name] = b
            elif p.endswith("#header"):
                headers[p[:-7]] = b
            else:
                files[p] = b
    if set(rows) - set(headers):
        raise Fail("金庫の中身がそろっていません（表の見出しが見つからない）")
    for p, h in headers.items():
        files[p] = join_rows(h, rows.get(p, {}))
    return files


def pack_bundle(d: dict) -> bytes:
    names = sorted(d)
    head = json.dumps([[n, len(d[n])] for n in names], ensure_ascii=False).encode("utf-8")
    return lzma.compress(b"YVB1\n" + head + b"\n" + b"".join(d[n] for n in names), preset=9)


def unpack_bundle(z: bytes) -> dict:
    raw = lzma.decompress(z)
    if not raw.startswith(b"YVB1\n"):
        raise Fail("金庫の束の形がちがいます")
    i = raw.index(b"\n", 5)
    pos, d = i + 1, {}
    for n, size in json.loads(raw[5:i]):
        d[n] = raw[pos:pos + size]
        pos += size
    if pos != len(raw):
        raise Fail("金庫の束の長さが合いません")
    return d


def seal_vault(files: dict, pw: str) -> dict:
    return {f"{n}.bin": sitecrypt.seal(pack_bundle(d), pw, f"vault/{n}") for n, d in group(files).items()}


def open_vault(blobs: dict, pw: str) -> dict:
    g = {}
    for name, blob in blobs.items():
        if name.endswith(".bin"):
            try:
                plain = sitecrypt.unseal(blob, pw)
            except Exception as e:
                raise Locked(f"{name} を開けません（{type(e).__name__}）") from None
            g[name[:-4]] = unpack_bundle(plain)
    return ungroup(g)


def content_hash(d: dict) -> str:
    h = hashlib.sha256()
    for n in sorted(d):
        h.update(n.encode("utf-8") + b"\0" + hashlib.sha256(d[n]).digest())
    return h.hexdigest()


def pack(pw: str) -> dict:
    """手元の元の記録から金庫（data/vault/）を作り直す。作ったものを開き直して、元と同じか確かめる。"""
    files = local_records()
    blobs = seal_vault(files, pw)
    if open_vault(blobs, pw) != files:
        raise Fail("金庫を作って開き直したら、元の記録と合いませんでした")
    content = content_hash(blobs)
    prev = load_json(VAULT / "index.json")
    saved = prev["saved_at"] if prev.get("content") == content and prev.get("saved_at") else now_stamp()
    VAULT.mkdir(parents=True, exist_ok=True)
    for f in VAULT.glob("*.bin"):
        if f.name not in blobs:
            f.unlink()
    for n, b in blobs.items():
        p = VAULT / n
        if not p.exists() or p.read_bytes() != b:
            p.write_bytes(b)
    index = {"format": 1, "saved_at": saved, "content": content,
             "days": sorted(n[:-4] for n in blobs if n != "common.bin"),
             "note": "予想・エア投票などの元の記録を、ウェブアプリと同じ合言葉で暗号化した金庫（src/records.py）"}
    text = json.dumps(index, ensure_ascii=False, indent=1, sort_keys=True) + "\n"
    if not (VAULT / "index.json").exists() or (VAULT / "index.json").read_text(encoding="utf-8") != text:
        (VAULT / "index.json").write_text(text, encoding="utf-8")
    log(f"金庫: {len(index['days'])}日分・{len(files)}ファイル（{'前と同じ' if saved == prev.get('saved_at') else '更新'}）")
    return files


def snapshot(vault_saved=""):
    """キャッシュ用の写し data/cache/records/ を作る（入れ替えは最後にまとめて）。
    vault_saved: いま作った金庫の保存時刻。写しは金庫と同じ中身なので、それより古い時刻にはしない。"""
    files = local_records()
    content = content_hash(files)
    prev = load_json(SNAP / "STAMP.json")
    saved = prev["saved_at"] if prev.get("content") == content and prev.get("saved_at") else now_stamp()
    saved = max(saved, vault_saved or "")
    tmp, old = SNAP.with_name("records.tmp"), SNAP.with_name("records.old")
    shutil.rmtree(tmp, ignore_errors=True)
    for p, b in read_files(DATA, RECORD_DIRS + RECORD_FILES + ("site", "vault")).items():
        (tmp / p).parent.mkdir(parents=True, exist_ok=True)
        (tmp / p).write_bytes(b)
    tmp.mkdir(parents=True, exist_ok=True)
    (tmp / "STAMP.json").write_text(json.dumps({"saved_at": saved, "content": content, "files": len(files)}),
                                    encoding="utf-8")
    shutil.rmtree(old, ignore_errors=True)
    if SNAP.exists():
        SNAP.rename(old)
    tmp.rename(SNAP)
    shutil.rmtree(old, ignore_errors=True)


# ---------- 記録の置き場（キャッシュの写し・git のブランチ） ----------

class Source:
    def __init__(self, label, files, saved_at, cache=False):
        self.label, self.files, self.saved_at, self.cache = label, files, saved_at, cache

    @property
    def has_vault(self):
        return "vault/index.json" in self.files

    def empty(self):
        return not any(is_record(p) or p.startswith("vault/") for p in self.files)

    def records(self, pw):
        """この置き場の元の記録 {相対パス: 中身}。金庫を開けないときは Locked。"""
        if self.cache or not self.has_vault:
            return {p: b for p, b in self.files.items() if is_record(p)}
        if not pw:
            raise Locked("合言葉（SITE_PASSWORD）が無い")
        return open_vault({p[6:]: b for p, b in self.files.items() if p.startswith("vault/")}, pw)

    def vault_days(self):
        try:
            return set(json.loads(self.files["vault/index.json"]).get("days", []))
        except Exception:
            return set()


def dir_source(base: Path, label):
    files = read_files(base, RECORD_DIRS + RECORD_FILES + ("site", "vault"))
    return Source(label, files, load_json(base / "STAMP.json").get("saved_at", ""), cache=True)


def tree_source(ref, label):
    ls = run(["git", "ls-tree", "-r", "-z", "--name-only", ref, "--"] + RECORD_PATHS + ["data/site", "data/vault"],
             check=True).stdout.decode("utf-8").split("\0")
    paths = [p for p in ls if p]
    files = {}
    if paths:
        r = run(["git", "cat-file", "--batch"], input="".join(f"{ref}:{p}\n" for p in paths).encode("utf-8"),
                check=True).stdout
        pos = 0
        for p in paths:
            nl = r.index(b"\n", pos)
            head = r[pos:nl].split()
            if len(head) != 3 or head[1] != b"blob":
                raise Fail(f"git から {p} を読めません")
            size = int(head[2])
            files[p[5:]] = r[nl + 1:nl + 1 + size]
            pos = nl + 1 + size + 1
    saved = ""
    if "vault/index.json" in files:
        try:
            saved = json.loads(files["vault/index.json"]).get("saved_at", "")
        except Exception:
            pass
    return Source(label, files, saved)


def remote_sha(branch):
    """GitHub のブランチの先頭。無ければ None。確かめられなければ Fail。"""
    for i in range(5):
        r = run(["git", "ls-remote", "--exit-code", "origin", f"refs/heads/{branch}"])
        if r.returncode == 0:
            return r.stdout.decode().split()[0]
        if r.returncode == 2:
            return None
        time.sleep(10)
    raise Fail(f"GitHub の {branch} ブランチを確かめられませんでした")


def fetch_site_branch():
    sha = remote_sha(BRANCH)
    if sha:
        run(["git", "fetch", "-q", "--no-tags", "origin", f"+refs/heads/{BRANCH}:refs/remotes/origin/{BRANCH}"],
            check=True)
        if out(["git", "rev-parse", f"refs/remotes/origin/{BRANCH}"]) != sha:
            raise Fail(f"{BRANCH} ブランチを取り込めませんでした")
    return sha


def guard(remote: Source, pw, site=False):
    """保存先(remote)より記録が減らないか。減るなら、その理由のリストを返す（中身は出さない）。"""
    local = local_records()
    probs = []
    try:
        rrec = remote.records(pw)
    except Locked as e:
        log(f"保存先（{remote.label}）の金庫を今の合言葉で開けないので（{e}）、日付だけ比べます")
        lost = remote.vault_days() - record_days(local)
        if lost:
            probs.append(f"記録の日が {len(lost)}日 減る（{min(lost)} など）")
        rrec = {}
    for p, b in sorted(rrec.items()):
        lb = local.get(p)
        if lb is None:
            probs.append(f"{p} が無くなる")
        elif nlines(lb) < nlines(b):
            probs.append(f"{p} の行が減る（{nlines(b)}→{nlines(lb)}）")
    if site:
        lost = site_days(remote.files) - site_days(read_files(DATA, ("site",)))
        if lost:
            probs.append(f"ウェブ用データの日が {len(lost)}日 減る（{min(lost)} など）")
    return probs


# ---------- 回の始め: 記録を戻す ----------

def restore():
    gd = git_dir()
    ok_mark, ng_mark = gd / "records-restored", gd / "records-restore-failed"
    ok_mark.unlink(missing_ok=True)
    ng_mark.unlink(missing_ok=True)
    pw = sitecrypt.password()
    try:
        cands = []
        if (SNAP / "STAMP.json").exists():
            cands.append(dir_source(SNAP, "キャッシュ"))
        if fetch_site_branch():
            cands.append(tree_source(f"refs/remotes/origin/{BRANCH}", f"{BRANCH} ブランチ"))
        cands.append(tree_source("HEAD", "main"))
        readable, locked = [], []
        for rank, c in enumerate(cands):   # 同じ時刻なら キャッシュ → site-data → main の順
            if c.empty():
                log(f"・{c.label}: 記録なし")
                continue
            try:
                recs = c.records(pw)
                readable.append((c.saved_at, -rank, c, recs))
                log(f"・{c.label}: {len(recs)}ファイル（保存 {c.saved_at or '時刻なし＝金庫を作る前の形'}）")
            except Locked as e:
                locked.append(c)
                log(f"・{c.label}: 金庫を開けません（{e}。保存 {c.saved_at or '?'}）")
        readable.sort(key=lambda x: x[:2], reverse=True)
        newest_locked = max((c.saved_at for c in locked), default=None)
        if newest_locked is not None and (not readable or newest_locked > readable[0][0]):
            raise Fail("いちばん新しい記録が金庫の中にあるのに、今の合言葉で開けません。"
                       "合言葉（SITE_PASSWORD）を変えた・消したなら元に戻してください。"
                       "古い記録や空から始めると記録が欠けるので、ここで止めます")
        if not readable:
            log("記録はまだありません（はじめから）")
        else:
            merged = {}   # ファイルごとに、いちばん行の多いもの（同じなら新しい置き場のもの）
            for _, _, c, recs in readable:
                for p, b in recs.items():
                    if p not in merged or nlines(b) > nlines(merged[p][0]):
                        merged[p] = (b, c.label)
            changed = 0
            for p, (b, _) in merged.items():
                dest = DATA / p
                if not dest.exists() or dest.read_bytes() != b:
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(b)
                    changed += 1
            site = {}
            for _, _, c, _ in reversed(readable):   # ウェブ用データは新しい置き場のものを優先
                site.update({p: b for p, b in c.files.items() if p.startswith("site/")})
            for p, b in site.items():
                dest = DATA / p
                if not dest.exists() or dest.read_bytes() != b:
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(b)
            best = readable[0][2]
            from_other = sum(1 for b, lbl in merged.values() if lbl != best.label)
            log(f"記録を戻しました: 戻し元＝{best.label}（保存 {best.saved_at or '時刻なし'}）"
                f"{f'＋ほかの置き場から {from_other}ファイル' if from_other else ''}・"
                f"{len(merged)}ファイル・{len(record_days({p: v[0] for p, v in merged.items()}))}日分"
                f"（書き直し {changed}）・ウェブ用 {len(site_days(site))}日分")
        ok_mark.write_text(now_stamp())
        return 0
    except (Fail, Locked) as e:
        ng_mark.write_text(str(e))
        gh("error", "記録を戻せません", str(e))
        return 1


# ---------- 保存 ----------

def make_tree(paths):
    idx = git_dir() / "site-data.index"
    idx.unlink(missing_ok=True)
    env = {**os.environ, "GIT_INDEX_FILE": str(idx)}
    run(["git", "read-tree", "--empty"], env=env, check=True)
    run(["git", "add", "-f", "--"] + paths, env=env, check=True)
    tree = out(["git", "write-tree"], env=env)
    idx.unlink(missing_ok=True)
    return tree


def push_site(pw):
    """data/site と data/vault を、site-data ブランチに1コミットだけの形で上書きする。"""
    # 合言葉が無いときは金庫を出さない（前の合言葉の古い金庫が残ると、次の回でそれを開けずに止まってしまうため）
    paths = [p for p in ("data/site", "data/vault") if (ROOT / p).is_dir() and any((ROOT / p).iterdir())
             and (pw or p != "data/vault")]
    if not paths:
        log("ウェブ用データがまだ無いので site-data には出しません")
        return True
    for i in range(3):
        sha = fetch_site_branch()
        if sha:
            probs = guard(tree_source(f"refs/remotes/origin/{BRANCH}", f"{BRANCH} ブランチ"), pw, site=True)
            if probs:
                gh("error", "site-data に保存しません", "今の記録が保存先より少ないため: " + "／".join(probs[:5]))
                return False
        tree = make_tree(paths)
        if sha and out(["git", "rev-parse", f"{sha}^{{tree}}"]) == tree:
            log(f"{BRANCH}: 前と同じなので送りません")
            return True
        commit = out(["git", "commit-tree", tree, "-m", f"ウェブ用データと記録の金庫（最新だけ） {dt.datetime.now(JST):%Y-%m-%d %H:%M}"])
        r = run(["git", "push", "-q", f"--force-with-lease=refs/heads/{BRANCH}:{sha or ''}", "origin",
                 f"{commit}:refs/heads/{BRANCH}"])
        if r.returncode == 0 and remote_sha(BRANCH) == commit:
            log(f"{BRANCH} に保存しました")
            return True
        log(f"{BRANCH} への保存に失敗（{i + 1}回目）: {r.stderr.decode('utf-8', 'replace').strip()[:200]}")
        time.sleep(5)
    return False


def push_main(pw, add, remove=()):
    """main にコミットして送る。送る前に、main の記録より減らないかを確かめる（取り込みのたびに確かめ直す）。"""
    run(["git", "fetch", "-q", "--no-tags", "origin", "+refs/heads/main:refs/remotes/origin/main"], check=True)
    probs = guard(tree_source("refs/remotes/origin/main", "main"), pw)
    if probs:
        gh("error", "main に保存しません", "今の記録が main より少ないため: " + "／".join(probs[:5]))
        return False
    add = [p for p in add if (ROOT / p).exists()]
    if add:
        run(["git", "add", "-f", "--"] + add, check=True)
    if remove:
        run(["git", "rm", "-r", "-q", "--cached", "--ignore-unmatch", "--"] + list(remove), check=True)
    if run(["git", "diff", "--cached", "--quiet"]).returncode == 0:
        log("main: 変わったところが無いので送りません")
        return True
    run(["git", "commit", "-q", "-m", f"予想記録 {dt.datetime.now(JST):%Y-%m-%d %H:%M}"], check=True)
    for i in range(3):
        if run(["git", "pull", "-q", "--rebase", "--autostash"]).returncode == 0:
            probs = guard(tree_source("refs/remotes/origin/main", "main"), pw)
            if probs:
                gh("error", "main に保存しません", "今の記録が main より少ないため: " + "／".join(probs[:5]))
                return False
            if run(["git", "push", "-q"]).returncode == 0:
                log("main に保存しました")
                return True
        run(["git", "rebase", "--abort"])   # 途中で止まった取り込みを片付けてからやり直す
        time.sleep(5)
    return False


def save(final):
    gd = git_dir()
    if (gd / "records-restore-failed").exists():
        gh("error", "記録の保存", "この回は記録を戻せなかったので、保存しません（空や古い記録で上書きしないため）")
        return 1
    if not (gd / "records-restored").exists():
        gh("warning", "記録の保存", "記録を戻す手順を通っていない回です。保存先より記録が減らないことを確かめてから保存します")
    pw = sitecrypt.password()
    legacy = not pw or os.environ.get("RECORDS_MODE") == "legacy"
    ok = True
    vault_ok = True
    vault_saved = ""
    if pw:
        try:
            pack(pw)
            vault_saved = load_json(VAULT / "index.json").get("saved_at", "")
        except Exception as e:
            gh("error", "記録の金庫", f"金庫を作れませんでした（{type(e).__name__}: {e}）")
            ok = vault_ok = False
    try:
        snapshot(vault_saved)
    except Exception as e:
        gh("warning", "記録の写し", f"キャッシュ用の写しを作れませんでした（{type(e).__name__}: {e}）")
    site_ok = vault_ok and push_site(pw)
    ok = ok and site_ok
    if legacy:   # 前と同じ形（元の記録をそのまま main へ。10分ごと）。合言葉があれば金庫も、無ければ古い金庫は外す
        ok = push_main(pw, RECORD_PATHS + ["data/site"] + (["data/vault"] if pw else []),
                       [] if pw else ["data/vault"]) and ok
    elif final:  # 回の最後だけ、金庫を main へ。元の記録とウェブ用データは main から外す（作業フォルダには残す）
        if site_ok:
            ok = push_main(pw, ["data/vault"], RECORD_PATHS + ["data/site"]) and ok
        else:
            gh("error", "main に保存しません", "site-data に保存できなかったので、main もそのままにします")
            ok = False
    return 0 if ok else 1


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    try:
        if cmd == "restore":
            return restore()
        if cmd == "save":
            return save(len(sys.argv) > 2 and sys.argv[2] == "final")
    except (Fail, Locked) as e:
        gh("error", "記録の保存", str(e))
        return 1
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())
