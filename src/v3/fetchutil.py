"""控えめに取るための道具（全体で1秒あたりの回数に上限をかける）。"""
import threading
import time
import urllib.error
import urllib.request

UA = "Mozilla/5.0 (compatible; personal-research)"


class RateLimiter:
    def __init__(self, per_sec: float):
        self.gap = 1.0 / per_sec
        self.lock = threading.Lock()
        self.next = 0.0

    def wait(self):
        with self.lock:
            now = time.monotonic()
            t = max(now, self.next)
            self.next = t + self.gap
        if t > now:
            time.sleep(t - now)


ERR = "!ERR"


def get_text(url, limiter, ok_prefix=None, encoding="utf-8", tries=3, timeout=20):
    """(text | None | ERR)。None=そのページは無い（404 や HTML のエラーページ）。ERR=通信の失敗（あとで取り直す）。"""
    for attempt in range(tries):
        limiter.wait()
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                t = r.read().decode(encoding, "replace")
            if ok_prefix is not None and not t.startswith(ok_prefix):
                return None
            return t
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            time.sleep(5 * (attempt + 1))
        except Exception:
            time.sleep(5 * (attempt + 1))
    return ERR
