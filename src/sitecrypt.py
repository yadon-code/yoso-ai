"""ウェブアプリ用データと記録の金庫に「合言葉」でカギをかける。

環境変数 SITE_PASSWORD（GitHub の Secret）に合言葉があるときだけ使う。
合言葉から鍵を作り（PBKDF2-SHA256 20万回）、AES-GCM で暗号化する。ウェブアプリ側は同じ合言葉で元に戻す。
ファイルの中身: salt 16バイト + nonce 12バイト + 暗号文（ウェブ用は全体を base64 にした文字）

同じ中身なら同じ暗号文になるようにしている（中身が変わらないファイルを保存し直しても、リポジトリが太らない）:
- salt はこのアプリで決まった値（合言葉は1つだけなので、ファイルごとに変える意味がない）
- nonce は「鍵」と「ファイルの名前＋中身」から作る（HMAC-SHA256）。同じ nonce になるのは名前も中身も同じときだけなので、
  AES-GCM の「同じ nonce を別の中身に使ってはいけない」を守れる。わかるのは「前と同じ中身かどうか」だけ。
読むときはファイルの salt と nonce をそのまま使うので、前の作り方（salt・nonce がでたらめ）のファイルも読める。
"""
import base64
import hashlib
import hmac
import os

ITERATIONS = 200_000
SALT = hashlib.sha256(b"yoso-ai sitecrypt salt v1").digest()[:16]
_KEYS = {}


def password():
    return os.environ.get("SITE_PASSWORD") or None


def _key(pw: str, salt: bytes) -> bytes:
    """合言葉から鍵を作る（重いので、同じ合言葉と salt なら1回だけ）。"""
    k = (hashlib.sha256(pw.encode("utf-8")).digest(), bytes(salt))
    if k not in _KEYS:
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
        _KEYS[k] = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=bytes(salt),
                              iterations=ITERATIONS).derive(pw.encode("utf-8"))
    return _KEYS[k]


def seal(data: bytes, pw: str, name: str = "") -> bytes:
    """バイト列を暗号化する。name はファイルの名前（同じ中身でも名前がちがえば別の暗号文になる）。"""
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    key = _key(pw, SALT)
    nkey = hmac.new(key, b"yoso-ai nonce v1", hashlib.sha256).digest()
    nonce = hmac.new(nkey, name.encode("utf-8") + b"\0" + data, hashlib.sha256).digest()[:12]
    return SALT + nonce + AESGCM(key).encrypt(nonce, data, None)


def unseal(raw: bytes, pw: str) -> bytes:
    """seal の逆。合言葉がちがう・壊れているときは例外（cryptography の InvalidTag など）。"""
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    salt, nonce, ct = raw[:16], raw[16:28], raw[28:]
    return AESGCM(_key(pw, salt)).decrypt(nonce, ct, None)


def encrypt(text: str, pw: str, name: str = "") -> str:
    return base64.b64encode(seal(text.encode("utf-8"), pw, name)).decode("ascii")


def decrypt(blob: str, pw: str) -> str:
    return unseal(base64.b64decode(blob), pw).decode("utf-8")
