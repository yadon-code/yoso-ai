"""ウェブアプリ用データに「合言葉」でカギをかける。

環境変数 SITE_PASSWORD（GitHub の Secret）に合言葉があるときだけ使う。
合言葉から鍵を作り（PBKDF2）、AES-GCM で暗号化する。ウェブアプリ側は同じ合言葉で元に戻す。
ファイルの中身: base64( salt 16バイト + nonce 12バイト + 暗号文 )
"""
import base64
import os

ITERATIONS = 200_000


def password():
    return os.environ.get("SITE_PASSWORD") or None


def encrypt(text: str, pw: str) -> str:
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    salt, nonce = os.urandom(16), os.urandom(12)
    key = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=ITERATIONS).derive(pw.encode("utf-8"))
    ct = AESGCM(key).encrypt(nonce, text.encode("utf-8"), None)
    return base64.b64encode(salt + nonce + ct).decode("ascii")


def decrypt(blob: str, pw: str) -> str:
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    raw = base64.b64decode(blob)
    salt, nonce, ct = raw[:16], raw[16:28], raw[28:]
    key = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=ITERATIONS).derive(pw.encode("utf-8"))
    return AESGCM(key).decrypt(nonce, ct, None).decode("utf-8")
