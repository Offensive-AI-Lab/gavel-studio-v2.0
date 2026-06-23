"""JWT + password hashing for the central server."""
import hashlib
import os
import time
from collections import OrderedDict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import BoundedSemaphore, Lock
from typing import Optional

from dotenv import load_dotenv
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from passlib.context import CryptContext

# Load .env early so SECRET_KEY below picks it up even if auth.py is
# imported before db.py.
_env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(dotenv_path=_env_path)

# The central server is the SOLE auth authority — it both signs and verifies
# tokens, and it is the only component that holds this key. Fail loudly rather
# than fall back to a publicly-known default (which would let anyone forge
# tokens). Local backends never hold this; they verify via GET /auth/verify.
SECRET_KEY = os.getenv("JWT_SECRET_KEY")
if not SECRET_KEY:
    raise RuntimeError(
        "JWT_SECRET_KEY is not set. The central server requires it to sign and "
        "verify auth tokens. Set a strong random value in the environment."
    )
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_HOURS = 24 * 7  # 7 days — local backend stays signed in across restarts

# argon2 tuned to OWASP-minimum (m=19 MiB, t=2, p=1) instead of passlib's
# ~64 MiB default. Under a login burst the default would each grab ~64 MiB and
# OOM a small instance; ~19 MiB is still strong and far lighter. Existing
# password hashes carry their own params, so they keep verifying — only NEW
# hashes use these settings.
pwd_context = CryptContext(
    schemes=["argon2"],
    deprecated="auto",
    argon2__memory_cost=19456,
    argon2__time_cost=2,
    argon2__parallelism=1,
)
# Bound how many argon2 hashes run at once so a flood of logins/registers can't
# exhaust CPU/RAM and crash the box. Excess requests wait briefly for a slot
# rather than all computing simultaneously.
_HASH_CONCURRENCY = max(1, int(os.getenv("ARGON2_MAX_CONCURRENCY", "4")))
_hash_semaphore = BoundedSemaphore(_HASH_CONCURRENCY)
_bearer = HTTPBearer()


# ---------------------------------------------------------------------------
# JWT decode cache.
#
# Every authed endpoint runs decode_access_token() per request. HMAC
# verification is cheap (microseconds) but JSON parsing + datetime
# conversion + python-jose's claim validation add up to ~0.5-2ms when
# you're handling many requests in a row — and the same client tends
# to reuse the same token for thousands of requests. Caching the
# decoded payload by token cuts the repeat-cost to a dict lookup.
#
# Why this is SAFE (the bar we have to clear is "this must not let
# a stolen, tampered, or expired token through"):
#
#  1. We cache the SHA-256 hash of the token, not the token string —
#     so a memory dump of central doesn't leak raw JWTs.
#  2. Cache hits STILL re-verify the `exp` claim against the current
#     wall-clock time. An entry that was valid when cached but has
#     since expired is rejected on the hit path, then evicted.
#  3. Cache hits are ONLY checked AFTER a successful initial decode.
#     A tampered token never reaches the cache because the signature
#     check fails before we hash it.
#  4. The cache key includes the SECRET_KEY's fingerprint — so if the
#     operator rotates the JWT secret, every previously-cached entry
#     becomes a miss automatically (no risk of accepting a token
#     signed with the old secret after rotation).
#  5. Bounded LRU (`maxsize=512`) caps memory. At ~64 bytes per entry
#     that's <40 KB; an attacker can't OOM us by spraying tokens.
#  6. Cache hits also enforce a short MAX_CACHE_TTL of 60 seconds —
#     even if a token's exp is days away, we never trust a cached
#     entry older than a minute. This bounds the worst-case window
#     for any future revocation feature we add (e.g., a denylist).
#
# To turn the cache off, set `CACHE_DECODE_TOKENS=0` in central's .env.
_DECODE_CACHE_ENABLED = os.getenv("CACHE_DECODE_TOKENS", "1") != "0"
_DECODE_CACHE_MAXSIZE = 512
_DECODE_CACHE_TTL = 60.0  # seconds — see point (6) above

_decode_cache: "OrderedDict[bytes, tuple[dict, float]]" = OrderedDict()
_decode_cache_lock = Lock()

# Mix the secret's fingerprint into the cache key so secret rotation
# invalidates everything automatically.
_SECRET_FINGERPRINT = hashlib.sha256(SECRET_KEY.encode("utf-8")).digest()[:8]


def _cache_key(token: str) -> bytes:
    h = hashlib.sha256()
    h.update(_SECRET_FINGERPRINT)
    h.update(b"|")
    h.update(token.encode("utf-8"))
    return h.digest()


def _exp_still_valid(payload: dict) -> bool:
    """python-jose validates exp at decode time, but cache HITs skip
    decoding. Re-check here so an entry that was valid when cached
    isn't returned after expiration."""
    exp = payload.get("exp")
    if exp is None:
        return False
    try:
        return float(exp) > time.time()
    except (TypeError, ValueError):
        return False


def hash_password(password: str) -> str:
    with _hash_semaphore:
        return pwd_context.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    with _hash_semaphore:
        return pwd_context.verify(plain, hashed)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(hours=ACCESS_TOKEN_EXPIRE_HOURS))
    to_encode["exp"] = expire
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def decode_access_token(token: str) -> dict:
    """Verify + decode a JWT. Fast path on cache hit (still re-checks
    exp). Slow path is the full python-jose decode + signature verify."""
    if _DECODE_CACHE_ENABLED:
        key = _cache_key(token)
        with _decode_cache_lock:
            cached = _decode_cache.get(key)
            if cached is not None:
                payload, cached_at = cached
                if (time.time() - cached_at) < _DECODE_CACHE_TTL and _exp_still_valid(payload):
                    # LRU touch: move to end so frequently-used tokens
                    # stay hot.
                    _decode_cache.move_to_end(key)
                    return payload
                # Stale or expired — evict and fall through to a real
                # decode. The full decode will raise on a tampered or
                # truly-expired token, which is what we want.
                _decode_cache.pop(key, None)

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if _DECODE_CACHE_ENABLED:
        with _decode_cache_lock:
            _decode_cache[_cache_key(token)] = (payload, time.time())
            # Trim to maxsize (LRU eviction).
            while len(_decode_cache) > _DECODE_CACHE_MAXSIZE:
                _decode_cache.popitem(last=False)

    return payload


def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(_bearer)) -> int:
    payload = decode_access_token(credentials.credentials)
    user_id = payload.get("sub")
    if user_id is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token missing subject")
    return int(user_id)
