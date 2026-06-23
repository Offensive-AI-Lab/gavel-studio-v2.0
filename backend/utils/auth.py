"""Local-backend authentication.

The CENTRAL server is the single auth authority — it holds the JWT signing key
and is the only component that can issue or cryptographically verify tokens.
This local backend deliberately does NOT hold the signing secret: it validates a
request's bearer token by asking the central server (GET /auth/verify), so a
local operator can never forge tokens for other users. Verifications are cached
briefly to avoid a central round-trip on every request.

`create_access_token` / `decode_access_token` below are TEST-ONLY helpers (the
test suite mints tokens with them, and conftest decodes them in place of a live
central server). They are NEVER used to authenticate real requests.
"""
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt

_bearer = HTTPBearer()

# --- Verified-token cache (token -> user_id) -------------------------------
# Short TTL so a revoked/expired token isn't honored for long; bounded so it
# can't grow without limit; thread-safe.
_VERIFY_TTL = 60.0
_VERIFY_MAX = 4096
_verify_cache: dict = {}
_verify_lock = threading.Lock()


def _cache_get(token: str) -> Optional[int]:
    now = time.monotonic()
    with _verify_lock:
        ent = _verify_cache.get(token)
        if ent is None:
            return None
        user_id, exp = ent
        if exp <= now:
            _verify_cache.pop(token, None)
            return None
        return user_id


def _cache_put(token: str, user_id: int) -> None:
    now = time.monotonic()
    with _verify_lock:
        if len(_verify_cache) >= _VERIFY_MAX:
            for k in [k for k, (_, e) in _verify_cache.items() if e <= now]:
                _verify_cache.pop(k, None)
            if len(_verify_cache) >= _VERIFY_MAX:
                _verify_cache.pop(next(iter(_verify_cache)), None)
        _verify_cache[token] = (user_id, now + _VERIFY_TTL)


def verify_bearer_token(token: str) -> int:
    """Validate a raw bearer token (central server is the authority) and return
    the user_id. Shared by the per-route `get_current_user` dependency AND the
    app-wide auth gate in main.py, so both hit the same verify cache. Raises
    HTTPException(401) for an invalid/expired token, 503 if auth isn't configured
    or the central server is unreachable."""
    cached = _cache_get(token)
    if cached is not None:
        return cached

    from services import central_server
    if not central_server.is_enabled():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication is not configured (CENTRAL_SERVER_URL unset).",
        )
    try:
        user_id = central_server.verify_token(token)
    except central_server.CentralServerError as e:
        if e.status_code in (401, 403):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired token",
                headers={"WWW-Authenticate": "Bearer"},
            )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication service unavailable",
        )

    _cache_put(token, user_id)
    return user_id


def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(_bearer)) -> int:
    """FastAPI dependency — the authenticated user_id, verified by the central
    server (never with a local secret)."""
    return verify_bearer_token(credentials.credentials)


# ---------------------------------------------------------------------------
# TEST-ONLY token helpers — NOT a production auth path. Kept so the test suite
# can mint tokens and conftest can decode them in place of a live central
# server. The secret here is a fixed test constant, never read from the env, so
# the backend has no JWT secret to configure or leak.
# ---------------------------------------------------------------------------
_TEST_SECRET = "gavel-test-only-token-secret"
_TEST_ALG = "HS256"


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(hours=24))
    to_encode["exp"] = expire
    return jwt.encode(to_encode, _TEST_SECRET, algorithm=_TEST_ALG)


def decode_access_token(token: str) -> dict:
    try:
        return jwt.decode(token, _TEST_SECRET, algorithms=[_TEST_ALG])
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )
