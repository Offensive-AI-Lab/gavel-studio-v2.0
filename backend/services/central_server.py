"""HTTP client for the GAVEL central server.

The central server owns:
    * users + auth
    * ratings + per-asset/user aggregates
    * bookmarks (by public_id, portable across machines)
    * HuggingFace write proxy (the HF token lives only on the server)

This module is the ONLY place in the local backend that knows about the
central server's URL. Swap `CENTRAL_SERVER_URL` (local dev → Render →
university server) and nothing else changes.

Each function returns a parsed JSON dict, or raises CentralServerError
on HTTP / network failures. Routes catch this and convert to
HTTPException for the frontend.
"""
import base64
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from dotenv import load_dotenv

env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

CENTRAL_SERVER_URL = os.getenv("CENTRAL_SERVER_URL", "").rstrip("/")
_TIMEOUT = httpx.Timeout(30.0, connect=10.0)

# ---------------------------------------------------------------------------
# Shared HTTP client.
#
# Why module-level: before this change every _request() built a fresh
# httpx.Client and tore it down on exit — so each call paid the cost of
# a new TCP three-way handshake plus (on HTTPS) a full TLS handshake.
# Over WAN that's 100-400ms wasted per request. With one shared client
# kept alive for the process lifetime, the first request pays the
# handshake, and every subsequent request reuses the open socket — a
# 5-50x speedup on repeated calls.
#
# Tuning rationale:
#   * max_keepalive_connections=10 — enough headroom for a handful of
#     concurrent flows (publish + sync poll + a profile page load
#     racing). On a single-user dev machine 2-3 is more than enough.
#   * keepalive_expiry=60 — most central-side endpoints sit idle longer
#     than the default 5s; holding sockets open for a minute amortises
#     reconnect cost over user think-time.
#   * http2 (optional) — lets multiple requests multiplex over one
#     connection. We probe for the `h2` package at import time and
#     enable HTTP/2 only when it's available; otherwise we fall back
#     to HTTP/1.1 + keep-alive (which already covers 80% of the win).
#     To turn http2 on: `pip install h2` and restart.
#   * follow_redirects=True — central uses path-normalised URLs;
#     trailing-slash redirects would otherwise lose a round-trip.
try:
    import h2 as _h2  # noqa: F401  — probe only
    _HTTP2 = True
except ImportError:
    _HTTP2 = False

_client = httpx.Client(
    timeout=_TIMEOUT,
    http2=_HTTP2,
    follow_redirects=True,
    limits=httpx.Limits(
        max_keepalive_connections=10,
        max_connections=20,
        keepalive_expiry=60.0,
    ),
)

# Per-process logger so the timing line lands in uvicorn's log stream
# rather than print()-ing through stdout (which uvicorn buffers).
logger = logging.getLogger("central-rpc")
# Default to INFO so the timing breadcrumbs show up in normal dev logs.
# If they become noisy in production, flip to logging.DEBUG and they
# disappear by default.
if not logger.handlers:
    logger.setLevel(logging.INFO)


class CentralServerError(Exception):
    """Raised when the central server is unreachable or returns an error."""

    def __init__(self, message: str, status_code: int = 500, payload: Optional[dict] = None):
        super().__init__(message)
        self.status_code = status_code
        self.payload = payload or {}


def is_enabled() -> bool:
    return bool(CENTRAL_SERVER_URL)


def _headers(token: Optional[str]) -> Dict[str, str]:
    h = {"Content-Type": "application/json"}
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


def _request(method: str, path: str, *, token: Optional[str] = None, json: Any = None, params: Any = None) -> Any:
    if not CENTRAL_SERVER_URL:
        raise CentralServerError("CENTRAL_SERVER_URL is not configured", status_code=503)
    url = f"{CENTRAL_SERVER_URL}{path}"
    t0 = time.perf_counter()
    try:
        resp = _client.request(method, url, headers=_headers(token), json=json, params=params)
    except httpx.RequestError as e:
        elapsed = (time.perf_counter() - t0) * 1000
        logger.warning("FAIL %s %s after %.0fms: %s", method, path, elapsed, e)
        raise CentralServerError(f"Central server unreachable: {e}", status_code=502)

    elapsed = (time.perf_counter() - t0) * 1000
    # One-line breadcrumb so latency regressions are obvious without a
    # profiler. `> 200ms` flagged loud because that's the threshold
    # where a UI starts feeling sluggish.
    tag = " SLOW" if elapsed > 200 else ""
    logger.info("%s %s -> %s in %.0fms%s", method, path, resp.status_code, elapsed, tag)

    if resp.status_code >= 400:
        try:
            payload = resp.json()
        except Exception:
            payload = {"detail": resp.text}
        detail = payload.get("detail") if isinstance(payload, dict) else str(payload)
        raise CentralServerError(detail or f"HTTP {resp.status_code}", status_code=resp.status_code, payload=payload)

    if resp.headers.get("content-type", "").startswith("application/json"):
        return resp.json()
    return resp.text


def close() -> None:
    """Tear down the shared client. Safe to call multiple times. Useful
    for tests that monkey-patch CENTRAL_SERVER_URL and want a clean
    socket state; not strictly needed in normal app shutdown since
    Python closes sockets on process exit anyway."""
    try:
        _client.close()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

def register(username: str, email: str, password: str) -> dict:
    return _request("POST", "/auth/register", json={"username": username, "email": email, "password": password})


def login(email: str, password: str) -> dict:
    return _request("POST", "/auth/login", json={"email": email, "password": password})


def verify_token(token: str) -> int:
    """Validate a bearer token with the central server (the single auth
    authority) and return the authenticated user_id.

    The local backend uses this instead of decoding the JWT itself, so it never
    needs the signing secret — no local operator can forge tokens. Raises
    CentralServerError (status 401) for an invalid/expired token.
    """
    resp = _request("GET", "/auth/verify", token=token)
    return int(resp["user_id"])


def get_me(token: str) -> dict:
    return _request("GET", "/auth/me", token=token)


def update_me(token: str, *, display_name: Optional[str] = None, bio: Optional[str] = None) -> dict:
    body: Dict[str, Any] = {}
    if display_name is not None:
        body["display_name"] = display_name
    if bio is not None:
        body["bio"] = bio
    return _request("PATCH", "/auth/me", token=token, json=body)


def mark_tutorial_seen(token: str) -> dict:
    return _request("PUT", "/auth/tutorial-seen", token=token)


def get_user_by_id(token: str, user_id: int) -> dict:
    return _request("GET", f"/auth/users/{user_id}", token=token)


def get_users_by_username(usernames: List[str]) -> List[dict]:
    """No auth required — used to populate the local users mirror after sync."""
    if not usernames:
        return []
    params = {"usernames": ",".join(usernames)}
    resp = _request("GET", "/auth/users/by-username", params=params)
    return resp.get("users", []) if isinstance(resp, dict) else []


def get_team_users() -> List[dict]:
    """All users with is_team=TRUE on the central server. No auth required.
    Used at startup to mirror the seed-library author(s) into the local
    users table so attribution and contributions queries resolve on a
    fresh local DB before any user has logged in."""
    resp = _request("GET", "/auth/team-users")
    return resp.get("users", []) if isinstance(resp, dict) else []


def record_publish_attribution(token: str, asset_type: str, published_at: Optional[str] = None) -> dict:
    """Bump the authed user's contribution_count_* on the central
    server. Called after a successful HF publish."""
    body: Dict[str, Any] = {"asset_type": asset_type}
    if published_at:
        body["published_at"] = published_at
    return _request("POST", "/auth/publish-attribution", token=token, json=body)


# ---------------------------------------------------------------------------
# User discovery
# ---------------------------------------------------------------------------

def get_profile(username: str) -> dict:
    return _request("GET", f"/users/profile/{username}")


def search_users(*, q: Optional[str] = None, page: int = 1, page_size: int = 20) -> dict:
    params: Dict[str, Any] = {"page": page, "page_size": page_size}
    if q:
        params["q"] = q
    return _request("GET", "/users/search", params=params)


def leaderboard(*, sort: str = "rating", page: int = 1, page_size: int = 20, min_ratings: int = 0) -> dict:
    return _request("GET", "/users/leaderboard", params={"sort": sort, "page": page, "page_size": page_size, "min_ratings": min_ratings})


# ---------------------------------------------------------------------------
# Ratings
# ---------------------------------------------------------------------------

def rate(token: str, asset_type: str, asset_public_id: str, score: int, created_by_username: str = None) -> dict:
    payload = {"asset_type": asset_type, "asset_public_id": asset_public_id, "score": score}
    if created_by_username:
        payload["created_by_username"] = created_by_username
    return _request("POST", "/ratings", token=token, json=payload)


def delete_rating(token: str, asset_type: str, asset_public_id: str, created_by_username: str = None) -> dict:
    params = {}
    if created_by_username:
        params["created_by_username"] = created_by_username
    return _request("DELETE", f"/ratings/{asset_type}/{asset_public_id}", token=token, params=params)


def get_rating_summary(token: str, asset_type: str, asset_public_id: str) -> dict:
    return _request("GET", f"/ratings/{asset_type}/{asset_public_id}", token=token)


# ---------------------------------------------------------------------------
# Bookmarks
# ---------------------------------------------------------------------------

def add_bookmark(token: str, asset_type: str, public_id: str) -> dict:
    return _request("POST", "/bookmarks", token=token,
                    json={"asset_type": asset_type, "public_id": public_id})


def remove_bookmark(token: str, asset_type: str, public_id: str) -> dict:
    return _request("DELETE", f"/bookmarks/{asset_type}/{public_id}", token=token)


def list_bookmarks(token: str, asset_type: str) -> List[dict]:
    resp = _request("GET", f"/bookmarks/{asset_type}", token=token)
    return resp.get("bookmarks", []) if isinstance(resp, dict) else []


# ---------------------------------------------------------------------------
# HuggingFace write proxy
# ---------------------------------------------------------------------------

def hf_head_sha(token: str) -> Optional[str]:
    resp = _request("GET", "/hf/head-sha", token=token)
    return resp.get("sha") if isinstance(resp, dict) else None


def hf_commit(token: str, *, operations: List[Dict[str, bytes]], commit_message: str,
              parent_commit: Optional[str] = None) -> dict:
    """Commit a batch of files to HF via the central server.

    `operations` is a list of {"path": str, "content": bytes}. We base64-
    encode bytes for transport.
    """
    encoded = [
        {"path": op["path"], "content_b64": base64.b64encode(op["content"]).decode("ascii")}
        for op in operations
    ]
    return _request("POST", "/hf/commit", token=token,
                    json={"operations": encoded, "commit_message": commit_message, "parent_commit": parent_commit})


# NOTE: cluster-job orchestration via the central server was removed — the
# backend talks to the SLURM cluster DIRECTLY (services/cluster_direct.py). The
# central server now only serves community features (auth / bookmarks / ratings /
# HF publish), which never touch the cluster.
