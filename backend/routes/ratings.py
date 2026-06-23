"""Ratings: proxied through the central server.

Local backend keeps no ratings tables. Every endpoint here is a thin
wrapper around `services.central_server.*` — the central server owns
the `ratings` and `asset_ratings_summary` tables.

Self-rating check stays on the local side because it needs to read
`created_by_username` from the LOCAL rules/cognitive_elements cache.
"""
from typing import Literal, Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from functools import lru_cache

from services import central_server
from services.central_server import CentralServerError
from utils.auth import get_current_user
from utils.PostgreSQL import execute_query_dict

router = APIRouter()
_bearer = HTTPBearer(auto_error=False)


# Username lookup cache.
#
# Why safe: usernames are PERMANENT by design (Phase 1 — see
# Register.jsx "username is permanent" notice and the user_scripts
# validation that has no UPDATE path). A given user_id will always map
# to the same username string for the lifetime of the account.
#
# Why bounded: maxsize=256 fits any plausible interactive session
# without memory growth. Older entries fall out under LRU pressure.
#
# Why no TTL: there's no scenario short of a manual DB UPDATE where
# the answer changes. If you ever add a username-change flow, drop
# the cache or add an invalidation hook here.
#
# Why no PII risk: usernames are public (visible on every rule/CE
# card via the "by @username" link). Nothing sensitive flows through
# this cache.
@lru_cache(maxsize=256)
def _username_for_user_id(user_id: int) -> Optional[str]:
    rows = execute_query_dict(
        "SELECT username FROM users WHERE user_id = %s", (user_id,)
    ) or []
    return rows[0]["username"] if rows else None


def _get_token(creds: HTTPAuthorizationCredentials = Depends(_bearer)) -> str:
    if not creds:
        raise HTTPException(status_code=401, detail="Missing bearer token")
    return creds.credentials


def _raise_for_central(err: CentralServerError):
    raise HTTPException(status_code=err.status_code, detail=str(err))


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class RatingRequest(BaseModel):
    asset_type: Literal["rule", "ce"]
    asset_public_id: str = Field(..., min_length=1, max_length=255)
    score: int = Field(..., ge=1, le=5)


class RatingSummary(BaseModel):
    asset_type: str
    asset_public_id: str
    rating_count: int
    rating_avg: Optional[float] = None
    your_score: Optional[int] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _artifact_owner_username(asset_type: str, public_id: str) -> Optional[str]:
    """Look up the artifact's creator in the LOCAL cache. The local
    backend has the rules/CEs synced from HF; the central server doesn't
    keep them."""
    table = "rules" if asset_type == "rule" else "cognitive_elements"
    rows = execute_query_dict(
        f"SELECT created_by_username FROM {table} WHERE public_id = %s",
        (public_id,),
    ) or []
    return rows[0]["created_by_username"] if rows else None


def _shape(asset_type: str, public_id: str, central_resp: dict) -> RatingSummary:
    return RatingSummary(
        asset_type=asset_type,
        asset_public_id=public_id,
        rating_count=central_resp.get("rating_count", 0),
        rating_avg=central_resp.get("rating_avg"),
        your_score=central_resp.get("your_score"),
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/", response_model=RatingSummary)
def upsert_rating(
    req: RatingRequest,
    user_id: int = Depends(get_current_user),
    token: str = Depends(_get_token),
):
    """Rate an artifact 1-5. Self-rating is rejected locally before the
    HTTP hop saves a round trip."""
    owner = _artifact_owner_username(req.asset_type, req.asset_public_id)
    if owner is None:
        raise HTTPException(
            status_code=404,
            detail=f"No {req.asset_type} with public_id {req.asset_public_id!r}",
        )

    # Self-rating guard. We need the authed user's username; the JWT only
    # carries user_id, so look it up in the local mirror (cached — see
    # _username_for_user_id docstring for why caching is safe here).
    my_username = _username_for_user_id(user_id)
    if my_username and my_username.lower() == (owner or "").lower():
        raise HTTPException(status_code=400, detail="You can't rate your own contribution.")

    try:
        # Pass the artifact's creator so central can update
        # user_ratings_summary for the author (the profile page's
        # "avg rating received" reads from that summary table).
        resp = central_server.rate(
            token, req.asset_type, req.asset_public_id, req.score,
            created_by_username=owner,
        )
    except CentralServerError as err:
        _raise_for_central(err)

    return _shape(req.asset_type, req.asset_public_id, resp)


@router.delete("/{asset_type}/{asset_public_id}", response_model=RatingSummary)
def delete_rating(
    asset_type: Literal["rule", "ce"],
    asset_public_id: str,
    token: str = Depends(_get_token),
):
    # Look up the creator so central can decrement their summary.
    owner = _artifact_owner_username(asset_type, asset_public_id)
    try:
        resp = central_server.delete_rating(token, asset_type, asset_public_id, created_by_username=owner)
    except CentralServerError as err:
        _raise_for_central(err)
    return _shape(asset_type, asset_public_id, resp)


@router.get("/{asset_type}/{asset_public_id}", response_model=RatingSummary)
def get_rating(
    asset_type: Literal["rule", "ce"],
    asset_public_id: str,
    token: str = Depends(_get_token),
):
    try:
        resp = central_server.get_rating_summary(token, asset_type, asset_public_id)
    except CentralServerError as err:
        _raise_for_central(err)
    return _shape(asset_type, asset_public_id, resp)
