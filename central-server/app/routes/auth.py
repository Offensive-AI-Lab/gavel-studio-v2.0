"""Authentication and user-profile endpoints."""
import re
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..utils.auth import (
    create_access_token,
    get_current_user,
    hash_password,
    verify_password,
)
from ..utils.db import execute_dict, execute
from ..utils.rate_limit import account_login_guard, rate_limit, record_login_failure

router = APIRouter(prefix="/auth", tags=["auth"])

# Per-IP throttles on the expensive/abusable auth endpoints. Login/register each
# trigger an argon2 hash, so a flood would otherwise pile up CPU/RAM work.
_rl_login = rate_limit("auth-login", limit=15, window_seconds=60)
_rl_register = rate_limit("auth-register", limit=5, window_seconds=60)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=30)
    email: str = Field(..., max_length=254)
    password: str = Field(..., min_length=8, max_length=128)


class RegisterResponse(BaseModel):
    status: str
    user_id: int
    username: str
    email: str


class LoginRequest(BaseModel):
    email: str = Field(..., max_length=254)
    password: str = Field(..., min_length=8, max_length=128)


class LoginResponse(BaseModel):
    status: str
    token: str
    user_id: int
    username: str
    email: str
    tutorial_seen: bool = False


class UpdateProfileRequest(BaseModel):
    display_name: Optional[str] = Field(None, max_length=255)
    bio: Optional[str] = Field(None, max_length=2000)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _valid_username(u: str) -> bool:
    return bool(re.match(r"^[a-zA-Z0-9_-]{3,30}$", u))


def _valid_email(e: str) -> bool:
    return bool(re.match(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$", e))


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/register", response_model=RegisterResponse)
def register(req: RegisterRequest, _rl=Depends(_rl_register)):
    if not _valid_username(req.username):
        raise HTTPException(status_code=400, detail="Username must be 3-30 chars (alphanumeric, underscore, dash)")
    if not _valid_email(req.email):
        raise HTTPException(status_code=400, detail="Invalid email format")

    existing = execute_dict(
        "SELECT user_id FROM users WHERE LOWER(email) = LOWER(%s) OR LOWER(username) = LOWER(%s)",
        (req.email, req.username),
    )
    if existing:
        raise HTTPException(status_code=409, detail="Username or email already taken")

    rows = execute_dict(
        "INSERT INTO users (username, email, password) VALUES (%s, %s, %s) "
        "RETURNING user_id, username, email",
        (req.username, req.email, hash_password(req.password)),
    )
    if not rows:
        raise HTTPException(status_code=500, detail="Failed to create user")

    user = rows[0]
    return RegisterResponse(status="ok", user_id=user["user_id"], username=user["username"], email=user["email"])


@router.post("/login", response_model=LoginResponse)
def login(req: LoginRequest, _rl=Depends(_rl_login)):
    # Per-account throttle (on top of the per-IP one above) so credential
    # stuffing rotating through many IPs against one account still gets locked.
    account_login_guard(req.email, max_failures=5, window_seconds=60)

    rows = execute_dict(
        "SELECT user_id, username, email, password, tutorial_seen FROM users "
        "WHERE LOWER(email) = LOWER(%s)",
        (req.email,),
    )
    if not rows:
        record_login_failure(req.email)
        raise HTTPException(status_code=401, detail="Invalid email or password")

    user = rows[0]
    if not verify_password(req.password, user["password"]):
        record_login_failure(req.email)
        raise HTTPException(status_code=401, detail="Invalid email or password")

    token = create_access_token({"sub": str(user["user_id"])})
    return LoginResponse(
        status="ok",
        token=token,
        user_id=user["user_id"],
        username=user["username"],
        email=user["email"],
        tutorial_seen=bool(user.get("tutorial_seen")),
    )


@router.get("/verify")
def verify_token(user_id: int = Depends(get_current_user)):
    """Token validity check for the local backends.

    The central server is the single auth authority — it holds the signing key.
    Local backends call this to validate a bearer token instead of holding the
    JWT secret themselves (so a local operator can never forge tokens). This is
    deliberately cheap: it only decodes the token (cached, no DB hit) and returns
    the user_id.
    """
    return {"user_id": user_id}


@router.get("/me")
def get_me(user_id: int = Depends(get_current_user)):
    rows = execute_dict(
        "SELECT user_id, username, email, display_name, bio, is_team, tutorial_seen "
        "FROM users WHERE user_id = %s",
        (user_id,),
    )
    if not rows:
        raise HTTPException(status_code=404, detail="User not found")
    return rows[0]


@router.patch("/me")
def update_me(req: UpdateProfileRequest, user_id: int = Depends(get_current_user)):
    updates, params = [], []
    if req.display_name is not None:
        updates.append("display_name = %s")
        params.append(req.display_name.strip() or None)
    if req.bio is not None:
        updates.append("bio = %s")
        params.append(req.bio.strip() or None)

    if updates:
        params.append(user_id)
        execute(f"UPDATE users SET {', '.join(updates)} WHERE user_id = %s", tuple(params))

    rows = execute_dict(
        "SELECT user_id, username, email, display_name, bio, is_team, tutorial_seen "
        "FROM users WHERE user_id = %s",
        (user_id,),
    )
    return rows[0] if rows else {}


@router.put("/tutorial-seen")
def mark_tutorial_seen(user_id: int = Depends(get_current_user)):
    execute("UPDATE users SET tutorial_seen = TRUE WHERE user_id = %s", (user_id,))
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Public lookup (no auth) — used by local backends to resolve usernames
# referenced by HF-synced rules/CEs.
# ---------------------------------------------------------------------------

@router.get("/team-users")
def get_team_users():
    """All users with is_team=TRUE. No auth required — used by every
    local backend at startup to mirror the team user(s) into its local
    users table. Without this, fresh local DBs don't have a row for
    the seed-library author, so rules/CEs attributed to that author
    can't JOIN to a display name and the @gavel profile shows an
    empty contributions list."""
    rows = execute_dict(
        "SELECT user_id, username, email, display_name, bio, is_team, tutorial_seen "
        "FROM users WHERE is_team = TRUE",
    )
    return {"users": rows or []}


@router.get("/users/by-username")
def get_users_by_username(usernames: str):
    """Comma-separated usernames → list of user rows. Used by the local
    backend after an HF sync to populate its `users` mirror so FK
    constraints and triggers resolve."""
    if not usernames:
        return {"users": []}
    names = [u.strip().lower() for u in usernames.split(",") if u.strip()]
    if not names:
        return {"users": []}
    rows = execute_dict(
        "SELECT user_id, username, email, display_name, bio, is_team, tutorial_seen "
        "FROM users WHERE LOWER(username) = ANY(%s)",
        (names,),
    )
    return {"users": rows or []}


@router.get("/users/{user_id}")
def get_user_by_id(user_id: int, _: int = Depends(get_current_user)):
    """Looking up a user by id (e.g. for the ratings self-rating guard)."""
    rows = execute_dict(
        "SELECT user_id, username, email FROM users WHERE user_id = %s",
        (user_id,),
    )
    if not rows:
        raise HTTPException(status_code=404, detail="User not found")
    return rows[0]


class PublishAttributionRequest(BaseModel):
    """Tells the central server that the authed user just published a
    rule or CE to HF. The server bumps user_ratings_summary counters."""
    asset_type: str = Field(..., pattern="^(rule|ce)$")
    published_at: Optional[str] = None  # ISO timestamp; defaults to now()


@router.post("/publish-attribution")
def record_publish(req: PublishAttributionRequest, user_id: int = Depends(get_current_user)):
    """Bump contribution count for the authenticated user. Called by the
    local backend after a successful HF publish through /hf/commit."""
    if req.asset_type == "rule":
        col = "contribution_count_rules"
    else:
        col = "contribution_count_ces"

    pub_clause = "COALESCE(%s::timestamptz, now())"
    execute(
        f"""
        INSERT INTO user_ratings_summary (user_id, {col}, last_published_at)
        VALUES (%s, 1, {pub_clause})
        ON CONFLICT (user_id) DO UPDATE
        SET {col} = user_ratings_summary.{col} + 1,
            last_published_at = GREATEST(
                user_ratings_summary.last_published_at,
                EXCLUDED.last_published_at
            )
        """,
        (user_id, req.published_at),
    )
    return {"status": "ok"}
