"""User discovery endpoints: public profile, search, leaderboard.

These join `users` with `user_ratings_summary` — both live on this
server, so a single SQL query produces the result. The local backend
calls these and may augment the response with its own data (e.g. the
profile-contributions endpoint adds the user's local rules/CEs cache).
"""
from typing import Literal, Optional

from fastapi import APIRouter, HTTPException, Query

from ..utils.db import execute_dict

router = APIRouter(prefix="/users", tags=["users"])


@router.get("/profile/{username}")
def get_profile(username: str):
    """Public profile + aggregate stats. Anyone can call this (no auth).

    Returns NULL/0 for unrated / no-contributions users — the user row
    itself is enough to consider them "registered". The artist-gate
    filtering lives in /users/search, not here."""
    username = (username or "").strip().lower()
    if not username:
        raise HTTPException(status_code=400, detail="username required")

    rows = execute_dict(
        """
        SELECT u.user_id, u.username,
               u.display_name, u.bio, u.is_team,
               u.created_at,
               COALESCE(s.contribution_count_rules, 0) AS contribution_count_rules,
               COALESCE(s.contribution_count_ces, 0)   AS contribution_count_ces,
               COALESCE(s.total_rating_count, 0)       AS total_rating_count,
               s.total_rating_sum,
               s.last_published_at
        FROM users u
        LEFT JOIN user_ratings_summary s ON s.user_id = u.user_id
        WHERE u.username = %s
        """,
        (username,),
    )
    if not rows:
        raise HTTPException(status_code=404, detail="User not found")
    row = dict(rows[0])

    avg = None
    if row["total_rating_count"] and row["total_rating_count"] > 0:
        avg = round((row["total_rating_sum"] or 0) / row["total_rating_count"], 2)

    member_since = row["created_at"].isoformat() if row.get("created_at") else None
    last_pub = row["last_published_at"].isoformat() if row.get("last_published_at") else None

    return {
        "user_id": row["user_id"],
        "username": row["username"],
        # NOTE: email is deliberately NOT exposed here — this is a PUBLIC
        # (unauthenticated) profile lookup, so returning it would let anyone
        # harvest any user's email by username. Your own email comes from the
        # authenticated /auth/me instead.
        "display_name": row.get("display_name"),
        "bio": row.get("bio"),
        "is_team": bool(row.get("is_team")),
        "member_since": member_since,
        "contribution_count_rules": row["contribution_count_rules"],
        "contribution_count_ces": row["contribution_count_ces"],
        "total_rating_count": row["total_rating_count"],
        "avg_rating_received": avg,
        "last_published_at": last_pub,
    }


@router.get("/search")
def search_users(
    q: Optional[str] = Query(None, max_length=120),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
):
    """Artist-gated user search. Matches username or display_name; users
    with zero contributions are filtered out so non-artists don't appear
    in discovery results."""
    offset = (page - 1) * page_size
    like = f"%{(q or '').strip().lower()}%"

    base_where = "(s.contribution_count_rules + s.contribution_count_ces) > 0"
    filter_clause = "AND (LOWER(u.username) LIKE %s OR LOWER(COALESCE(u.display_name, '')) LIKE %s)" if q else ""
    params = [like, like] if q else []

    total_row = execute_dict(
        f"""
        SELECT COUNT(*) AS n
        FROM users u
        JOIN user_ratings_summary s ON s.user_id = u.user_id
        WHERE {base_where} {filter_clause}
        """,
        tuple(params),
    )
    total = total_row[0]["n"] if total_row else 0

    items = execute_dict(
        f"""
        SELECT u.user_id, u.username, u.display_name, u.bio, u.is_team,
               s.contribution_count_rules,
               s.contribution_count_ces,
               s.total_rating_count,
               s.total_rating_sum,
               s.last_published_at
        FROM users u
        JOIN user_ratings_summary s ON s.user_id = u.user_id
        WHERE {base_where} {filter_clause}
        ORDER BY (s.contribution_count_rules + s.contribution_count_ces) DESC,
                 s.last_published_at DESC NULLS LAST
        LIMIT %s OFFSET %s
        """,
        tuple(params + [page_size, offset]),
    ) or []

    for it in items:
        if it.get("total_rating_count") and it["total_rating_count"] > 0:
            it["avg_rating_received"] = round((it.get("total_rating_sum") or 0) / it["total_rating_count"], 2)
        else:
            it["avg_rating_received"] = None
        it.pop("total_rating_sum", None)
        if it.get("last_published_at"):
            it["last_published_at"] = it["last_published_at"].isoformat()

    return {"q": q, "page": page, "page_size": page_size, "total": total, "items": items}


@router.get("/leaderboard")
def leaderboard(
    sort: Literal["rating", "contributions"] = Query("rating"),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    min_ratings: int = Query(0, ge=0, le=100000),
):
    """Top contributors. Sort options:
        rating         — by avg rating received
        contributions  — by total artifacts published
    Both filter out non-artists (zero contributions).

    `min_ratings` is a caller-controlled floor on how many ratings a
    contributor must have received to appear (the UI exposes it as a
    "minimum ratings" filter). For the `rating` sort we always require at
    least 1 rating — an unrated user has no average to rank by — so the
    effective floor is max(1, min_ratings). For `contributions` the floor is
    only applied when min_ratings > 0, leaving the raw "most published" view
    unfiltered by default."""
    offset = (page - 1) * page_size

    where = "(s.contribution_count_rules + s.contribution_count_ces) > 0"
    params: list = []
    if sort == "rating":
        eff_min = max(1, min_ratings)
        where += " AND s.total_rating_count >= %s"
        params.append(eff_min)
        order = "(s.total_rating_sum::float / NULLIF(s.total_rating_count, 0)) DESC, s.total_rating_count DESC"
    else:
        if min_ratings > 0:
            where += " AND s.total_rating_count >= %s"
            params.append(min_ratings)
        order = "(s.contribution_count_rules + s.contribution_count_ces) DESC, s.last_published_at DESC NULLS LAST"

    items = execute_dict(
        f"""
        SELECT u.user_id, u.username, u.display_name, u.bio, u.is_team,
               s.contribution_count_rules,
               s.contribution_count_ces,
               s.total_rating_count,
               s.total_rating_sum
        FROM users u
        JOIN user_ratings_summary s ON s.user_id = u.user_id
        WHERE {where}
        ORDER BY {order}
        LIMIT %s OFFSET %s
        """,
        tuple(params + [page_size, offset]),
    ) or []

    for it in items:
        if it.get("total_rating_count") and it["total_rating_count"] > 0:
            it["avg_rating_received"] = round((it.get("total_rating_sum") or 0) / it["total_rating_count"], 2)
        else:
            it["avg_rating_received"] = None
        it.pop("total_rating_sum", None)

    return {"sort": sort, "page": page, "page_size": page_size, "items": items}
