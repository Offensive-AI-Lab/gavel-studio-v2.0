"""Ratings endpoints — one row per (user, asset)."""
from typing import Literal, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..utils.auth import get_current_user
from ..utils.db import execute, execute_dict

router = APIRouter(prefix="/ratings", tags=["ratings"])

AssetType = Literal["rule", "ce"]


class RateRequest(BaseModel):
    asset_type: AssetType
    asset_public_id: str = Field(..., min_length=1, max_length=255)
    score: int = Field(..., ge=1, le=5)
    # Sent by the local backend so central can update the creator's
    # user_ratings_summary. Optional for backward compat — if missing,
    # only asset_ratings_summary is updated (via trigger), not the
    # author's profile aggregate.
    created_by_username: Optional[str] = None


class RatingSummary(BaseModel):
    rating_count: int = 0
    rating_avg: Optional[float] = None
    your_score: Optional[int] = None


def _summary(asset_type: str, asset_public_id: str, user_id: Optional[int]) -> RatingSummary:
    rows = execute_dict(
        "SELECT rating_count, rating_sum FROM asset_ratings_summary "
        "WHERE asset_type = %s AND asset_public_id = %s",
        (asset_type, asset_public_id),
    )
    count, total = (rows[0]["rating_count"], rows[0]["rating_sum"]) if rows else (0, 0)
    avg = round(total / count, 2) if count else None

    your_score = None
    if user_id is not None:
        own = execute_dict(
            "SELECT score FROM ratings WHERE user_id = %s AND asset_type = %s AND asset_public_id = %s",
            (user_id, asset_type, asset_public_id),
        )
        if own:
            your_score = own[0]["score"]

    return RatingSummary(rating_count=count, rating_avg=avg, your_score=your_score)


@router.post("")
def upsert_rating(req: RateRequest, user_id: int = Depends(get_current_user)):
    # Check if this is a new rating or a re-rate so we can compute the
    # delta for user_ratings_summary.
    old = execute_dict(
        "SELECT score FROM ratings WHERE user_id = %s AND asset_type = %s AND asset_public_id = %s",
        (user_id, req.asset_type, req.asset_public_id),
    )
    old_score = old[0]["score"] if old else None

    execute(
        """
        INSERT INTO ratings (user_id, asset_type, asset_public_id, score, creator_username)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (user_id, asset_type, asset_public_id) DO UPDATE
        SET score = EXCLUDED.score, updated_at = now(),
            creator_username = COALESCE(EXCLUDED.creator_username, ratings.creator_username)
        """,
        (user_id, req.asset_type, req.asset_public_id, req.score, req.created_by_username),
    )

    # Update the artifact creator's user_ratings_summary so the profile
    # page shows the correct "avg rating received". The trigger handles
    # asset_ratings_summary; this handles the per-user aggregate.
    if req.created_by_username:
        creator = execute_dict(
            "SELECT user_id FROM users WHERE username = %s",
            (req.created_by_username,),
        )
        if creator:
            creator_uid = creator[0]["user_id"]
            delta_count = 0 if old_score is not None else 1
            delta_sum = req.score - (old_score or 0)
            execute(
                """
                INSERT INTO user_ratings_summary (user_id, total_rating_count, total_rating_sum)
                VALUES (%s, %s, %s)
                ON CONFLICT (user_id) DO UPDATE
                SET total_rating_count = user_ratings_summary.total_rating_count + EXCLUDED.total_rating_count,
                    total_rating_sum   = user_ratings_summary.total_rating_sum   + EXCLUDED.total_rating_sum
                """,
                (creator_uid, delta_count, delta_sum),
            )

    return _summary(req.asset_type, req.asset_public_id, user_id).dict()


@router.delete("/{asset_type}/{asset_public_id}")
def delete_rating(
    asset_type: AssetType,
    asset_public_id: str,
    created_by_username: Optional[str] = None,
    user_id: int = Depends(get_current_user),
):
    # Read the score AND the creator that was recorded when THIS rating was made,
    # so the decrement targets exactly the user the original rate incremented. The
    # client-supplied `created_by_username` query param is IGNORED for the decrement
    # (trusting it let a delete arbitrarily dock any user's score).
    old = execute_dict(
        "SELECT score, creator_username FROM ratings "
        "WHERE user_id = %s AND asset_type = %s AND asset_public_id = %s",
        (user_id, asset_type, asset_public_id),
    )
    execute(
        "DELETE FROM ratings WHERE user_id = %s AND asset_type = %s AND asset_public_id = %s",
        (user_id, asset_type, asset_public_id),
    )

    # Decrement the creator's user_ratings_summary — creator from the STORED row.
    stored_creator = old[0]["creator_username"] if old else None
    if old and stored_creator:
        old_score = old[0]["score"]
        creator = execute_dict(
            "SELECT user_id FROM users WHERE username = %s",
            (stored_creator,),
        )
        if creator:
            execute(
                """
                UPDATE user_ratings_summary
                SET total_rating_count = GREATEST(total_rating_count - 1, 0),
                    total_rating_sum   = GREATEST(total_rating_sum - %s, 0)
                WHERE user_id = %s
                """,
                (old_score, creator[0]["user_id"]),
            )

    return _summary(asset_type, asset_public_id, user_id).dict()


@router.get("/{asset_type}/{asset_public_id}")
def get_rating_summary(
    asset_type: AssetType,
    asset_public_id: str,
    user_id: int = Depends(get_current_user),
):
    return _summary(asset_type, asset_public_id, user_id).dict()
