"""Per-user bookmarks for public rules and CEs.

Stored by `public_id` (the HuggingFace identifier) so they survive across
local-DB rebuilds and follow the user from machine to machine.
"""
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..utils.auth import get_current_user
from ..utils.db import execute, execute_dict

router = APIRouter(prefix="/bookmarks", tags=["bookmarks"])

AssetType = Literal["rule", "ce", "rule_set"]


class BookmarkRequest(BaseModel):
    asset_type: AssetType
    public_id: str = Field(..., min_length=1, max_length=255)


def _table_and_col(asset_type: str):
    if asset_type == "rule":
        return "rule_bookmarks", "rule_public_id"
    if asset_type == "rule_set":
        return "rule_set_bookmarks", "rule_set_public_id"
    return "ce_bookmarks", "ce_public_id"


@router.post("")
def add_bookmark(req: BookmarkRequest, user_id: int = Depends(get_current_user)):
    table, col = _table_and_col(req.asset_type)
    execute(
        f"INSERT INTO {table} (user_id, {col}) VALUES (%s, %s) "
        f"ON CONFLICT (user_id, {col}) DO NOTHING",
        (user_id, req.public_id),
    )
    return {"status": "ok"}


@router.delete("/{asset_type}/{public_id}")
def remove_bookmark(asset_type: AssetType, public_id: str, user_id: int = Depends(get_current_user)):
    table, col = _table_and_col(asset_type)
    execute(f"DELETE FROM {table} WHERE user_id = %s AND {col} = %s", (user_id, public_id))
    return {"status": "ok"}


@router.get("/{asset_type}")
def list_bookmarks(asset_type: AssetType, user_id: int = Depends(get_current_user)):
    """Returns the list of public_ids this user has bookmarked for the
    given asset type. The local backend joins these against its local
    rules/CEs cache to render the full bookmark list."""
    table, col = _table_and_col(asset_type)
    rows = execute_dict(
        f"SELECT {col} AS public_id, created_at FROM {table} "
        f"WHERE user_id = %s ORDER BY created_at DESC",
        (user_id,),
    )
    return {"bookmarks": rows or []}
