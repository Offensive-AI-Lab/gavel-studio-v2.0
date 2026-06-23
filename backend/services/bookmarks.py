"""Bookmark CRUD service — proxied through the central server.

The central server stores bookmarks by `public_id` (the HF identifier),
not by local SERIAL ids. That makes bookmarks portable across machines.
This service translates between the two:

  * On add/remove: local id -> public_id (lookup in local cache) -> central
  * On list: public_ids from central -> local rows (join against cache)
"""
from dataclasses import dataclass
from typing import List

from services import central_server
from services.central_server import CentralServerError
from utils.PostgreSQL import execute_query_dict


class BookmarkLookupError(Exception):
    """Raised when a local asset id cannot be resolved to a public_id
    (i.e., the user tried to bookmark a draft that hasn't been published).
    Routes turn this into a 400."""


@dataclass(frozen=True)
class BookmarkSpec:
    asset_type: str         # "rule" | "ce"
    asset_table: str        # "rules" | "cognitive_elements"
    asset_id_col: str       # "rule_id" | "ce_id"
    list_columns: str       # comma-separated columns to surface in list view


BOOKMARK_REGISTRY: dict = {
    "rule": BookmarkSpec(
        asset_type="rule",
        asset_table="rules",
        asset_id_col="rule_id",
        list_columns="a.name, a.predicate, a.description",
    ),
    "ce": BookmarkSpec(
        asset_type="ce",
        asset_table="cognitive_elements",
        asset_id_col="ce_id",
        list_columns="a.name, a.definition, a.category",
    ),
    "rule_set": BookmarkSpec(
        asset_type="rule_set",
        asset_table="rule_sets",
        asset_id_col="rule_set_id",
        list_columns="a.name, a.description",
    ),
}


def _spec(asset_type: str) -> BookmarkSpec:
    spec = BOOKMARK_REGISTRY.get(asset_type)
    if not spec:
        raise ValueError(f"Unknown bookmark asset type: {asset_type!r}")
    return spec


def _local_id_to_public_id(spec: BookmarkSpec, asset_id: int) -> str:
    rows = execute_query_dict(
        f"SELECT public_id FROM {spec.asset_table} WHERE {spec.asset_id_col} = %s",
        (asset_id,),
    ) or []
    if not rows:
        raise BookmarkLookupError(f"{spec.asset_type} #{asset_id} not found locally")
    public_id = rows[0]["public_id"]
    if not public_id:
        raise BookmarkLookupError(
            f"Cannot bookmark a draft {spec.asset_type} (no public_id). "
            "Publish it to HF first."
        )
    return public_id


class BookmarkService:
    """Token-based bookmark API. The token authoritatively identifies
    the user (no need to pass user_id separately)."""

    @staticmethod
    def add(asset_type: str, token: str, asset_id: int) -> bool:
        spec = _spec(asset_type)
        public_id = _local_id_to_public_id(spec, asset_id)
        try:
            central_server.add_bookmark(token, spec.asset_type, public_id)
            return True
        except CentralServerError as err:
            raise RuntimeError(f"Central server: {err}") from err

    @staticmethod
    def remove(asset_type: str, token: str, asset_id: int) -> bool:
        spec = _spec(asset_type)
        try:
            public_id = _local_id_to_public_id(spec, asset_id)
        except BookmarkLookupError:
            # Local row gone? Bookmark might still exist on central — but
            # we don't know its public_id, so this is a no-op. Returning
            # True keeps the UI idempotent.
            return True
        try:
            central_server.remove_bookmark(token, spec.asset_type, public_id)
            return True
        except CentralServerError as err:
            raise RuntimeError(f"Central server: {err}") from err

    @staticmethod
    def list(asset_type: str, token: str) -> List[dict]:
        """Return the user's bookmarks as a list of full local rows.

        Pulls public_ids from central, then joins against the local
        asset table to surface the columns the UI cares about. Bookmarks
        for public_ids the local cache hasn't synced yet are silently
        dropped — they'll show up on the next HF sync.
        """
        spec = _spec(asset_type)
        try:
            bookmarks = central_server.list_bookmarks(token, spec.asset_type)
        except CentralServerError as err:
            raise RuntimeError(f"Central server: {err}") from err

        public_ids = [b["public_id"] for b in bookmarks if b.get("public_id")]
        if not public_ids:
            return []

        rows = execute_query_dict(
            f"""
            SELECT a.{spec.asset_id_col}, a.public_id, {spec.list_columns}
            FROM {spec.asset_table} a
            WHERE a.public_id = ANY(%s)
            """,
            (public_ids,),
        ) or []

        # Preserve the central server's ordering (most-recent-bookmark first)
        order = {pid: i for i, pid in enumerate(public_ids)}
        rows.sort(key=lambda r: order.get(r["public_id"], 1_000_000))
        return rows
