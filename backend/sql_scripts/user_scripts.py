"""Local user table helpers.

Auth (register, login, profile reads/writes) lives on the central server
now — this module only handles the LOCAL mirror used for FK constraints
and triggers. After every login/register, the central server's user row
is copied here via `sync_user_to_local`.
"""
from utils.PostgreSQL import execute_query, execute_query_dict


def sync_user_to_local(user: dict):
    """Upsert a user row into the local DB so FKs and triggers resolve.

    Copies all available profile fields so local JOINs (profile page,
    leaderboard) return complete data.

    The local mirror is keyed by the central server's `user_id`. The table
    ALSO has UNIQUE(username) and UNIQUE(email), so a plain upsert-on-id breaks
    when the SAME username/email comes back under a DIFFERENT id — which
    happens after the central DB is reset or the backend is pointed at a
    different central server (ids get reassigned). Clear any stale row holding
    this username/email under another id first, so the mirror always reflects
    the CURRENT central identity instead of 500-ing on the unique constraint.
    (FKs cascade, so data owned by the now-defunct id mapping is removed with
    it — correct, since that identity no longer exists centrally.)
    """
    execute_query(
        "DELETE FROM users WHERE (username = %s OR email = %s) AND user_id <> %s",
        (user["username"], user["email"], user["user_id"]),
    )
    execute_query(
        """
        INSERT INTO users (user_id, username, email, password,
                           display_name, bio, is_team, tutorial_seen)
        VALUES (%s, %s, %s, '', %s, %s, %s, %s)
        ON CONFLICT (user_id) DO UPDATE SET
            username      = EXCLUDED.username,
            email         = EXCLUDED.email,
            display_name  = EXCLUDED.display_name,
            bio           = EXCLUDED.bio,
            is_team       = EXCLUDED.is_team,
            tutorial_seen = EXCLUDED.tutorial_seen
        """,
        (
            user["user_id"],
            user["username"],
            user["email"],
            user.get("display_name"),
            user.get("bio"),
            bool(user.get("is_team", False)),
            bool(user.get("tutorial_seen", False)),
        ),
    )


def get_user_by_id(user_id: int):
    """Read a user from the LOCAL mirror. Used by dashboard and other
    routes where the user is already authenticated (so they must exist
    in the local mirror)."""
    result = execute_query_dict(
        "SELECT user_id, username, email, tutorial_seen FROM users WHERE user_id = %s",
        (user_id,),
    )
    return result[0] if result else None


def ensure_creators_in_local(usernames: list):
    """After an HF sync imports rules/CEs, ensure every creator username
    referenced by those records exists in the local users mirror.

    Fetches missing users from the CENTRAL SERVER and upserts them
    locally so FK constraints, triggers, and profile JOINs resolve for
    users who have never logged into this machine.
    """
    if not usernames:
        return

    unique = list({u.lower() for u in usernames if u})
    if not unique:
        return

    local_rows = execute_query(
        "SELECT LOWER(username) FROM users WHERE LOWER(username) = ANY(%s)",
        (unique,),
    )
    local_set = {r[0] for r in local_rows} if local_rows else set()
    missing = [u for u in unique if u not in local_set]
    if not missing:
        return

    try:
        from services import central_server
        remote_rows = central_server.get_users_by_username(missing)
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"Could not fetch creators from central server: {e}")
        return

    for row in (remote_rows or []):
        sync_user_to_local(row)
