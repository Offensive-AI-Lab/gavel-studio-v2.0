"""Seed the GAVEL team user on the central server's database.

Run once on the VM after first deploy:
    cd /opt/gavel/central-server
    source .venv/bin/activate
    python seed_gavel_user.py

Reads GAVEL_TEAM_* values from .env (same file the central server
reads). Creates the team user with an unusable password hash so nobody
can log in as this account. Idempotent — safe to run multiple times.

This script only touches the CENTRAL DB (users + user_ratings_summary).
The LOCAL backend's rules/CEs backfill (created_by_username='gavel')
is a separate step that runs on the laptop via backend/scripts/seed_team_user.py.
"""
import os
import sys
from pathlib import Path

import psycopg2
from dotenv import load_dotenv

SCRIPT_DIR = Path(__file__).resolve().parent
load_dotenv(dotenv_path=SCRIPT_DIR / ".env")

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    sys.exit("[seed] DATABASE_URL not set in .env")

email = os.getenv("GAVEL_TEAM_EMAIL", "").strip()
username = os.getenv("GAVEL_TEAM_USERNAME", "").strip().lower()
display_name = os.getenv("GAVEL_TEAM_DISPLAY_NAME", "").strip()
bio = os.getenv("GAVEL_TEAM_BIO", "").strip()

if not email or not username:
    sys.exit(
        "[seed] GAVEL_TEAM_EMAIL and GAVEL_TEAM_USERNAME must be set in .env.\n"
        "Example:\n"
        "  GAVEL_TEAM_EMAIL=GavelTeamSupport@gmail.com\n"
        "  GAVEL_TEAM_USERNAME=gavel\n"
        "  GAVEL_TEAM_DISPLAY_NAME=GAVEL\n"
        "  GAVEL_TEAM_BIO=The seed library curated by the GAVEL team."
    )


def run():
    conn = psycopg2.connect(DATABASE_URL)
    conn.autocommit = True
    cur = conn.cursor()

    # Check if the user already exists.
    cur.execute("SELECT user_id FROM users WHERE username = %s", (username,))
    row = cur.fetchone()

    if row:
        user_id = row[0]
        print(f"[seed] Team user '{username}' already exists (user_id={user_id}).")
    else:
        # Unusable password hash — argon2 format but derived from random
        # bytes so no plaintext maps to it. Prevents login as this account.
        import secrets
        from passlib.context import CryptContext
        pwd_ctx = CryptContext(schemes=["argon2"], deprecated="auto")
        unusable_hash = pwd_ctx.hash(secrets.token_hex(64))

        cur.execute(
            """
            INSERT INTO users (username, email, password, display_name, bio, is_team)
            VALUES (%s, %s, %s, %s, %s, TRUE)
            RETURNING user_id
            """,
            (username, email, unusable_hash, display_name or username.upper(), bio or ""),
        )
        user_id = cur.fetchone()[0]
        print(f"[seed] Created team user '{username}' (user_id={user_id}).")

    # Ensure a user_ratings_summary row exists. The contribution counts
    # are placeholders — they reflect what's on HF, not what's in the
    # central DB (central doesn't have rules/CEs tables). The local
    # backend's seed script sets the real counts. Here we just make sure
    # the row exists so profile queries don't 404.
    cur.execute(
        """
        INSERT INTO user_ratings_summary (user_id, contribution_count_rules, contribution_count_ces)
        VALUES (%s, 0, 0)
        ON CONFLICT (user_id) DO NOTHING
        """,
        (user_id,),
    )
    print(f"[seed] user_ratings_summary row ensured for user_id={user_id}.")

    cur.close()
    conn.close()
    print("[seed] Done.")


if __name__ == "__main__":
    run()
