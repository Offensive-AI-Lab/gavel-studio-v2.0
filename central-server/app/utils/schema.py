"""Schema initialization for the central server.

Tables (all live on this server):
    users                      — identity + profile
    ratings                    — per-user rating of a public rule/CE (by public_id)
    asset_ratings_summary      — denormalized aggregate per (asset_type, public_id)
    user_ratings_summary       — denormalized aggregate per user (contributions, totals)
    rule_bookmarks             — per-user bookmark of a public rule (by public_id)
    ce_bookmarks               — per-user bookmark of a public CE (by public_id)
    rule_set_bookmarks         — per-user bookmark of a public rule set (by public_id)

Bookmarks reference assets by `public_id` (the HuggingFace identifier), not
by local SERIAL id, so they're portable across machines.
"""
from .db import execute


def init_schema() -> None:
    print("--- Initializing Central Server Schema ---")

    try:
        execute("CREATE EXTENSION IF NOT EXISTS citext;")
    except Exception as e:
        print(f"Warning: citext extension not available: {e}")

    # USERS
    execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id       SERIAL PRIMARY KEY,
            username      CITEXT NOT NULL UNIQUE,
            password      VARCHAR(255) NOT NULL,
            email         VARCHAR(255) NOT NULL UNIQUE,
            display_name  VARCHAR(255),
            bio           TEXT,
            is_team       BOOLEAN NOT NULL DEFAULT FALSE,
            tutorial_seen BOOLEAN DEFAULT FALSE,
            created_at    TIMESTAMPTZ DEFAULT now()
        );
    """)

    # RATINGS — one row per (user, asset). asset_public_id is the HF id.
    execute("""
        CREATE TABLE IF NOT EXISTS ratings (
            rating_id       SERIAL PRIMARY KEY,
            user_id         INTEGER NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
            asset_type      VARCHAR(10) NOT NULL CHECK (asset_type IN ('rule', 'ce', 'rule_set')),
            asset_public_id TEXT NOT NULL,
            score           SMALLINT NOT NULL CHECK (score BETWEEN 1 AND 5),
            created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
            UNIQUE (user_id, asset_type, asset_public_id)
        );
    """)
    # v19: public rule sets are rateable too. Existing DBs created the table
    # with CHECK (asset_type IN ('rule','ce')); drop and re-add the (auto-named)
    # constraint to admit 'rule_set'. Idempotent — re-adds the same named
    # constraint on fresh DBs. asset_ratings_summary/user_ratings_summary are
    # already generic over (asset_type, asset_public_id), so no other change.
    execute("ALTER TABLE ratings DROP CONSTRAINT IF EXISTS ratings_asset_type_check;")
    execute("ALTER TABLE ratings ADD CONSTRAINT ratings_asset_type_check CHECK (asset_type IN ('rule', 'ce', 'rule_set'));")
    execute("CREATE INDEX IF NOT EXISTS idx_ratings_asset ON ratings (asset_type, asset_public_id);")
    execute("CREATE INDEX IF NOT EXISTS idx_ratings_user ON ratings (user_id);")
    # Creator recorded AT RATE TIME, so un-rating decrements exactly the user that
    # rating incremented — a delete can't be made to target an arbitrary user via a
    # spoofed created_by_username query param.
    execute("ALTER TABLE ratings ADD COLUMN IF NOT EXISTS creator_username TEXT;")

    # ASSET RATINGS SUMMARY — denormalized aggregate
    execute("""
        CREATE TABLE IF NOT EXISTS asset_ratings_summary (
            asset_type      VARCHAR(10) NOT NULL,
            asset_public_id TEXT NOT NULL,
            rating_count    INTEGER NOT NULL DEFAULT 0,
            rating_sum      INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (asset_type, asset_public_id)
        );
    """)

    # USER RATINGS SUMMARY — denormalized aggregate per user
    # creator_username is denormalized so the trigger does not need a
    # cross-table lookup (the central server doesn't store rules/CEs).
    execute("""
        CREATE TABLE IF NOT EXISTS user_ratings_summary (
            user_id                   INTEGER PRIMARY KEY REFERENCES users(user_id) ON DELETE CASCADE,
            contribution_count_rules  INTEGER NOT NULL DEFAULT 0,
            contribution_count_ces    INTEGER NOT NULL DEFAULT 0,
            total_rating_count        INTEGER NOT NULL DEFAULT 0,
            total_rating_sum          INTEGER NOT NULL DEFAULT 0,
            last_published_at         TIMESTAMPTZ
        );
    """)

    # BOOKMARKS — by public_id so they're portable across machines
    execute("""
        CREATE TABLE IF NOT EXISTS rule_bookmarks (
            id              SERIAL PRIMARY KEY,
            user_id         INTEGER NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
            rule_public_id  TEXT NOT NULL,
            created_at      TIMESTAMPTZ DEFAULT now(),
            UNIQUE (user_id, rule_public_id)
        );
    """)
    execute("""
        CREATE TABLE IF NOT EXISTS ce_bookmarks (
            id              SERIAL PRIMARY KEY,
            user_id         INTEGER NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
            ce_public_id    TEXT NOT NULL,
            created_at      TIMESTAMPTZ DEFAULT now(),
            UNIQUE (user_id, ce_public_id)
        );
    """)
    # v19: bookmarks for public rule sets, keyed by the rule set's HF public_id
    # (portable across machines, exactly like rule/ce bookmarks).
    execute("""
        CREATE TABLE IF NOT EXISTS rule_set_bookmarks (
            id                  SERIAL PRIMARY KEY,
            user_id             INTEGER NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
            rule_set_public_id  TEXT NOT NULL,
            created_at          TIMESTAMPTZ DEFAULT now(),
            UNIQUE (user_id, rule_set_public_id)
        );
    """)

    # Ratings -> summaries trigger. Updates asset_ratings_summary on every
    # change and (when the creator_uid is resolvable via creator_username
    # passed by the caller) bumps user_ratings_summary. The publish flow
    # is responsible for bumping contribution_count_* via a direct UPDATE,
    # not via trigger, because creator identity lives outside this server.
    execute("""
        CREATE OR REPLACE FUNCTION update_asset_ratings_summary() RETURNS TRIGGER AS $$
        DECLARE
            delta_count    INTEGER;
            delta_sum      INTEGER;
            effective_type VARCHAR(10);
            effective_pid  TEXT;
        BEGIN
            IF TG_OP = 'INSERT' THEN
                delta_count := 1;
                delta_sum   := NEW.score;
                effective_type := NEW.asset_type;
                effective_pid  := NEW.asset_public_id;
            ELSIF TG_OP = 'UPDATE' THEN
                delta_count := 0;
                delta_sum   := NEW.score - OLD.score;
                effective_type := NEW.asset_type;
                effective_pid  := NEW.asset_public_id;
            ELSIF TG_OP = 'DELETE' THEN
                delta_count := -1;
                delta_sum   := -OLD.score;
                effective_type := OLD.asset_type;
                effective_pid  := OLD.asset_public_id;
            END IF;

            INSERT INTO asset_ratings_summary (asset_type, asset_public_id, rating_count, rating_sum)
            VALUES (effective_type, effective_pid, delta_count, delta_sum)
            ON CONFLICT (asset_type, asset_public_id) DO UPDATE
            SET rating_count = asset_ratings_summary.rating_count + EXCLUDED.rating_count,
                rating_sum   = asset_ratings_summary.rating_sum   + EXCLUDED.rating_sum;

            RETURN COALESCE(NEW, OLD);
        END;
        $$ LANGUAGE plpgsql;
    """)
    execute("DROP TRIGGER IF EXISTS trg_asset_ratings_summary ON ratings;")
    execute("""
        CREATE TRIGGER trg_asset_ratings_summary
        AFTER INSERT OR UPDATE OR DELETE ON ratings
        FOR EACH ROW EXECUTE FUNCTION update_asset_ratings_summary();
    """)

    print("✓ Central schema initialized")
