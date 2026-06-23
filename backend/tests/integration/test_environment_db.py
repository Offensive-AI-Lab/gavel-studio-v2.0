"""DB-dependent environment checks.

These verify that the running database has the schema and extensions the
application expects.
"""
import pytest


class TestDatabaseConnectivity:
    """Verify env vars resolve to a working DB connection."""

    def test_can_get_connection(self):
        from utils.PostgreSQL import get_connection, release_connection
        conn = get_connection()
        try:
            assert conn is not None
        finally:
            release_connection(conn)


class TestDatabaseInitialization:
    """Verify schema initialization is idempotent and complete."""

    def test_init_database_idempotent(self):
        """Running init_database twice should not error or duplicate data."""
        from utils.DButils import init_database
        from utils.PostgreSQL import execute_query_dict

        before = execute_query_dict("SELECT COUNT(*) AS c FROM categories")
        count_before = before[0]["c"]

        init_database()

        after = execute_query_dict("SELECT COUNT(*) AS c FROM categories")
        count_after = after[0]["c"]
        assert count_after == count_before

    def test_required_tables_exist(self):
        """Critical tables must exist after init."""
        from utils.PostgreSQL import execute_query_dict
        # Bookmarks + ratings tables moved to the central server when we
        # migrated users to a shared identity service; only model/training
        # data lives locally now.
        required = [
            "users", "target_models", "classifiers",
            "cognitive_elements", "rules", "categories",
            "rule_setup", "setup_ce_link", "rule_ce_link",
            "excitation_datasets", "calibration_datasets",
            "evaluation_results", "test_datasets",
        ]
        result = execute_query_dict(
            "SELECT tablename FROM pg_tables WHERE schemaname = 'public'"
        )
        existing = {r["tablename"] for r in result or []}
        missing = [t for t in required if t not in existing]
        assert not missing, f"Missing tables: {missing}"

    def test_required_extensions_loaded(self):
        """pg_trgm and vector extensions must be present for search/embeddings."""
        from utils.PostgreSQL import execute_query_dict
        result = execute_query_dict(
            "SELECT extname FROM pg_extension WHERE extname IN ('pg_trgm', 'vector')"
        )
        extensions = {r["extname"] for r in result or []}
        assert "pg_trgm" in extensions, "pg_trgm extension required for fuzzy search"
        assert "vector" in extensions, "pgvector extension required for semantic search"
