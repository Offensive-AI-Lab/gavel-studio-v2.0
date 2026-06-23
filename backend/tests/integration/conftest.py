"""Integration-test fixtures.

Owns everything that needs a live database:
  * The FastAPI TestClient.
  * Session-level fixtures that create a test user / model / classifier.
  * The DB snapshot+restore mechanism that cleans up after every test.

Unit tests do not see these fixtures because pytest's conftest discovery is
directory-scoped — only files under tests/integration/ pick up this file.

The cleanup strategy:

  1. At session start (`_session_baseline`) — snapshot the current PK state of
     every tracked table. This is the "safety net" that catches anything the
     session-scoped fixtures themselves create (test_user, test_model,
     test_classifier).

  2. Before every test (`_per_test_cleanup`) — snapshot the current state
     again, run the test, then delete any rows that appeared during the test.
     With per-test cleanup, even an interrupted run only pollutes the rows
     created by the single test that was in flight.

  3. At session end — restore to the session baseline so the dev DB is in the
     exact state we found it.

Foreign keys in the schema all use ON DELETE CASCADE, so deleting parent rows
automatically removes children. The deletion order in `_TRACKED_TABLES` is
children-first to avoid relying on cascade behavior, which keeps the cleanup
robust if a future migration weakens any FK to RESTRICT.
"""
import time

import pytest
from fastapi.testclient import TestClient

from main import app
from utils.auth import create_access_token


@pytest.fixture(autouse=True)
def _mock_central_verify(monkeypatch):
    """The local backend now verifies bearer tokens via the central server
    (it no longer holds the JWT secret). There's no live central server in
    tests, so stand in for it: decode the locally-minted test token and return
    its subject. Also clear the per-request verify cache so a token from one
    test can't leak into the next."""
    import utils.auth as _auth
    from services import central_server

    def _fake_verify(token):
        try:
            payload = _auth.decode_access_token(token)
        except Exception:
            raise central_server.CentralServerError("invalid token", status_code=401)
        sub = payload.get("sub")
        if sub is None:
            raise central_server.CentralServerError("token missing subject", status_code=401)
        return int(sub)

    monkeypatch.setattr(central_server, "verify_token", _fake_verify)
    monkeypatch.setattr(central_server, "is_enabled", lambda: True)
    _auth._verify_cache.clear()
    yield
    _auth._verify_cache.clear()


# Tables we track for per-test cleanup, listed children-first so deletes never
# trip an FK constraint. Tables with seed data (categories) are intentionally
# excluded — wiping them would break every test.
#
# Each entry is (table_name, pk_columns) where pk_columns is a TUPLE of the
# column(s) forming the row identity. The junction tables (setup_ce_link,
# rule_ce_link) have COMPOSITE primary keys — there is no surrogate `id`
# column — so they must be tracked by their full key tuple. Tracking them by
# a non-existent "id" column silently broke their cleanup (the query raised
# "column id does not exist", the exception was swallowed, and orphaned link
# rows leaked across the whole session).
_TRACKED_TABLES = [
    # Junction tables (reference rules / cognitive_elements / classifiers).
    # COMPOSITE keys — see note above.
    ("setup_ce_link", ("setup_id", "ce_id", "role", "fallback_group")),
    ("rule_ce_link", ("rule_id", "ce_id", "role", "fallback_group")),
    # Bookmarks + ratings + their summary tables moved to the central
    # server when we extracted the shared identity service. There's
    # nothing left to clean up locally.
    # Datasets attached to CEs.
    ("excitation_datasets", ("dataset_id",)),
    ("calibration_datasets", ("dataset_id",)),
    # Datasets / results attached to classifiers.
    ("evaluation_results", ("eval_id",)),
    ("test_datasets", ("dataset_id",)),
    # Per-classifier rule wiring.
    ("rule_setup", ("setup_id",)),
    # Classifiers and the models they belong to.
    ("classifiers", ("classifier_id",)),
    ("target_models", ("model_id",)),
    # Top-level definitions.
    ("rules", ("rule_id",)),
    ("cognitive_elements", ("ce_id",)),
    # In-flight pipeline state (orphaned by interrupted AI-pipeline tests).
    ("pipeline_runs", ("run_id",)),
    # Users last.
    ("users", ("user_id",)),
]


def _snapshot_pks() -> dict:
    """Capture the set of primary keys currently present in every tracked
    table. Returned dict is `{table_name: set(pk_tuples)}` — each row's
    identity is the tuple of its pk column values, which supports both single-
    column and composite primary keys. Failures are swallowed (return empty
    set) because some tables may be missing in tests that initialise a partial
    schema."""
    from utils.PostgreSQL import execute_query_dict
    snap: dict = {}
    for tbl, pk_cols in _TRACKED_TABLES:
        cols = ", ".join(pk_cols)
        try:
            rows = execute_query_dict(f"SELECT {cols} FROM {tbl}") or []
            snap[tbl] = {tuple(r[c] for c in pk_cols) for r in rows}
        except Exception:
            snap[tbl] = set()
    return snap


def _restore_to(snapshot: dict) -> None:
    """Delete any row whose pk tuple is not in the snapshot. Iterates the
    tracked tables in declared (children-first) order. We make two passes: the
    second catches anything that lingered because of weird FK interactions in
    the first pass. Errors are swallowed per-row so one stuck row never blocks
    the rest of the cleanup."""
    from utils.PostgreSQL import execute_query, execute_query_dict
    for _ in range(2):
        for tbl, pk_cols in _TRACKED_TABLES:
            cols = ", ".join(pk_cols)
            where = " AND ".join(f"{c} = %s" for c in pk_cols)
            try:
                rows = execute_query_dict(f"SELECT {cols} FROM {tbl}") or []
                current = {tuple(r[c] for c in pk_cols) for r in rows}
                stale = current - snapshot.get(tbl, set())
                for pk_values in stale:
                    try:
                        execute_query(
                            f"DELETE FROM {tbl} WHERE {where}", tuple(pk_values)
                        )
                    except Exception:
                        pass
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session", autouse=True)
def _session_baseline():
    """Outer safety net. Snapshot at session start, restore at session end so
    the dev database returns exactly to the state we found it — including
    rows that the session-scoped fixtures (test_user et al.) created."""
    baseline = _snapshot_pks()
    yield baseline
    _restore_to(baseline)


@pytest.fixture(autouse=True)
def _per_test_cleanup(_session_baseline):
    """Inner cleanup. Captures DB state right before the test runs, then on
    teardown deletes anything new. Depending on `_session_baseline` ensures
    the session-scoped fixtures have run by the time we take this snapshot,
    so test_user / test_model rows are part of the per-test baseline and do
    NOT get deleted between tests."""
    pre = _snapshot_pks()
    yield
    _restore_to(pre)


@pytest.fixture(scope="session")
def client():
    """FastAPI TestClient — lives for the whole test session."""
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="session")
def test_user(client):
    """Register a dedicated test user once per session.

    Returns a dict with user_id, username, email, password, token. The user
    is created via the real /user/register endpoint so password hashing,
    token issuance, and DB writes all exercise production code paths.
    """
    suffix = int(time.time()) % 100000
    username = f"testuser_{suffix}"
    email = f"testuser_{suffix}@test.com"
    password = "TestPass123!"

    res = client.post("/user/register", json={
        "username": username,
        "email": email,
        "password": password,
    })
    data = res.json()

    # If the user already exists from a prior run, login instead.
    if res.status_code != 200 or "user_id" not in data:
        res = client.post("/user/login", json={"email": email, "password": password})
        data = res.json()

    user_id = data.get("user_id", 1)
    token = data.get("token") or create_access_token({"sub": str(user_id)})
    return {
        "user_id": user_id,
        "username": username,
        "email": email,
        "password": password,
        "token": token,
    }


@pytest.fixture(scope="session")
def auth_headers(test_user):
    """Authorization headers for authenticated requests."""
    return {"Authorization": f"Bearer {test_user['token']}"}


@pytest.fixture(scope="session")
def test_model(client, test_user, auth_headers):
    """Create a test model once per session using SmolLM2-360M-Instruct."""
    models_res = client.get(f"/models/{test_user['user_id']}", headers=auth_headers)
    models_data = models_res.json()
    models_list = models_data.get("models", models_data) if isinstance(models_data, dict) else models_data
    if isinstance(models_list, list) and models_list:
        return models_list[0]

    res = client.post("/models/create", json={
        "user_id": test_user["user_id"],
        "name": "SmolLM2-Test",
        "storage_path": "HuggingFaceTB/SmolLM2-360M-Instruct",
    }, headers=auth_headers)
    if res.status_code == 200:
        data = res.json()
        return data.get("model", data)
    pytest.skip(f"Could not create test model: {res.status_code} {res.text[:200]}")


@pytest.fixture(scope="session")
def test_classifier(client, test_model, auth_headers):
    """Create a test classifier once per session."""
    model_id = test_model.get("model_id")
    if not model_id:
        pytest.skip("No model_id in test_model")

    cls_res = client.get(f"/classifiers/{model_id}", headers=auth_headers)
    cls_data = cls_res.json()
    cls_list = cls_data if isinstance(cls_data, list) else cls_data.get("classifiers", [])
    if cls_list:
        return cls_list[0]

    res = client.post("/classifiers/create", json={
        "model_id": model_id,
        "name": "TestClassifier",
    }, headers=auth_headers)
    if res.status_code == 200:
        data = res.json()
        return data.get("classifier", data)
    pytest.skip(f"Could not create test classifier: {res.status_code} {res.text[:200]}")
