"""Route tests for the control-plane endpoints (GET /versions, POST /webhook, WS).

The background watcher is OFF (ENABLE_CONTROL_PLANE=0 in conftest), so we drive
the module singletons directly: set the watcher's state, stub its trigger, and
toggle the provider's webhook secret.
"""
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services import control_plane as cp


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


# --------------------------------------------------------------------------- #
# GET /api/v1/versions
# --------------------------------------------------------------------------- #
def test_versions_returns_map_with_etag(client, monkeypatch):
    state = {"commit": "sha9", "global_signature": "v1:abc",
             "namespaces": {"public_rules": {"signature": "v1:r"}}}
    monkeypatch.setattr(cp.WATCHER, "_state", state)

    r = client.get("/api/v1/versions")
    assert r.status_code == 200
    assert r.json() == state
    assert r.headers["etag"] == 'W/"v1:abc"'


def test_versions_304_on_matching_if_none_match(client, monkeypatch):
    monkeypatch.setattr(cp.WATCHER, "_state",
                        {"commit": "s", "global_signature": "v1:abc", "namespaces": {}})
    r = client.get("/api/v1/versions", headers={"If-None-Match": 'W/"v1:abc"'})
    assert r.status_code == 304


# --------------------------------------------------------------------------- #
# POST /api/v1/webhook  (doorbell only)
# --------------------------------------------------------------------------- #
def test_webhook_valid_secret_triggers_watcher(client, monkeypatch):
    calls = []
    monkeypatch.setattr(cp.PROVIDER, "webhook_secret", "sekret")
    monkeypatch.setattr(cp.WATCHER, "trigger", lambda: calls.append(1))

    r = client.post("/api/v1/webhook", headers={"X-Webhook-Secret": "sekret"},
                    content=b'{"repo": {"name": "GavelPublicData/public-library"}}')
    assert r.status_code == 200 and r.json() == {"ok": True}
    assert calls == [1]


def test_webhook_bad_secret_is_rejected_and_does_not_trigger(client, monkeypatch):
    calls = []
    monkeypatch.setattr(cp.PROVIDER, "webhook_secret", "sekret")
    monkeypatch.setattr(cp.WATCHER, "trigger", lambda: calls.append(1))

    r = client.post("/api/v1/webhook", headers={"X-Webhook-Secret": "wrong"}, content=b"{}")
    assert r.status_code == 401
    r = client.post("/api/v1/webhook", content=b"{}")   # no secret header
    assert r.status_code == 401
    assert calls == []


# --------------------------------------------------------------------------- #
# WS /api/v1/ws  (PUBLIC — non-sensitive version_update signal, no JWT required)
# --------------------------------------------------------------------------- #
def test_ws_is_public_no_token_required(client):
    with client.websocket_connect("/api/v1/ws") as ws:
        assert cp.WS_MANAGER.count() >= 1   # connected + registered, no token
    # context exit disconnects → manager prunes it
