"""App-level tests for the central server: the health/root endpoints and the
three pieces of middleware every request passes through — the request-size cap,
the security headers, and the CORS origin policy.

These run against a real FastAPI TestClient. No database is needed: the schema
init on startup is wrapped in try/except (it just logs when DATABASE_URL is
unset), and none of the routes exercised here issue a query.
"""
import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# Liveness endpoints
# ---------------------------------------------------------------------------


class TestEndpoints:
    def test_health(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json() == {"status": "ok"}

    def test_root(self, client):
        r = client.get("/")
        assert r.status_code == 200
        body = r.json()
        assert body["service"] == "gavel-central-server"
        assert body["status"] == "ok"


# ---------------------------------------------------------------------------
# Security headers — applied to every response
# ---------------------------------------------------------------------------


class TestSecurityHeaders:
    def test_hardening_headers_present(self, client):
        h = client.get("/health").headers
        assert h["X-Content-Type-Options"] == "nosniff"
        assert h["X-Frame-Options"] == "DENY"
        assert h["Referrer-Policy"] == "no-referrer"

    def test_hsts_is_set(self, client):
        # This server is internet-facing over real HTTPS, so HSTS is correct
        # here (unlike the local http backend).
        hsts = client.get("/health").headers["Strict-Transport-Security"]
        assert "max-age=" in hsts
        assert "includeSubDomains" in hsts


# ---------------------------------------------------------------------------
# Request-size cap — reject oversized bodies before they're read into memory
# ---------------------------------------------------------------------------


class TestRequestSizeCap:
    def test_oversized_body_is_rejected_with_413(self, client, monkeypatch):
        from app import main
        monkeypatch.setattr(main, "_MAX_REQUEST_BYTES", 5)
        r = client.post("/health", content=b"way more than five bytes")
        assert r.status_code == 413
        assert "too large" in r.json()["detail"].lower()

    def test_within_limit_passes_the_cap(self, client, monkeypatch):
        from app import main
        monkeypatch.setattr(main, "_MAX_REQUEST_BYTES", 1024)
        # Small body clears the size middleware; it then reaches routing, where
        # /health (GET-only) answers 405 — the point is it is NOT a 413.
        r = client.post("/health", content=b"tiny")
        assert r.status_code != 413


# ---------------------------------------------------------------------------
# CORS — localhost always allowed, the open internet is not
# ---------------------------------------------------------------------------


class TestCors:
    def test_localhost_origin_is_allowed(self, client):
        r = client.get("/health", headers={"Origin": "http://localhost:5173"})
        assert r.headers.get("access-control-allow-origin") == "http://localhost:5173"

    def test_127_0_0_1_origin_is_allowed(self, client):
        r = client.get("/health", headers={"Origin": "http://127.0.0.1:8000"})
        assert r.headers.get("access-control-allow-origin") == "http://127.0.0.1:8000"

    def test_unknown_origin_gets_no_cors_header(self, client):
        # An arbitrary internet origin is not in ALLOWED_ORIGINS and does not
        # match the localhost regex, so no ACAO header is returned to it.
        r = client.get("/health", headers={"Origin": "https://evil.example.com"})
        assert r.headers.get("access-control-allow-origin") is None
