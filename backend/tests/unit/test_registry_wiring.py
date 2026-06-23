"""Tests for the backend wiring: build_subscriber() gating and that the
reconcile action PROBES freshness (without mutating the DB) and pushes the
badge state to the frontend."""
import pytest

from services.registry_sync import wiring
from services.registry_sync.subscriber import RegistrySyncSubscriber


def test_build_subscriber_none_without_central_url(monkeypatch):
    monkeypatch.delenv("CENTRAL_SERVER_URL", raising=False)
    assert wiring.build_subscriber() is None


def test_build_subscriber_none_when_disabled(monkeypatch):
    monkeypatch.setenv("CENTRAL_SERVER_URL", "https://central.example")
    monkeypatch.setenv("ENABLE_REGISTRY_SUBSCRIBER", "0")
    assert wiring.build_subscriber() is None


def test_build_subscriber_returns_wired_subscriber(monkeypatch):
    monkeypatch.setenv("CENTRAL_SERVER_URL", "https://central.example/")
    monkeypatch.delenv("ENABLE_REGISTRY_SUBSCRIBER", raising=False)
    sub = wiring.build_subscriber()
    assert isinstance(sub, RegistrySyncSubscriber)
    assert sub.ws_url == "wss://central.example/api/v1/ws"
    # The notification socket is PUBLIC — no credential is captured/held here.
    assert sub.token_provider is None


def test_notifier_checks_and_pushes_badge_without_mutating(monkeypatch):
    seen = []
    monkeypatch.setattr("services.hf_sync.check_for_updates",
                        lambda: {"available": True, "checked": True, "reason": None})
    monkeypatch.setattr("services.library_events.set_available",
                        lambda available: seen.append(available))
    # The notifier must NEVER pull records — that's the user's click.
    def _no_sync(*a, **k):
        raise AssertionError("reconcile must not sync_library")
    monkeypatch.setattr("services.hf_sync.sync_library", _no_sync)

    out = wiring._LibraryUpdateNotifier().reconcile()

    assert out["available"] is True
    assert seen == [True]


def test_notifier_pushes_synced_when_up_to_date(monkeypatch):
    seen = []
    monkeypatch.setattr("services.hf_sync.check_for_updates",
                        lambda: {"available": False, "checked": True, "reason": None})
    monkeypatch.setattr("services.library_events.set_available",
                        lambda available: seen.append(available))
    wiring._LibraryUpdateNotifier().reconcile()
    assert seen == [False]


def test_built_subscriber_checks_freshness_on_version_update(monkeypatch):
    """Integration: the REAL wired subscriber, fed a version_update over a fake
    socket, runs the real notifier -> check_for_updates (no DB mutation)."""
    import asyncio
    import json
    from contextlib import asynccontextmanager

    monkeypatch.setenv("CENTRAL_SERVER_URL", "http://central")
    monkeypatch.delenv("ENABLE_REGISTRY_SUBSCRIBER", raising=False)
    checks = []
    monkeypatch.setattr("services.hf_sync.check_for_updates",
                        lambda: checks.append(1) or {"available": False, "checked": True})
    monkeypatch.setattr("services.library_events.set_available", lambda available: None)

    sub = wiring.build_subscriber()
    assert sub is not None

    class FakeWS:
        def __aiter__(self):
            self._msgs = [json.dumps({"event": "version_update"})]
            return self

        async def __anext__(self):
            if self._msgs:
                return self._msgs.pop(0)
            raise StopAsyncIteration

    @asynccontextmanager
    async def fake_connect(url):
        assert url.endswith("/api/v1/ws")   # public socket, no token
        yield FakeWS()
        sub._stop.set()

    sub._connect = fake_connect

    async def run():
        await sub.start()
        await asyncio.wait_for(sub.wait(), timeout=3)

    asyncio.run(run())
    # reconnect-reconcile + the version_update both ran the freshness check
    assert sum(checks) >= 2
