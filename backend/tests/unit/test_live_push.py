"""Tests for the backend -> frontend live push and the publisher's
no-phantom-update fix.

Covers three pieces:
  * library_events — the in-process SSE bus (subscribe / publish / drop-oldest)
    and set_available()'s `update_available` / `synced` events.
  * the registry notifier — PROBES freshness (never pulls) and pushes the badge
    state, so the user applies updates on their click and a publisher's own
    commit comes back "synced".
  * _record_pushed_manifest_hash — caches the central server's authoritative
    (post-stamp) manifest hash, falling back to the local hash if absent.
"""
import asyncio
import hashlib

from services import library_events as bus
from services.registry_sync import wiring


# --------------------------------------------------------------------------- #
# library_events bus
# --------------------------------------------------------------------------- #
def test_publish_is_noop_without_subscribers():
    # No subscribers (and possibly no remembered loop) -> must not raise.
    bus.publish({"event": "library_updated"})
    assert bus.subscriber_count() == 0


def test_subscribe_publish_delivers_then_unsubscribe():
    async def run():
        q = bus.subscribe()
        assert bus.subscriber_count() == 1
        bus.publish({"event": "library_updated", "n": 1})
        evt = await asyncio.wait_for(q.get(), timeout=1)
        bus.unsubscribe(q)
        return evt

    evt = asyncio.run(run())
    assert evt == {"event": "library_updated", "n": 1}
    assert bus.subscriber_count() == 0


def test_full_queue_drops_oldest_keeps_newest():
    async def run():
        q = bus.subscribe()
        for i in range(bus._QUEUE_MAXSIZE + 5):
            bus.publish({"i": i})
        await asyncio.sleep(0.05)  # let the call_soon_threadsafe callbacks run
        items = []
        while not q.empty():
            items.append(q.get_nowait())
        bus.unsubscribe(q)
        return items

    items = asyncio.run(run())
    assert len(items) == bus._QUEUE_MAXSIZE          # capped, never unbounded
    assert items[0]["i"] == 5                        # oldest five dropped
    assert items[-1]["i"] == bus._QUEUE_MAXSIZE + 4  # newest survived


def test_set_available_updates_state_and_emits():
    async def run():
        q = bus.subscribe()
        bus.set_available(True)
        evt = await asyncio.wait_for(q.get(), timeout=1)
        state = bus.current_state()
        bus.unsubscribe(q)
        return evt, state

    evt, state = asyncio.run(run())
    assert evt == {"event": "update_available"}
    assert state["available"] is True
    bus.set_available(False)  # reset shared state for other tests
    assert bus.current_state()["available"] is False


# --------------------------------------------------------------------------- #
# notifier — probe freshness (never pull) and push the badge state
# --------------------------------------------------------------------------- #
def test_notifier_flags_available(monkeypatch):
    monkeypatch.setattr("services.hf_sync.check_for_updates",
                        lambda: {"available": True, "checked": True, "reason": None})
    seen = []
    monkeypatch.setattr("services.library_events.set_available",
                        lambda available: seen.append(available))

    out = wiring._LibraryUpdateNotifier().reconcile()

    assert out["available"] is True
    assert seen == [True]


def test_notifier_flags_synced_and_never_pulls(monkeypatch):
    monkeypatch.setattr("services.hf_sync.check_for_updates",
                        lambda: {"available": False, "checked": True, "reason": None})
    seen = []
    monkeypatch.setattr("services.library_events.set_available",
                        lambda available: seen.append(available))

    def _no_sync(*a, **k):
        raise AssertionError("notifier must not sync_library")
    monkeypatch.setattr("services.hf_sync.sync_library", _no_sync)

    wiring._LibraryUpdateNotifier().reconcile()

    assert seen == [False]


# --------------------------------------------------------------------------- #
# publisher hash — prefer the central server's authoritative stamped hash
# --------------------------------------------------------------------------- #
def test_record_pushed_manifest_hash_prefers_returned_sha(monkeypatch):
    from services import hf_publish

    captured = []
    monkeypatch.setattr("services.hf_publish.execute_query",
                        lambda sql, params: captured.append(params))

    hf_publish._record_pushed_manifest_hash(
        {"status": "success", "manifest_sha256": "deadbeef"}, {"rules": {}})

    assert captured and captured[0] == ("deadbeef",)


def test_record_pushed_manifest_hash_falls_back_to_local(monkeypatch):
    from services import hf_publish

    captured = []
    monkeypatch.setattr("services.hf_publish.execute_query",
                        lambda sql, params: captured.append(params))

    manifest = {"rules": {"r": "t"}}
    hf_publish._record_pushed_manifest_hash({"status": "success"}, manifest)  # no sha

    expected = hashlib.sha256(hf_publish._to_bytes(manifest)).hexdigest()
    assert captured and captured[0] == (expected,)
