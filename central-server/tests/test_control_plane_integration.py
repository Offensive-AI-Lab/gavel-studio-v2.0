"""Integration: the REAL central loop wired together.

No mocks between the components — a real SourceWatcher reconciles a manifest (as
produced by augment_manifest at the commit chokepoint), advancing through the real
WSManager's thread-safe broadcaster to a connected socket. Proves a publish-shaped
change actually reaches a client, and that an unchanged manifest does not.
"""
import asyncio

from app.services.manifest_versions import augment_manifest
from app.services.source_watcher import SourceWatcher
from app.services.ws_manager import WSManager


class FakeWS:
    def __init__(self):
        self.sent = []

    async def accept(self):
        pass

    async def close(self, code=None):
        pass

    async def send_json(self, msg):
        self.sent.append(msg)


class FakeProvider:
    def __init__(self, manifest, head="sha1"):
        self._manifest = manifest
        self.head = head

    def head_version(self):
        return self.head

    def fetch_manifest(self, revision=None):
        return self._manifest


def _wire(manifest):
    mgr = WSManager()
    saved = []
    watcher = SourceWatcher(
        FakeProvider(manifest), repo="GavelPublicData/public-library",
        broadcast=lambda _state: mgr.broadcast_threadsafe({"event": "version_update"}),
        load_state=lambda: None, save_state=saved.append, hf_timeout_s=2,
    )
    return mgr, watcher, saved


def test_publish_change_broadcasts_to_a_connected_socket():
    manifest = augment_manifest({"rules": {"r1": "t1"}, "ces": {}, "neutral": {}})
    mgr, watcher, saved = _wire(manifest)
    ws = FakeWS()

    async def scenario():
        mgr.set_loop(asyncio.get_running_loop())
        await mgr.connect(ws)
        advanced = await asyncio.to_thread(watcher.reconcile_now)   # runs on a thread, like prod
        await asyncio.sleep(0.05)   # let the cross-thread broadcast run on the loop
        return advanced

    advanced = asyncio.run(scenario())
    assert advanced is True
    assert ws.sent == [{"event": "version_update"}]
    assert saved and saved[0]["global_signature"] == manifest["global_signature"]


def test_unchanged_manifest_does_not_rebroadcast():
    manifest = augment_manifest({"rules": {"r1": "t1"}})
    mgr, watcher, _saved = _wire(manifest)
    ws = FakeWS()

    async def scenario():
        mgr.set_loop(asyncio.get_running_loop())
        await mgr.connect(ws)
        await asyncio.to_thread(watcher.reconcile_now)            # advance + broadcast
        await asyncio.sleep(0.05)
        second = await asyncio.to_thread(watcher.reconcile_now)   # same manifest → dedup
        await asyncio.sleep(0.05)
        return second

    second = asyncio.run(scenario())
    assert second is False
    assert ws.sent == [{"event": "version_update"}]               # exactly once
