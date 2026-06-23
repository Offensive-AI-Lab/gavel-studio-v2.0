"""Unit tests for the SourceWatcher reconcile logic.

A FakeProvider stands in for HF; persistence + broadcast are injected, so these
run with no DB, no network, and no threads — they call reconcile_now() directly
and assert the four behaviours that matter: advance+broadcast on a new
global_signature, dedup (no broadcast) on an unchanged one, graceful abort on HF
failure/timeout, and "persist failed → don't advance or broadcast".
"""
import time

from app.services.source_watcher import SourceWatcher


class FakeProvider:
    def __init__(self, head="sha1", manifest=None, *, fail=False, delay=0.0):
        self._head = head
        self._manifest = manifest if manifest is not None else {}
        self.fail = fail
        self.delay = delay

    def head_version(self):
        if self.delay:
            time.sleep(self.delay)
        if self.fail:
            raise RuntimeError("HF down")
        return self._head

    def fetch_manifest(self, revision=None):
        if self.fail:
            raise RuntimeError("HF down")
        return self._manifest


def _watcher(provider, saved, broadcasts, *, loaded=None, save=None, **kw):
    return SourceWatcher(
        provider, repo="GavelPublicData/public-library",
        broadcast=broadcasts.append,
        load_state=lambda: loaded,
        save_state=save or saved.append,
        hf_timeout_s=kw.pop("hf_timeout_s", 2.0),
        lock_timeout_s=2.0, debounce_s=0.0, safety_poll_s=999.0,
    )


MANIFEST_A = {"global_signature": "v1:aaa",
              "namespaces": {"public_rules": {"signature": "v1:r"}}}


def test_advances_and_broadcasts_on_new_global_signature():
    saved, broadcasts = [], []
    w = _watcher(FakeProvider("sha1", MANIFEST_A), saved, broadcasts)

    assert w.reconcile_now() is True
    assert w.current_versions["commit"] == "sha1"
    assert w.current_versions["global_signature"] == "v1:aaa"
    assert w.current_versions["namespaces"] == {"public_rules": {"signature": "v1:r"}}
    assert len(saved) == 1 and saved[0]["global_signature"] == "v1:aaa"
    assert len(broadcasts) == 1


def test_dedup_unchanged_global_does_not_broadcast():
    saved, broadcasts = [], []
    w = _watcher(FakeProvider("sha1", MANIFEST_A), saved, broadcasts)
    assert w.reconcile_now() is True            # first advance
    assert w.reconcile_now() is False           # same global → no advance
    assert len(broadcasts) == 1                 # NOT re-broadcast (self-push safe)


def test_graceful_abort_when_hf_is_down():
    saved, broadcasts = [], []
    w = _watcher(FakeProvider(fail=True), saved, broadcasts)
    assert w.reconcile_now() is False
    assert w.current_versions["global_signature"] == ""   # state untouched
    assert broadcasts == [] and saved == []


def test_graceful_abort_on_hf_timeout():
    saved, broadcasts = [], []
    w = _watcher(FakeProvider("sha1", MANIFEST_A, delay=0.5), saved, broadcasts,
                 hf_timeout_s=0.1)
    assert w.reconcile_now() is False           # strict timeout fired
    assert broadcasts == []


def test_persist_failure_does_not_advance_or_broadcast():
    broadcasts = []

    def bad_save(_state):
        raise RuntimeError("db down")
    w = _watcher(FakeProvider("sha1", MANIFEST_A), [], broadcasts, save=bad_save)

    assert w.reconcile_now() is False
    assert w.current_versions["global_signature"] == ""   # not advanced
    assert broadcasts == []


def test_manifest_without_global_signature_is_skipped():
    saved, broadcasts = [], []
    w = _watcher(FakeProvider("sha1", {"namespaces": {}}), saved, broadcasts)
    assert w.reconcile_now() is False
    assert broadcasts == []


def test_persisted_state_is_loaded_on_start():
    # Survives a reboot: load_state returns a persisted map; a provider that fails
    # means the boot reconcile aborts, so the loaded state is what we keep.
    persisted = {"commit": "old", "global_signature": "v1:old", "namespaces": {"x": 1}}
    saved, broadcasts = [], []
    w = _watcher(FakeProvider(fail=True), saved, broadcasts, loaded=persisted)
    try:
        w.start()
        assert w.current_versions == persisted
    finally:
        w.stop()
