"""HuggingFaceSource (central read side) — chiefly that fetch_manifest derives the
version map on read, so a LEGACY manifest (published before this feature, with no
global_signature) still drives the watcher without any backfill/migration."""
import json

from app.services.source_provider import HuggingFaceSource


def test_fetch_manifest_augments_a_legacy_manifest(monkeypatch, tmp_path):
    # A pre-feature manifest: real record indices, but NO global_signature/namespaces.
    legacy = tmp_path / "manifest.json"
    legacy.write_text(json.dumps({"rules": {"r1": "t1"}, "ces": {}, "neutral": {}}),
                      encoding="utf-8")
    monkeypatch.setattr("huggingface_hub.hf_hub_download", lambda **kw: str(legacy))

    m = HuggingFaceSource("repo/lib", "dataset").fetch_manifest("rev")

    assert m["rules"] == {"r1": "t1"}                      # existing data untouched
    assert m["global_signature"].startswith("v1:")        # derived on read
    assert m["namespaces"]["public_rules"]["signature"].startswith("v1:")


def test_fetch_manifest_missing_returns_empty(monkeypatch):
    def _missing(**kw):
        raise RuntimeError("404 EntryNotFound")
    monkeypatch.setattr("huggingface_hub.hf_hub_download", _missing)
    assert HuggingFaceSource("repo/lib", "dataset").fetch_manifest("rev") == {}
