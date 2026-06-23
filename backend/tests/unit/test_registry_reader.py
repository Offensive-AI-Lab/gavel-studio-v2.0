"""Tests for the read-side port + HuggingFaceReader adapter.

`hf_hub_download` / `HfApi` are monkeypatched, so these exercise the adapter's
real logic (path handling, not-found mapping, json convenience) with no network.
"""
import json

import pytest

from services.registry_sync.reader import (
    HuggingFaceReader,
    RegistryNotFound,
    RegistryReader,
    RegistryReadError,
    build_reader,
)


def _reader():
    return HuggingFaceReader("repo/lib", "dataset")


def test_adapter_conforms_to_the_port():
    assert isinstance(_reader(), RegistryReader)


def test_fetch_bytes_returns_file_contents(monkeypatch, tmp_path):
    f = tmp_path / "manifest.json"
    f.write_bytes(b'{"rules":{}}')
    monkeypatch.setattr("huggingface_hub.hf_hub_download", lambda **kw: str(f))

    assert _reader().fetch_bytes("manifest.json") == b'{"rules":{}}'


def test_fetch_json_parses_via_fetch_bytes(monkeypatch, tmp_path):
    f = tmp_path / "rec.json"
    f.write_text(json.dumps({"id": 1, "name": "x"}), encoding="utf-8")
    monkeypatch.setattr("huggingface_hub.hf_hub_download", lambda **kw: str(f))

    assert _reader().fetch_json("public_rules/rec.json") == {"id": 1, "name": "x"}


def test_missing_file_maps_to_registry_not_found(monkeypatch):
    def _missing(**kw):
        raise RuntimeError("404 Client Error: EntryNotFound")
    monkeypatch.setattr("huggingface_hub.hf_hub_download", _missing)

    with pytest.raises(RegistryNotFound):
        _reader().fetch_bytes("public_ces/nope.json")


def test_other_transport_error_maps_to_read_error(monkeypatch):
    def _boom(**kw):
        raise RuntimeError("connection reset")
    monkeypatch.setattr("huggingface_hub.hf_hub_download", _boom)

    with pytest.raises(RegistryReadError):
        _reader().fetch_bytes("manifest.json")


def test_head_version_reads_repo_sha(monkeypatch):
    class _Info:
        sha = "abc123"

    class _Api:
        def __init__(self, *a, **k):
            pass

        def repo_info(self, **kw):
            return _Info()

    monkeypatch.setattr("huggingface_hub.HfApi", _Api)
    assert _reader().head_version() == "abc123"


def test_fetch_json_is_provided_on_any_adapter():
    # A trivial in-memory adapter only implements fetch_bytes/head_version;
    # fetch_json must work for free from the base class.
    class MemReader(RegistryReader):
        name = "mem"

        def head_version(self):
            return "v0"

        def fetch_bytes(self, path, *, revision=None):
            return b'{"ok": true}'

    assert MemReader().fetch_json("anything.json") == {"ok": True}


# --------------------------------------------------------------------------- #
# factory — returns the active reader (swap is a code change, not env)
# --------------------------------------------------------------------------- #
def test_build_reader_returns_huggingface():
    r = build_reader()
    assert isinstance(r, HuggingFaceReader) and r.name == "huggingface"
