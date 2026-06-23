"""Unit tests for the classifier bundle FORMAT (DB-free paths only).

Covers the parts of services.classifier_bundle that don't touch the database:
the machine-independent rule fingerprint, the hardened zip extractor, and the
full parse/validate read path (including a real TopicRNN strict load and the
integrity/tamper checks). The build/import sides need a live DB + classifier and
are exercised by integration tests.
"""
import io
import json
import zipfile

import pytest

from services import classifier_bundle as cb


# ---------------------------------------------------------------------------
# Fingerprint
# ---------------------------------------------------------------------------

def test_rule_fingerprint_is_order_independent():
    a = [
        {"ce_public_id": "ce_2", "role": "necessary", "fallback_group": 0},
        {"ce_public_id": "ce_1", "role": "necessary", "fallback_group": 0},
        {"ce_public_id": "ce_3", "role": "fallback", "fallback_group": 1},
    ]
    b = list(reversed(a))
    assert cb._rule_fp_from_public_links(a) == cb._rule_fp_from_public_links(b)


def test_rule_fingerprint_distinguishes_roles():
    base = [{"ce_public_id": "ce_1", "role": "necessary", "fallback_group": 0}]
    other = [{"ce_public_id": "ce_1", "role": "sufficient", "fallback_group": 0}]
    assert cb._rule_fp_from_public_links(base) != cb._rule_fp_from_public_links(other)


# ---------------------------------------------------------------------------
# Safe extraction
# ---------------------------------------------------------------------------

def _zip_of(files: dict) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in files.items():
            zf.writestr(name, data)
    return buf.getvalue()


def test_safe_extract_rejects_non_zip():
    with pytest.raises(cb.BundleError):
        cb._safe_extract(b"not a zip at all")


def test_safe_extract_rejects_path_traversal():
    with pytest.raises(cb.BundleError):
        cb._safe_extract(_zip_of({"../evil.json": b"{}"}))


def test_safe_extract_reads_clean_zip():
    out = cb._safe_extract(_zip_of({"a.json": b"{}", "sub/b.txt": b"hi"}))
    assert out["a.json"] == b"{}"
    assert out["sub/b.txt"] == b"hi"


# ---------------------------------------------------------------------------
# Full parse/validate round-trip with a real model
# ---------------------------------------------------------------------------

def _build_valid_bundle(tamper_meta: bool = False) -> bytes:
    """A minimal but genuine bundle: a tiny TopicRNN's state_dict + matching
    meta + manifest with correct integrity hashes. `tamper_meta` flips a byte in
    the meta AFTER hashing, to prove the integrity check fires."""
    torch = pytest.importorskip("torch")
    from classifier_engine.RNN import TopicRNN

    meta = {
        "labels": {"a": 0, "b": 1, "c": 2},
        "readout_dim": 4,
        "n_layers": 2,
        "hidden_dim": 8,
        "num_rnn_layers": 1,
        "num_classes": 3,
        "selected_layers": [0, 2],
        "rnn_sequence_length": 5,
        "model_path": "some/base-model",
    }
    rnn = TopicRNN(input_dim=4, num_layers=2, hidden_dim=8,
                   num_rnn_layers=1, num_topics=3, rnn_type="GRU")
    pth_buf = io.BytesIO()
    torch.save(rnn.state_dict(), pth_buf)
    pth_bytes = pth_buf.getvalue()

    meta_bytes = json.dumps(meta).encode("utf-8")
    files = {
        "model/trained_rnn.pth": pth_bytes,
        "model/classifier_meta.json": meta_bytes,
    }
    integrity = {p: cb._sha256(b) for p, b in files.items()}

    if tamper_meta:
        # Corrupt the meta payload but keep the (now-wrong) hash in the manifest.
        files["model/classifier_meta.json"] = meta_bytes + b" "

    manifest = {
        "format": cb.FORMAT,
        "format_version": cb.FORMAT_VERSION,
        "tier": cb.TIER_MODEL,
        "base_model": {"storage_path": "some/base-model", "display_name": "Base"},
        "source": {"classifier_name": "demo"},
        "policy": {"trained_policy_fingerprint": "", "ces": [], "rules": []},
        "integrity": integrity,
    }
    files["bundle_manifest.json"] = json.dumps(manifest).encode("utf-8")
    return _zip_of(files)


def test_parse_valid_bundle_roundtrip():
    pytest.importorskip("torch")
    parsed = cb.parse_and_validate_bundle(_build_valid_bundle())
    assert parsed["tier"] == cb.TIER_MODEL
    assert parsed["manifest"]["format"] == cb.FORMAT
    assert parsed["meta"]["num_classes"] == 3


def test_parse_rejects_tampered_file():
    pytest.importorskip("torch")
    with pytest.raises(cb.BundleError) as ei:
        cb.parse_and_validate_bundle(_build_valid_bundle(tamper_meta=True))
    assert "integrity" in str(ei.value).lower()


def test_parse_rejects_non_bundle_zip():
    with pytest.raises(cb.BundleError):
        cb.parse_and_validate_bundle(_zip_of({"random.txt": b"hello"}))


def test_parse_rejects_wrong_format_marker():
    bad = _zip_of({"bundle_manifest.json": json.dumps({"format": "something-else"}).encode()})
    with pytest.raises(cb.BundleError):
        cb.parse_and_validate_bundle(bad)


def test_validate_model_loads_rejects_geometry_mismatch():
    """A state_dict that doesn't fit the declared architecture must be rejected
    before any DB write."""
    torch = pytest.importorskip("torch")
    from classifier_engine.RNN import TopicRNN

    rnn = TopicRNN(input_dim=4, num_layers=2, hidden_dim=8,
                   num_rnn_layers=1, num_topics=3, rnn_type="GRU")
    buf = io.BytesIO()
    torch.save(rnn.state_dict(), buf)

    wrong_meta = {  # hidden_dim differs → strict load fails
        "readout_dim": 4, "n_layers": 2, "hidden_dim": 16,
        "num_rnn_layers": 1, "num_classes": 3,
    }
    with pytest.raises(cb.BundleError):
        cb._validate_model_loads(wrong_meta, buf.getvalue())


# ---------------------------------------------------------------------------
# Base-model resolution (exact path, then unambiguous basename fallback)
# ---------------------------------------------------------------------------

def test_base_model_key_normalizes_paths():
    # HF repo id and a local folder ending in the same name both reduce equal.
    assert cb._base_model_key("meta-llama/Llama-2-7b-chat") == "llama-2-7b-chat"
    assert cb._base_model_key("D:/models/meta-llama/Llama-2-7b-chat") == "llama-2-7b-chat"
    assert cb._base_model_key("models/huggingface/distilbert-base-uncased") == "distilbert-base-uncased"
    assert cb._base_model_key("distilbert-base-uncased") == "distilbert-base-uncased"
    assert cb._base_model_key("meta-llama/Llama-2-7b-chat/") == "llama-2-7b-chat"
    assert cb._base_model_key("") == ""


def _patch_models(monkeypatch, *, exact, all_rows):
    """Fake execute_query_dict: the exact lookup carries 'storage_path = %s'."""
    def fake(query, params=None):
        return exact if "storage_path = %s" in query else all_rows
    monkeypatch.setattr(cb, "execute_query_dict", fake)


def test_resolve_base_model_prefers_exact(monkeypatch):
    _patch_models(
        monkeypatch,
        exact=[{"model_id": 7, "name": "L", "storage_path": "meta-llama/Llama-2-7b-chat"}],
        all_rows=[],
    )
    assert cb._resolve_base_model(1, "meta-llama/Llama-2-7b-chat")["model_id"] == 7


def test_resolve_base_model_basename_fallback(monkeypatch):
    # Same model, different storage_path (HF id vs local folder) → matched.
    _patch_models(
        monkeypatch,
        exact=[],
        all_rows=[{"model_id": 9, "name": "L", "storage_path": "D:/models/meta-llama/Llama-2-7b-chat"}],
    )
    assert cb._resolve_base_model(1, "meta-llama/Llama-2-7b-chat")["model_id"] == 9


def test_resolve_base_model_ambiguous_basename_blocks(monkeypatch):
    # Two models share the basename → don't guess; refuse to bind.
    _patch_models(
        monkeypatch,
        exact=[],
        all_rows=[
            {"model_id": 1, "name": "a", "storage_path": "meta-llama/Llama-2-7b-chat"},
            {"model_id": 2, "name": "b", "storage_path": "other-org/Llama-2-7b-chat"},
        ],
    )
    assert cb._resolve_base_model(1, "someorg/Llama-2-7b-chat") is None


def test_resolve_base_model_no_match(monkeypatch):
    _patch_models(
        monkeypatch,
        exact=[],
        all_rows=[{"model_id": 1, "name": "a", "storage_path": "meta-llama/other-model"}],
    )
    assert cb._resolve_base_model(1, "meta-llama/Llama-2-7b-chat") is None
