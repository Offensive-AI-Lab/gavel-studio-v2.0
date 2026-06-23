"""Unit tests for the manifest version stamping injected at /hf/commit."""
from app.services.manifest_versions import SIG_ALGO, augment_manifest


def test_augment_adds_global_and_namespace_signatures():
    m = augment_manifest({"rules": {"r1": "t1"}, "ces": {"c1": "t1"}, "neutral": {}})
    assert m["global_signature"].startswith(SIG_ALGO + ":")
    assert set(m["namespaces"]) == {"public_rules", "public_ces", "public_rule_sets", "neutral"}
    assert m["namespaces"]["public_rules"]["signature"].startswith("v1:")


def test_global_signature_moves_when_a_record_is_published():
    a = augment_manifest({"rules": {"r1": "t1"}})["global_signature"]
    b = augment_manifest({"rules": {"r1": "t1", "r2": "t2"}})["global_signature"]
    assert a != b


def test_namespace_signature_isolates_the_changed_folder():
    base = augment_manifest({"rules": {"r1": "t1"}, "ces": {"c1": "t1"}})
    moved = augment_manifest({"rules": {"r1": "t1"}, "ces": {"c1": "t2"}})  # only ces changed
    assert base["namespaces"]["public_rules"] == moved["namespaces"]["public_rules"]
    assert base["namespaces"]["public_ces"] != moved["namespaces"]["public_ces"]


def test_recompute_is_idempotent():
    once = augment_manifest({"rules": {"r1": "t1"}, "ces": {}})
    twice = augment_manifest(dict(once))   # re-augment an already-augmented manifest
    assert once["global_signature"] == twice["global_signature"]
    assert once["namespaces"] == twice["namespaces"]
