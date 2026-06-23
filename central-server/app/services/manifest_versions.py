"""Stamp a content version map onto the public-library manifest.

The control-plane watcher serves { global_signature, namespaces } as the version
token clients reconcile against. We derive it from the manifest's OWN record
indices (rules / ces / neutral, which already carry per-record published_at), so
it changes whenever anything is published — and inject it at the single write
chokepoint (/hf/commit), so every publish carries it. No file re-reads needed.

NOTE: these signatures are over the manifest's record INDEX (not file-content
signatures like the backend's registry_sync.signatures) — that's all the watcher
needs to detect "the library changed" and trigger a sync.
"""
import hashlib
import json

SIG_ALGO = "v1"
_DERIVED = ("global_signature", "namespaces")

# public namespace folder -> the manifest record sub-map that tracks it
_NAMESPACE_KEYS = {
    "public_rules": "rules",
    "public_ces": "ces",
    "public_rule_sets": "rule_sets",
    "neutral": "neutral",
}


def _sig(value) -> str:
    """v1:<sha256 of the canonical JSON of `value>`."""
    raw = json.dumps(value if value is not None else {}, sort_keys=True,
                     separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return f"{SIG_ALGO}:{hashlib.sha256(raw).hexdigest()}"


def augment_manifest(manifest: dict) -> dict:
    """Return `manifest` with `namespaces` + `global_signature` stamped in.

    global_signature hashes the WHOLE record manifest (minus the derived fields),
    so ANY publish moves it; the per-namespace signatures give clients a surgical
    map. Idempotent — recomputing from the same records yields the same values."""
    base = {k: v for k, v in manifest.items() if k not in _DERIVED}
    ns_sigs = {ns: _sig(base.get(key)) for ns, key in _NAMESPACE_KEYS.items()}
    manifest["namespaces"] = {n: {"signature": s} for n, s in ns_sigs.items()}
    manifest["global_signature"] = _sig(base)
    return manifest
