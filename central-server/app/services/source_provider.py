"""Source reader for the control plane (read + webhook surface only).

The central server is a separate deployable from the local backend, so it can't
import the backend's `registry_sync` package — but it doesn't need to. The watcher
only READS (head + manifest), and the manifest already carries `global_signature`
and the per-namespace map. This reader mirrors the SAME port shape (head_version /
fetch_manifest / verify_and_normalize_webhook) so it stays GitHub-swappable.

Writes (the CAS commit) are NOT here — they go through the existing /hf/commit
proxy. This is read + doorbell only.
"""
from __future__ import annotations

import hmac
import json
import logging
from dataclasses import dataclass, field
from typing import Mapping, Optional

logger = logging.getLogger(__name__)


class WebhookRejected(Exception):
    """Raised when a webhook fails authenticity verification."""


@dataclass(frozen=True)
class ChangeEvent:
    """A doorbell, not data. `version` is best-effort (HF doesn't reliably carry
    the new SHA) — the watcher treats it as a trigger and reconciles to HEAD."""
    source: str
    repo: str
    version: str = ""
    raw: Mapping = field(default_factory=dict)


class HuggingFaceSource:
    name = "huggingface"

    def __init__(self, repo_id: str, repo_type: str, *,
                 hf_token: Optional[str] = None, webhook_secret: Optional[str] = None):
        self.repo_id = repo_id
        self.repo_type = repo_type
        self.hf_token = hf_token
        self.webhook_secret = webhook_secret

    def head_version(self) -> str:
        from huggingface_hub import HfApi
        api = HfApi(token=self.hf_token)
        return api.repo_info(repo_id=self.repo_id, repo_type=self.repo_type).sha

    def fetch_manifest(self, revision: Optional[str] = None) -> dict:
        from huggingface_hub import hf_hub_download
        try:
            path = hf_hub_download(repo_id=self.repo_id, repo_type=self.repo_type,
                                   filename="manifest.json", revision=revision,
                                   token=self.hf_token)
        except Exception as e:
            if "404" in str(e) or "EntryNotFound" in type(e).__name__:
                return {}
            raise
        with open(path, "rb") as f:
            manifest = json.loads(f.read().decode("utf-8"))
        # Derive the version map on read (idempotent) so a manifest published
        # BEFORE this feature — which has no global_signature — still drives the
        # watcher. No backfill/migration of existing data is required.
        from .manifest_versions import augment_manifest
        return augment_manifest(manifest)

    def verify_and_normalize_webhook(self, headers: Mapping[str, str],
                                     raw_body: bytes) -> ChangeEvent:
        lower = {str(k).lower(): v for k, v in dict(headers).items()}
        got = lower.get("x-webhook-secret")
        if not self.webhook_secret or not got or \
                not hmac.compare_digest(str(got), str(self.webhook_secret)):
            raise WebhookRejected("invalid or missing webhook secret")
        try:
            payload = json.loads(raw_body.decode("utf-8"))
        except Exception:
            raise WebhookRejected("webhook body is not valid JSON")
        repo = (payload.get("repo") or {}).get("name") or self.repo_id
        return ChangeEvent(source=self.name, repo=repo, version="", raw=payload)
