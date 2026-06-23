"""Control-plane singletons + lifecycle.

One process-wide provider / WS manager / watcher, wired together. The routes
import these; main.py calls start()/stop() in the app lifecycle. Disabled in
tests via ENABLE_CONTROL_PLANE=0 (so no background thread / DB / HF on import).
"""
from __future__ import annotations

import logging
import os

from .source_provider import HuggingFaceSource
from .source_watcher import SourceWatcher
from .ws_manager import WSManager

logger = logging.getLogger(__name__)

REPO_ID = os.getenv("HF_REPO_ID", "GavelPublicData/public-library")
REPO_TYPE = os.getenv("HF_REPO_TYPE", "dataset")
HF_TOKEN = os.getenv("HF_TOKEN")
WEBHOOK_SECRET = os.getenv("REGISTRY_WEBHOOK_SECRET")

# Tunables (env-overridable) for the watcher's robustness knobs.
DEBOUNCE_S = float(os.getenv("REGISTRY_DEBOUNCE_S", "1.0"))
SAFETY_POLL_S = float(os.getenv("REGISTRY_SAFETY_POLL_S", "300.0"))
HF_TIMEOUT_S = float(os.getenv("REGISTRY_HF_TIMEOUT_S", "5.0"))

WS_MAX_CONN = int(os.getenv("REGISTRY_WS_MAX_CONN", "2000"))

PROVIDER = HuggingFaceSource(REPO_ID, REPO_TYPE, hf_token=HF_TOKEN, webhook_secret=WEBHOOK_SECRET)
WS_MANAGER = WSManager(max_connections=WS_MAX_CONN)


def _on_advance(_state: dict) -> None:
    # Lightweight notification — the client then hits GET /api/v1/versions.
    WS_MANAGER.broadcast_threadsafe({"event": "version_update"})


WATCHER = SourceWatcher(
    PROVIDER, repo=REPO_ID, broadcast=_on_advance,
    debounce_s=DEBOUNCE_S, safety_poll_s=SAFETY_POLL_S, hf_timeout_s=HF_TIMEOUT_S,
)


def start(loop) -> None:
    """Bind the event loop (for thread→socket broadcasts) and start the watcher.
    No-op when ENABLE_CONTROL_PLANE=0 (tests / a deployment that doesn't want it)."""
    if os.getenv("ENABLE_CONTROL_PLANE", "1") == "0":
        logger.info("control plane disabled (ENABLE_CONTROL_PLANE=0)")
        return
    if not WEBHOOK_SECRET:
        logger.warning("REGISTRY_WEBHOOK_SECRET unset — webhooks will be rejected; "
                       "the safety poll still keeps clients converged.")
    WS_MANAGER.set_loop(loop)
    try:
        WATCHER.start()
        logger.info("control plane started (repo=%s)", REPO_ID)
    except Exception as e:
        logger.error(f"control plane start failed: {e}")


def stop() -> None:
    try:
        WATCHER.stop()
    except Exception:
        pass
