"""Factory + wiring for the client-side registry subscriber (local backend).

`build_subscriber()` assembles the live observer and returns it (or None when the
backend isn't pointed at a central server). Its `.reconcile()` action PROBES the
registry on every central `version_update` (and on each reconnect) and tells the
frontend whether this backend is behind — so the sidebar surfaces a "click to
sync" badge the instant an update is published.

It deliberately does NOT apply the update: pulling records mid-session would
silently change the user's library underneath them. The user applies updates on
their own click (sidebar indicator / manual sync); login still does its own
fire-and-forget sync for a fresh start.

The central notification socket is PUBLIC (the version_update signal is
non-sensitive), so the subscriber connects with NO credential — we deliberately do
not capture or hold the user's JWT in the backend for this.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

from .subscriber import RegistrySyncSubscriber

logger = logging.getLogger(__name__)


class _LibraryUpdateNotifier:
    """Adapts the subscriber's `.reconcile()` to a NON-mutating freshness probe:
    `check_for_updates()` (a cheap manifest-hash compare, anonymous, no records
    pulled), then `library_events.set_available()` to push the badge state to the
    frontend. Safe to call on every notification / reconnect."""

    def reconcile(self):
        from services.hf_sync import check_for_updates
        from services import library_events
        try:
            status = check_for_updates()
            if status.get("checked"):
                library_events.set_available(bool(status.get("available")))
            return status
        except Exception as e:
            logger.warning("[registry] update check failed: %s", e)
            return None


def build_subscriber() -> Optional[RegistrySyncSubscriber]:
    """Build the registry-sync subscriber from env, or None if not configured."""
    central = os.getenv("CENTRAL_SERVER_URL", "").rstrip("/")
    if not central:
        logger.info("[registry] CENTRAL_SERVER_URL unset — registry subscriber disabled")
        return None
    if os.getenv("ENABLE_REGISTRY_SUBSCRIBER", "1") == "0":
        logger.info("[registry] subscriber disabled (ENABLE_REGISTRY_SUBSCRIBER=0)")
        return None
    return RegistrySyncSubscriber(_LibraryUpdateNotifier(), central_url=central)
