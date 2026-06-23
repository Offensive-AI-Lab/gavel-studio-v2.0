"""SourceWatcher — the Subject of the control plane.

Holds the active { commit, global_signature, namespaces } version map in memory,
backed by a persistent DB row so it survives a server reboot. It converges to HF's
*authoritative* HEAD and advances ONLY when the manifest's `global_signature`
actually changes (dedup — so the server's own publish, which fires a webhook back
at us, doesn't re-broadcast).

Robustness, per the control-plane spec:
  * one background worker thread → reconciles are serialized; a `Lock(timeout)`
    additionally guards against a request-thread reconcile racing the worker.
  * a STRICT timeout on the HF fetch (runs it on an executor and `.result(timeout)`),
    so a hung/slow HF can't wedge the watcher. On any HF failure: log, abort, leave
    state untouched, do NOT broadcast stale data.
  * a DEBOUNCE window coalesces a burst of triggers into one HEAD check.
  * a low-frequency SAFETY POLL is the backstop for missed webhooks (it's just the
    worker's wait() timing out → one reconcile).
"""
from __future__ import annotations

import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, Optional

logger = logging.getLogger(__name__)

EMPTY_STATE: Dict = {"commit": "", "global_signature": "", "namespaces": {}}


class SourceWatcher:
    def __init__(self, provider, *, repo: str,
                 broadcast: Optional[Callable[[dict], None]] = None,
                 load_state: Optional[Callable[[], Optional[dict]]] = None,
                 save_state: Optional[Callable[[dict], None]] = None,
                 debounce_s: float = 1.0, safety_poll_s: float = 300.0,
                 hf_timeout_s: float = 5.0, lock_timeout_s: float = 5.0):
        self.provider = provider
        self.repo = repo
        self._broadcast = broadcast or (lambda state: None)
        self._load_state = load_state or (lambda: _db_load_state(repo))
        self._save_state = save_state or (lambda s: _db_save_state(repo, s))
        self.debounce_s = debounce_s
        self.safety_poll_s = safety_poll_s
        self.hf_timeout_s = hf_timeout_s
        self.lock_timeout_s = lock_timeout_s

        self._state: Dict = dict(EMPTY_STATE)
        self._lock = threading.Lock()            # per-repo advisory lock
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="hf-fetch")
        self._trigger = threading.Event()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------ #
    # lifecycle
    # ------------------------------------------------------------------ #
    @property
    def current_versions(self) -> dict:
        return dict(self._state)

    def start(self) -> None:
        try:
            loaded = self._load_state()
            if loaded:
                self._state = loaded
        except Exception as e:
            logger.error(f"[watcher] load_state failed (starting empty): {e}")
        self._stop.clear()
        self._thread = threading.Thread(target=self._run_loop, name="source-watcher",
                                        daemon=True)
        self._thread.start()
        self.trigger()  # converge to HEAD on boot

    def stop(self) -> None:
        self._stop.set()
        self._trigger.set()
        if self._thread:
            self._thread.join(timeout=5)
        self._executor.shutdown(wait=False)

    def trigger(self) -> None:
        """The doorbell — called by the webhook route. Cheap and non-blocking."""
        self._trigger.set()

    # ------------------------------------------------------------------ #
    # the worker loop: trigger (debounced) OR safety-poll → reconcile
    # ------------------------------------------------------------------ #
    def _run_loop(self) -> None:
        while not self._stop.is_set():
            fired = self._trigger.wait(timeout=self.safety_poll_s)
            if self._stop.is_set():
                break
            if fired:
                self._trigger.clear()
                time.sleep(self.debounce_s)   # coalesce a burst into one reconcile
                self._trigger.clear()         # absorb triggers that landed mid-debounce
            try:
                self.reconcile_now()
            except Exception as e:            # never let the loop die
                logger.error(f"[watcher] reconcile crashed: {e}")

    # ------------------------------------------------------------------ #
    # reconcile: fetch HEAD+manifest (strict timeout) → dedup → persist → broadcast
    # ------------------------------------------------------------------ #
    def reconcile_now(self) -> bool:
        """Returns True iff the version map advanced (and a broadcast was fired)."""
        if not self._lock.acquire(timeout=self.lock_timeout_s):
            logger.warning("[watcher] reconcile skipped: another reconcile in progress")
            return False
        try:
            return self._reconcile_locked()
        finally:
            self._lock.release()

    def _reconcile_locked(self) -> bool:
        try:
            head = self._with_timeout(self.provider.head_version)
            manifest = self._with_timeout(lambda: self.provider.fetch_manifest(head))
        except Exception as e:
            # HF down / slow / timed out → graceful: state untouched, no broadcast.
            logger.warning(f"[watcher] reconcile aborted (HF unreachable/timeout): {e}")
            return False

        new_global = (manifest or {}).get("global_signature") or ""
        if not new_global:
            logger.warning("[watcher] manifest has no global_signature; skipping")
            return False
        if new_global == self._state.get("global_signature"):
            return False  # dedup: self-push or unchanged HEAD → no broadcast

        new_state = {
            "commit": head,
            "global_signature": new_global,
            "namespaces": manifest.get("namespaces") or {},
        }
        try:
            self._save_state(new_state)
        except Exception as e:
            # Persist failed → do NOT advance memory or broadcast, so we don't
            # promise clients a version we couldn't durably record.
            logger.error(f"[watcher] persist failed, not broadcasting: {e}")
            return False

        self._state = new_state
        try:
            self._broadcast(new_state)
        except Exception as e:
            logger.error(f"[watcher] broadcast failed (state already advanced): {e}")
        return True

    def _with_timeout(self, fn: Callable):
        """Run a blocking HF call with a STRICT wall-clock timeout."""
        return self._executor.submit(fn).result(timeout=self.hf_timeout_s)


# --------------------------------------------------------------------------- #
# Default DB-backed persistence (central-server Postgres). Survives reboots.
# --------------------------------------------------------------------------- #
_table_ready = False


def _ensure_table() -> None:
    global _table_ready
    if _table_ready:
        return
    from app.utils.db import execute
    execute("""
        CREATE TABLE IF NOT EXISTS registry_version (
            repo             TEXT PRIMARY KEY,
            commit_sha       TEXT,
            global_signature TEXT,
            namespaces       JSONB,
            updated_at       TIMESTAMPTZ DEFAULT now()
        )
    """)
    _table_ready = True


def _db_load_state(repo: str) -> Optional[dict]:
    from app.utils.db import execute_dict
    _ensure_table()
    rows = execute_dict(
        "SELECT commit_sha, global_signature, namespaces FROM registry_version WHERE repo = %s",
        (repo,))
    if not rows:
        return None
    r = rows[0]
    ns = r["namespaces"]
    if isinstance(ns, str):
        ns = json.loads(ns)
    return {
        "commit": r["commit_sha"] or "",
        "global_signature": r["global_signature"] or "",
        "namespaces": ns or {},
    }


def _db_save_state(repo: str, state: dict) -> None:
    from app.utils.db import execute
    _ensure_table()
    execute("""
        INSERT INTO registry_version (repo, commit_sha, global_signature, namespaces, updated_at)
        VALUES (%s, %s, %s, %s, now())
        ON CONFLICT (repo) DO UPDATE SET
            commit_sha       = EXCLUDED.commit_sha,
            global_signature = EXCLUDED.global_signature,
            namespaces       = EXCLUDED.namespaces,
            updated_at       = now()
    """, (repo, state.get("commit") or "", state.get("global_signature") or "",
          json.dumps(state.get("namespaces") or {})))
