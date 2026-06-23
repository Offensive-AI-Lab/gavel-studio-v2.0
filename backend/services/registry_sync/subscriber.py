"""Client-side observer: the WebSocket subscriber (transport + reconnect).

Wraps a reconcile action (see wiring._LibraryUpdateNotifier) with the live loop:

  * connect to the central server's public notification WebSocket,
  * RECONCILE on every (re)connect — the reconnect-reconciliation that catches any
    `version_update` missed while disconnected (push is the fast path, reconcile is
    the correctness backbone),
  * reconcile again on each {"event":"version_update"} message,
  * on disconnect/error, reconnect with exponential backoff + jitter (capped).

There is deliberately NO periodic poll here — clients must not hit HuggingFace on a
timer. Freshness comes from the real-time push plus the sync-on-reconnect; the
missed-webhook backstop poll lives on the CENTRAL server (one entity), not on every
user's backend.

`client.reconcile()` does blocking HF I/O, so it's dispatched to a thread. The WS
transport is injectable (`connect=`) so the loop is testable without a real socket;
the default uses the `websockets` library (lazy import — add it to requirements).
"""
from __future__ import annotations

import asyncio
import json
import logging
import random
from contextlib import asynccontextmanager
from typing import Callable, Optional

logger = logging.getLogger(__name__)


def derive_ws_url(central_url: str) -> str:
    """http(s)://host  ->  ws(s)://host/api/v1/ws"""
    u = central_url.rstrip("/")
    if u.startswith("https://"):
        return "wss://" + u[len("https://"):] + "/api/v1/ws"
    if u.startswith("http://"):
        return "ws://" + u[len("http://"):] + "/api/v1/ws"
    return u + "/api/v1/ws"


class RegistrySyncSubscriber:
    def __init__(self, client, *, central_url: str,
                 token_provider: Optional[Callable[[], Optional[str]]] = None,
                 backoff_base: float = 1.0, backoff_max: float = 60.0,
                 connect=None, sleep=None):
        self.client = client
        self.ws_url = derive_ws_url(central_url)
        self.token_provider = token_provider
        self.backoff_base = backoff_base
        self.backoff_max = backoff_max
        self._connect = connect or self._default_connect
        self._sleep = sleep or asyncio.sleep
        self._stop = asyncio.Event()
        self._tasks: list = []

    # ------------------------------------------------------------------ #
    # lifecycle
    # ------------------------------------------------------------------ #
    async def start(self) -> None:
        self._stop.clear()
        self._tasks = [asyncio.create_task(self._run_loop(), name="registry-ws")]

    async def stop(self) -> None:
        self._stop.set()
        for t in self._tasks:
            t.cancel()
        for t in self._tasks:
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass
        self._tasks = []

    async def wait(self) -> None:
        """Await the loop tasks (used by tests / a blocking host)."""
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)

    # ------------------------------------------------------------------ #
    # the WS loop + reconnect
    # ------------------------------------------------------------------ #
    async def _run_loop(self) -> None:
        attempt = 0
        while not self._stop.is_set():
            delay = 0.0
            try:
                # The central notification socket is public — connect with no token.
                # token_provider is only for a future PRIVATE registry; if set and it
                # returns a token, we pass it through.
                token = self.token_provider() if self.token_provider else None
                url = f"{self.ws_url}?token={token}" if token else self.ws_url
                async with self._connect(url) as ws:
                    attempt = 0                       # good connection → reset backoff
                    await self._reconcile()           # reconnect reconciliation
                    async for raw in ws:
                        if self._stop.is_set():
                            break
                        try:
                            data = json.loads(raw)
                        except Exception:
                            continue
                        if data.get("event") == "version_update":
                            await self._reconcile()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                if self._stop.is_set():
                    break
                delay = self._backoff(attempt)
                attempt += 1
                logger.warning("[registry] WS down (%s); reconnecting in %.1fs", e, delay)
            if await self._sleep_or_stop(delay):
                break

    async def _sleep_or_stop(self, delay: float) -> bool:
        """Sleep up to `delay` seconds; return True if stop() fired meanwhile."""
        if delay <= 0:
            return self._stop.is_set()
        try:
            await asyncio.wait_for(self._stop.wait(), timeout=delay)
            return True
        except asyncio.TimeoutError:
            return False

    async def _reconcile(self):
        # reconcile() is blocking (HF I/O + disk) → run off the event loop.
        try:
            return await asyncio.to_thread(self.client.reconcile)
        except Exception as e:
            logger.error("[registry] reconcile crashed: %s", e)

    def _backoff(self, attempt: int) -> float:
        base = min(self.backoff_max, self.backoff_base * (2 ** attempt))
        return base * (0.5 + random.random() * 0.5)    # jitter to avoid a thundering herd

    @asynccontextmanager
    async def _default_connect(self, url: str):
        import websockets  # lazy — add `websockets` to requirements to use the default
        async with websockets.connect(url) as ws:
            yield ws
