"""WebSocket connection manager + the broadcaster.

Holds the set of authenticated client sockets and fans a tiny notification out to
all of them. The watcher runs on a background THREAD, so it can't `await` a send;
`broadcast_threadsafe` hops the message onto the server's event loop via
`run_coroutine_threadsafe`. Dead sockets are pruned on send failure.
"""
from __future__ import annotations

import asyncio
import logging
import threading
from typing import Optional, Set

logger = logging.getLogger(__name__)


class WSManager:
    def __init__(self, max_connections: int = 2000):
        self._conns: Set = set()
        self._lock = threading.Lock()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._max = max_connections

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Bind the server's running event loop (called once at startup) so the
        watcher thread can schedule broadcasts onto it."""
        self._loop = loop

    async def connect(self, ws) -> bool:
        """Accept + register a socket. Returns False (and closes the socket) when
        the connection cap is hit — the gate on this unauthenticated endpoint."""
        with self._lock:
            full = len(self._conns) >= self._max
        if full:
            await ws.close(code=1013)   # 1013 = "try again later"
            return False
        await ws.accept()
        with self._lock:
            self._conns.add(ws)
        return True

    def disconnect(self, ws) -> None:
        with self._lock:
            self._conns.discard(ws)

    def count(self) -> int:
        with self._lock:
            return len(self._conns)

    async def broadcast(self, message: dict) -> None:
        with self._lock:
            targets = list(self._conns)
        dead = []
        for ws in targets:
            try:
                await ws.send_json(message)
            except Exception:
                dead.append(ws)
        if dead:
            with self._lock:
                for ws in dead:
                    self._conns.discard(ws)

    def broadcast_threadsafe(self, message: dict) -> None:
        """Schedule a broadcast from a non-async thread (the watcher). No-op if the
        loop isn't bound yet (e.g. before startup / in tests)."""
        loop = self._loop
        if loop is None:
            logger.warning("WS broadcast skipped: event loop not bound")
            return
        try:
            asyncio.run_coroutine_threadsafe(self.broadcast(message), loop)
        except Exception as e:
            logger.error(f"WS broadcast schedule failed: {e}")
