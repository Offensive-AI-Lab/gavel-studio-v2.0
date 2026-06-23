"""Tests for the WS subscriber: ws-url derivation, backoff bounds, and the
connect→reconcile / message→reconcile loop driven by a fake transport."""
import asyncio
import json
from contextlib import asynccontextmanager

from services.registry_sync.subscriber import RegistrySyncSubscriber, derive_ws_url


def test_derive_ws_url():
    assert derive_ws_url("http://localhost:8001") == "ws://localhost:8001/api/v1/ws"
    assert derive_ws_url("https://central.example/") == "wss://central.example/api/v1/ws"


def test_backoff_is_bounded_and_jittered():
    sub = RegistrySyncSubscriber(_FakeClient(), central_url="http://c",
                                 token_provider=lambda: "t",
                                 backoff_base=1.0, backoff_max=10.0)
    # attempt 0: within [0.5, 1.0]
    for _ in range(50):
        assert 0.5 <= sub._backoff(0) <= 1.0
    # large attempt: capped at backoff_max, jittered down to half of it
    for _ in range(50):
        assert 5.0 <= sub._backoff(20) <= 10.0


class _FakeClient:
    def __init__(self):
        self.reconciles = 0

    def reconcile(self):
        self.reconciles += 1


class _FakeWS:
    """Async-iterable that yields one version_update then ends."""
    def __init__(self, messages):
        self._messages = list(messages)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._messages:
            return self._messages.pop(0)
        raise StopAsyncIteration


def test_reconciles_on_connect_and_on_version_update():
    client = _FakeClient()
    # No token_provider → public socket, no credential in the URL.
    sub = RegistrySyncSubscriber(client, central_url="http://c")

    @asynccontextmanager
    async def fake_connect(url):
        assert url.endswith("/api/v1/ws")      # no ?token=
        yield _FakeWS([json.dumps({"event": "version_update"})])
        sub._stop.set()        # after the socket drains, end the loop

    sub._connect = fake_connect

    async def run():
        await sub.start()
        await asyncio.wait_for(sub.wait(), timeout=3)

    asyncio.run(run())
    # one reconcile on (re)connect + one for the version_update message
    assert client.reconciles >= 2


def test_reconnects_with_backoff_after_a_connection_error():
    client = _FakeClient()
    attempts = {"n": 0}
    sleeps = []

    sub = RegistrySyncSubscriber(client, central_url="http://c",
                                 token_provider=lambda: "tok")

    @asynccontextmanager
    async def flaky_connect(url):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise ConnectionError("refused")      # first connect fails → backoff
        yield _FakeWS([])                          # second connect ok, then end
        sub._stop.set()

    sub._connect = flaky_connect
    sub._backoff = lambda attempt: 0.01            # keep the test fast/deterministic

    async def run():
        await sub.start()
        await asyncio.wait_for(sub.wait(), timeout=3)

    asyncio.run(run())
    assert attempts["n"] >= 2                       # it retried after the failure
    assert client.reconciles >= 1                   # reconciled on the good connect
