"""Unit tests for the WSManager broadcaster (asyncio.run drives the coroutines —
no pytest-asyncio needed)."""
import asyncio

from app.services.ws_manager import WSManager


class FakeWS:
    def __init__(self, *, fail=False):
        self.sent = []
        self.fail = fail
        self.closed = None

    async def accept(self):
        pass

    async def close(self, code=None):
        self.closed = code

    async def send_json(self, msg):
        if self.fail:
            raise RuntimeError("socket closed")
        self.sent.append(msg)


def test_broadcast_delivers_to_all_and_prunes_dead():
    m = WSManager()
    good, dead = FakeWS(), FakeWS(fail=True)

    async def scenario():
        await m.connect(good)
        await m.connect(dead)
        assert m.count() == 2
        await m.broadcast({"event": "version_update"})

    asyncio.run(scenario())
    assert good.sent == [{"event": "version_update"}]
    assert m.count() == 1            # the failing socket was pruned


def test_disconnect_removes_socket():
    m = WSManager()
    ws = FakeWS()

    async def scenario():
        await m.connect(ws)
        assert m.count() == 1
        m.disconnect(ws)
        assert m.count() == 0

    asyncio.run(scenario())


def test_broadcast_threadsafe_is_noop_without_a_bound_loop():
    # No event loop bound (e.g. before startup): must not raise.
    WSManager().broadcast_threadsafe({"event": "version_update"})


def test_connect_rejects_over_capacity():
    m = WSManager(max_connections=1)
    a, b = FakeWS(), FakeWS()

    async def scenario():
        assert await m.connect(a) is True
        assert await m.connect(b) is False    # over cap → rejected
        return m.count()

    assert asyncio.run(scenario()) == 1
    assert b.closed == 1013                   # closed with "try again later"
