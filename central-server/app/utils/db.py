"""Single PostgreSQL connection pool for the central server.

Reads DATABASE_URL from the environment (Render injects this automatically
when a Postgres service is linked). Falls back to discrete DB_* vars for
local dev so you can point at a docker-compose postgres.
"""
import os
import threading
import time
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from psycopg2 import extras, pool

env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

DATABASE_URL: Optional[str] = os.getenv("DATABASE_URL")

# This is a SHARED, multi-tenant server — every request runs on a threadpool
# thread, so the pool MUST be thread-safe. SimpleConnectionPool is NOT; under
# concurrency it can hand one connection to two threads or corrupt its free
# list. ThreadedConnectionPool is the thread-safe sibling (same API). Size it
# against the Postgres connection limit (DB_POOL_MAX, default 20).
_POOL_MAX = int(os.getenv("DB_POOL_MAX", "20"))

_pool: Optional[pool.ThreadedConnectionPool] = None
_pool_lock = threading.Lock()


def _get_pool() -> pool.ThreadedConnectionPool:
    global _pool
    if _pool is None:
        # Guard the lazy init too — without the lock, a cold-start burst could
        # have two threads each build a pool (leaking the first).
        with _pool_lock:
            if _pool is None:
                if not DATABASE_URL:
                    raise RuntimeError(
                        "DATABASE_URL is not set. Configure it in central-server/.env "
                        "or in the Render service environment."
                    )
                _pool = pool.ThreadedConnectionPool(1, _POOL_MAX, dsn=DATABASE_URL)
    return _pool


# When all pooled connections are checked out, ThreadedConnectionPool.getconn()
# raises immediately. Under a brief spike we'd rather queue for ~1s than 500 the
# request, so retry a few times with a short backoff before giving up.
_POOL_GET_RETRIES = int(os.getenv("DB_POOL_GET_RETRIES", "40"))
_POOL_GET_WAIT = float(os.getenv("DB_POOL_GET_WAIT", "0.025"))


def get_conn():
    p = _get_pool()
    last_err = None
    for _ in range(max(1, _POOL_GET_RETRIES)):
        try:
            return p.getconn()
        except pool.PoolError as e:
            last_err = e
            time.sleep(_POOL_GET_WAIT)
    raise last_err


def release_conn(conn):
    if conn:
        _get_pool().putconn(conn)


def execute(query: str, params=None):
    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute(query, params) if params else cur.execute(query)
        results = cur.fetchall() if cur.description else None
        conn.commit()
        cur.close()
        return results
    except Exception:
        conn.rollback()
        raise
    finally:
        release_conn(conn)


def execute_dict(query: str, params=None):
    conn = get_conn()
    try:
        cur = conn.cursor(cursor_factory=extras.RealDictCursor)
        cur.execute(query, params) if params else cur.execute(query)
        results = cur.fetchall() if cur.description else None
        conn.commit()
        cur.close()
        return results
    except Exception:
        conn.rollback()
        raise
    finally:
        release_conn(conn)


def close_pool():
    global _pool
    if _pool is not None:
        _pool.closeall()
        _pool = None
