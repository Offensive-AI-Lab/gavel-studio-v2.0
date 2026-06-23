"""Pytest config for the central-server suite.

Two responsibilities, both handled at import time so they take effect before
any test module pulls in `app.*`:

  1. Put the central-server root on sys.path so `from app.utils import auth`
     resolves when pytest is run from this directory.

  2. Set a deterministic JWT_SECRET_KEY in the environment. `app.utils.auth`
     raises at import if the key is missing (it is the sole auth authority and
     must never fall back to a default), so we provide one here. We set it on
     os.environ *before* the app is imported; auth.py's load_dotenv() uses
     override=False, so it will not clobber what we set.

These are pure-logic tests. None of them touch a database: the DB pool is built
lazily on first query, and the endpoints exercised here (/health, /, the
middleware chain) never issue one.
"""
import os
import sys
from pathlib import Path

# 1. import path -----------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# 2. deterministic secrets / config, set BEFORE app import -----------------
os.environ.setdefault("JWT_SECRET_KEY", "test-secret-key-do-not-use-in-production")
# Default to exactly one trusted proxy hop (the production default) so the
# X-Forwarded-For tests assert against the real behaviour. Individual tests
# monkeypatch rate_limit._TRUSTED_HOPS when they need a different value.
os.environ.setdefault("TRUSTED_PROXY_HOPS", "1")
# Keep the decode cache on by default — it is on in production and several
# tests assert its behaviour explicitly.
os.environ.setdefault("CACHE_DECODE_TOKENS", "1")
# Keep the control-plane background watcher OFF during tests — no thread, no DB,
# no HF calls on app startup. Tests drive the watcher / routes directly instead.
os.environ.setdefault("ENABLE_CONTROL_PLANE", "0")
