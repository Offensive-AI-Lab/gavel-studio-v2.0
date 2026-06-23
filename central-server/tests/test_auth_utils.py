"""Unit tests for app.utils.auth — the central server's JWT signing/verifying
and password hashing.

The central server is the SOLE auth authority: it signs every login token and
is the only component that verifies them. A bug here would let a forged or
expired token through, or lock every user out, so the round-trip, the tamper
rejection, the expiry check, and the decode-cache safety properties all deserve
explicit coverage.

No database and no network are involved — these are pure crypto/logic units.
"""
import time
from datetime import timedelta

import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials
from jose import jwt

from app.utils import auth


# ---------------------------------------------------------------------------
# Password hashing
# ---------------------------------------------------------------------------


class TestPasswordHashing:
    def test_hash_is_not_the_plaintext(self):
        h = auth.hash_password("hunter2")
        assert h != "hunter2"
        # argon2 hashes carry an identifiable prefix.
        assert h.startswith("$argon2")

    def test_verify_accepts_correct_password(self):
        h = auth.hash_password("correct horse battery staple")
        assert auth.verify_password("correct horse battery staple", h) is True

    def test_verify_rejects_wrong_password(self):
        h = auth.hash_password("the-right-one")
        assert auth.verify_password("the-wrong-one", h) is False

    def test_same_password_hashes_differently(self):
        # A per-hash random salt means two hashes of the same password must
        # differ — otherwise identical passwords would be visibly equal in the
        # DB. Both must still verify.
        a = auth.hash_password("repeat")
        b = auth.hash_password("repeat")
        assert a != b
        assert auth.verify_password("repeat", a)
        assert auth.verify_password("repeat", b)


# ---------------------------------------------------------------------------
# JWT create / decode round-trip
# ---------------------------------------------------------------------------


class TestTokenRoundTrip:
    def test_decode_returns_the_signed_claims(self):
        token = auth.create_access_token({"sub": "42"})
        payload = auth.decode_access_token(token)
        assert payload["sub"] == "42"
        # create_access_token always stamps an expiry.
        assert "exp" in payload

    def test_default_expiry_is_in_the_future(self):
        token = auth.create_access_token({"sub": "1"})
        payload = auth.decode_access_token(token)
        assert payload["exp"] > time.time()

    def test_extra_claims_are_preserved(self):
        token = auth.create_access_token({"sub": "7", "role": "admin"})
        payload = auth.decode_access_token(token)
        assert payload["role"] == "admin"


# ---------------------------------------------------------------------------
# JWT rejection paths — the security-critical ones
# ---------------------------------------------------------------------------


class TestTokenRejection:
    def test_tampered_token_is_rejected(self):
        token = auth.create_access_token({"sub": "5"})
        # Flip the last character of the signature segment.
        tampered = token[:-1] + ("a" if token[-1] != "a" else "b")
        with pytest.raises(HTTPException) as ei:
            auth.decode_access_token(tampered)
        assert ei.value.status_code == 401

    def test_token_signed_with_a_different_secret_is_rejected(self):
        # An attacker who guesses the algorithm but not the key must not get in.
        forged = jwt.encode({"sub": "5"}, "some-other-secret", algorithm=auth.ALGORITHM)
        with pytest.raises(HTTPException) as ei:
            auth.decode_access_token(forged)
        assert ei.value.status_code == 401

    def test_expired_token_is_rejected(self):
        # A token whose exp is already in the past must fail on decode.
        token = auth.create_access_token({"sub": "5"}, expires_delta=timedelta(seconds=-10))
        with pytest.raises(HTTPException) as ei:
            auth.decode_access_token(token)
        assert ei.value.status_code == 401

    def test_garbage_string_is_rejected(self):
        with pytest.raises(HTTPException) as ei:
            auth.decode_access_token("not-a-jwt-at-all")
        assert ei.value.status_code == 401


# ---------------------------------------------------------------------------
# get_current_user — the FastAPI dependency
# ---------------------------------------------------------------------------


def _creds(token: str) -> HTTPAuthorizationCredentials:
    return HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)


class TestGetCurrentUser:
    def test_returns_subject_as_int(self):
        token = auth.create_access_token({"sub": "123"})
        assert auth.get_current_user(_creds(token)) == 123
        assert isinstance(auth.get_current_user(_creds(token)), int)

    def test_missing_subject_raises_401(self):
        # A validly-signed token that simply has no `sub` claim is unusable.
        token = auth.create_access_token({"not_sub": "x"})
        with pytest.raises(HTTPException) as ei:
            auth.get_current_user(_creds(token))
        assert ei.value.status_code == 401

    def test_invalid_token_propagates_401(self):
        with pytest.raises(HTTPException) as ei:
            auth.get_current_user(_creds("garbage"))
        assert ei.value.status_code == 401


# ---------------------------------------------------------------------------
# Decode cache — the fast path must never weaken the security guarantees
# ---------------------------------------------------------------------------


class TestDecodeCache:
    @pytest.fixture(autouse=True)
    def _clear_cache(self):
        """Each test starts with an empty cache so ordering doesn't matter."""
        with auth._decode_cache_lock:
            auth._decode_cache.clear()
        yield
        with auth._decode_cache_lock:
            auth._decode_cache.clear()

    def test_first_decode_populates_the_cache(self):
        token = auth.create_access_token({"sub": "9"})
        assert len(auth._decode_cache) == 0
        auth.decode_access_token(token)
        assert len(auth._decode_cache) == 1

    def test_second_decode_returns_equal_payload(self):
        token = auth.create_access_token({"sub": "9"})
        first = auth.decode_access_token(token)
        second = auth.decode_access_token(token)
        assert first == second

    def test_cache_key_mixes_in_secret_fingerprint(self):
        # The key must depend on the signing secret so a secret rotation makes
        # every prior entry a miss (no token signed with the old key survives).
        token = auth.create_access_token({"sub": "9"})
        k1 = auth._cache_key(token)
        saved = auth._SECRET_FINGERPRINT
        try:
            auth._SECRET_FINGERPRINT = b"different"
            k2 = auth._cache_key(token)
        finally:
            auth._SECRET_FINGERPRINT = saved
        assert k1 != k2

    def test_cache_key_is_not_the_raw_token(self):
        # We store a hash, not the JWT itself, so a memory dump can't leak it.
        token = auth.create_access_token({"sub": "9"})
        assert token.encode("utf-8") not in auth._cache_key(token)

    def test_exp_still_valid_helper(self):
        assert auth._exp_still_valid({"exp": time.time() + 100}) is True
        assert auth._exp_still_valid({"exp": time.time() - 100}) is False
        assert auth._exp_still_valid({}) is False
        assert auth._exp_still_valid({"exp": "not-a-number"}) is False

    def test_cached_entry_with_expired_exp_is_never_served(self):
        # Plant a cache entry whose stored payload has already expired, keyed to
        # a token that would never decode for real. If the hit path trusted the
        # cache it would return the planted (privileged) payload; instead it must
        # reject the stale exp, evict, and fall through to a real decode — which
        # raises on the bogus token. This is the property that stops an expired
        # token being honoured just because it was once cached.
        bogus = "bogus.token.value"
        key = auth._cache_key(bogus)
        with auth._decode_cache_lock:
            auth._decode_cache[key] = ({"sub": "hacker", "exp": time.time() - 100}, time.time())
        with pytest.raises(HTTPException):
            auth.decode_access_token(bogus)
        # The stale entry was evicted on the way out.
        assert key not in auth._decode_cache

    def test_cached_entry_older_than_ttl_is_never_served(self):
        # Even with a far-future exp, an entry cached longer ago than the 60s TTL
        # must not be served — this bounds the window for any future revocation.
        bogus = "another.bogus.token"
        key = auth._cache_key(bogus)
        with auth._decode_cache_lock:
            stale_age = time.time() - (auth._DECODE_CACHE_TTL + 10)
            auth._decode_cache[key] = ({"sub": "hacker", "exp": time.time() + 9999}, stale_age)
        with pytest.raises(HTTPException):
            auth.decode_access_token(bogus)
        assert key not in auth._decode_cache

    def test_works_with_cache_disabled(self, monkeypatch):
        monkeypatch.setattr(auth, "_DECODE_CACHE_ENABLED", False)
        token = auth.create_access_token({"sub": "9"})
        assert auth.decode_access_token(token)["sub"] == "9"
        # Nothing was cached.
        assert len(auth._decode_cache) == 0

    def test_cache_is_bounded(self, monkeypatch):
        # Spraying many distinct tokens must not grow the cache without bound.
        monkeypatch.setattr(auth, "_DECODE_CACHE_MAXSIZE", 8)
        for i in range(50):
            auth.decode_access_token(auth.create_access_token({"sub": str(i)}))
        assert len(auth._decode_cache) <= 8
