"""Unit tests for services.central_server — the HTTP client wrapping the
GAVEL central identity/bookmark/ratings server.

These are pure-logic tests: no real network, no live central server. The
module under test uses a single module-level httpx.Client (`_client`) and a
module-level `CENTRAL_SERVER_URL`. We monkeypatch both:

  * `central_server._client` -> a tiny fake client whose `.request(...)`
    returns a canned `_FakeResponse` (or raises an httpx.RequestError to
    model timeouts / connection failures).
  * `central_server.CENTRAL_SERVER_URL` -> a known base URL so request
    shaping (method, path, headers, json body, params) can be asserted on.

The fake client records every call so we can assert on request shaping +
auth header. We never touch a real socket.
"""
import httpx
import pytest

from services import central_server as cs


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _FakeResponse:
    """Mimics the slice of httpx.Response that _request() touches:
    status_code, headers (dict-like with .get), .json(), .text."""

    def __init__(self, status_code=200, json_body=None, text="",
                 content_type="application/json", raise_on_json=False):
        self.status_code = status_code
        self._json_body = json_body
        self.text = text
        self.headers = {"content-type": content_type}
        self._raise_on_json = raise_on_json

    def json(self):
        if self._raise_on_json:
            raise ValueError("malformed JSON")
        return self._json_body


class _RecordingClient:
    """Stands in for the module-level httpx.Client. Records each request and
    returns a queued response (or raises a queued exception)."""

    def __init__(self, response=None, exc=None):
        self.response = response
        self.exc = exc
        self.calls = []

    def request(self, method, url, headers=None, json=None, params=None):
        self.calls.append({
            "method": method,
            "url": url,
            "headers": headers,
            "json": json,
            "params": params,
        })
        if self.exc is not None:
            raise self.exc
        return self.response

    @property
    def last(self):
        return self.calls[-1]


@pytest.fixture
def base_url(monkeypatch):
    """Configure a known central server base URL on the module."""
    url = "https://central.example.test"
    monkeypatch.setattr(cs, "CENTRAL_SERVER_URL", url)
    return url


def _install_client(monkeypatch, response=None, exc=None):
    client = _RecordingClient(response=response, exc=exc)
    monkeypatch.setattr(cs, "_client", client)
    return client


# ---------------------------------------------------------------------------
# _headers — auth header shaping
# ---------------------------------------------------------------------------


class TestHeaders:
    def test_includes_bearer_token_when_present(self):
        h = cs._headers("abc123")
        assert h["Authorization"] == "Bearer abc123"
        assert h["Content-Type"] == "application/json"

    def test_omits_auth_header_when_token_none(self):
        h = cs._headers(None)
        assert "Authorization" not in h
        assert h["Content-Type"] == "application/json"

    def test_empty_token_treated_as_no_auth(self):
        # "" is falsy, so no Authorization header. This is the documented
        # "no auth required" path (e.g. get_team_users).
        h = cs._headers("")
        assert "Authorization" not in h


# ---------------------------------------------------------------------------
# is_enabled
# ---------------------------------------------------------------------------


class TestIsEnabled:
    def test_enabled_when_url_set(self, monkeypatch):
        monkeypatch.setattr(cs, "CENTRAL_SERVER_URL", "https://x")
        assert cs.is_enabled() is True

    def test_disabled_when_url_empty(self, monkeypatch):
        monkeypatch.setattr(cs, "CENTRAL_SERVER_URL", "")
        assert cs.is_enabled() is False


# ---------------------------------------------------------------------------
# _request — core request shaping + response handling
# ---------------------------------------------------------------------------


class TestRequestCore:
    def test_unconfigured_url_raises_503(self, monkeypatch):
        monkeypatch.setattr(cs, "CENTRAL_SERVER_URL", "")
        # The client should never be hit if URL is unset.
        client = _install_client(monkeypatch, response=_FakeResponse())
        with pytest.raises(cs.CentralServerError) as ei:
            cs._request("GET", "/anything")
        assert ei.value.status_code == 503
        assert client.calls == []

    def test_builds_full_url_and_passes_method(self, base_url, monkeypatch):
        client = _install_client(
            monkeypatch, response=_FakeResponse(json_body={"ok": True}))
        out = cs._request("GET", "/auth/me", token="tok")
        assert out == {"ok": True}
        assert client.last["method"] == "GET"
        assert client.last["url"] == base_url + "/auth/me"

    def test_auth_header_threaded_through(self, base_url, monkeypatch):
        client = _install_client(
            monkeypatch, response=_FakeResponse(json_body={}))
        cs._request("GET", "/x", token="secret-token")
        assert client.last["headers"]["Authorization"] == "Bearer secret-token"

    def test_no_auth_header_when_token_absent(self, base_url, monkeypatch):
        client = _install_client(
            monkeypatch, response=_FakeResponse(json_body={}))
        cs._request("GET", "/auth/team-users")
        assert "Authorization" not in client.last["headers"]

    def test_json_body_and_params_forwarded(self, base_url, monkeypatch):
        client = _install_client(
            monkeypatch, response=_FakeResponse(json_body={}))
        cs._request("POST", "/x", json={"a": 1}, params={"p": "q"})
        assert client.last["json"] == {"a": 1}
        assert client.last["params"] == {"p": "q"}

    def test_parses_json_when_content_type_json(self, base_url, monkeypatch):
        _install_client(
            monkeypatch,
            response=_FakeResponse(json_body={"hello": "world"}))
        assert cs._request("GET", "/x") == {"hello": "world"}

    def test_returns_text_for_non_json_content_type(self, base_url, monkeypatch):
        _install_client(
            monkeypatch,
            response=_FakeResponse(
                json_body=None, text="plain body",
                content_type="text/plain"))
        assert cs._request("GET", "/x") == "plain body"

    def test_missing_content_type_returns_text(self, base_url, monkeypatch):
        # headers.get("content-type", "") -> "" -> not application/json -> text
        resp = _FakeResponse(text="raw", content_type="")
        # Force an empty content-type header.
        resp.headers = {}
        _install_client(monkeypatch, response=resp)
        assert cs._request("GET", "/x") == "raw"


# ---------------------------------------------------------------------------
# _request — error branches
# ---------------------------------------------------------------------------


class TestRequestErrors:
    def test_non_200_raises_with_detail_from_payload(self, base_url, monkeypatch):
        _install_client(
            monkeypatch,
            response=_FakeResponse(
                status_code=404, json_body={"detail": "not found"}))
        with pytest.raises(cs.CentralServerError) as ei:
            cs._request("GET", "/x")
        assert ei.value.status_code == 404
        assert "not found" in str(ei.value)
        assert ei.value.payload == {"detail": "not found"}

    def test_error_payload_without_detail_falls_back_to_http_code(self, base_url, monkeypatch):
        # payload is a dict but has no "detail" key -> detail is None ->
        # falls back to "HTTP {status}".
        _install_client(
            monkeypatch,
            response=_FakeResponse(status_code=500, json_body={"other": "x"}))
        with pytest.raises(cs.CentralServerError) as ei:
            cs._request("GET", "/x")
        assert "HTTP 500" in str(ei.value)
        assert ei.value.status_code == 500

    def test_error_with_malformed_json_uses_text(self, base_url, monkeypatch):
        # resp.json() raises -> payload = {"detail": resp.text}
        _install_client(
            monkeypatch,
            response=_FakeResponse(
                status_code=400, raise_on_json=True,
                text="gateway exploded"))
        with pytest.raises(cs.CentralServerError) as ei:
            cs._request("GET", "/x")
        assert "gateway exploded" in str(ei.value)
        assert ei.value.status_code == 400

    def test_error_with_non_dict_json_payload(self, base_url, monkeypatch):
        # JSON parses but is a list, not a dict -> detail = str(payload).
        _install_client(
            monkeypatch,
            response=_FakeResponse(status_code=422, json_body=["bad", "input"]))
        with pytest.raises(cs.CentralServerError) as ei:
            cs._request("GET", "/x")
        assert ei.value.status_code == 422
        # detail derived via str(payload) since payload isn't a dict
        assert "bad" in str(ei.value)

    def test_connection_error_raises_502(self, base_url, monkeypatch):
        _install_client(
            monkeypatch,
            exc=httpx.ConnectError("connection refused"))
        with pytest.raises(cs.CentralServerError) as ei:
            cs._request("GET", "/x")
        assert ei.value.status_code == 502
        assert "unreachable" in str(ei.value)

    def test_timeout_raises_502(self, base_url, monkeypatch):
        # httpx.TimeoutException is a subclass of httpx.RequestError.
        assert issubclass(httpx.TimeoutException, httpx.RequestError)
        _install_client(
            monkeypatch,
            exc=httpx.ConnectTimeout("timed out"))
        with pytest.raises(cs.CentralServerError) as ei:
            cs._request("GET", "/x")
        assert ei.value.status_code == 502

    def test_400_is_error_boundary(self, base_url, monkeypatch):
        # status_code >= 400 is the error boundary; 400 raises.
        _install_client(
            monkeypatch,
            response=_FakeResponse(status_code=400, json_body={"detail": "bad"}))
        with pytest.raises(cs.CentralServerError):
            cs._request("GET", "/x")

    def test_399_is_not_error(self, base_url, monkeypatch):
        # Just below the boundary: treated as success. 399 isn't a real HTTP
        # code but it exercises the < 400 branch precisely.
        _install_client(
            monkeypatch,
            response=_FakeResponse(status_code=399, json_body={"ok": 1}))
        assert cs._request("GET", "/x") == {"ok": 1}


# ---------------------------------------------------------------------------
# CentralServerError construction
# ---------------------------------------------------------------------------


class TestCentralServerError:
    def test_defaults(self):
        e = cs.CentralServerError("boom")
        assert e.status_code == 500
        assert e.payload == {}
        assert str(e) == "boom"

    def test_payload_none_becomes_empty_dict(self):
        e = cs.CentralServerError("boom", status_code=404, payload=None)
        assert e.payload == {}

    def test_custom_payload_retained(self):
        e = cs.CentralServerError("boom", status_code=409, payload={"k": "v"})
        assert e.status_code == 409
        assert e.payload == {"k": "v"}


# ---------------------------------------------------------------------------
# Bookmarks — request shaping + list parsing
# ---------------------------------------------------------------------------


class TestBookmarks:
    def test_add_bookmark_shape(self, base_url, monkeypatch):
        client = _install_client(
            monkeypatch, response=_FakeResponse(json_body={"created": True}))
        out = cs.add_bookmark("tok", "rule", "pub-123")
        assert out == {"created": True}
        assert client.last["method"] == "POST"
        assert client.last["url"] == base_url + "/bookmarks"
        assert client.last["json"] == {"asset_type": "rule", "public_id": "pub-123"}
        assert client.last["headers"]["Authorization"] == "Bearer tok"

    def test_remove_bookmark_shape(self, base_url, monkeypatch):
        client = _install_client(
            monkeypatch, response=_FakeResponse(json_body={"removed": True}))
        out = cs.remove_bookmark("tok", "ce", "pub-9")
        assert out == {"removed": True}
        assert client.last["method"] == "DELETE"
        assert client.last["url"] == base_url + "/bookmarks/ce/pub-9"
        # DELETE carries no JSON body.
        assert client.last["json"] is None

    def test_list_bookmarks_extracts_list(self, base_url, monkeypatch):
        rows = [{"public_id": "a"}, {"public_id": "b"}]
        client = _install_client(
            monkeypatch, response=_FakeResponse(json_body={"bookmarks": rows}))
        out = cs.list_bookmarks("tok", "rule")
        assert out == rows
        assert client.last["url"] == base_url + "/bookmarks/rule"

    def test_list_bookmarks_missing_key_returns_empty(self, base_url, monkeypatch):
        _install_client(
            monkeypatch, response=_FakeResponse(json_body={"other": 1}))
        assert cs.list_bookmarks("tok", "rule") == []

    def test_list_bookmarks_non_dict_response_returns_empty(self, base_url, monkeypatch):
        # If the server returns a bare list (or anything non-dict), the helper
        # must defensively return [] rather than blow up on .get().
        _install_client(
            monkeypatch,
            response=_FakeResponse(
                json_body=None, text="surprise", content_type="text/plain"))
        assert cs.list_bookmarks("tok", "rule") == []


# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------


class TestAuthHelpers:
    def test_register_shape(self, base_url, monkeypatch):
        client = _install_client(
            monkeypatch, response=_FakeResponse(json_body={"id": 1}))
        cs.register("alice", "a@x.io", "pw")
        assert client.last["method"] == "POST"
        assert client.last["url"] == base_url + "/auth/register"
        assert client.last["json"] == {
            "username": "alice", "email": "a@x.io", "password": "pw"}
        # register is unauthenticated.
        assert "Authorization" not in client.last["headers"]

    def test_login_shape(self, base_url, monkeypatch):
        client = _install_client(
            monkeypatch, response=_FakeResponse(json_body={"token": "t"}))
        cs.login("a@x.io", "pw")
        assert client.last["json"] == {"email": "a@x.io", "password": "pw"}
        assert "Authorization" not in client.last["headers"]

    def test_get_me_uses_token(self, base_url, monkeypatch):
        client = _install_client(
            monkeypatch, response=_FakeResponse(json_body={"me": True}))
        cs.get_me("mytoken")
        assert client.last["url"] == base_url + "/auth/me"
        assert client.last["headers"]["Authorization"] == "Bearer mytoken"

    def test_update_me_only_includes_provided_fields(self, base_url, monkeypatch):
        client = _install_client(
            monkeypatch, response=_FakeResponse(json_body={}))
        cs.update_me("tok", display_name="New Name")
        assert client.last["method"] == "PATCH"
        assert client.last["json"] == {"display_name": "New Name"}
        # bio omitted entirely since it was None.
        assert "bio" not in client.last["json"]

    def test_update_me_empty_when_nothing_provided(self, base_url, monkeypatch):
        client = _install_client(
            monkeypatch, response=_FakeResponse(json_body={}))
        cs.update_me("tok")
        assert client.last["json"] == {}

    def test_update_me_includes_empty_string_bio(self, base_url, monkeypatch):
        # "" is not None, so an explicit empty bio (clearing it) IS sent.
        client = _install_client(
            monkeypatch, response=_FakeResponse(json_body={}))
        cs.update_me("tok", bio="")
        assert client.last["json"] == {"bio": ""}

    def test_get_users_by_username_empty_short_circuits(self, base_url, monkeypatch):
        # Empty list returns [] WITHOUT hitting the network at all.
        client = _install_client(
            monkeypatch, response=_FakeResponse(json_body={"users": [1]}))
        assert cs.get_users_by_username([]) == []
        assert client.calls == []

    def test_get_users_by_username_joins_params(self, base_url, monkeypatch):
        client = _install_client(
            monkeypatch,
            response=_FakeResponse(json_body={"users": [{"id": 1}]}))
        out = cs.get_users_by_username(["alice", "bob"])
        assert out == [{"id": 1}]
        assert client.last["params"] == {"usernames": "alice,bob"}
        # No auth required.
        assert "Authorization" not in client.last["headers"]

    def test_get_users_by_username_non_dict_returns_empty(self, base_url, monkeypatch):
        _install_client(
            monkeypatch,
            response=_FakeResponse(
                json_body=None, text="x", content_type="text/plain"))
        assert cs.get_users_by_username(["alice"]) == []

    def test_get_team_users_extracts_users(self, base_url, monkeypatch):
        client = _install_client(
            monkeypatch,
            response=_FakeResponse(json_body={"users": [{"is_team": True}]}))
        out = cs.get_team_users()
        assert out == [{"is_team": True}]
        assert client.last["url"] == base_url + "/auth/team-users"
        assert "Authorization" not in client.last["headers"]

    def test_get_team_users_missing_key_returns_empty(self, base_url, monkeypatch):
        _install_client(
            monkeypatch, response=_FakeResponse(json_body={}))
        assert cs.get_team_users() == []

    def test_get_team_users_non_dict_returns_empty(self, base_url, monkeypatch):
        _install_client(
            monkeypatch,
            response=_FakeResponse(
                json_body=None, text="x", content_type="text/plain"))
        assert cs.get_team_users() == []

    def test_record_publish_attribution_omits_published_at(self, base_url, monkeypatch):
        client = _install_client(
            monkeypatch, response=_FakeResponse(json_body={}))
        cs.record_publish_attribution("tok", "rule")
        assert client.last["json"] == {"asset_type": "rule"}

    def test_record_publish_attribution_includes_published_at(self, base_url, monkeypatch):
        client = _install_client(
            monkeypatch, response=_FakeResponse(json_body={}))
        cs.record_publish_attribution("tok", "rule", published_at="2026-01-01")
        assert client.last["json"] == {
            "asset_type": "rule", "published_at": "2026-01-01"}


# ---------------------------------------------------------------------------
# Ratings
# ---------------------------------------------------------------------------


class TestRatings:
    def test_rate_minimal_payload(self, base_url, monkeypatch):
        client = _install_client(
            monkeypatch, response=_FakeResponse(json_body={}))
        cs.rate("tok", "rule", "pub-1", 5)
        assert client.last["method"] == "POST"
        assert client.last["url"] == base_url + "/ratings"
        assert client.last["json"] == {
            "asset_type": "rule", "asset_public_id": "pub-1", "score": 5}
        assert "created_by_username" not in client.last["json"]

    def test_rate_includes_created_by(self, base_url, monkeypatch):
        client = _install_client(
            monkeypatch, response=_FakeResponse(json_body={}))
        cs.rate("tok", "rule", "pub-1", 3, created_by_username="alice")
        assert client.last["json"]["created_by_username"] == "alice"

    def test_delete_rating_params(self, base_url, monkeypatch):
        client = _install_client(
            monkeypatch, response=_FakeResponse(json_body={}))
        cs.delete_rating("tok", "ce", "pub-7", created_by_username="bob")
        assert client.last["method"] == "DELETE"
        assert client.last["url"] == base_url + "/ratings/ce/pub-7"
        assert client.last["params"] == {"created_by_username": "bob"}

    def test_delete_rating_no_created_by_empty_params(self, base_url, monkeypatch):
        client = _install_client(
            monkeypatch, response=_FakeResponse(json_body={}))
        cs.delete_rating("tok", "ce", "pub-7")
        assert client.last["params"] == {}

    def test_get_rating_summary_path(self, base_url, monkeypatch):
        client = _install_client(
            monkeypatch, response=_FakeResponse(json_body={"avg": 4.0}))
        out = cs.get_rating_summary("tok", "rule", "pub-2")
        assert out == {"avg": 4.0}
        assert client.last["url"] == base_url + "/ratings/rule/pub-2"


# ---------------------------------------------------------------------------
# User discovery + search
# ---------------------------------------------------------------------------


class TestUserDiscovery:
    def test_get_profile_path(self, base_url, monkeypatch):
        client = _install_client(
            monkeypatch, response=_FakeResponse(json_body={"u": 1}))
        cs.get_profile("alice")
        assert client.last["url"] == base_url + "/users/profile/alice"

    def test_search_users_default_params(self, base_url, monkeypatch):
        client = _install_client(
            monkeypatch, response=_FakeResponse(json_body={}))
        cs.search_users()
        assert client.last["params"] == {"page": 1, "page_size": 20}
        assert "q" not in client.last["params"]

    def test_search_users_includes_query(self, base_url, monkeypatch):
        client = _install_client(
            monkeypatch, response=_FakeResponse(json_body={}))
        cs.search_users(q="bob", page=2, page_size=5)
        assert client.last["params"] == {"page": 2, "page_size": 5, "q": "bob"}

    def test_leaderboard_params(self, base_url, monkeypatch):
        client = _install_client(
            monkeypatch, response=_FakeResponse(json_body={}))
        cs.leaderboard(sort="contributions", page=3, page_size=10)
        assert client.last["params"] == {
            "sort": "contributions", "page": 3, "page_size": 10, "min_ratings": 0}

    def test_leaderboard_forwards_min_ratings(self, base_url, monkeypatch):
        client = _install_client(
            monkeypatch, response=_FakeResponse(json_body={}))
        cs.leaderboard(sort="rating", page=1, page_size=20, min_ratings=5)
        assert client.last["params"] == {
            "sort": "rating", "page": 1, "page_size": 20, "min_ratings": 5}


# ---------------------------------------------------------------------------
# HuggingFace write proxy
# ---------------------------------------------------------------------------


class TestHfProxy:
    def test_hf_head_sha_extracts_sha(self, base_url, monkeypatch):
        _install_client(
            monkeypatch, response=_FakeResponse(json_body={"sha": "deadbeef"}))
        assert cs.hf_head_sha("tok") == "deadbeef"

    def test_hf_head_sha_missing_returns_none(self, base_url, monkeypatch):
        _install_client(
            monkeypatch, response=_FakeResponse(json_body={}))
        assert cs.hf_head_sha("tok") is None

    def test_hf_head_sha_non_dict_returns_none(self, base_url, monkeypatch):
        _install_client(
            monkeypatch,
            response=_FakeResponse(
                json_body=None, text="x", content_type="text/plain"))
        assert cs.hf_head_sha("tok") is None

    def test_hf_commit_base64_encodes_content(self, base_url, monkeypatch):
        import base64
        client = _install_client(
            monkeypatch, response=_FakeResponse(json_body={"commit": "c1"}))
        ops = [{"path": "a.txt", "content": b"hello"}]
        cs.hf_commit("tok", operations=ops, commit_message="msg")
        sent = client.last["json"]
        assert sent["commit_message"] == "msg"
        assert sent["parent_commit"] is None
        # Content is base64-encoded for transport; decode round-trips.
        enc = sent["operations"][0]["content_b64"]
        assert base64.b64decode(enc) == b"hello"
        assert sent["operations"][0]["path"] == "a.txt"

    def test_hf_commit_passes_parent_commit(self, base_url, monkeypatch):
        client = _install_client(
            monkeypatch, response=_FakeResponse(json_body={}))
        cs.hf_commit("tok", operations=[{"path": "p", "content": b""}],
                     commit_message="m", parent_commit="abc")
        assert client.last["json"]["parent_commit"] == "abc"

    def test_hf_commit_empty_operations(self, base_url, monkeypatch):
        client = _install_client(
            monkeypatch, response=_FakeResponse(json_body={}))
        cs.hf_commit("tok", operations=[], commit_message="m")
        assert client.last["json"]["operations"] == []


# ---------------------------------------------------------------------------
# RPC logging breadcrumbs
# ---------------------------------------------------------------------------


class TestRpcLogging:
    def test_success_logs_breadcrumb(self, base_url, monkeypatch, caplog):
        _install_client(
            monkeypatch, response=_FakeResponse(status_code=200, json_body={}))
        with caplog.at_level("INFO", logger="central-rpc"):
            cs._request("GET", "/auth/me", token="tok")
        # The one-line breadcrumb records method, path, and status code.
        msgs = " ".join(r.getMessage() for r in caplog.records)
        assert "GET" in msgs
        assert "/auth/me" in msgs
        assert "200" in msgs

    def test_failure_logs_warning(self, base_url, monkeypatch, caplog):
        _install_client(monkeypatch, exc=httpx.ConnectError("refused"))
        with caplog.at_level("WARNING", logger="central-rpc"):
            with pytest.raises(cs.CentralServerError):
                cs._request("GET", "/auth/me", token="tok")
        msgs = " ".join(r.getMessage() for r in caplog.records)
        assert "FAIL" in msgs
        assert "/auth/me" in msgs


# ---------------------------------------------------------------------------
# close()
# ---------------------------------------------------------------------------


class TestClose:
    def test_close_calls_client_close(self, monkeypatch):
        calls = []

        class _Closeable:
            def close(self):
                calls.append(True)

        monkeypatch.setattr(cs, "_client", _Closeable())
        cs.close()
        assert calls == [True]

    def test_close_swallows_exceptions(self, monkeypatch):
        class _Exploding:
            def close(self):
                raise RuntimeError("already closed")

        monkeypatch.setattr(cs, "_client", _Exploding())
        # Must not raise.
        cs.close()
