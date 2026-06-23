"""Pure unit tests for the rule structural fingerprint helpers in
``sql_scripts.junction_scripts``.

These cover:
  * ``compute_rule_fingerprint_from_links`` — the ce_id-shaped path. No DB
    access at all, so it is exercised directly.
  * ``compute_rule_fingerprint_from_names`` — the name-shaped path that does a
    single ``execute_query_dict`` name->id lookup. That DB call is
    monkeypatched (the symbol is imported INTO the target module, so we patch
    it on the module object) to return canned rows; no database is touched.

The two functions are designed so that a names-based fingerprint equals the
links-based fingerprint of the same underlying ce_ids — several tests assert
exactly that equivalence.
"""
import pytest

import sql_scripts.junction_scripts as js
from sql_scripts.junction_scripts import (
    compute_rule_fingerprint_from_links,
    compute_rule_fingerprint_from_names,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _patch_lookup(monkeypatch, name_to_id, capture=None):
    """Replace execute_query_dict (as imported into junction_scripts) with a
    fake that maps the requested names to canned {ce_id, name} rows.

    Mirrors the real query ``SELECT ce_id, name FROM cognitive_elements
    WHERE name = ANY(%s)`` — only names present in ``name_to_id`` come back,
    so unknown names are silently dropped just like the real DB would do.
    ``capture`` (a list) optionally records (query, params) for assertions.
    """
    def fake(query, params):
        if capture is not None:
            capture.append((query, params))
        requested = params[0]
        return [
            {"ce_id": name_to_id[n], "name": n}
            for n in requested
            if n in name_to_id
        ]

    monkeypatch.setattr(js, "execute_query_dict", fake)


# ===========================================================================
# compute_rule_fingerprint_from_links
# ===========================================================================

class TestFingerprintFromLinks:
    def test_empty_list_canonical_form(self):
        assert compute_rule_fingerprint_from_links([]) == "N:()|F:[]|S:()"

    def test_none_input_treated_as_empty(self):
        # ``ce_links or []`` makes None behave like an empty list.
        assert compute_rule_fingerprint_from_links(None) == "N:()|F:[]|S:()"

    def test_single_necessary(self):
        links = [{"ce_id": 5, "role": "necessary", "fallback_group": 0}]
        assert compute_rule_fingerprint_from_links(links) == "N:(5,)|F:[]|S:()"

    def test_role_defaults_to_necessary_when_missing(self):
        # No 'role' key -> (None or "necessary") -> "necessary".
        links = [{"ce_id": 7}]
        assert compute_rule_fingerprint_from_links(links) == "N:(7,)|F:[]|S:()"

    def test_role_none_defaults_to_necessary(self):
        links = [{"ce_id": 7, "role": None}]
        assert compute_rule_fingerprint_from_links(links) == "N:(7,)|F:[]|S:()"

    def test_role_is_case_insensitive(self):
        links = [
            {"ce_id": 1, "role": "NECESSARY"},
            {"ce_id": 2, "role": "Sufficient"},
        ]
        assert compute_rule_fingerprint_from_links(links) == "N:(1,)|F:[]|S:(2,)"

    def test_necessary_ids_are_sorted(self):
        links = [
            {"ce_id": 9, "role": "necessary"},
            {"ce_id": 2, "role": "necessary"},
            {"ce_id": 5, "role": "necessary"},
        ]
        assert compute_rule_fingerprint_from_links(links) == "N:(2, 5, 9)|F:[]|S:()"

    def test_sufficient_ids_are_sorted(self):
        links = [
            {"ce_id": 30, "role": "sufficient"},
            {"ce_id": 10, "role": "sufficient"},
        ]
        assert compute_rule_fingerprint_from_links(links) == "N:()|F:[]|S:(10, 30)"

    def test_ce_id_none_is_skipped(self):
        links = [
            {"ce_id": None, "role": "necessary"},
            {"ce_id": 4, "role": "necessary"},
        ]
        assert compute_rule_fingerprint_from_links(links) == "N:(4,)|F:[]|S:()"

    def test_missing_ce_id_key_is_skipped(self):
        links = [
            {"role": "necessary"},
            {"ce_id": 4, "role": "necessary"},
        ]
        assert compute_rule_fingerprint_from_links(links) == "N:(4,)|F:[]|S:()"

    def test_unknown_role_is_ignored(self):
        # A role that is not necessary/sufficient/fallback falls through all
        # branches and contributes nothing.
        links = [
            {"ce_id": 1, "role": "necessary"},
            {"ce_id": 2, "role": "bogus"},
        ]
        assert compute_rule_fingerprint_from_links(links) == "N:(1,)|F:[]|S:()"

    def test_fallback_single_group(self):
        links = [
            {"ce_id": 3, "role": "fallback", "fallback_group": 1},
            {"ce_id": 1, "role": "fallback", "fallback_group": 1},
        ]
        # Group members are sorted within the group.
        assert compute_rule_fingerprint_from_links(links) == "N:()|F:[(1, 3)]|S:()"

    def test_fallback_groups_partition_and_sort(self):
        links = [
            {"ce_id": 4, "role": "fallback", "fallback_group": 2},
            {"ce_id": 3, "role": "fallback", "fallback_group": 2},
            {"ce_id": 1, "role": "fallback", "fallback_group": 1},
            {"ce_id": 2, "role": "fallback", "fallback_group": 1},
        ]
        # Each group sorted internally; list-of-groups sorted as tuples.
        assert (
            compute_rule_fingerprint_from_links(links)
            == "N:()|F:[(1, 2), (3, 4)]|S:()"
        )

    def test_fallback_group_numbering_does_not_matter(self):
        # The docstring promises the user's group *numbering* is discarded —
        # only the partition matters. These two should be identical.
        a = [
            {"ce_id": 1, "role": "fallback", "fallback_group": 1},
            {"ce_id": 2, "role": "fallback", "fallback_group": 2},
        ]
        b = [
            {"ce_id": 1, "role": "fallback", "fallback_group": 7},
            {"ce_id": 2, "role": "fallback", "fallback_group": 99},
        ]
        assert (
            compute_rule_fingerprint_from_links(a)
            == compute_rule_fingerprint_from_links(b)
        )

    def test_fallback_group_defaults_to_zero(self):
        # Missing fallback_group -> default 0; both land in the same group.
        links = [
            {"ce_id": 2, "role": "fallback"},
            {"ce_id": 1, "role": "fallback", "fallback_group": None},
        ]
        assert compute_rule_fingerprint_from_links(links) == "N:()|F:[(1, 2)]|S:()"

    def test_fallback_group_string_is_coerced_to_int(self):
        # int(link.get("fallback_group", 0) or 0) coerces numeric strings.
        links = [
            {"ce_id": 1, "role": "fallback", "fallback_group": "1"},
            {"ce_id": 2, "role": "fallback", "fallback_group": "1"},
        ]
        assert compute_rule_fingerprint_from_links(links) == "N:()|F:[(1, 2)]|S:()"

    def test_mixed_roles_full_fingerprint(self):
        links = [
            {"ce_id": 5, "role": "necessary"},
            {"ce_id": 1, "role": "necessary"},
            {"ce_id": 8, "role": "sufficient"},
            {"ce_id": 10, "role": "fallback", "fallback_group": 1},
            {"ce_id": 9, "role": "fallback", "fallback_group": 1},
            {"ce_id": 12, "role": "fallback", "fallback_group": 2},
        ]
        assert (
            compute_rule_fingerprint_from_links(links)
            == "N:(1, 5)|F:[(9, 10), (12,)]|S:(8,)"
        )

    def test_order_of_input_does_not_change_fingerprint(self):
        links1 = [
            {"ce_id": 1, "role": "necessary"},
            {"ce_id": 2, "role": "sufficient"},
            {"ce_id": 3, "role": "fallback", "fallback_group": 1},
        ]
        links2 = list(reversed(links1))
        assert (
            compute_rule_fingerprint_from_links(links1)
            == compute_rule_fingerprint_from_links(links2)
        )


# ===========================================================================
# compute_rule_fingerprint_from_names
# ===========================================================================

class TestFingerprintFromNames:
    def test_all_empty_inputs_short_circuit_without_db(self, monkeypatch):
        # When there are no names at all, the function must NOT touch the DB;
        # it returns the empty-links fingerprint directly. We install a fake
        # that explodes to prove the lookup is never called.
        def boom(*args, **kwargs):
            raise AssertionError("execute_query_dict should not be called")

        monkeypatch.setattr(js, "execute_query_dict", boom)
        result = compute_rule_fingerprint_from_names([], [], [])
        assert result == "N:()|F:[]|S:()"
        assert result == compute_rule_fingerprint_from_links([])

    def test_all_none_inputs_short_circuit_without_db(self, monkeypatch):
        def boom(*args, **kwargs):
            raise AssertionError("execute_query_dict should not be called")

        monkeypatch.setattr(js, "execute_query_dict", boom)
        assert compute_rule_fingerprint_from_names(None, None, None) == "N:()|F:[]|S:()"

    def test_empty_fallback_groups_only_still_short_circuits(self, monkeypatch):
        # fallback is a list of empty groups -> no names collected -> no DB.
        def boom(*args, **kwargs):
            raise AssertionError("execute_query_dict should not be called")

        monkeypatch.setattr(js, "execute_query_dict", boom)
        assert compute_rule_fingerprint_from_names([], [[], []], []) == "N:()|F:[]|S:()"

    def test_necessary_names_translated_to_ids(self, monkeypatch):
        _patch_lookup(monkeypatch, {"alpha": 1, "beta": 2})
        result = compute_rule_fingerprint_from_names(["alpha", "beta"], [], [])
        assert result == "N:(1, 2)|F:[]|S:()"

    def test_sufficient_names_translated_to_ids(self, monkeypatch):
        _patch_lookup(monkeypatch, {"gamma": 10, "delta": 20})
        result = compute_rule_fingerprint_from_names([], [], ["gamma", "delta"])
        assert result == "N:()|F:[]|S:(10, 20)"

    def test_fallback_groups_get_sequential_group_indices(self, monkeypatch):
        _patch_lookup(monkeypatch, {"a": 1, "b": 2, "c": 3})
        # enumerate(..., start=1) -> group ["a","b"] = group 1, ["c"] = group 2.
        result = compute_rule_fingerprint_from_names([], [["a", "b"], ["c"]], [])
        assert result == "N:()|F:[(1, 2), (3,)]|S:()"

    def test_unknown_names_are_dropped(self, monkeypatch):
        # "ghost" has no row -> name_to_id.get returns None -> skipped.
        _patch_lookup(monkeypatch, {"known": 5})
        result = compute_rule_fingerprint_from_names(["known", "ghost"], [], [])
        assert result == "N:(5,)|F:[]|S:()"

    def test_all_names_unknown_yields_empty_fingerprint(self, monkeypatch):
        _patch_lookup(monkeypatch, {})  # DB returns no rows for anything
        result = compute_rule_fingerprint_from_names(["x", "y"], [], [])
        assert result == "N:()|F:[]|S:()"

    def test_db_returns_none_is_handled(self, monkeypatch):
        # ``execute_query_dict(...) or []`` guards a None return.
        monkeypatch.setattr(js, "execute_query_dict", lambda q, p: None)
        result = compute_rule_fingerprint_from_names(["x", "y"], [], [])
        assert result == "N:()|F:[]|S:()"

    def test_full_mixed_fingerprint(self, monkeypatch):
        _patch_lookup(
            monkeypatch,
            {"n1": 1, "n2": 5, "s1": 8, "f1": 9, "f2": 10, "f3": 12},
        )
        result = compute_rule_fingerprint_from_names(
            necessary=["n2", "n1"],
            fallback=[["f2", "f1"], ["f3"]],
            sufficient=["s1"],
        )
        assert result == "N:(1, 5)|F:[(9, 10), (12,)]|S:(8,)"

    def test_query_uses_anyof_distinct_names(self, monkeypatch):
        # The lookup should pass a *list* of the deduped set of all names.
        captured = []
        _patch_lookup(monkeypatch, {"a": 1, "b": 2}, capture=captured)
        compute_rule_fingerprint_from_names(["a", "a"], [["b"]], ["a"])
        assert len(captured) == 1
        query, params = captured[0]
        assert "name = ANY(%s)" in query
        passed_names = params[0]
        assert isinstance(passed_names, list)
        # Deduped via set(): {"a", "b"} regardless of input repetition.
        assert sorted(passed_names) == ["a", "b"]

    def test_names_path_equals_links_path_for_same_ids(self, monkeypatch):
        # The core equivalence guarantee: a names fingerprint must equal the
        # links fingerprint built from the resolved ce_ids.
        _patch_lookup(
            monkeypatch,
            {"nn": 3, "ss": 4, "fa": 7, "fb": 8},
        )
        from_names = compute_rule_fingerprint_from_names(
            necessary=["nn"],
            fallback=[["fa", "fb"]],
            sufficient=["ss"],
        )
        from_links = compute_rule_fingerprint_from_links(
            [
                {"ce_id": 3, "role": "necessary", "fallback_group": 0},
                {"ce_id": 4, "role": "sufficient", "fallback_group": 0},
                {"ce_id": 7, "role": "fallback", "fallback_group": 1},
                {"ce_id": 8, "role": "fallback", "fallback_group": 1},
            ]
        )
        assert from_names == from_links

    def test_duplicate_name_across_roles_resolves_independently(self, monkeypatch):
        # A name appearing in two roles produces a link in each role.
        _patch_lookup(monkeypatch, {"shared": 42})
        result = compute_rule_fingerprint_from_names(["shared"], [], ["shared"])
        assert result == "N:(42,)|F:[]|S:(42,)"
