"""Pure unit tests for the policy-drift / retraining helpers in
`sql_scripts.model_scripts`.

There is NO database in this environment, so every DB seam is monkeypatched:
`execute_query` / `execute_query_dict` are imported INTO model_scripts at
module load, so we patch them on the `model_scripts` module namespace. The
draft-creation path also calls `upsert_rule_with_links`, imported at call time
from `gavel_pipeline.db_access`, so we patch it there.

The real `compute_rule_fingerprint_from_links` (a pure, DB-free helper) is left
in place — the fingerprint functions are deterministic and exercising the real
one keeps the fingerprint-equivalence assertions honest.

Functions covered:
  * compute_classifier_policy_fingerprint
  * reconcile_classifier_status
  * create_draft_rule_from_bookmarked
  * _build_predicate_from_roles
"""
import hashlib

import pytest

from sql_scripts import model_scripts
from sql_scripts.junction_scripts import compute_rule_fingerprint_from_links


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _DictDB:
    """Records calls to execute_query_dict and replays a queued list of
    return values (one per call). Lets a test assert on the SQL/params seen."""

    def __init__(self, returns):
        # `returns` is a list; pop from the front for each call.
        self._returns = list(returns)
        self.calls = []

    def __call__(self, sql, params=None):
        self.calls.append((sql, params))
        if self._returns:
            return self._returns.pop(0)
        return []


class _RecordingWriter:
    """Records calls to execute_query (the write/UPDATE seam)."""

    def __init__(self):
        self.calls = []

    def __call__(self, sql, params=None):
        self.calls.append((sql, params))
        return None


# ===========================================================================
# compute_classifier_policy_fingerprint
# ===========================================================================


class TestComputeFingerprint:
    def test_no_active_links_returns_empty_string(self, monkeypatch):
        # The query returns no rows -> by_setup is empty -> canonical == "" -> ''.
        db = _DictDB([[]])
        monkeypatch.setattr(model_scripts, "execute_query_dict", db)
        assert model_scripts.compute_classifier_policy_fingerprint(1) == ""
        assert len(db.calls) == 1
        # classifier_id should be threaded into the params.
        assert db.calls[0][1] == (1,)

    def test_none_rows_treated_as_empty(self, monkeypatch):
        # `execute_query_dict(...) or []` must turn a None result into ''.
        monkeypatch.setattr(
            model_scripts, "execute_query_dict", lambda *a, **k: None
        )
        assert model_scripts.compute_classifier_policy_fingerprint(99) == ""

    def test_single_rule_matches_sha256_of_fingerprint(self, monkeypatch):
        rows = [
            {"setup_id": 10, "ce_id": 5, "role": "necessary", "fallback_group": 0},
        ]
        monkeypatch.setattr(
            model_scripts, "execute_query_dict", lambda *a, **k: rows
        )
        result = model_scripts.compute_classifier_policy_fingerprint(1)

        rule_fp = compute_rule_fingerprint_from_links(
            [{"ce_id": 5, "role": "necessary", "fallback_group": 0}]
        )
        expected = hashlib.sha256(rule_fp.encode("utf-8")).hexdigest()
        assert result == expected
        assert len(result) == 64  # sha256 hex digest length

    def test_fingerprint_independent_of_rule_order(self, monkeypatch):
        # Two setups; the function sorts per-rule fingerprints before hashing,
        # so swapping setup order in the rows must NOT change the result.
        rows_a = [
            {"setup_id": 1, "ce_id": 5, "role": "necessary", "fallback_group": 0},
            {"setup_id": 2, "ce_id": 9, "role": "necessary", "fallback_group": 0},
        ]
        rows_b = [
            {"setup_id": 2, "ce_id": 9, "role": "necessary", "fallback_group": 0},
            {"setup_id": 1, "ce_id": 5, "role": "necessary", "fallback_group": 0},
        ]
        monkeypatch.setattr(model_scripts, "execute_query_dict", lambda *a, **k: rows_a)
        fp_a = model_scripts.compute_classifier_policy_fingerprint(1)
        monkeypatch.setattr(model_scripts, "execute_query_dict", lambda *a, **k: rows_b)
        fp_b = model_scripts.compute_classifier_policy_fingerprint(1)
        assert fp_a == fp_b

    def test_fingerprint_independent_of_setup_id(self, monkeypatch):
        # Same CE composition but different setup_ids (re-added rule mints a new
        # setup_id) must yield the SAME fingerprint — the documented property.
        rows_a = [
            {"setup_id": 100, "ce_id": 5, "role": "necessary", "fallback_group": 0},
        ]
        rows_b = [
            {"setup_id": 777, "ce_id": 5, "role": "necessary", "fallback_group": 0},
        ]
        monkeypatch.setattr(model_scripts, "execute_query_dict", lambda *a, **k: rows_a)
        fp_a = model_scripts.compute_classifier_policy_fingerprint(1)
        monkeypatch.setattr(model_scripts, "execute_query_dict", lambda *a, **k: rows_b)
        fp_b = model_scripts.compute_classifier_policy_fingerprint(1)
        assert fp_a == fp_b

    def test_different_composition_yields_different_fingerprint(self, monkeypatch):
        rows_a = [
            {"setup_id": 1, "ce_id": 5, "role": "necessary", "fallback_group": 0},
        ]
        rows_b = [
            {"setup_id": 1, "ce_id": 6, "role": "necessary", "fallback_group": 0},
        ]
        monkeypatch.setattr(model_scripts, "execute_query_dict", lambda *a, **k: rows_a)
        fp_a = model_scripts.compute_classifier_policy_fingerprint(1)
        monkeypatch.setattr(model_scripts, "execute_query_dict", lambda *a, **k: rows_b)
        fp_b = model_scripts.compute_classifier_policy_fingerprint(1)
        assert fp_a != fp_b


# ===========================================================================
# reconcile_classifier_status
# ===========================================================================


class TestReconcileStatus:
    def test_missing_classifier_returns_untrained(self, monkeypatch):
        monkeypatch.setattr(model_scripts, "execute_query_dict", lambda *a, **k: [])
        writer = _RecordingWriter()
        monkeypatch.setattr(model_scripts, "execute_query", writer)
        assert model_scripts.reconcile_classifier_status(1) == "untrained"
        assert writer.calls == []  # no write-back

    def test_none_rows_returns_untrained(self, monkeypatch):
        monkeypatch.setattr(model_scripts, "execute_query_dict", lambda *a, **k: None)
        assert model_scripts.reconcile_classifier_status(1) == "untrained"

    @pytest.mark.parametrize("status", ["untrained", "training", "error"])
    def test_passthrough_non_trainable_status(self, status, monkeypatch):
        rows = [{"status": status, "trained_policy_fingerprint": "abc"}]
        monkeypatch.setattr(model_scripts, "execute_query_dict", lambda *a, **k: rows)
        writer = _RecordingWriter()
        monkeypatch.setattr(model_scripts, "execute_query", writer)
        assert model_scripts.reconcile_classifier_status(1) == status
        assert writer.calls == []

    def test_no_fingerprint_snapshot_passes_through(self, monkeypatch):
        # active but no trained_policy_fingerprint (legacy model) -> unchanged.
        rows = [{"status": "active", "trained_policy_fingerprint": None}]
        monkeypatch.setattr(model_scripts, "execute_query_dict", lambda *a, **k: rows)
        writer = _RecordingWriter()
        monkeypatch.setattr(model_scripts, "execute_query", writer)
        assert model_scripts.reconcile_classifier_status(1) == "active"
        assert writer.calls == []

    def test_missing_fingerprint_key_passes_through(self, monkeypatch):
        # `.get("trained_policy_fingerprint")` returns None when key absent.
        rows = [{"status": "active"}]
        monkeypatch.setattr(model_scripts, "execute_query_dict", lambda *a, **k: rows)
        assert model_scripts.reconcile_classifier_status(1) == "active"

    def test_no_drift_stays_active_no_writeback(self, monkeypatch):
        trained_fp = "deadbeef"
        rows = [{"status": "active", "trained_policy_fingerprint": trained_fp}]
        monkeypatch.setattr(model_scripts, "execute_query_dict", lambda *a, **k: rows)
        monkeypatch.setattr(
            model_scripts,
            "compute_classifier_policy_fingerprint",
            lambda cid: trained_fp,
        )
        writer = _RecordingWriter()
        monkeypatch.setattr(model_scripts, "execute_query", writer)
        assert model_scripts.reconcile_classifier_status(1) == "active"
        # new_status == status -> no UPDATE.
        assert writer.calls == []

    def test_drift_flips_to_needs_retraining_and_writes_back(self, monkeypatch):
        rows = [{"status": "active", "trained_policy_fingerprint": "trained"}]
        monkeypatch.setattr(model_scripts, "execute_query_dict", lambda *a, **k: rows)
        monkeypatch.setattr(
            model_scripts,
            "compute_classifier_policy_fingerprint",
            lambda cid: "current_differs",
        )
        writer = _RecordingWriter()
        monkeypatch.setattr(model_scripts, "execute_query", writer)
        assert model_scripts.reconcile_classifier_status(7) == "needs_retraining"
        # One write-back UPDATE with the new status + classifier_id.
        assert len(writer.calls) == 1
        sql, params = writer.calls[0]
        assert "UPDATE classifiers" in sql
        assert params == ("needs_retraining", 7)

    def test_drift_resolved_heals_back_to_active(self, monkeypatch):
        # Stored status is needs_retraining but live policy now matches trained
        # snapshot again -> self-heals to active AND writes it back.
        rows = [{"status": "needs_retraining", "trained_policy_fingerprint": "fp"}]
        monkeypatch.setattr(model_scripts, "execute_query_dict", lambda *a, **k: rows)
        monkeypatch.setattr(
            model_scripts, "compute_classifier_policy_fingerprint", lambda cid: "fp"
        )
        writer = _RecordingWriter()
        monkeypatch.setattr(model_scripts, "execute_query", writer)
        assert model_scripts.reconcile_classifier_status(3) == "active"
        assert len(writer.calls) == 1
        assert writer.calls[0][1] == ("active", 3)

    def test_still_needs_retraining_no_redundant_writeback(self, monkeypatch):
        # Stored needs_retraining and still drifting -> stays needs_retraining,
        # and since new_status == status, no write-back occurs.
        rows = [{"status": "needs_retraining", "trained_policy_fingerprint": "fp"}]
        monkeypatch.setattr(model_scripts, "execute_query_dict", lambda *a, **k: rows)
        monkeypatch.setattr(
            model_scripts, "compute_classifier_policy_fingerprint", lambda cid: "other"
        )
        writer = _RecordingWriter()
        monkeypatch.setattr(model_scripts, "execute_query", writer)
        assert model_scripts.reconcile_classifier_status(3) == "needs_retraining"
        assert writer.calls == []


# ===========================================================================
# _build_predicate_from_roles
# ===========================================================================


class TestBuildPredicate:
    def test_empty_roles_returns_empty(self):
        assert model_scripts._build_predicate_from_roles([], {}) == ""

    def test_single_necessary(self):
        roles = [{"ce_id": 1, "role": "necessary"}]
        assert model_scripts._build_predicate_from_roles(roles, {1: "A"}) == "A"

    def test_multiple_necessary_joined_with_and(self):
        roles = [
            {"ce_id": 1, "role": "necessary"},
            {"ce_id": 2, "role": "necessary"},
        ]
        out = model_scripts._build_predicate_from_roles(roles, {1: "A", 2: "B"})
        assert out == "A AND B"

    def test_default_role_is_necessary(self):
        # No "role" key -> treated as necessary.
        roles = [{"ce_id": 1}]
        assert model_scripts._build_predicate_from_roles(roles, {1: "A"}) == "A"

    def test_missing_name_falls_back_to_ce_placeholder(self):
        # name_map has no entry for ce_id 42 -> "CE_42".
        roles = [{"ce_id": 42, "role": "necessary"}]
        assert model_scripts._build_predicate_from_roles(roles, {}) == "CE_42"

    def test_fallback_group_wrapped_in_or_parens(self):
        roles = [
            {"ce_id": 1, "role": "fallback", "fallback_group": 1},
            {"ce_id": 2, "role": "fallback", "fallback_group": 1},
        ]
        out = model_scripts._build_predicate_from_roles(roles, {1: "A", 2: "B"})
        assert out == "(A OR B)"

    def test_fallback_group_zero_grouped(self):
        # group 0 is a valid group key (no longer promoted); both CEs share it.
        roles = [
            {"ce_id": 1, "role": "fallback", "fallback_group": 0},
            {"ce_id": 2, "role": "fallback", "fallback_group": 0},
        ]
        out = model_scripts._build_predicate_from_roles(roles, {1: "A", 2: "B"})
        assert out == "(A OR B)"

    def test_distinct_fallback_groups_zero_and_one_stay_separate(self):
        # Regression: the old max(fb,1) collapsed groups 0 and 1 into one.
        roles = [
            {"ce_id": 1, "role": "fallback", "fallback_group": 0},
            {"ce_id": 2, "role": "fallback", "fallback_group": 1},
        ]
        out = model_scripts._build_predicate_from_roles(roles, {1: "A", 2: "B"})
        assert out == "(A) AND (B)"

    def test_necessary_and_fallback_combined(self):
        roles = [
            {"ce_id": 1, "role": "necessary"},
            {"ce_id": 2, "role": "fallback", "fallback_group": 1},
            {"ce_id": 3, "role": "fallback", "fallback_group": 1},
        ]
        out = model_scripts._build_predicate_from_roles(
            roles, {1: "A", 2: "B", 3: "C"}
        )
        assert out == "A AND (B OR C)"

    def test_multiple_fallback_groups_sorted(self):
        roles = [
            {"ce_id": 1, "role": "fallback", "fallback_group": 2},
            {"ce_id": 2, "role": "fallback", "fallback_group": 1},
        ]
        out = model_scripts._build_predicate_from_roles(roles, {1: "G2", 2: "G1"})
        # Groups iterated in sorted key order: group 1 first, then group 2.
        assert out == "(G1) AND (G2)"

    def test_sufficient_is_helpful_only_excluded_from_predicate(self):
        # 'sufficient' CEs are HELPFUL-only: they never appear in the firing
        # predicate (reference detect_uc ignores supporting CEs).
        roles = [{"ce_id": 1, "role": "sufficient"}]
        assert model_scripts._build_predicate_from_roles(roles, {1: "A"}) == ""

    def test_multiple_sufficient_all_excluded(self):
        roles = [
            {"ce_id": 1, "role": "sufficient"},
            {"ce_id": 2, "role": "sufficient"},
        ]
        out = model_scripts._build_predicate_from_roles(roles, {1: "A", 2: "B"})
        assert out == ""

    def test_base_predicate_drops_sufficient(self):
        # necessary stays; sufficient is omitted entirely (no OR tail).
        roles = [
            {"ce_id": 1, "role": "necessary"},
            {"ce_id": 2, "role": "sufficient"},
        ]
        out = model_scripts._build_predicate_from_roles(roles, {1: "A", 2: "B"})
        assert out == "A"

    def test_none_fallback_group_coerced_to_zero(self):
        # fallback_group None -> int(None or 0) == 0 -> group 0 (valid key).
        roles = [{"ce_id": 1, "role": "fallback", "fallback_group": None}]
        out = model_scripts._build_predicate_from_roles(roles, {1: "A"})
        assert out == "(A)"


# ===========================================================================
# create_draft_rule_from_bookmarked
# ===========================================================================


class TestCreateDraftFromBookmarked:
    def _patch_upsert(self, monkeypatch, capture):
        """Patch the call-time import target so model_scripts picks up our fake
        upsert_rule_with_links from gavel_pipeline.db_access."""
        import gavel_pipeline.db_access as dba

        def fake_upsert(rule_data, mark_pending=False):
            capture["rule_data"] = rule_data
            capture["mark_pending"] = mark_pending
            return 4242

        monkeypatch.setattr(dba, "upsert_rule_with_links", fake_upsert)

    def test_empty_ce_roles_raises(self, monkeypatch):
        with pytest.raises(ValueError, match="ce_roles cannot be empty"):
            model_scripts.create_draft_rule_from_bookmarked("R", [])

    def test_none_ce_roles_raises(self, monkeypatch):
        with pytest.raises(ValueError):
            model_scripts.create_draft_rule_from_bookmarked("R", None)

    def test_happy_path_returns_rule_id_and_predicate(self, monkeypatch):
        ce_roles = [
            {"ce_id": 1, "role": "necessary"},
            {"ce_id": 2, "role": "fallback", "fallback_group": 1},
            {"ce_id": 3, "role": "fallback", "fallback_group": 1},
            {"ce_id": 4, "role": "sufficient"},
        ]
        name_rows = [
            {"ce_id": 1, "name": "Alpha"},
            {"ce_id": 2, "name": "Beta"},
            {"ce_id": 3, "name": "Gamma"},
            {"ce_id": 4, "name": "Delta"},
        ]
        db = _DictDB([name_rows])
        monkeypatch.setattr(model_scripts, "execute_query_dict", db)
        capture = {}
        self._patch_upsert(monkeypatch, capture)

        rule_id, predicate = model_scripts.create_draft_rule_from_bookmarked(
            "My Draft", ce_roles, categories=[7, 8]
        )

        assert rule_id == 4242
        # 'Delta' is sufficient/helpful — excluded from the firing predicate,
        # but still carried in rule_data["sufficient"] for the "Helpful" UI.
        assert predicate == "Alpha AND (Beta OR Gamma)"
        # mark_pending must be True (documented hidden-draft behavior).
        assert capture["mark_pending"] is True
        rd = capture["rule_data"]
        assert rd["rule_name"] == "My Draft"
        assert rd["predicate"] == predicate
        assert rd["necessary"] == ["Alpha"]
        assert rd["sufficient"] == ["Delta"]
        # fallback is a list of groups (ordered by group key).
        assert rd["fallback"] == [["Beta", "Gamma"]]
        assert rd["categories"] == [7, 8]
        assert rd["description"] == ""

    def test_categories_default_to_empty_list(self, monkeypatch):
        ce_roles = [{"ce_id": 1, "role": "necessary"}]
        db = _DictDB([[{"ce_id": 1, "name": "Alpha"}]])
        monkeypatch.setattr(model_scripts, "execute_query_dict", db)
        capture = {}
        self._patch_upsert(monkeypatch, capture)

        rule_id, predicate = model_scripts.create_draft_rule_from_bookmarked(
            "X", ce_roles
        )
        assert predicate == "Alpha"
        assert capture["rule_data"]["categories"] == []

    def test_ce_with_no_name_row_is_skipped(self, monkeypatch):
        # ce_id 2 has no matching name row -> dropped from buckets/predicate,
        # but the predicate built by _build_predicate_from_roles still uses
        # the CE_<id> placeholder. Buckets, however, skip unknown names.
        ce_roles = [
            {"ce_id": 1, "role": "necessary"},
            {"ce_id": 2, "role": "necessary"},
        ]
        db = _DictDB([[{"ce_id": 1, "name": "Alpha"}]])
        monkeypatch.setattr(model_scripts, "execute_query_dict", db)
        capture = {}
        self._patch_upsert(monkeypatch, capture)

        rule_id, predicate = model_scripts.create_draft_rule_from_bookmarked(
            "X", ce_roles
        )
        # Bucket only contains the named CE (unknown skipped via `if not ce_name`).
        assert capture["rule_data"]["necessary"] == ["Alpha"]
        # Predicate uses placeholder for the unnamed CE.
        assert predicate == "Alpha AND CE_2"

    def test_roles_with_none_ce_id_excluded_from_lookup(self, monkeypatch):
        # ce_id None should not be added to the IN (...) lookup list.
        ce_roles = [
            {"ce_id": 1, "role": "necessary"},
            {"role": "necessary"},  # no ce_id
        ]
        db = _DictDB([[{"ce_id": 1, "name": "Alpha"}]])
        monkeypatch.setattr(model_scripts, "execute_query_dict", db)
        capture = {}
        self._patch_upsert(monkeypatch, capture)

        model_scripts.create_draft_rule_from_bookmarked("X", ce_roles)
        # The CE-name lookup params must only carry the non-None ce_id.
        sql, params = db.calls[0]
        assert params == (1,)

    def test_no_valid_ce_ids_skips_db_lookup(self, monkeypatch):
        # Every role has ce_id None -> ce_ids == [] -> ce_rows stays [] and NO
        # execute_query_dict call is made (the `if ce_ids else []` branch).
        ce_roles = [{"role": "necessary"}, {"role": "sufficient"}]
        db = _DictDB([])  # would raise if popped, but should never be called
        monkeypatch.setattr(model_scripts, "execute_query_dict", db)
        capture = {}
        self._patch_upsert(monkeypatch, capture)

        rule_id, predicate = model_scripts.create_draft_rule_from_bookmarked(
            "X", ce_roles
        )
        assert db.calls == []  # no DB lookup attempted
        assert rule_id == 4242
        # All names unknown -> buckets empty; predicate uses CE_None placeholders.
        assert capture["rule_data"]["necessary"] == []
        assert capture["rule_data"]["sufficient"] == []
