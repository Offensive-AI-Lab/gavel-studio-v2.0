"""Pure unit tests for the browse/extract helpers in routes.realtime.

These cover the non-LLM, non-torch helpers that only touch the DB and a couple
of normalization utilities:

  * _usable_dialogue        — the user+assistant-content predicate
  * _extract_ce_conversations — table fallback, JSON decoding, item shapes
  * _list_sample_groups     — test-dataset catalog + per-CE calibration groups
  * _load_sample_group      — testds:/ce:/unknown key dispatch

Every DB call goes through `execute_query_dict`, which is imported at module
top into routes.realtime, so we monkeypatch `routes.realtime.execute_query_dict`.
The two helpers that lazy-import `_normalize_conversation` /
`get_classifier_labels` pull them from their source modules at call time, so we
patch those sources (`evaluation.inference._normalize_conversation`,
`evaluation.ruleset_builder.get_classifier_labels`). No DB, no torch, no LLM is
ever touched.
"""
import json

import pytest

import routes.realtime as rt


# ---------------------------------------------------------------------------
# Helpers / fakes
# ---------------------------------------------------------------------------

def _identity_normalize(conv):
    """A stand-in for evaluation.inference._normalize_conversation.

    For a list of message dicts it returns them coerced to {role, content};
    for anything else it returns a single empty user message. This is enough
    to drive the realtime helpers without importing the heavy real one.
    """
    if isinstance(conv, list):
        out = []
        for i, m in enumerate(conv):
            if isinstance(m, dict):
                role = m.get("role") or ("assistant" if i % 2 else "user")
                content = m.get("content") or ""
                out.append({"role": str(role), "content": content})
            elif isinstance(m, str):
                out.append({"role": "assistant" if i % 2 else "user", "content": m})
        return out
    if isinstance(conv, str):
        return [{"role": "assistant", "content": conv}]
    return [{"role": "user", "content": ""}]


def _patch_normalize(monkeypatch):
    """Patch _normalize_conversation in its source module (lazy-imported)."""
    import evaluation.inference as inf
    monkeypatch.setattr(inf, "_normalize_conversation", _identity_normalize)


def _make_eqd(monkeypatch, handler):
    """Install a fake execute_query_dict on routes.realtime.

    `handler(sql, params)` returns the canned rows (a list) for each call.
    Also records the calls for assertions.
    """
    calls = []

    def fake(sql, params=None):
        calls.append((sql, params))
        return handler(sql, params)

    monkeypatch.setattr(rt, "execute_query_dict", fake)
    return calls


# ===========================================================================
# _usable_dialogue
# ===========================================================================

class TestUsableDialogue:
    def test_happy_path_user_then_assistant(self):
        msgs = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        assert rt._usable_dialogue(msgs) is True

    def test_empty_list(self):
        assert rt._usable_dialogue([]) is False

    def test_only_user(self):
        assert rt._usable_dialogue([{"role": "user", "content": "hi"}]) is False

    def test_only_assistant(self):
        assert rt._usable_dialogue([{"role": "assistant", "content": "hi"}]) is False

    def test_assistant_with_empty_content_is_not_usable(self):
        msgs = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": ""},
        ]
        assert rt._usable_dialogue(msgs) is False

    def test_role_is_case_insensitive(self):
        msgs = [
            {"role": "User", "content": "hi"},
            {"role": "ASSISTANT", "content": "yo"},
        ]
        assert rt._usable_dialogue(msgs) is True

    def test_missing_role_key_does_not_crash(self):
        # m.get("role") -> None, ("" ) .lower() guards against None.
        msgs = [
            {"content": "hi"},
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a"},
        ]
        assert rt._usable_dialogue(msgs) is True

    def test_none_role_value(self):
        msgs = [
            {"role": None, "content": "x"},
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a"},
        ]
        assert rt._usable_dialogue(msgs) is True

    def test_assistant_content_none_not_usable(self):
        msgs = [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": None},
        ]
        assert rt._usable_dialogue(msgs) is False


# ===========================================================================
# _extract_ce_conversations
# ===========================================================================

class TestExtractCeConversations:
    def test_prefers_calibration_table(self, monkeypatch):
        _patch_normalize(monkeypatch)
        convo = [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a"},
        ]

        def handler(sql, params):
            if "calibration_datasets" in sql:
                return [{"dataset": [convo]}]
            raise AssertionError("should not query excitation when calibration hits")

        calls = _make_eqd(monkeypatch, handler)
        out = rt._extract_ce_conversations(7)
        assert out == [convo]
        # Only the calibration table was queried.
        assert len(calls) == 1
        assert "calibration_datasets" in calls[0][0]
        assert calls[0][1] == (7,)

    def test_falls_back_to_excitation_when_calibration_empty(self, monkeypatch):
        _patch_normalize(monkeypatch)
        convo = [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a"},
        ]

        def handler(sql, params):
            if "calibration_datasets" in sql:
                return []  # nothing in calibration
            if "excitation_datasets" in sql:
                return [{"dataset": [convo]}]
            return []

        _make_eqd(monkeypatch, handler)
        out = rt._extract_ce_conversations(3)
        assert out == [convo]

    def test_falls_back_when_calibration_dataset_is_falsy(self, monkeypatch):
        # Row exists but dataset is None -> not usable, must fall through.
        _patch_normalize(monkeypatch)
        convo = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]

        def handler(sql, params):
            if "calibration_datasets" in sql:
                return [{"dataset": None}]
            return [{"dataset": [convo]}]

        _make_eqd(monkeypatch, handler)
        out = rt._extract_ce_conversations(3)
        assert out == [convo]

    def test_no_rows_at_all_returns_empty(self, monkeypatch):
        _patch_normalize(monkeypatch)
        _make_eqd(monkeypatch, lambda sql, params: [])
        assert rt._extract_ce_conversations(99) == []

    def test_execute_returns_none_handled(self, monkeypatch):
        # execute_query_dict may return None; the `or []` guards both reads.
        _patch_normalize(monkeypatch)
        _make_eqd(monkeypatch, lambda sql, params: None)
        assert rt._extract_ce_conversations(1) == []

    def test_dataset_as_json_string_is_decoded(self, monkeypatch):
        _patch_normalize(monkeypatch)
        convo = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]
        raw = json.dumps([convo])

        def handler(sql, params):
            if "calibration_datasets" in sql:
                return [{"dataset": raw}]
            return []

        _make_eqd(monkeypatch, handler)
        out = rt._extract_ce_conversations(1)
        assert out == [convo]

    def test_invalid_json_string_returns_empty(self, monkeypatch):
        _patch_normalize(monkeypatch)

        def handler(sql, params):
            if "calibration_datasets" in sql:
                return [{"dataset": "{not valid json"}]
            return []

        _make_eqd(monkeypatch, handler)
        assert rt._extract_ce_conversations(1) == []

    def test_dict_dataset_uses_conversations_key(self, monkeypatch):
        _patch_normalize(monkeypatch)
        convo = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]

        def handler(sql, params):
            if "calibration_datasets" in sql:
                return [{"dataset": {"conversations": [convo]}}]
            return []

        _make_eqd(monkeypatch, handler)
        assert rt._extract_ce_conversations(1) == [convo]

    def test_dict_dataset_samples_key_fallback(self, monkeypatch):
        _patch_normalize(monkeypatch)
        convo = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]

        def handler(sql, params):
            if "calibration_datasets" in sql:
                return [{"dataset": {"samples": [convo]}}]
            return []

        _make_eqd(monkeypatch, handler)
        assert rt._extract_ce_conversations(1) == [convo]

    def test_dict_dataset_training_data_key_fallback(self, monkeypatch):
        _patch_normalize(monkeypatch)
        convo = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]

        def handler(sql, params):
            if "calibration_datasets" in sql:
                return [{"dataset": {"training_data": [convo]}}]
            return []

        _make_eqd(monkeypatch, handler)
        assert rt._extract_ce_conversations(1) == [convo]

    def test_dict_dataset_with_no_known_keys_yields_empty(self, monkeypatch):
        _patch_normalize(monkeypatch)

        def handler(sql, params):
            if "calibration_datasets" in sql:
                return [{"dataset": {"unrelated": [1, 2, 3]}}]
            return []

        _make_eqd(monkeypatch, handler)
        assert rt._extract_ce_conversations(1) == []

    def test_training_pair_item_input_output(self, monkeypatch):
        # Items shaped as {input, output} become a 2-message convo without
        # touching _normalize_conversation.
        import evaluation.inference as inf
        monkeypatch.setattr(inf, "_normalize_conversation",
                            lambda c: (_ for _ in ()).throw(AssertionError("should not be called")))

        def handler(sql, params):
            if "calibration_datasets" in sql:
                return [{"dataset": [{"input": "hello", "output": "world"}]}]
            return []

        _make_eqd(monkeypatch, handler)
        out = rt._extract_ce_conversations(1)
        assert out == [[
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "world"},
        ]]

    def test_training_pair_prompt_response(self, monkeypatch):
        def handler(sql, params):
            if "calibration_datasets" in sql:
                return [{"dataset": [{"prompt": "p", "response": "r"}]}]
            return []

        _make_eqd(monkeypatch, handler)
        out = rt._extract_ce_conversations(1)
        assert out == [[
            {"role": "user", "content": "p"},
            {"role": "assistant", "content": "r"},
        ]]

    def test_training_pair_coerces_non_str_to_str(self, monkeypatch):
        def handler(sql, params):
            if "calibration_datasets" in sql:
                return [{"dataset": [{"input": 42, "output": {"a": 1}}]}]
            return []

        _make_eqd(monkeypatch, handler)
        out = rt._extract_ce_conversations(1)
        assert out[0][0]["content"] == "42"
        assert out[0][1]["content"] == str({"a": 1})

    def test_item_with_messages_key_uses_normalize(self, monkeypatch):
        _patch_normalize(monkeypatch)
        inner = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]

        def handler(sql, params):
            if "calibration_datasets" in sql:
                return [{"dataset": [{"messages": inner}]}]
            return []

        _make_eqd(monkeypatch, handler)
        out = rt._extract_ce_conversations(1)
        assert out == [inner]

    def test_unusable_dialogue_filtered_out(self, monkeypatch):
        # A pair whose assistant content is empty -> filtered (no assistant
        # content), so the result is empty.
        _patch_normalize(monkeypatch)

        def handler(sql, params):
            if "calibration_datasets" in sql:
                return [{"dataset": [{"input": "hi", "output": ""}]}]
            return []

        _make_eqd(monkeypatch, handler)
        assert rt._extract_ce_conversations(1) == []

    def test_limit_caps_output(self, monkeypatch):
        _patch_normalize(monkeypatch)
        pair = {"input": "q", "output": "a"}
        items = [pair] * 10

        def handler(sql, params):
            if "calibration_datasets" in sql:
                return [{"dataset": items}]
            return []

        _make_eqd(monkeypatch, handler)
        out = rt._extract_ce_conversations(1, limit=3)
        assert len(out) == 3

    def test_raw_is_unexpected_scalar_type_yields_empty(self, monkeypatch):
        # dataset is an int -> not str/dict/list -> items=[] -> []
        _patch_normalize(monkeypatch)

        def handler(sql, params):
            if "calibration_datasets" in sql:
                return [{"dataset": 12345}]
            return []

        _make_eqd(monkeypatch, handler)
        assert rt._extract_ce_conversations(1) == []


# ===========================================================================
# _list_sample_groups
# ===========================================================================

def _patch_labels(monkeypatch, labels):
    import evaluation.ruleset_builder as rb
    monkeypatch.setattr(rb, "get_classifier_labels", lambda cid: labels)


class TestListSampleGroups:
    def test_test_datasets_mapped_with_type_labels(self, monkeypatch):
        _patch_labels(monkeypatch, {})  # no CEs

        def handler(sql, params):
            if "test_datasets" in sql:
                return [
                    {"dataset_id": 1, "dataset_type": "positive", "rule_name": "R1", "count": 5},
                    {"dataset_id": 2, "dataset_type": "negative", "rule_name": "R1", "count": 2},
                    {"dataset_id": 3, "dataset_type": "positive_calibration", "rule_name": "R2", "count": 4},
                ]
            return []

        _make_eqd(monkeypatch, handler)
        groups = rt._list_sample_groups(10)
        assert groups == [
            {"key": "testds:1", "label": "Test · positive · R1", "count": 5},
            {"key": "testds:2", "label": "Test · negative · R1", "count": 2},
            {"key": "testds:3", "label": "Test · calibration · R2", "count": 4},
        ]

    def test_zero_or_negative_count_test_datasets_skipped(self, monkeypatch):
        _patch_labels(monkeypatch, {})

        def handler(sql, params):
            if "test_datasets" in sql:
                return [
                    {"dataset_id": 1, "dataset_type": "positive", "rule_name": "R1", "count": 0},
                    {"dataset_id": 2, "dataset_type": "positive", "rule_name": "R2", "count": None},
                    {"dataset_id": 3, "dataset_type": "positive", "rule_name": "R3", "count": 1},
                ]
            return []

        _make_eqd(monkeypatch, handler)
        groups = rt._list_sample_groups(10)
        assert [g["key"] for g in groups] == ["testds:3"]

    def test_unknown_dataset_type_passes_through(self, monkeypatch):
        _patch_labels(monkeypatch, {})

        def handler(sql, params):
            if "test_datasets" in sql:
                return [{"dataset_id": 9, "dataset_type": "weird", "rule_name": "R", "count": 3}]
            return []

        _make_eqd(monkeypatch, handler)
        groups = rt._list_sample_groups(10)
        assert groups[0]["label"] == "Test · weird · R"

    def test_ce_groups_appended_when_conversations_present(self, monkeypatch):
        _patch_labels(monkeypatch, {"ce_alpha": 0, "ce_beta": 1})
        _patch_normalize(monkeypatch)
        convo = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]

        def handler(sql, params):
            if "test_datasets" in sql:
                return []
            if "cognitive_elements" in sql:
                # name = ANY(%s) -> params is (ce_names_list,)
                assert params == (["ce_alpha", "ce_beta"],)
                return [
                    {"ce_id": 100, "name": "ce_alpha"},
                    {"ce_id": 200, "name": "ce_beta"},
                ]
            if "calibration_datasets" in sql:
                ce_id = params[0]
                if ce_id == 100:
                    return [{"dataset": [convo, convo]}]  # 2 convos
                return []  # ce 200 calibration empty
            if "excitation_datasets" in sql:
                return []  # ce 200 has nothing -> skipped
            return []

        _make_eqd(monkeypatch, handler)
        groups = rt._list_sample_groups(10)
        # Only ce_alpha (count 2) survives; ce_beta has 0 convos.
        assert groups == [{"key": "ce:100", "label": "CE · ce_alpha", "count": 2}]

    def test_no_ce_labels_skips_ce_query(self, monkeypatch):
        _patch_labels(monkeypatch, {})  # falsy -> no CE branch

        seen = {"ce_query": False}

        def handler(sql, params):
            if "cognitive_elements" in sql:
                seen["ce_query"] = True
            if "test_datasets" in sql:
                return []
            return []

        _make_eqd(monkeypatch, handler)
        groups = rt._list_sample_groups(10)
        assert groups == []
        assert seen["ce_query"] is False

    def test_get_classifier_labels_returns_none(self, monkeypatch):
        # `(get_classifier_labels(...) or {})` guards a None return.
        _patch_labels(monkeypatch, None)

        def handler(sql, params):
            return []

        _make_eqd(monkeypatch, handler)
        assert rt._list_sample_groups(10) == []

    def test_test_datasets_query_returns_none(self, monkeypatch):
        # The `or []` after the first query guards a None result.
        _patch_labels(monkeypatch, {})

        def handler(sql, params):
            return None

        _make_eqd(monkeypatch, handler)
        assert rt._list_sample_groups(10) == []


# ===========================================================================
# _load_sample_group
# ===========================================================================

class TestLoadSampleGroup:
    def test_unknown_key_returns_empty(self, monkeypatch):
        _patch_normalize(monkeypatch)
        # No DB call expected.
        _make_eqd(monkeypatch, lambda sql, params: (_ for _ in ()).throw(AssertionError("no DB")))
        assert rt._load_sample_group("bogus:1") == []

    def test_ce_key_delegates_to_extract(self, monkeypatch):
        _patch_normalize(monkeypatch)
        convo = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]

        def handler(sql, params):
            if "calibration_datasets" in sql:
                assert params == (55,)
                return [{"dataset": [convo]}]
            return []

        _make_eqd(monkeypatch, handler)
        assert rt._load_sample_group("ce:55") == [convo]

    def test_testds_key_normalizes_and_filters(self, monkeypatch):
        _patch_normalize(monkeypatch)
        good = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]
        bad = [{"role": "user", "content": "q"}]  # no assistant -> filtered

        def handler(sql, params):
            assert "test_datasets" in sql
            assert params == (77,)
            return [{"conversations": [good, bad]}]

        _make_eqd(monkeypatch, handler)
        out = rt._load_sample_group("testds:77")
        assert out == [good]

    def test_testds_conversations_as_json_string(self, monkeypatch):
        _patch_normalize(monkeypatch)
        good = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]

        def handler(sql, params):
            return [{"conversations": json.dumps([good])}]

        _make_eqd(monkeypatch, handler)
        assert rt._load_sample_group("testds:1") == [good]

    def test_testds_no_rows_returns_empty(self, monkeypatch):
        _patch_normalize(monkeypatch)
        _make_eqd(monkeypatch, lambda sql, params: [])
        assert rt._load_sample_group("testds:1") == []

    def test_testds_null_conversations_returns_empty(self, monkeypatch):
        # Row present but conversations is None -> `(convos or [])` -> [].
        _patch_normalize(monkeypatch)
        _make_eqd(monkeypatch, lambda sql, params: [{"conversations": None}])
        assert rt._load_sample_group("testds:1") == []
