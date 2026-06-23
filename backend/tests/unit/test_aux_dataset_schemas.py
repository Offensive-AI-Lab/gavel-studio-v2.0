"""Unit tests for the auxiliary-dataset Pydantic schemas + manifest fields.

These verify the schema contract that the bootstrap script writes and that
hf_sync reads. Mismatches between what gets pushed to HF and what local
clients can parse would silently truncate calibration / eval data on every
client at once — exactly the kind of regression a unit test should catch
before a PR lands.

What's covered:
  * the new Manifest fields default to empty maps (so old manifests still
    validate cleanly under the new schema)
  * each new record type validates a happy-path payload and round-trips
    its `samples` / partition lists unchanged
  * extra fields are dropped silently (forward-compat: a v2 record with
    new fields shouldn't break v1 clients)

What's NOT covered: the actual HF push. That's an integration concern and
would require either a real HF token (don't want side effects in CI) or a
lot of huggingface_hub mocking (low value vs effort). The publish path is
exercised in production runs via the bootstrap script.
"""
import pytest

from services.library_schemas import (
    Manifest,
    CECalibrationRecord,
)


# ---------------------------------------------------------------------------
# Manifest — backward compat + the three new sections
# ---------------------------------------------------------------------------


class TestManifestBackwardCompat:
    def test_old_manifest_without_aux_sections_still_validates(self):
        # An old manifest (no ce_calibration key) must validate cleanly so
        # existing clients keep syncing.
        old = {"schema_version": 1, "rules": {}, "ces": {}}
        m = Manifest.model_validate(old)
        assert m.ce_calibration == {}

    def test_new_manifest_round_trips_aux_sections(self):
        new = {
            "schema_version": 1,
            "rules": {"rule_a": "2025-01-01T00:00:00Z"},
            "ces": {"ce_a": "2025-01-01T00:00:00Z"},
            "ce_calibration": {"ce_a": "2025-01-01T00:00:00Z"},
        }
        m = Manifest.model_validate(new)
        assert "ce_a" in m.ce_calibration

    def test_legacy_rule_calibration_field_silently_ignored(self):
        # The old `rule_calibration` section (replaced by rule_datasets) must
        # validate cleanly and be dropped — extra="ignore".
        legacy = {
            "schema_version": 1,
            "rules": {}, "ces": {},
            "rule_calibration": {"rule_a": "2025-01-01T00:00:00Z"},
        }
        m = Manifest.model_validate(legacy)
        assert not hasattr(m, "rule_calibration")

    def test_legacy_rule_evaluation_field_silently_ignored(self):
        # An older manifest that carried `rule_evaluation` (decision walked
        # back) must validate cleanly — extra="ignore" drops the field
        # so legacy registry state doesn't break clients.
        legacy = {
            "schema_version": 1,
            "rules": {}, "ces": {},
            "rule_evaluation": {"rule_a": "2025-01-01T00:00:00Z"},
        }
        m = Manifest.model_validate(legacy)
        assert not hasattr(m, "rule_evaluation")

    def test_unknown_extra_fields_dropped(self):
        # Forward compat — a future schema_version 2 manifest with new
        # fields still validates under v1 (extra="ignore").
        with_extras = {
            "schema_version": 1,
            "rules": {}, "ces": {},
            "ce_calibration": {},
            "rule_calibration": {},
            "future_field": {"some": "thing"},
        }
        m = Manifest.model_validate(with_extras)
        assert not hasattr(m, "future_field")


# ---------------------------------------------------------------------------
# CECalibrationRecord
# ---------------------------------------------------------------------------


class TestCECalibrationRecord:
    def test_happy_path(self):
        sample_convo = [{"role": "user", "content": "hi"}]
        rec = CECalibrationRecord.model_validate({
            "schema_version": 1,
            "ce_public_id": "ce_abc",
            "samples": [sample_convo, sample_convo],
            "sample_count": 2,
            "published_at": "2025-01-01T00:00:00Z",
        })
        assert rec.ce_public_id == "ce_abc"
        assert len(rec.samples) == 2
        assert rec.sample_count == 2

    def test_samples_default_to_empty_list(self):
        # Defensive: a malformed record without samples still validates,
        # just empty. The downstream upsert will write an empty dataset
        # rather than raising.
        rec = CECalibrationRecord.model_validate({
            "schema_version": 1,
            "ce_public_id": "ce_x",
        })
        assert rec.samples == []
        assert rec.sample_count == 0


# ---------------------------------------------------------------------------
# RuleCalibrationRecord tests removed — the schema itself is gone now
# that rule calibration is local-only and lives in `test_datasets`.
# ---------------------------------------------------------------------------
# RuleEvaluationRecord
# ---------------------------------------------------------------------------


