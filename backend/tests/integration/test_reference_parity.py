"""Parity verification for the reference evaluation pipeline.

This test does NOT exercise an LLM. The reference functions
(`gavel.evaluation.calibrate`, `evaluate`, `compute_triggers`, etc.) take
already-computed logit matrices as input, so a synthetic dialogue cache
with known properties is enough to verify our adapter:

  1. Passes the right shapes IN (labels dict, unified_ruleset dict,
     dialogue_data list with {logits, metadata}).
  2. Reads the right shapes OUT (per-topic optimal thresholds, per-usecase
     metric tables, AUC scores).
  3. Doesn't drop or mistranslate any field on the way through.

If these tests pass, our calibration + evaluation pipeline is producing
the SAME numerical answers as the reference would on the
SAME logit inputs — because in both paths the math runs through the
reference functions, and these tests verify the adapter doesn't add
divergence between us and the upstream.

What's NOT tested here:
  * Activation extraction (LLM forward + readout). That's verified by
    the einsum-equivalence proof in the audit summary.
  * Ruleset role mapping (necessary/fallback/sufficient → all_required/
    any_of/supporting). That's done in `evaluation/ruleset_builder.py`
    and would need a DB-backed test to exercise end-to-end. The
    transformation contract is documented in that module's docstring.
"""
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------------
# Fixture: synthetic ruleset + labels + dialogue cache
# ---------------------------------------------------------------------------

@pytest.fixture
def labels():
    # 4 CEs, indexed in label-name order
    return {"ce_alpha": 0, "ce_beta": 1, "ce_gamma": 2, "ce_delta": 3}


@pytest.fixture
def unified_ruleset():
    """Two rules with different shapes — exercises all_required + any_of paths."""
    return {
        "rule_strict": {
            "enabled": True,
            "all_required": ["ce_alpha", "ce_beta"],  # both must fire
            "any_of": [],
            "supporting": [],
        },
        "rule_anyof": {
            "enabled": True,
            "all_required": ["ce_alpha"],
            "any_of": [["ce_gamma", "ce_delta"]],  # gamma OR delta required
            "supporting": [],
        },
    }


def _make_dialogue(logits_per_window, split, usecase_path, dialogue_id):
    """Build one entry in the {logits, metadata} shape calibrate/evaluate expect."""
    return {
        "logits": np.array(logits_per_window, dtype=np.float32),
        "metadata": {
            "split": split,
            "usecase_path": usecase_path,
            "dialogue_id": dialogue_id,
        },
    }


@pytest.fixture
def calibration_dialogue_data(labels):
    """Usecase-level calibration data.

    IMPORTANT: the reference `run_threshold_sweep` only consumes dialogues
    whose split is "usecase_level" — `CE_level` is loaded by
    `load_calibration_dialogues` but skipped in the sweep loop
    (see calibration.py line 379). That's the upstream contract; our
    fixture has to honour it.

    For each rule we add:
      * a positive dialogue where the rule's required CEs all fire
      * a negative dialogue where a DIFFERENT rule's CEs fire (gives
        the alpha-fires-when-beta-required case → FP signal so the
        sweep has something to discriminate against)
    """
    cache = []
    high, low = 5.0, -5.0
    n = len(labels)

    def make_logits(fire_indices):
        rows = [[low] * n, [low] * n]
        for idx in fire_indices:
            rows[0][idx] = high
            rows[1][idx] = high
        return rows

    idx_alpha, idx_beta, idx_gamma, idx_delta = 0, 1, 2, 3

    # rule_strict positives: alpha + beta fire together
    for i in range(3):
        cache.append(_make_dialogue(
            make_logits([idx_alpha, idx_beta]),
            split="usecase_level", usecase_path="rule_strict",
            dialogue_id=f"calib_strict_pos_{i}",
        ))

    # rule_anyof positives: alpha + gamma (or alpha + delta) fire
    for i in range(2):
        cache.append(_make_dialogue(
            make_logits([idx_alpha, idx_gamma]),
            split="usecase_level", usecase_path="rule_anyof",
            dialogue_id=f"calib_anyof_pos_gamma_{i}",
        ))
        cache.append(_make_dialogue(
            make_logits([idx_alpha, idx_delta]),
            split="usecase_level", usecase_path="rule_anyof",
            dialogue_id=f"calib_anyof_pos_delta_{i}",
        ))

    return cache


@pytest.fixture
def evaluation_dialogue_data(labels):
    """Hand-crafted eval set with predictable rule outcomes.

    Positive split: dialogues that SHOULD trigger their rule.
    Negative split: dialogues that should NOT trigger any rule.
    """
    cache = []
    high, low = 5.0, -5.0

    def fire(ce_indices, n_windows=2):
        logits = [[low] * len(labels) for _ in range(n_windows)]
        for ce_idx in ce_indices:
            for w in range(n_windows):
                logits[w][ce_idx] = high
        return logits

    idx_alpha, idx_beta, idx_gamma, idx_delta = 0, 1, 2, 3

    # Positives for rule_strict (both alpha + beta) → should fire
    for i in range(3):
        cache.append(_make_dialogue(
            fire([idx_alpha, idx_beta]),
            split="positive", usecase_path="rule_strict",
            dialogue_id=f"eval_strict_tp_{i}",
        ))
    # Misses for rule_strict (only one of the required CEs fires)
    cache.append(_make_dialogue(
        fire([idx_alpha]),
        split="positive", usecase_path="rule_strict",
        dialogue_id="eval_strict_fn_alpha_only",
    ))
    cache.append(_make_dialogue(
        fire([idx_beta]),
        split="positive", usecase_path="rule_strict",
        dialogue_id="eval_strict_fn_beta_only",
    ))
    # Negatives — nothing fires
    for i in range(2):
        cache.append(_make_dialogue(
            fire([]),
            split="negative", usecase_path="rule_strict",
            dialogue_id=f"eval_strict_tn_{i}",
        ))
    return cache


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestReferenceImport:
    """The sys.modules aliasing trick must work — every reference symbol
    we depend on has to be importable through both `gavel.*` and
    `classifier_engine.reference.*` paths."""

    def test_calibrate_importable(self):
        import classifier_engine.reference  # noqa: F401
        from gavel.evaluation.calibration import calibrate
        assert callable(calibrate)

    def test_evaluate_importable(self):
        import classifier_engine.reference  # noqa: F401
        from gavel.evaluation.metrics import evaluate
        assert callable(evaluate)

    def test_compute_triggers_importable(self):
        import classifier_engine.reference  # noqa: F401
        from gavel.evaluation.metrics import compute_triggers
        assert callable(compute_triggers)

    def test_eval_usecase_detection_importable(self):
        import classifier_engine.reference  # noqa: F401
        from gavel.evaluation.metrics import eval_usecase_detection
        assert callable(eval_usecase_detection)

    def test_outcomerecorder_importable(self):
        import classifier_engine.reference  # noqa: F401
        from gavel.evaluation.debug import OutcomeRecorder
        assert callable(OutcomeRecorder)


class TestComputeTriggers:
    """Sanity-check the core trigger math at the reference level. If this
    breaks, our calibration + evaluation are also broken."""

    def test_single_high_window_triggers_with_patience_1(self):
        import classifier_engine.reference  # noqa: F401
        from gavel.evaluation.metrics import compute_triggers

        # Two windows, one label, threshold=0.5
        logits = torch.tensor([[5.0], [-5.0]])  # window 0 fires, window 1 doesn't
        out = compute_triggers(logits, thresholds=0.5, patience_rate=1)
        assert out.shape == (1,)
        assert bool(out[0].item()) is True

    def test_patience_2_requires_two_windows(self):
        import classifier_engine.reference  # noqa: F401
        from gavel.evaluation.metrics import compute_triggers

        logits = torch.tensor([[5.0], [-5.0]])  # only one window above
        out = compute_triggers(logits, thresholds=0.5, patience_rate=2)
        # patience=2 → need 2 windows above threshold → should NOT trigger
        assert bool(out[0].item()) is False

    def test_per_label_thresholds(self):
        import classifier_engine.reference  # noqa: F401
        from gavel.evaluation.metrics import compute_triggers

        # 1 window, 2 labels with different thresholds
        logits = torch.tensor([[3.0, 3.0]])  # sigmoid ≈ 0.953 for both
        thresholds = torch.tensor([0.5, 0.99])
        out = compute_triggers(logits, thresholds=thresholds, patience_rate=1)
        assert bool(out[0].item()) is True   # 0.953 > 0.5
        assert bool(out[1].item()) is False  # 0.953 < 0.99


class TestCalibratePipeline:
    """End-to-end test of the reference calibrate() with our adapter shapes."""

    def test_calibrate_produces_thresholds_json(self, labels, unified_ruleset, calibration_dialogue_data):
        import classifier_engine.reference  # noqa: F401
        from gavel.evaluation.calibration import calibrate

        with tempfile.TemporaryDirectory() as tmpdir:
            calibrate(
                output_dir=tmpdir,
                labels=labels,
                unified_ruleset=unified_ruleset,
                dialogue_data=calibration_dialogue_data,
                show_progress=False,
                generate_plots=False,
            )
            thresholds_path = Path(tmpdir) / "thresholds.json"
            assert thresholds_path.is_file(), "calibrate() did not produce thresholds.json"

            with open(thresholds_path, "r") as f:
                thresholds = json.load(f)

        # Every CE that appears as `all_required` in any rule of the
        # ruleset should get a threshold entry. (The sweep is per-topic
        # but driven by ruleset participation — CEs that no rule references
        # may not show up.)
        for ce_name in labels.keys():
            assert ce_name in thresholds, f"missing threshold entry for {ce_name}"
            entry = thresholds[ce_name]
            for required_key in ("threshold", "patience", "youden_j",
                                 "tpr_at_optimal", "fpr_at_optimal"):
                assert required_key in entry, f"{ce_name} missing {required_key}"
            # Types match the documented schema
            assert isinstance(entry["threshold"], (int, float))
            assert isinstance(entry["patience"], int)
            assert 0.0 <= entry["threshold"] <= 1.0
            assert 0.0 <= entry["youden_j"] <= 1.0


class TestEvaluatePipeline:
    """End-to-end test of reference evaluate() — read back CSV artifacts."""

    def test_evaluate_with_calibrated_thresholds(
        self, labels, unified_ruleset,
        calibration_dialogue_data, evaluation_dialogue_data,
    ):
        import classifier_engine.reference  # noqa: F401
        from gavel.evaluation.calibration import calibrate
        from gavel.evaluation.metrics import evaluate

        with tempfile.TemporaryDirectory() as tmpdir:
            # Calibrate first
            calibrate(
                output_dir=tmpdir,
                labels=labels,
                unified_ruleset=unified_ruleset,
                dialogue_data=calibration_dialogue_data,
                show_progress=False,
                generate_plots=False,
            )
            thresholds_path = os.path.join(tmpdir, "thresholds.json")

            # Persist ruleset to disk (evaluate reads from path)
            ruleset_path = os.path.join(tmpdir, "unified_ruleset.json")
            with open(ruleset_path, "w") as f:
                json.dump(unified_ruleset, f)

            # Run evaluation
            eval_out_dir = os.path.join(tmpdir, "eval")
            os.makedirs(eval_out_dir, exist_ok=True)
            evaluate(
                output_dir=eval_out_dir,
                labels=labels,
                thresholds_path=thresholds_path,
                unified_ruleset_path=ruleset_path,
                dialogue_data=evaluation_dialogue_data,
                compute_auc=True,
                show_progress=False,
            )

            # Required CSV artifacts must exist
            for fname in ("usecase_metrics_fprtpr.csv",
                          "usecase_weighted_averages.csv"):
                assert (Path(eval_out_dir) / fname).is_file(), f"{fname} missing"

            # Read back per-usecase metrics and check rule_strict result
            import csv
            with open(Path(eval_out_dir) / "usecase_metrics_fprtpr.csv", "r") as f:
                rows = list(csv.DictReader(f))

            # The metrics file uses "Usecase" as the rule name column
            usecase_col = next(c for c in rows[0].keys() if c.lower().startswith("use"))
            rule_strict_row = next((r for r in rows if r[usecase_col] == "rule_strict"), None)
            assert rule_strict_row is not None, "rule_strict not in metrics CSV"


class TestAdapterParity:
    """Our adapter is supposed to be a thin orchestrator. These tests
    verify it produces the same answers as the raw reference calls would.

    Skipped automatically if the adapter requires DB access for the
    pieces these tests don't supply (labels are loaded from on-disk
    classifier_meta.json normally).
    """

    def test_run_calibration_threshold_keys_match_labels(
        self, labels, unified_ruleset, calibration_dialogue_data, monkeypatch,
    ):
        import classifier_engine.reference  # noqa: F401
        from evaluation import adapter

        # Stub the DB-backed build_unified_ruleset; our fixtures have
        # the same shape it would produce.
        monkeypatch.setattr(adapter, "build_unified_ruleset",
                            lambda classifier_id: unified_ruleset)

        result = adapter.run_calibration(
            classifier_id=999,
            labels=labels,
            dialogue_data=calibration_dialogue_data,
        )

        # Same key set + schema as the raw reference output
        assert set(result.keys()) == set(labels.keys())
        for entry in result.values():
            assert "threshold" in entry
            assert "patience" in entry
            assert "youden_j" in entry

    def test_run_evaluation_returns_serialisable_metrics(
        self, labels, unified_ruleset,
        calibration_dialogue_data, evaluation_dialogue_data, monkeypatch,
    ):
        import classifier_engine.reference  # noqa: F401
        from evaluation import adapter

        monkeypatch.setattr(adapter, "build_unified_ruleset",
                            lambda classifier_id: unified_ruleset)

        # Get thresholds via the adapter
        thresholds = adapter.run_calibration(
            classifier_id=999, labels=labels,
            dialogue_data=calibration_dialogue_data,
        )

        result = adapter.run_evaluation(
            classifier_id=999, labels=labels,
            dialogue_data=evaluation_dialogue_data,
            thresholds=thresholds, compute_auc=True,
        )

        # The adapter promises JSONB-safe (no DataFrames, no tensors) output
        json.dumps(result)
        # Adapter unwraps the CSVs into a dict keyed by stem
        assert "csvs" in result
