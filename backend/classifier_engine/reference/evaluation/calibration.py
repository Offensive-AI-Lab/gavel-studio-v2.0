"""Calibration utilities for GAVEL.

This module contains functions for calibrating threshold and patience parameters
using Youden's J-statistic on calibration datasets.
"""
from __future__ import annotations
import json
import logging
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)


def update_label_level_stats(
    triggers: torch.Tensor,  # bool/int tensor [L]
    all_required_labels: torch.Tensor,  # bool/int tensor [L]
    supporting_labels: torch.Tensor,  # bool/int tensor [L]
    any_of_conditions: dict[str, list[torch.Tensor]],
    use_case: str,
    labels_statistics: list[dict],
) -> None:
    """Update label-level statistics based on rule-based ground truth.

    Args:
        triggers: Predicted trigger vector
        all_required_labels: Labels that must trigger
        supporting_labels: Labels that are supporting alone
        any_of_conditions: Any-of condition groups
        use_case: Name of the use case
        labels_statistics: List of stat dicts to update
    """
    pred = triggers.bool()
    req = all_required_labels.bool()
    supp = supporting_labels.bool()

    L = pred.numel()

    def idxs_to_mask(idxs: torch.Tensor) -> torch.Tensor:
        m = torch.zeros(L, dtype=torch.bool, device=pred.device)
        if idxs.numel() > 0:
            m[idxs] = True
        return m

    any_of_groups_idx = any_of_conditions.get(use_case, []) or []
    any_of_groups = [idxs_to_mask(g) for g in any_of_groups_idx]

    any_of_union = torch.zeros_like(pred, dtype=torch.bool)
    for g in any_of_groups:
        any_of_union |= g

    if (req & any_of_union).any() or (supp & any_of_union).any():
        raise ValueError("A label appears in both any_of and all_required/supporting.")
    if (req & supp).any():
        raise ValueError("A label is marked both all_required and supporting.")
    if len(any_of_groups) > 1:
        stacked = torch.stack(any_of_groups, dim=0)
        overlaps = stacked.sum(dim=0) > 1
        if overlaps.any():
            raise ValueError("A label appears in multiple any_of groups.")

    allowed = req | supp | any_of_union
    irrelevant = ~allowed

    tp_req = pred & req
    fn_req = (~pred) & req

    tp_supp = pred & supp
    tn_supp = (~pred) & supp

    tp_any = torch.zeros_like(pred)
    tn_any = torch.zeros_like(pred)
    fn_any = torch.zeros_like(pred)
    for g in any_of_groups:
        any_fired = bool((pred & g).any())
        if any_fired:
            tp_any |= pred & g
            tn_any |= (~pred) & g
        else:
            fn_any |= g

    fp_irr = pred & irrelevant
    tn_irr = (~pred) & irrelevant

    tp = tp_req | tp_supp | tp_any
    fp = fp_irr
    tn = tn_supp | tn_any | tn_irr
    fn = fn_req | fn_any

    assert (tp | fp | tn | fn).all(), "Every label must land in a bucket."
    assert not ((tp & fp) | (tp & tn) | (tp & fn) | (fp & tn) | (fp & fn) | (tn & fn)).any(), (
        "Label landed in multiple buckets."
    )

    def _bump(mask: torch.Tensor, field: str) -> None:
        for i in mask.nonzero(as_tuple=True)[0].tolist():
            labels_statistics[i][field] += 1

    _bump(tp, "true_positive")
    _bump(fp, "false_positive")
    _bump(tn, "true_negative")
    _bump(fn, "false_negative")


def plot_calibration_curves(
    df_results: pd.DataFrame,
    id_col: str,
    optimal_params: dict,
    out_dir: str,
    patience_to_plot: int = 1,
    mosaic_max_cols: int = 4,
    suffix: str = "",
) -> None:
    """Plot calibration curves showing TPR, FPR, and Youden's J vs threshold.

    Args:
        df_results: DataFrame with calibration results
        id_col: Column name for identifiers (e.g., "Topic")
        optimal_params: Dictionary of optimal parameters per ID
        out_dir: Output directory for plots
        patience_to_plot: Patience value to plot
        mosaic_max_cols: Maximum columns in mosaic plot
        suffix: Suffix for output filenames
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    for _id, opt in optimal_params.items():
        df_plot = df_results[
            (df_results[id_col] == _id) & (df_results["Patience"] == patience_to_plot)
        ]
        if df_plot.empty:
            continue

        plt.figure(figsize=(12, 7))
        plt.plot(df_plot["Threshold"], df_plot["TPR"], marker="o", label="TPR")
        plt.plot(df_plot["Threshold"], df_plot["FPR"], marker="s", label="FPR")
        plt.plot(df_plot["Threshold"], df_plot["YoudenJ"], marker="x", label="TPR – FPR")

        plt.plot(
            opt["threshold"],
            opt["youden_j"],
            marker="*",
            markersize=14,
            color="red",
            label=f"Optimal J = {opt['youden_j']:.2f}",
        )

        plt.title(f'{id_col}: "{_id}" (Patience = {patience_to_plot})')
        plt.xlabel("Threshold")
        plt.ylabel("Rate / J-Statistic")
        plt.grid(True, linestyle="--", linewidth=0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig(Path(out_dir) / f"calibration_{_id}.png")
        plt.close()

    ids = list(optimal_params.keys())
    if not ids:
        return
    n_plots = len(ids)
    ncols = min(mosaic_max_cols, n_plots)
    nrows = math.ceil(n_plots / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), sharex=True, sharey=True)
    fig.suptitle(
        f"Youden's J vs Threshold – {id_col}s (Patience = {patience_to_plot})",
        fontsize=20,
    )
    axes = axes.flatten()

    for i, _id in enumerate(ids):
        ax = axes[i]
        df_plot = df_results[
            (df_results[id_col] == _id) & (df_results["Patience"] == patience_to_plot)
        ]

        ax.set_title(_id, fontsize=10)
        if df_plot.empty:
            ax.text(0.5, 0.5, "No Data", ha="center", va="center")
            continue

        ax.plot(df_plot["Threshold"], df_plot["YoudenJ"], marker=".", linestyle="-")
        opt = optimal_params[_id]
        ax.plot(opt["threshold"], opt["youden_j"], marker="*", markersize=10, color="red")
        ax.grid(True, linestyle="--", linewidth=0.5)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.text(0.5, 0.04, "Threshold", ha="center", fontsize=16)
    fig.text(0.06, 0.5, "Youden's J (TPR – FPR)", va="center", rotation="vertical", fontsize=16)
    plt.tight_layout(rect=[0.07, 0.05, 0.98, 0.95])
    plt.savefig(Path(out_dir).parent / f"summary_youdens_j_all_{id_col.lower()}s_{suffix}.png")
    plt.close()


def compute_optimal_params(
    df: pd.DataFrame,
    id_col: str,
    out_json: str | None = None,
    verbose: bool = True,
) -> tuple[Dict[str, Dict[str, Any]], pd.DataFrame]:
    """Compute optimal threshold and patience parameters using Youden's J.

    Args:
        df: DataFrame with TPR, FPR, threshold, and patience columns
        id_col: Column name for identifier (e.g., "Topic")
        out_json: Optional path to save results as JSON
        verbose: Whether to print progress

    Returns:
        Tuple of (optimal_params_dict, updated_dataframe_with_youdenJ)
    """
    if "YoudenJ" not in df.columns:
        df = df.copy()
        df["YoudenJ"] = df["TPR"] - df["FPR"]

    optimal: Dict[str, Dict[str, Any]] = {}

    for _id in df[id_col].unique():
        sub = df[df[id_col] == _id]
        if sub.empty:
            continue
        tol = 0.01
        max_j = sub["YoudenJ"].max()
        candidates = sub[sub["YoudenJ"] >= max_j - tol * abs(max_j)]
        best = candidates.loc[candidates["Threshold"].idxmax()]
        optimal[_id] = {
            "threshold": best["Threshold"],
            "patience": int(best["Patience"]),
            "youden_j": best["YoudenJ"],
            "tpr_at_optimal": best["TPR"],
            "fpr_at_optimal": best["FPR"],
        }
        if verbose:
            logger.debug(
                f"Optimal for {id_col.lower()} '{_id}': "
                f"Thr={best['Threshold']:.2f}, Pat={best['Patience']}, "
                f"J={best['YoudenJ']:.4f}"
            )

    if out_json is not None:
        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(out_json, "w") as f:
            json.dump(optimal, f, indent=4)
        if verbose:
            logger.debug(f"Optimal {id_col.lower()} parameters saved to {out_json}")

    return optimal, df


def load_calibration_dialogues(
    logits_root: str,
    labels: Dict[str, int],
    unified_ruleset: dict,
    logger=None,
) -> List[dict]:
    """Load calibration dialogues from logits directory.

    Args:
        logits_root: Root directory containing logits
        labels: Label name to index mapping
        unified_ruleset: Unified ruleset with all_required/supporting labels per use case
        logger: Optional logger

    Returns:
        List of dialogue data dicts with logits, gt_req, gt_supp, etc.
    """
    from gavel.utils.io import iter_dialogue_files
    from gavel.evaluation.metrics import convert_labels_to_tensors

    malicious_use_cases_ruleset = convert_labels_to_tensors(unified_ruleset, labels)
    num_topics = len(labels)

    dialogue_data_cache = []
    for meta_path, npy_path, meta in iter_dialogue_files(logits_root):
        split = meta.get("split", "")
        # Only process calibration splits (usecase_level, CE_level)
        if split not in ["usecase_level", "CE_level"]:
            continue

        gt_uc_name = meta.get("usecase_path", "")
        dialogue_id = meta.get("dialogue_id", meta_path.stem)

        if split == "usecase_level":
            if gt_uc_name not in malicious_use_cases_ruleset:
                if logger:
                    logger.warning(f"Unknown usecase '{gt_uc_name}' — skipping")
                continue
            gt_req = malicious_use_cases_ruleset[gt_uc_name]["all_required_labels"]
            gt_supp = malicious_use_cases_ruleset[gt_uc_name]["supporting_labels"]
        else:
            # CE_level: UC name is a topic label
            idx = labels.get(gt_uc_name, None)
            if idx is None:
                if logger:
                    logger.warning(f"Unknown label '{gt_uc_name}' — skipping")
                continue
            gt_req = torch.zeros(num_topics, dtype=torch.float32)
            gt_req[idx] = 1.0
            gt_supp = torch.zeros(num_topics, dtype=torch.float32)

        logits_np = np.load(npy_path)
        dialogue_logits = torch.from_numpy(logits_np).float()

        dialogue_data_cache.append(
            {
                "logits": dialogue_logits,
                "gt_req": gt_req,
                "gt_supp": gt_supp,
                "gt_uc_name": gt_uc_name,
                "dialogue_name": dialogue_id,
                "split": split,
            }
        )

    if logger:
        logger.info(f"Loaded {len(dialogue_data_cache)} calibration dialogues")

    return dialogue_data_cache


def run_threshold_sweep(
    dialogue_cache: List[dict],
    labels: Dict[str, int],
    any_of_conditions: dict,
    thresholds: np.ndarray = None,
    patience_values: List[int] = None,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Run grid search over thresholds and patience values.

    Args:
        dialogue_cache: List of dialogue data from load_calibration_dialogues
        labels: Label name to index mapping
        any_of_conditions: Any-of condition groups
        thresholds: Array of threshold values to try (default: 0.05 to 1.0 step 0.05)
        patience_values: List of patience values to try (default: [1])
        show_progress: Show progress bar

    Returns:
        DataFrame with columns: Topic, Patience, Threshold, TPR, FPR
    """
    from gavel.evaluation.metrics import compute_triggers

    if thresholds is None:
        thresholds = np.arange(0.05, 1.05, 0.05)
    if patience_values is None:
        patience_values = [1]

    num_topics = len(labels)
    idx_to_label = {v: k for k, v in labels.items()}

    all_results = []
    total_iterations = len(patience_values) * len(thresholds)

    iterator = tqdm(total=total_iterations, desc="Calibrating") if show_progress else None

    for patience in patience_values:
        for threshold in thresholds:
            thr = float(round(threshold, 2))

            topic_stats = [
                {"true_positive": 0, "true_negative": 0, "false_positive": 0, "false_negative": 0}
                for _ in range(num_topics)
            ]

            for data in dialogue_cache:
                triggers = compute_triggers(data["logits"], thresholds=thr, patience_rate=patience)

                if data["split"] == "usecase_level":
                    update_label_level_stats(
                        triggers=triggers,
                        all_required_labels=data["gt_req"],
                        supporting_labels=data["gt_supp"],
                        use_case=data["gt_uc_name"],
                        any_of_conditions=any_of_conditions,
                        labels_statistics=topic_stats,
                    )

            # Compute TPR/FPR for each topic
            for i, stats in enumerate(topic_stats):
                tp, fn = stats["true_positive"], stats["false_negative"]
                fp, tn = stats["false_positive"], stats["true_negative"]
                tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
                all_results.append(
                    {
                        "Topic": idx_to_label[i],
                        "Patience": patience,
                        "Threshold": thr,
                        "TPR": tpr,
                        "FPR": fpr,
                    }
                )

            if iterator:
                iterator.update(1)

    if iterator:
        iterator.close()

    return pd.DataFrame(all_results)


def calibrate(
    output_dir: str,
    labels: Dict[str, int],
    unified_ruleset: dict,
    dialogue_data: Optional[List[Dict]] = None,
    logits_root: Optional[str] = None,
    thresholds: np.ndarray = None,
    patience_values: List[int] = None,
    show_progress: bool = True,
    generate_plots: bool = True,
    logger=None,
) -> Dict[str, Dict[str, Any]]:
    """Run full calibration pipeline.

    Accepts either in-memory dialogue data OR a path to saved logits on disk.
    Exactly one of `dialogue_data` or `logits_root` must be provided.

    Args:
        output_dir: Output directory for results
        labels: Label name to index mapping
        unified_ruleset: Unified ruleset with use case definitions
        dialogue_data: List of dicts from extract_dialogues_in_memory (in-memory mode)
        logits_root: Root directory containing saved logits (disk mode)
        thresholds: Array of threshold values (default: 0.05 to 1.0 step 0.05)
        patience_values: List of patience values (default: [1])
        show_progress: Show progress bar
        generate_plots: Generate calibration plots
        logger: Optional logger

    Returns:
        Dictionary of optimal parameters per topic

    Raises:
        ValueError: If neither or both of dialogue_data/logits_root are provided
    """
    from gavel.evaluation.metrics import convert_labels_to_tensors, load_any_of_conditions

    # Validate inputs
    if dialogue_data is None and logits_root is None:
        raise ValueError("Must provide either dialogue_data or logits_root")
    if dialogue_data is not None and logits_root is not None:
        raise ValueError("Provide only one of dialogue_data or logits_root, not both")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "topic_plots"), exist_ok=True)

    # Load any-of conditions
    any_of_conditions = load_any_of_conditions(unified_ruleset, labels)

    # Build dialogue cache from either source
    if logits_root is not None:
        # Disk-based mode
        if logger:
            logger.info("Loading calibration dialogues from disk...")
        dialogue_cache = load_calibration_dialogues(logits_root, labels, unified_ruleset, logger)
    else:
        # In-memory mode: convert to internal format
        if logger:
            logger.info("Processing in-memory calibration dialogues...")
        malicious_use_cases_ruleset = convert_labels_to_tensors(unified_ruleset, labels)
        num_topics = len(labels)

        dialogue_cache = []
        for dialogue in dialogue_data:
            meta = dialogue["metadata"]
            split = meta.get("split", "")

            # Only process calibration splits
            if split not in ["usecase_level", "CE_level"]:
                continue

            gt_uc_name = meta.get("usecase_path", "")
            dialogue_id = meta.get("dialogue_id", "")

            if split == "usecase_level":
                if gt_uc_name not in malicious_use_cases_ruleset:
                    if logger:
                        logger.warning(f"Unknown usecase '{gt_uc_name}' — skipping")
                    continue
                gt_req = malicious_use_cases_ruleset[gt_uc_name]["all_required_labels"]
                gt_supp = malicious_use_cases_ruleset[gt_uc_name]["supporting_labels"]
            else:
                # CE_level: UC name is a topic label
                idx = labels.get(gt_uc_name, None)
                if idx is None:
                    if logger:
                        logger.warning(f"Unknown label '{gt_uc_name}' — skipping")
                    continue
                gt_req = torch.zeros(num_topics, dtype=torch.float32)
                gt_req[idx] = 1.0
                gt_supp = torch.zeros(num_topics, dtype=torch.float32)

            # Convert logits to tensor if numpy
            logits = dialogue["logits"]
            if isinstance(logits, np.ndarray):
                logits = torch.from_numpy(logits).float()

            dialogue_cache.append(
                {
                    "logits": logits,
                    "gt_req": gt_req,
                    "gt_supp": gt_supp,
                    "gt_uc_name": gt_uc_name,
                    "dialogue_name": dialogue_id,
                    "split": split,
                }
            )

        if logger:
            logger.info(f"Processing {len(dialogue_cache)} calibration dialogues")

    # Run threshold sweep
    if logger:
        logger.info("Running threshold sweep...")
    df_results = run_threshold_sweep(
        dialogue_cache, labels, any_of_conditions, thresholds, patience_values, show_progress
    )

    # Compute optimal parameters
    if logger:
        logger.info("Computing optimal parameters...")
    optimal_params, df_results = compute_optimal_params(
        df=df_results,
        id_col="Topic",
        out_json=os.path.join(output_dir, "thresholds.json"),
        verbose=logger is not None,
    )

    # Generate plots
    if generate_plots:
        if logger:
            logger.info("Generating calibration plots...")
        plot_calibration_curves(
            df_results=df_results,
            id_col="Topic",
            optimal_params=optimal_params,
            out_dir=os.path.join(output_dir, "topic_plots"),
            suffix="",
        )

    if logger:
        logger.info(f"Calibration complete! Results saved to: {output_dir}")

    return optimal_params


# Legacy alias for backward compatibility
def calibrate_from_logits(
    logits_root: str,
    output_dir: str,
    labels: Dict[str, int],
    unified_ruleset: dict,
    thresholds: np.ndarray = None,
    patience_values: List[int] = None,
    show_progress: bool = True,
    generate_plots: bool = True,
    logger=None,
) -> Dict[str, Dict[str, Any]]:
    """Legacy wrapper for calibrate(). Use calibrate() instead."""
    return calibrate(
        output_dir=output_dir,
        labels=labels,
        unified_ruleset=unified_ruleset,
        logits_root=logits_root,
        thresholds=thresholds,
        patience_values=patience_values,
        show_progress=show_progress,
        generate_plots=generate_plots,
        logger=logger,
    )
