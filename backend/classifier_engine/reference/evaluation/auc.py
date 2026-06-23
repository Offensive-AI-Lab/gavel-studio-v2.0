#!/usr/bin/env python3
"""AUC computation utilities for GAVEL evaluation."""

import csv
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from gavel.evaluation.metrics import convert_labels_to_tensors, load_any_of_conditions
from gavel.utils.io import iter_dialogue_files

logger = logging.getLogger(__name__)


def max_over_windows_probs(logits_np: np.ndarray) -> torch.Tensor:
    """Aggregate (W, L) logits -> [L] scores via sigmoid + max-over-windows."""
    logits = torch.from_numpy(logits_np).float()
    probs = torch.sigmoid(logits)
    scores, _ = probs.max(dim=0)
    return scores


def build_usecase_score(
    topic_scores: torch.Tensor,
    all_required_mask: torch.Tensor,
    any_of_groups: List[torch.Tensor],
) -> float:
    """
    score_uc = min( min(all_required), min_g max(group g) )
    """
    # all_required
    if int(all_required_mask.sum()) > 0:
        s_req = float(topic_scores[all_required_mask.bool()].min().item())
    else:
        s_req = 1.0

    # any_of groups
    if any_of_groups and len(any_of_groups) > 0:
        group_scores = []
        for g in any_of_groups:
            g = g.to(topic_scores.device)
            group_scores.append(0.0 if g.numel() == 0 else float(topic_scores[g].max().item()))
        s_any = min(group_scores)
    else:
        s_any = 1.0

    return min(s_req, s_any)


def safe_auc(y_true, y_score) -> Tuple[Optional[float], Optional[float]]:
    y = np.asarray(y_true, dtype=np.int32)
    s = np.asarray(y_score, dtype=np.float32)
    if len(np.unique(y)) < 2:
        return None, None
    try:
        roc = roc_auc_score(y, s)
    except Exception:
        roc = None
    try:
        pr = average_precision_score(y, s)
    except Exception:
        pr = None
    return roc, pr


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main(config_path: str = "config.json"):
    from gavel.config import load_config

    config = load_config(config_path)
    LABELS = config.labels
    LOGITS_ROOT = config.paths.logits_dir
    OUTPUT_DIR = os.path.join(config.paths.base_dir, "results")
    PLOTS_DIR = os.path.join(OUTPUT_DIR, "roc_plots")
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # ── Rules ──
    unified_ruleset_path = f"models/{config.model_name}/Rules.json"
    with open(unified_ruleset_path, "r") as f:
        unified_ruleset = json.load(f)

    # Filter to enabled rulesets only
    enabled_ruleset = {k: v for k, v in unified_ruleset.items() if v.get("enabled", True)}

    any_of_conditions = load_any_of_conditions(enabled_ruleset, LABELS)
    uc_rules = convert_labels_to_tensors(enabled_ruleset, LABELS)
    usecases = list(enabled_ruleset.keys())

    # ── Collect paired (pos vs neg) y_true and scores for each UC ──
    uc_true: Dict[str, List[int]] = {uc: [] for uc in usecases}
    uc_score: Dict[str, List[float]] = {uc: [] for uc in usecases}

    for meta_path, npy_path, meta in iter_dialogue_files(LOGITS_ROOT):
        split = meta.get("split", "")
        if split in ["usecase_level", "CE_level"]:
            continue
        uc_in_meta = meta.get("usecase_path", "")

        # Per-dialogue per-topic scores
        topic_scores = max_over_windows_probs(np.load(npy_path))

        # Score each UC and attach GT where it's paired (pos/neg) for that UC
        for uc in usecases:
            req = uc_rules[uc]["all_required_labels"]
            groups = any_of_conditions.get(uc, []) or []
            s_uc = build_usecase_score(topic_scores, req, groups)

            # Paired labeling (pos if positive & same UC, neg if negative & same UC)
            if split == "positive" and uc_in_meta == uc:
                uc_true[uc].append(1)
                uc_score[uc].append(s_uc)
            elif split == "negative" and uc_in_meta == uc:
                uc_true[uc].append(0)
                uc_score[uc].append(s_uc)
            else:
                # ignore other dialogues for this UC's paired ROC
                pass

    # ── Plot per-UC ROC curves and save AUCs ──
    combined_fig, combined_ax = plt.subplots(figsize=(6, 6))
    combined_ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)

    rows = []
    for uc in usecases:
        y_true = np.array(uc_true[uc], dtype=int)
        y_score = np.array(uc_score[uc], dtype=float)

        if len(y_true) == 0 or len(np.unique(y_true)) < 2:
            # Not enough data to plot; skip
            continue

        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = roc_auc_score(y_true, y_score)
        pr_auc = average_precision_score(y_true, y_score)

        # # Per-UC plot
        # fig, ax = plt.subplots(figsize=(6, 6))
        # ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
        # ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
        # ax.set_xlabel("False Positive Rate")
        # ax.set_ylabel("True Positive Rate")
        # ax.set_title(f"ROC — {uc}")
        # ax.legend(loc="lower right")
        # fig.tight_layout()
        # fig.savefig(Path(PLOTS_DIR) / f"roc_{uc}.png", dpi=150)
        # plt.close(fig)

        # Add to combined plot
        combined_ax.plot(fpr, tpr, label=f"{uc} (AUC {roc_auc:.3f})")

        rows.append(
            {
                "use_case": uc,
                "roc_auc": roc_auc,
                "pr_auc": pr_auc,
                "num_pos": int((y_true == 1).sum()),
                "num_neg": int((y_true == 0).sum()),
            }
        )

    # Save combined ROC
    combined_ax.set_xlabel("False Positive Rate")
    combined_ax.set_ylabel("True Positive Rate")
    combined_ax.set_title("ROC (Paired Pos vs Neg) — All Use Cases")
    combined_ax.legend(loc="lower right", fontsize=8)
    combined_fig.tight_layout()
    combined_fig.savefig(Path(PLOTS_DIR) / "roc_all_usecases.png", dpi=150)
    plt.close(combined_fig)

    # Also write CSV of AUCs
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(Path(OUTPUT_DIR) / "usecase_auc.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["use_case", "roc_auc", "pr_auc", "num_pos", "num_neg"])
        w.writeheader()
        w.writerows(rows)

    # Summary file
    def macro_mean(vals):
        vals = [v for v in vals if isinstance(v, (float, np.floating))]
        return float(np.mean(vals)) if vals else None

    with open(Path(OUTPUT_DIR) / "usecase_auc_summary.txt", "w") as f:
        macro_roc = macro_mean([r["roc_auc"] for r in rows])
        macro_pr = macro_mean([r["pr_auc"] for r in rows])
        f.write(f"Macro ROC-AUC: {macro_roc}\n")
        f.write(f"Macro PR-AUC : {macro_pr}\n")

    logger.info(f"Saved per-UC ROC plots to: {PLOTS_DIR}")
    logger.info(f"Saved per-UC AUC CSV to: {Path(OUTPUT_DIR) / 'usecase_auc.csv'}")


if __name__ == "__main__":
    main()
