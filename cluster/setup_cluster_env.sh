#!/bin/bash
### One-time setup script for the GAVEL conda environment on the BGU cluster.
###
### Run this ONCE from the login node (slurm.bgu.ac.il):
###     chmod +x setup_cluster_env.sh
###     ./setup_cluster_env.sh [ENV_NAME] [RECREATE]
###
###   ENV_NAME   conda env name (default: gavel)
###   RECREATE   "1" => wipe and rebuild the env from scratch
###
### SAFE TO RE-RUN. The env is VERSIONED via a marker file:
###   * if the env exists and is already at this SETUP_VERSION, we do nothing
###     (no reinstall, no package drift/collisions);
###   * if the env spec changed (SETUP_VERSION bumped) — or this is the first
###     run with the versioned script — we REMOVE and rebuild it cleanly;
###   * RECREATE=1 forces a clean rebuild regardless.
### Bump SETUP_VERSION below whenever you change the python version or the
### pip-installed package set, so existing envs get rebuilt instead of layered.

set -eu

ENV_NAME="${1:-gavel}"
RECREATE="${2:-0}"
SETUP_VERSION="3"                                 # bump on any dependency change
MARKER="$HOME/gavel_code/.gavel_env_version"      # survives the code re-sync

echo "Setting up conda environment: $ENV_NAME (setup v$SETUP_VERSION)"

module load anaconda

# `conda activate` doesn't work in a fresh shell unless the user has
# run `conda init` first — without this hook line, the activate call
# errors and every subsequent pip install runs against the wrong
# Python (or fails). `conda shell.bash hook` injects the shell
# functions we need for this script only; it does not modify .bashrc.
eval "$(conda shell.bash hook)"

env_exists() { conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; }
stored_version() { [ -f "$MARKER" ] && cat "$MARKER" 2>/dev/null || echo ""; }

# --- Decide: skip (up to date) / rebuild clean / create fresh ---------------
if env_exists; then
    if [ "$RECREATE" = "1" ]; then
        echo "Recreating '$ENV_NAME' from scratch (RECREATE=1)..."
        conda env remove -n "$ENV_NAME" -y || true
    elif [ "$(stored_version)" = "$SETUP_VERSION" ]; then
        echo "Environment '$ENV_NAME' is already at setup v$SETUP_VERSION — nothing to do."
        echo "(Re-run with RECREATE=1 to force a clean rebuild.)"
        exit 0
    else
        echo "Env spec changed (have: '$(stored_version)', want: v$SETUP_VERSION)."
        echo "Removing the old '$ENV_NAME' and rebuilding cleanly to avoid drift..."
        conda env remove -n "$ENV_NAME" -y || true
    fi
fi

echo "Creating environment '$ENV_NAME'..."
conda create -n "$ENV_NAME" python=3.12 -y

conda activate "$ENV_NAME"

# Sanity check: confirm we're inside the env. If CONDA_DEFAULT_ENV
# isn't what we asked for, the rest of this script would silently
# pip-install into the wrong Python — fail loud instead.
if [[ "${CONDA_DEFAULT_ENV:-}" != "$ENV_NAME" ]]; then
    echo "ERROR: failed to activate $ENV_NAME (got: ${CONDA_DEFAULT_ENV:-<none>})"
    exit 1
fi

echo "Active conda env: $CONDA_DEFAULT_ENV"
echo "Python: $(which python)"

# Install PyTorch with CUDA support (per cluster guide: use pip for pytorch).
# PIN the exact versions: an UNPINNED install resolves to "latest on the cu118
# index", so two env rebuilds on different dates can land on different torch/cuDNN
# binaries — which shifts extraction logits and GRU training numerics and shows up
# as large run-to-run variance. 2.7.1 is the latest cu118+cp312 build (what the
# unpinned line already resolved to), so this freezes the current stack rather than
# changing it. Keep all N variance-comparison runs on this same pinned env.
pip3 install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu118

# Install transformers + sentence-transformers for LLM loading.
# PIN transformers to 4.57.x: the attention-value feature extraction reads the
# LLM's attention weights + KV-cache internals, which transformers 5.x refactored
# — 5.x extracts subtly different features and breaks parity with the reference
# 4.x-trained reference (her RNN scored ~1.0 on 4.x features, ~0.8 on 5.x). 4.57.6
# is in her uv.lock and matches the backend's local transformers.
pip install "transformers==4.57.6" accelerate sentence-transformers

# Other dependencies the training/eval scripts pull in:
#   pytorch-ignite — used by classifier_engine.RNN for trainer/evaluator
#   matplotlib + Pillow — confusion matrix plotting at end of training
#   scipy/scikit-learn — feature extraction utilities
pip install numpy scipy scikit-learn matplotlib Pillow pytorch-ignite

# Stamp the env so future re-runs can skip it (or detect a spec change).
mkdir -p "$(dirname "$MARKER")"
echo "$SETUP_VERSION" > "$MARKER"

echo ""
echo "=========================================="
echo "Environment '$ENV_NAME' is ready (setup v$SETUP_VERSION)."
echo "Test from your shell with:"
echo "    module load anaconda && conda activate $ENV_NAME && python -c 'import torch'"
echo "=========================================="
