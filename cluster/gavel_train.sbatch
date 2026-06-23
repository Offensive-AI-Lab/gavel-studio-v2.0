#!/bin/bash
### GAVEL classifier training job for the BGU SLURM cluster.
### All SBATCH params (partition, GPU, time, QoS) are passed on the
### command line by the agent — NOT in #SBATCH directives here.
### This avoids the bash-variable-in-SBATCH parsing bug.

#SBATCH --job-name gavel-train
#SBATCH --output gavel-train-id-%J.out
#SBATCH --mem=24G

echo "=========================================="
echo "GAVEL Training Job"
echo "=========================================="
echo "SLURM_JOBID        = $SLURM_JOBID"
echo "SLURM_JOB_NODELIST = $SLURM_JOB_NODELIST"
echo "JOB_DIR            = $JOB_DIR"
echo "CALLBACK_URL       = $CALLBACK_URL"
echo "Date               = $(date)"
echo "GPU                = $(nvidia-smi -L 2>/dev/null || echo 'no GPU')"
echo "=========================================="

### Validate ###
if [ -z "$JOB_DIR" ]; then
    echo "[ERROR] JOB_DIR not set."
    exit 1
fi
if [ ! -f "$JOB_DIR/job_payload.json" ]; then
    echo "[ERROR] $JOB_DIR/job_payload.json not found."
    exit 1
fi

echo "$SLURM_JOBID" > "$JOB_DIR/slurm_job_id"
JOB_ID=$(python3 -c "import json; print(json.load(open('$JOB_DIR/job_payload.json'))['job_id'])" 2>/dev/null || echo "unknown")

# Always remove THIS job's HuggingFace cache when the job ends — on success,
# error, OOM, timeout, or scancel. The cache holds the ~14 GB base model and
# lives only inside $JOB_DIR, so it never lingers on the cluster even if the
# backend crashes or never collects the result. Only the cache subfolder is
# removed; the trained model + result files stay for the backend to download
# (cleanup_job removes the rest on terminal states).
_clean_hf() { rm -rf "$JOB_DIR/hf_cache" 2>/dev/null || true; }
trap _clean_hf EXIT
trap 'exit' TERM INT   # SLURM sends SIGTERM on timeout/scancel -> run the EXIT trap

### Activate conda ###
module load anaconda
source activate ${GAVEL_ENV:-gavel}

### Run training ###
# Use $HOME for absolute path — the relative ../gavel_code doesn't
# resolve correctly from the job directory.
python -u "$HOME/gavel_code/compute_jobs/train_job.py" \
    --job-dir "$JOB_DIR" \
    --device auto

EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Job finished: exit_code=$EXIT_CODE at $(date)"
echo "=========================================="

### Callback ###
if [ -n "$CALLBACK_URL" ]; then
    echo "[callback] Notifying $CALLBACK_URL ..."
    STATUS=$(python3 -c "import json; print(json.load(open('$JOB_DIR/status.json'))['status'])" 2>/dev/null || echo "unknown")
    ERROR=$(python3 -c "import json; d=json.load(open('$JOB_DIR/status.json')); print(d.get('error',''))" 2>/dev/null || echo "")
    ELAPSED=$(python3 -c "import json; d=json.load(open('$JOB_DIR/status.json')); print(d.get('elapsed_s',0))" 2>/dev/null || echo "0")

    curl -s -X POST "$CALLBACK_URL" \
        -H "Content-Type: application/json" \
        -d "{\"job_id\":\"$JOB_ID\",\"slurm_job_id\":\"$SLURM_JOBID\",\"exit_code\":$EXIT_CODE,\"status\":\"$STATUS\",\"error\":\"$ERROR\",\"elapsed_s\":$ELAPSED}" \
        || echo "[callback] WARNING: callback failed"
    echo "[callback] Done."
fi

exit $EXIT_CODE
