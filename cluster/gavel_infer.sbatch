#!/bin/bash
### GAVEL inference job (calibration/evaluation compute). SBATCH params come
### from the command line (GPU type, time limit), like the training job.

#SBATCH --job-name gavel-infer
#SBATCH --output gavel-infer-id-%J.out
#SBATCH --mem=24G

echo "=========================================="
echo "GAVEL Inference Job | SLURM $SLURM_JOBID | $(date)"
echo "JOB_DIR=$JOB_DIR | GPU=$(nvidia-smi -L 2>/dev/null | head -1)"
echo "=========================================="

if [ -z "$JOB_DIR" ] || [ ! -f "$JOB_DIR/job_payload.json" ]; then
    echo "[ERROR] Missing JOB_DIR or job_payload.json"
    exit 1
fi

echo "$SLURM_JOBID" > "$JOB_DIR/slurm_job_id"
JOB_ID=$(python3 -c "import json; print(json.load(open('$JOB_DIR/job_payload.json')).get('job_id','unknown'))" 2>/dev/null || echo "unknown")

# Remove THIS job's HuggingFace cache on any exit (success/error/timeout/scancel)
# so the ~14 GB base model never lingers, even if the backend never collects the
# result. Only the cache subfolder is removed; results stay for download.
_clean_hf() { rm -rf "$JOB_DIR/hf_cache" 2>/dev/null || true; }
trap _clean_hf EXIT
trap 'exit' TERM INT

module load anaconda
source activate ${GAVEL_ENV:-gavel}

python -u "$HOME/gavel_code/compute_jobs/infer_job.py" \
    --job-dir "$JOB_DIR" \
    --device auto

EXIT_CODE=$?

### Callback (optional) ###
if [ -n "$CALLBACK_URL" ]; then
    STATUS=$(python3 -c "import json; print(json.load(open('$JOB_DIR/status.json'))['status'])" 2>/dev/null || echo "unknown")
    ERROR=$(python3 -c "import json; d=json.load(open('$JOB_DIR/status.json')); print(d.get('error',''))" 2>/dev/null || echo "")
    ELAPSED=$(python3 -c "import json; d=json.load(open('$JOB_DIR/status.json')); print(d.get('elapsed_s',0))" 2>/dev/null || echo "0")

    curl -s -X POST "$CALLBACK_URL" \
        -H "Content-Type: application/json" \
        -d "{\"job_id\":\"$JOB_ID\",\"slurm_job_id\":\"$SLURM_JOBID\",\"exit_code\":$EXIT_CODE,\"status\":\"$STATUS\",\"error\":\"$ERROR\",\"elapsed_s\":$ELAPSED}" \
        || echo "[callback] WARNING: callback failed"
fi

exit $EXIT_CODE
