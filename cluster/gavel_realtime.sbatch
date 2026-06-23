#!/bin/bash
### GAVEL WARM realtime inference server. Stays alive serving realtime requests
### from the backend until: a stop sentinel, an idle timeout (set in realtime_job.py),
### or the SLURM wall limit below. SBATCH params (GPU type, time) come from the
### command line, like the train/infer jobs.

#SBATCH --job-name gavel-realtime
#SBATCH --output gavel-realtime-id-%J.out
#SBATCH --mem=24G

echo "=========================================="
echo "GAVEL Warm Realtime Job | SLURM $SLURM_JOBID | $(date)"
echo "JOB_DIR=$JOB_DIR | GPU=$(nvidia-smi -L 2>/dev/null | head -1)"
echo "=========================================="

if [ -z "$JOB_DIR" ] || [ ! -f "$JOB_DIR/job_payload.json" ]; then
    echo "[ERROR] Missing JOB_DIR or job_payload.json"
    exit 1
fi

echo "$SLURM_JOBID" > "$JOB_DIR/slurm_job_id"

# Remove THIS job's HuggingFace cache on any exit (success/idle/scancel/timeout)
# so the ~14 GB base model never lingers. Only the cache subfolder is removed.
_clean_hf() { rm -rf "$JOB_DIR/hf_cache" 2>/dev/null || true; }
trap _clean_hf EXIT
trap 'exit' TERM INT

module load anaconda
source activate ${GAVEL_ENV:-gavel}

python -u "$HOME/gavel_code/compute_jobs/realtime_job.py" \
    --job-dir "$JOB_DIR" \
    --device auto

EXIT_CODE=$?

### Callback (optional) ###
if [ -n "$CALLBACK_URL" ]; then
    STATUS=$(python3 -c "import json; print(json.load(open('$JOB_DIR/status.json'))['status'])" 2>/dev/null || echo "unknown")
    curl -s -X POST "$CALLBACK_URL" \
        -H "Content-Type: application/json" \
        -d "{\"slurm_job_id\":\"$SLURM_JOBID\",\"exit_code\":$EXIT_CODE,\"status\":\"$STATUS\",\"kind\":\"realtime\"}" \
        || echo "[callback] WARNING: callback failed"
fi

exit $EXIT_CODE
