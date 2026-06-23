"""Direct SSH-based cluster job submission from the local backend.

No central server in the training loop. The local backend SSH's directly
to slurm.bgu.ac.il to submit, poll, and download results. The connection
opens for ~5 seconds per operation, then closes.

Requires:
    - VPN active (user can reach slurm.bgu.ac.il)
    - SSH key configured (no password prompt)
    - Conda env on the cluster with all dependencies

Config (backend/.env):
    SLURM_HOST=slurm.bgu.ac.il
    SLURM_USER=avigofek
    SLURM_SSH_KEY=~/.ssh/id_ed25519_cluster
    SLURM_JOBS_DIR=~/gavel_jobs
    SLURM_CODE_DIR=~/gavel_code
    SLURM_CONDA_ENV=gavel
"""
import json
import os
import re
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

_env = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=_env)

def _normalize_key_path(p: str) -> str:
    """Make an SSH key path usable by whatever ssh binary we shell out to.

    cluster/setup_local.sh runs in Git Bash on Windows, where $HOME is an MSYS
    path, so it writes SLURM_SSH_KEY as e.g. '/c/Users/me/.ssh/id_ed25519_cluster'.
    But the backend runs as native Windows Python and shells out to Windows
    OpenSSH (ssh.exe), which can't read '/c/...' and silently falls back to
    password auth ("Identity file ... not accessible" -> Permission denied).
    Convert a leading '/<drive>/' to '<DRIVE>:/' on Windows so the key is found.
    """
    if not p:
        return p
    p = os.path.expanduser(p)
    if os.name == "nt":
        m = re.match(r"^/([a-zA-Z])/(.*)$", p)
        if m:
            p = f"{m.group(1).upper()}:/{m.group(2)}"
    return p


SLURM_HOST = os.getenv("SLURM_HOST", "")
SLURM_USER = os.getenv("SLURM_USER", "")
SLURM_SSH_KEY = _normalize_key_path(os.getenv("SLURM_SSH_KEY", "~/.ssh/id_ed25519_cluster"))
SLURM_JOBS_DIR = os.getenv("SLURM_JOBS_DIR", "~/gavel_jobs")
SLURM_CODE_DIR = os.getenv("SLURM_CODE_DIR", "~/gavel_code")
SLURM_CONDA_ENV = os.getenv("SLURM_CONDA_ENV", "gavel")
SLURM_GPU_TYPE = os.getenv("SLURM_GPU_TYPE", "rtx_6000:1")
# Partition + QoS for every sbatch we submit (training / inference / realtime).
# Defaults match the PUBLIC RTX 6000 launcher (vscode_6000.sh): the shared 'main'
# partition + 'normal' QoS, requesting an rtx_6000 GPU. The "golden" pool
# (partition 'rtx6000' + QoS 'yisroel') is PRIVATE to another group — using it
# fails with "Partition is not public" / "Invalid qos specification". Set the env
# vars to retarget if your account has access to a private pool.
SLURM_PARTITION = os.getenv("SLURM_PARTITION", "main")
SLURM_QOS = os.getenv("SLURM_QOS", "normal")
SLURM_TIME_LIMIT = os.getenv("SLURM_TIME_LIMIT", "0-04:00:00")
SLURM_CALLBACK_URL = os.getenv("SLURM_CALLBACK_URL", "")
# When true (default), each job's HuggingFace cache lives inside its job dir and
# is deleted with it — the cluster never accumulates base-model weights across
# runs (at the cost of re-downloading per job). Set to 0/false to use the shared
# ~/.cache/huggingface instead (faster reuse, but it grows unbounded).
SLURM_EPHEMERAL_HF_CACHE = os.getenv("SLURM_EPHEMERAL_HF_CACHE", "1").lower() not in ("0", "false", "no")

# --- GPU downgrade race -----------------------------------------------------
# We prefer the most powerful GPU (PRIMARY), but a busy cluster can leave a job
# queued behind it for a long time. So: submit the PRIMARY; if it hasn't STARTED
# within DOWNGRADE_AFTER_S, ALSO submit the same work on a weaker-but-more-
# available SECONDARY GPU and race them — whichever reaches RUNNING first wins,
# and the loser is cancelled. Set SLURM_GPU_SECONDARY="" to disable the downgrade
# entirely (then we just queue on the primary as before).
SLURM_GPU_PRIMARY = os.getenv("SLURM_GPU_PRIMARY", SLURM_GPU_TYPE)          # rtx_6000:1
SLURM_GPU_SECONDARY = os.getenv("SLURM_GPU_SECONDARY", "rtx_4090:1")        # fallback tier
SLURM_GPU_DOWNGRADE_AFTER_S = int(os.getenv("SLURM_GPU_DOWNGRADE_AFTER_S", "180"))  # 3 min

# Job states that mean "this job has begun running on a GPU" (won the race) vs
# "this job died before starting" (drop it from the race).
_RACE_STARTED = {"running", "completed"}
_RACE_BAD = {"failed", "oom", "timeout", "cancelled"}


def is_enabled() -> bool:
    """True if cluster SSH is configured."""
    return bool(SLURM_HOST and SLURM_USER)


# Shared HF-token resolver — moved to utils/model_hf_token.py (not cluster-specific).
from utils.model_hf_token import resolve_model_hf_token as _resolve_model_hf_token

def ping(timeout: int = 20, attempts: int = 3) -> bool:
    """Quick reachability probe: a trivial SSH round-trip, with retries.

    Used before the heavy upload so an unreachable/overloaded cluster fails in
    ~`timeout`s instead of hanging ~2 min on the SCP banner exchange before we
    fall back to local. The retries ride out a TRANSIENT blip — a brief VPN
    reconnect or a momentarily slow login node — that would otherwise wrongly
    force a slow local run.

    On failure it LOGS the real SSH error (stripped of the harmless post-quantum
    warning) so a fallback is never silent again — the cause shows in the log.
    Returns True only if a round-trip succeeds.
    """
    last_err = None
    for i in range(attempts):
        out, err, rc = _ssh("echo ok", timeout=timeout)
        if rc == 0 and out.strip() == "ok":
            return True
        # Drop the harmless post-quantum SSH warning so the real reason is visible.
        real = "\n".join(
            ln for ln in (err or "").splitlines()
            if "post-quantum" not in ln and "store now" not in ln
            and "openssh.com/pq" not in ln and not ln.strip().startswith("**")
        ).strip()
        last_err = real or f"rc={rc}, stdout={out!r}"
        print(f"[cluster] reachability probe {i + 1}/{attempts} failed: {last_err[:200]}")
        if i < attempts - 1:
            time.sleep(2)
    print(f"[cluster] cluster unreachable after {attempts} probes — last error: {last_err[:200]}")
    return False


def _ssh(cmd: str, timeout: int = 30) -> tuple:
    """Run a command on the cluster via SSH. Returns (stdout, stderr, exit_code)."""
    args = [
        "ssh",
        "-o", "StrictHostKeyChecking=no",
        "-o", "ConnectTimeout=10",
        "-o", "BatchMode=yes",
        "-i", SLURM_SSH_KEY,
        f"{SLURM_USER}@{SLURM_HOST}",
        cmd,
    ]
    try:
        r = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        return r.stdout.strip(), r.stderr.strip(), r.returncode
    except subprocess.TimeoutExpired:
        return "", "SSH timed out", 1
    except FileNotFoundError:
        return "", "ssh not found", 1
    except Exception as e:
        return "", str(e), 1


# Files larger than this are uploaded in CHUNKS (see _scp_to_chunked). scp has no
# resume, so a 95MB .pth over the BGU VPN — which drops every ~2min — restarts from
# zero on every blip and never completes. Splitting into small chunks means a drop
# costs only the current ~8MB chunk (seconds), and already-sent chunks are skipped.
_SCP_CHUNK_THRESHOLD = 16 * 1024 * 1024   # 16 MB — above this, chunk it
_SCP_CHUNK_SIZE = 8 * 1024 * 1024         # 8 MB per chunk


def _scp_to_direct(local_path: str, remote_path: str, retries: int = 3, per_attempt_timeout: int = 120):
    """One file -> one scp, retrying with a FRESH connection on a transient stall.

    Hardened for the BGU VPN / login-node gateway:
      * ConnectionAttempts=3 — ssh re-dials the TCP connect itself (round-robin
        DNS across several login nodes; some are momentarily unreachable).
      * ServerAlive/TCPKeepAlive — keep a mid-transfer pipe alive across a stall.
      * ConnectTimeout + backoff — let a rate-limited gateway recover before retry."""
    args = [
        "scp",
        "-o", "StrictHostKeyChecking=no",
        "-o", "ConnectTimeout=15",
        "-o", "ConnectionAttempts=3",
        "-o", "ServerAliveInterval=15",
        "-o", "ServerAliveCountMax=8",
        "-o", "TCPKeepAlive=yes",
        "-i", SLURM_SSH_KEY,
        local_path,
        f"{SLURM_USER}@{SLURM_HOST}:{remote_path}",
    ]
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            r = subprocess.run(args, capture_output=True, text=True, timeout=per_attempt_timeout)
            if r.returncode == 0:
                return
            last_err = (r.stderr or "").strip() or f"scp exit code {r.returncode}"
        except subprocess.TimeoutExpired:
            last_err = f"timed out after {per_attempt_timeout}s"
        except Exception as e:
            last_err = str(e)
        print(f"[cluster] scp upload attempt {attempt}/{retries} failed "
              f"({local_path} -> {remote_path}): {last_err}")
        if attempt < retries:
            time.sleep(min(8 * attempt, 30))
    raise RuntimeError(f"SCP upload failed after {retries} attempts: {last_err}")


def _file_sha256(path: str) -> str:
    """Streaming SHA-256 of a local file (1MB reads) — content identity for the
    chunked-upload skip/resume so a byte-count match never masquerades as 'same'."""
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _scp_to_chunked(local_path: str, remote_path: str,
                    chunk_size: int = _SCP_CHUNK_SIZE, retries_per_chunk: int = 15):
    """Resilient large-file upload over a flaky link: split the file locally, scp
    each small chunk (resuming any already fully uploaded), then reassemble +
    verify on the cluster. A VPN drop costs only the current chunk, so a big .pth
    that plain scp could never push in one shot completes across many short windows.

    Content-addressed: the resume/skip keys off a SHA-256 of the local file, NOT
    its byte count. So a retry of the SAME file resumes its chunks, but a DIFFERENT
    file uploaded to the same path (e.g. a retrained .pth of the identical 95MB
    size) is never mistaken for 'already there' and never mixes stale chunks in."""
    size = os.path.getsize(local_path)
    digest = _file_sha256(local_path)
    slash = remote_path.rfind("/")
    remote_dir = remote_path[:slash] if slash > 0 else "."
    base = remote_path[slash + 1:] if slash >= 0 else remote_path
    # parts dir + completion marker namespaced by content hash (16 hex is plenty).
    parts_dir = f"{remote_dir}/.{base}.{digest[:16]}.parts"
    marker = f"{remote_path}.sha256"

    # Already uploaded with IDENTICAL content? (marker hash matches) -> skip.
    out, _, _ = _ssh(f"cat {marker} 2>/dev/null || true", timeout=15)
    if (out or "").strip() == digest:
        return

    _ssh(f"mkdir -p {parts_dir}", timeout=20)
    # Which chunks already landed (resume): name -> byte size.
    existing = {}
    out, _, _ = _ssh(
        f'cd {parts_dir} && for f in part_*; do [ -e "$f" ] && echo "$f $(stat -c %s "$f")"; done',
        timeout=20,
    )
    for line in (out or "").splitlines():
        cols = line.split()
        if len(cols) == 2 and cols[1].isdigit():
            existing[cols[0]] = int(cols[1])

    idx = sent = 0
    with open(local_path, "rb") as f:
        while True:
            data = f.read(chunk_size)
            if not data:
                break
            name = f"part_{idx:05d}"
            if existing.get(name) == len(data):
                idx += 1
                continue                      # resume: this chunk already uploaded
            with tempfile.NamedTemporaryFile(delete=False) as tf:
                tf.write(data)
                tmp = tf.name
            try:
                _scp_to_direct(tmp, f"{parts_dir}/{name}",
                               retries=retries_per_chunk, per_attempt_timeout=90)
                sent += 1
            finally:
                os.unlink(tmp)
            idx += 1
    print(f"[cluster] chunked upload {base}: {idx} chunks ({sent} sent, {idx - sent} resumed) -> reassembling")

    # Reassemble in order (zero-padded names glob-sort correctly), verify, clean up.
    out, err, rc = _ssh(f"cat {parts_dir}/part_* > {remote_path} && stat -c %s {remote_path}", timeout=240)
    got = (out or "").strip().splitlines()[-1] if (out or "").strip() else ""
    if rc != 0 or got != str(size):
        raise RuntimeError(f"chunked upload reassembly mismatch for {base} "
                           f"(got {got!r}, want {size}): {(err or '')[:200]}")
    # Stamp the content marker (so an identical re-upload skips) and drop the parts.
    _ssh(f"printf %s {digest} > {marker}; rm -rf {parts_dir}", timeout=20)


def _scp_to(local_path: str, remote_path: str, retries: int = 3, per_attempt_timeout: int = 120):
    """Upload a file to the cluster, resilient to flaky-VPN drops.

    Small files: a single scp (with connection retries). LARGE files (> ~16MB,
    e.g. the 95MB trained .pth): chunked, because scp has no resume and a big file
    over the BGU VPN otherwise restarts from zero on every blip and never finishes.
    Same signature for both, so every caller (inference / realtime / training)
    gets the resilient path automatically for big files."""
    try:
        size = os.path.getsize(local_path)
    except OSError:
        size = 0
    if size > _SCP_CHUNK_THRESHOLD:
        return _scp_to_chunked(local_path, remote_path)
    return _scp_to_direct(local_path, remote_path, retries=retries, per_attempt_timeout=per_attempt_timeout)


def _scp_from(remote_path: str, local_path: str, timeout_s: int = 300):
    """Download a file from the cluster via SSH cat.

    Plain binary pipe: ssh cat → local file. No gzip (dense .pth
    tensors don't compress). ~50s for 26MB over the university VPN."""
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    tmp_fd = tempfile.NamedTemporaryFile(
        dir=os.path.dirname(local_path), suffix=".tmp", delete=False,
    )
    tmp_path = tmp_fd.name
    tmp_fd.close()

    args = [
        "ssh",
        "-o", "StrictHostKeyChecking=no",
        "-o", "ConnectTimeout=10",
        "-o", "BatchMode=yes",
        "-o", "ServerAliveInterval=15",
        "-o", "ServerAliveCountMax=8",
        "-i", SLURM_SSH_KEY,
        f"{SLURM_USER}@{SLURM_HOST}",
        f"cat {remote_path}",
    ]
    try:
        with open(tmp_path, "wb") as f:
            r = subprocess.run(args, stdout=f, stderr=subprocess.PIPE, timeout=timeout_s)
        if r.returncode != 0:
            raise RuntimeError(f"SSH failed: {r.stderr.decode().strip()}")
        if os.path.getsize(tmp_path) == 0:
            raise RuntimeError(f"Empty file")
        if os.path.exists(local_path):
            os.remove(local_path)
        os.rename(tmp_path, local_path)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"Download timed out for {remote_path}")
    except RuntimeError:
        raise
    except Exception as e:
        raise RuntimeError(f"Download failed: {e}")
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass


def _resolve_remote_home() -> str:
    """Get the absolute home directory on the cluster (resolves ~ properly)."""
    out, _, rc = _ssh("echo $HOME", timeout=10)
    if rc != 0 or not out:
        return f"/home/{SLURM_USER}"
    return out.strip()


def submit_training_job(
    classifier_id: int,
    user_id: int,
    model_hf_path: str,
    labels: Dict[str, int],
    training_config: dict,
    dataset_files: Dict[str, Any],
    calibration_entries: Optional[list] = None,
) -> dict:
    """Upload training data to the cluster and submit a SLURM job.

    Returns {"job_id": "...", "slurm_job_id": "...", "mode": "cluster"}.
    Raises RuntimeError on any failure."""
    # Resolve ~ to absolute path (~ doesn't expand in sbatch --export)
    home = _resolve_remote_home()
    jobs_dir = SLURM_JOBS_DIR.replace("~", home)
    code_dir = SLURM_CODE_DIR.replace("~", home)

    job_id = str(uuid.uuid4())
    remote_job_dir = f"{jobs_dir}/{job_id}"

    # 1. Create job directory
    out, err, rc = _ssh(f"mkdir -p {remote_job_dir}/dataset")
    if rc != 0:
        raise RuntimeError(f"Failed to create job dir: {err}")

    # 2. Upload job_payload.json
    payload = {
        "model_hf_path": model_hf_path,
        "labels": labels,
        "config": training_config,
        "classifier_id": classifier_id,
        "user_id": user_id,
        "job_id": job_id,
        # Shipped so the cluster can pull a gated base model (no DB there).
        "hf_token": _resolve_model_hf_token(model_hf_path),
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(payload, f, indent=2)
        tmp_payload = f.name
    try:
        _scp_to(tmp_payload, f"{remote_job_dir}/job_payload.json")
    finally:
        os.unlink(tmp_payload)

    # 3. Upload dataset files
    with tempfile.TemporaryDirectory() as tmpdir:
        for filename, conversations in dataset_files.items():
            filepath = os.path.join(tmpdir, filename)
            with open(filepath, "w") as f:
                json.dump(conversations, f, indent=2)
            _scp_to(filepath, f"{remote_job_dir}/dataset/{filename}")
        # Calibration dialogues for candidate selection (optional — the job
        # falls back to a single fit when the file is absent).
        if calibration_entries:
            calib_path = os.path.join(tmpdir, "calibration_input.json")
            with open(calib_path, "w") as f:
                json.dump(calibration_entries, f, indent=2)
            _scp_to(calib_path, f"{remote_job_dir}/calibration_input.json")

    # 4. Submit sbatch
    gpu_type = training_config.get("gpu_type", SLURM_GPU_TYPE)
    time_limit = training_config.get("time_limit", SLURM_TIME_LIMIT)
    callback = SLURM_CALLBACK_URL

    # Per-job HF cache lives INSIDE the job dir, so the (~14 GB) base model the
    # job pulls from HF is removed together with the dir by cleanup_job — the
    # cluster never accumulates model weights across runs. Trade-off: each job
    # re-downloads its base model (no cross-job cache); set SLURM_EPHEMERAL_HF_CACHE
    # to 0 to keep the shared ~/.cache instead (faster, but it grows unbounded).
    export_vars = f"JOB_DIR={remote_job_dir},GAVEL_ENV={SLURM_CONDA_ENV}"
    if SLURM_EPHEMERAL_HF_CACHE:
        # Point ALL HuggingFace caches into the job dir, not just HF_HOME. Older
        # huggingface_hub / transformers read HF_HUB_CACHE / TRANSFORMERS_CACHE
        # directly and ignore HF_HOME, so they would still write the multi-GB base
        # model into the shared ~/.cache/huggingface (the home-quota bloat). With
        # all three set, every byte lands in the job dir and cleanup_job removes it.
        _hf = f"{remote_job_dir}/hf_cache"
        export_vars += f",HF_HOME={_hf},HF_HUB_CACHE={_hf}/hub,TRANSFORMERS_CACHE={_hf}/hub"
    if callback:
        export_vars += f",CALLBACK_URL={callback}"

    sbatch_cmd = (
        f"cd {code_dir}/cluster && "
        f"sbatch "
        # Write the SLURM log INTO the job dir so cleanup_job removes it with
        # everything else — otherwise gavel-train-id-*.out files pile up in the
        # code dir forever, one per run.
        f"--output={remote_job_dir}/slurm-%J.out "
        f"--partition={SLURM_PARTITION} "
        f"--qos={SLURM_QOS} "
        f"--gpus={gpu_type} "
        f"--time={time_limit} "
        f"--export=ALL,{export_vars} "
        f"gavel_train.sbatch"
    )
    out, err, rc = _ssh(sbatch_cmd)
    if rc != 0:
        raise RuntimeError(f"sbatch failed: {err}")

    match = re.search(r"Submitted batch job (\d+)", out)
    if not match:
        raise RuntimeError(f"Could not parse SLURM job ID: {out}")

    slurm_job_id = match.group(1)
    # Record the SLURM id in the job dir at SUBMIT time (the sbatch script writes
    # it too, but only once the job RUNS). Having it now lets the orphan sweep
    # recognize a still-QUEUED job and never delete its dir. Best-effort.
    _ssh(f"echo {slurm_job_id} > {remote_job_dir}/slurm_job_id", timeout=15)
    return {
        "job_id": job_id,
        "slurm_job_id": slurm_job_id,
        "remote_job_dir": remote_job_dir,
        "mode": "cluster",
        "gpu": gpu_type,
    }


# ---------------------------------------------------------------------------
# Inference jobs (the GPU-heavy part of calibration / evaluation).
#
# Only the inference (target-LLM + RNN windowed logits) is offloaded; the light
# metric/threshold math runs back on the backend. The on-cluster job uses the
# SAME shared core (classifier_engine.inference_core) as the local path, so the
# logits are identical regardless of where they were computed.
# ---------------------------------------------------------------------------

def submit_inference_job(
    classifier_id: int,
    model_hf_path: str,
    classifier_meta: dict,
    dialogues: List[dict],
    rnn_path: str,
    *,
    max_length: Optional[int] = None,   # None = keep the WHOLE dialogue (reference-parity); the on-cluster core's per-dialogue OOM fallback truncates only what won't fit
    window_stride: int = 0,   # 0 => non-overlapping (stride = window_size)
    gpu_type: str = None,
    time_limit: str = None,
) -> dict:
    """Upload the inference inputs + trained RNN and submit a SLURM job.

    Returns {"job_id", "slurm_job_id", "remote_job_dir", "mode": "cluster"}.
    Raises RuntimeError on any failure."""
    home = _resolve_remote_home()
    jobs_dir = SLURM_JOBS_DIR.replace("~", home)
    code_dir = SLURM_CODE_DIR.replace("~", home)

    job_id = str(uuid.uuid4())
    remote_job_dir = f"{jobs_dir}/{job_id}"

    out, err, rc = _ssh(f"mkdir -p {remote_job_dir}")
    if rc != 0:
        raise RuntimeError(f"Failed to create job dir: {err}")

    # job_payload.json — model id, guardrail geometry/meta, the dialogues to
    # infer (conversation + metadata), and the windowing params.
    payload = {
        "model_hf_path": model_hf_path,
        "classifier_meta": classifier_meta,
        "dialogues": dialogues,
        "max_length": max_length,
        "window_stride": window_stride,
        "classifier_id": classifier_id,
        "job_id": job_id,
        # Shipped so the cluster can pull a gated base model (no DB there).
        "hf_token": _resolve_model_hf_token(model_hf_path),
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(payload, f)
        tmp_payload = f.name
    try:
        _scp_to(tmp_payload, f"{remote_job_dir}/job_payload.json")
    finally:
        os.unlink(tmp_payload)

    # The trained RNN weights (the LLM is pulled from HF on the cluster).
    if not os.path.isfile(rnn_path):
        raise RuntimeError(f"Trained RNN not found locally: {rnn_path}")
    _scp_to(rnn_path, f"{remote_job_dir}/trained_rnn.pth")

    gpu = gpu_type or SLURM_GPU_TYPE
    tlimit = time_limit or SLURM_TIME_LIMIT
    # Per-job HF cache inside the job dir → removed by cleanup_job (see
    # submit_training_job). Inference re-pulls the base model each run instead of
    # leaving 14 GB on the cluster.
    export_vars = f"JOB_DIR={remote_job_dir},GAVEL_ENV={SLURM_CONDA_ENV}"
    if SLURM_EPHEMERAL_HF_CACHE:
        # All HF caches into the job dir (see submit_training_job for why HF_HOME
        # alone isn't enough on older huggingface_hub / transformers versions).
        _hf = f"{remote_job_dir}/hf_cache"
        export_vars += f",HF_HOME={_hf},HF_HUB_CACHE={_hf}/hub,TRANSFORMERS_CACHE={_hf}/hub"
    if SLURM_CALLBACK_URL:
        export_vars += f",CALLBACK_URL={SLURM_CALLBACK_URL}"

    sbatch_cmd = (
        f"cd {code_dir}/cluster && "
        f"sbatch --partition={SLURM_PARTITION} --qos={SLURM_QOS} "
        # Log into the job dir so it's removed with cleanup_job (no pile-up).
        f"--output={remote_job_dir}/slurm-%J.out "
        f"--gpus={gpu} --time={tlimit} "
        f"--export=ALL,{export_vars} "
        f"gavel_infer.sbatch"
    )
    out, err, rc = _ssh(sbatch_cmd)
    if rc != 0:
        raise RuntimeError(f"sbatch failed: {err}")
    match = re.search(r"Submitted batch job (\d+)", out)
    if not match:
        raise RuntimeError(f"Could not parse SLURM job ID: {out}")

    slurm_job_id = match.group(1)
    # Record the SLURM id at submit time so the orphan sweep can tell a queued
    # job's dir apart from an abandoned one (see submit_training_job).
    _ssh(f"echo {slurm_job_id} > {remote_job_dir}/slurm_job_id", timeout=15)
    return {
        "job_id": job_id,
        "slurm_job_id": slurm_job_id,
        "remote_job_dir": remote_job_dir,
        "mode": "cluster",
        "gpu": gpu,
    }


def download_inference_results(remote_job_dir: str, retries: int = 3,
                               per_attempt_timeout: int = 180) -> dict:
    """Download + parse logits.json from a finished inference job.

    The SSH 'cat' download occasionally stalls at 0 bytes on the university
    VPN (a transient connection hang, not a size problem). Rather than wait out
    one long 300s timeout and then fall all the way back to local, retry with a
    FRESH connection a few times — a stalled pipe almost always clears on the
    next attempt. Each attempt fails fast (`per_attempt_timeout`), so the whole
    retry budget is shorter than the old single long wait.
    """
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                local = os.path.join(tmpdir, "logits.json")
                _scp_from(f"{remote_job_dir}/logits.json", local, timeout_s=per_attempt_timeout)
                with open(local) as f:
                    return json.load(f)
        except Exception as e:
            last_err = e
            print(f"[cluster] logits download attempt {attempt}/{retries} failed: {e}")
            if attempt < retries:
                time.sleep(min(5 * attempt, 15))
    raise RuntimeError(f"logits download failed after {retries} attempts: {last_err}")


def run_inference_blocking(
    classifier_id: int,
    model_hf_path: str,
    classifier_meta: dict,
    dialogues: List[dict],
    rnn_path: str,
    *,
    max_length: Optional[int] = None,   # None = keep the WHOLE dialogue (reference-parity); the on-cluster core's per-dialogue OOM fallback truncates only what won't fit
    window_stride: int = 0,   # 0 => non-overlapping (stride = window_size)
    on_submit=None,
    on_phase=None,
    poll_interval: int = 10,
    timeout_s: int = 3 * 3600,
) -> List[dict]:
    """Submit an inference job, BLOCK-poll until it finishes, download the
    logits, and return them as [{"logits": np.ndarray, "metadata": {...}}] —
    the exact shape run_inference_on_dialogues returns locally.

    Crash-safety: `on_submit(info)` is called right after submission with
    {"slurm_job_id", "remote_job_dir", "job_id"} so the caller can persist it
    and cancel/clean the cluster job if the backend dies mid-run. On ANY failure
    (job failed / timeout / download error) this raises, so the caller treats it
    as 'never ran' (and may fall back to local). The remote job dir is cleaned
    up on terminal states.

    `on_phase(text)` is an optional progress callback fired (deduped) as the
    job moves pending -> running -> downloading, so the caller can surface the
    live stage to the user.
    """
    import numpy as np

    _last_phase = {"v": None}

    def _phase(text):
        if on_phase and text != _last_phase["v"]:
            _last_phase["v"] = text
            try:
                on_phase(text)
            except Exception:
                pass

    info = submit_inference_job(
        classifier_id, model_hf_path, classifier_meta, dialogues, rnn_path,
        max_length=max_length, window_stride=window_stride,
        gpu_type=SLURM_GPU_PRIMARY,
    )
    if on_submit:
        try:
            on_submit(info)
        except Exception:
            pass

    # GPU downgrade race: if the powerful primary GPU stays queued, also try a
    # weaker one and keep whichever starts first (re-tracking the winner so a
    # cancel-on-delete still targets the live job).
    def _resubmit_inference(gpu):
        return submit_inference_job(
            classifier_id, model_hf_path, classifier_meta, dialogues, rnn_path,
            max_length=max_length, window_stride=window_stride, gpu_type=gpu,
        )
    winner = run_gpu_race(info, _resubmit_inference, on_phase=_phase)
    if winner is not info:
        info = winner
        if on_submit:
            try:
                on_submit(info)
            except Exception:
                pass

    slurm_job_id = info["slurm_job_id"]
    remote_job_dir = info["remote_job_dir"]
    deadline = time.time() + timeout_s
    _phase("Queued on the compute cluster…")

    try:
        while True:
            st = get_job_status(slurm_job_id).get("status", "unknown")
            if st == "completed":
                break
            if st in ("failed", "oom", "timeout", "cancelled"):
                res = get_job_result(remote_job_dir) or {}
                raise RuntimeError(f"cluster inference {st}: {res.get('error', '')[:300]}")
            if st == "pending":
                _phase("Waiting for a cluster GPU…")
            elif st == "running":
                _phase("Running on the cluster GPU…")
            if st == "unknown":
                # Job purged from SLURM — fall back to status.json on disk.
                res = get_job_result(remote_job_dir)
                if res and res.get("status") == "success":
                    break
                if res and res.get("status") == "failed":
                    raise RuntimeError(f"cluster inference failed: {res.get('error', '')[:300]}")
                # else: still finishing — keep polling
            if time.time() > deadline:
                cancel_job(slurm_job_id)
                raise TimeoutError(f"cluster inference exceeded {timeout_s}s")
            time.sleep(poll_interval)

        _phase("Downloading results from the cluster…")
        data = download_inference_results(remote_job_dir)
        return [
            {"logits": np.asarray(r["logits"], dtype=np.float32), "metadata": r.get("metadata", {})}
            for r in (data.get("results") or [])
        ]
    finally:
        # Terminal cleanup either way (success or raise) — we never reuse a dir.
        try:
            cleanup_job(remote_job_dir)
        except Exception:
            pass


def get_job_status(slurm_job_id: str) -> dict:
    """Check SLURM job status via sacct. Returns {"status": "...", "error": "..."}."""
    out, err, rc = _ssh(
        f"sacct -j {slurm_job_id} --format=State --noheader --parsable2 | head -1",
        timeout=15,
    )
    if rc != 0 or not out:
        # Try squeue for very new jobs
        out, _, _ = _ssh(f"squeue -j {slurm_job_id} --format=%T --noheader | head -1", timeout=15)

    state = out.strip().upper()
    status_map = {
        "PENDING": "pending",
        "RUNNING": "running",
        "COMPLETED": "completed",
        "FAILED": "failed",
        "TIMEOUT": "timeout",
        "OUT_OF_MEMORY": "oom",
        "OUT_OF_ME+": "oom",
        "CANCELLED": "cancelled",
        "CANCELLED+": "cancelled",
    }
    status = status_map.get(state, "unknown")

    # If completed/failed, try to read status.json for details
    error = None
    elapsed = None
    if status in ("completed", "failed", "oom", "timeout"):
        # We need the job_id to find the dir — caller should pass it
        pass

    return {"status": status, "slurm_state": state}


def get_job_result(remote_job_dir: str) -> Optional[dict]:
    """Read status.json from the cluster job directory."""
    out, err, rc = _ssh(f"cat {remote_job_dir}/status.json", timeout=15)
    if rc != 0:
        return None
    try:
        return json.loads(out)
    except json.JSONDecodeError:
        return None


REQUIRED_FILES = ("trained_rnn.pth", "classifier_meta.json")
OPTIONAL_FILES = ("training_log.json",)


def download_results(remote_job_dir: str, local_dir: str) -> bool:
    """Download trained_rnn.pth + classifier_meta.json from the cluster.

    Returns True only if ALL required files downloaded successfully.
    Both files are needed for the guardrail to load — a partial download
    leaves the guardrail unusable, so we keep the cluster files for retry
    and don't mark success unless every required file made it."""
    os.makedirs(local_dir, exist_ok=True)
    for filename in REQUIRED_FILES:
        try:
            _scp_from(f"{remote_job_dir}/{filename}", os.path.join(local_dir, filename))
            print(f"[cluster] Downloaded {filename}")
        except RuntimeError as e:
            print(f"[cluster] Download {filename} failed: {e}")
            return False
    for filename in OPTIONAL_FILES:
        try:
            _scp_from(f"{remote_job_dir}/{filename}", os.path.join(local_dir, filename))
        except RuntimeError as e:
            print(f"[cluster] Optional file {filename} not downloaded: {e}")
    return True


def cleanup_job(remote_job_dir: str):
    """Remove the job directory from the cluster."""
    _ssh(f"rm -rf {remote_job_dir}", timeout=15)


def sweep_orphan_job_dirs(max_age_hours: int = 6) -> dict:
    """Reclaim abandoned job dirs left under SLURM_JOBS_DIR.

    The normal path cleans up after itself: a finished job removes its own HF
    cache (sbatch trap) and the backend calls cleanup_job on every terminal
    state. The hole this closes is the backend dying BEFORE it observes a
    terminal state — then the job's small result files (the big HF cache is
    already gone via the trap) sit in the dir forever.

    Safety on a SHARED cluster account is the priority — a dir is removed only
    when ALL of these hold:
      * it is OLDER than `max_age_hours` (a backend mid-collection is never
        touched), AND
      * it carries a recorded SLURM id (submit-time or run-time), AND
      * that SLURM id is NOT in the active set (running or pending).
    Dirs without a recorded id are LEFT ALONE (could be a just-submitted job).
    If the active-job lookup itself fails, the whole sweep aborts and removes
    nothing — a transient squeue error must never delete a live job's dir.

    Best-effort, side-effect-only. Returns a summary dict for logging.
    """
    out = {"checked": 0, "removed": 0, "kept": 0, "errors": []}
    if not (SLURM_HOST and SLURM_USER):
        return out  # cluster not configured — nothing to sweep

    try:
        home = _resolve_remote_home()
    except Exception as e:
        out["errors"].append(f"resolve home failed: {e}")
        return out
    jobs_dir = SLURM_JOBS_DIR.replace("~", home)

    # 1. Candidate dirs: direct children older than the age threshold.
    age_min = max(1, int(max_age_hours * 60))
    listing, err, rc = _ssh(
        f"find {jobs_dir} -mindepth 1 -maxdepth 1 -type d -mmin +{age_min} 2>/dev/null",
        timeout=30,
    )
    if rc != 0:
        out["errors"].append(f"list job dirs failed: {err or 'rc!=0'}")
        return out
    dirs = [d for d in listing.splitlines() if d.strip()]
    if not dirs:
        return out

    # 2. Snapshot active SLURM ids (running + pending — squeue shows both by
    #    default). If this lookup fails, ABORT: never delete on an unknown set.
    sq_out, sq_err, sq_rc = _ssh("squeue -h -u $USER -o %i", timeout=30)
    if sq_rc != 0:
        out["errors"].append(f"squeue failed; aborting sweep: {sq_err or 'rc!=0'}")
        return out
    active_ids = {x.strip() for x in sq_out.split() if x.strip()}

    # 3. Remove each old dir whose recorded SLURM job is no longer active.
    for d in dirs:
        out["checked"] += 1
        jid, _, jrc = _ssh(f"cat {d}/slurm_job_id 2>/dev/null", timeout=15)
        jid = jid.strip()
        if not jid or jid in active_ids:
            # No id recorded yet (possibly just submitted) or still active → keep.
            out["kept"] += 1
            continue
        _, rm_err, rm_rc = _ssh(f"rm -rf {d}", timeout=20)
        if rm_rc == 0:
            out["removed"] += 1
        else:
            out["errors"].append(f"rm {d}: {rm_err or 'rc!=0'}")
    return out


def cancel_job(slurm_job_id: str) -> bool:
    """Cancel a SLURM job (running or queued) via scancel. Best-effort —
    returns True if scancel succeeded, False otherwise. Used by the
    guardrail/model delete paths so a user removing a guardrail
    mid-training doesn't leave a job grinding through GPU time."""
    if not slurm_job_id:
        return False
    _, err, rc = _ssh(f"scancel {slurm_job_id}", timeout=15)
    if rc != 0:
        print(f"[cluster] scancel {slurm_job_id} failed: {err}")
        return False
    print(f"[cluster] Cancelled SLURM job {slurm_job_id}")
    return True


def run_gpu_race(
    primary_info: dict,
    resubmit_fn,
    *,
    on_switch=None,
    on_phase=None,
    secondary_gpu: Optional[str] = None,
    wait_s: Optional[int] = None,
    poll_interval: int = 10,
    _status_fn=None,
    _sleep=None,
) -> dict:
    """Race an already-submitted PRIMARY (powerful) GPU job against a later,
    weaker SECONDARY so a long queue on the primary doesn't block the user.

    Flow:
      1. Wait up to `wait_s` for the primary to START on its own. If it does
         (or it dies), return the primary — no downgrade.
      2. Still queued → submit the SAME work on the secondary GPU via
         `resubmit_fn(secondary_gpu)` and race: whichever reaches RUNNING first
         WINS; the loser is `scancel`'d and its dir cleaned.

    `on_switch(winner)` (optional) is invoked the moment a NON-primary winner is
    chosen, BEFORE the loser is cancelled — so an async caller (training poller /
    realtime session map) can atomically re-point its tracking to the winner and
    never observe the soon-to-be-cancelled loser as a "failed" job.

    Returns the WINNER's info dict. Pure-ish: `_status_fn`/`_sleep` are injectable
    so the race is unit-testable without a real cluster."""
    status = _status_fn or (lambda jid: get_job_status(str(jid)).get("status", "unknown"))
    sleep = _sleep or time.sleep
    secondary_gpu = SLURM_GPU_SECONDARY if secondary_gpu is None else secondary_gpu
    wait_s = SLURM_GPU_DOWNGRADE_AFTER_S if wait_s is None else wait_s

    a = primary_info
    a_id = str(a.get("slurm_job_id"))
    primary_gpu = a.get("gpu") or SLURM_GPU_PRIMARY

    # Downgrade disabled / pointless (same GPU) → behave exactly as before.
    if not secondary_gpu or secondary_gpu == primary_gpu:
        return a

    def _safe(cb, *args):
        if cb:
            try:
                cb(*args)
            except Exception:
                pass

    # Phase 1: give the primary the full wait window to start by itself.
    waited = 0
    while waited < wait_s:
        sa = status(a_id)
        if sa in _RACE_STARTED or sa in _RACE_BAD:
            return a  # started (or died) on its own → no downgrade needed
        sleep(poll_interval)
        waited += poll_interval

    # Phase 2: still queued on the primary → submit the secondary and race.
    _safe(on_phase, f"Still waiting for {primary_gpu} after {max(1, wait_s // 60)} min — "
                    f"also requested {secondary_gpu}; first to start wins…")
    try:
        b = resubmit_fn(secondary_gpu)
    except Exception as e:
        print(f"[cluster] GPU downgrade submit failed ({e}); staying on the primary GPU")
        return a
    b_id = str(b.get("slurm_job_id"))

    def _drop(info: dict, cancel: bool):
        """Cancel (if still live) + clean the loser's cluster dir."""
        try:
            if cancel and info.get("slurm_job_id"):
                cancel_job(str(info["slurm_job_id"]))
        except Exception:
            pass
        try:
            if info.get("remote_job_dir"):
                cleanup_job(info["remote_job_dir"])
        except Exception:
            pass

    def _finish(winner: dict, loser: Optional[dict], cancel_loser: bool) -> dict:
        if winner is not a:
            _safe(on_switch, winner)   # re-point tracking BEFORE the loser is cancelled
        if loser is not None:
            _drop(loser, cancel_loser)
        return winner

    while True:
        sa = status(a_id)
        sb = status(b_id)
        if sa in _RACE_STARTED:
            return _finish(a, b, cancel_loser=True)     # primary wins (incl. both-started → prefer stronger)
        if sb in _RACE_STARTED:
            return _finish(b, a, cancel_loser=True)     # secondary wins
        a_bad = sa in _RACE_BAD
        b_bad = sb in _RACE_BAD
        if a_bad and b_bad:
            return _finish(a, None, cancel_loser=False)  # both died before starting → report via primary
        if a_bad:
            return _finish(b, a, cancel_loser=False)     # primary died on its own → commit to secondary
        if b_bad:
            return _finish(a, b, cancel_loser=False)     # secondary died → stay on primary
        sleep(poll_interval)


def get_training_log(remote_job_dir: str) -> Optional[list]:
    """Read the partial training_log.json for live progress."""
    out, _, rc = _ssh(f"cat {remote_job_dir}/training_log.json", timeout=15)
    if rc != 0:
        return None
    try:
        return json.loads(out)
    except json.JSONDecodeError:
        return None
