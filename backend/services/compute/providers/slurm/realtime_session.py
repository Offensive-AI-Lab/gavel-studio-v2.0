"""Warm realtime cluster-session manager.

Drives the lifecycle of a long-lived `gavel_realtime.sbatch` job (loaded once,
serves requests) so realtime works on ANY client PC — the heavy LLM forward runs
on the cluster GPU, not the laptop.

Lifecycle (all over the shared filesystem via cluster_direct's SSH helpers):
  start_session  -> mkdir session dir, upload payload + RNN, sbatch the warm job
  session_status -> queued | loading | ready | dead | stopped   (for startup polling)
  send_request   -> write requests/<id>.json, poll responses/<id>.json
  keepalive      -> touch the cluster keepalive file (resets the job's idle clock)
  end_session    -> stop sentinel + scancel + cleanup

Robust cleanup (so a GPU never leaks across "the many ways to leave realtime"):
  * explicit end_session on a clean exit,
  * a background sweep that ends sessions whose client stopped pinging (covers
    tab-close / browser-crash / network-loss / force-quit),
  * the job ITSELF self-terminates on an idle timeout / wall limit (covers a dead
    backend) — see compute_jobs/realtime_job.py.

One warm session per guardrail (the registry is keyed by classifier_id).
"""
import json
import os
import re
import tempfile
import threading
import time
import uuid
from typing import Optional

from . import cluster_direct as cd

# How long after the last client ping we consider a session abandoned and scancel
# it. Shorter than the job's own idle timeout so the backend reclaims promptly;
# the job idle-timeout is the backstop if the backend itself dies.
STALE_SESSION_S = int(os.getenv("SLURM_REALTIME_STALE_S", "90"))
SWEEP_INTERVAL_S = 45
REALTIME_TIME_LIMIT = os.getenv("SLURM_REALTIME_TIME_LIMIT", cd.SLURM_TIME_LIMIT)
# Boot-time orphan recovery blanket-scancels every running gavel-realtime job,
# which is correct for a SINGLE-tenant SLURM account (one backend) but would kill
# a concurrent backend's live session on a SHARED account. Default on; set to 0
# on a shared account and rely on the job's own idle timeout instead.
RECOVER_ORPHANS = os.getenv("SLURM_REALTIME_RECOVER_ORPHANS", "1").lower() not in ("0", "false", "no")
# Consecutive SLURM-'unknown' status polls tolerated before declaring a session
# dead — rides out transient SSH/sacct/VPN blips so one hiccup can't orphan a GPU.
UNKNOWN_TOLERANCE = 4

_SESSIONS: dict = {}          # classifier_id -> session dict
_LOCK = threading.RLock()
_sweeper_started = False


# ---------------------------------------------------------------------------
# Registry + background sweep
# ---------------------------------------------------------------------------

def _ensure_sweeper():
    global _sweeper_started
    with _LOCK:
        if _sweeper_started:
            return
        _sweeper_started = True
    t = threading.Thread(target=_sweep_loop, name="rt-session-sweeper", daemon=True)
    t.start()


def _sweep_loop():
    while True:
        time.sleep(SWEEP_INTERVAL_S)
        try:
            _sweep_once()
        except Exception as e:  # noqa: BLE001
            print(f"[realtime] sweep error: {e}")


def _sweep_once():
    now = time.time()
    stale = []
    with _LOCK:
        for cid, s in list(_SESSIONS.items()):
            if now - s.get("last_seen", 0) > STALE_SESSION_S:
                stale.append((cid, s))
    for cid, s in stale:
        print(f"[realtime] session for classifier {cid} idle "
              f"{now - s.get('last_seen', 0):.0f}s — ending (client gone)")
        _end(cid, s)


def get_session(classifier_id: int) -> Optional[dict]:
    with _LOCK:
        return _SESSIONS.get(classifier_id)


def touch_session(classifier_id: int) -> bool:
    """Mark the session as alive (client ping) + reset the job's idle clock.
    Returns False if there's no session for this guardrail."""
    with _LOCK:
        s = _SESSIONS.get(classifier_id)
        if not s:
            return False
        s["last_seen"] = time.time()
        remote = s.get("remote_job_dir")
    if not remote:
        return True  # still starting — no remote keepalive file to touch yet
    # Log (don't silently swallow) a failed remote touch: the local last_seen
    # above keeps the backend sweep from ending the session, but if the remote
    # touch keeps failing the JOB's own idle clock won't advance and it will
    # eventually self-exit — surfacing the degradation in the log helps diagnose.
    _, err, rc = cd._ssh(f"touch {remote}/keepalive", timeout=15)
    if rc != 0:
        print(f"[realtime] keepalive touch failed for classifier {classifier_id}: {(err or '')[:120]}")
    return True


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

def _provision_job(classifier_id: int, model_hf_path: str, classifier_meta: dict,
                   rnn_path: str, gpu: str, session_id: str, *,
                   time_limit: str = None) -> tuple:
    """Create a session dir, upload the payload + trained RNN, and sbatch a warm
    realtime job on `gpu`. Returns (remote_dir, slurm_job_id).

    Self-cleaning: on ANY failure it scancels/cleans whatever it created and
    re-raises, so a half-provisioned job never leaks (the caller can't clean a
    dir it never learned the path of). Used for BOTH the primary submit and the
    GPU-race secondary submit."""
    home = cd._resolve_remote_home()
    jobs_dir = cd.SLURM_JOBS_DIR.replace("~", home)
    code_dir = cd.SLURM_CODE_DIR.replace("~", home)
    remote = f"{jobs_dir}/{session_id}"
    slurm_job_id = None
    try:
        out, err, rc = cd._ssh(f"mkdir -p {remote}/requests {remote}/responses")
        if rc != 0:
            raise RuntimeError(f"Failed to create session dir: {err}")

        payload = {
            "model_hf_path": model_hf_path, "classifier_meta": classifier_meta,
            "classifier_id": classifier_id, "session_id": session_id, "kind": "realtime",
            # Shipped so the cluster can pull a gated base model (no DB there).
            "hf_token": cd._resolve_model_hf_token(model_hf_path),
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(payload, f)
            tmp = f.name
        try:
            cd._scp_to(tmp, f"{remote}/job_payload.json")
        finally:
            os.unlink(tmp)

        if not os.path.isfile(rnn_path):
            raise RuntimeError(f"Trained RNN not found locally: {rnn_path}")
        cd._scp_to(rnn_path, f"{remote}/trained_rnn.pth")
        cd._ssh(f"touch {remote}/keepalive", timeout=15)   # start the idle clock fresh

        gpu_eff = gpu or cd.SLURM_GPU_TYPE
        tlimit = time_limit or REALTIME_TIME_LIMIT
        export_vars = f"JOB_DIR={remote},GAVEL_ENV={cd.SLURM_CONDA_ENV}"
        if cd.SLURM_EPHEMERAL_HF_CACHE:
            _hf = f"{remote}/hf_cache"
            export_vars += f",HF_HOME={_hf},HF_HUB_CACHE={_hf}/hub,TRANSFORMERS_CACHE={_hf}/hub"
        if cd.SLURM_CALLBACK_URL:
            export_vars += f",CALLBACK_URL={cd.SLURM_CALLBACK_URL}"

        sbatch_cmd = (
            f"cd {code_dir}/cluster && sbatch --partition={cd.SLURM_PARTITION} --qos={cd.SLURM_QOS} "
            f"--output={remote}/slurm-%J.out --gpus={gpu_eff} --time={tlimit} "
            f"--export=ALL,{export_vars} gavel_realtime.sbatch"
        )
        out, err, rc = cd._ssh(sbatch_cmd)
        if rc != 0:
            raise RuntimeError(f"sbatch failed: {err}")
        m = re.search(r"Submitted batch job (\d+)", out)
        if not m:
            raise RuntimeError(f"Could not parse SLURM job ID: {out}")
        slurm_job_id = m.group(1)
        cd._ssh(f"echo {slurm_job_id} > {remote}/slurm_job_id", timeout=15)
        return remote, slurm_job_id
    except Exception:
        if slurm_job_id:
            try:
                cd.cancel_job(slurm_job_id)
            except Exception:
                pass
        try:
            cd.cleanup_job(remote)
        except Exception:
            pass
        raise


def _maybe_start_gpu_race(classifier_id: int, session_id: str, model_hf_path: str,
                          classifier_meta: dict, rnn_path: str, primary_job_id: str,
                          primary_remote: str, time_limit: str = None) -> None:
    """If a secondary GPU is configured, race the just-submitted primary realtime
    job against a weaker GPU in the background (see cluster_direct.run_gpu_race).
    The winner is swapped into the session registry atomically BEFORE the loser is
    cancelled, so session_status never sees the loser's cancellation as a death."""
    secondary = cd.SLURM_GPU_SECONDARY
    primary_gpu = cd.SLURM_GPU_PRIMARY
    if not secondary or secondary == primary_gpu:
        return

    def _resub(gpu):
        sid = str(uuid.uuid4())
        remote, jid = _provision_job(classifier_id, model_hf_path, classifier_meta,
                                     rnn_path, gpu, sid, time_limit=time_limit)
        return {"slurm_job_id": jid, "remote_job_dir": remote, "session_id": sid, "gpu": gpu}

    def _switch(winner):
        with _LOCK:
            cur = _SESSIONS.get(classifier_id)
            if cur and cur.get("session_id") == session_id and not cur.get("ending"):
                cur.update({"slurm_job_id": winner["slurm_job_id"],
                            "remote_job_dir": winner["remote_job_dir"],
                            "session_id": winner["session_id"],
                            "last_seen": time.time()})
                return
        # Our session was ended/superseded mid-race → don't resurrect it; tear the
        # winner down so the GPU it just claimed is released.
        try:
            cd.cancel_job(winner["slurm_job_id"])
        except Exception:
            pass
        try:
            cd.cleanup_job(winner["remote_job_dir"])
        except Exception:
            pass

    primary = {"slurm_job_id": primary_job_id, "remote_job_dir": primary_remote,
               "session_id": session_id, "gpu": primary_gpu}

    def _run():
        try:
            cd.run_gpu_race(primary, _resub, on_switch=_switch)
        except Exception as e:
            print(f"[realtime] GPU race failed for classifier {classifier_id}: {e}")

    threading.Thread(target=_run, daemon=True, name=f"gpu-race-rt-{classifier_id}").start()


def start_session(classifier_id: int, model_hf_path: str, classifier_meta: dict,
                  rnn_path: str, *, gpu_type: str = None, time_limit: str = None) -> dict:
    """Submit a warm realtime job and register the session. Returns immediately
    after submit (the job still has to load the model — poll session_status).

    Reuse-or-claim is ATOMIC: a placeholder is inserted UNDER THE LOCK before the
    slow remote submit, so two concurrent starts for the same guardrail (React
    StrictMode double-mount, two tabs, Restart) can never both launch a GPU job —
    the second caller finds the placeholder and reuses it."""
    _ensure_sweeper()

    session_id = str(uuid.uuid4())
    with _LOCK:
        existing = _SESSIONS.get(classifier_id)
        if existing and not existing.get("ending"):
            existing["last_seen"] = time.time()
            return {"reused": True,
                    **{k: existing.get(k) for k in ("session_id", "slurm_job_id", "remote_job_dir")}}
        # Claim the slot with a placeholder so a concurrent start reuses it.
        _SESSIONS[classifier_id] = {
            "session_id": session_id, "slurm_job_id": None, "remote_job_dir": None,
            "classifier_id": classifier_id, "last_seen": time.time(), "starting": True,
        }

    slurm_job_id = None
    remote = None
    try:
        # Submit the PRIMARY (powerful) GPU. A background race may later add a
        # weaker GPU if this one stays queued — see _maybe_start_gpu_race below.
        remote, slurm_job_id = _provision_job(
            classifier_id, model_hf_path, classifier_meta, rnn_path,
            gpu_type or cd.SLURM_GPU_PRIMARY, session_id, time_limit=time_limit,
        )
    except Exception:
        # Roll back the placeholder + best-effort kill anything we created.
        with _LOCK:
            cur = _SESSIONS.get(classifier_id)
            if cur is not None and cur.get("session_id") == session_id:
                _SESSIONS.pop(classifier_id, None)
        if slurm_job_id:
            try:
                cd.cancel_job(slurm_job_id)
            except Exception:
                pass
        if remote:
            try:
                cd.cleanup_job(remote)
            except Exception:
                pass
        raise

    # Fill the placeholder with the real ids — UNLESS we were superseded / ended
    # while submitting (then tear down what we just made so it doesn't leak).
    with _LOCK:
        cur = _SESSIONS.get(classifier_id)
        superseded = cur is None or cur.get("session_id") != session_id
        if not superseded:
            cur.update({"slurm_job_id": slurm_job_id, "remote_job_dir": remote,
                        "last_seen": time.time(), "starting": False})
    if superseded:
        try:
            cd.cancel_job(slurm_job_id)
        except Exception:
            pass
        try:
            cd.cleanup_job(remote)
        except Exception:
            pass
        raise RuntimeError("realtime session was superseded during startup")

    # Kick off the GPU downgrade race in the background (no-op if no secondary GPU
    # is configured). start_session still returns immediately on the primary.
    _maybe_start_gpu_race(classifier_id, session_id, model_hf_path, classifier_meta,
                          rnn_path, slurm_job_id, remote, time_limit)

    return {"reused": False, "session_id": session_id,
            "slurm_job_id": slurm_job_id, "remote_job_dir": remote}


def session_status(classifier_id: int) -> dict:
    """Coarse session state for startup polling + crash detection.

    CRITICAL: a transient SSH/sacct/VPN blip makes get_job_status() return
    'unknown'. We must NOT treat that as terminal — it would evict a healthy
    running job from the registry and orphan the GPU. We ride out a few
    consecutive 'unknown's, and when a session truly is gone we _end() it
    (which scancels) rather than a bare _drop()."""
    s = get_session(classifier_id)
    if not s:
        return {"state": "none"}
    s["last_seen"] = time.time()
    # Placeholder (mid-submit, no SLURM id yet) — still coming up.
    if s.get("starting") or not s.get("slurm_job_id"):
        return {"state": "starting"}
    slurm_job_id = s["slurm_job_id"]
    remote = s["remote_job_dir"]

    st = cd.get_job_status(slurm_job_id).get("status", "unknown")
    if st in ("failed", "timeout", "cancelled", "oom"):
        res = cd.get_job_result(remote) or {}
        _end(classifier_id, s)
        return {"state": "dead", "error": res.get("error") or f"cluster job {st}"}
    if st == "pending":
        s["unknown_polls"] = 0
        return {"state": "queued"}
    if st == "unknown":
        # Transient — keep the session, report the last known state. Only after
        # UNKNOWN_TOLERANCE consecutive unknowns do we conclude it's gone (and
        # then scancel, in case it IS somehow still running).
        n = s.get("unknown_polls", 0) + 1
        s["unknown_polls"] = n
        if n < UNKNOWN_TOLERANCE:
            return {"state": s.get("last_state", "loading")}
        _end(classifier_id, s)
        return {"state": "dead", "error": "cluster job is unreachable"}
    s["unknown_polls"] = 0

    out, _, _ = cd._ssh(
        f"cat {remote}/status.json 2>/dev/null; echo '__SEP__'; cat {remote}/heartbeat 2>/dev/null",
        timeout=15,
    )
    status_part, _, _ = out.partition("__SEP__")
    job_status = None
    try:
        job_status = (json.loads(status_part.strip()) or {}).get("status")
    except Exception:
        pass

    if job_status == "failed":
        res = cd.get_job_result(remote) or {}
        _end(classifier_id, s)
        return {"state": "dead", "error": res.get("error") or "realtime job failed"}
    if job_status == "ready":
        s["last_state"] = "ready"
        return {"state": "ready"}
    if st == "completed" or job_status == "stopped":
        # SLURM finished it (idle-exit / scancel / preemption) — reclaim the dir.
        _end(classifier_id, s)
        return {"state": "stopped"}
    s["last_state"] = "loading"
    return {"state": "loading"}


def send_request(classifier_id: int, payload: dict, *, timeout_s: int = 240,
                 poll: float = 1.0) -> dict:
    """Write a request into the warm job's queue and block for its response.
    Raises on timeout or if the job is no longer running."""
    s = get_session(classifier_id)
    if not s:
        raise RuntimeError("No realtime session — start one first.")
    if s.get("starting") or not s.get("remote_job_dir"):
        raise RuntimeError("Realtime session is still starting up.")
    s["last_seen"] = time.time()
    slurm_job_id = s["slurm_job_id"]
    remote = s["remote_job_dir"]

    rid = uuid.uuid4().hex
    body = {**payload, "id": rid}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(body, f)
        tmp = f.name
    try:
        # Upload to a non-.json name, then atomically rename so the job (which
        # globs requests/*.json) never reads a half-written request.
        cd._scp_to(tmp, f"{remote}/requests/{rid}.part")
    finally:
        os.unlink(tmp)
    # Check the rename rc: if the mv blips, the request would sit as a .part the
    # job never picks up and we'd block the FULL timeout. Fail fast instead.
    _, mv_err, mv_rc = cd._ssh(f"mv {remote}/requests/{rid}.part {remote}/requests/{rid}.json", timeout=20)
    if mv_rc != 0:
        cd._ssh(f"rm -f {remote}/requests/{rid}.part", timeout=10)
        raise RuntimeError(f"failed to enqueue realtime request: {(mv_err or 'mv failed')[:160]}")

    resp = f"{remote}/responses/{rid}.json"
    deadline = time.time() + timeout_s
    last_liveness = 0.0
    gone_strikes = 0
    while time.time() < deadline:
        # Keep the session fresh DURING a long request (a live generation can run
        # well past STALE_SESSION_S) so the stale-session sweep can't scancel the
        # job out from under an in-flight request.
        s["last_seen"] = time.time()

        out, _, rc = cd._ssh(f"cat {resp} 2>/dev/null", timeout=25)
        if rc == 0 and out.strip():
            try:
                data = json.loads(out)
            except json.JSONDecodeError:
                time.sleep(poll)
                continue
            cd._ssh(f"rm -f {resp}", timeout=10)
            s["last_seen"] = time.time()
            if not data.get("ok"):
                raise RuntimeError(f"realtime job error: {data.get('error', 'unknown')}")
            return data.get("result") or {}
        # Periodically confirm the job is still alive so a dead session fails fast
        # instead of hanging the whole timeout — but tolerate a transient blip
        # (a single 'unknown'/'completed' from sacct lag must not false-fail).
        if time.time() - last_liveness > 12:
            last_liveness = time.time()
            jst = cd.get_job_status(slurm_job_id).get("status")
            if jst in ("failed", "timeout", "cancelled", "oom"):
                _end(classifier_id, s)
                raise RuntimeError(f"realtime job stopped ({jst})")
            if jst in ("completed", "unknown"):
                gone_strikes += 1
                if gone_strikes >= 2:
                    _end(classifier_id, s)
                    raise RuntimeError(f"realtime job stopped ({jst})")
            else:
                gone_strikes = 0
        time.sleep(poll)
    raise TimeoutError(f"realtime request timed out after {timeout_s}s")


def end_session(classifier_id: int) -> dict:
    s = get_session(classifier_id)
    if not s:
        return {"ended": False}
    _end(classifier_id, s)
    return {"ended": True}


# ---------------------------------------------------------------------------
# internals
# ---------------------------------------------------------------------------

def end_all_sessions() -> int:
    """Tear down EVERY live session. Called from the backend's graceful-shutdown
    hook so closing the backend (Ctrl+C / SIGTERM) scancels its warm jobs at once
    instead of leaving them to idle-time-out."""
    with _LOCK:
        items = list(_SESSIONS.items())
    for cid, s in items:
        try:
            _end(cid, s)
        except Exception:
            pass
    return len(items)


def recover_orphans() -> dict:
    """Scancel warm realtime jobs left running by a PRIOR backend that died while
    a session was active. Safe to call on startup: a fresh backend has NO live
    sessions, so any running `gavel-realtime` job is necessarily an orphan. This
    is the recovery path for a HARD backend kill (where the shutdown hook never
    ran); a clean stop is already handled by end_all_sessions(), and a still-alive
    job also self-exits on its idle timeout. Gated by SLURM_REALTIME_RECOVER_ORPHANS
    (default on) since the blanket scancel is only safe on a single-tenant account."""
    out = {"found": 0, "cancelled": 0}
    if not RECOVER_ORPHANS:
        return out
    try:
        sq_out, _, sq_rc = cd._ssh(
            "squeue -h -u $USER --name=gavel-realtime -o %i", timeout=30,
        )
    except Exception:
        return out
    if sq_rc != 0:
        return out
    ids = [x.strip() for x in sq_out.split() if x.strip()]
    out["found"] = len(ids)
    for jid in ids:
        try:
            if cd.cancel_job(jid):
                out["cancelled"] += 1
        except Exception:
            pass
    return out


def _drop(classifier_id: int):
    with _LOCK:
        _SESSIONS.pop(classifier_id, None)


def _end(classifier_id: int, s: dict):
    # Mark ending FIRST so a concurrent start_session won't reuse a torn-down
    # session (its reuse check skips entries with ending=True).
    s["ending"] = True
    remote = s.get("remote_job_dir")
    slurm_job_id = s.get("slurm_job_id")
    # Ask the job to exit cleanly, then scancel + remove the dir (each best-effort).
    # Guard on `remote`/`slurm_job_id` — a placeholder (mid-submit) has neither.
    if remote:
        try:
            cd._ssh(f"touch {remote}/stop", timeout=15)
        except Exception:
            pass
    if slurm_job_id:
        try:
            cd.cancel_job(str(slurm_job_id))
        except Exception:
            pass
    if remote:
        try:
            cd.cleanup_job(remote)
        except Exception:
            pass
    _drop(classifier_id)
