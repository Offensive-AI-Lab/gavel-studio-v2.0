"""SlurmProvider — the university SSH+SLURM cluster, wrapping the existing
`cluster_direct` (and, later, `realtime_session`) helpers. Behavior is identical
to the pre-abstraction "cluster" branch; this is purely an adapter.
"""
from typing import Callable, List, Optional

from ...base import (
    Accelerator, Capabilities, ComputeError, ComputeProvider, InferenceSpec,
    JobState, RealtimeSession, RealtimeSpec, TrainingJob, TrainingSpec, TrainingStatus,
)


class SlurmProvider(ComputeProvider):
    name = "slurm"

    def capabilities(self) -> Capabilities:
        return Capabilities(
            name=self.name, accelerator=Accelerator.REMOTE, is_local=False,
            supports_training=True, supports_inference=True, supports_realtime=True,
            max_realtime_sessions=1, detail="SLURM cluster",
        )

    def is_available(self) -> bool:
        from . import cluster_direct as cd
        try:
            return bool(cd.is_enabled()) and bool(cd.ping(timeout=20))
        except Exception:
            return False

    # --- inference (calibration + evaluation) ---
    def run_inference(self, spec: InferenceSpec, on_phase: Optional[Callable] = None,
                      on_submit: Optional[Callable] = None) -> List[dict]:
        from . import cluster_direct as cd
        from classifier_engine.cancellation import InferenceCancelled

        try:
            return cd.run_inference_blocking(
                spec.classifier_id, spec.model_hf_path, spec.classifier_meta,
                spec.dialogues, spec.rnn_path,
                max_length=spec.max_length, window_stride=spec.window_stride,
                on_submit=on_submit, on_phase=on_phase,
            )
        except InferenceCancelled:
            # Guardrail was deleted mid-run — propagate, don't fall back.
            raise
        except BaseException as e:
            # If the job died specifically because the guardrail was deleted, that
            # is a cancellation, not a cluster outage — surface it as such.
            try:
                from classifier_engine.trainer import _classifier_deleted
                if _classifier_deleted(spec.classifier_id):
                    raise InferenceCancelled(spec.classifier_id)
            except InferenceCancelled:
                raise
            except Exception:
                pass
            # Any other failure → let the dispatcher retry on the LocalProvider,
            # exactly as the old code's cluster→local fallback did.
            raise ComputeError(f"Cluster inference failed: {e}", retryable_local=True)

    # --- training (async batch) ---
    # Thin wrappers over cluster_direct so the /train + training-status routes can
    # treat SLURM and a remote worker uniformly. Job pointers live in `raw`
    # ({slurm_job_id, remote_job_dir, job_id, mode}) so the existing DB recording
    # and cancel-on-delete logic carry over unchanged.
    def submit_training(self, spec: TrainingSpec) -> TrainingJob:
        from . import cluster_direct as cd
        result = cd.submit_training_job(
            classifier_id=spec.classifier_id, user_id=spec.user_id,
            model_hf_path=spec.model_hf_path, labels=spec.labels,
            training_config=spec.training_config, dataset_files=spec.dataset_files,
            calibration_entries=spec.calibration_entries,
        )
        return TrainingJob(
            provider=self.name, classifier_id=spec.classifier_id,
            id=str(result["slurm_job_id"]),
            raw={"slurm_job_id": result["slurm_job_id"],
                 "remote_job_dir": result["remote_job_dir"],
                 "job_id": result["job_id"], "mode": "cluster"},
        )

    def poll_training(self, job: TrainingJob) -> TrainingStatus:
        """Self-contained cluster poll: encapsulates every SLURM quirk the status
        route used to handle inline (sacct purge → status.json recovery, live
        progress, error extraction) so the route can stay transport-agnostic."""
        from . import cluster_direct as cd
        remote_dir = job.raw.get("remote_job_dir")
        try:
            st = cd.get_job_status(job.id) or {}
        except Exception as e:
            # SSH/cluster unreachable → RUNNING-but-UNREACHABLE so the caller's
            # dead-timeout can fail a job whose cluster vanished (parity with the
            # remote worker), instead of polling a dead job forever.
            return TrainingStatus(state=JobState.RUNNING, detail=f"poll failed: {e}",
                                  reachable=False)
        cs = st.get("status", "unknown")

        # SLURM purges finished jobs from sacct fast, so 'unknown' usually means
        # "done but aged out" — consult the job's status.json on the cluster.
        if cs == "unknown" and remote_dir:
            try:
                info = cd.get_job_result(remote_dir)
            except Exception as e:
                return TrainingStatus(state=JobState.RUNNING, detail=f"poll failed: {e}",
                                      reachable=False)
            if info and info.get("status") == "success":
                return TrainingStatus(state=JobState.DONE, phase="complete",
                                      detail="Trained on the cluster")
            if info and info.get("status") == "error":
                return TrainingStatus(state=JobState.ERROR, phase="failed",
                                      error=info.get("error", "Training failed on the cluster"))
            return TrainingStatus(state=JobState.RUNNING, phase="Finishing on cluster",
                                  detail="Waiting for results…")

        state = {
            "completed": JobState.DONE, "failed": JobState.ERROR, "oom": JobState.ERROR,
            "timeout": JobState.ERROR, "cancelled": JobState.CANCELLED,
            "running": JobState.RUNNING, "pending": JobState.QUEUED,
        }.get(cs, JobState.RUNNING)

        if state == JobState.ERROR:
            info = (cd.get_job_result(remote_dir) if remote_dir else None) or {}
            return TrainingStatus(state=state, phase=cs,
                                  error=info.get("error", f"Cluster job {cs}"))
        if state == JobState.DONE:
            return TrainingStatus(state=state, phase="complete", detail="Trained on the cluster")

        # Running/queued — pull live progress for a friendly detail line.
        phase, detail = "Training on cluster", f"SLURM {job.id}: {cs}"
        try:
            log = cd.get_training_log(remote_dir) if remote_dir else None
            if log and isinstance(log, list) and log:
                last = log[-1]
                if last.get("progress") is not None:
                    detail = f"Optimizing rule set — {last['progress']}% (val_acc: {last.get('val_accuracy', '?')})"
                else:
                    detail = f"Epoch {last.get('epoch', '?')}/{last.get('total_epochs', '?')} — val_acc: {last.get('val_accuracy', '?')}"
                phase = "train_rnn"
        except Exception:
            pass
        return TrainingStatus(state=state, phase=phase, detail=detail)

    def fetch_trained_model(self, job: TrainingJob, dest_dir: str) -> None:
        from . import cluster_direct as cd
        remote_dir = job.raw.get("remote_job_dir")
        if not cd.download_results(remote_dir, dest_dir):
            raise ComputeError("Downloading the trained model from the cluster failed.")
        # Clean the cluster scratch once the model is safely local (cluster-only
        # housekeeping that the old status route did inline — now encapsulated).
        try:
            if remote_dir:
                cd.cleanup_job(remote_dir)
        except Exception:
            pass

    def cancel_training(self, job: TrainingJob) -> None:
        from . import cluster_direct as cd
        try:
            cd.cancel_job(str(job.raw.get("slurm_job_id") or job.id))
        except Exception:
            pass
        try:
            if job.raw.get("remote_job_dir"):
                cd.cleanup_job(job.raw["remote_job_dir"])
        except Exception:
            pass

    # --- realtime (warm session) ---
    # Wrap realtime_session (the SSH warm-job manager) so the realtime routes can
    # treat SLURM uniformly with the remote worker. realtime_session is keyed by
    # classifier_id and is STATELESS (it reads the cluster filesystem each call),
    # so a RealtimeSession here just carries classifier_id — the route can rebuild
    # one after a backend restart and still reach a session that's still running.
    def start_realtime(self, spec: RealtimeSpec) -> RealtimeSession:
        from . import realtime_session as rs
        info = rs.start_session(
            spec.classifier_id, spec.model_hf_path, spec.classifier_meta, spec.rnn_path,
        ) or {}
        return RealtimeSession(
            provider=self.name, classifier_id=spec.classifier_id,
            id=str(spec.classifier_id), raw={"mode": "cluster", **info},
        )

    def realtime_status(self, session: RealtimeSession) -> str:
        from . import realtime_session as rs
        st = rs.session_status(session.classifier_id) or {}
        return st.get("status", "dead")

    def realtime_analyze(self, session: RealtimeSession, payload: dict) -> dict:
        from . import realtime_session as rs
        # Live (generation) mode needs longer than stored classification.
        timeout_s = 300 if payload.get("mode") == "live" else 240
        return rs.send_request(session.classifier_id, payload, timeout_s=timeout_s) or {}

    def realtime_keepalive(self, session: RealtimeSession) -> bool:
        from . import realtime_session as rs
        try:
            return bool(rs.touch_session(session.classifier_id))
        except Exception:
            return False

    def end_realtime(self, session: RealtimeSession) -> None:
        from . import realtime_session as rs
        try:
            rs.end_session(session.classifier_id)
        except Exception:
            pass

    # --- housekeeping (lifecycle hooks, was inline in main.py / crash_recovery) ---
    def recover_orphans(self) -> dict:
        """Boot-time: scancel orphaned warm-realtime jobs + sweep abandoned job
        dirs left by a prior backend that died mid-run. No-op when the cluster
        isn't configured or is unreachable right now."""
        from . import cluster_direct as cd
        if not cd.SLURM_HOST or not cd.ping(timeout=10):
            return {}
        out: dict = {}
        try:
            from . import realtime_session as rs
            rt = rs.recover_orphans()
            if rt.get("cancelled"):
                out["realtime_cancelled"] = rt["cancelled"]
        except Exception:
            pass
        try:
            swept = cd.sweep_orphan_job_dirs() or {}
            if swept.get("removed") or swept.get("errors"):
                out["job_dirs"] = swept
        except Exception:
            pass
        return out

    def end_all_realtime(self) -> int:
        from . import realtime_session as rs
        try:
            return int(rs.end_all_sessions() or 0)
        except Exception:
            return 0

    def cancel_inference(self, pointer: dict) -> None:
        """Cancel an in-flight cluster calibration/evaluation job from its stashed
        {slurm_job_id, remote_job_dir}. Ignores pointers that aren't ours."""
        if not isinstance(pointer, dict) or not pointer.get("slurm_job_id"):
            return
        from . import cluster_direct as cd
        try:
            if not cd.is_enabled():
                return
            cd.cancel_job(str(pointer["slurm_job_id"]))
            if pointer.get("remote_job_dir"):
                cd.cleanup_job(pointer["remote_job_dir"])
        except Exception:
            pass
