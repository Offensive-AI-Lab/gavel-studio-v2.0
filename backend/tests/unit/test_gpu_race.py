"""Unit tests for the cluster GPU downgrade race (cluster_direct.run_gpu_race).

The race submits a powerful PRIMARY GPU job; if it hasn't started within a wait
window, it ALSO submits a weaker SECONDARY and keeps whichever STARTS first,
cancelling the loser. Status polling + sleep are injected so the whole race runs
synchronously with no cluster.
"""
import pytest

from services.compute.providers.slurm import cluster_direct as cd


def _status_from(seqs):
    """Build a _status_fn from {job_id: [s0, s1, ...]}; each call pops the next
    state, and the LAST one sticks for all further polls."""
    state = {str(k): list(v) for k, v in seqs.items()}

    def _status(jid):
        seq = state.get(str(jid), ["pending"])
        return seq.pop(0) if len(seq) > 1 else seq[0]

    return _status


@pytest.fixture
def spy(monkeypatch):
    calls = {"cancel": [], "cleanup": [], "switch": []}
    monkeypatch.setattr(cd, "cancel_job", lambda jid: calls["cancel"].append(str(jid)) or True)
    monkeypatch.setattr(cd, "cleanup_job", lambda d: calls["cleanup"].append(d))
    return calls


PRIMARY = {"slurm_job_id": "A", "remote_job_dir": "/a", "gpu": "rtx_6000:1"}
SECONDARY = {"slurm_job_id": "B", "remote_job_dir": "/b", "gpu": "rtx_4090:1"}


def _race(status, *, on_switch=None, resub=None, secondary="rtx_4090:1", wait_s=30):
    resubmitted = {"n": 0}

    def _resub(gpu):
        resubmitted["n"] += 1
        assert gpu == secondary
        return SECONDARY

    winner = cd.run_gpu_race(
        PRIMARY, resub or _resub, on_switch=on_switch,
        secondary_gpu=secondary, wait_s=wait_s, poll_interval=10,
        _status_fn=status, _sleep=lambda _s: None,
    )
    return winner, resubmitted["n"]


def test_primary_starts_within_window_no_downgrade(spy):
    winner, resubs = _race(_status_from({"A": ["running"]}))
    assert winner is PRIMARY
    assert resubs == 0                 # secondary never submitted
    assert spy["cancel"] == [] and spy["cleanup"] == []


def test_disabled_secondary_returns_primary(spy):
    winner, resubs = _race(_status_from({"A": ["pending"]}), secondary="")
    assert winner is PRIMARY
    assert resubs == 0


def test_primary_dies_in_window_no_downgrade(spy):
    winner, resubs = _race(_status_from({"A": ["failed"]}))
    assert winner is PRIMARY           # let the caller's normal handling report the failure
    assert resubs == 0


def test_secondary_wins_when_primary_stays_queued(spy):
    switched = []
    # A stays pending throughout; B is RUNNING by the first race poll.
    winner, resubs = _race(
        _status_from({"A": ["pending"], "B": ["running"]}),
        on_switch=lambda w: switched.append(w),
    )
    assert winner is SECONDARY
    assert resubs == 1
    assert switched == [SECONDARY]      # re-pointed to the winner BEFORE cancel
    assert spy["cancel"] == ["A"]       # the queued primary is cancelled
    assert "/a" in spy["cleanup"]


def test_primary_wins_the_race_after_downgrade(spy):
    switched = []
    # A is still pending through phase 1, then RUNNING on the first race poll.
    winner, resubs = _race(
        _status_from({"A": ["pending", "pending", "pending", "running"], "B": ["pending"]}),
        on_switch=lambda w: switched.append(w),
    )
    assert winner is PRIMARY
    assert resubs == 1
    assert switched == []               # no switch — primary won
    assert spy["cancel"] == ["B"]       # the weaker secondary is cancelled
    assert "/b" in spy["cleanup"]


def test_primary_fails_after_downgrade_commits_to_secondary(spy):
    switched = []
    winner, resubs = _race(
        _status_from({"A": ["pending", "pending", "pending", "failed"], "B": ["pending", "running"]}),
        on_switch=lambda w: switched.append(w),
    )
    assert winner is SECONDARY
    assert switched == [SECONDARY]
    assert spy["cancel"] == []          # primary already dead → not cancelled
    assert "/a" in spy["cleanup"]       # but its dir is cleaned


def test_resubmit_failure_stays_on_primary(spy):
    def _boom(_gpu):
        raise RuntimeError("sbatch down")

    winner, _ = _race(_status_from({"A": ["pending"]}), resub=_boom)
    assert winner is PRIMARY
    assert spy["cancel"] == []
