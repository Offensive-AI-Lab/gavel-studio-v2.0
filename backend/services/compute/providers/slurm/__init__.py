"""SLURM provider package — all university-cluster code isolated here:
the provider (provider.py), the SSH transport (cluster_direct.py), and the warm
realtime-session manager (realtime_session.py). Delete this folder + cluster/ to
remove the cluster entirely."""
from .provider import SlurmProvider

__all__ = ["SlurmProvider"]
