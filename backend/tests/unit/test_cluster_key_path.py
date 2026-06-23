"""Unit cover for cluster_direct._normalize_key_path.

Regression: cluster/setup_local.sh runs in Git Bash and writes the SSH key as an
MSYS path (/c/Users/...). The backend runs as native Windows Python and shells
out to Windows OpenSSH, which can't read /c/... -> 'Identity file not accessible'
-> Permission denied -> falls back to local. We convert it to C:/... on Windows.
"""
from services.compute.providers.slurm import cluster_direct as cd


def test_msys_path_converted_on_windows(monkeypatch):
    monkeypatch.setattr(cd.os, "name", "nt")
    assert cd._normalize_key_path("/c/Users/me/.ssh/k") == "C:/Users/me/.ssh/k"
    assert cd._normalize_key_path("/d/keys/k") == "D:/keys/k"
    # An already-Windows path is left alone.
    assert cd._normalize_key_path("C:/Users/me/.ssh/k") == "C:/Users/me/.ssh/k"


def test_posix_path_left_alone(monkeypatch):
    monkeypatch.setattr(cd.os, "name", "posix")
    # On Linux/macOS a /c/... path is a real path — never rewrite it.
    assert cd._normalize_key_path("/c/Users/me/.ssh/k") == "/c/Users/me/.ssh/k"
    assert cd._normalize_key_path("/home/me/.ssh/k") == "/home/me/.ssh/k"


def test_empty_is_passthrough(monkeypatch):
    monkeypatch.setattr(cd.os, "name", "nt")
    assert cd._normalize_key_path("") == ""
