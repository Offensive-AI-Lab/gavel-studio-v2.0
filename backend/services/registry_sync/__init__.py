"""Registry sync — the client-side observer wired into the local backend.

The backend ingests the public library into its DB via
`services.hf_sync.sync_library`. This package adds the real-time trigger: a
subscriber that listens to the central server's `version_update` notifications
(with WS reconnect + a safety poll as the backstop) and drives that sync.

Entry point: `build_subscriber()` (started in the backend's lifespan).
"""
from .reader import (
    HuggingFaceReader,
    RegistryNotFound,
    RegistryReader,
    RegistryReadError,
    build_reader,
)
from .subscriber import RegistrySyncSubscriber, derive_ws_url
from .wiring import build_subscriber

__all__ = [
    "RegistrySyncSubscriber", "derive_ws_url", "build_subscriber",
    # read-side port + adapters
    "RegistryReader", "HuggingFaceReader", "build_reader",
    "RegistryReadError", "RegistryNotFound",
]
