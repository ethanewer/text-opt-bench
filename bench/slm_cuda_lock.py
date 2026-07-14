"""One cross-process lease for every operator-run local SLM CUDA job."""

from contextlib import contextmanager
import fcntl
import hashlib
import os
from pathlib import Path
import time


DEFAULT_CUDA_LOCK = Path("/tmp/text-opt-bm-slm-cuda.lock")


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_cuda_lock_identity():
    """Return the content-bound identity shared by CUDA SLM jobs."""
    return {
        "path": str(DEFAULT_CUDA_LOCK),
        "helper_sha256": _sha256(Path(__file__).resolve()),
    }


def require_canonical_cuda_lock_identity(value, label="SLM CUDA lock"):
    """Reject missing, alternate-path, or stale CUDA lock provenance."""
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} identity must be an object")
    expected = canonical_cuda_lock_identity()
    actual = {key: value.get(key) for key in expected}
    if actual != expected:
        raise RuntimeError(
            f"{label} identity mismatch: expected {expected}, got {actual}")
    return expected


@contextmanager
def exclusive_cuda_lock(timeout_seconds=3600.0, purpose="slm-work"):
    """Serialize CUDA task work on a host and report lock-wait telemetry."""
    identity = canonical_cuda_lock_identity()
    path = DEFAULT_CUDA_LOCK
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+")
    started = time.monotonic()
    acquired = None
    try:
        while acquired is None:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = time.monotonic()
            except BlockingIOError:
                if (timeout_seconds is not None and timeout_seconds >= 0 and
                        time.monotonic() - started >= timeout_seconds):
                    raise TimeoutError(f"timed out waiting for SLM CUDA lock {path}")
                time.sleep(0.1)
        acquired_unix = time.time()
        record = {
            "pid": os.getpid(),
            "purpose": str(purpose),
            **identity,
            "wait_started_unix": acquired_unix - (acquired - started),
            "acquired_unix": acquired_unix,
            "wait_seconds": acquired - started,
        }
        yield record
    finally:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()
