"""One cross-process lease for every operator-run local SLM MPS job."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import fcntl
from functools import wraps
import hashlib
import json
import os
from pathlib import Path
import time


# Shared with the paper-native runner so datagen, compilation, and native
# baseline jobs cannot accidentally place two models on MPS at once.
DEFAULT_MPS_LOCK = Path("/tmp/text-opt-bm-slm-mps.lock")
CAMPAIGN_PHASE_LOCK = Path("/tmp/text-opt-bm-slm-campaign-phase.lock")
_ACTIVE_CANONICAL_MPS_LEASES = ContextVar(
    "text_opt_bm_active_canonical_mps_leases", default=0)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_mps_lock_identity():
    """Return the content-bound identity shared by every local SLM job."""
    return {
        "path": str(DEFAULT_MPS_LOCK),
        "helper_sha256": _sha256(Path(__file__).resolve()),
    }


def require_canonical_mps_lock_identity(value, label="SLM MPS lock"):
    """Reject missing, alternate-path, or stale lock-helper provenance."""
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} identity must be an object")
    expected = canonical_mps_lock_identity()
    actual = {key: value.get(key) for key in expected}
    if actual != expected:
        raise RuntimeError(
            f"{label} identity mismatch: expected {expected}, got {actual}")
    return expected


def require_active_mps_lock(label="SLM model work"):
    """Require the current execution context to hold the canonical lease.

    This closes the Python-level ``__wrapped__``/helper-call bypass: importing a
    decorated compiler or evaluator function and calling its undecorated body
    can no longer reach model placement or compute without first acquiring the
    same process-wide Metal lock.  A ContextVar keeps concurrent threads or
    async contexts in one process from borrowing another context's lease.
    """
    if _ACTIVE_CANONICAL_MPS_LEASES.get() <= 0:
        raise RuntimeError(
            f"{label} requires the active canonical SLM MPS lease")
    return canonical_mps_lock_identity()


@contextmanager
def operator_mps_phase(purpose="operator-slm-work"):
    """Prevent operator-side model work from overlapping a live campaign.

    Operator preparation/baseline jobs take a shared phase lease for their
    complete model-bearing lifetime. The campaign launcher takes the exclusive
    counterpart before launching any optimizer, making phase separation a
    fail-closed invariant rather than an operator convention.
    """
    CAMPAIGN_PHASE_LOCK.parent.mkdir(parents=True, exist_ok=True)
    handle = CAMPAIGN_PHASE_LOCK.open("a+")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                f"cannot start {purpose}: an optimization campaign is active") from exc
        yield {"path": str(CAMPAIGN_PHASE_LOCK), "purpose": str(purpose)}
    finally:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


@contextmanager
def exclusive_campaign_mps_phase(purpose="optimization-campaign"):
    """Exclude operator-side MPS preparation/baselines for a whole campaign."""
    CAMPAIGN_PHASE_LOCK.parent.mkdir(parents=True, exist_ok=True)
    handle = CAMPAIGN_PHASE_LOCK.open("a+")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                "cannot start optimization campaign while operator-side SLM "
                "MPS work is active") from exc
        yield {"path": str(CAMPAIGN_PHASE_LOCK), "purpose": str(purpose)}
    finally:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


@contextmanager
def exclusive_mps_lock(path=DEFAULT_MPS_LOCK, timeout_seconds=3600.0,
                       purpose="slm-work", allow_noncanonical_for_test=False):
    path = Path(path)
    identity = canonical_mps_lock_identity()
    if str(path) != identity["path"] and not allow_noncanonical_for_test:
        raise RuntimeError(
            f"SLM MPS lock path is fixed at {identity['path']}; got {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+")
    started = time.monotonic()
    acquired = None
    lease_token = None
    try:
        while acquired is None:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = time.monotonic()
            except BlockingIOError:
                if (timeout_seconds is not None and timeout_seconds >= 0 and
                        time.monotonic() - started >= timeout_seconds):
                    raise TimeoutError(f"timed out waiting for SLM MPS lock {path}")
                time.sleep(0.1)
        wait_seconds = acquired - started
        acquired_unix = time.time()
        record = {
            "pid": os.getpid(),
            "purpose": str(purpose),
            **(identity if not allow_noncanonical_for_test else {
                "path": str(path), "helper_sha256": identity["helper_sha256"]}),
            "wait_started_unix": acquired_unix - wait_seconds,
            "acquired_unix": acquired_unix,
            "wait_seconds": wait_seconds,
        }
        handle.seek(0)
        handle.truncate()
        json.dump(record, handle, sort_keys=True)
        handle.write("\n")
        handle.flush()
        if str(path) == identity["path"]:
            lease_token = _ACTIVE_CANONICAL_MPS_LEASES.set(
                _ACTIVE_CANONICAL_MPS_LEASES.get() + 1)
        yield record
    finally:
        if lease_token is not None:
            _ACTIVE_CANONICAL_MPS_LEASES.reset(lease_token)
        if acquired is not None:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


@contextmanager
def shared_mps_lock(path=DEFAULT_MPS_LOCK, timeout_seconds=3600.0,
                    purpose="slm-work"):
    """Take a canonical shared MPS lease for bounded contention trials.

    Campaign resource slots remain the concurrency limit.  The shared lease
    lets those jobs overlap while continuing to exclude operator work and
    legacy evaluators that take the exclusive lease.
    """
    path = Path(path)
    identity = canonical_mps_lock_identity()
    if str(path) != identity["path"]:
        raise RuntimeError(
            f"SLM MPS lock path is fixed at {identity['path']}; got {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+")
    started = time.monotonic()
    acquired = None
    lease_token = None
    try:
        while acquired is None:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
                acquired = time.monotonic()
            except BlockingIOError:
                if (timeout_seconds is not None and timeout_seconds >= 0 and
                        time.monotonic() - started >= timeout_seconds):
                    raise TimeoutError(f"timed out waiting for SLM MPS lock {path}")
                time.sleep(0.1)
        wait_seconds = acquired - started
        acquired_unix = time.time()
        record = {
            "pid": os.getpid(), "purpose": str(purpose), **identity,
            "mode": "shared",
            "wait_started_unix": acquired_unix - wait_seconds,
            "acquired_unix": acquired_unix, "wait_seconds": wait_seconds,
        }
        lease_token = _ACTIVE_CANONICAL_MPS_LEASES.set(
            _ACTIVE_CANONICAL_MPS_LEASES.get() + 1)
        yield record
    finally:
        if lease_token is not None:
            _ACTIVE_CANONICAL_MPS_LEASES.reset(lease_token)
        if acquired is not None:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def serialized_mps_job(purpose, timeout_seconds=3600.0,
                       operator_phase=False):
    """Decorate a complete local SLM job with the shared MPS lease."""
    def decorate(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            if operator_phase:
                with operator_mps_phase(purpose):
                    with exclusive_mps_lock(
                            timeout_seconds=timeout_seconds, purpose=purpose):
                        return function(*args, **kwargs)
            with exclusive_mps_lock(
                    timeout_seconds=timeout_seconds, purpose=purpose):
                return function(*args, **kwargs)
        return wrapped
    return decorate
