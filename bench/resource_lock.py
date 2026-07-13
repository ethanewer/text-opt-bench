"""Cross-process evaluation concurrency gates.

Campaign processes spend most of their lives waiting on an agent/API.  These
locks therefore gate only the expensive evaluator subprocess, rather than the
whole optimization loop.  Limits are supplied by the trusted campaign
launcher through the parent environment; evaluator children never receive
these variables.
"""

import fcntl
import json
import os
import secrets
import time
from contextlib import contextmanager
from pathlib import Path

LOCK_DIR_ENV = "TEXTOPT_EVAL_LOCK_DIR"
LIMITS_ENV = "TEXTOPT_EVAL_LIMITS"
WAIT_LOG_ENV = "TEXTOPT_EVAL_WAIT_LOG"


def _log_wait(event, wait_id):
    """Append one queue interval event for live campaign time accounting."""
    path = os.environ.get(WAIT_LOG_ENV)
    if not path:
        return
    record = json.dumps({"event": event, "id": wait_id,
                         "ts": time.time(), "pid": os.getpid()}) + "\n"
    # One O_APPEND write keeps records from concurrent self-evaluations from
    # interleaving. Accounting is telemetry and must never break evaluation.
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            os.write(fd, record.encode())
        finally:
            os.close(fd)
    except OSError:
        pass


def record_wait_interval(start_unix, end_unix, category="external"):
    """Append a trusted completed queue interval to the optimizer wait log.

    Evaluator children deliberately do not receive the wait-log path. Their
    nonce-authenticated result may nevertheless report an inner shared-device
    lease interval; the trusted runner parent records it here after validation.
    """
    path = os.environ.get(WAIT_LOG_ENV)
    if not path:
        return
    try:
        start_unix = float(start_unix)
        end_unix = float(end_unix)
    except (TypeError, ValueError):
        raise RuntimeError("external evaluation wait timestamps must be numeric")
    if not (0 < start_unix <= end_unix):
        raise RuntimeError("external evaluation wait interval is invalid")
    wait_id = f"{os.getpid()}-{category}-{secrets.token_hex(8)}"
    records = "".join(json.dumps({
        "event": event, "id": wait_id, "ts": timestamp,
        "pid": os.getpid(), "category": str(category),
    }) + "\n" for event, timestamp in (
        ("start", start_unix), ("end", end_unix)))
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            os.write(fd, records.encode())
        finally:
            os.close(fd)
    except OSError:
        pass


def configured_limits():
    """Return configured resource limits, or None outside a campaign."""
    lock_dir = os.environ.get(LOCK_DIR_ENV)
    raw = os.environ.get(LIMITS_ENV)
    if not lock_dir or not raw:
        return None
    try:
        limits = json.loads(raw)
    except (TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid {LIMITS_ENV}: {exc}") from exc
    if not isinstance(limits, dict):
        raise RuntimeError(f"{LIMITS_ENV} must be a JSON object")
    clean = {}
    for resource_name, slots in limits.items():
        if not isinstance(resource_name, str) or not resource_name:
            raise RuntimeError("evaluation resource names must be nonempty strings")
        if not isinstance(slots, int) or isinstance(slots, bool) or slots < 1:
            raise RuntimeError(f"slot count for {resource_name!r} must be >= 1")
        clean[resource_name] = slots
    return Path(lock_dir), clean


def _active_foreground_waiters(lock_dir, resource_name):
    """Return whether a live foreground waiter marker exists.

    Marker files may survive SIGKILL, but their advisory locks do not.  A
    background waiter can therefore identify and remove stale markers without
    relying on PID reuse or process-liveness heuristics.
    """
    active = False
    pattern = f"{resource_name}.foreground.*.wait"
    for path in lock_dir.glob(pattern):
        try:
            handle = open(path, "a+")
        except OSError:
            continue
        try:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                active = True
                continue
            try:
                path.unlink()
            except OSError:
                pass
        finally:
            handle.close()
    return active


@contextmanager
def evaluation_slot(resource_name="cpu", priority="foreground"):
    """Acquire one named evaluation slot and yield queue wait in seconds.

    POSIX advisory locks are released by the kernel if a loop is killed, so a
    timed-out campaign cannot leave a stale semaphore behind.  Resource names
    use separate slot pools: an accelerator evaluation may overlap a CPU one.
    """
    if priority not in ("foreground", "background"):
        raise ValueError("evaluation priority must be foreground or background")
    configured = configured_limits()
    if configured is None:
        yield 0.0
        return

    lock_dir, limits = configured
    if resource_name not in limits:
        known = ", ".join(sorted(limits))
        raise RuntimeError(
            f"no campaign evaluation limit for resource {resource_name!r}; "
            f"configured resources: {known}"
        )
    if not resource_name.replace("_", "").replace("-", "").isalnum():
        raise RuntimeError(f"unsafe evaluation resource name: {resource_name!r}")

    lock_dir.mkdir(parents=True, exist_ok=True)
    handles = [open(lock_dir / f"{resource_name}.{i}.lock", "a+")
               for i in range(limits[resource_name])]
    gate = open(lock_dir / f"{resource_name}.gate.lock", "a+")
    waiter_path = None
    waiter = None
    if priority == "foreground":
        # Register atomically with respect to a background slot acquisition.
        # Once this marker exists, later background work yields until every
        # foreground waiter has acquired a slot.
        fcntl.flock(gate.fileno(), fcntl.LOCK_EX)
        try:
            waiter_path = lock_dir / (
                f"{resource_name}.foreground.{os.getpid()}."
                f"{secrets.token_hex(8)}.wait")
            waiter = open(waiter_path, "a+")
            fcntl.flock(waiter.fileno(), fcntl.LOCK_EX)
        finally:
            fcntl.flock(gate.fileno(), fcntl.LOCK_UN)
    acquired = None
    started = time.monotonic()
    wait_id = f"{os.getpid()}-{secrets.token_hex(8)}"
    _log_wait("start", wait_id)
    wait_ended = False
    try:
        while acquired is None:
            # The gate closes the check/acquire race between a newly arriving
            # foreground waiter and low-priority test work.
            fcntl.flock(gate.fileno(), fcntl.LOCK_EX)
            try:
                if (priority == "background" and
                        _active_foreground_waiters(lock_dir, resource_name)):
                    pass
                else:
                    for handle in handles:
                        try:
                            fcntl.flock(handle.fileno(),
                                        fcntl.LOCK_EX | fcntl.LOCK_NB)
                        except BlockingIOError:
                            continue
                        acquired = handle
                        break
            finally:
                fcntl.flock(gate.fileno(), fcntl.LOCK_UN)
            if acquired is None:
                time.sleep(0.05)
        if waiter is not None:
            fcntl.flock(waiter.fileno(), fcntl.LOCK_UN)
            waiter.close()
            waiter = None
            try:
                waiter_path.unlink()
            except OSError:
                pass
        _log_wait("end", wait_id)
        wait_ended = True
        yield time.monotonic() - started
    finally:
        if not wait_ended:
            _log_wait("end", wait_id)
        if acquired is not None:
            fcntl.flock(acquired.fileno(), fcntl.LOCK_UN)
        if waiter is not None:
            fcntl.flock(waiter.fileno(), fcntl.LOCK_UN)
            waiter.close()
            try:
                waiter_path.unlink()
            except OSError:
                pass
        for handle in handles:
            handle.close()
        gate.close()
