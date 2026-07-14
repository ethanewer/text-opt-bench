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
from contextlib import ExitStack, contextmanager
from pathlib import Path

LOCK_DIR_ENV = "TEXTOPT_EVAL_LOCK_DIR"
LIMITS_ENV = "TEXTOPT_EVAL_LIMITS"
WAIT_LOG_ENV = "TEXTOPT_EVAL_WAIT_LOG"
REQUESTS_ENV = "TEXTOPT_EVAL_REQUESTS"


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


def configured_requests():
    """Return per-resource capacity units requested by this evaluator.

    The campaign gives every rollout a task-specific request.  A cheap task
    normally requests one unit, while a CPU-heavy task can request several.
    Outside the weighted runner, or for an omitted resource, one unit keeps
    the original semaphore behavior.
    """
    raw = os.environ.get(REQUESTS_ENV)
    if not raw:
        return {}
    try:
        requests = json.loads(raw)
    except (TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid {REQUESTS_ENV}: {exc}") from exc
    if not isinstance(requests, dict):
        raise RuntimeError(f"{REQUESTS_ENV} must be a JSON object")
    clean = {}
    for resource_name, slots in requests.items():
        if not isinstance(resource_name, str) or not resource_name:
            raise RuntimeError("evaluation resource names must be nonempty strings")
        if not isinstance(slots, int) or isinstance(slots, bool) or slots < 1:
            raise RuntimeError(
                f"requested units for {resource_name!r} must be >= 1")
        clean[resource_name] = slots
    return clean


@contextmanager
def evaluation_slots(primary_resource="cpu", priority="foreground"):
    """Acquire every resource requested by the trusted campaign launcher.

    Most evaluators use one pool. Accelerator evaluators can also consume CPU
    capacity, however, so the weighted runner may request both. Non-CPU pools
    are acquired first in a stable order: an MPS grader never occupies scarce
    CPU units while it is still queued behind another MPS grader, and all
    multi-resource callers use the same order.

    The yielded value is the sum of sequential queue waits. Individual waits
    are also logged by :func:`evaluation_slot`; active-time accounting takes
    their union and therefore remains correct if this grows to more resources.
    """
    requests = configured_requests()
    if not requests:
        requests = {primary_resource: 1}
    elif primary_resource not in requests:
        raise RuntimeError(
            f"campaign resource request omits the evaluator's primary "
            f"resource {primary_resource!r}")

    ordered = sorted(requests, key=lambda name: (name == "cpu", name))
    waited = 0.0
    with ExitStack() as stack:
        for resource_name in ordered:
            waited += stack.enter_context(evaluation_slot(
                resource_name, priority=priority,
                slots=requests[resource_name]))
        yield waited


def _next_ticket(lock_dir, resource_name):
    """Allocate a FIFO ticket while the caller holds the resource gate."""
    path = lock_dir / f"{resource_name}.ticket"
    with open(path, "a+") as handle:
        handle.seek(0)
        raw = handle.read().strip()
        try:
            prior = int(raw) if raw else -1
        except ValueError:
            prior = -1
        # Recover monotonically even if the counter file was truncated by a
        # hard kill after waiter markers had already been published.
        for marker in lock_dir.glob(f"{resource_name}.wait.*"):
            try:
                encoded = marker.name.split(".wait.", 1)[1].split(".", 1)[0]
                prior = max(prior, int(encoded))
            except (IndexError, ValueError):
                continue
        ticket = prior + 1
        handle.seek(0)
        handle.truncate()
        handle.write(str(ticket))
        handle.flush()
        os.fsync(handle.fileno())
    return ticket


def _live_waiters(lock_dir, resource_name, own_path=None):
    """Return live waiter metadata and remove markers left by killed jobs."""
    live = []
    for path in lock_dir.glob(f"{resource_name}.wait.*"):
        handle = None
        try:
            handle = open(path, "r+")
            if own_path is not None and path == own_path:
                locked_elsewhere = True
            else:
                try:
                    fcntl.flock(
                        handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    locked_elsewhere = True
                else:
                    locked_elsewhere = False
            if not locked_elsewhere:
                try:
                    path.unlink()
                except OSError:
                    pass
                continue
            handle.seek(0)
            payload = json.load(handle)
            ticket = int(payload["ticket"])
            priority = payload["priority"]
            slots = int(payload["slots"])
            if priority not in ("foreground", "background") or slots < 1:
                raise ValueError("invalid waiter marker")
            live.append((priority, ticket, path, slots))
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            # A malformed marker whose lock is still held belongs to a broken
            # live process. Leave it in place, but do not let it wedge the
            # trusted queue. Unlocked malformed markers are removed above.
            continue
        finally:
            if handle is not None:
                handle.close()
    return live


def _first_waiter(waiters):
    """Strict foreground priority, FIFO within a priority class."""
    foreground = [item for item in waiters if item[0] == "foreground"]
    eligible = foreground or [item for item in waiters
                              if item[0] == "background"]
    return min(eligible, key=lambda item: item[1]) if eligible else None


@contextmanager
def evaluation_slot(resource_name="cpu", priority="foreground", slots=None):
    """Acquire named evaluation capacity and yield queue wait in seconds.

    POSIX advisory locks are released by the kernel if a loop is killed, so a
    timed-out campaign cannot leave a stale semaphore behind. Resource names
    use separate pools: an accelerator evaluation may overlap a CPU one.

    ``slots`` is a capacity-unit cost. If omitted, the weighted benchmark
    runner supplies it through ``TEXTOPT_EVAL_REQUESTS``; legacy callers cost
    one unit. Admission is FIFO among foreground evaluations. This prevents a
    four-unit task from starving behind a continuous stream of one-unit work.
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
    if slots is None:
        slots = configured_requests().get(resource_name, 1)
    if not isinstance(slots, int) or isinstance(slots, bool) or slots < 1:
        raise RuntimeError("evaluation capacity request must be a positive integer")
    if slots > limits[resource_name]:
        raise RuntimeError(
            f"evaluation requests {slots} {resource_name!r} units but the "
            f"campaign capacity is {limits[resource_name]}")

    lock_dir.mkdir(parents=True, exist_ok=True)
    handles = [open(lock_dir / f"{resource_name}.{i}.lock", "a+")
               for i in range(limits[resource_name])]
    gate = open(lock_dir / f"{resource_name}.gate.lock", "a+")
    waiter_path = None
    waiter = None
    acquired = []
    started = time.monotonic()
    wait_id = f"{os.getpid()}-{secrets.token_hex(8)}"
    wait_ended = False
    try:
        # Ticket allocation and marker publication are one gate-serialized
        # action, giving all contenders a stable order across processes.
        fcntl.flock(gate.fileno(), fcntl.LOCK_EX)
        try:
            ticket = _next_ticket(lock_dir, resource_name)
            waiter_path = lock_dir / (
                f"{resource_name}.wait.{ticket:020d}.{os.getpid()}."
                f"{secrets.token_hex(8)}")
            waiter = open(waiter_path, "w+")
            json.dump({"ticket": ticket, "priority": priority,
                       "slots": slots, "pid": os.getpid()}, waiter)
            waiter.flush()
            fcntl.flock(waiter.fileno(), fcntl.LOCK_EX)
        finally:
            fcntl.flock(gate.fileno(), fcntl.LOCK_UN)
        _log_wait("start", wait_id)

        while len(acquired) < slots:
            # The gate closes both waiter-order and multi-slot acquisition
            # races. A request either takes all its units or releases every
            # tentative lock, so weighted contenders cannot deadlock.
            fcntl.flock(gate.fileno(), fcntl.LOCK_EX)
            try:
                first = _first_waiter(
                    _live_waiters(lock_dir, resource_name, waiter_path))
                if first is not None and first[2] == waiter_path:
                    for handle in handles:
                        if handle in acquired:
                            continue
                        try:
                            fcntl.flock(handle.fileno(),
                                        fcntl.LOCK_EX | fcntl.LOCK_NB)
                        except BlockingIOError:
                            continue
                        acquired.append(handle)
                        if len(acquired) == slots:
                            break
                    if len(acquired) < slots:
                        for handle in acquired:
                            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                        acquired.clear()
            finally:
                fcntl.flock(gate.fileno(), fcntl.LOCK_UN)
            if len(acquired) < slots:
                time.sleep(0.05)
        fcntl.flock(gate.fileno(), fcntl.LOCK_EX)
        try:
            fcntl.flock(waiter.fileno(), fcntl.LOCK_UN)
            waiter.close()
            waiter = None
            try:
                waiter_path.unlink()
            except OSError:
                pass
        finally:
            fcntl.flock(gate.fileno(), fcntl.LOCK_UN)
        _log_wait("end", wait_id)
        wait_ended = True
        yield time.monotonic() - started
    finally:
        if not wait_ended:
            _log_wait("end", wait_id)
        for handle in acquired:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
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
