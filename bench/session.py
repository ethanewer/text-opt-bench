"""Benchmark sessions: the canonical record of an optimization run.

A *session* is the benchmark-side artifact of any optimization run,
whatever produced it — the bundled loop in `loop/`, a coding CLI's goal
mode, a custom evolutionary search, or a human editing by hand. It is an
append-only history of actual submissions: the exact program text
submitted, the score it received, and the timeline (wall-clock
timestamps and the lapse between submissions). Timing is *metadata*
only — scores never depend on it.

Layout of a run directory:

    session.json          task, feedback mode, creation time
    submissions.jsonl     one record per submission (append-only, hash-chained)
    submissions/NNN.py    exact bytes of submission NNN (what was scored)
    best_program.py       the best (lowest guide score) valid submission so far

Each record carries the SHA-256 of the previous record line (`prev`),
so any edit to the history breaks the chain, and the SHA-256 of the
program snapshot. `python3.12 -m bench verify RUN_DIR` checks both and
can optionally re-score every submission — scores are deterministic, so
a re-score must reproduce every record exactly.

Information regimes: the plaintext part of each record is exactly what
the optimizer is allowed to see under the session's feedback mode.
Hidden parts (held-out test scores always; validation scores too in
train-only mode) are stored in the record's `sealed` field, obfuscated
the same way as the repo's held-out task data (`bench/heldout.py`) —
so an agent that happens to read the run directory mid-run learns
nothing it shouldn't, while experimenter tooling (`bench report
--unseal`, `Session.full_result`) recovers the full picture. This is
casual-leak protection under the cooperative threat model, not
encryption.

Nothing in this module knows about git, codex, or any particular
optimization algorithm.
"""

import base64
import datetime
import fcntl
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path

from bench import heldout, runner

FORMAT = 1

# Metric keys hidden from the optimizer in each feedback mode.
# "full": train data + validation scores visible, test always hidden.
# "train-only": only train scores visible (blind mode).
HIDDEN_KEYS = {
    "full": ("test_score", "test_ratio", "n_test"),
    "train-only": ("val_score", "val_ratio", "n_val",
                   "test_score", "test_ratio", "n_test"),
}
FEEDBACK_MODES = tuple(HIDDEN_KEYS)


def guide_score(result, feedback):
    """The score an optimizer is allowed to select on in this mode."""
    if feedback == "train-only" and "train_score" in (result.get("metrics") or {}):
        return result["metrics"]["train_score"]
    return result["score"]


def visible_metrics(metrics, feedback):
    # Prefix rules cover task-specific diagnostics (test_perplexity,
    # test_nll_delta, val_accuracy, ...), not just the legacy fixed names.
    # A new evaluator can therefore add held-out metrics without silently
    # creating an information leak in the session layer.
    def allowed(key):
        if key.startswith("test_"):
            return False
        if feedback == "train-only" and key.startswith("val_"):
            return False
        return key not in HIDDEN_KEYS[feedback]

    return {k: v for k, v in (metrics or {}).items() if allowed(k)}


def _sha256_text(text):
    return hashlib.sha256(text.encode()).hexdigest()


def _seal(obj):
    return base64.b64encode(heldout.encode(obj)).decode()


def _unseal(blob):
    return heldout.decode(base64.b64decode(blob))


def _benchmark_fingerprint(task):
    """Use the same protocol/data identity as deferred holdout caches."""
    # Local import avoids making the generic session module depend on deferred
    # scoring during module initialization. bench.deferred itself imports only
    # runner/heldout, so this runtime import is acyclic.
    from bench.deferred import benchmark_fingerprint
    return benchmark_fingerprint(task)


def _require_current_fingerprint(meta, task):
    """Reject unbound legacy runs and runs from a different task protocol."""
    recorded = meta.get("benchmark_fingerprint")
    if not isinstance(recorded, str) or len(recorded) != 64:
        raise ValueError(
            "legacy session is not bound to a benchmark fingerprint; "
            "start a fresh run directory (legacy runs remain inspectable as "
            "raw files, but cannot be resumed, read as benchmark results, or "
            "verified under the current protocol)")
    current = _benchmark_fingerprint(task)
    if recorded != current:
        raise ValueError(
            f"benchmark fingerprint mismatch for task {task!r}: session "
            f"records {recorded}, current protocol/data are {current}; "
            "start a fresh run directory")
    return current


def _atomic_write_json(path, value):
    """Durably publish JSON without exposing a partially written file."""
    path = Path(path)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            fd = -1  # ownership transferred to ``handle``
            json.dump(value, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)

        # Persist the directory entry where the platform supports fsync on a
        # directory. A failed directory fsync does not compromise the atomic
        # visibility guarantee, so tolerate filesystems that reject it.
        directory_fd = None
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
            os.fsync(directory_fd)
        except OSError:
            pass
        finally:
            if directory_fd is not None:
                os.close(directory_fd)
    finally:
        if fd != -1:
            os.close(fd)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


class Session:
    """Append-only submission history for one benchmark run."""

    def __init__(self, run_dir, meta):
        self.run_dir = Path(run_dir)
        if not isinstance(meta, dict) or meta.get("format") != FORMAT:
            raise ValueError("invalid benchmark session metadata format")
        task = meta.get("task")
        config = runner.load_config(task)
        feedback = meta.get("feedback")
        allowed_feedback = tuple(config.get("feedback_modes", FEEDBACK_MODES))
        if feedback not in HIDDEN_KEYS or feedback not in allowed_feedback:
            raise ValueError(
                f"session feedback {feedback!r} is not allowed for task "
                f"{task!r}; expected one of {allowed_feedback}")
        expected_kind = config.get("kind", "perfect")
        if meta.get("kind") != expected_kind:
            raise ValueError(
                f"session kind {meta.get('kind')!r} does not match task "
                f"kind {expected_kind!r}")
        self.benchmark_fingerprint = _require_current_fingerprint(meta, task)
        self.meta = meta
        self.task = task
        self.feedback = feedback
        self._replay()

    # -- construction --------------------------------------------------

    @classmethod
    def create(cls, run_dir, task, feedback="full"):
        run_dir = Path(run_dir)
        runner.task_dir(task)  # validates the task name
        if feedback not in HIDDEN_KEYS:
            raise ValueError(f"feedback must be one of {FEEDBACK_MODES}")
        config = runner.load_config(task)
        allowed_feedback = tuple(config.get("feedback_modes", FEEDBACK_MODES))
        if feedback not in allowed_feedback:
            raise ValueError(
                f"task {task!r} supports feedback modes {allowed_feedback}, "
                f"not {feedback!r}")
        run_dir.mkdir(parents=True, exist_ok=True)
        session_path = run_dir / "session.json"
        with open(run_dir / ".lock", "a+") as lock:
            # Serialize creators with submissions/readers using the same lock.
            # The existence check belongs inside the lock so two concurrent
            # creators cannot both publish metadata for one run directory.
            fcntl.flock(lock, fcntl.LOCK_EX)
            if session_path.exists():
                raise FileExistsError(
                    f"session already exists in {run_dir}")
            (run_dir / "submissions").mkdir(exist_ok=True)
            now = time.time()
            meta = {
                "format": FORMAT,
                "task": task,
                "kind": config.get("kind", "perfect"),
                "feedback": feedback,
                "benchmark_fingerprint": _benchmark_fingerprint(task),
                "created": datetime.datetime.fromtimestamp(now).isoformat(
                    timespec="seconds"),
                "created_ts": round(now, 3),
            }
            _atomic_write_json(session_path, meta)
        return cls(run_dir, meta)

    @classmethod
    def open(cls, run_dir):
        run_dir = Path(run_dir)
        meta = json.loads((run_dir / "session.json").read_text())
        return cls(run_dir, meta)

    @classmethod
    def open_or_create(cls, run_dir, task=None, feedback=None):
        """Open an existing session (checking task/feedback match if given),
        or create one (task required; feedback defaults to "full")."""
        run_dir = Path(run_dir)
        if (run_dir / "session.json").exists():
            s = cls.open(run_dir)
            if task is not None and task != s.task:
                raise ValueError(
                    f"run dir {run_dir} is a session for task {s.task!r}, "
                    f"not {task!r}")
            if feedback is not None and feedback != s.feedback:
                raise ValueError(
                    f"session feedback mode is {s.feedback!r} and cannot be "
                    f"changed to {feedback!r}")
            return s
        if task is None:
            raise ValueError(
                f"no session in {run_dir}; pass a task to create one")
        return cls.create(run_dir, task, feedback or "full")

    # -- state ----------------------------------------------------------

    def _replay(self):
        """Rebuild in-memory state (records, best, chain tip) from disk."""
        self.records = []
        self.best = None
        self._prev_sha = None
        path = self.run_dir / "submissions.jsonl"
        if path.exists():
            for line in path.read_text().splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                if record.get("benchmark_fingerprint") != self.benchmark_fingerprint:
                    raise ValueError(
                        "submission benchmark fingerprint is missing or does "
                        "not match its session; legacy/mixed-protocol histories "
                        "cannot be resumed")
                self.records.append(record)
                self._prev_sha = _sha256_text(line)
        for rec in self.records:
            # guide_score None on an ok record can only come from a
            # corrupted/hand-edited history; never let it brick the session.
            if (rec["ok"] and rec.get("guide_score") is not None
                    and (self.best is None
                         or rec["guide_score"] < self.best["guide_score"])):
                self.best = rec

    def _score_hidden(self):
        # In blind mode on a generalization task, the raw score (validation
        # error) is itself hidden information; only the train-based guide
        # score may appear in plaintext.
        return (self.meta.get("kind") == "generalization"
                and self.feedback == "train-only")

    # -- the one verb ----------------------------------------------------

    def submit(self, program_path, note=""):
        """Score a program and append it to the run's history.

        The returned record's plaintext fields are exactly what the
        optimizer may see; hidden metrics live in its `sealed` field.
        Snapshots the exact bytes first and scores the snapshot, so the
        record always matches what was scored. Safe across processes: a
        lock file serializes concurrent submissions.
        """
        data = Path(program_path).read_bytes()
        with open(self.run_dir / ".lock", "a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            self._replay()  # pick up submissions from other processes
            # A long-lived Session object must not silently cross a coherent
            # evaluator/data/environment update. Rehash at every submission,
            # not only when the Session was constructed.
            _require_current_fingerprint(self.meta, self.task)
            n = len(self.records)
            snap = self.run_dir / "submissions" / f"{n:03d}.py"
            snap.write_bytes(data)

            ts = time.time()
            t0 = time.monotonic()
            config = runner.load_config(self.task)
            # Accelerator-heavy tasks can seal accepted-incumbent tests in a
            # separate low-priority queue.  Their trajectory score contains
            # only the task's declared online objective (validation-only for
            # SLM compression); ordinary generalization tasks preserve the
            # historical synchronous-final behavior.
            deferred_test = bool(config.get("deferred_test", False))
            final = (self.meta.get("kind") == "generalization"
                     and not deferred_test)
            development_profile = None
            if config.get("development_profile"):
                development_profile = config["development_profile"]
            result = runner.evaluate(
                self.task, snap, final=final,
                development_profile=development_profile)
            # Also close the update-during-evaluation window. Never append a
            # score under the old identity if any bound byte changed while the
            # evaluator was running; the unrecorded numbered snapshot is safe
            # to overwrite on the next valid attempt.
            _require_current_fingerprint(self.meta, self.task)

            metrics = result.get("metrics") or {}
            vis = visible_metrics(metrics, self.feedback)
            hidden = {k: v for k, v in metrics.items() if k not in vis}
            score_hidden = self._score_hidden()
            sealed = None
            if hidden or (score_hidden and result["score"] is not None):
                sealed = _seal({"score": result["score"], "metrics": hidden})
            rec = {
                "n": n,
                "time": datetime.datetime.fromtimestamp(ts).isoformat(
                    timespec="seconds"),
                "ts": round(ts, 3),
                "dt": (round(ts - self.records[-1]["ts"], 3)
                       if self.records else None),
                "eval_seconds": round(time.monotonic() - t0, 3),
                # local-compute cost of this grading (bench/trace.py rescale
                # basis): CPU seconds is contention-immune, wall for context.
                "eval_wall_seconds": result.get("eval_wall_seconds"),
                "eval_cpu_seconds": result.get("eval_cpu_seconds"),
                "eval_queue_seconds": result.get("eval_queue_seconds"),
                "note": note,
                "program": f"submissions/{n:03d}.py",
                "program_sha256": hashlib.sha256(data).hexdigest(),
                "benchmark_fingerprint": self.benchmark_fingerprint,
                "ok": bool(result["ok"]),
                "score": None if score_hidden else result["score"],
                "guide_score": (guide_score(result, self.feedback)
                                if result["ok"] else None),
                "metrics": vis,
                "sealed": sealed,
                "error": result.get("error"),
                "best": False,
                "prev": self._prev_sha,
            }
            if development_profile is not None:
                rec["development_profile"] = development_profile
            if (rec["ok"] and rec["guide_score"] is not None
                    and (self.best is None
                         or rec["guide_score"] < self.best["guide_score"])):
                rec["best"] = True
            if deferred_test and rec["ok"]:
                rec["holdout"] = "pending" if rec["best"] else "not-required"
            line = json.dumps(rec)
            with open(self.run_dir / "submissions.jsonl", "a") as f:
                f.write(line + "\n")
            self._prev_sha = _sha256_text(line)
            self.records.append(rec)
            if rec["best"]:
                self.best = rec
                (self.run_dir / "best_program.py").write_bytes(data)
        return rec

    # -- views ------------------------------------------------------------

    def visible(self, rec):
        """The part of a record an optimizing agent may see."""
        return {
            "n": rec["n"],
            "time": rec["time"],
            "dt": rec["dt"],
            "ok": rec["ok"],
            "score": rec["guide_score"],
            "metrics": rec["metrics"],
            "error": rec["error"],
            "best": rec["best"],
            "best_score": self.best["guide_score"] if self.best else None,
        }

    def full_result(self, rec, include_deferred=True):
        """Raw score + complete metrics, unsealing hidden parts.

        For experimenter tooling only — never show this to an optimizing
        agent mid-run.
        """
        metrics = dict(rec["metrics"])
        score = rec["score"]
        if rec.get("sealed"):
            hidden = _unseal(rec["sealed"])
            metrics.update(hidden["metrics"])
            if score is None:
                score = hidden["score"]
        if include_deferred and runner.load_config(self.task).get("deferred_test"):
            from bench.deferred import result_for
            deferred = result_for(
                self.run_dir, rec["n"], rec["program_sha256"])
            if deferred is not None:
                metrics.update(deferred["metrics"])
            elif rec.get("holdout") == "pending":
                metrics["test_pending"] = True
        return {"score": score, "metrics": metrics}

    def summary(self):
        valid = [r for r in self.records if r["ok"]]
        deferred = runner.load_config(self.task).get("deferred_test", False)
        pending = complete = 0
        if deferred:
            from bench.deferred import read_results
            completed = read_results(self.run_dir)
            required = [r for r in self.records
                        if r.get("holdout") == "pending"]
            complete = sum(r["n"] in completed for r in required)
            pending = len(required) - complete
        return {
            "task": self.task,
            "feedback": self.feedback,
            "submissions": len(self.records),
            "valid": len(valid),
            "best_score": self.best["guide_score"] if self.best else None,
            "best_n": self.best["n"] if self.best else None,
            "holdouts_pending": pending,
            "holdouts_complete": complete,
            "span_seconds": (round(self.records[-1]["ts"] - self.records[0]["ts"], 3)
                             if len(self.records) > 1 else 0.0),
        }


def verify_run(run_dir, rescore=False):
    """Check a run's submission history for integrity.

    Verifies: records parse and are correctly numbered, the prev-hash
    chain is unbroken, every program snapshot matches its recorded
    SHA-256, and best flags are consistent. With rescore=True, re-scores
    every submission — deterministic scoring means the guide score, the
    visible metrics, and the sealed hidden parts must all reproduce
    exactly.

    Returns a list of problem strings (empty = intact).
    """
    run_dir = Path(run_dir)
    problems = []
    try:
        session = Session.open(run_dir)
    except (OSError, json.JSONDecodeError, KeyError, ValueError) as e:
        return [f"cannot open session: {e}"]

    path = run_dir / "submissions.jsonl"
    lines = [l for l in path.read_text().splitlines() if l.strip()] \
        if path.exists() else []
    prev_sha = None
    best = None
    for i, line in enumerate(lines):
        where = f"record {i}"
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            problems.append(f"{where}: not valid JSON")
            prev_sha = _sha256_text(line)
            continue
        if rec.get("n") != i:
            problems.append(f"{where}: numbered {rec.get('n')}, expected {i}")
        if rec.get("benchmark_fingerprint") != session.benchmark_fingerprint:
            problems.append(f"{where}: benchmark fingerprint mismatch")
        if rec.get("prev") != prev_sha:
            problems.append(f"{where}: hash chain broken (prev mismatch)")
        prev_sha = _sha256_text(line)

        snap = run_dir / rec.get("program", f"submissions/{i:03d}.py")
        if not snap.is_file():
            problems.append(f"{where}: program snapshot {snap.name} missing")
        elif hashlib.sha256(snap.read_bytes()).hexdigest() != rec.get("program_sha256"):
            problems.append(f"{where}: program snapshot does not match recorded sha256")

        if rec.get("ok") and rec.get("guide_score") is None:
            problems.append(f"{where}: ok record has no guide_score")
        is_best = (bool(rec.get("ok"))
                   and rec.get("guide_score") is not None
                   and (best is None or rec.get("guide_score") < best))
        if is_best:
            best = rec.get("guide_score")
        if bool(rec.get("best")) != is_best:
            problems.append(f"{where}: best flag inconsistent with history")

        if rescore and snap.is_file():
            deferred_test = runner.load_config(session.task).get(
                "deferred_test", False)
            final = (session.meta.get("kind") == "generalization"
                     and not deferred_test)
            # Honor the task's score_tolerance: low-variance tasks (e.g. a
            # peak-memory metric with a small pymalloc-arena flicker) are
            # not bit-exact, so a rescore may differ by up to the tolerance
            # — the same allowance `bench determinism` makes.
            cfg = runner.load_config(session.task)
            tol = cfg.get("score_tolerance", 0)
            tolerant_metrics = set(cfg.get("tolerant_metrics", ()))
            development_profile = None
            if cfg.get("development_profile"):
                development_profile = cfg["development_profile"]
            result = runner.evaluate(
                session.task, snap, final=final,
                development_profile=development_profile)
            if bool(result["ok"]) != bool(rec.get("ok")):
                problems.append(f"{where}: re-score ok={result['ok']}, "
                                f"recorded ok={rec.get('ok')}")
            elif result["ok"]:
                fresh_full = result.get("metrics") or {}
                try:
                    recorded = session.full_result(rec, include_deferred=False)
                except Exception:
                    problems.append(f"{where}: sealed field is unreadable")
                    continue
                if not (_close(rec.get("guide_score"),
                               guide_score(result, session.feedback), tol)
                        and _close(recorded["score"], result["score"], tol)
                        and _metrics_close(recorded["metrics"], fresh_full,
                                           tol, tolerant_metrics)):
                    problems.append(f"{where}: re-score does not reproduce the "
                                    f"recorded score/metrics (beyond "
                                    f"tolerance {tol})")
    try:
        from bench.deferred import verify_results
        problems.extend(verify_results(run_dir))
    except Exception as exc:
        problems.append(f"cannot verify heldout results: {exc}")
    return problems


def _close(a, b, tol):
    """Equal, or numerically within tol."""
    if a == b:
        return True
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(a - b) <= tol
    return False


def _metrics_close(a, b, tol, tolerant_metrics=()):
    """Metric dicts must match exactly, EXCEPT keys named in
    `tolerant_metrics` — the score-like memory metrics — which may differ by
    up to `tol` (per-element for lists like peak_bytes_per_instance). The
    tolerance deliberately does NOT reach invariant metadata/counts
    (n_instances, prompt_len, n_gen, n_docs, ...), so tampering with those is
    still caught."""
    if set(a) != set(b):
        return False
    for k in a:
        av, bv = a[k], b[k]
        key_tol = tol if k in tolerant_metrics else 0
        if isinstance(av, list) and isinstance(bv, list):
            if len(av) != len(bv) or not all(_close(x, y, key_tol)
                                             for x, y in zip(av, bv)):
                return False
        elif not _close(av, bv, key_tol):
            return False
    return True
