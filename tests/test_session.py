"""Tests for bench.session — the benchmark's canonical submission record.

The invariants under test: every submission (valid, worse, or invalid)
is recorded with its exact program bytes, score, and timeline; best
tracking is strictly-better; feedback modes filter what is *visible*
but never what is *recorded*; and verify_run catches any tampering
with the record.

Run with:  python3.12 tests/test_session.py
"""

import json
import shutil
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bench import deferred, runner, session as session_module
from bench.session import Session, verify_run, visible_metrics

failures = []


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    if not cond:
        failures.append(name)


def main():
    tmp = Path(tempfile.mkdtemp(prefix="textopt_session_test_"))
    try:
        run(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print()
    if failures:
        sys.exit(f"{len(failures)} check(s) FAILED: {failures}")
    print("all session checks passed")


def run(tmp):
    # -- creation publication is atomic ---------------------------------
    # Pause immediately before rename: readers must see no session at all,
    # while the complete temporary JSON is private. This deterministically
    # covers the race between Session.create and the deferred queue scanner.
    race_dir = tmp / "create_race"
    replace_reached = threading.Event()
    allow_replace = threading.Event()
    creation_errors = []
    real_replace = session_module.os.replace

    def delayed_replace(source, destination):
        if Path(destination) == race_dir / "session.json":
            replace_reached.set()
            if not allow_replace.wait(timeout=5):
                raise TimeoutError("test did not release metadata publication")
        return real_replace(source, destination)

    def create_race_session():
        try:
            Session.create(race_dir, "mem_index")
        except BaseException as exc:
            creation_errors.append(exc)

    session_module.os.replace = delayed_replace
    creator = threading.Thread(target=create_race_session, daemon=True)
    try:
        creator.start()
        reached = replace_reached.wait(timeout=5)
        check("session creation reaches atomic publication point", reached)
        check("session metadata is absent before atomic publication",
              reached and not (race_dir / "session.json").exists())
        temporary_files = list(race_dir.glob(".session.json.*.tmp"))
        complete_temporary = False
        if len(temporary_files) == 1:
            try:
                temporary_meta = json.loads(temporary_files[0].read_text())
                complete_temporary = temporary_meta.get("task") == "mem_index"
            except (OSError, json.JSONDecodeError):
                pass
        check("private metadata temporary is already complete",
              complete_temporary)
        check("deferred reader skips an unpublished session cleanly",
              deferred.pending_request([race_dir], tmp / "race_cache") is None)
    finally:
        allow_replace.set()
        creator.join(timeout=5)
        session_module.os.replace = real_replace
    check("atomic session publication completes without error",
          not creator.is_alive() and not creation_errors,
          repr(creation_errors[:1]))
    check("published session is complete and temporary is removed",
          Session.open(race_dir).task == "mem_index"
          and not list(race_dir.glob(".session.json.*.tmp")))

    run_dir = tmp / "run"
    baseline = runner.initial_program("mem_index")
    solution = ROOT / "tests" / "solutions" / "mem_index.py"
    broken = tmp / "broken.py"
    broken.write_text("def store(pairs:\n")  # syntax error

    # -- lifecycle: baseline, improvement, worse resubmit, invalid ------
    s = Session.create(run_dir, "mem_index")
    expected_fingerprint = deferred.benchmark_fingerprint("mem_index")
    check("session is bound to current benchmark fingerprint",
          s.meta["benchmark_fingerprint"] == expected_fingerprint)
    r0 = s.submit(baseline, note="baseline")
    check("baseline recorded ok", r0["ok"] and r0["best"] and r0["n"] == 0)
    check("submission is bound to session fingerprint",
          r0["benchmark_fingerprint"] == expected_fingerprint)
    check("first dt is None", r0["dt"] is None)

    # A Session object may live for an hour while task code/data are updated.
    # Every submit must rebind at the boundary instead of recording a score
    # from changed bytes under the construction-time fingerprint.
    real_session_fingerprint = session_module._benchmark_fingerprint
    session_module._benchmark_fingerprint = lambda task: (
        "e" * 64 if task == "mem_index" else real_session_fingerprint(task))
    try:
        s.submit(baseline, note="must reject changed protocol")
    except ValueError:
        changed_before_score_rejected = True
    else:
        changed_before_score_rejected = False
    finally:
        session_module._benchmark_fingerprint = real_session_fingerprint
    check("submit rechecks a live session fingerprint",
          changed_before_score_rejected and len(s.records) == 1)

    # Recheck once more after the evaluator returns so an update that lands
    # during a long model score cannot be appended under the old identity.
    real_evaluate = session_module.runner.evaluate
    changed_during_score = {"value": False}

    def change_after_evaluate(*args, **kwargs):
        result = real_evaluate(*args, **kwargs)
        changed_during_score["value"] = True
        return result

    session_module.runner.evaluate = change_after_evaluate
    session_module._benchmark_fingerprint = lambda task: (
        "d" * 64 if task == "mem_index" and changed_during_score["value"]
        else real_session_fingerprint(task))
    try:
        s.submit(baseline, note="must reject mid-score protocol change")
    except ValueError:
        changed_during_score_rejected = True
    else:
        changed_during_score_rejected = False
    finally:
        session_module.runner.evaluate = real_evaluate
        session_module._benchmark_fingerprint = real_session_fingerprint
    check("submit rejects a fingerprint change during evaluation",
          changed_during_score_rejected and len(s.records) == 1)

    r1 = s.submit(solution, note="reference solution")
    check("improvement becomes best",
          r1["ok"] and r1["best"] and r1["guide_score"] < r0["guide_score"])
    check("dt recorded", isinstance(r1["dt"], float) and r1["dt"] >= 0)
    r2 = s.submit(baseline, note="worse again")
    check("worse resubmission recorded but not best",
          r2["ok"] and not r2["best"] and r2["n"] == 2)
    r3 = s.submit(broken, note="broken")
    check("invalid submission recorded",
          not r3["ok"] and not r3["best"] and r3["error"])
    check("best_program.py is the best submission",
          (run_dir / "best_program.py").read_bytes() == solution.read_bytes())
    check("snapshots preserve exact bytes",
          (run_dir / "submissions" / "000.py").read_bytes()
          == baseline.read_bytes())

    # -- reopen: state replays from disk --------------------------------
    s2 = Session.open(run_dir)
    check("reopen replays records",
          len(s2.records) == 4 and s2.best["n"] == r1["n"])
    r4 = s2.submit(baseline, note="after reopen")
    check("indices continue after reopen", r4["n"] == 4)

    # -- open_or_create guards -------------------------------------------
    try:
        Session.open_or_create(run_dir, task="mem_str")
        check("task mismatch rejected", False)
    except ValueError:
        check("task mismatch rejected", True)
    try:
        Session.open_or_create(run_dir, task="mem_index", feedback="train-only")
        check("feedback change rejected", False)
    except ValueError:
        check("feedback change rejected", True)

    # -- verify: intact, then every tampering mode ----------------------
    check("verify passes on intact run", verify_run(run_dir) == [])

    # Session reads/resumes/verifications fail closed across protocol/data
    # changes. Legacy sessions remain raw-file inspectable but are deliberately
    # not treated as current benchmark records.
    session_path = run_dir / "session.json"
    original_session = session_path.read_text()
    stale_meta = json.loads(original_session)
    stale_meta["benchmark_fingerprint"] = "0" * 64
    session_path.write_text(json.dumps(stale_meta))
    try:
        Session.open(run_dir)
        check("stale session fingerprint rejected", False)
    except ValueError as exc:
        check("stale session fingerprint rejected",
              "fingerprint mismatch" in str(exc))
    check("verify rejects stale session fingerprint",
          any("fingerprint mismatch" in problem
              for problem in verify_run(run_dir)))
    legacy_meta = json.loads(original_session)
    legacy_meta.pop("benchmark_fingerprint")
    session_path.write_text(json.dumps(legacy_meta))
    try:
        Session.open(run_dir)
        check("legacy unbound session rejected", False)
    except ValueError as exc:
        check("legacy unbound session rejected", "legacy session" in str(exc))
    session_path.write_text(original_session)
    # Rescore-reproducibility is checked on a bit-exact task (ops_connect,
    # instruction-counted). The mem_index reference can jitter by a few
    # allocator bytes — low-variance but not bit-exact, so it is unsuitable for a
    # strict rescore==record assertion. Memory-metric determinism is
    # covered by `bench determinism` on the (stable) initial programs.
    det_run = run_dir.parent / "det_run"
    det = Session.create(det_run, "ops_connect")
    det.submit(runner.initial_program("ops_connect"), note="baseline")
    det.submit(ROOT / "tests" / "solutions" / "ops_connect.py", note="solution")
    check("verify --rescore reproduces the record (bit-exact task)",
          verify_run(det_run, rescore=True) == [])

    jsonl = run_dir / "submissions.jsonl"
    original = jsonl.read_text()
    lines = original.splitlines()

    wrong_protocol = json.loads(lines[0])
    wrong_protocol["benchmark_fingerprint"] = "f" * 64
    jsonl.write_text("\n".join([json.dumps(wrong_protocol)] + lines[1:]) + "\n")
    check("submission fingerprint mismatch rejected",
          any("fingerprint" in problem for problem in verify_run(run_dir)))
    jsonl.write_text(original)

    doctored = json.loads(lines[1])
    doctored["guide_score"] = 1.0  # fake a better score
    jsonl.write_text("\n".join([lines[0], json.dumps(doctored)] + lines[2:]) + "\n")
    check("edited record breaks the hash chain",
          any("chain" in p for p in verify_run(run_dir)))
    jsonl.write_text(original)

    jsonl.write_text("\n".join([lines[0]] + lines[2:]) + "\n")
    check("deleted record detected", verify_run(run_dir) != [])
    jsonl.write_text(original)

    snap = run_dir / "submissions" / "001.py"
    snap_bytes = snap.read_bytes()
    snap.write_bytes(b"# swapped program\n")
    check("swapped snapshot detected",
          any("sha256" in p for p in verify_run(run_dir)))
    snap.write_bytes(snap_bytes)
    check("verify passes after restore", verify_run(run_dir) == [])

    # -- feedback filtering + sealing on a generalization task -----------
    g = Session.create(tmp / "run_blind", "word_problems",
                       feedback="train-only")
    gr = g.submit(runner.initial_program("word_problems"), note="baseline")
    check("plaintext record hides val+test in train-only mode",
          not any(k.startswith(("val", "test")) or k in ("n_val", "n_test")
                  for k in gr["metrics"])
          and gr["score"] is None and gr["sealed"])
    full = g.full_result(gr)
    check("full_result unseals held-out split",
          "test_score" in full["metrics"]
          and full["score"] == full["metrics"]["train_score"])
    check("guide score is the train score in train-only mode",
          gr["guide_score"] == full["metrics"]["train_score"])
    raw = (tmp / "run_blind" / "submissions.jsonl").read_text()
    check("hidden keys never appear in plaintext on disk",
          "test_score" not in raw and "val_score" not in raw)
    check("blind-session rescore reproduces sealed record",
          verify_run(tmp / "run_blind", rescore=True) == [])
    check("visible_metrics full mode hides only test",
          visible_metrics({"train_score": 1, "val_score": 2,
                           "test_score": 3}, "full")
          == {"train_score": 1, "val_score": 2})

    # Active CPU research tasks must never execute their sealed test split in
    # the online submission. Only validation determines validity/selection;
    # accepted incumbents are handed to the background deferred queue.
    real_evaluate = runner.evaluate
    deferred_calls = []

    def validation_only_evaluate(task, _program, **kwargs):
        deferred_calls.append((task, dict(kwargs)))
        return {
            "ok": True, "score": 0.25,
            "metrics": {"train_score": 0.3, "val_score": 0.25},
            "error": None, "eval_wall_seconds": 0.01,
            "eval_cpu_seconds": 0.01, "eval_queue_seconds": 0.0,
        }

    runner.evaluate = validation_only_evaluate
    try:
        for task in ("llm_routing", "optimizer_generalization"):
            active = Session.create(tmp / ("deferred_" + task), task)
            record = active.submit(runner.initial_program(task), note="baseline")
            check(f"{task} online submission is validation-only",
                  deferred_calls[-1][0] == task
                  and deferred_calls[-1][1].get("final") is False
                  and not any(key.startswith("test_")
                              for key in record["metrics"]))
            check(f"{task} accepted incumbent queues sealed test",
                  record["ok"] and record["best"]
                  and record.get("holdout") == "pending")
    finally:
        runner.evaluate = real_evaluate

    # -- CLI round trip ---------------------------------------------------
    cli_run = tmp / "cli_run"
    def bench(*a):
        return subprocess.run(
            [sys.executable, "-m", "bench", *a],
            capture_output=True, text=True, cwd=ROOT)
    sub = bench("submit", str(cli_run), str(baseline), "--task", "mem_index",
                "--note", "cli baseline")
    check("CLI submit works",
          sub.returncode == 0 and "NEW BEST" in sub.stdout, sub.stdout.strip()[:80])
    sub2 = bench("submit", str(cli_run), str(broken))
    check("CLI submit reports invalid with exit 1",
          sub2.returncode == 1 and "INVALID" in sub2.stdout)
    rep = bench("report", str(cli_run))
    check("CLI report shows history",
          rep.returncode == 0 and "cli baseline" in rep.stdout
          and "2 submissions" in rep.stdout)
    ver = bench("verify", str(cli_run))
    check("CLI verify passes", ver.returncode == 0 and "OK" in ver.stdout,
          (ver.stdout + ver.stderr).strip()[:240])

    # -- workspace command (goal-mode setup) ------------------------------
    ws = tmp / "ws"
    wsp = bench("workspace", "mem_index", str(ws))
    goal = (ws / "GOAL.md").read_text() if (ws / "GOAL.md").exists() else ""
    check("workspace materializes program/spec/GOAL",
          wsp.returncode == 0 and (ws / "program.py").exists()
          and (ws / "spec.md").exists() and "-m bench submit" in goal)
    check("workspace creates a session", (ws / "run" / "session.json").exists())


if __name__ == "__main__":
    main()
