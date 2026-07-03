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
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bench import runner
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
    run_dir = tmp / "run"
    baseline = runner.initial_program("mem_kv")
    solution = ROOT / "tests" / "solutions" / "mem_kv.py"
    broken = tmp / "broken.py"
    broken.write_text("def store(pairs:\n")  # syntax error

    # -- lifecycle: baseline, improvement, worse resubmit, invalid ------
    s = Session.create(run_dir, "mem_kv")
    r0 = s.submit(baseline, note="baseline")
    check("baseline recorded ok", r0["ok"] and r0["best"] and r0["n"] == 0)
    check("first dt is None", r0["dt"] is None)
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
        Session.open_or_create(run_dir, task="compress")
        check("task mismatch rejected", False)
    except ValueError:
        check("task mismatch rejected", True)
    try:
        Session.open_or_create(run_dir, task="mem_kv", feedback="train-only")
        check("feedback change rejected", False)
    except ValueError:
        check("feedback change rejected", True)

    # -- verify: intact, then every tampering mode ----------------------
    check("verify passes on intact run", verify_run(run_dir) == [])
    check("verify --rescore reproduces the record",
          verify_run(run_dir, rescore=True) == [])

    jsonl = run_dir / "submissions.jsonl"
    original = jsonl.read_text()
    lines = original.splitlines()

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
    check("full_result unseals held-out splits",
          "test_score" in full["metrics"] and "val_score" in full["metrics"]
          and full["score"] == full["metrics"]["val_score"])
    check("guide score is the train score in train-only mode",
          gr["guide_score"] == full["metrics"]["train_score"])
    raw = (tmp / "run_blind" / "submissions.jsonl").read_text()
    check("hidden keys never appear in plaintext on disk",
          "test_score" not in raw and "val_score" not in raw)
    check("blind-session rescore reproduces sealed record",
          verify_run(tmp / "run_blind", rescore=True) == [])
    check("visible_metrics full mode hides only test",
          "val_score" in visible_metrics(full["metrics"], "full")
          and "test_score" not in visible_metrics(full["metrics"], "full"))

    # -- CLI round trip ---------------------------------------------------
    cli_run = tmp / "cli_run"
    def bench(*a):
        return subprocess.run(
            [sys.executable, "-m", "bench", *a],
            capture_output=True, text=True, cwd=ROOT)
    sub = bench("submit", str(cli_run), str(baseline), "--task", "mem_kv",
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
    check("CLI verify passes", ver.returncode == 0 and "OK" in ver.stdout)

    # -- workspace command (goal-mode setup) ------------------------------
    ws = tmp / "ws"
    wsp = bench("workspace", "mem_kv", str(ws))
    goal = (ws / "GOAL.md").read_text() if (ws / "GOAL.md").exists() else ""
    check("workspace materializes program/spec/GOAL",
          wsp.returncode == 0 and (ws / "program.py").exists()
          and (ws / "spec.md").exists() and "-m bench submit" in goal)
    check("workspace creates a session", (ws / "run" / "session.json").exists())


if __name__ == "__main__":
    main()
