"""Benchmark CLI.

Usage (run with Python 3.12+):
    python3.12 -m bench list
    python3.12 -m bench spec TASK
    python3.12 -m bench evaluate TASK PROGRAM.py [--json] [--train-only]
    python3.12 -m bench baseline [TASK ...]
    python3.12 -m bench determinism [TASK ...] [--runs N]

    # sessions: the canonical benchmark record (works with ANY optimizer)
    python3.12 -m bench submit RUN_DIR PROGRAM.py [--task TASK] [--feedback MODE] [--note ...]
    python3.12 -m bench report RUN_DIR [--unseal]
    python3.12 -m bench verify RUN_DIR [--rescore]
    python3.12 -m bench audit RUN_DIR
    python3.12 -m bench workspace TASK DIR [--run-dir RUN_DIR] [--feedback MODE]
"""

import argparse
import hashlib
import json
import os
import shlex
import shutil
import sys
import time
from pathlib import Path

from bench import runner
from bench.session import FEEDBACK_MODES, Session, verify_run

REPO_ROOT = Path(__file__).resolve().parents[1]


def effective_evaluate_train_only(config, requested=False, full=False):
    """Resolve CLI scoring mode without inventing a train objective."""
    if full:
        return False
    allows_train_only = "train-only" in config.get(
        "feedback_modes", FEEDBACK_MODES)
    return bool(requested or (
        config.get("kind") == "generalization" and allows_train_only))

GOAL_TEMPLATE = """\
# Goal: minimize the benchmark score of `program.py`

This directory is a workspace for benchmark task `{task}`; the full
specification is in `spec.md`. Lower score is better. Improve the score
as much as you can through repeated edit-and-submit iterations.

## Submitting

Every candidate you want scored on the record:

    {submit_cmd}

It prints the score, whether the candidate is valid, and whether it is
the best so far. Submit the unmodified `program.py` once first, so the
baseline is on the record. Submissions are append-only and timestamped —
they ARE the benchmark result, so submit real attempts, not noise.
{feedback_note}
## Rules

1. Edit `program.py` in place; keep the required API functions exactly.
2. The program must be deterministic, self-contained, and stdlib-only.
3. Do not modify anything under the benchmark repo or the run directory;
   the only file you edit is `program.py`.
4. HARD RULE: no benchmark gaming. Do not hardcode, precompute, memorize,
   or re-generate answers/data for the specific benchmark inputs (e.g. by
   reading the evaluator's seeds or source). Solutions must be general
   algorithms that would work on fresh data from the same distribution;
   the evaluator checks this on unseen validation data where applicable.

## Self-testing (does not record a submission)

    {selftest_cmd}

It prints one JSON line; the `score` field is what you are judged on
(`ok` must be true). Evaluating takes a few seconds to a few minutes.
Run both commands exactly as written — other flags are off-limits.
"""


def _check_python():
    if sys.version_info < (3, 12):
        sys.exit(
            f"bench requires Python 3.12+ (found {sys.version.split()[0]}); "
            "run e.g. `python3.12 -m bench ...`"
        )


def _print_result(task, result):
    if result["ok"]:
        print(f"{task}: score={result['score']:g}  metrics={json.dumps(result['metrics'])}")
    else:
        print(f"{task}: FAILED — {result['error']}")


def _submit_cmd(run_dir):
    return (f"PYTHONPATH={shlex.quote(str(REPO_ROOT))} "
            f"{shlex.quote(sys.executable)} -m bench "
            f"submit {shlex.quote(str(Path(run_dir).resolve()))} program.py")


def _selftest_cmd(task, feedback):
    # Route self-tests through `bench evaluate` (the same runner code path
    # as submissions), so agent-observed scores are bit-identical to what
    # the harness records. On generalization tasks `bench evaluate`
    # defaults to blind (validation hidden); a full-feedback run passes
    # --full so the agent can see validation. Blind runs pass nothing —
    # the safe default holds even if the agent drops the flag.
    flag = ""
    if runner.load_config(task).get("kind") == "generalization" \
            and feedback == "full":
        flag = " --full"
    return (f"PYTHONPATH={shlex.quote(str(REPO_ROOT))} "
            f"{shlex.quote(sys.executable)} -m bench "
            f"evaluate {task} program.py --json{flag}")


def main():
    _check_python()
    parser = argparse.ArgumentParser(prog="bench")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("list", help="list available tasks and release status")
    p.add_argument("--status", choices=("official", "legacy"),
                   help="show only one release-status group")
    p.add_argument("--names-only", action="store_true",
                   help="print bare task names for scripts")

    p = sub.add_parser("spec", help="print a task's specification")
    p.add_argument("task")

    p = sub.add_parser("evaluate", help="evaluate a program against a task (no record)")
    p.add_argument("task")
    p.add_argument("program")
    p.add_argument("--json", action="store_true",
                   help="print the raw result as one JSON line")
    p.add_argument("--full", action="store_true",
                   help="reveal validation scores on generalization tasks "
                        "(default is blind: validation hidden). Operator use.")
    p.add_argument("--train-only", action="store_true",
                   help="force blind mode (redundant with the default on "
                        "generalization tasks; kept for explicitness)")
    p.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"),
                   help="explicit scoring backend (task support is enforced)")

    p = sub.add_parser("baseline", help="evaluate initial programs")
    p.add_argument("tasks", nargs="*")
    p.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"))

    p = sub.add_parser("determinism", help="run initial programs repeatedly, check identical scores")
    p.add_argument("tasks", nargs="*")
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"))

    p = sub.add_parser(
        "submit",
        help="score a program and append it to a run's submission history")
    p.add_argument("run_dir")
    p.add_argument("program")
    p.add_argument("--task", help="task name (required on first submission)")
    p.add_argument("--feedback", choices=FEEDBACK_MODES,
                   help="information regime, fixed at session creation "
                        "(default: full)")
    p.add_argument("--note", default="", help="free-form label for the record")
    p.add_argument("--json", action="store_true",
                   help="print the visible result as JSON")
    p.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"))

    p = sub.add_parser("report", help="print a run's submission history")
    p.add_argument("run_dir")
    p.add_argument("--unseal", action="store_true",
                   help="also print hidden metrics (held-out scores). "
                        "Experimenter use only — never mid-run where an "
                        "optimizing agent could see the output.")

    p = sub.add_parser("verify", help="check a run's history for integrity")
    p.add_argument("run_dir")
    p.add_argument("--rescore", action="store_true",
                   help="also re-score every submission (deterministic "
                        "scoring must reproduce each record exactly)")

    p = sub.add_parser(
        "workspace",
        help="set up an agent-facing workspace + session for a task "
             "(for goal-mode / any external optimizer)")
    p.add_argument("task")
    p.add_argument("dir")
    p.add_argument("--run-dir", default=None,
                   help="where the session lives (default: DIR/run). For "
                        "hidden-information experiments put this outside "
                        "the agent's reach.")
    p.add_argument("--feedback", default="full", choices=FEEDBACK_MODES)
    p.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"))

    p = sub.add_parser("calibrate",
                       help="measure host local-compute rate, pick concurrency")
    p.add_argument("--json", action="store_true")
    p.add_argument("--repeats", type=int, default=5)

    p = sub.add_parser("trace",
                       help="print a run's unified grading trace "
                            "(best-so-far vs wall/gradings, model/local split)")
    p.add_argument("run_dir")
    p.add_argument("--rescale-to", default=None, metavar="PROFILE.json",
                   help="rescale local time onto a reference machine profile")
    p.add_argument("--csv", action="store_true", help="emit CSV for plotting")

    p = sub.add_parser("audit",
                       help="scan a run's recorded submission sources for "
                            "escape-gadget signatures + implausible scores "
                            "(detect non-cooperative agents)")
    p.add_argument("run_dir")

    args = parser.parse_args()

    if args.cmd == "list":
        for t in runner.list_tasks(args.status):
            print(t if args.names_only else f"{runner.task_status(t)}\t{t}")

    elif args.cmd == "spec":
        print(runner.read_spec(args.task))

    elif args.cmd == "evaluate":
        # Blind-by-default when a generalization task actually supports a
        # train-only objective.  Calibration-only tasks explicitly advertise
        # full feedback because they have no scored training split; for those,
        # bare evaluation returns the task's 64-row validation objective.
        config = runner.load_config(args.task)
        train_only = effective_evaluate_train_only(
            config, requested=args.train_only, full=args.full)
        result = runner.evaluate(args.task, args.program,
                                 train_only=train_only, device=args.device)
        # Passive eval telemetry: when TEXTOPT_EVAL_LOG is set (the loop
        # points it into the agent workspace), record every evaluation —
        # not just end-of-iteration submissions — with the exact program
        # bytes. Contains only what this command printed anyway, and must
        # never interfere with scoring.
        log_path = os.environ.get("TEXTOPT_EVAL_LOG")
        if log_path:
            try:
                data = Path(args.program).read_bytes()
                sha = hashlib.sha256(data).hexdigest()
                # ts is the grading's START time (consistent with session
                # submissions), so bench/trace.py can place each grading and
                # add its own duration to get the eval-end elapsed axis.
                start_ts = time.time() - (result.get("eval_wall_seconds") or 0.0)
                rec = {"ts": round(start_ts, 3), "task": args.task,
                       "program_sha256": sha,
                       "train_only": bool(train_only),  # effective mode
                       "ok": result["ok"], "score": result["score"],
                       "metrics": result.get("metrics") or {},
                       "eval_wall_seconds": result.get("eval_wall_seconds"),
                       "eval_cpu_seconds": result.get("eval_cpu_seconds"),
                       "eval_queue_seconds": result.get("eval_queue_seconds"),
                       "error": result.get("error")}
                with open(log_path, "a") as f:
                    f.write(json.dumps(rec) + "\n")
                progs = Path(log_path).parent / "eval_programs"
                progs.mkdir(exist_ok=True)
                snap = progs / f"{sha[:16]}.py"
                if not snap.exists():
                    snap.write_bytes(data)
            except OSError:
                pass
        if args.json:
            print(json.dumps(result))
        else:
            _print_result(args.task, result)
        sys.exit(0 if result["ok"] else 1)

    elif args.cmd == "baseline":
        tasks = args.tasks or runner.default_tasks()
        for t in tasks:
            _print_result(t, runner.evaluate(
                t, runner.initial_program(t), device=args.device))

    elif args.cmd == "determinism":
        tasks = args.tasks or runner.default_tasks()
        all_ok = True
        for t in tasks:
            # Tasks are bit-exact by default (tolerance 0). A task may
            # declare score_tolerance in config.json if its metric is
            # low-variance rather than bit-exact (per the "low variance OR
            # deterministic" design) — e.g. mem_infer's peak has a ~60-byte
            # pymalloc arena flicker. Within tolerance counts as passing.
            tol = runner.load_config(t).get("score_tolerance", 0)
            scores = []
            failed = False
            for _ in range(args.runs):
                r = runner.evaluate(
                    t, runner.initial_program(t), device=args.device)
                if r["ok"]:
                    scores.append(r["score"])
                else:
                    scores.append(f"FAIL({r['error'][:80]})")
                    failed = True
            if failed:
                status, ok = "NOT DETERMINISTIC", False
            else:
                spread = max(scores) - min(scores)
                if spread == 0:
                    status, ok = "DETERMINISTIC", True
                elif spread <= tol:
                    status, ok = f"LOW-VARIANCE (spread {spread:g} <= {tol})", True
                else:
                    status, ok = "NOT DETERMINISTIC", False
            all_ok &= ok
            print(f"{t}: {status}  scores={scores}")
        sys.exit(0 if all_ok else 1)

    elif args.cmd == "submit":
        try:
            session = Session.open_or_create(
                args.run_dir, task=args.task, feedback=args.feedback,
                device=args.device)
            rec = session.submit(args.program, note=args.note)
        except (ValueError, FileExistsError, OSError) as e:
            sys.exit(f"submit failed: {e}")
        vis = session.visible(rec)
        if args.json:
            print(json.dumps(vis, indent=2))
        elif vis["ok"]:
            verdict = ("NEW BEST" if vis["best"]
                       else f"not better (best {vis['best_score']:g})")
            print(f"submission #{vis['n']}: score={vis['score']:g}  "
                  f"{verdict}  metrics={json.dumps(vis['metrics'])}")
        else:
            err = " ".join(str(vis["error"]).split())[:400]
            print(f"submission #{vis['n']}: INVALID — {err}")
        sys.exit(0 if vis["ok"] else 1)

    elif args.cmd == "report":
        try:
            session = Session.open(args.run_dir)
        except (OSError, ValueError, KeyError) as e:
            sys.exit(f"cannot open session in {args.run_dir}: {e}")
        if args.unseal and os.environ.get("TEXTOPT_UNSEAL") != "1":
            # --unseal prints held-out (validation/test) scores. Gate it
            # behind an explicit operator env var so it cannot be run
            # casually from inside an agent's workspace to leak the very
            # signal a blind/generalization run is meant to hide.
            sys.exit("refusing to --unseal without TEXTOPT_UNSEAL=1 "
                     "(operator-only: reveals held-out scores)")
        print(f"# task={session.task} feedback={session.feedback} "
              f"run={Path(args.run_dir).resolve()}")
        print(f"{'n':>4}  {'time':19}  {'+dt':>9}  {'score':>14}  "
              f"{'':9}  note")
        for rec in session.records:
            dt = "" if rec["dt"] is None else f"+{rec['dt']:.1f}s"
            if rec["ok"]:
                score = f"{rec['guide_score']:g}"
                mark = "BEST" if rec["best"] else ""
            else:
                score = "-"
                mark = "INVALID"
            extra = ""
            if args.unseal and rec["ok"]:
                extra = "  " + json.dumps(session.full_result(rec)["metrics"])
            print(f"{rec['n']:>4}  {rec['time']:19}  {dt:>9}  {score:>14}  "
                  f"{mark:9}  {rec['note']}{extra}")
        s = session.summary()
        best = "n/a" if s["best_score"] is None else f"{s['best_score']:g}"
        print(f"# {s['submissions']} submissions ({s['valid']} valid) over "
              f"{s['span_seconds']:.1f}s; best score {best}"
              + ("" if s["best_n"] is None else f" at #{s['best_n']}"))

    elif args.cmd == "verify":
        problems = verify_run(args.run_dir, rescore=args.rescore)
        if problems:
            for msg in problems:
                print(f"PROBLEM: {msg}")
            sys.exit(1)
        try:
            n = len(Session.open(args.run_dir).records)
        except (OSError, ValueError, KeyError) as e:
            sys.exit(f"cannot open session in {args.run_dir}: {e}")
        mode = "hash chain, snapshots, best flags" + \
            (", re-scored" if args.rescore else "")
        print(f"OK: {n} submission(s) intact ({mode})")

    elif args.cmd == "workspace":
        ws = Path(args.dir)
        if (ws / "program.py").exists():
            sys.exit(f"workspace already set up: {ws / 'program.py'} exists")
        run_dir = Path(args.run_dir) if args.run_dir else ws / "run"
        ws.mkdir(parents=True, exist_ok=True)
        try:
            session = Session.open_or_create(
                run_dir, task=args.task, feedback=args.feedback,
                device=args.device)
        except (ValueError, OSError, KeyError) as e:
            sys.exit(f"cannot set up session in {run_dir}: {e}")
        shutil.copyfile(runner.initial_program(args.task), ws / "program.py")
        (ws / "spec.md").write_text(runner.read_spec(args.task))
        feedback_note = ""
        if runner.load_config(args.task).get("kind") == "generalization":
            if session.feedback == "train-only":
                feedback_note = (
                    "\nThis task scores generalization, and this session is "
                    "blind: the score you see is computed on the visible "
                    "train data only, while hidden validation and test "
                    "scores are recorded sealed and never shown to you. "
                    "Programs overfit to the train data will record poorly.\n")
            else:
                feedback_note = (
                    "\nThis task scores generalization: reported scores come "
                    "from data you cannot see. A held-out test split is "
                    "never reported at all.\n")
        (ws / "GOAL.md").write_text(GOAL_TEMPLATE.format(
            task=args.task,
            submit_cmd=_submit_cmd(run_dir),
            feedback_note=feedback_note,
            selftest_cmd=_selftest_cmd(args.task, session.feedback),
        ))
        print(f"workspace ready: {ws}")
        print(f"  program.py  — baseline program to optimize")
        print(f"  spec.md     — task specification")
        print(f"  GOAL.md     — goal + submit instructions for the agent")
        print(f"session (submission record): {run_dir.resolve()}")
        print(f"submit command:\n  {_submit_cmd(run_dir)}")
        # For generalization tasks the session holds held-out scores. They
        # are sealed (obfuscated), but the seal is casual-leak protection,
        # not encryption. If the run dir sits inside the agent's workspace
        # (the default), a determined agent could decode it — so for a real
        # blind experiment put the session out of the agent's reach.
        if runner.load_config(args.task).get("kind") == "generalization" \
                and args.run_dir is None:
            print("\nNOTE: this generalization run dir is inside the agent "
                  "workspace. Its held-out scores are only sealed, not "
                  "hidden. For a rigorous blind experiment pass --run-dir "
                  "pointing outside the agent's reach.")

    elif args.cmd == "calibrate":
        from bench import calibrate
        prof = calibrate.machine_profile(args.repeats)
        if args.json:
            print(json.dumps(prof, indent=2))
        else:
            print(f"host rate      : {prof['rate']:,.0f} units/CPU-s "
                  f"(speed_factor {prof['speed_factor']}x vs reference)")
            print(f"meets floor    : {prof['meets_floor']} "
                  f"(floor {prof['floor_rate']:,.0f})")
            print(f"cores          : {prof['physical_cores']} physical / "
                  f"{prof['logical_cores']} logical")
            print(f"concurrency    : {prof['recommended_concurrency']} streams")
            if not prof["meets_floor"]:
                print("WARNING: below reasonable-machine floor; traces may "
                      "not rescale reliably.")

    elif args.cmd == "trace":
        from bench import trace as trace_mod
        ref = json.loads(Path(args.rescale_to).read_text()) if args.rescale_to else None
        trace_mod.print_trace(args.run_dir, reference_profile=ref, csv=args.csv)

    elif args.cmd == "audit":
        from bench import audit
        target = Path(args.run_dir)
        if target.is_file():
            # Single-program spot-check: scan just this source.
            hits = audit.scan_source(target.read_text(errors="replace"))
            print(f"# audit {target}")
            if hits:
                sigs = sorted({h[0] for h in hits})
                print(f"SUSPICIOUS ({len(hits)} hit(s)): {', '.join(sigs)}")
                for name, ln, snippet in hits[:12]:
                    print(f"    L{ln}: {snippet}")
                print("(escape-gadget hits are near-certain cheats; memorization/"
                      "packed-table hits are advisory — spot-check by hand.)")
            else:
                print("no known escape-gadget or memorization signatures found.")
                print("(clean != safe: a novel obfuscation could evade this.)")
            sys.exit(1 if hits else 0)
        result = audit.audit_run(args.run_dir)
        report, flagged = audit.format_report(args.run_dir, result)
        print(report)
        sys.exit(1 if flagged else 0)


if __name__ == "__main__":
    main()
