"""Minimal AutoResearch-style text-optimization loop.

This is the *default optimizer* shipped with the benchmark, not part of
it: the benchmark's interface is `bench.session.Session` (submit a
program, get a score, everything recorded), and any other optimization
algorithm can drive it the same way. This loop is a greedy hill-climb:
each iteration hands the current best program to a codex CLI agent with
the task spec, score, and full attempt history (as a browsable git
clone — see loop/history.py); the agent edits program.py in a scratch
workspace; the harness submits the result to the session; strictly
better programs are accepted as the new best.

Usage:
    python3.12 -m loop.optimize --task mem_kv --iterations 10
"""

import argparse
import datetime
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from bench import calibrate, runner
from bench.session import Session, visible_metrics
from loop.history import HistoryRepo

PROMPT_TEMPLATE = """\
You are one step in an automated optimization loop. Your ONLY goal is to
reduce the numeric score of the Python program in `program.py` (in the
current directory). Lower score is better.

{spec}

## Current state

- Current best score: {score}
- Current best metrics: {metrics}
- `program.py` currently contains the best program found so far.

## Recent attempt history (most recent last)

{history}
{git_section}
## What to do

1. Edit `program.py` IN PLACE. Keep the required API functions exactly.
2. Make a focused improvement; a working small win beats a broken rewrite.
   If a previous attempt failed, avoid repeating its mistake.
3. You can score your candidate exactly as the harness will:
       PYTHONPATH={repo_root} python3.12 -m bench evaluate {task} program.py --json{eval_flag}
   It prints one JSON line; the `score` field is what you are judged on
   (`ok` must be true). Evaluating takes a few seconds to a few minutes.
   Run it exactly as written — other flags are off-limits.
4. Do not create or modify any other file. Do not touch the harness or
   benchmark files. The program must be deterministic and self-contained.
5. HARD RULE: no benchmark gaming. Do not hardcode, precompute, memorize,
   or re-generate answers/data for the specific benchmark inputs (e.g. by
   reading the evaluator's seeds or source). Solutions must be general
   algorithms that would work on fresh data from the same distribution;
   the evaluator checks this on unseen validation data where applicable.
"""


GIT_SECTION = """
## Full attempt history (git)

This directory is a git clone carrying the complete history of this
optimization run. `main` is the lineage of accepted improvements;
`origin/attempts/iter-*` are rejected or invalid attempts. Commit
messages carry each attempt's score, metrics, or failure reason.

    git log --oneline --all          # every attempt, one line each
    git log -p origin/main           # accepted improvements as diffs
    git show origin/attempts/iter-004  # what a failed attempt tried
{attempt_hint}
Check the history before trying an idea — failed attempts show exactly
what did not work and why. Do NOT commit, push, tag, or otherwise change
git state: the harness records your attempt automatically after scoring,
and your clone is discarded afterwards (origin is read-only to you).
"""


def render_history(history, limit=8):
    if not history:
        return "(none yet — this is the first attempt)"
    lines = []
    for h in history[-limit:]:
        if h["ok"]:
            verdict = "ACCEPTED (new best)" if h["accepted"] else "rejected (not better)"
            lines.append(f"- iter {h['iter']}: score {h['guide_score']:g} — {verdict}")
        else:
            err = " ".join(str(h["error"]).split())[:300]
            lines.append(f"- iter {h['iter']}: INVALID — {err}")
    return "\n".join(lines)


def run_codex(prompt, workspace, model, effort, timeout):
    cmd = [
        "codex", "exec",
        "--model", model,
        "-c", f"model_reasoning_effort={effort}",
        "--sandbox", "workspace-write",
        "--skip-git-repo-check",
        "--cd", str(workspace),
        "--output-last-message", str(workspace / "codex_last_message.txt"),
        "--color", "never",
        prompt,
    ]
    # Record every self-evaluation the agent runs (bench/cli.py appends to
    # TEXTOPT_EVAL_LOG), so discarded candidates are preserved too. The
    # workspace persists under the run dir, keeping the full record.
    env = dict(os.environ,
               TEXTOPT_EVAL_LOG=str((workspace / "evals.jsonl").resolve()))
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
            cwd=workspace, env=env,
        )
    except subprocess.TimeoutExpired:
        return f"codex timed out after {timeout}s"
    (workspace / "codex_stdout.txt").write_text(proc.stdout or "")
    (workspace / "codex_stderr.txt").write_text(proc.stderr or "")
    if proc.returncode != 0:
        return f"codex exited {proc.returncode}: {(proc.stderr or '')[-400:]}"
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", required=True, choices=runner.list_tasks())
    ap.add_argument("--iterations", type=int, default=10)
    ap.add_argument("--model", default="gpt-5.5")
    ap.add_argument("--effort", default="low",
                    choices=["none", "minimal", "low", "medium", "high"])
    ap.add_argument("--feedback", default="full", choices=["full", "train-only"],
                    help="what the agent sees on generalization tasks: "
                         "'full' = train + validation scores; 'train-only' = "
                         "train score only (selection also uses it)")
    ap.add_argument("--codex-timeout", type=int, default=900)
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--start-program", default=None,
                    help="start from this program instead of the task baseline")
    args = ap.parse_args()

    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    config_tag = f"{args.model}-{args.effort}"
    run_dir = Path(args.run_dir or REPO_ROOT / "runs" / args.task / f"{stamp}-{config_tag}")
    session = Session.open_or_create(run_dir, task=args.task,
                                     feedback=args.feedback)
    log_path = run_dir / "log.jsonl"

    # Record the machine this run executed on, so its wall-clock trace can
    # be rescaled onto a reference machine later (bench/trace.py). Cheap
    # (~1s) and one-time; skipped on resume if already present.
    prof_path = run_dir / "machine_profile.json"
    if not prof_path.exists():
        try:
            prof_path.write_text(json.dumps(calibrate.machine_profile(), indent=1))
        except Exception as e:
            print(f"[loop] calibration skipped: {e}")

    def log(entry):
        with open(log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    spec = runner.read_spec(args.task)
    generalization = runner.load_config(args.task).get("kind") == "generalization"

    start = Path(args.start_program or runner.initial_program(args.task))
    best_source = start.read_text()
    print(f"[loop] task={args.task} model={args.model} effort={args.effort} "
          f"feedback={args.feedback}")
    print(f"[loop] run dir: {run_dir}")
    print("[loop] scoring baseline ...", flush=True)
    baseline = session.submit(start, note="loop baseline")
    if not baseline["ok"]:
        sys.exit(f"baseline program is invalid: {baseline['error']}")
    best_guide = baseline["guide_score"]
    best_metrics = baseline["metrics"]
    if session.best["n"] != baseline["n"]:
        # Resumed into an existing session whose best beats our start
        # program: adopt it, so greedy acceptance (rec["best"]) and the
        # loop's notion of "current best" stay consistent.
        best_source = (run_dir / "best_program.py").read_text()
        best_guide = session.best["guide_score"]
        best_metrics = session.best["metrics"]
        print(f"[loop] resuming session: best so far {best_guide:g} "
              f"(submission #{session.best['n']})")
    print(f"[loop] baseline score: {best_guide:g}")
    log({"iter": 0, "event": "baseline", "task": args.task,
         "model": args.model, "effort": args.effort,
         "feedback": args.feedback, "score": baseline["score"],
         "guide_score": best_guide, "metrics": best_metrics,
         "time": datetime.datetime.now().isoformat()})

    hist = HistoryRepo(run_dir)
    if hist.exists():
        # Resumed run: never re-init (that would orphan the recorded
        # lineage). main already holds the session's best program; only a
        # strictly better --start-program needs recording.
        if baseline["best"]:
            hist.record(
                best_source,
                f"resume: new baseline, score {best_guide:g} ACCEPTED\n\n"
                f"metrics: {json.dumps(visible_metrics(best_metrics, args.feedback))}\n",
                0, True)
    else:
        hist.init(
            best_source,
            f"iter 0: baseline, score {best_guide:g}\n\n"
            f"metrics: {json.dumps(visible_metrics(best_metrics, args.feedback))}\n",
        )

    # Iteration numbers continue across resumed invocations so attempts/
    # refs and workspace dirs never collide with earlier ones.
    prior_iters = [int(r["note"].rsplit(" ", 1)[1]) for r in session.records
                   if str(r.get("note", "")).startswith("loop iter ")]
    for ref in hist.attempt_refs():
        prior_iters.append(int(ref.rsplit("-", 1)[1]))
    first_iter = max(prior_iters, default=0) + 1
    failed_refs = [f"origin/{ref}" for ref in hist.attempt_refs()]

    history = []
    for i in range(first_iter, first_iter + args.iterations):
        t_start = time.monotonic()
        ws = run_dir / f"iter_{i:03d}"
        hist.clone_workspace(ws)
        (ws / "program.py").write_text(best_source)

        if failed_refs:
            attempt_hint = (
                "\nBEFORE writing code, review the failed attempts "
                f"({', '.join(failed_refs[-4:])}) with `git show <ref>` or "
                "`git diff origin/main <ref>` so you do not repeat them.\n"
            )
        else:
            attempt_hint = ""
        prompt = PROMPT_TEMPLATE.format(
            spec=spec,
            score=f"{best_guide:g}",
            metrics=json.dumps(visible_metrics(best_metrics, args.feedback)),
            history=render_history(history),
            git_section=(GIT_SECTION.format(attempt_hint=attempt_hint)
                         if hist.enabled else ""),
            repo_root=REPO_ROOT,
            task=args.task,
            # `bench evaluate` is blind by default on generalization tasks;
            # a full-feedback run passes --full so the agent may see
            # validation. Blind runs pass nothing — the safe default stands
            # even if the agent omits the flag.
            eval_flag=(" --full"
                       if generalization and args.feedback == "full"
                       else ""),
        )
        (ws / "PROMPT.md").write_text(prompt)

        print(f"[iter {i}] codex editing ...", flush=True)
        codex_error = run_codex(prompt, ws, args.model, args.effort,
                                args.codex_timeout)

        entry = {"iter": i, "ok": False, "score": None, "guide_score": None,
                 "error": None, "accepted": False}
        try:
            # errors="replace": an undecodable file still gets recorded in
            # history and reported, instead of crashing the loop.
            candidate = (ws / "program.py").read_text(errors="replace")
        except OSError:
            candidate = None
        if candidate is None:
            entry["error"] = "program.py was deleted or is unreadable"
            candidate = best_source
        elif codex_error is not None:
            entry["error"] = codex_error
        elif candidate == best_source:
            entry["error"] = "codex made no change to program.py"
        else:
            # The submission IS the benchmark record; the loop only decides
            # what to do with the result (greedy accept).
            rec = session.submit(ws / "program.py", note=f"loop iter {i}")
            entry.update(ok=rec["ok"], score=rec["score"],
                         guide_score=rec["guide_score"], error=rec["error"],
                         metrics=rec["metrics"], accepted=rec["best"])
            if rec["best"]:
                best_source = candidate
                best_guide = rec["guide_score"]
                best_metrics = rec["metrics"]

        if entry["accepted"]:
            print(f"[iter {i}] score {entry['guide_score']:g}  <-- new best")
        elif entry["ok"]:
            print(f"[iter {i}] score {entry['guide_score']:g}  "
                  f"(rejected, best {best_guide:g})")
        else:
            print(f"[iter {i}] invalid: {' '.join(str(entry['error']).split())[:160]}")
        entry["seconds"] = round(time.monotonic() - t_start, 1)
        entry["best_guide_score"] = best_guide

        # Record every attempt in the loop's git history (a no-change or
        # invalid attempt becomes an empty-diff commit whose message says
        # what went wrong — the record stays complete). Messages respect
        # the feedback mode: the agent reads them.
        agent_note = ""
        note_file = ws / "codex_last_message.txt"
        if note_file.exists():
            agent_note = " ".join(
                note_file.read_text(errors="replace").split()
            )[:300]
        if entry["accepted"]:
            subject = f"iter {i}: score {entry['guide_score']:g} ACCEPTED"
            detail = (f"metrics: "
                      f"{json.dumps(visible_metrics(entry.get('metrics'), args.feedback))}")
        elif entry["ok"]:
            subject = (f"iter {i}: score {entry['guide_score']:g} "
                       f"rejected (best {best_guide:g})")
            detail = (f"metrics: "
                      f"{json.dumps(visible_metrics(entry.get('metrics'), args.feedback))}")
        else:
            subject = f"iter {i}: INVALID"
            detail = "error: " + " ".join(str(entry["error"]).split())[:400]
        message = subject + "\n\n" + detail
        if agent_note:
            message += f"\nagent: {agent_note}"
        hist.record(candidate, message + "\n", i, entry["accepted"])
        if not entry["accepted"]:
            failed_refs.append(f"origin/attempts/iter-{i:03d}")

        log(entry)
        history.append(entry)

    baseline_guide = baseline["guide_score"]
    improvement = baseline_guide / best_guide if best_guide else float("inf")
    print(f"[loop] done. baseline {baseline_guide:g} -> best {best_guide:g} "
          f"({improvement:.3f}x better)")
    if generalization:
        # Full splits (incl. held-out test) go to the operator's stdout
        # only; on disk they exist solely inside the session's sealed
        # fields, where a mid-run agent reading the run dir can't casually
        # see them.
        print(f"[loop] final best-program splits: "
              f"{json.dumps(session.full_result(session.best)['metrics'])}")
        log({"event": "final", "best_n": session.best["n"]})
    print(f"[loop] best program: {run_dir / 'best_program.py'}")
    print(f"[loop] submission record: {run_dir / 'submissions.jsonl'} "
          f"(python3.12 -m bench report {run_dir})")


if __name__ == "__main__":
    main()
