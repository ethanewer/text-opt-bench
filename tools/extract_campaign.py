"""Extract a durable dataset + report from a benchmark campaign.

Reads every runs/<task>/<prefix>*/ run, merges each run's gradings
(harness submissions from submissions.jsonl + agent self-tests from
iter_*/evals.jsonl) into a single time-ordered trace, and derives:

  - best-score-so-far vs grading-index and vs wall-seconds
  - per-grading local eval cost and the model/local wall split
  - convergence: gradings/seconds to reach within eps of the run's best
  - per-task aggregates across the 5 runs

Writes runs/_campaign/dataset.json (machine-readable, full traces) and
runs/_campaign/report.md (human summary). Read-only over run dirs.

Usage: python3.12 tools/extract_campaign.py [--prefix 5x-]
"""

import argparse
import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bench import trace as trace_mod  # canonical trace assembly

EXPECTED = {"easy_word_problems": 10}
DEFAULT_ITERS = 15


def build_trace(run_dir):
    """Unified grading trace (delegates to the canonical bench.trace)."""
    t = trace_mod.build_trace(run_dir, speed_factor=1.0)
    return t or None


def convergence(trace, eps):
    """gradings and wall-seconds to first reach within eps of final best."""
    finals = [p["best"] for p in trace if p["best"] is not None]
    if not finals:
        return None, None
    final_best = finals[-1]
    if final_best == 0:
        target = 0.0
    else:
        target = final_best * (1 + eps) if final_best > 0 else final_best * (1 - eps)
    for p in trace:
        if p["best"] is not None and p["best"] <= target:
            return p["i"] + 1, p["wall"]
    return len(trace), trace[-1]["wall"]


def summarize_run(run_dir):
    trace = build_trace(run_dir)
    if not trace:
        return None
    valid = [p for p in trace if p["ok"] and p["score"] is not None]
    final_best = valid[-1]["best"] if valid else None
    g99, s99 = convergence(trace, 0.01)
    total_wall = trace[-1]["wall"]
    total_local = trace[-1]["cum_local"]
    return {
        "run": run_dir.name,
        "gradings": len(trace),
        "submits": sum(1 for p in trace if p["kind"] == "submit"),
        "selftests": sum(1 for p in trace if p["kind"] == "selftest"),
        "final_best": final_best,
        "wall_seconds": round(total_wall, 1),
        "local_seconds": round(total_local, 1),
        "local_frac": round(total_local / total_wall, 3) if total_wall else None,
        "gradings_to_1pct": g99,
        "seconds_to_1pct": round(s99, 1) if s99 is not None else None,
        "has_telemetry": any(p["kind"] == "selftest" for p in trace),
        "trace": trace,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="5x-")
    args = ap.parse_args()

    dataset = {"prefix": args.prefix, "tasks": {}}
    for task_dir in sorted((ROOT / "runs").glob("*")):
        if not task_dir.is_dir() or task_dir.name.startswith("_"):
            continue
        runs = []
        for run_dir in sorted(task_dir.glob(f"{args.prefix}*")):
            s = summarize_run(run_dir)
            if s:
                runs.append(s)
        if runs:
            dataset["tasks"][task_dir.name] = runs

    out_dir = ROOT / "runs" / "_campaign"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "dataset.json").write_text(json.dumps(dataset, indent=1))

    # Human report
    lines = ["# Campaign dataset summary\n"]
    lines.append(f"prefix: `{args.prefix}`  |  tasks: {len(dataset['tasks'])}  "
                 f"|  runs: {sum(len(v) for v in dataset['tasks'].values())}\n")
    hdr = (f"| task | runs | best (min) | median gradings | median g→1% | "
           f"median local% | median wall(s) |")
    lines += [hdr, "|" + "---|" * 7]
    for task in sorted(dataset["tasks"]):
        rs = dataset["tasks"][task]
        bests = [r["final_best"] for r in rs if r["final_best"] is not None]
        grd = [r["gradings"] for r in rs]
        g1 = [r["gradings_to_1pct"] for r in rs if r["gradings_to_1pct"]]
        lf = [r["local_frac"] for r in rs if r["local_frac"] is not None]
        wl = [r["wall_seconds"] for r in rs]
        def m(xs, f="{:.0f}"):
            return f.format(st.median(xs)) if xs else "-"
        best = f"{min(bests):g}" if bests else "-"
        lines.append(f"| {task} | {len(rs)} | {best} | {m(grd)} | {m(g1)} | "
                     f"{m(lf,'{:.0%}') if lf else '-'} | {m(wl)} |")
    (out_dir / "report.md").write_text("\n".join(lines) + "\n")

    print(f"wrote {out_dir/'dataset.json'} and {out_dir/'report.md'}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
