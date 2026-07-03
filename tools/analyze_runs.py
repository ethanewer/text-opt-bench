"""Summarize optimization runs from runs/*/*/log.jsonl.

Usage:  python3.12 tools/analyze_runs.py [--task TASK]

Prints, per run: config, baseline, the best-score trajectory across
iterations, acceptance/invalid counts, and (for generalization tasks)
train/val/test split scores to expose overfitting.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bench.session import Session


def load_splits(run_dir):
    """Map loop iteration -> full metrics (unsealed) from the run's session.

    New runs keep hidden split scores sealed inside submissions.jsonl;
    old runs (no session.json) fall back to metrics in log.jsonl entries.
    """
    try:
        session = Session.open(run_dir)
    except (OSError, json.JSONDecodeError, KeyError):
        return None, None
    by_iter = {}
    for rec in session.records:
        note = rec.get("note", "")
        if rec["ok"] and note.startswith("loop iter "):
            by_iter[int(note.rsplit(" ", 1)[1])] = session.full_result(rec)["metrics"]
        elif rec["ok"] and note.startswith("loop baseline"):
            by_iter[0] = session.full_result(rec)["metrics"]
    best_full = session.full_result(session.best)["metrics"] if session.best else None
    return by_iter, best_full


def load_runs(task_filter=None):
    runs = []
    for log_path in sorted(ROOT.glob("runs/*/*/log.jsonl")):
        task = log_path.parent.parent.name
        if task_filter and task != task_filter:
            continue
        entries = [json.loads(l) for l in log_path.read_text().splitlines() if l.strip()]
        if not entries:
            continue
        runs.append((task, log_path.parent.name, entries))
    return runs


def fmt(x):
    if x is None:
        return "-"
    if isinstance(x, float) and x != int(x):
        return f"{x:g}"
    return f"{int(x):,}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default=None)
    args = ap.parse_args()

    for task, run_name, entries in load_runs(args.task):
        base = next((e for e in entries if e.get("event") == "baseline"), None)
        iters = [e for e in entries if "accepted" in e]
        final = next((e for e in entries if e.get("event") == "final"), None)
        if base is None:
            continue
        model = base.get("model", "?")
        effort = base.get("effort", "?")
        feedback = base.get("feedback", "-")
        session_splits, session_final = load_splits(log_path.parent)
        print(f"\n== {task} | {run_name} | {model}/{effort} | feedback={feedback}")
        base_guide = base.get("guide_score", base.get("score"))
        print(f"   baseline: {fmt(base_guide)}")
        best = base_guide
        for e in iters:
            gs = e.get("guide_score", e.get("score"))
            if e["accepted"]:
                mark, best = "ACCEPT", gs
            elif e["ok"]:
                mark = "reject"
            else:
                mark = "INVALID"
            err = ""
            if not e["ok"]:
                err = " | " + " ".join(str(e.get("error", "")).split())[:80]
            m = e.get("metrics") or {}
            if session_splits and e["iter"] in session_splits:
                m = session_splits[e["iter"]]
            splits = ""
            if "val_score" in m or "test_score" in m:
                splits = (f" | train={fmt(m.get('train_score'))}"
                          f" val={fmt(m.get('val_score'))}"
                          f" test={fmt(m.get('test_score'))}")
            print(f"   iter {e['iter']:>2}: {fmt(gs):>12} {mark:7}"
                  f" best={fmt(best):>12} ({e.get('seconds', '?')}s){splits}{err}")
        n_acc = sum(1 for e in iters if e["accepted"])
        n_inv = sum(1 for e in iters if not e["ok"])
        impr = (base_guide / best) if best else float("inf")
        print(f"   summary: {n_acc}/{len(iters)} accepted, {n_inv} invalid, "
              f"improvement {impr:.3f}x")
        if session_final and ("val_score" in session_final
                              or "test_score" in session_final):
            print(f"   final splits: {json.dumps(session_final)}")
        elif final and "metrics" in final:
            print(f"   final splits: {json.dumps(final['metrics'])}")


if __name__ == "__main__":
    main()
