#!/usr/bin/env python3
"""Measure validation/test overfitting across complete LFM2.5 trajectories."""

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench.session import _unseal  # noqa: E402

TASK = "slm_weight_compression_lfm25"


def _mean(values):
    return sum(values) / len(values) if values else None


def _ranks(values):
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + 1 + end) / 2
        for index in order[start:end]:
            ranks[index] = rank
        start = end
    return ranks


def _pearson(xs, ys):
    if len(xs) < 2:
        return None
    xm, ym = _mean(xs), _mean(ys)
    dx = [value - xm for value in xs]
    dy = [value - ym for value in ys]
    denominator = math.sqrt(sum(v * v for v in dx) * sum(v * v for v in dy))
    return sum(a * b for a, b in zip(dx, dy)) / denominator if denominator else None


def _model_name(path):
    name = path.name
    if "gpt-5.6-sol-high" in name:
        return "gpt-5.6-sol high"
    if "gpt-5.5-high" in name:
        return "gpt-5.5 high"
    return "other"


def _holdouts(run_dir):
    path = run_dir / "holdouts.jsonl"
    results = {}
    if not path.exists():
        return results
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        payload = _unseal(json.loads(line)["sealed"])
        if payload.get("ok"):
            results[(int(payload["n"]), payload["program_sha256"])] = payload
    return results


def load_run(run_dir):
    run_dir = Path(run_dir)
    session = json.loads((run_dir / "session.json").read_text())
    if session.get("task") != TASK:
        raise RuntimeError(f"{run_dir} is not a {TASK} run")
    records = [json.loads(line)
               for line in (run_dir / "submissions.jsonl").read_text().splitlines()
               if line.strip()]
    valid = [record for record in records if record.get("ok")]
    holdouts = _holdouts(run_dir)
    points = []
    for record in valid:
        payload = holdouts.get((int(record["n"]), record["program_sha256"]))
        if payload is None:
            continue
        metrics = payload.get("metrics") or {}
        points.append({
            "n": int(record["n"]),
            "best": bool(record.get("best")),
            "validation": float(record["guide_score"]),
            "test": float(metrics["test_score"]),
            "test_cells": dict(metrics["test_dataset_regression_rates"]),
            "program_sha256": record["program_sha256"],
        })
    return {
        "path": str(run_dir),
        "name": run_dir.name,
        "model": _model_name(run_dir),
        "fingerprint": session.get("benchmark_fingerprint"),
        "valid_submissions": len(valid),
        "scored_submissions": len(points),
        "points": points,
    }


def _round_floats(value):
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, dict):
        return {key: _round_floats(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_round_floats(item) for item in value]
    return value


def _summarize(runs):
    complete_runs = [run for run in runs
                     if run["points"] and
                     run["valid_submissions"] == run["scored_submissions"]]
    points = [point for run in complete_runs for point in run["points"]]
    xs = [point["validation"] for point in points]
    ys = [point["test"] for point in points]
    selected_rows = []
    transition_counts = {"improved": 0, "same": 0, "worsened": 0}
    for run in complete_runs:
        ordered = sorted(run["points"], key=lambda point: point["n"])
        accepted = [point for point in ordered if point["best"]]
        selected = accepted[-1]
        start = ordered[0]
        oracle = min(ordered, key=lambda point: (point["test"], point["n"]))
        selected_rows.append({
            "run": run["name"],
            "model": run["model"],
            "start_validation": start["validation"],
            "start_test": start["test"],
            "selected_n": selected["n"],
            "selected_validation": selected["validation"],
            "selected_test": selected["test"],
            "generalization_gap": selected["test"] - selected["validation"],
            "validation_improvement": start["validation"] - selected["validation"],
            "test_improvement": start["test"] - selected["test"],
            "oracle_n": oracle["n"],
            "oracle_test": oracle["test"],
            "selection_regret": selected["test"] - oracle["test"],
            "selected_test_cells": selected["test_cells"],
        })
        for before, after in zip(accepted, accepted[1:]):
            delta = after["test"] - before["test"]
            key = "improved" if delta < -1e-12 else "worsened" if delta > 1e-12 else "same"
            transition_counts[key] += 1
    return {
        "runs": len(runs),
        "complete_runs": len(complete_runs),
        "valid_submissions": sum(run["valid_submissions"] for run in runs),
        "scored_submissions": sum(run["scored_submissions"] for run in runs),
        "unique_scored_programs": len({point["program_sha256"] for point in points}),
        "pearson_validation_test": _pearson(xs, ys),
        "spearman_validation_test": _pearson(_ranks(xs), _ranks(ys)),
        "mean_selected_validation": _mean([row["selected_validation"] for row in selected_rows]),
        "mean_selected_test": _mean([row["selected_test"] for row in selected_rows]),
        "mean_generalization_gap": _mean([row["generalization_gap"] for row in selected_rows]),
        "mean_validation_improvement": _mean([row["validation_improvement"] for row in selected_rows]),
        "mean_test_improvement": _mean([row["test_improvement"] for row in selected_rows]),
        "mean_selection_regret": _mean([row["selection_regret"] for row in selected_rows]),
        "median_selection_regret": (statistics.median(row["selection_regret"]
                                                       for row in selected_rows)
                                    if selected_rows else None),
        "selected_test_cells": {
            cell: _mean([row["selected_test_cells"][cell]
                         for row in selected_rows])
            for cell in ("gpqa", "ifbench", "bfcl")
        } if selected_rows else {},
        "accepted_validation_improvement_test_changes": transition_counts,
        "per_run": selected_rows,
    }


def analyze(run_dirs):
    runs = [load_run(path) for path in sorted(map(Path, run_dirs), key=str)]
    result = {"task": TASK, "all": _summarize(runs), "models": {}}
    for model in sorted({run["model"] for run in runs}):
        result["models"][model] = _summarize(
            [run for run in runs if run["model"] == model])
    return _round_floats(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = analyze(args.run_dirs)
    all_results = result["all"]
    if (not args.allow_incomplete and
            all_results["valid_submissions"] != all_results["scored_submissions"]):
        raise SystemExit(
            f"audit incomplete: {all_results['scored_submissions']}/"
            f"{all_results['valid_submissions']} valid submissions scored")
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
