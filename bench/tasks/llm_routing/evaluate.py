"""Evaluator-owned offline routing on compact official RouterBench outcomes."""

import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, __file__.rsplit("/bench/", 1)[0])
from bench import eval_lib, heldout
from bench.ml_eval import call, integer, load_candidate, split_metrics

DATA = Path(__file__).resolve().parent / "data"
PENALTIES = (0.0, 0.1, 0.3, 1.0, 3.0)


def load_data(final):
    visible = json.loads((DATA / "train.json").read_text())
    test = heldout.read(DATA / "heldout_test.bin") if final else None
    return visible, heldout.read(DATA / "heldout_val.bin"), test


def score_rows(mod, rows, state, model_stats, scale):
    gaps = []
    accuracies = []
    costs = []
    for prompt, quality, cost in rows:
        for penalty in PENALTIES:
            choice = integer(call(mod.route, prompt, model_stats, penalty, state),
                             "route result", 0, len(model_stats) - 1)
            utility = quality[choice] - penalty * cost[choice] / scale
            oracle = max(q - penalty * c / scale for q, c in zip(quality, cost))
            gaps.append(oracle - utility)
            if penalty == 0.3:
                accuracies.append(quality[choice])
                costs.append(cost[choice])
    return {
        "score": sum(gaps) / len(gaps),
        "avg_accuracy": round(sum(accuracies) / len(accuracies), 6),
        "avg_cost": round(sum(costs) / len(costs), 8),
        "n_prompts": len(rows),
    }


def main():
    final = "--final" in sys.argv[2:]
    train_only = "--train-only" in sys.argv[2:]
    program_path = sys.argv[1]
    visible, val, test = load_data(final)
    fit_rows = visible["fit"]
    all_costs = [c for _, _, costs in fit_rows for c in costs if c > 0]
    scale = statistics.median(all_costs)
    n_models = len(fit_rows[0][1])
    model_stats = []
    for model in range(n_models):
        model_stats.append([
            sum(row[1][model] for row in fit_rows) / len(fit_rows),
            sum(row[2][model] for row in fit_rows) / len(fit_rows) / scale,
        ])
    # Training labels are legitimate meta-training information. The ranked
    # rows are disjoint and candidate calls never receive their outcomes.
    def fresh_score(rows):
        candidate = load_candidate(program_path, ("fit", "route"))
        state = call(candidate.fit, fit_rows)
        return score_rows(candidate, rows, state, model_stats, scale)

    train = fresh_score(visible["score"])
    if train_only:
        eval_lib.succeed(train["score"], split_metrics(train))
    validation = fresh_score(val)
    test_result = fresh_score(test) if final else None
    metrics = split_metrics(train, validation, test_result)
    metrics.update(dataset="RouterBench", models=n_models,
                   paper_metrics="AvgAcc, mean cost, oracle utility gap")
    eval_lib.succeed(validation["score"], metrics)


if __name__ == "__main__":
    main()
