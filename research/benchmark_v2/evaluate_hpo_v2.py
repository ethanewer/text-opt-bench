"""Prototype expanded HPO-B transfer evaluator for benchmark v2."""

import json
import math
import sys
from pathlib import Path

from bench import eval_lib, heldout
from bench.ml_eval import call, int_list, load_candidate, split_metrics


DATA = Path(__file__).resolve().parent / "data/hpo_transfer_v2"
BUDGETS = (1.0, 2.0, 4.0, 8.0)


def task_score(mod, task, state):
    kind, features, configurations, curves = task
    fidelities = len(curves[0])
    costs = [1.0] if fidelities == 1 else [0.125, 0.25, 0.5, 1.0]
    observations, used = [], set()
    spent = 0.0
    incumbent = None
    checkpoints = []
    final = [curve[-1] for curve in curves]
    optimum = min(final)
    robust_worst = sorted(final)[max(0, int(0.9 * len(final)) - 1)]
    scale = max(robust_worst - optimum, 1e-12)
    info = [kind, features, fidelities]
    while spent < BUDGETS[-1] - 1e-12:
        answer = int_list(call(mod.suggest, info, configurations, observations,
                               BUDGETS[-1] - spent, state),
                          "suggest result", max_len=2)
        if len(answer) != 2:
            eval_lib.fail("suggest must return [configuration_index, fidelity_index]")
        index, fidelity = answer
        if not (0 <= index < len(configurations) and 0 <= fidelity < fidelities):
            eval_lib.fail("suggest returned an out-of-range index")
        if (index, fidelity) in used:
            eval_lib.fail("suggest repeated a configuration/fidelity pair")
        cost = costs[fidelity]
        if spent + cost > BUDGETS[-1] + 1e-12:
            break
        completion = spent + cost
        while (len(checkpoints) < len(BUDGETS) and
               BUDGETS[len(checkpoints)] < completion - 1e-12):
            value = robust_worst if incumbent is None else incumbent
            checkpoints.append(max(0.0, (value - optimum) / scale))
        used.add((index, fidelity))
        spent = completion
        value = curves[index][fidelity]
        observations.append([index, fidelity, value, cost])
        if fidelity == fidelities - 1:
            incumbent = value if incumbent is None else min(incumbent, value)
        while len(checkpoints) < len(BUDGETS) and spent >= BUDGETS[len(checkpoints)] - 1e-12:
            value = robust_worst if incumbent is None else incumbent
            checkpoints.append(max(0.0, (value - optimum) / scale))
    while len(checkpoints) < len(BUDGETS):
        value = robust_worst if incumbent is None else incumbent
        checkpoints.append(max(0.0, (value - optimum) / scale))
    return sum(checkpoints) / len(checkpoints)


def score_split(mod, tasks, state):
    by_space = {}
    values = []
    for task in tasks:
        value = task_score(mod, task, state)
        values.append(value)
        by_space.setdefault(str(task[1][0]), []).append(value)
    space_means = {key: sum(rows) / len(rows) for key, rows in by_space.items()}
    macro = sum(space_means.values()) / len(space_means)
    score = macro + 0.25 * max(space_means.values())
    sample_mean = sum(values) / len(values)
    sample_variance = (sum((x - sample_mean) ** 2 for x in values) /
                       max(1, len(values) - 1))
    return {
        "score": score,
        "macro_space_regret_auc": round(macro, 8),
        "worst_space_regret_auc": round(max(space_means.values()), 8),
        "space_regret_auc": {key: round(value, 8)
                             for key, value in sorted(space_means.items())},
        "task_standard_error": round(math.sqrt(sample_variance / len(values)), 8),
        "n_tasks": len(tasks),
    }


def main():
    final = "--final" in sys.argv[2:]
    train_only = "--train-only" in sys.argv[2:]
    path = sys.argv[1]
    visible = json.loads((DATA / "train.json").read_text())
    validation = heldout.read(DATA / "heldout_val.bin")
    test = heldout.read(DATA / "heldout_test.bin") if final else None

    def fresh(tasks):
        candidate = load_candidate(path, ("prepare", "suggest"))
        state = call(candidate.prepare, visible["meta"])
        return score_split(candidate, tasks, state)

    train = fresh(visible["score"])
    if train_only:
        eval_lib.succeed(train["score"], split_metrics(train))
    val = fresh(validation)
    test_result = fresh(test) if final else None
    metrics = split_metrics(train, val, test_result)
    metrics.update(datasets="HPO-B-v3 expanded", budgets=list(BUDGETS),
                   paper_metric="space-macro normalized regret AUC")
    eval_lib.succeed(val["score"], metrics)


if __name__ == "__main__":
    main()
