"""Sequential lookup benchmark over compact official HPO-B/TaskSet data."""

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, __file__.rsplit("/bench/", 1)[0])
from bench import eval_lib, heldout
from bench.ml_eval import call, int_list, load_candidate, split_metrics

DATA = Path(__file__).resolve().parent / "data"
BUDGETS = (1.0, 2.0, 4.0, 8.0)


def task_score(mod, task, state):
    # task = [kind, task_features, configurations, curves]
    kind, features, configurations, curves = task
    fidelities = len(curves[0])
    costs = [1.0] if fidelities == 1 else [0.125, 0.25, 0.5, 1.0]
    info = [kind, features, fidelities]
    observations = []
    used_pairs = set()
    spent = 0.0
    incumbent = None
    checkpoints = []
    final_values = [curve[-1] for curve in curves]
    optimum = min(final_values)
    worst = sorted(final_values)[max(0, int(.9 * len(final_values)) - 1)]
    scale = max(worst - optimum, 1e-12)
    while spent < BUDGETS[-1] - 1e-12:
        answer = int_list(call(mod.suggest, info, configurations, observations,
                               BUDGETS[-1] - spent, state),
                          "suggest result", max_len=2)
        if len(answer) != 2:
            eval_lib.fail("suggest must return [configuration_index, fidelity_index]")
        index, fidelity = answer
        if not (0 <= index < len(configurations) and 0 <= fidelity < fidelities):
            eval_lib.fail("suggest returned an out-of-range index")
        pair = (index, fidelity)
        if pair in used_pairs:
            eval_lib.fail("suggest repeated an evaluated configuration/fidelity pair")
        cost = costs[fidelity]
        if spent + cost > BUDGETS[-1] + 1e-12:
            break
        completion = spent + cost
        # A result that starts before a checkpoint but completes after it was
        # not available at that budget. Fill crossed checkpoints from the old
        # incumbent before applying the new observation. Exact-boundary
        # completions are handled below and may use the new result.
        while (len(checkpoints) < len(BUDGETS) and
               BUDGETS[len(checkpoints)] < completion - 1e-12):
            value_at_budget = worst if incumbent is None else incumbent
            checkpoints.append(max(0.0, (value_at_budget - optimum) / scale))
        used_pairs.add(pair)
        spent = completion
        value = curves[index][fidelity]
        observations.append([index, fidelity, value, cost])
        if fidelity == fidelities - 1:
            incumbent = value if incumbent is None else min(incumbent, value)
        while (len(checkpoints) < len(BUDGETS) and
               spent >= BUDGETS[len(checkpoints)] - 1e-12):
            value_at_budget = worst if incumbent is None else incumbent
            checkpoints.append(max(0.0, (value_at_budget - optimum) / scale))
    while len(checkpoints) < len(BUDGETS):
        value_at_budget = worst if incumbent is None else incumbent
        checkpoints.append(max(0.0, (value_at_budget - optimum) / scale))
    return sum(checkpoints) / len(checkpoints), checkpoints


def score_split(mod, tasks, state):
    values = []
    hpob = []
    taskset = []
    for task in tasks:
        value, _ = task_score(mod, task, state)
        values.append(value)
        (hpob if task[0] == "hpob" else taskset).append(value)
    ordered = sorted(values)
    worst_quartile = ordered[-max(1, len(ordered) // 4):]
    mean = sum(values) / len(values)
    return {
        "score": mean + 0.25 * sum(worst_quartile) / len(worst_quartile),
        "mean_normalized_regret_auc": round(mean, 8),
        "hpob_regret_auc": round(sum(hpob) / len(hpob), 8),
        "taskset_regret_auc": round(sum(taskset) / len(taskset), 8),
        "n_tasks": len(tasks),
    }


def main():
    final = "--final" in sys.argv[2:]
    train_only = "--train-only" in sys.argv[2:]
    program_path = sys.argv[1]
    visible = json.loads((DATA / "train.json").read_text())
    validation = heldout.read(DATA / "heldout_val.bin")
    test = heldout.read(DATA / "heldout_test.bin") if final else None
    def fresh_score(tasks):
        # A fresh module and freshly prepared state make every split an
        # independent deployment. Candidate mutations while scoring train or
        # validation cannot influence the sealed test.
        candidate = load_candidate(program_path, ("prepare", "suggest"))
        state = call(candidate.prepare, visible["meta"])
        return score_split(candidate, tasks, state)

    train = fresh_score(visible["score"])
    if train_only:
        eval_lib.succeed(train["score"], split_metrics(train))
    val = fresh_score(validation)
    test_result = fresh_score(test) if final else None
    metrics = split_metrics(train, val, test_result)
    metrics.update(
        datasets="HPO-B-v3 + original TaskSet paper archives",
        budgets=list(BUDGETS),
        paper_metric="normalized regret versus trials/training-equivalent budget",
    )
    eval_lib.succeed(val["score"], metrics)


if __name__ == "__main__":
    main()
