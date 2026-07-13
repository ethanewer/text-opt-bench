"""Prototype bounded optimizer-generalization metric on expanded workloads."""

import json
import math
import sys
from pathlib import Path

from bench import eval_lib, heldout
from bench.ml_eval import call, finite, load_candidate, split_metrics
from bench.tasks.optimizer_synthesis.evaluate import loss_grad


DATA = Path(__file__).resolve().parent / "data/optimizer_generalization_v2"
STEPS = 120
INTERVAL = 10


def task_loss_grad(task, params, validation=False):
    if task[0] != "robust":
        return loss_grad(task, params, validation)
    rows = task[4] if validation else task[3]
    delta = 1.0
    gradients = [0.0] * len(params)
    loss = 0.0
    for row in rows:
        x, target = row[:-1], row[-1]
        error = sum(a * b for a, b in zip(x, params)) - target
        absolute = abs(error)
        if absolute <= delta:
            loss += 0.5 * error * error
            derivative = error
        else:
            loss += delta * (absolute - 0.5 * delta)
            derivative = delta if error > 0 else -delta
        for index, value in enumerate(x):
            gradients[index] += derivative * value
    return loss / len(rows), [value / len(rows) for value in gradients]


def run_task(mod, task):
    info = task[:3]
    params = list(task[-1] if task[0] != "factorization" else task[7])
    state = call(mod.init, info, len(params))
    initial, _ = task_loss_grad(task, params, validation=True)
    curve = [initial]
    for step in range(1, STEPS + 1):
        _, gradients = task_loss_grad(task, params)
        answer = call(mod.update, list(params), gradients, state, step, info)
        if type(answer) not in (list, tuple) or len(answer) != 2:
            eval_lib.fail("update must return [new_parameters, new_state]")
        new_params, state = answer
        if type(new_params) not in (list, tuple) or len(new_params) != len(params):
            eval_lib.fail("update returned the wrong parameter count")
        params = [finite(value, "updated parameter") for value in new_params]
        if step % INTERVAL == 0:
            value, _ = task_loss_grad(task, params, validation=True)
            curve.append(value if math.isfinite(value) else initial * math.exp(2))
    logs = [max(-6.0, min(2.0, math.log(max(value, 1e-12) /
                                        max(initial, 1e-12)))) for value in curve]
    bounded_auc = sum((value + 6.0) / 8.0 for value in logs) / len(logs)
    first_10x = next((i * INTERVAL for i, value in enumerate(curve)
                      if value <= initial * 0.1), STEPS + INTERVAL)
    first_100x = next((i * INTERVAL for i, value in enumerate(curve)
                       if value <= initial * 0.01), STEPS + INTERVAL)
    return bounded_auc, first_10x, first_100x, curve[-1] / max(initial, 1e-12)


def score_split(mod, tasks):
    rows = [(task[0],) + run_task(mod, task) for task in tasks]
    by_family = {}
    for kind, auc, *_ in rows:
        by_family.setdefault(kind, []).append(auc)
    family = {key: sum(values) / len(values) for key, values in by_family.items()}
    macro = sum(family.values()) / len(family)
    return {"score": macro + 0.2 * max(family.values()),
            "macro_bounded_auc": round(macro, 8),
            "worst_family_auc": round(max(family.values()), 8),
            "family_auc": {k: round(v, 8) for k, v in sorted(family.items())},
            "mean_steps_to_10x": round(sum(r[2] for r in rows) / len(rows), 4),
            "mean_steps_to_100x": round(sum(r[3] for r in rows) / len(rows), 4),
            "mean_final_loss_ratio": round(sum(r[4] for r in rows) / len(rows), 8),
            "n_workloads": len(rows)}


def main():
    final = "--final" in sys.argv[2:]
    path = sys.argv[1]
    train = json.loads((DATA / "train.json").read_text())
    validation = heldout.read(DATA / "heldout_val.bin")
    test = heldout.read(DATA / "heldout_test.bin") if final else None
    def fresh(tasks):
        return score_split(load_candidate(path, ("init", "update")), tasks)
    train_result, val = fresh(train), fresh(validation)
    test_result = fresh(test) if final else None
    metrics = split_metrics(train_result, val, test_result)
    metrics.update(steps=STEPS, workload_families=4,
                   paper_metric="bounded normalized validation-loss progress")
    eval_lib.succeed(val["score"], metrics)


if __name__ == "__main__":
    main()
