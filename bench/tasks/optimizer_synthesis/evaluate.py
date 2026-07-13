"""Evaluator-owned first-order optimizer generalization workloads."""

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, __file__.rsplit("/bench/", 1)[0])
from bench import eval_lib, heldout
from bench.ml_eval import call, finite, load_candidate, split_metrics

DATA = Path(__file__).resolve().parent / "data"
STEPS = 120
INTERVAL = 10


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def loss_grad(task, params, validation=False):
    kind = task[0]
    if kind == "quadratic":
        matrix = task[3]
        grad = [dot(row, params) for row in matrix]
        return 0.5 * dot(params, grad), grad
    if kind == "logistic":
        rows = task[4] if validation else task[3]
        grad = [0.0] * len(params)
        loss = 0.0
        for row in rows:
            x, y = row[:-1], row[-1]
            z = max(-40.0, min(40.0, dot(x, params)))
            prediction = 1.0 / (1.0 + math.exp(-z))
            loss += -(y * math.log(max(prediction, 1e-12)) +
                      (1.0 - y) * math.log(max(1.0 - prediction, 1e-12)))
            error = prediction - y
            for i, value in enumerate(x):
                grad[i] += error * value
        n = len(rows)
        return loss / n, [value / n for value in grad]
    # Matrix factorization: params contain U then V.
    rows, cols, rank, target = task[3], task[4], task[5], task[6]
    cut = rows * rank
    u, v = params[:cut], params[cut:]
    gu, gv = [0.0] * len(u), [0.0] * len(v)
    loss = 0.0
    for i in range(rows):
        for j in range(cols):
            pred = sum(u[i * rank + k] * v[j * rank + k] for k in range(rank))
            error = pred - target[i][j]
            loss += error * error
            for k in range(rank):
                gu[i * rank + k] += 2 * error * v[j * rank + k]
                gv[j * rank + k] += 2 * error * u[i * rank + k]
    scale = rows * cols
    return loss / scale, [(x / scale) for x in gu + gv]


def run_task(mod, task):
    info = task[:3]
    params = list(task[-1] if task[0] != "factorization" else task[7])
    state = call(mod.init, info, len(params))
    initial, _ = loss_grad(task, params, validation=True)
    curve = [initial]
    for step in range(1, STEPS + 1):
        _, gradients = loss_grad(task, params)
        answer = call(mod.update, list(params), gradients, state, step, info)
        if type(answer) not in (list, tuple) or len(answer) != 2:
            eval_lib.fail("update must return [new_parameters, new_state]")
        new_params, state = answer
        if type(new_params) not in (list, tuple) or len(new_params) != len(params):
            eval_lib.fail("update returned the wrong number of parameters")
        params = [finite(value, "updated parameter") for value in new_params]
        if step % INTERVAL == 0:
            value, _ = loss_grad(task, params, validation=True)
            if not math.isfinite(value):
                value = initial * 1e6
            curve.append(value)
    auc = sum(math.log(max(value, 1e-12) / max(initial, 1e-12))
              for value in curve) / len(curve)
    return auc, curve[-1] / max(initial, 1e-12)


def score_split(mod, tasks):
    aucs, finals = [], []
    for task in tasks:
        auc, final = run_task(mod, task)
        aucs.append(auc)
        finals.append(final)
    ordered = sorted(aucs)
    worst = ordered[-max(1, len(ordered) // 4):]
    mean = sum(aucs) / len(aucs)
    return {
        "score": mean + 0.25 * sum(worst) / len(worst),
        "mean_normalized_log_auc": round(mean, 8),
        "worst_quartile_auc": round(sum(worst) / len(worst), 8),
        "mean_final_loss_ratio": round(sum(finals) / len(finals), 8),
        "n_workloads": len(tasks),
    }


def main():
    final = "--final" in sys.argv[2:]
    train_only = "--train-only" in sys.argv[2:]
    program_path = sys.argv[1]
    train = json.loads((DATA / "train.json").read_text())
    val_tasks = heldout.read(DATA / "heldout_val.bin")
    test_tasks = heldout.read(DATA / "heldout_test.bin") if final else None
    def fresh_score(tasks):
        return score_split(load_candidate(program_path, ("init", "update")), tasks)

    train_result = fresh_score(train)
    if train_only:
        eval_lib.succeed(train_result["score"], split_metrics(train_result))
    val = fresh_score(val_tasks)
    test = fresh_score(test_tasks) if final else None
    metrics = split_metrics(train_result, val, test)
    metrics.update(steps=STEPS, workload_families=3,
                   paper_metric="normalized validation-loss curve area")
    eval_lib.succeed(val["score"], metrics)


if __name__ == "__main__":
    main()
