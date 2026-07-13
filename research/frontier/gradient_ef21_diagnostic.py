"""Evaluator-owned EF21 / EF21-HB diagnostic on local gradient workloads.

The public candidate API hard-codes classical residual error feedback after
`encode`, so it cannot express EF21's worker-side Markov gradient estimator.
This script runs the paper algorithm on the same data, steps, exact sparse wire
accounting, and downstream validation metric.  Step-size and momentum are
selected on the visible train workloads, then frozen for validation.
"""

import json
import math
from pathlib import Path

from bench import heldout
from bench.tasks.gradient_compression.evaluate import gradient, validation_loss


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "bench/tasks/gradient_compression/data"
STEPS = 100


def run_workload(workload, momentum, step_scale):
    info, workers, validation = workload
    dims = len(workers[0][0]) - 1
    weight = [0.0] * dims
    estimates = [[0.0] * dims for _ in workers]
    velocity = [0.0] * dims
    initial, _ = validation_loss(validation, weight)
    curve = [initial]
    max_items = max(1, dims // 10)
    index_bits = max(1, math.ceil(math.log2(dims)))
    bits = 0
    for step in range(STEPS):
        for worker, rows in enumerate(workers):
            grad = gradient(rows, weight, step * 17 + worker * 7)
            delta = [a - b for a, b in zip(grad, estimates[worker])]
            chosen = sorted(range(dims), key=lambda i: abs(delta[i]),
                            reverse=True)[:max_items]
            for index in chosen:
                estimates[worker][index] += delta[index]
            bits += 16 + len(chosen) * (32 + index_bits)
        for i in range(dims):
            mean_estimate = sum(row[i] for row in estimates) / len(workers)
            velocity[i] = momentum * velocity[i] + mean_estimate
            weight[i] -= info[2] * step_scale * velocity[i]
        if step % 10 == 9:
            curve.append(validation_loss(validation, weight)[0])
    final_loss, accuracy = validation_loss(validation, weight)
    dense_bits = STEPS * len(workers) * dims * 32
    quality_auc = sum(value / initial for value in curve) / len(curve)
    return quality_auc, bits / dense_bits, final_loss, accuracy


def score(workloads, momentum, step_scale):
    rows = [run_workload(w, momentum, step_scale) for w in workloads]
    values = [r[0] + 0.15 * r[1] for r in rows]
    worst = sorted(values)[-max(1, len(values) // 4):]
    return {
        "score": sum(values) / len(values) + 0.2 * sum(worst) / len(worst),
        "mean_quality_bit_score": sum(values) / len(values),
        "communicated_fraction_of_dense": sum(r[1] for r in rows) / len(rows),
        "final_validation_loss": sum(r[2] for r in rows) / len(rows),
        "final_accuracy": sum(r[3] for r in rows) / len(rows),
    }


def tune(train, momentum_values):
    candidates = []
    for momentum in momentum_values:
        for scale in (0.025, 0.05, 0.1, 0.2, 0.5, 1.0):
            result = score(train, momentum, scale)
            candidates.append((result["score"], momentum, scale, result))
    return min(candidates, key=lambda row: row[0])


def main():
    train = json.loads((DATA / "train.json").read_text())
    validation = heldout.read(DATA / "heldout_val.bin")
    result = {}
    for name, momenta in (("ef21", (0.0,)),
                          ("ef21_hb", (0.5, 0.7, 0.9))):
        _, momentum, scale, train_score = tune(train, momenta)
        result[name] = {
            "selected_on_train": {"momentum": momentum,
                                  "step_scale": scale},
            "train": train_score,
            "validation": score(validation, momentum, scale),
        }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
