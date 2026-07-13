"""Evaluator-owned distributed training with exact communication accounting."""

import json
import math
import struct
import sys
from pathlib import Path

sys.path.insert(0, __file__.rsplit("/bench/", 1)[0])
from bench import eval_lib, heldout
from bench.ml_eval import call, finite, int_list, load_candidate, split_metrics

DATA = Path(__file__).resolve().parent / "data"
STEPS = 100


def sigmoid(value):
    value = max(-40.0, min(40.0, value))
    return 1.0 / (1.0 + math.exp(-value))


def gradient(rows, weight, start):
    dims = len(weight)
    grad = [0.0] * dims
    batch = 24
    for offset in range(batch):
        row = rows[(start + offset) % len(rows)]
        x, y = row[:-1], row[-1]
        error = sigmoid(sum(a * b for a, b in zip(x, weight))) - y
        for i in range(dims):
            grad[i] += error * x[i] / batch
    return grad


def validation_loss(rows, weight):
    total = 0.0
    correct = 0
    for row in rows:
        x, y = row[:-1], row[-1]
        p = sigmoid(sum(a * b for a, b in zip(x, weight)))
        total += -(y * math.log(max(p, 1e-12)) +
                   (1 - y) * math.log(max(1 - p, 1e-12)))
        correct += (p >= .5) == bool(y)
    return total / len(rows), correct / len(rows)


def run_workload(mod, workload):
    info, workers, validation = workload
    dims = len(workers[0][0]) - 1
    weight = [0.0] * dims
    residuals = [[0.0] * dims for _ in workers]
    states = call(mod.init, dims, len(workers), info)
    if type(states) not in (list, tuple) or len(states) != len(workers):
        eval_lib.fail("init must return one state per worker")
    states = list(states)
    initial, _ = validation_loss(validation, weight)
    curve = [initial]
    bits = 0
    max_items = max(1, dims // 10)
    index_bits = max(1, math.ceil(math.log2(dims)))
    for step in range(STEPS):
        decoded_workers = []
        for worker, rows in enumerate(workers):
            grad = gradient(rows, weight, step * 17 + worker * 7)
            source = [g + r for g, r in zip(grad, residuals[worker])]
            answer = call(mod.encode, source, max_items,
                          [worker, step, info], states[worker])
            if type(answer) not in (list, tuple) or len(answer) != 3:
                eval_lib.fail("encode must return [indices, values, new_state]")
            indices = int_list(answer[0], "encoded indices", unique=True,
                               low=0, high=dims - 1, max_len=max_items)
            values = answer[1]
            if type(values) not in (list, tuple) or len(values) != len(indices):
                eval_lib.fail("encoded values must match encoded indices")
            decoded = [0.0] * dims
            for index, value in zip(indices, values):
                value = finite(value, "encoded value")
                try:
                    value = struct.unpack("<f", struct.pack("<f", value))[0]
                except OverflowError:
                    eval_lib.fail("encoded value overflows its 32-bit wire format")
                if not math.isfinite(value):
                    eval_lib.fail("encoded value is non-finite after float32 encoding")
                decoded[index] = value
            residuals[worker] = [a - b for a, b in zip(source, decoded)]
            states[worker] = answer[2]
            decoded_workers.append(decoded)
            bits += 16 + len(indices) * (32 + index_bits)
        for i in range(dims):
            update = sum(row[i] for row in decoded_workers) / len(workers)
            weight[i] -= info[2] * update
        if step % 10 == 9:
            curve.append(validation_loss(validation, weight)[0])
    final_loss, accuracy = validation_loss(validation, weight)
    dense_bits = STEPS * len(workers) * dims * 32
    quality_auc = sum(value / initial for value in curve) / len(curve)
    return quality_auc, bits / dense_bits, final_loss, accuracy


def score_split(mod, workloads):
    scores, ratios, losses, accuracies = [], [], [], []
    for workload in workloads:
        auc, ratio, loss, accuracy = run_workload(mod, workload)
        scores.append(auc + 0.15 * ratio)
        ratios.append(ratio)
        losses.append(loss)
        accuracies.append(accuracy)
    ordered = sorted(scores)
    worst = ordered[-max(1, len(ordered) // 4):]
    mean = sum(scores) / len(scores)
    return {
        "score": mean + .2 * sum(worst) / len(worst),
        "mean_quality_bit_score": round(mean, 8),
        "communicated_fraction_of_dense": round(sum(ratios) / len(ratios), 8),
        "final_validation_loss": round(sum(losses) / len(losses), 8),
        "final_accuracy": round(sum(accuracies) / len(accuracies), 8),
        "n_workloads": len(workloads),
    }


def main():
    final = "--final" in sys.argv[2:]
    train_only = "--train-only" in sys.argv[2:]
    program_path = sys.argv[1]
    train_data = json.loads((DATA / "train.json").read_text())
    val_data = heldout.read(DATA / "heldout_val.bin")
    test_data = heldout.read(DATA / "heldout_test.bin") if final else None
    def fresh_score(tasks):
        return score_split(load_candidate(program_path, ("init", "encode")), tasks)

    train = fresh_score(train_data)
    if train_only:
        eval_lib.succeed(train["score"], split_metrics(train))
    val = fresh_score(val_data)
    test = fresh_score(test_data) if final else None
    metrics = split_metrics(train, val, test)
    metrics.update(steps=STEPS, value_bits=32,
                   paper_metric="validation quality versus transmitted bits")
    eval_lib.succeed(val["score"], metrics)


if __name__ == "__main__":
    main()
