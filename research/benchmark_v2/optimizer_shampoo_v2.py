"""Run Meta's official Distributed Shampoo on the compact optimizer workloads.

The candidate API flattens parameters, erasing tensor structure.  This
evaluator-owned diagnostic restores the natural U/V matrix shapes for matrix
factorization and otherwise uses the same losses, steps, curve metric, and
visible-to-validation tuning discipline as the task.

Run with `PYTHONPATH=/tmp/fboptimizers:.` after cloning the official
facebookresearch/optimizers repository to `/tmp/fboptimizers`.
"""

import json
import math
from pathlib import Path

import torch

from bench import heldout
from research.benchmark_v2.evaluate_optimizer_v2 import task_loss_grad
from distributed_shampoo import AdamPreconditionerConfig, DistributedShampoo


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "research/benchmark_v2/data/optimizer_generalization_v2"


def tensors_for(task):
    initial = task[-1] if task[0] != "factorization" else task[7]
    if task[0] != "factorization":
        return [torch.nn.Parameter(torch.tensor(initial, dtype=torch.float64))]
    rows, cols, rank = task[3:6]
    cut = rows * rank
    return [
        torch.nn.Parameter(torch.tensor(initial[:cut], dtype=torch.float64).reshape(rows, rank)),
        torch.nn.Parameter(torch.tensor(initial[cut:], dtype=torch.float64).reshape(cols, rank)),
    ]


def flatten(params):
    return torch.cat([p.detach().flatten() for p in params]).tolist()


def run_task(task, lr):
    params = tensors_for(task)
    optimizer = DistributedShampoo(
        params, lr=lr, betas=(0.9, 0.999), epsilon=1e-12,
        precondition_frequency=10, start_preconditioning_step=10,
        max_preconditioner_dim=256,
        grafting_config=AdamPreconditionerConfig(beta2=0.999, epsilon=1e-8),
    )
    initial, _ = task_loss_grad(task, flatten(params), validation=True)
    curve = [initial]
    for step in range(1, 121):
        _, gradient = task_loss_grad(task, flatten(params))
        offset = 0
        for param in params:
            count = param.numel()
            param.grad = torch.tensor(gradient[offset:offset + count],
                                      dtype=param.dtype).reshape_as(param)
            offset += count
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if step % 10 == 0:
            value, _ = task_loss_grad(task, flatten(params), validation=True)
            curve.append(value if math.isfinite(value) else initial * 1e6)
    logs = [max(-6.0, min(2.0, math.log(max(v, 1e-12) /
                                        max(initial, 1e-12)))) for v in curve]
    auc = sum((value + 6.0) / 8.0 for value in logs) / len(logs)
    first_10x = next((i * 10 for i, value in enumerate(curve)
                      if value <= initial * 0.1), 130)
    first_100x = next((i * 10 for i, value in enumerate(curve)
                       if value <= initial * 0.01), 130)
    return auc, first_10x, first_100x, curve[-1] / max(initial, 1e-12)


def score(tasks, lr):
    rows = [(task[0],) + run_task(task, lr) for task in tasks]
    by_family = {}
    for kind, auc, *_ in rows:
        by_family.setdefault(kind, []).append(auc)
    family = {key: sum(values) / len(values) for key, values in by_family.items()}
    macro = sum(family.values()) / len(family)
    return {
        "score": macro + 0.2 * max(family.values()),
        "macro_bounded_auc": macro,
        "worst_family_auc": max(family.values()),
        "family_auc": family,
        "mean_steps_to_10x": sum(row[2] for row in rows) / len(rows),
        "mean_steps_to_100x": sum(row[3] for row in rows) / len(rows),
        "mean_final_loss_ratio": sum(row[4] for row in rows) / len(rows),
    }


def main():
    torch.manual_seed(0)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    train = json.loads((DATA / "train.json").read_text())
    validation = heldout.read(DATA / "heldout_val.bin")
    test = heldout.read(DATA / "heldout_test.bin")
    tuned = [(score(train, lr)["score"], lr) for lr in (0.003, 0.01, 0.03)]
    _, lr = min(tuned)
    print(json.dumps({"selected_lr": lr, "train": score(train, lr),
                      "validation": score(validation, lr),
                      "test": score(test, lr)}, indent=2,
                     sort_keys=True))


if __name__ == "__main__":
    main()
