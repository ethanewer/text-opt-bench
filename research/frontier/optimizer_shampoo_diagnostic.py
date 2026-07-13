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
from bench.tasks.optimizer_synthesis.evaluate import loss_grad
from distributed_shampoo import AdamPreconditionerConfig, DistributedShampoo


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "bench/tasks/optimizer_synthesis/data"


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
    initial, _ = loss_grad(task, flatten(params), validation=True)
    curve = [initial]
    for step in range(1, 121):
        _, gradient = loss_grad(task, flatten(params))
        offset = 0
        for param in params:
            count = param.numel()
            param.grad = torch.tensor(gradient[offset:offset + count],
                                      dtype=param.dtype).reshape_as(param)
            offset += count
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if step % 10 == 0:
            value, _ = loss_grad(task, flatten(params), validation=True)
            curve.append(value if math.isfinite(value) else initial * 1e6)
    auc = sum(math.log(max(v, 1e-12) / max(initial, 1e-12)) for v in curve) / len(curve)
    return auc, curve[-1] / max(initial, 1e-12)


def score(tasks, lr):
    rows = [run_task(task, lr) for task in tasks]
    aucs = [row[0] for row in rows]
    worst = sorted(aucs)[-max(1, len(aucs) // 4):]
    return {
        "score": sum(aucs) / len(aucs) + 0.25 * sum(worst) / len(worst),
        "mean_normalized_log_auc": sum(aucs) / len(aucs),
        "worst_quartile_auc": sum(worst) / len(worst),
        "mean_final_loss_ratio": sum(row[1] for row in rows) / len(rows),
    }


def main():
    torch.manual_seed(0)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    train = json.loads((DATA / "train.json").read_text())
    validation = heldout.read(DATA / "heldout_val.bin")
    tuned = [(score(train, lr)["score"], lr) for lr in (0.003, 0.01, 0.03)]
    _, lr = min(tuned)
    print(json.dumps({"selected_lr": lr, "train": score(train, lr),
                      "validation": score(validation, lr)}, indent=2,
                     sort_keys=True))


if __name__ == "__main__":
    main()
