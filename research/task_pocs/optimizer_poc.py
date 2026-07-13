"""A small TaskSet/AlgoPerf-style optimizer evaluator.

The score is fixed-step validation-loss-curve area, not wall-clock time. The
same implementation runs on CPU, MPS, or CUDA.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from common import choose_device, dump, timed


class Lion(torch.optim.Optimizer):
    def __init__(self, params, lr=3e-3, betas=(0.9, 0.99), weight_decay=0.0):
        super().__init__(params, dict(lr=lr, betas=betas,
                                     weight_decay=weight_decay))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                grad = parameter.grad
                if group["weight_decay"]:
                    parameter.mul_(1 - group["lr"] * group["weight_decay"])
                state = self.state[parameter]
                if "momentum" not in state:
                    state["momentum"] = torch.zeros_like(parameter)
                momentum = state["momentum"]
                update = momentum.mul(beta1).add(grad, alpha=1 - beta1)
                parameter.add_(update.sign(), alpha=-group["lr"])
                momentum.mul_(beta2).add_(grad, alpha=1 - beta2)


def generator(seed):
    return torch.Generator(device="cpu").manual_seed(seed)


def workloads(device):
    result = []
    for seed, condition in [(1, 30.0), (2, 300.0), (3, 3000.0)]:
        g = generator(seed)
        q, _ = torch.linalg.qr(torch.randn(48, 48, generator=g))
        eigen = torch.logspace(0, math.log10(condition), 48)
        matrix = (q @ torch.diag(eigen) @ q.T).to(device)
        initial = torch.randn(48, generator=g).to(device) * .2

        def build_quadratic(init=initial, a=matrix):
            p = torch.nn.Parameter(init.clone())
            return [p], lambda: .5 * p @ a @ p, lambda: .5 * p @ a @ p

        result.append((f"quadratic_c{int(condition)}", build_quadratic))

    for seed, scale in [(11, 1.0), (12, 8.0), (13, 32.0)]:
        g = generator(seed)
        x_train = torch.randn(256, 32, generator=g)
        x_test = torch.randn(256, 32, generator=g)
        scales = torch.logspace(0, math.log10(scale), 32)
        x_train *= scales
        x_test *= scales
        truth = torch.randn(32, generator=g) / scales
        y_train = (x_train @ truth + .3 * torch.randn(256, generator=g) > 0).float()
        y_test = (x_test @ truth + .3 * torch.randn(256, generator=g) > 0).float()
        x_train, x_test = x_train.to(device), x_test.to(device)
        y_train, y_test = y_train.to(device), y_test.to(device)

        def build_logistic(xt=x_train, xv=x_test, yt=y_train, yv=y_test):
            w = torch.nn.Parameter(torch.zeros(32, device=device))
            return ([w],
                    lambda: F.binary_cross_entropy_with_logits(xt @ w, yt),
                    lambda: F.binary_cross_entropy_with_logits(xv @ w, yv))

        result.append((f"logistic_scale{int(scale)}", build_logistic))

    for seed, rank in [(21, 2), (22, 4)]:
        g = generator(seed)
        left = torch.randn(28, rank, generator=g)
        right = torch.randn(24, rank, generator=g)
        target = (left @ right.T).to(device)
        init_u = (torch.randn(28, rank + 2, generator=g) * .05).to(device)
        init_v = (torch.randn(24, rank + 2, generator=g) * .05).to(device)

        def build_factorization(t=target, u0=init_u, v0=init_v):
            u = torch.nn.Parameter(u0.clone())
            v = torch.nn.Parameter(v0.clone())
            loss = lambda: F.mse_loss(u @ v.T, t)
            return [u, v], loss, loss

        result.append((f"factorization_r{rank}", build_factorization))
    return result


def optimizer(name, params):
    if name == "sgd":
        return torch.optim.SGD(params, lr=3e-3)
    if name == "momentum":
        return torch.optim.SGD(params, lr=3e-3, momentum=.9)
    if name == "rmsprop":
        return torch.optim.RMSprop(params, lr=3e-3, alpha=.99)
    if name == "adam":
        return torch.optim.Adam(params, lr=3e-3)
    if name == "lion":
        return Lion(params, lr=3e-4)
    raise ValueError(name)


def run_method(name, task_builders, steps, interval):
    task_results = []
    for task_name, build in task_builders:
        params, train_loss, validation_loss = build()
        opt = optimizer(name, params)
        with torch.no_grad():
            initial = float(validation_loss().cpu())
        curve = [initial]
        valid = True
        for step in range(1, steps + 1):
            opt.zero_grad(set_to_none=True)
            loss = train_loss()
            if not torch.isfinite(loss):
                valid = False
                break
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 100.0)
            opt.step()
            if step % interval == 0:
                with torch.no_grad():
                    value = float(validation_loss().cpu())
                if not math.isfinite(value):
                    valid = False
                    break
                curve.append(value)
        if not valid:
            curve.extend([initial * 1e6] * (steps // interval + 1 - len(curve)))
        # Dimensionless area under log(relative validation loss). Lower is
        # better; zero means no progress and negative values mean improvement.
        normalized_auc = sum(math.log(max(value, 1e-12) /
                                       max(initial, 1e-12)) for value in curve)
        normalized_auc /= len(curve)
        task_results.append({
            "task": task_name,
            "initial_loss": initial,
            "final_loss": curve[-1],
            "normalized_log_auc": normalized_auc,
        })
    scores = [value["normalized_log_auc"] for value in task_results]
    scores.sort()
    mean = sum(scores) / len(scores)
    worst_quartile = sum(scores[-max(1, len(scores) // 4):]) / max(1, len(scores) // 4)
    return {
        "method": name,
        "score": mean + .25 * worst_quartile,
        "mean_normalized_log_auc": mean,
        "worst_quartile_auc": worst_quartile,
        "tasks": task_results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto")
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--interval", type=int, default=10)
    args = parser.parse_args()
    torch.manual_seed(0)
    device = choose_device(torch, args.device)
    task_builders = workloads(device)
    results = []
    for name in ["sgd", "momentum", "rmsprop", "adam", "lion"]:
        value, seconds = timed(
            torch, device,
            lambda method=name: run_method(method, task_builders,
                                           args.steps, args.interval))
        value["eval_seconds"] = seconds
        results.append(value)
    dump({
        "device": str(device),
        "steps": args.steps,
        "workloads": len(task_builders),
        "results": sorted(results, key=lambda value: value["score"]),
    })


if __name__ == "__main__":
    main()
