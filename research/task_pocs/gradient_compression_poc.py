"""Communication-budgeted distributed-training compression prototype."""

from __future__ import annotations

import argparse
import json
import math
import time

import torch
import torch.nn.functional as F


def make_workers(seed, workers, samples, dims, non_iid):
    g = torch.Generator().manual_seed(seed)
    truth = torch.randn(dims, generator=g)
    data = []
    for worker in range(workers):
        shift = non_iid * (worker - (workers - 1) / 2) / workers
        x = torch.randn(samples, dims, generator=g) + shift
        logits = x @ truth + .35 * torch.randn(samples, generator=g)
        data.append((x, (logits > 0).float()))
    xv = torch.randn(samples * 2, dims, generator=g)
    yv = (xv @ truth + .35 * torch.randn(samples * 2, generator=g) > 0).float()
    return data, (xv, yv)


def compress(vector, method, fraction, residual):
    source = vector + residual if residual is not None else vector
    n = source.numel()
    if method == "dense":
        decoded = source.clone()
        bits = n * 32
    elif method == "sign":
        scale = source.abs().mean()
        decoded = source.sign() * scale
        bits = n + 32
    elif method in {"topk", "topk_ef"}:
        k = max(1, int(n * fraction))
        indices = torch.topk(source.abs(), k).indices
        decoded = torch.zeros_like(source)
        decoded[indices] = source[indices]
        index_bits = max(1, math.ceil(math.log2(n)))
        bits = k * (32 + index_bits)
    else:
        raise ValueError(method)
    new_residual = source - decoded if method == "topk_ef" else torch.zeros_like(source)
    return decoded, new_residual, bits


def run(method, workers, validation, steps, fraction, lr):
    dims = workers[0][0].shape[1]
    weight = torch.zeros(dims, requires_grad=True)
    residuals = [torch.zeros(dims) for _ in workers]
    bits = 0
    curve = []
    xv, yv = validation
    for step in range(steps):
        decoded = []
        for index, (x, y) in enumerate(workers):
            start = (step * 17) % (len(x) - 32)
            loss = F.binary_cross_entropy_with_logits(
                x[start:start + 32] @ weight, y[start:start + 32])
            grad, = torch.autograd.grad(loss, weight)
            sent, residuals[index], used = compress(
                grad.detach(), method, fraction, residuals[index])
            decoded.append(sent)
            bits += used
        with torch.no_grad():
            weight -= lr * torch.stack(decoded).mean(dim=0)
        if step % 5 == 0 or step == steps - 1:
            with torch.no_grad():
                val = float(F.binary_cross_entropy_with_logits(xv @ weight, yv))
                acc = float(((xv @ weight > 0) == yv.bool()).float().mean())
            curve.append({"step": step + 1, "bits": bits,
                          "validation_loss": val, "accuracy": acc})
    return {"method": method, "communicated_bits": bits,
            "final_validation_loss": curve[-1]["validation_loss"],
            "final_accuracy": curve[-1]["accuracy"], "curve": curve}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--fraction", type=float, default=.1)
    args = parser.parse_args()
    torch.set_num_threads(max(1, min(torch.get_num_threads(), 4)))
    started = time.perf_counter()
    workloads = []
    for name, non_iid in [("iid", 0.0), ("non_iid", 2.0)]:
        workers, validation = make_workers(31, args.workers, 256, 128, non_iid)
        methods = [run(method, workers, validation, args.steps,
                       args.fraction, .2)
                   for method in ["dense", "sign", "topk", "topk_ef"]]
        workloads.append({"workload": name, "methods": methods})
    print(json.dumps({
        "metric": "validation quality versus exact communicated bits",
        "steps": args.steps,
        "workers": args.workers,
        "workloads": workloads,
        "eval_seconds": time.perf_counter() - started,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
