"""Fast HPO-B + TaskSet-style transfer and multi-fidelity search prototype.

This synthetic table is only a mechanics smoke test.  The benchmark version
will replace ``make_suite`` with the released HPO-B and TaskSet tables while
keeping the same evaluator-owned query API and normalized-regret score.
"""

from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np


def make_suite(seed: int, tasks: int, configs: int, dims: int, fidelities: int):
    rng = np.random.default_rng(seed)
    config_x = rng.uniform(-1, 1, (configs, dims))
    task_features = rng.normal(size=(tasks, dims))
    task_features /= np.linalg.norm(task_features, axis=1, keepdims=True)
    optima = np.tanh(task_features @ rng.normal(size=(dims, dims)))
    curvature = rng.lognormal(mean=0.0, sigma=.8, size=(tasks, dims))
    final = ((config_x[None] - optima[:, None]) ** 2 *
             curvature[:, None]).mean(axis=2)
    final += .04 * np.sin(7 * config_x[None, :, 0] +
                          3 * task_features[:, None, 0])
    final -= final.min(axis=1, keepdims=True)
    speed = rng.lognormal(mean=0.0, sigma=.7, size=(tasks, configs))
    fractions = np.geomspace(.05, 1.0, fidelities)
    curves = np.stack([
        final + (1.0 + .25 * np.abs(config_x[:, 1])[None]) *
        np.exp(-speed * 4 * fraction)
        for fraction in fractions
    ], axis=2)
    return task_features, config_x, curves


def normalized_regret(values, optimum, worst):
    return (values - optimum) / np.maximum(worst - optimum, 1e-12)


def evaluate_order(order, curves, budgets):
    optimum = curves[:, :, -1].min(axis=1)
    worst = np.quantile(curves[:, :, -1], .9, axis=1)
    traces = []
    for budget in budgets:
        chosen = np.asarray(order[:, :budget])
        rows = np.arange(len(curves))[:, None]
        best = curves[rows, chosen, -1].min(axis=1)
        traces.append(normalized_regret(best, optimum, worst))
    trace = np.stack(traces, axis=1)
    return {
        "mean_regret_auc": float(trace.mean()),
        "final_mean_regret": float(trace[:, -1].mean()),
        "worst_quartile_final_regret": float(
            np.quantile(trace[:, -1], .75)),
        "budget_trace": [float(x) for x in trace.mean(axis=0)],
    }


def evaluate_successive_promotion(initial_order, curves):
    """Promote a transfer portfolio using progressively longer curve prefixes.

    The cost is reported in full-training equivalents, including every
    partially observed run. It is deliberately separate from the query-count
    score above because treating 64 short prefixes as one query would make the
    multi-fidelity method look artificially free.
    """
    fractions = np.geomspace(.05, 1.0, curves.shape[2])
    pools = initial_order[:, :64].copy()
    previous_fraction = 0.0
    cost = 0.0
    for fidelity, keep in [(0, 32), (2, 16), (4, 8), (curves.shape[2] - 1, 4)]:
        cost += pools.shape[1] * (fractions[fidelity] - previous_fraction)
        ranked = []
        for task, pool in enumerate(pools):
            ranked.append(pool[np.argsort(curves[task, pool, fidelity])[:keep]])
        pools = np.stack(ranked)
        previous_fraction = fractions[fidelity]
    optimum = curves[:, :, -1].min(axis=1)
    worst = np.quantile(curves[:, :, -1], .9, axis=1)
    rows = np.arange(len(curves))[:, None]
    best = curves[rows, pools, -1].min(axis=1)
    regret = normalized_regret(best, optimum, worst)
    return {"method": "transfer_successive_promotion",
            "full_training_equivalents": float(cost),
            "mean_normalized_regret": float(regret.mean()),
            "worst_quartile_regret": float(np.quantile(regret, .75))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=int, default=48)
    parser.add_argument("--configs", type=int, default=512)
    parser.add_argument("--dims", type=int, default=6)
    parser.add_argument("--fidelities", type=int, default=8)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    started = time.perf_counter()

    features, _configs, curves = make_suite(
        args.seed, args.tasks, args.configs, args.dims, args.fidelities)
    split = args.tasks * 2 // 3
    train_curves, test_curves = curves[:split], curves[split:]
    train_features, test_features = features[:split], features[split:]
    budgets = [1, 2, 4, 8, 16, 32]
    rng = np.random.default_rng(args.seed + 1)

    random_order = np.stack([
        rng.permutation(args.configs) for _ in range(len(test_curves))])
    global_rank = np.argsort(train_curves[:, :, -1].mean(axis=0))
    global_order = np.tile(global_rank, (len(test_curves), 1))
    similarities = test_features @ train_features.T
    neighbors = np.argpartition(-similarities, kth=4, axis=1)[:, :5]
    knn_order = np.stack([
        np.argsort(train_curves[index, :, -1].mean(axis=0))
        for index in neighbors
    ])
    oracle_order = np.argsort(test_curves[:, :, -1], axis=1)

    methods = {
        "random": random_order,
        "global_portfolio": global_order,
        "feature_knn_transfer": knn_order,
        "oracle": oracle_order,
    }
    results = []
    for name, order in methods.items():
        value = evaluate_order(order, test_curves, budgets)
        value["method"] = name
        results.append(value)
    print(json.dumps({
        "kind": "synthetic_hpo_taskset_mechanics_poc",
        "train_tasks": split,
        "test_tasks": args.tasks - split,
        "configs": args.configs,
        "fidelities": args.fidelities,
        "query_budgets": budgets,
        "results": sorted(results, key=lambda x: x["mean_regret_auc"]),
        "multi_fidelity_result": evaluate_successive_promotion(
            global_order, test_curves),
        "eval_seconds": time.perf_counter() - started,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
