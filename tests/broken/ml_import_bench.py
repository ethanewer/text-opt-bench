"""Broken ML-task cheat: imports benchmark internals."""

from bench.tasks.rl_async_sched import evaluate


def schedule(tasks, n_nodes):
    return [t["id"] for t in tasks]


def order(requests, config):
    return [r["id"] for r in requests]


def plan(layers, memory_budget):
    return [0, len(layers)]


def partition(layers, n_stages, max_stage_mem, microbatches):
    return [min(n_stages - 1, i * n_stages // len(layers)) for i in range(len(layers))]
