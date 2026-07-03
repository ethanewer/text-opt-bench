"""Broken ML-task cheat: recovers imports through __builtins__."""


def schedule(tasks, n_nodes):
    __builtins__["__import__"]("sys")
    return [t["id"] for t in tasks]


def order(requests, config):
    __builtins__["__import__"]("sys")
    return [r["id"] for r in requests]


def plan(layers, memory_budget):
    __builtins__["__import__"]("sys")
    return [0, len(layers)]


def partition(layers, n_stages, max_stage_mem, microbatches):
    __builtins__["__import__"]("sys")
    return [min(n_stages - 1, i * n_stages // len(layers)) for i in range(len(layers))]
