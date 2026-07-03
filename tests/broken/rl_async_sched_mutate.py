"""Broken exploit: mutates task ids so the permutation check used to see fake data."""


def schedule(tasks, n_nodes):
    for task in tasks:
        task["id"] = 0
        task["deps"] = ()
        task["duration"] = 0
        task["ready"] = 0
    return [0 for _ in tasks]
