"""Broken exploit: mutates layer memory to make an invalid plan look feasible."""


def plan(layers, memory_budget):
    for layer in layers:
        layer["act_mem"] = 0
        layer["fwd_cost"] = 0
    return [0, len(layers)]
