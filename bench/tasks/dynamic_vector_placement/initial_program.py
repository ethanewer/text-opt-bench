"""Starter: first fit, with no attempt to preserve balanced free space."""


def init(capacity):
    return None


def place(request, machines, state):
    for machine_id, free, age, jobs in machines:
        if all(free[i] >= request[i] for i in range(4)):
            return machine_id, state
    return -1, state
