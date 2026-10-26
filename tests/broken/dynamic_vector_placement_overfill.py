def init(capacity):
    return None


def place(request, machines, state):
    if machines:
        return machines[0][0], state
    return -1, state
