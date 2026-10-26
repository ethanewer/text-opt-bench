def prepare(problem):
    return None


def branch(search, state):
    return [search["unassigned"][0], 1]


def restart(conflict, state):
    return 0
