def prepare(meta_tasks):
    return None


def suggest(task_info, configurations, observations, remaining_budget, state):
    seen = set()
    for observation in observations:
        seen.add((observation[0], observation[1]))
    fidelities = task_info[2]
    for index in range(len(configurations) - 1, -1, -1):
        pair = (index, fidelities - 1)
        if pair not in seen:
            return [index, fidelities - 1]
    return [0, fidelities - 1]
