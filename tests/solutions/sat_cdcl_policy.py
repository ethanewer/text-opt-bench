"""Jeroslow-Wang/activity policy used as a task-quality reference."""


def prepare(problem):
    return {"family": problem["family"], "next_restart": 24}


def branch(search, state):
    # The regular symmetry of pigeonhole formulas favors stable variable
    # order; dynamic occurrence scoring is much better on the other families.
    if state["family"] == "pigeonhole":
        return [search["unassigned"][0], 1]
    best = search["unassigned"][0]
    best_score = -1.0
    phase = 1
    for variable in search["unassigned"]:
        positive = search["jw_positive"][variable]
        negative = search["jw_negative"][variable]
        score = positive + negative + 0.035 * search["activity"][variable]
        if score > best_score:
            best_score = score
            best = variable
            phase = 1 if positive >= negative else 0
    if search["activity"][best] > 0:
        phase = search["saved_phase"][best]
    return [best, phase]


def restart(conflict, state):
    if state["family"] == "pigeonhole":
        return False
    if conflict["conflicts"] >= state["next_restart"]:
        state["next_restart"] = state["next_restart"] * 3 // 2 + 8
        return True
    return False
