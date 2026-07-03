"""Reference solution: global longest-rollout-first (LPT), updates last.

LPT is the canonical multiprocessor-scheduling heuristic; putting all
updates after all rollouts keeps nodes off dependency waits.
"""


def schedule(tasks, n_nodes):
    rollouts = [t for t in tasks if t["kind"] == "rollout"]
    updates = [t for t in tasks if t["kind"] == "update"]
    rollouts.sort(key=lambda t: (-t["duration"], t["id"]))
    out = [t["id"] for t in rollouts]
    out.extend(t["id"] for t in sorted(updates, key=lambda t: t["id"]))
    return out
