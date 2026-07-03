"""Baseline: dependency-safe FIFO by group."""


def schedule(tasks, n_nodes):
    out = []
    groups = sorted({t["group"] for t in tasks})
    for g in groups:
        rollouts = [t for t in tasks if t["group"] == g and t["kind"] == "rollout"]
        updates = [t for t in tasks if t["group"] == g and t["kind"] == "update"]
        out.extend(t["id"] for t in sorted(rollouts, key=lambda t: (t["ready"], t["id"])))
        out.extend(t["id"] for t in sorted(updates, key=lambda t: t["id"]))
    return out
