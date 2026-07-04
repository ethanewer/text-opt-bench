"""Evaluator for rl_async_sched. Score = deterministic simulated cluster cost."""

import random
import sys

sys.path.insert(0, __file__.rsplit("/bench/", 1)[0])
from bench import eval_lib, opcount

N_NODES = 8
SEED = 0xA51A
N_TRACES = 6
VALIDATION = [(0xA501, 4), (0xA502, 5), (0xA503, 6)]
BUDGET = 5_000_000
MAX_SOURCE_BYTES = 12_000
MAX_LITERAL_ITEMS = 80
MAX_TOTAL_LITERAL_ITEMS = 300
MAX_STRING_LITERAL_BYTES = 2_000

FORBIDDEN = frozenset(
    {
        "sys", "os", "io", "open", "mmap", "ctypes", "socket", "subprocess",
        "multiprocessing", "threading", "tempfile", "pathlib", "shutil",
        "sqlite3", "dbm", "shelve", "importlib", "signal", "inspect",
        "time", "resource",
        "builtins", "__builtins__", "bench", "__import__", "eval", "exec", "compile",
        "globals", "locals", "vars", "dir", "getattr", "setattr",
        "delattr", "type", "object", "super", "input", "breakpoint",
    }
)
FORBIDDEN_ATTRS = frozenset(
    {
        "__class__", "__dict__", "__globals__", "__code__", "__closure__",
        "__mro__", "__subclasses__", "__getattribute__", "__builtins__",
        "__traceback__", "tb_frame", "tb_next", "f_back", "f_globals",
        "f_locals", "gi_frame", "cr_frame",
    }
)


def rollout_duration(prompt, gen, env):
    # Prefill-ish quadratic term plus decode/env linear terms.
    return 18 + prompt * prompt // 80 + gen * 7 + env * 11


def update_duration(n_rollouts, tokens):
    # Simulated learner step plus communication overhead.
    return 120 + n_rollouts * 18 + tokens // 12 + int((tokens ** 0.5) * 7)


def gen_trace(seed, groups):
    rng = random.Random(seed)
    tasks = []
    task_id = 0
    for g in range(groups):
        deps = []
        n_rollouts = rng.randint(18, 34)
        wave_ready = g * rng.randint(35, 70)
        total_tokens = 0
        for _ in range(n_rollouts):
            r = rng.random()
            if r < 0.55:
                gen = rng.randint(16, 80)
            elif r < 0.88:
                gen = rng.randint(120, 360)
            else:
                gen = rng.randint(650, 1400)
            prompt = rng.randint(24, 220)
            env = rng.randint(1, 8)
            duration = rollout_duration(prompt, gen, env)
            ready = wave_ready + rng.randint(0, 55)
            tasks.append(
                {
                    "id": task_id,
                    "kind": "rollout",
                    "group": g,
                    "ready": ready,
                    "duration": duration,
                    "tokens": prompt + gen,
                    "deps": (),
                }
            )
            deps.append(task_id)
            task_id += 1
            total_tokens += prompt + gen
        tasks.append(
            {
                "id": task_id,
                "kind": "update",
                "group": g,
                "ready": wave_ready,
                "duration": update_duration(n_rollouts, total_tokens),
                "tokens": total_tokens,
                "deps": tuple(deps),
            }
        )
        task_id += 1
    return tasks


def scoring_traces():
    return [gen_trace(SEED + i * 101, 4 + (i % 3)) for i in range(N_TRACES)]


def simulate(tasks, order, label):
    ids = [t["id"] for t in tasks]
    # Element-check before sorted(): a non-int in the list must produce a
    # clean protocol failure, not an uncaught TypeError inside sorted().
    if (not isinstance(order, list)
            or any(not isinstance(x, int) for x in order)
            or sorted(order) != ids):
        eval_lib.fail(f"{label}: schedule() must return every task id exactly once")
    by_id = {t["id"]: t for t in tasks}
    rank = {tid: i for i, tid in enumerate(order)}
    remaining = set(ids)
    done = {}
    node_free = [0] * N_NODES
    rollout_done_times = []
    update_lags = []
    while remaining:
        node = min(range(N_NODES), key=node_free.__getitem__)
        now = node_free[node]
        available = [
            tid
            for tid in remaining
            if by_id[tid]["ready"] <= now and all(dep in done for dep in by_id[tid]["deps"])
        ]
        if not available:
            next_times = []
            for tid in remaining:
                task = by_id[tid]
                if all(dep in done for dep in task["deps"]):
                    next_times.append(max(now, task["ready"]))
            if not next_times:
                eval_lib.fail(f"{label}: no schedulable task remains")
            node_free[node] = min(next_times)
            continue
        tid = min(available, key=rank.__getitem__)
        task = by_id[tid]
        deps_ready = max((done[dep] for dep in task["deps"]), default=0)
        start = max(now, task["ready"], deps_ready)
        finish = start + task["duration"]
        node_free[node] = finish
        done[tid] = finish
        remaining.remove(tid)
        if task["kind"] == "rollout":
            rollout_done_times.append(finish - task["ready"])
        else:
            update_lags.append(start - deps_ready)
    makespan = max(done.values())
    mean_rollout = sum(rollout_done_times) / len(rollout_done_times)
    mean_lag = sum(update_lags) / max(1, len(update_lags))
    score = makespan + 0.10 * mean_rollout + 0.35 * mean_lag
    return score, makespan, mean_rollout, mean_lag


def heuristic_order(tasks):
    # Dependency-safe: rollouts by ready time, longest first within ready buckets,
    # then updates as soon as their dependencies have appeared.
    rollouts = [t for t in tasks if t["kind"] == "rollout"]
    updates = [t for t in tasks if t["kind"] == "update"]
    rollouts.sort(key=lambda t: (t["group"], t["ready"], -t["duration"], t["id"]))
    out = []
    emitted = set()
    updates_by_group = {t["group"]: t for t in updates}
    for r in rollouts:
        out.append(r["id"])
        emitted.add(r["id"])
        upd = updates_by_group[r["group"]]
        if all(dep in emitted for dep in upd["deps"]):
            out.append(upd["id"])
            emitted.add(upd["id"])
    return out


def load_candidate(program_path):
    return eval_lib.load_program(
        program_path,
        FORBIDDEN,
        required=("schedule",),
        forbidden_attrs=FORBIDDEN_ATTRS,
        safe_builtins=True,
        import_budget=BUDGET,
        max_source_bytes=MAX_SOURCE_BYTES,
        max_literal_items=MAX_LITERAL_ITEMS,
        max_total_literal_items=MAX_TOTAL_LITERAL_ITEMS,
        max_string_literal_bytes=MAX_STRING_LITERAL_BYTES,
    )


def schedule_guarded(program_path, tasks, label):
    mod = load_candidate(program_path)
    task_input = [dict(t) for t in tasks]
    eval_lib.set_candidate_active(True)   # guard the direct call (outside opcount)
    opcount.start(budget=BUDGET)
    try:
        order = mod.schedule(task_input, N_NODES)
    except opcount.BudgetExceeded:
        opcount.stop()
        eval_lib.fail(
            f"{label}: instruction budget of {BUDGET} exceeded "
            "(return a bounded deterministic heuristic)"
        )
    except BaseException as e:
        opcount.stop()
        eval_lib.fail(f"{label}: schedule() raised {type(e).__name__}: {e}")
    used = opcount.stop()
    eval_lib.set_candidate_active(False)
    if used > BUDGET:
        eval_lib.fail(f"{label}: instruction budget exceeded ({used} > {BUDGET})")
    return order, used


def main():
    program_path = sys.argv[1]

    for seed, groups in VALIDATION:
        label = f"validation seed {seed} ({groups} groups)"
        tasks = gen_trace(seed, groups)
        order, _ = schedule_guarded(program_path, tasks, label)
        score, _, _, _ = simulate(tasks, order, label)
        hscore, _, _, _ = simulate(tasks, heuristic_order(tasks), f"validation heuristic {seed}")
        if score > hscore * 1.35:
            eval_lib.fail(
                f"validation seed {seed}: score {score:.3f} is above quality limit {hscore * 1.35:.3f}"
            )

    total = 0.0
    makespans = []
    rollout_means = []
    update_lags = []
    instructions = []
    for k, tasks in enumerate(scoring_traces()):
        order, used = schedule_guarded(program_path, tasks, f"trace {k}")
        score, makespan, rollout_mean, update_lag = simulate(tasks, order, f"trace {k}")
        total += score
        instructions.append(used)
        makespans.append(makespan)
        rollout_means.append(round(rollout_mean, 3))
        update_lags.append(round(update_lag, 3))

    eval_lib.succeed(
        round(total, 6),
        metrics={
            "makespan_per_trace": makespans,
            "mean_rollout_completion": rollout_means,
            "mean_update_lag": update_lags,
            "n_nodes": N_NODES,
            "per_trace_instructions": instructions,
            "budget_per_trace": BUDGET,
        },
    )


if __name__ == "__main__":
    main()
