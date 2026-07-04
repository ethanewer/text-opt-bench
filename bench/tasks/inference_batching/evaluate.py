"""Evaluator for inference_batching. Score = deterministic simulated serving cost."""

import random
import sys

sys.path.insert(0, __file__.rsplit("/bench/", 1)[0])
from bench import eval_lib, opcount

SEED = 0x1F0
N_TRACES = 7
VALIDATION = [(0x1F01, 85), (0x1F02, 118), (0x1F03, 151)]
CONFIG = {"max_batch": 12, "max_prefill_tokens": 3000, "kv_capacity": 9000}
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


def gen_requests(seed, n):
    rng = random.Random(seed)
    reqs = []
    t = 0
    for i in range(n):
        t += rng.randint(0, 9) if rng.random() < 0.82 else rng.randint(20, 70)
        r = rng.random()
        if r < 0.52:
            prompt = rng.randint(32, 220)
            output = rng.randint(12, 96)
        elif r < 0.86:
            prompt = rng.randint(240, 950)
            output = rng.randint(80, 260)
        else:
            prompt = rng.randint(1000, 2400)
            output = rng.randint(260, 900)
        priority = 2 if rng.random() < 0.12 else (1 if rng.random() < 0.35 else 0)
        reqs.append({"id": i, "arrival": t, "prompt": prompt, "output": output, "priority": priority})
    return reqs


def scoring_traces():
    return [gen_requests(SEED + i * 149, 85 + i * 11) for i in range(N_TRACES)]


def prefill_cost(batch):
    tokens = sum(r["prompt"] for r in batch)
    max_prompt = max(r["prompt"] for r in batch)
    return 25 + tokens // 18 + max_prompt // 5 + len(batch) * 3


def decode_step_cost(active):
    tokens = sum(r["prompt"] + r["generated"] for r in active)
    return 7 + len(active) * 2 + tokens // 900


def percentile(values, pct):
    vals = sorted(values)
    idx = int((len(vals) - 1) * pct / 100)
    return vals[idx]


def simulate(requests, order, label):
    ids = [r["id"] for r in requests]
    # Element-check before sorted(): a non-int in the list must produce a
    # clean protocol failure, not an uncaught TypeError inside sorted().
    if (not isinstance(order, list)
            or any(not isinstance(x, int) for x in order)
            or sorted(order) != ids):
        eval_lib.fail(f"{label}: order() must return every request id exactly once")
    by_id = {r["id"]: r for r in requests}
    for r in requests:
        if r["prompt"] > CONFIG["max_prefill_tokens"] or r["prompt"] + r["output"] > CONFIG["kv_capacity"]:
            eval_lib.fail(f"{label}: generated request {r['id']} cannot fit serving caps")
    pending = [by_id[i].copy() for i in order]
    active = []
    completed = {}
    now = 0
    while pending or active:
        admitted = False
        while pending:
            ready = [r for r in pending if r["arrival"] <= now]
            if not ready:
                break
            batch = []
            batch_tokens = 0
            active_tokens = sum(r["prompt"] + r["output"] for r in active)
            for r in list(pending):
                if r["arrival"] > now:
                    continue
                if len(batch) >= CONFIG["max_batch"]:
                    break
                if batch_tokens + r["prompt"] > CONFIG["max_prefill_tokens"]:
                    continue
                if active_tokens + batch_tokens + r["prompt"] + r["output"] > CONFIG["kv_capacity"]:
                    continue
                batch.append(r)
                batch_tokens += r["prompt"]
            if not batch:
                break
            for r in batch:
                pending.remove(r)
            now += prefill_cost(batch)
            for r in batch:
                r["generated"] = 0
                active.append(r)
            admitted = True
        if not active:
            if pending:
                now = max(now, min(r["arrival"] for r in pending))
                continue
            break
        # Decode one token for every active sequence.
        now += decode_step_cost(active)
        done = []
        for r in active:
            r["generated"] += 1
            if r["generated"] >= r["output"]:
                done.append(r)
        for r in done:
            active.remove(r)
            completed[r["id"]] = now
        if not admitted and pending and not active:
            now = max(now, min(r["arrival"] for r in pending))
    latencies = [completed[r["id"]] - r["arrival"] for r in requests]
    weights = [1.0 + 0.8 * r["priority"] for r in requests]
    weighted = [latency * weight for latency, weight in zip(latencies, weights)]
    mean = sum(weighted) / sum(weights)
    p95 = percentile(latencies, 95)
    score = mean + 0.30 * p95 + 0.04 * max(completed.values())
    return score, mean, p95, max(completed.values())


def heuristic_order(requests):
    return [
        r["id"]
        for r in sorted(
            requests,
            key=lambda r: (r["arrival"] // 20, -r["priority"], r["prompt"] + r["output"], r["id"]),
        )
    ]


def load_candidate(program_path):
    return eval_lib.load_program(
        program_path,
        FORBIDDEN,
        required=("order",),
        forbidden_attrs=FORBIDDEN_ATTRS,
        safe_builtins=True,
        import_budget=BUDGET,
        max_source_bytes=MAX_SOURCE_BYTES,
        max_literal_items=MAX_LITERAL_ITEMS,
        max_total_literal_items=MAX_TOTAL_LITERAL_ITEMS,
        max_string_literal_bytes=MAX_STRING_LITERAL_BYTES,
    )


def order_guarded(program_path, requests, label):
    mod = load_candidate(program_path)
    request_input = [dict(r) for r in requests]
    eval_lib.set_candidate_active(True)   # guard the direct call (outside opcount)
    opcount.start(budget=BUDGET)
    try:
        order = mod.order(request_input, dict(CONFIG))
    except opcount.BudgetExceeded:
        opcount.stop()
        eval_lib.fail(
            f"{label}: instruction budget of {BUDGET} exceeded "
            "(return a bounded deterministic heuristic)"
        )
    except BaseException as e:
        opcount.stop()
        eval_lib.fail(f"{label}: order() raised {type(e).__name__}: {e}")
    used = opcount.stop()
    eval_lib.set_candidate_active(False)
    if used > BUDGET:
        eval_lib.fail(f"{label}: instruction budget exceeded ({used} > {BUDGET})")
    return order, used


def main():
    program_path = sys.argv[1]

    for seed, n_requests in VALIDATION:
        label = f"validation seed {seed} ({n_requests} requests)"
        requests = gen_requests(seed, n_requests)
        got, _ = order_guarded(program_path, requests, label)
        score, _, _, _ = simulate(requests, got, label)
        hscore, _, _, _ = simulate(requests, heuristic_order(requests), f"validation heuristic {seed}")
        if score > hscore * 1.45:
            eval_lib.fail(
                f"validation seed {seed}: score {score:.3f} is above quality limit {hscore * 1.45:.3f}"
            )

    total = 0.0
    means = []
    p95s = []
    makespans = []
    instructions = []
    for k, requests in enumerate(scoring_traces()):
        got, used = order_guarded(program_path, requests, f"trace {k}")
        score, mean, p95, makespan = simulate(requests, got, f"trace {k}")
        total += score
        instructions.append(used)
        means.append(round(mean, 3))
        p95s.append(p95)
        makespans.append(makespan)

    eval_lib.succeed(
        round(total, 6),
        metrics={
            "weighted_mean_latency": means,
            "p95_latency": p95s,
            "makespan": makespans,
            "per_trace_instructions": instructions,
            "budget_per_trace": BUDGET,
            "config": CONFIG,
        },
    )


if __name__ == "__main__":
    main()
