"""Evaluator for spec_decode_plan. Score = expected speculative-decoding cost."""

import random
import sys

sys.path.insert(0, __file__.rsplit("/bench/", 1)[0])
from bench import eval_lib, opcount

SEED = 0x5D3C
N_TRACES = 7
VALIDATION = [(0x5D01, 18), (0x5D02, 24), (0x5D03, 30)]
BUDGET = 25_000_000
MAX_SOURCE_BYTES = 12_000
MAX_LITERAL_ITEMS = 80
MAX_TOTAL_LITERAL_ITEMS = 300
MAX_STRING_LITERAL_BYTES = 2_000
CONFIG = {
    "max_draft": 8,
    "target_base": 86.0,
    "target_per_token": 5.5,
    "draft_base": 7.0,
    "draft_per_token": 3.2,
    "stall_penalty": 0.08,
}

FORBIDDEN = frozenset({
    "sys", "os", "io", "open", "mmap", "ctypes", "socket", "subprocess",
    "multiprocessing", "threading", "tempfile", "pathlib", "shutil",
    "sqlite3", "dbm", "shelve", "importlib", "signal", "inspect",
    "time", "resource",
    "builtins", "__builtins__", "bench", "__import__", "eval", "exec", "compile",
    "globals", "locals", "vars", "dir", "getattr", "setattr",
    "delattr", "type", "object", "super", "input", "breakpoint",
})
FORBIDDEN_ATTRS = frozenset({
    "__class__", "__dict__", "__globals__", "__code__", "__closure__",
    "__mro__", "__subclasses__", "__getattribute__", "__builtins__",
    "__traceback__", "tb_frame", "tb_next", "f_back", "f_globals",
    "f_locals", "gi_frame", "cr_frame",
})


def gen_requests(seed, n):
    rng = random.Random(seed)
    reqs = []
    for i in range(n):
        length = rng.randint(42, 180)
        if rng.random() < 0.18:
            length += rng.randint(120, 260)
        base = rng.uniform(0.45, 0.88)
        volatility = rng.uniform(0.04, 0.18)
        topic_shift = rng.randint(18, 55)
        acc = []
        for pos in range(length + CONFIG["max_draft"] + 2):
            drift = 0.08 * ((pos % topic_shift) / max(1, topic_shift - 1) - 0.5)
            if rng.random() < 0.08:
                local = base - rng.uniform(0.12, 0.30)
            elif rng.random() < 0.10:
                local = base + rng.uniform(0.05, 0.14)
            else:
                local = base + rng.uniform(-volatility, volatility) + drift
            local = max(0.08, min(0.96, local))
            acc.append(round(local, 4))
        draft_scale = rng.uniform(0.78, 1.32)
        verify_scale = rng.uniform(0.86, 1.18)
        reqs.append({
            "id": i,
            "output_tokens": length,
            "accept": acc,
            "draft_scale": round(draft_scale, 4),
            "verify_scale": round(verify_scale, 4),
        })
    return reqs


def scoring_traces():
    return [gen_requests(SEED + i * 173, 20 + i * 5) for i in range(N_TRACES)]


def round_cost(k, req, config):
    target = (config["target_base"] + config["target_per_token"] * (k + 1)) * req["verify_scale"]
    draft = (config["draft_base"] + config["draft_per_token"] * k) * req["draft_scale"]
    stall = config["stall_penalty"] * k * k
    return target + draft + stall


def expected_request_cost(req, policy, config, label):
    n = req["output_tokens"]
    max_draft = config["max_draft"]
    if not isinstance(policy, list) or len(policy) != n:
        eval_lib.fail(f"{label}: each policy must contain one draft length per token position")
    value = [0.0] * (n + max_draft + 2)
    acc = req["accept"]
    for pos in range(n - 1, -1, -1):
        k = policy[pos]
        if not isinstance(k, int) or k < 1 or k > max_draft:
            eval_lib.fail(f"{label}: draft lengths must be integers in [1, {max_draft}]")
        rem = n - pos
        if k > rem:
            k = rem
        cost = round_cost(k, req, config)
        prob_prefix = 1.0
        future = 0.0
        for accepted in range(k):
            reject_p = prob_prefix * (1.0 - acc[pos + accepted])
            future += reject_p * value[min(n, pos + accepted + 1)]
            prob_prefix *= acc[pos + accepted]
        future += prob_prefix * value[min(n, pos + k + 1)]
        value[pos] = cost + future
    return value[0]


def heuristic_policy(req, config):
    n = req["output_tokens"]
    out = []
    for pos in range(n):
        p = req["accept"][pos]
        if p > 0.82:
            k = 7
        elif p > 0.68:
            k = 5
        elif p > 0.52:
            k = 3
        else:
            k = 1
        out.append(min(k, n - pos))
    return out


def load_candidate(program_path):
    return eval_lib.load_program(
        program_path,
        FORBIDDEN,
        required=("plan",),
        forbidden_attrs=FORBIDDEN_ATTRS,
        safe_builtins=True,
        import_budget=BUDGET,
        max_source_bytes=MAX_SOURCE_BYTES,
        max_literal_items=MAX_LITERAL_ITEMS,
        max_total_literal_items=MAX_TOTAL_LITERAL_ITEMS,
        max_string_literal_bytes=MAX_STRING_LITERAL_BYTES,
    )


def plan_guarded(program_path, requests, label):
    mod = load_candidate(program_path)
    req_input = []
    for r in requests:
        item = dict(r)
        item["accept"] = list(r["accept"])
        req_input.append(item)
    opcount.start(budget=BUDGET)
    try:
        policies = mod.plan(req_input, dict(CONFIG))
    except opcount.BudgetExceeded:
        opcount.stop()
        eval_lib.fail(f"{label}: instruction budget of {BUDGET} exceeded")
    except BaseException as e:
        opcount.stop()
        eval_lib.fail(f"{label}: plan() raised {type(e).__name__}: {e}")
    used = opcount.stop()
    if not isinstance(policies, list) or len(policies) != len(requests):
        eval_lib.fail(f"{label}: plan() must return one policy list per request")
    return policies, used


def score_policies(requests, policies, label):
    total = 0.0
    per_req = []
    for req, policy in zip(requests, policies):
        cost = expected_request_cost(req, policy, CONFIG, label)
        total += cost
        per_req.append(cost)
    return total / len(requests), per_req


def main():
    program_path = sys.argv[1]

    for seed, n in VALIDATION:
        requests = gen_requests(seed, n)
        label = f"validation seed {seed}"
        policies, _ = plan_guarded(program_path, requests, label)
        score, _ = score_policies(requests, policies, label)
        hpol = [heuristic_policy(r, CONFIG) for r in requests]
        hscore, _ = score_policies(requests, hpol, f"validation heuristic {seed}")
        if score > hscore * 1.18:
            eval_lib.fail(f"{label}: score {score:.3f} is above quality limit {hscore * 1.18:.3f}")

    total = 0.0
    trace_scores = []
    instructions = []
    for idx, requests in enumerate(scoring_traces()):
        label = f"trace {idx}"
        policies, used = plan_guarded(program_path, requests, label)
        score, _ = score_policies(requests, policies, label)
        total += score
        trace_scores.append(round(score, 4))
        instructions.append(used)

    eval_lib.succeed(
        round(total, 6),
        metrics={
            "trace_expected_cost": trace_scores,
            "plan_instructions": instructions,
            "budget_per_trace": BUDGET,
            "config": CONFIG,
        },
    )


if __name__ == "__main__":
    main()
