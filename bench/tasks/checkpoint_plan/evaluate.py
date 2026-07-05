"""Evaluator for checkpoint_plan. Score = recompute cost under memory caps."""

import random
import sys

sys.path.insert(0, __file__.rsplit("/bench/", 1)[0])
from bench import eval_lib, opcount

SEED = 0xC4EC
N_PROFILES = 10
# Unseen-data validation: MANY fresh-seed profiles spanning the full scoring
# size range, each run under the SAME instruction budget. A plan() that
# overfits to the fixed scoring instances (branching on size/budget as an
# instance id, or tuning search effort to land just under the budget on those
# specific instances) fails here — it either exceeds the budget on a fresh
# instance or misroutes to a bad plan. A general bounded search passes.
VALIDATION_SEED_BASE = 0xDA7A
VALIDATION_SIZES = [20, 24, 28, 32, 36]
VALIDATION_PER_SIZE = 3
# The overfit winners are caught by the instruction-budget enforcement on the
# fresh validation instances (below), not by this quality floor, so keep the
# floor loose enough that the honest baseline heuristic passes.
VALIDATION_QUALITY_MULT = 2.5
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


def gen_profile(seed, n_layers):
    rng = random.Random(seed)
    layers = []
    width = rng.choice([768, 1024, 1280, 1536])
    seq = rng.choice([512, 1024, 1536, 2048])
    for i in range(n_layers):
        attn = 1 + (i % 4 == 0)
        mlp = 1 + (i % 3 == 1)
        act = int((width * seq) / 9000) + rng.randint(30, 160) + attn * rng.randint(50, 130)
        cost = int((width * width * seq) / 1_200_000) + rng.randint(80, 260) + mlp * 90
        layers.append({"act_mem": act, "fwd_cost": cost, "kind": "attn" if attn > 1 else "mlp"})
    full_mem = sum(l["act_mem"] for l in layers)
    budget = int(full_mem * rng.uniform(0.55, 0.72))
    return layers, budget


def scoring_profiles():
    return [gen_profile(SEED + i * 97, 20 + (i % 5) * 4) for i in range(N_PROFILES)]


def validation_profiles():
    sizes = VALIDATION_SIZES * VALIDATION_PER_SIZE
    return [(gen_profile(VALIDATION_SEED_BASE + j * 131, n), n)
            for j, n in enumerate(sizes)]


def evaluate_plan(layers, memory_budget, boundaries, label):
    n = len(layers)
    if not isinstance(boundaries, list):
        eval_lib.fail(f"{label}: plan() must return a list[int]")
    try:
        b = sorted(set(int(x) for x in boundaries))
    except Exception:
        eval_lib.fail(f"{label}: checkpoint boundaries must be integers")
    b = sorted(set([0, n] + b))
    if b[0] != 0 or b[-1] != n or any(x < 0 or x > n for x in b):
        eval_lib.fail(f"{label}: boundaries must be in 0..{n}")
    stored = sum(layers[i - 1]["act_mem"] for i in b if 0 < i < n)
    max_segment = 0
    recompute = 0
    for a, c in zip(b, b[1:]):
        if c <= a:
            eval_lib.fail(f"{label}: boundaries must be strictly increasing")
        seg_mem = sum(l["act_mem"] for l in layers[a:c])
        max_segment = max(max_segment, seg_mem)
        # If every layer boundary is stored, backward can use stored
        # activations directly. Longer segments recompute the interior.
        recompute += sum(l["fwd_cost"] for l in layers[a : max(a, c - 1)])
    peak = stored + max_segment
    if peak > memory_budget:
        eval_lib.fail(f"{label}: peak activation memory {peak} exceeds budget {memory_budget}")
    return recompute, peak, b


def greedy_boundaries(layers, memory_budget):
    n = len(layers)
    # Small deterministic search over target segment counts.
    best = None
    for nseg in range(2, min(n, 16) + 1):
        target = sum(l["act_mem"] for l in layers) / nseg
        b = [0]
        acc = 0
        for i, layer in enumerate(layers):
            acc += layer["act_mem"]
            if acc >= target and i + 1 < n:
                b.append(i + 1)
                acc = 0
        b.append(n)
        b = sorted(set([0, len(layers)] + b))
        stored = sum(layers[i - 1]["act_mem"] for i in b if 0 < i < len(layers))
        max_segment = 0
        for a, c in zip(b, b[1:]):
            max_segment = max(max_segment, sum(l["act_mem"] for l in layers[a:c]))
        peak = stored + max_segment
        if peak > memory_budget:
            continue
        score = sum(
            sum(l["fwd_cost"] for l in layers[a : max(a, c - 1)])
            for a, c in zip(b, b[1:])
        )
        out = b
        cand = (score, peak, out)
        if best is None or cand < best:
            best = cand
    return best[2] if best else list(range(n + 1))


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


def plan_guarded(program_path, layers, budget, label):
    mod = load_candidate(program_path)
    layer_input = [dict(l) for l in layers]
    eval_lib.set_candidate_active(True)   # guard the direct call (outside opcount)
    opcount.start(budget=BUDGET)
    try:
        got = mod.plan(layer_input, budget)
    except opcount.BudgetExceeded:
        opcount.stop()
        eval_lib.fail(
            f"{label}: instruction budget of {BUDGET} exceeded "
            "(return a bounded deterministic heuristic)"
        )
    except BaseException as e:
        opcount.stop()
        eval_lib.fail(f"{label}: plan() raised {type(e).__name__}: {e}")
    used = opcount.stop()
    eval_lib.set_candidate_active(False)
    if used > BUDGET:
        eval_lib.fail(f"{label}: instruction budget exceeded ({used} > {BUDGET})")
    return got, used


def main():
    program_path = sys.argv[1]

    for j, ((layers, budget), n_layers) in enumerate(validation_profiles()):
        label = f"validation {j} ({n_layers} layers)"
        # plan_guarded enforces the instruction budget on this fresh instance:
        # an effort-overfit plan() that exceeds it here is rejected.
        got, _ = plan_guarded(program_path, layers, budget, label)
        score, _, _ = evaluate_plan(layers, budget, got, label)
        hscore, _, _ = evaluate_plan(layers, budget,
                                     greedy_boundaries(layers, budget),
                                     f"validation heuristic {j}")
        if score > hscore * VALIDATION_QUALITY_MULT:
            eval_lib.fail(
                f"validation {j} ({n_layers} layers): recompute cost {score} "
                f"is above quality limit {int(hscore * VALIDATION_QUALITY_MULT)} "
                f"— plan must generalize to unseen profiles, not specialize to "
                f"the scoring set"
            )

    total = 0
    recompute_costs = []
    peaks = []
    budgets = []
    n_checkpoints = []
    instructions = []
    for k, (layers, budget) in enumerate(scoring_profiles()):
        got, used = plan_guarded(program_path, layers, budget, f"profile {k}")
        score, peak, b = evaluate_plan(layers, budget, got, f"profile {k}")
        total += score
        instructions.append(used)
        recompute_costs.append(score)
        peaks.append(peak)
        budgets.append(budget)
        n_checkpoints.append(len(b))

    eval_lib.succeed(
        float(total),
        metrics={
            "recompute_cost_per_profile": recompute_costs,
            "peak_memory_per_profile": peaks,
            "memory_budget_per_profile": budgets,
            "n_boundaries": n_checkpoints,
            "per_profile_instructions": instructions,
            "budget_per_profile": BUDGET,
        },
    )


if __name__ == "__main__":
    main()
