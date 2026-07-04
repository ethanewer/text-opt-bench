"""Evaluator for spec_tree_select. Score = expected tree-verification cost per token."""

import random
import sys

sys.path.insert(0, __file__.rsplit("/bench/", 1)[0])
from bench import eval_lib, opcount

SEED = 0x71EE
N_TRACES = 7
VALIDATION = [(0x7101, 12), (0x7102, 16), (0x7103, 20)]
BUDGET = 30_000_000
MAX_SOURCE_BYTES = 12_000
MAX_LITERAL_ITEMS = 80
MAX_TOTAL_LITERAL_ITEMS = 300
MAX_STRING_LITERAL_BYTES = 2_000
CONFIG = {
    "max_nodes": 28,
    "max_depth": 6,
    "verify_base": 92.0,
    "verify_per_node": 4.7,
    "draft_per_node": 1.8,
    "depth_overhead": 2.4,
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


def gen_tree(seed):
    rng = random.Random(seed)
    nodes = []
    frontier = [(None, 0, 1.0, rng.uniform(0.58, 0.88))]
    next_id = 0
    while frontier and len(nodes) < 95:
        parent, depth, path_prob, quality = frontier.pop(0)
        if depth >= CONFIG["max_depth"]:
            continue
        if parent is None:
            width = rng.randint(3, 6)
        elif depth < 2:
            width = rng.randint(2, 5)
        else:
            width = rng.randint(1, 4)
        raw = []
        for _ in range(width):
            raw.append(max(0.01, rng.expovariate(1.0) * quality))
        # Draft candidates cover only part of the target distribution; keep
        # sibling edge probabilities as a proper subdistribution.
        z = sum(raw) / rng.uniform(0.62, 0.96)
        probs = [min(0.94, x / z) for x in raw]
        probs.sort(reverse=True)
        for rank, edge_p in enumerate(probs):
            if len(nodes) >= 95:
                break
            p = path_prob * edge_p
            nid = next_id
            next_id += 1
            node = {
                "id": nid,
                "parent": parent if parent is not None else -1,
                "depth": depth + 1,
                "edge_prob": round(edge_p, 5),
                "path_prob": round(p, 7),
                "rank": rank,
            }
            nodes.append(node)
            if p > 0.002 and depth + 1 < CONFIG["max_depth"]:
                child_quality = max(0.18, min(0.92, quality * rng.uniform(0.72, 1.04)))
                frontier.append((nid, depth + 1, p, child_quality))
    return nodes


def gen_trace(seed, n):
    return [{"id": i, "nodes": gen_tree(seed + i * 37)} for i in range(n)]


def scoring_traces():
    return [gen_trace(SEED + i * 211, 12 + i * 3) for i in range(N_TRACES)]


def heuristic_select(tree):
    nodes = tree["nodes"]
    children = {}
    by_id = {}
    for node in nodes:
        by_id[node["id"]] = node
        children.setdefault(node["parent"], []).append(node)
    selected = set()
    frontier = list(children.get(-1, ()))
    while frontier and len(selected) < CONFIG["max_nodes"]:
        best_i = 0
        best_score = -1.0
        for i, node in enumerate(frontier):
            score = node["path_prob"] / (1.0 + 0.08 * node["depth"] + 0.05 * node["rank"])
            if score > best_score:
                best_score = score
                best_i = i
        node = frontier.pop(best_i)
        selected.add(node["id"])
        for child in children.get(node["id"], ()):
            frontier.append(child)
    return sorted(selected)


def score_selection(tree, selected_ids, label):
    if not isinstance(selected_ids, list) or any(not isinstance(x, int) for x in selected_ids):
        eval_lib.fail(f"{label}: select() must return lists of integer node ids")
    if len(selected_ids) > CONFIG["max_nodes"]:
        eval_lib.fail(f"{label}: selected {len(selected_ids)} nodes over cap {CONFIG['max_nodes']}")
    if len(set(selected_ids)) != len(selected_ids):
        eval_lib.fail(f"{label}: selected node ids must be unique")
    by_id = {n["id"]: n for n in tree["nodes"]}
    selected = set(selected_ids)
    gain = 0.0
    max_depth = 0
    for nid in selected_ids:
        node = by_id.get(nid)
        if node is None:
            eval_lib.fail(f"{label}: selected unknown node id {nid}")
        parent = node["parent"]
        if parent != -1 and parent not in selected:
            eval_lib.fail(f"{label}: selected tree must be prefix-closed")
        gain += node["path_prob"]
        if node["depth"] > max_depth:
            max_depth = node["depth"]
    if not selected:
        return CONFIG["verify_base"], 0.0, 0
    cost = (
        CONFIG["verify_base"]
        + CONFIG["verify_per_node"] * len(selected)
        + CONFIG["draft_per_node"] * len(selected)
        + CONFIG["depth_overhead"] * max_depth * max_depth
    )
    expected_tokens = 1.0 + gain
    return cost / expected_tokens, gain, max_depth


def load_candidate(program_path):
    return eval_lib.load_program(
        program_path,
        FORBIDDEN,
        required=("select",),
        forbidden_attrs=FORBIDDEN_ATTRS,
        safe_builtins=True,
        import_budget=BUDGET,
        max_source_bytes=MAX_SOURCE_BYTES,
        max_literal_items=MAX_LITERAL_ITEMS,
        max_total_literal_items=MAX_TOTAL_LITERAL_ITEMS,
        max_string_literal_bytes=MAX_STRING_LITERAL_BYTES,
    )


def select_guarded(program_path, trace, label):
    mod = load_candidate(program_path)
    trace_input = [{"id": t["id"], "nodes": [dict(n) for n in t["nodes"]]} for t in trace]
    opcount.start(budget=BUDGET)
    try:
        selections = mod.select(trace_input, dict(CONFIG))
    except opcount.BudgetExceeded:
        opcount.stop()
        eval_lib.fail(f"{label}: instruction budget of {BUDGET} exceeded")
    except BaseException as e:
        opcount.stop()
        eval_lib.fail(f"{label}: select() raised {type(e).__name__}: {e}")
    used = opcount.stop()
    if not isinstance(selections, list) or len(selections) != len(trace):
        eval_lib.fail(f"{label}: select() must return one node-id list per tree")
    return selections, used


def score_trace(trace, selections, label):
    total = 0.0
    gains = []
    depths = []
    for tree, selected in zip(trace, selections):
        score, gain, depth = score_selection(tree, selected, label)
        total += score
        gains.append(gain)
        depths.append(depth)
    return total / len(trace), gains, depths


def main():
    program_path = sys.argv[1]

    for seed, n in VALIDATION:
        trace = gen_trace(seed, n)
        label = f"validation seed {seed}"
        selections, _ = select_guarded(program_path, trace, label)
        score, _, _ = score_trace(trace, selections, label)
        hsel = [heuristic_select(tree) for tree in trace]
        hscore, _, _ = score_trace(trace, hsel, f"validation heuristic {seed}")
        if score > hscore * 1.18:
            eval_lib.fail(f"{label}: score {score:.3f} is above quality limit {hscore * 1.18:.3f}")

    total = 0.0
    trace_scores = []
    avg_gains = []
    avg_depths = []
    instructions = []
    for idx, trace in enumerate(scoring_traces()):
        label = f"trace {idx}"
        selections, used = select_guarded(program_path, trace, label)
        score, gains, depths = score_trace(trace, selections, label)
        total += score
        trace_scores.append(round(score, 4))
        avg_gains.append(round(sum(gains) / len(gains), 5))
        avg_depths.append(round(sum(depths) / len(depths), 3))
        instructions.append(used)

    eval_lib.succeed(
        round(total, 6),
        metrics={
            "trace_cost_per_token": trace_scores,
            "avg_expected_extra_tokens": avg_gains,
            "avg_tree_depth": avg_depths,
            "select_instructions": instructions,
            "budget_per_trace": BUDGET,
            "config": CONFIG,
        },
    )


if __name__ == "__main__":
    main()
