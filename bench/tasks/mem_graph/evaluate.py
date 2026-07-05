"""Evaluator for mem_graph. Score = resident traced bytes after build (lower better).

Store a directed graph (given as an edge list) so out-neighbor queries are
answered exactly, using as little retained memory as possible. Mirrors the
hardened mem_kv harness: the full query workload is served INSIDE the
measurement window (so a marker-returning build() that defers construction is
still measured), the import/file guard spans the whole measured region, and
unseen-data validation (different seed) catches regenerate/hardcode.
"""

import gc
import random
import sys
import tracemalloc

sys.path.insert(0, __file__.rsplit("/bench/", 1)[0])
from bench import eval_lib

N_NODES = 30_000
M_EDGES = 300_000
N_HIT_QUERIES = 30_000
N_MISS_QUERIES = 10_000
SEED = 0x6A6A
# Small unseen dataset (different seed), checked for correctness after the
# memory measurement: catches programs that re-generate the known graph from
# its seed instead of storing what they are given.
VALIDATION_SEED = 0xB0B0
VALIDATION_NODES = 4_000
VALIDATION_EDGES = 40_000

FORBIDDEN = frozenset(
    {
        "os", "io", "open", "mmap", "ctypes", "socket", "subprocess",
        "multiprocessing", "threading", "tempfile", "pathlib", "shutil",
        "sqlite3", "dbm", "shelve", "importlib", "__import__",
        # Metric-control surfaces: a program must never touch its own scorer.
        "tracemalloc", "sys", "resource",
    }
)


def gen_edges(seed=SEED, n_nodes=N_NODES, m_edges=M_EDGES):
    """Directed edges with duplicates and hub skew (compressible structure)."""
    rng = random.Random(seed)
    n_hub = max(1, n_nodes // 20)
    edges = []
    for _ in range(m_edges):
        u = rng.randrange(n_nodes)
        v = rng.randrange(n_hub) if rng.random() < 0.3 else rng.randrange(n_nodes)
        edges.append((u, v))
    return edges


def build_reference(edges):
    adj = {}
    for u, v in edges:
        s = adj.get(u)
        if s is None:
            adj[u] = s = set()
        s.add(v)
    return {u: sorted(s) for u, s in adj.items()}


def main():
    program_path = sys.argv[1]

    # Pre-compute queries + expected answers OUTSIDE the tracing window.
    ref = build_reference(gen_edges())
    rng = random.Random(SEED + 1)
    nodes_with_edges = list(ref.keys())
    queries = []
    for _ in range(N_HIT_QUERIES):
        u = nodes_with_edges[rng.randrange(len(nodes_with_edges))]
        queries.append((u, ref[u]))
    for _ in range(N_MISS_QUERIES):
        u = rng.randrange(N_NODES)
        queries.append((u, ref.get(u, [])))
    del ref
    gc.collect()

    eval_lib.preimport(program_path)
    tracemalloc.start()
    mod = eval_lib.load_program(program_path, FORBIDDEN, required=("build", "neighbors"))
    # Guard ON from here — AFTER load_program has read the program file —
    # across input generation, build, and the post-build gc.collect(), so a
    # candidate finalizer collected at any gc point can't import and stop
    # tracemalloc. No allocation happens between load_program and this call.
    eval_lib.set_candidate_active(True)
    edges = gen_edges()
    # Disable automatic cyclic GC during the measured build (fires at
    # allocation-count thresholds that jitter the score); gc.collect() after.
    gc.disable()
    index = eval_lib.run_program(mod.build, edges)
    del edges
    # Serve the FULL query workload INSIDE the measurement window, so a
    # marker-returning build() that defers real construction (or a
    # regenerate-and-cache) to the first neighbors() call is still measured.
    # Results are discarded; correctness recorded and checked after the sample.
    wrong = 0
    first_wrong = None
    for u, expected in queries:
        got = eval_lib.run_program(mod.neighbors, index, u)
        if not isinstance(got, list) or got != expected:
            wrong += 1
            if first_wrong is None:
                first_wrong = (u, expected, got)
    got = None
    gc.enable()
    gc.collect()
    current, peak = tracemalloc.get_traced_memory()
    eval_lib.set_candidate_active(False)
    tracemalloc.stop()

    if wrong:
        u, expected, got = first_wrong
        eval_lib.fail(
            f"{wrong}/{len(queries)} neighbor queries wrong; first: node={u} "
            f"expected {len(expected) if isinstance(expected, list) else '?'} "
            f"ids got={str(got)[:120]!r}",
            metrics={"resident_bytes": current},
        )

    # Unseen-data validation (after measurement; memory here is unscored): a
    # program that re-generates the known graph answers these wrongly.
    vedges = gen_edges(VALIDATION_SEED, VALIDATION_NODES, VALIDATION_EDGES)
    vref = build_reference(vedges)
    vindex = eval_lib.run_program(mod.build, list(vedges))
    vrng = random.Random(VALIDATION_SEED + 1)
    vkeys = list(vref.keys())
    for _ in range(2_000):
        u = vkeys[vrng.randrange(len(vkeys))]
        if eval_lib.run_program(mod.neighbors, vindex, u) != vref[u]:
            eval_lib.fail(
                "validation failed on unseen data (different seed): the index "
                "must hold the edges it is given, not answers specialized to "
                "the scoring graph"
            )
    if eval_lib.run_program(mod.neighbors, vindex, VALIDATION_NODES + 12345) != []:
        eval_lib.fail("validation failed on unseen data: unknown node must return []")

    eval_lib.succeed(
        float(current),
        metrics={
            "resident_bytes": current,
            "peak_bytes_during_build": peak,
            "n_nodes": N_NODES,
            "n_edges": M_EDGES,
            "n_queries": len(queries),
        },
    )


if __name__ == "__main__":
    main()
