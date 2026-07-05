"""Evaluator for mem_intset. Score = peak traced bytes while SERVING membership
queries (lower better) — retained set plus per-query transient.

Store a large set of integers (clustered distribution) so membership queries
are answered exactly, using as little SERVING memory as possible. Same
hardened harness as mem_kv/mem_graph: full query workload served INSIDE the
measurement window, guard across the measured region, unseen-seed validation.
"""

import gc
import random
import sys
import tracemalloc

sys.path.insert(0, __file__.rsplit("/bench/", 1)[0])
from bench import eval_lib

N_INTS = 150_000
UNIVERSE = 5_000_000
N_HIT_QUERIES = 30_000
N_MISS_QUERIES = 10_000
SEED = 0x1257
VALIDATION_SEED = 0x9E55
VALIDATION_INTS = 15_000

FORBIDDEN = frozenset(
    {
        "os", "io", "open", "mmap", "ctypes", "socket", "subprocess",
        "multiprocessing", "threading", "tempfile", "pathlib", "shutil",
        "sqlite3", "dbm", "shelve", "importlib", "__import__",
        "tracemalloc", "sys", "resource",
    }
)


def gen_ints(seed=SEED, n=N_INTS, universe=UNIVERSE):
    """Clustered integers: ~85% near a set of cluster centers (dense runs),
    ~15% uniform (sparse noise). Returns a shuffled list of distinct ints."""
    rng = random.Random(seed)
    s = set()
    centers = [rng.randrange(universe) for _ in range(max(1, n // 300))]
    while len(s) < n:
        if rng.random() < 0.85:
            x = centers[rng.randrange(len(centers))] + rng.randrange(-150, 150)
            if 0 <= x < universe:
                s.add(x)
        else:
            s.add(rng.randrange(universe))
    out = list(s)
    rng.shuffle(out)
    return out


def main():
    program_path = sys.argv[1]

    members = gen_ints()
    member_set = set(members)
    rng = random.Random(SEED + 1)
    queries = []
    for _ in range(N_HIT_QUERIES):
        queries.append((members[rng.randrange(len(members))], True))
    for _ in range(N_MISS_QUERIES):
        x = rng.randrange(UNIVERSE)
        queries.append((x, x in member_set))
    del member_set
    gc.collect()

    eval_lib.preimport(program_path)
    tracemalloc.start()
    mod = eval_lib.load_program(program_path, FORBIDDEN, required=("build", "contains"))
    eval_lib.set_candidate_active(True)
    ints = gen_ints()
    gc.disable()
    index = eval_lib.run_program(mod.build, ints)
    del ints
    # Score the SERVING peak (retained set + per-query transient), not just
    # retained bytes. reset_peak() after build charges the high-water mark of
    # answering the workload, closing the compress-then-decompress-per-query
    # trick (tiny retained blob, huge transient block on every contains()).
    # Build transients are excluded. See mem_kv for the full rationale.
    tracemalloc.reset_peak()
    # Full query workload INSIDE the window (defeats deferred construction).
    wrong = 0
    first_wrong = None
    for x, expected in queries:
        got = eval_lib.run_program(mod.contains, index, x)
        if bool(got) != expected:
            wrong += 1
            if first_wrong is None:
                first_wrong = (x, expected, got)
    got = None
    gc.enable()
    gc.collect()
    current, peak = tracemalloc.get_traced_memory()
    eval_lib.set_candidate_active(False)
    tracemalloc.stop()

    if wrong:
        x, expected, got = first_wrong
        eval_lib.fail(
            f"{wrong}/{len(queries)} membership queries wrong; first: x={x} "
            f"expected={expected} got={got!r}",
            metrics={"resident_bytes": current},
        )

    # Unseen-data validation (different seed): a program that regenerates the
    # known set instead of storing what it is given answers these wrongly.
    vmembers = gen_ints(VALIDATION_SEED, VALIDATION_INTS)
    vset = set(vmembers)
    vindex = eval_lib.run_program(mod.build, list(vmembers))
    vrng = random.Random(VALIDATION_SEED + 1)
    for _ in range(2_000):
        x = vmembers[vrng.randrange(len(vmembers))]
        if not bool(eval_lib.run_program(mod.contains, vindex, x)):
            eval_lib.fail("validation failed on unseen data (different seed): "
                          "the set must hold the integers it is given")
    for _ in range(2_000):
        x = vrng.randrange(UNIVERSE)
        if bool(eval_lib.run_program(mod.contains, vindex, x)) != (x in vset):
            eval_lib.fail("validation failed on unseen data: membership wrong "
                          "for a non-member (must not over-report)")

    eval_lib.succeed(
        float(peak),
        metrics={
            "serving_peak_bytes": peak,
            "resident_bytes": current,
            "n_ints": N_INTS,
            "universe": UNIVERSE,
            "n_queries": len(queries),
        },
    )


if __name__ == "__main__":
    main()
