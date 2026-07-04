"""Evaluator for mem_kv. Score = resident traced bytes after build (lower better)."""

import gc
import random
import sys
import tracemalloc

sys.path.insert(0, __file__.rsplit("/bench/", 1)[0])
from bench import eval_lib

N_PAIRS = 150_000
N_HIT_QUERIES = 30_000
N_MISS_QUERIES = 10_000
SEED = 0xC0FFEE
# Small unseen dataset (different seed), checked for correctness after the
# memory measurement: catches programs that re-generate the known dataset
# from its seed instead of storing what they are given.
VALIDATION_SEED = 0xFEED
VALIDATION_PAIRS = 10_000

FORBIDDEN = frozenset(
    {
        "os", "io", "open", "mmap", "ctypes", "socket", "subprocess",
        "multiprocessing", "threading", "tempfile", "pathlib", "shutil",
        "sqlite3", "dbm", "shelve", "importlib", "__import__",
        # Metric-control surfaces: a program must never touch its own
        # scorer. tracemalloc.stop()/clear_traces()/reset_peak() would
        # zero the memory score; sys is forbidden too because
        # sys.modules["tracemalloc"] reaches the same API indirectly.
        "tracemalloc", "sys", "resource",
    }
)

CATS = ["user", "order", "invoice", "session", "product", "event", "account", "ticket"]
CITIES = [
    "tokyo", "lagos", "berlin", "austin", "mumbai", "quito", "oslo", "seoul",
    "lima", "cairo", "perth", "denver", "hanoi", "turin", "boise", "dakar",
]
WORDS = [
    "alpha", "bravo", "cargo", "delta", "ember", "flint", "gamma", "harbor",
    "ivory", "jumbo", "krill", "lunar", "mango", "nylon", "ocean", "pixel",
    "quartz", "raven", "sonar", "tundra", "umber", "vivid", "willow", "xenon",
    "yield", "zephyr", "basalt", "cobalt", "drift", "easel", "fjord", "gusto",
]


def gen_pairs(seed=SEED, n_pairs=N_PAIRS):
    rng = random.Random(seed)
    pairs = []
    for i in range(n_pairs):
        cat = CATS[i % len(CATS)]
        key = f"{cat}:{i:08d}:{rng.randrange(16 ** 6):06x}"
        tags = " ".join(rng.choice(WORDS) for _ in range(rng.randrange(2, 6)))
        value = (
            f'{{"id": {i}, "city": "{rng.choice(CITIES)}", '
            f'"score": {rng.randrange(100000)}, "active": {rng.choice(["true", "false"])}, '
            f'"tags": "{tags}"}}'
        )
        pairs.append((key, value))
    return pairs


def main():
    program_path = sys.argv[1]

    # Pre-compute queries and expected answers OUTSIDE the tracing window.
    reference = gen_pairs()
    rng = random.Random(SEED + 1)
    queries = []
    for _ in range(N_HIT_QUERIES):
        k, v = reference[rng.randrange(len(reference))]
        queries.append((k, v))
    for _ in range(N_MISS_QUERIES):
        i = rng.randrange(N_PAIRS)
        queries.append((f"{CATS[i % len(CATS)]}:{i:08d}:zzzzzz", None))
    del reference
    gc.collect()

    # Measurement window: program import + data allocation + build all traced.
    # (Modules the program imports are pre-warmed outside the window so
    # module loading cannot jitter the score; see eval_lib.preimport.)
    eval_lib.preimport(program_path)
    tracemalloc.start()
    mod = eval_lib.load_program(program_path, FORBIDDEN, required=("build", "lookup"))
    # Guard ON from here — AFTER load_program has read the program file (the
    # audit hook blocks repo-file reads while the guard is active) — across
    # input generation, build, AND the post-build gc.collect(). A candidate
    # cyclic __del__ finalizer runs whenever GC collects (an auto-gc during
    # gen_pairs, or the explicit collect below) and would otherwise (guard
    # off) import tracemalloc and stop it before the score is read. Holding
    # the guard across all of it blocks that import at every collection
    # point. No allocation happens between load_program and this call, so GC
    # cannot fire in the gap. The depth-counter span encloses run_program's
    # inner span.
    eval_lib.set_candidate_active(True)
    pairs = gen_pairs()
    # Disable automatic cyclic GC during the measured build: it fires at
    # allocation-count thresholds that vary run to run, jittering the score
    # by tens of bytes; we gc.collect() deterministically after instead.
    gc.disable()
    store = eval_lib.run_program(mod.build, pairs)
    del pairs
    gc.enable()
    gc.collect()
    current, peak = tracemalloc.get_traced_memory()
    eval_lib.set_candidate_active(False)
    tracemalloc.stop()

    # Correctness after measurement.
    wrong = 0
    first_wrong = None
    for k, expected in queries:
        got = eval_lib.run_program(mod.lookup, store, k)
        if got != expected:
            wrong += 1
            if first_wrong is None:
                first_wrong = (k, expected, got)
    if wrong:
        k, expected, got = first_wrong
        eval_lib.fail(
            f"{wrong}/{len(queries)} lookups wrong; first: key={k!r} "
            f"expected={expected!r} got={str(got)[:120]!r}",
            metrics={"resident_bytes": current},
        )

    # Unseen-data validation (after measurement; memory here is unscored):
    # a program that re-generates the known dataset instead of storing what
    # it is given answers these wrongly.
    vpairs = gen_pairs(VALIDATION_SEED, VALIDATION_PAIRS)
    vstore = eval_lib.run_program(mod.build, list(vpairs))
    vrng = random.Random(VALIDATION_SEED + 1)
    for _ in range(2_000):
        k, v = vpairs[vrng.randrange(len(vpairs))]
        if eval_lib.run_program(mod.lookup, vstore, k) != v:
            eval_lib.fail(
                "validation failed on unseen data (different seed): the store "
                "must hold the pairs it is given, not answers specialized to "
                "the scoring dataset"
            )
    if eval_lib.run_program(mod.lookup, vstore, "user:99999999:zzzzzz") is not None:
        eval_lib.fail("validation failed on unseen data: missing key must return None")

    eval_lib.succeed(
        float(current),
        metrics={
            "resident_bytes": current,
            "peak_bytes_during_build": peak,
            "n_pairs": N_PAIRS,
            "n_queries": len(queries),
        },
    )


if __name__ == "__main__":
    main()
