"""Evaluator for mem_str. Score = resident traced bytes after build (lower better).

Store a list of strings (with heavy duplication and shared prefixes) so each
can be retrieved EXACTLY by its index, using as little retained memory as
possible. Same hardened harness as mem_kv/mem_graph: the full retrieval
workload runs INSIDE the measurement window, the guard spans the measured
region, and unseen-data validation (different seed) catches regenerate/hardcode.
"""

import gc
import random
import sys
import tracemalloc

sys.path.insert(0, __file__.rsplit("/bench/", 1)[0])
from bench import eval_lib

N = 100_000
N_QUERIES = 40_000
SEED = 0x57A2
VALIDATION_SEED = 0xC0DE
VALIDATION_N = 12_000

PREFIXES = [
    "/api/v1/users", "/api/v2/orders", "com.example.service.handler",
    "us-east-1/prod/bucket", "org.apache.commons.lang", "/var/log/app/worker",
    "cdn.assets.example.com/static", "internal.rpc.v3.gateway",
]
WORDS = [
    "alpha", "bravo", "cargo", "delta", "ember", "flint", "gamma", "harbor",
    "ivory", "jumbo", "krill", "lunar", "mango", "nylon", "ocean", "pixel",
    "quartz", "raven", "sonar", "tundra", "umber", "vivid", "willow", "xenon",
    "yield", "zephyr", "basalt", "cobalt", "drift", "easel", "fjord", "gusto",
]

FORBIDDEN = frozenset(
    {
        "os", "io", "open", "mmap", "ctypes", "socket", "subprocess",
        "multiprocessing", "threading", "tempfile", "pathlib", "shutil",
        "sqlite3", "dbm", "shelve", "importlib", "__import__",
        "tracemalloc", "sys", "resource",
    }
)


def gen_strings(seed=SEED, n=N):
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        p = PREFIXES[rng.randrange(len(PREFIXES))]
        w = WORDS[rng.randrange(len(WORDS))]
        k = rng.randrange(60)   # small range -> heavy duplication
        out.append(p + "/" + w + "/" + str(k))   # fresh object every time
    return out


def main():
    program_path = sys.argv[1]

    strings = gen_strings()
    rng = random.Random(SEED + 1)
    queries = [rng.randrange(N) for _ in range(N_QUERIES)]
    expected = [strings[i] for i in queries]
    del strings
    gc.collect()

    eval_lib.preimport(program_path)
    tracemalloc.start()
    mod = eval_lib.load_program(program_path, FORBIDDEN, required=("build", "get"))
    eval_lib.set_candidate_active(True)
    data = gen_strings()
    gc.disable()
    index = eval_lib.run_program(mod.build, data)
    del data
    # Full retrieval workload INSIDE the window (defeats deferred construction).
    wrong = 0
    first_wrong = None
    for qi, want in zip(queries, expected):
        got = eval_lib.run_program(mod.get, index, qi)
        if got != want:
            wrong += 1
            if first_wrong is None:
                first_wrong = (qi, want, got)
    got = None
    gc.enable()
    gc.collect()
    current, peak = tracemalloc.get_traced_memory()
    eval_lib.set_candidate_active(False)
    tracemalloc.stop()

    if wrong:
        qi, want, got = first_wrong
        eval_lib.fail(
            f"{wrong}/{len(queries)} retrievals wrong; first: index {qi} "
            f"expected {want!r} got {str(got)[:80]!r}",
            metrics={"resident_bytes": current},
        )

    # Unseen-data validation (different seed): a program that regenerates the
    # known strings instead of storing what it is given answers these wrongly.
    vstrings = gen_strings(VALIDATION_SEED, VALIDATION_N)
    vindex = eval_lib.run_program(mod.build, list(vstrings))
    vrng = random.Random(VALIDATION_SEED + 1)
    for _ in range(2_000):
        j = vrng.randrange(VALIDATION_N)
        if eval_lib.run_program(mod.get, vindex, j) != vstrings[j]:
            eval_lib.fail("validation failed on unseen data (different seed): "
                          "the store must hold the strings it is given")

    eval_lib.succeed(
        float(current),
        metrics={
            "resident_bytes": current,
            "peak_bytes_during_build": peak,
            "n_strings": N,
            "n_queries": len(queries),
        },
    )


if __name__ == "__main__":
    main()
