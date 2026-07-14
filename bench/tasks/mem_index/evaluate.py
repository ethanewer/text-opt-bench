"""Evaluator for mem_index. Score = peak traced bytes while SERVING queries
(lower better) — retained inverted index plus per-query transient. Charging the
serving peak rewards structures cheap to both hold and query."""

import gc
import random
import string
import sys
import tracemalloc

sys.path.insert(0, __file__.rsplit("/bench/", 1)[0])
from bench import eval_lib

N_DOCS = 20_000
VOCAB_SIZE = 25_000
N_QUERIES_PRESENT = 3_000
N_QUERIES_ABSENT = 1_000
SEED = 0x1DEA
# Small unseen dataset (different seed), checked for correctness after the
# memory measurement: catches programs that re-generate the known dataset
# from its seed instead of indexing what they are given.
VALIDATION_SEED = 0xABBA
VALIDATION_DOCS = 2_000

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


def gen_vocab():
    rng = random.Random(SEED)
    vocab = []
    seen = set()
    while len(vocab) < VOCAB_SIZE:
        w = "".join(rng.choice(string.ascii_lowercase) for _ in range(rng.randrange(3, 11)))
        if w not in seen:
            seen.add(w)
            vocab.append(w)
    # Zipf-like sampling weights: weight of rank r is 1 / (r + 5).
    cum = []
    total = 0.0
    for r in range(VOCAB_SIZE):
        total += 1.0 / (r + 5)
        cum.append(total)
    return vocab, cum


def gen_docs(vocab, cum, seed=SEED + 1, n_docs=N_DOCS):
    rng = random.Random(seed)
    docs = []
    for _ in range(n_docs):
        n_words = rng.randrange(20, 101)
        words = rng.choices(vocab, cum_weights=cum, k=n_words)
        docs.append(" ".join(words))
    return docs


def build_reference(docs):
    ref = {}
    for doc_id, doc in enumerate(docs):
        for word in set(doc.split()):
            ref.setdefault(word, []).append(doc_id)
    return {w: sorted(ids) for w, ids in ref.items()}


def main():
    program_path = sys.argv[1]

    vocab, cum = gen_vocab()

    # Reference answers computed OUTSIDE the tracing window.
    ref = build_reference(gen_docs(vocab, cum))
    rng = random.Random(SEED + 2)
    present = list(ref.keys())
    queries = []
    for _ in range(N_QUERIES_PRESENT):
        w = present[rng.randrange(len(present))]
        queries.append((w, ref[w]))
    n_absent = 0
    while n_absent < N_QUERIES_ABSENT:
        w = "".join(rng.choice(string.ascii_lowercase) for _ in range(rng.randrange(3, 11)))
        if w not in ref:
            queries.append((w, []))
            n_absent += 1
    del ref
    gc.collect()

    # Measurement window.
    eval_lib.preimport(program_path)
    tracemalloc.start()
    mod = eval_lib.load_program(program_path, FORBIDDEN, required=("build", "query"))
    # Guard ON from here — AFTER load_program has read the program file (the
    # audit hook blocks repo-file reads while the guard is active) — across
    # input generation, build, AND the post-build gc.collect(), so a
    # candidate cyclic __del__ finalizer collected at ANY gc point (incl. an
    # auto-gc during gen_docs) can't import tracemalloc and stop it before
    # the score is read. No allocation happens between load_program and this
    # call, so GC cannot fire in the gap.
    eval_lib.set_candidate_active(True)
    docs = gen_docs(vocab, cum)
    # Disable automatic cyclic GC during the measured build: it fires at
    # allocation-count thresholds that vary run to run, jittering the score
    # by tens of bytes; we gc.collect() deterministically after instead.
    gc.disable()
    index = eval_lib.run_program(mod.build, docs)
    del docs
    # Score the SERVING peak (retained index + per-query transient), not just
    # retained bytes. reset_peak() after build charges the high-water mark of
    # answering the workload, closing the compress-then-decompress-per-query
    # trick (tiny retained blob, huge transient block on every query). Build
    # transients are excluded.
    tracemalloc.reset_peak()
    # Serve the FULL query workload INSIDE the measurement window. This also
    # defeats deferred construction: a build() that returns a marker and defers
    # the real inverted-index (regenerate-and-cache) to the first query() is
    # forced to build and retain it here, inside the window. Results are
    # discarded; the first wrong result is recorded and reported after the
    # sample.
    first_wrong = None
    for term, expected in queries:
        got = eval_lib.run_program(mod.query, index, term)
        if first_wrong is None and (not isinstance(got, list) or got != expected):
            first_wrong = (term, expected, got)
    got = None  # don't retain the last query result in the sample
    gc.enable()
    gc.collect()
    current, peak = tracemalloc.get_traced_memory()
    eval_lib.set_candidate_active(False)
    tracemalloc.stop()

    if first_wrong is not None:
        term, expected, got = first_wrong
        eval_lib.fail(
            f"wrong result for term {term!r}: expected list of "
            f"{len(expected)} ids, got {str(got)[:120]!r}",
            metrics={"resident_bytes": current},
        )

    # Unseen-data validation (after measurement; memory here is unscored).
    vdocs = gen_docs(vocab, cum, seed=VALIDATION_SEED, n_docs=VALIDATION_DOCS)
    vref = build_reference(vdocs)
    vindex = eval_lib.run_program(mod.build, list(vdocs))
    vrng = random.Random(VALIDATION_SEED + 1)
    vterms = list(vref.keys())
    for _ in range(500):
        w = vterms[vrng.randrange(len(vterms))]
        if eval_lib.run_program(mod.query, vindex, w) != vref[w]:
            eval_lib.fail(
                "validation failed on unseen data (different seed): the index "
                "must reflect the documents it is given, not answers "
                "specialized to the scoring dataset"
            )
    if eval_lib.run_program(mod.query, vindex, "zzzzzzzzzzzz") != []:
        eval_lib.fail("validation failed on unseen data: absent term must return []")

    eval_lib.succeed(
        float(peak),
        metrics={
            "serving_peak_bytes": peak,
            "resident_bytes": current,
            "n_docs": N_DOCS,
            "n_queries": len(queries),
        },
    )


if __name__ == "__main__":
    main()
