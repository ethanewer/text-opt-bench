"""Evaluator for mem_index. Score = resident traced bytes after build (lower better)."""

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
    docs = gen_docs(vocab, cum)
    index = eval_lib.run_program(mod.build, docs)
    del docs
    gc.collect()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Correctness after measurement.
    for term, expected in queries:
        got = eval_lib.run_program(mod.query, index, term)
        if not isinstance(got, list) or got != expected:
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
        float(current),
        metrics={
            "resident_bytes": current,
            "peak_bytes_during_build": peak,
            "n_docs": N_DOCS,
            "n_queries": len(queries),
        },
    )


if __name__ == "__main__":
    main()
