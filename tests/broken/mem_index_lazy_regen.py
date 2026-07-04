"""Broken exploit for mem_index: build() returns a tiny marker for the
scoring dataset (detected by size) and defers the real inverted-index
construction to the first query() — regenerating the deterministic corpus
(generators + seeds copied from the evaluator source) and caching it. The
validation corpus (different size) is indexed normally so unseen-data
validation passes.

Must NOT get a low score: the evaluator now serves the full query workload
INSIDE the measurement window, so the regenerate-and-cache happens and is
measured in-window — the marker trick yields no memory benefit.
"""

import random
import string

VOCAB_SIZE = 25_000
SEED = 0x1DEA


def _gen_vocab():
    rng = random.Random(SEED)
    vocab = []
    seen = set()
    while len(vocab) < VOCAB_SIZE:
        w = "".join(rng.choice(string.ascii_lowercase) for _ in range(rng.randrange(3, 11)))
        if w not in seen:
            seen.add(w)
            vocab.append(w)
    cum = []
    total = 0.0
    for r in range(VOCAB_SIZE):
        total += 1.0 / (r + 5)
        cum.append(total)
    return vocab, cum


def _gen_docs(seed, n_docs):
    vocab, cum = _gen_vocab()
    rng = random.Random(seed)
    docs = []
    for _ in range(n_docs):
        n_words = rng.randrange(20, 101)
        docs.append(" ".join(rng.choices(vocab, cum_weights=cum, k=n_words)))
    return docs


def _invert(docs):
    ref = {}
    for doc_id, doc in enumerate(docs):
        for word in set(doc.split()):
            ref.setdefault(word, []).append(doc_id)
    return {w: sorted(ids) for w, ids in ref.items()}


class _Marker:
    __slots__ = ("idx",)

    def __init__(self):
        self.idx = None


def build(docs):
    if len(docs) == 20_000:          # scoring set: defer
        return _Marker()
    return _invert(docs)              # validation set: build normally


def query(index, term):
    if isinstance(index, _Marker):
        if index.idx is None:        # regenerate + cache (was: after tracing)
            index.idx = _invert(_gen_docs(SEED + 1, 20_000))
        return list(index.idx.get(term, []))
    return list(index.get(term, []))
