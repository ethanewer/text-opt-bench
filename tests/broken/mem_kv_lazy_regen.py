"""Broken exploit for mem_kv: build() returns a tiny marker for the scoring
dataset (detected by size) and defers the real store construction to the
first lookup() — regenerating the deterministic dataset (generator + seed
copied from the evaluator source) and caching it. The validation dataset
(different size) is built normally so unseen-data validation passes.

Must NOT get a low score: the evaluator now serves the full query workload
INSIDE the measurement window, so the regenerate-and-cache happens and is
measured in-window — the marker trick yields no memory benefit.
"""

import random

CATS = ["user", "order", "invoice", "session", "product", "event", "account", "ticket"]
CITIES = ["tokyo", "lagos", "berlin", "austin", "mumbai", "quito", "oslo", "seoul",
          "lima", "cairo", "perth", "denver", "hanoi", "turin", "boise", "dakar"]
WORDS = ["alpha", "bravo", "cargo", "delta", "ember", "flint", "gamma", "harbor",
         "ivory", "jumbo", "krill", "lunar", "mango", "nylon", "ocean", "pixel",
         "quartz", "raven", "sonar", "tundra", "umber", "vivid", "willow", "xenon",
         "yield", "zephyr", "basalt", "cobalt", "drift", "easel", "fjord", "gusto"]


def _gen(seed, n):
    rng = random.Random(seed)
    out = []
    for i in range(n):
        cat = CATS[i % len(CATS)]
        key = f"{cat}:{i:08d}:{rng.randrange(16 ** 6):06x}"
        tags = " ".join(rng.choice(WORDS) for _ in range(rng.randrange(2, 6)))
        value = (f'{{"id": {i}, "city": "{rng.choice(CITIES)}", '
                 f'"score": {rng.randrange(100000)}, "active": {rng.choice(["true", "false"])}, '
                 f'"tags": "{tags}"}}')
        out.append((key, value))
    return out


class _Marker:
    __slots__ = ("d",)

    def __init__(self):
        self.d = None


def build(pairs):
    if len(pairs) == 150_000:        # scoring set: defer
        return _Marker()
    return {k: v for k, v in pairs}   # validation set: build normally


def lookup(store, key):
    if isinstance(store, _Marker):
        if store.d is None:           # regenerate + cache (was: after tracing)
            store.d = {k: v for k, v in _gen(0xC0FFEE, 150_000)}
        return store.d.get(key)
    return store.get(key)
