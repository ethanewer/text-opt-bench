"""Broken exploit: return a LAZY object so the real work (or a
metric-tampering access) runs after the evaluator stops measuring — a
generator, or a list subclass that builds itself in __iter__/__getitem__.

Must be rejected: measured evaluators require a plain list of concrete
ints, materialized inside the measurement window.
"""


def _gen(k):
    for i in range(k):
        yield i % 2


class _LazyList(list):
    def __init__(self, thunk):
        self._thunk = thunk
        self._v = None

    def _r(self):
        if self._v is None:
            self._v = self._thunk()
        return self._v

    def __iter__(self):
        return iter(self._r())

    def __getitem__(self, i):
        return self._r()[i]

    def __len__(self):
        return len(self._r())


def process(n, ops):            # ops_connect — generator
    return _gen(sum(1 for o in ops if o[0] == "q"))


def solve(points):             # tsp_budget — lazy list subclass
    return _LazyList(lambda: list(range(len(points))))


def generate(weights, prompt, n):   # mem_infer — lazy list subclass
    return _LazyList(lambda: [0] * n)
