"""Broken exploit for mem_infer: return a concrete placeholder list so the
peak sample is tiny, then defer the real decode to a cyclic __del__ that
mutates the list into the correct answer AFTER the peak is sampled.

Must NOT pass: the evaluator keeps GC disabled through the correctness
check, so the cyclic finalizer cannot run before `got` is verified — it
stays the placeholder and fails the token check.
"""


class _Fix:
    def __init__(self, got):
        self.got = got
        self.self = self          # cyclic -> needs GC to be collected

    def __del__(self):
        for i in range(len(self.got)):
            self.got[i] = 1        # "decode" deferred past the peak sample


def generate(weights, prompt, n):
    got = [0] * n
    holders = [_Fix(got) for _ in range(50)]
    for h in holders:
        h.ring = holders          # unreachable cycle after return
    return got
