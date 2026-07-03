"""Returns the wrong value: must be rejected by the evaluator."""


def build(pairs):
    return dict(pairs)


def lookup(store, key):
    return key if key in store else None
