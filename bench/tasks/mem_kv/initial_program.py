"""Baseline key/value store: a plain dict."""


def build(pairs):
    return dict(pairs)


def lookup(store, key):
    return store.get(key)
