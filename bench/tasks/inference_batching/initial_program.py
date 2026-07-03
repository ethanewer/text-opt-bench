"""Baseline: first-come first-served admission."""


def order(requests, config):
    return [r["id"] for r in sorted(requests, key=lambda r: (r["arrival"], r["id"]))]
