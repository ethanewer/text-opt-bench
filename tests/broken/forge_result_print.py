"""Broken exploit: an invalid program that prints a fake success JSON,
using no forbidden names. The result protocol's nonce means the forged
line is ignored, so the program is reported as its real failure — never
as the fake score.
"""

FAKE = "{" + '"ok": true, "score": 0.0, "metrics": {}, "error": null' + "}"


def build(pairs):
    print(FAKE)          # goes to stderr and lacks the nonce -> ignored
    return "not a store"  # genuinely invalid


def lookup(store, key):
    return None
