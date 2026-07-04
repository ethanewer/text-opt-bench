"""Broken exploits: the obvious builtins-reaching gadgets the escape
blocklist closes (literal forms). Each must be rejected.

Note: string-hidden attribute access ("{0.__globals__}".format(x),
operator.attrgetter(...)) is NOT caught by the AST scan and is out of
scope under the cooperative threat model — not tested here.
"""

import posixpath   # posixpath.os is the real os module -> attr launder


def _via_posix():
    return posixpath.os.environ            # .os attribute -> blocked


def _via_self():
    return print.__self__                  # builtins module -> blocked (__self__)


def build(pairs):
    _via_posix()
    return {k: v for k, v in pairs}


def lookup(store, key):
    _via_self()
    return store.get(key)


def process(n, ops):
    _via_posix()
    return []


def solve(points):
    _via_self()
    return list(range(len(points)))


def compress(data):
    _via_posix()
    return data


def decompress(blob):
    return blob
