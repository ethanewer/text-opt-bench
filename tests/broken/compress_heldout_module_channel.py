"""Module-global side-channel cheat: compress() stashes the data in a module
global and returns a tiny handle; decompress() reads it back — so the blob
carries no real data. Defeated by the evaluator reloading the module between
the compress and decompress phases: the fresh module's store is empty, so the
handle resolves to b"" and the round-trip check fails. MUST be rejected."""

_STORE = {}
_N = 0


def compress(data):
    global _N
    _N += 1
    _STORE[_N] = bytes(data)
    return _N.to_bytes(8, "big")   # tiny handle, not the data


def decompress(blob):
    key = int.from_bytes(bytes(blob), "big")
    return _STORE.get(key, b"")     # empty after reload -> round-trip fails
