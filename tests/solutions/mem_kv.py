"""Reference improved solution: sorted key blocks, zlib-compressed."""

import bisect
import zlib

BLOCK = 256


def build(pairs):
    items = sorted(pairs)
    first_keys = []
    blobs = []
    for i in range(0, len(items), BLOCK):
        chunk = items[i : i + BLOCK]
        first_keys.append(chunk[0][0])
        raw = "\x00".join(k + "\x01" + v for k, v in chunk).encode()
        blobs.append(zlib.compress(raw, 9))
    return first_keys, blobs


def lookup(store, key):
    first_keys, blobs = store
    idx = bisect.bisect_right(first_keys, key) - 1
    if idx < 0:
        return None
    raw = zlib.decompress(blobs[idx]).decode()
    for rec in raw.split("\x00"):
        k, sep, v = rec.partition("\x01")
        if k == key:
            return v
    return None
