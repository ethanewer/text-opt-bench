"""Encoding for held-out task data committed to the repo.

Light obfuscation (zlib + XOR), not encryption: it keeps validation/test
data out of casual view (grep, code search, an agent skimming files).
Decoding it from an optimizing agent or program is cheating by rule; the
harness relies on the cooperative threat model documented in the README.
"""

import json
import zlib

_MASK = 0x5A


def encode(obj):
    raw = zlib.compress(json.dumps(obj).encode(), 9)
    return bytes(b ^ _MASK for b in raw)


def decode(blob):
    return json.loads(zlib.decompress(bytes(b ^ _MASK for b in blob)))


def write(path, obj):
    with open(path, "wb") as f:
        f.write(encode(obj))


def read(path):
    with open(path, "rb") as f:
        return decode(f.read())
