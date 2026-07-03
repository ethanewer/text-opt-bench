"""Uses a forbidden codec: must be rejected by the AST scan."""

import zlib


def compress(data):
    return zlib.compress(data)


def decompress(blob):
    return zlib.decompress(blob)
