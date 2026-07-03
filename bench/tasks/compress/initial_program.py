"""Baseline compressor: identity (no compression at all)."""


def compress(data):
    return bytes(data)


def decompress(blob):
    return bytes(blob)
