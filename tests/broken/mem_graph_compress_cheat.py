"""Compressed-block neighbor index: store per-block LZMA-compressed
(degree + varint-delta) streams. Retained memory = sum of compressed blocks;
each query transiently decompresses one block and discards it."""

import lzma

BLOCK = 250


def _put_varint(buf, x):
    while True:
        b = x & 0x7F
        x >>= 7
        if x:
            buf.append(b | 0x80)
        else:
            buf.append(b)
            return


def build(edges):
    adj = {}
    maxu = -1
    for u, v in edges:
        s = adj.get(u)
        if s is None:
            adj[u] = s = set()
        s.add(v)
        if u > maxu:
            maxu = u
    N = maxu + 1
    nblocks = (N + BLOCK - 1) // BLOCK
    blocks = []
    for bi in range(nblocks):
        u0 = bi * BLOCK
        u1 = min(u0 + BLOCK, N)
        stream = bytearray()
        for u in range(u0, u1):
            s = adj.get(u)
            if s is None:
                _put_varint(stream, 0)
                continue
            sl = sorted(s)
            _put_varint(stream, len(sl))
            prev = 0
            for v in sl:
                _put_varint(stream, v - prev)
                prev = v
        blocks.append(lzma.compress(bytes(stream), preset=6))
    return (N, blocks)


def neighbors(index, u):
    N, blocks = index
    if u < 0 or u >= N:
        return []
    bi = u // BLOCK
    raw = lzma.decompress(blocks[bi])
    local = u - bi * BLOCK
    pos = 0
    # skip `local` preceding node records
    j = 0
    while j < local:
        # read degree varint
        shift = 0
        deg = 0
        while True:
            b = raw[pos]
            pos += 1
            deg |= (b & 0x7F) << shift
            if not (b & 0x80):
                break
            shift += 7
        # skip deg delta varints
        k = 0
        while k < deg:
            if not (raw[pos] & 0x80):
                k += 1
            pos += 1
        j += 1
    # read target node degree
    shift = 0
    deg = 0
    while True:
        b = raw[pos]
        pos += 1
        deg |= (b & 0x7F) << shift
        if not (b & 0x80):
            break
        shift += 7
    out = []
    prev = 0
    for _ in range(deg):
        shift = 0
        x = 0
        while True:
            b = raw[pos]
            pos += 1
            x |= (b & 0x7F) << shift
            if not (b & 0x80):
                break
            shift += 7
        prev += x
        out.append(prev)
    return out
