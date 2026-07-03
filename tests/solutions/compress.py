"""Reference improved solution: byte-level canonical Huffman coding."""

import heapq

MAGIC = b"HUF1"


def _lengths(freq):
    heap = []
    tie = 0
    for s, f in enumerate(freq):
        if f:
            heap.append((f, tie, [s]))
            tie += 1
    if not heap:
        return {}
    if len(heap) == 1:
        return {heap[0][2][0]: 1}
    depth = [0] * 256
    heapq.heapify(heap)
    while len(heap) > 1:
        f1, _, s1 = heapq.heappop(heap)
        f2, _, s2 = heapq.heappop(heap)
        for s in s1:
            depth[s] += 1
        for s in s2:
            depth[s] += 1
        heapq.heappush(heap, (f1 + f2, tie, s1 + s2))
        tie += 1
    return {s: depth[s] for s in range(256) if freq[s]}


def _canonical_codes(lengths):
    codes = {}
    code = 0
    prev_len = 0
    for length, sym in sorted((l, s) for s, l in lengths.items()):
        code <<= length - prev_len
        codes[sym] = format(code, f"0{length}b")
        code += 1
        prev_len = length
    return codes


def compress(data):
    if not data:
        return MAGIC
    freq = [0] * 256
    for b in data:
        freq[b] += 1
    lengths = _lengths(freq)
    codes = _canonical_codes(lengths)
    bits = "1" + "".join(map(codes.__getitem__, data))
    payload = int(bits, 2).to_bytes((len(bits) + 7) // 8, "big")
    header = bytes(lengths.get(s, 0) for s in range(256))
    return MAGIC + header + len(data).to_bytes(8, "big") + payload


def decompress(blob):
    if blob == MAGIC:
        return b""
    header = blob[4:260]
    n = int.from_bytes(blob[260:268], "big")
    lengths = {s: l for s, l in enumerate(header) if l}
    codes = _canonical_codes(lengths)
    decode = {(len(c), int(c, 2)): s for s, c in codes.items()}
    bits = bin(int.from_bytes(blob[268:], "big"))[3:]  # strip "0b1" sentinel
    out = bytearray()
    val = 0
    length = 0
    for bit in bits:
        val = (val << 1) | (bit == "1")
        length += 1
        sym = decode.get((length, val))
        if sym is not None:
            out.append(sym)
            val = 0
            length = 0
            if len(out) == n:
                break
    return bytes(out)
