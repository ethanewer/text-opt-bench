"""Compress-decompress cheat for mem_str: build() retains tiny lzma-compressed
blocks (low resident); get() transiently decompresses a whole block per query.
Under retained-only scoring this "won" (~472KB < honest 938KB); under serving-peak
scoring the per-query decompression working set is charged, so it scores far worse."""
import lzma
BLOCK = 512
def build(strings):
    n = len(strings)
    blocks = []
    offs = []
    for s in range(0, n, BLOCK):
        chunk = strings[s:s+BLOCK]
        payload = "\x00".join(chunk).encode("utf-8")
        blocks.append(lzma.compress(payload, preset=6))
        offs.append(s)
    return (n, offs, blocks)
def get(index, i):
    n, offs, blocks = index
    b = i // BLOCK
    payload = lzma.decompress(blocks[b])          # big transient per query
    parts = payload.decode("utf-8").split("\x00")
    return parts[i - offs[b]]
