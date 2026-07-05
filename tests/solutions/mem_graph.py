"""Reference improved solution: compressed sparse row (CSR) with varint
delta-encoded neighbor lists.

Index = (node_arr, offsets, data):
  - node_arr: array('i') of the sorted nodes that have out-edges
  - offsets:  array('I'), offsets[i]..offsets[i+1] slice `data` for node i
  - data:     one bytes buffer of varint(delta) of each node's sorted
              distinct neighbors (deltas are non-negative since sorted)

Retains ~1-2 bytes/edge + 8 bytes/node, vs the baseline's boxed-int lists.
"""

from array import array


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
    for u, v in edges:
        s = adj.get(u)
        if s is None:
            adj[u] = s = set()
        s.add(v)
    nodes = sorted(adj)
    node_arr = array("i", nodes)
    offsets = array("I", [0])
    data = bytearray()
    for u in nodes:
        prev = 0
        for v in sorted(adj[u]):
            _put_varint(data, v - prev)
            prev = v
        offsets.append(len(data))
    return (node_arr, offsets, bytes(data))


def _find(node_arr, u):
    lo, hi = 0, len(node_arr)
    while lo < hi:
        mid = (lo + hi) >> 1
        if node_arr[mid] < u:
            lo = mid + 1
        else:
            hi = mid
    if lo < len(node_arr) and node_arr[lo] == u:
        return lo
    return -1


def neighbors(index, u):
    node_arr, offsets, data = index
    i = _find(node_arr, u)
    if i < 0:
        return []
    out = []
    prev = 0
    pos = offsets[i]
    end = offsets[i + 1]
    while pos < end:
        shift = 0
        x = 0
        while True:
            b = data[pos]
            pos += 1
            x |= (b & 0x7F) << shift
            if not (b & 0x80):
                break
            shift += 7
        prev += x
        out.append(prev)
    return out
