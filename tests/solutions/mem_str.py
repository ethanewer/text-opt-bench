"""Reference: deduplicate + store unique strings once in a single bytes buffer
(offset-indexed), plus an int array mapping each position to its unique id.
Removes per-string object overhead and all duplicate content."""

from array import array


def build(strings):
    uniq = {}
    ids = array("i")
    buf = bytearray()
    offs = array("I", [0])
    for s in strings:
        u = uniq.get(s)
        if u is None:
            u = len(uniq)
            uniq[s] = u
            buf.extend(s.encode())
            offs.append(len(buf))
        ids.append(u)
    return (bytes(buf), offs, ids)


def get(index, i):
    buf, offs, ids = index
    u = ids[i]
    return buf[offs[u]:offs[u + 1]].decode()
