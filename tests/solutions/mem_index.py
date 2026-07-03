"""Reference improved solution: varint delta-encoded postings blobs."""


def build(docs):
    index = {}
    for doc_id, doc in enumerate(docs):
        for w in set(doc.split()):
            index.setdefault(w, []).append(doc_id)
    out = {}
    for w, ids in index.items():
        prev = 0
        ba = bytearray()
        for d in sorted(ids):
            delta = d - prev
            prev = d
            while delta >= 128:
                ba.append((delta & 127) | 128)
                delta >>= 7
            ba.append(delta)
        out[w] = bytes(ba)
    return out


def query(index, term):
    blob = index.get(term)
    if blob is None:
        return []
    ids = []
    cur = 0
    shift = 0
    acc = 0
    for byte in blob:
        if byte & 128:
            acc |= (byte & 127) << shift
            shift += 7
        else:
            cur += acc | (byte << shift)
            ids.append(cur)
            acc = 0
            shift = 0
    return ids
