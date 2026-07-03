"""Baseline inverted index: dict of word -> list of doc ids."""


def build(docs):
    index = {}
    for doc_id, doc in enumerate(docs):
        for word in set(doc.split()):
            index.setdefault(word, []).append(doc_id)
    return index


def query(index, term):
    return sorted(index.get(term, []))
