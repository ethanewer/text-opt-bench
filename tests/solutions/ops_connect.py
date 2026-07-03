"""Reference improved solution: union-find with path halving."""


def process(n, ops):
    parent = list(range(n))
    answers = []
    append = answers.append
    for op, a, b in ops:
        while parent[a] != a:
            parent[a] = a = parent[parent[a]]
        while parent[b] != b:
            parent[b] = b = parent[parent[b]]
        if op == "u":
            if a != b:
                parent[a] = b
        else:
            append(a == b)
    return answers
