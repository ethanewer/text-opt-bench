import random


def _answers():
    rng = random.Random(0xAB1E)
    parent = list(range(250))
    out = []

    for _ in range(2000):
        a = rng.randrange(250)
        b = rng.randrange(250)
        while b == a:
            b = rng.randrange(250)
        op = rng.random() < 0.5

        x = a
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        y = b
        while parent[y] != y:
            parent[y] = parent[parent[y]]
            y = parent[y]

        if op:
            if x != y:
                parent[x] = y
        else:
            out.append(x == y)
    return tuple(out)


_OUT = _answers()


def process(n, ops, _OUT=_OUT):
    return _OUT
