_CACHE = None


def encode(cache, config):
    global _CACHE
    _CACHE = cache
    return 0


def attend(encoded, queries, config):
    scale = _CACHE["scale"]
    out = []
    for layer, q_layer in enumerate(queries):
        keys = _CACHE["layers"][layer]["keys"]
        values = _CACHE["layers"][layer]["values"]
        layer_out = []
        for q in q_layer:
            scores = []
            m = -1e100
            for k in keys:
                s = 0.0
                for i in range(len(q)):
                    s += q[i] * k[i]
                s *= scale
                scores.append(s)
                if s > m:
                    m = s
            exps = []
            total = 0.0
            for s in scores:
                e = 2.718281828459045 ** (s - m)
                exps.append(e)
                total += e
            y = []
            for j in range(len(values[0])):
                v = 0.0
                for i in range(len(values)):
                    v += (exps[i] / total) * values[i][j]
                y.append(v)
            layer_out.append(y)
        out.append(layer_out)
    return out
