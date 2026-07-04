def encode(cache, config):
    return {
        "layers": [
            {
                "keys": [list(row) for row in layer["keys"]],
                "values": [list(row) for row in layer["values"]],
            }
            for layer in cache["layers"]
        ],
        "scale": cache["scale"],
    }


def attend(encoded, queries, config):
    out = []
    scale = encoded["scale"]
    for layer_index, q_layer in enumerate(queries):
        layer = encoded["layers"][layer_index]
        keys = layer["keys"]
        values = layer["values"]
        layer_out = []
        for q in q_layer:
            scores = []
            m = -10**30
            for k in keys:
                s = 0.0
                for i in range(len(q)):
                    s += q[i] * k[i]
                s *= scale
                scores.append(s)
                if s > m:
                    m = s
            weights = []
            total = 0.0
            for s in scores:
                w = 2.718281828459045 ** (s - m)
                weights.append(w)
                total += w
            y = [0.0 for _ in range(len(values[0]))]
            for pos, v in enumerate(values):
                w = weights[pos] / total
                for j in range(len(y)):
                    y[j] += w * v[j]
            layer_out.append(y)
        out.append(layer_out)
    return out
