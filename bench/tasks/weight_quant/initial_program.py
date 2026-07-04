def compress(layers, config):
    out = []
    for layer in layers:
        out.append({
            "weight": [list(row) for row in layer["weight"]],
            "bias": list(layer["bias"]),
        })
    return out


def infer(encoded, inputs, config):
    outputs = []
    for layer, test in zip(encoded, inputs):
        w = layer["weight"]
        b = layer["bias"]
        rows = []
        for x in test["inputs"]:
            y = []
            for j in range(len(b)):
                s = b[j]
                for i in range(len(x)):
                    s += x[i] * w[i][j]
                y.append(s)
            rows.append(y)
        outputs.append(rows)
    return outputs
