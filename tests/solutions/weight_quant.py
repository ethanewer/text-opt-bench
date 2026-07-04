def _quantize_columns(weight, levels, rms, alpha):
    rows = len(weight)
    cols = len(weight[0])
    mean = sum(rms) / len(rms)
    scales_in = []
    for r in rms:
        s = (r / mean) ** alpha if mean > 0.0 and r > 0.0 else 1.0
        if s < 0.35:
            s = 0.35
        elif s > 3.0:
            s = 3.0
        scales_in.append(s)

    qcols = []
    mins = []
    steps = []
    top = levels - 1
    for j in range(cols):
        lo = weight[0][j] * scales_in[0]
        hi = lo
        for i in range(1, rows):
            v = weight[i][j] * scales_in[i]
            if v < lo:
                lo = v
            elif v > hi:
                hi = v
        step = (hi - lo) / top if hi > lo else 1.0
        mins.append(lo)
        steps.append(step)
        col = []
        for i in range(rows):
            q = int((weight[i][j] * scales_in[i] - lo) / step + 0.5) if hi > lo else 0
            if q < 0:
                q = 0
            elif q > top:
                q = top
            col.append(q)
        qcols.append(col)
    return {
        "q": qcols,
        "mn": mins,
        "st": steps,
        "si": scales_in,
        "b": None,
    }


def compress(layers, config):
    encoded = []
    for idx, layer in enumerate(layers):
        levels = 49 if idx in (1, 3) else 33
        enc = _quantize_columns(layer["weight"], levels, layer["input_rms"], 0.55)
        enc["b"] = list(layer["bias"])
        enc["lv"] = levels
        encoded.append(enc)
    return encoded


def infer(encoded, inputs, config):
    outputs = []
    for enc, test in zip(encoded, inputs):
        qcols = enc["q"]
        mins = enc["mn"]
        steps = enc["st"]
        scales_in = enc["si"]
        bias = enc["b"]
        cols = len(qcols)
        rows_in = len(scales_in)
        layer_out = []
        for x in test["inputs"]:
            y = []
            for j in range(cols):
                qcol = qcols[j]
                mn = mins[j]
                st = steps[j]
                s = bias[j]
                for i in range(rows_in):
                    s += (x[i] / scales_in[i]) * (mn + st * qcol[i])
                y.append(s)
            layer_out.append(y)
        outputs.append(layer_out)
    return outputs
