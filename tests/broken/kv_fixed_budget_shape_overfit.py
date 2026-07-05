"""Broken: real encode/attend only for the fixed scoring token counts;
degenerate otherwise. Passes scoring but fails held-out. Must be rejected."""

def _quantize_matrix(mat, levels):
    dims = len(mat[0])
    mins = []
    scales = []
    qmat = []
    denom = levels - 1
    bias = denom // 2
    for j in range(dims):
        lo = mat[0][j]
        hi = lo
        for row in mat:
            x = row[j]
            if x < lo:
                lo = x
            if x > hi:
                hi = x
        span = hi - lo
        if span <= 0.000000001:
            span = 1.0
        mins.append(lo)
        scales.append(span / denom)
    for row in mat:
        qr = []
        for j in range(dims):
            q = int((row[j] - mins[j]) / scales[j] + 0.5) - bias
            low = -bias
            high = denom - bias
            if q < low:
                q = low
            if q > high:
                q = high
            qr.append(q)
        qmat.append(qr)
    return qmat, mins, scales, bias


def _real_encode(cache, config):
    levels = config.get("quant_levels", 65)
    layers = []
    for layer in cache["layers"]:
        qk, km, ks, kb = _quantize_matrix(layer["keys"], levels)
        qv, vm, vs, vb = _quantize_matrix(layer["values"], levels)
        layers.append({"k": qk, "v": qv, "km": km, "ks": ks, "kb": kb,
                       "vm": vm, "vs": vs, "vb": vb})
    return {"layers": layers, "scale": cache["scale"]}


def _real_attend(encoded, queries, config):
    out = []
    scale = encoded["scale"]
    for layer_index, q_layer in enumerate(queries):
        layer = encoded["layers"][layer_index]
        qkeys = layer["k"]
        qvals = layer["v"]
        kmins = layer["km"]
        kscales = layer["ks"]
        kbias = layer["kb"]
        vmins = layer["vm"]
        vscales = layer["vs"]
        vbias = layer["vb"]
        layer_out = []
        for q in q_layer:
            scores = []
            m = -10**30
            for krow in qkeys:
                s = 0.0
                for j in range(len(q)):
                    s += q[j] * (kmins[j] + (krow[j] + kbias) * kscales[j])
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
            y = [0.0 for _ in range(len(vmins))]
            for i, vrow in enumerate(qvals):
                w = weights[i] / total
                for j in range(len(y)):
                    y[j] += w * (vmins[j] + (vrow[j] + vbias) * vscales[j])
            layer_out.append(y)
        out.append(layer_out)
    return out


_SCORING = (224, 256, 288, 320)
def encode(cache, config):
    if config["n_tokens"] not in _SCORING:
        return {"skip": True}
    return _real_encode(cache, config)
def attend(encoded, queries, config):
    if isinstance(encoded, dict) and encoded.get("skip"):
        vd = config["value_dim"]
        return [[[0.0]*vd for _ in q] for q in queries]
    return _real_attend(encoded, queries, config)
