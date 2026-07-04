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


def encode(cache, config):
    # Layer-dependent retention plus asymmetric-ish quantization. This follows
    # the same broad pattern as H2O/SnapKV/PyramidKV: keep attention sinks,
    # keep a recent window, and spend the remaining budget on observed heavy
    # hitters with larger budgets in later/more sensitive layers.
    key_levels = [49, 65, 97, 129]
    value_levels = [49, 65, 97, 129]
    keep_fracs = [0.48, 0.54, 0.62, 0.70]
    layers = []
    for idx, layer in enumerate(cache["layers"]):
        n = len(layer["keys"])
        keep = int(n * keep_fracs[idx])
        if keep < 48:
            keep = 48
        if keep > n:
            keep = n
        sink = config["sink_tokens"]
        recent = keep // 3
        chosen = set()
        for i in range(sink):
            if i < n:
                chosen.add(i)
        for i in range(n - recent, n):
            if i >= 0:
                chosen.add(i)
        ranked = sorted(
            [(-layer["importance"][i], i) for i in range(n) if i not in chosen]
        )
        for _, i in ranked:
            if len(chosen) >= keep:
                break
            chosen.add(i)
        indices = sorted(chosen)
        keys = [layer["keys"][i] for i in indices]
        values = [layer["values"][i] for i in indices]
        qk, km, ks, kb = _quantize_matrix(keys, key_levels[idx])
        qv, vm, vs, vb = _quantize_matrix(values, value_levels[idx])
        layers.append({
            "i": indices,
            "k": qk,
            "v": qv,
            "km": km,
            "ks": ks,
            "kb": kb,
            "vm": vm,
            "vs": vs,
            "vb": vb,
        })
    return {"layers": layers, "scale": cache["scale"]}


def attend(encoded, queries, config):
    out = []
    scale = encoded["scale"]
    for layer_index, q_layer in enumerate(queries):
        layer = encoded["layers"][layer_index]
        indices = layer["i"]
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
                    kval = kmins[j] + (krow[j] + kbias) * kscales[j]
                    s += q[j] * kval
                s *= scale
                scores.append(s)
                if s > m:
                    m = s
            # Approximate the contribution of evicted tokens with a tiny
            # uniform tail. This avoids severe overconfidence when many
            # unimportant tokens were dropped.
            dropped = config["n_tokens"] - len(indices)
            weights = []
            total = 0.000001 * dropped
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
