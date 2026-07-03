"""Reference improved solution: no histories, array-backed KV cache."""

import math
from array import array


def _layernorm(x, g, b):
    n = len(x)
    m = sum(x) / n
    var = sum((v - m) ** 2 for v in x) / n
    inv = 1.0 / math.sqrt(var + 1e-5)
    return [(x[i] - m) * inv * g[i] + b[i] for i in range(n)]


def _matvec(mat, x):
    rows = len(x)
    cols = len(mat[0])
    return [sum(x[r] * mat[r][c] for r in range(rows)) for c in range(cols)]


def generate(weights, prompt, n_tokens):
    d = len(weights["wte"][0])
    n_heads = 4
    dh = d // n_heads
    layers = weights["layers"]
    inv_sqrt_dh = 1.0 / math.sqrt(dh)

    kcaches = [array("d") for _ in layers]
    vcaches = [array("d") for _ in layers]

    out = []
    prev_token = None
    pos = 0
    n_prompt = len(prompt)
    while len(out) < n_tokens:
        t = prompt[pos] if pos < n_prompt else prev_token
        wte_t = weights["wte"][t]
        wpe_p = weights["wpe"][pos]
        x = [wte_t[i] + wpe_p[i] for i in range(d)]
        for li, lw in enumerate(layers):
            h = _layernorm(x, lw["ln1_g"], lw["ln1_b"])
            q = _matvec(lw["wq"], h)
            kcaches[li].extend(_matvec(lw["wk"], h))
            vcaches[li].extend(_matvec(lw["wv"], h))
            kc = kcaches[li]
            vc = vcaches[li]
            n_cached = (pos + 1)
            attn = [0.0] * d
            for head in range(n_heads):
                lo = head * dh
                mx = -1e30
                scores = array("d")
                for j in range(n_cached):
                    base = j * d + lo
                    s = 0.0
                    for i in range(dh):
                        s += q[lo + i] * kc[base + i]
                    s *= inv_sqrt_dh
                    scores.append(s)
                    if s > mx:
                        mx = s
                z = 0.0
                for j in range(n_cached):
                    e = math.exp(scores[j] - mx)
                    scores[j] = e
                    z += e
                for j in range(n_cached):
                    w = scores[j] / z
                    base = j * d + lo
                    for i in range(dh):
                        attn[lo + i] += w * vc[base + i]
            proj = _matvec(lw["wo"], attn)
            for i in range(d):
                x[i] += proj[i]
            h = _layernorm(x, lw["ln2_g"], lw["ln2_b"])
            ff = _matvec(lw["w1"], h)
            b1 = lw["b1"]
            ff = [max(0.0, ff[i] + b1[i]) for i in range(len(ff))]
            ff2 = _matvec(lw["w2"], ff)
            b2 = lw["b2"]
            for i in range(d):
                x[i] += ff2[i] + b2[i]
        if pos >= n_prompt - 1:
            hf = _layernorm(x, weights["lnf_g"], weights["lnf_b"])
            best = -1
            best_val = -1e30
            for tok, row in enumerate(weights["wte"]):
                s = 0.0
                for i in range(d):
                    s += hf[i] * row[i]
                if s > best_val:
                    best_val = s
                    best = tok
            out.append(best)
            prev_token = best
        pos += 1
    return out
