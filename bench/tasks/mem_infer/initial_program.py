"""Baseline decode: correct but memory-hungry.

Keeps the full history of every intermediate activation for every layer
and step, plus full attention matrices and all logits.
"""

import math


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

    history = []          # every hidden state ever computed
    kcaches = [[] for _ in layers]
    vcaches = [[] for _ in layers]
    attn_matrices = []    # every attention row ever computed
    all_logits = []       # logits for every generation step

    tokens = list(prompt)
    out = []
    pos = 0
    while len(out) < n_tokens:
        t = tokens[pos]
        x = [weights["wte"][t][i] + weights["wpe"][pos][i] for i in range(d)]
        history.append(list(x))
        for li, lw in enumerate(layers):
            h = _layernorm(x, lw["ln1_g"], lw["ln1_b"])
            history.append(list(h))
            q = _matvec(lw["wq"], h)
            k = _matvec(lw["wk"], h)
            v = _matvec(lw["wv"], h)
            kcaches[li].append(list(k))
            vcaches[li].append(list(v))
            attn_out = [0.0] * d
            for head in range(n_heads):
                lo = head * dh
                scores = [
                    sum(q[lo + i] * kk[lo + i] for i in range(dh)) / math.sqrt(dh)
                    for kk in kcaches[li]
                ]
                mx = max(scores)
                exps = [math.exp(s - mx) for s in scores]
                z = sum(exps)
                probs = [e / z for e in exps]
                attn_matrices.append(list(probs))
                for j, vv in enumerate(vcaches[li]):
                    for i in range(dh):
                        attn_out[lo + i] += probs[j] * vv[lo + i]
            proj = _matvec(lw["wo"], attn_out)
            x = [x[i] + proj[i] for i in range(d)]
            history.append(list(x))
            h2 = _layernorm(x, lw["ln2_g"], lw["ln2_b"])
            history.append(list(h2))
            ff = _matvec(lw["w1"], h2)
            ff = [max(0.0, ff[i] + lw["b1"][i]) for i in range(len(ff))]
            history.append(list(ff))
            ff2 = _matvec(lw["w2"], ff)
            x = [x[i] + ff2[i] + lw["b2"][i] for i in range(d)]
            history.append(list(x))
        if pos >= len(prompt) - 1:
            hf = _layernorm(x, weights["lnf_g"], weights["lnf_b"])
            logits = [
                sum(hf[i] * row[i] for i in range(d)) for row in weights["wte"]
            ]
            all_logits.append(logits)
            best = max(range(len(logits)), key=logits.__getitem__)
            out.append(best)
            tokens.append(best)
        pos += 1
    return out
