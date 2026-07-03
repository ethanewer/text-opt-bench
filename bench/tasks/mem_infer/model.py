"""Shared reference model for mem_infer: a tiny GPT-style decoder.

Used by the evaluator to build weights and compute reference outputs.
Architecture (all plain Python floats/lists):

  token embedding (tied output head) + learned positional embedding
  N decoder layers: LayerNorm -> causal multi-head attention -> residual
                    LayerNorm -> feed-forward (ReLU) -> residual
  final LayerNorm -> logits = hidden @ wte^T -> greedy argmax
"""

import math
import random

VOCAB = 64
D_MODEL = 48
N_HEADS = 4
D_FF = 96
N_LAYERS = 2
CTX = 20
PROMPT_LEN = 8
N_GEN = 12


def _mat(rng, rows, cols, scale):
    return [[rng.uniform(-scale, scale) for _ in range(cols)] for _ in range(rows)]


def build_weights(seed):
    rng = random.Random(seed)
    s = 0.6 / math.sqrt(D_MODEL)
    layers = []
    for _ in range(N_LAYERS):
        layers.append({
            "wq": _mat(rng, D_MODEL, D_MODEL, s),
            "wk": _mat(rng, D_MODEL, D_MODEL, s),
            "wv": _mat(rng, D_MODEL, D_MODEL, s),
            "wo": _mat(rng, D_MODEL, D_MODEL, s),
            "w1": _mat(rng, D_MODEL, D_FF, s),
            "b1": [rng.uniform(-0.02, 0.02) for _ in range(D_FF)],
            "w2": _mat(rng, D_FF, D_MODEL, s),
            "b2": [rng.uniform(-0.02, 0.02) for _ in range(D_MODEL)],
            "ln1_g": [1.0 + rng.uniform(-0.05, 0.05) for _ in range(D_MODEL)],
            "ln1_b": [rng.uniform(-0.02, 0.02) for _ in range(D_MODEL)],
            "ln2_g": [1.0 + rng.uniform(-0.05, 0.05) for _ in range(D_MODEL)],
            "ln2_b": [rng.uniform(-0.02, 0.02) for _ in range(D_MODEL)],
        })
    return {
        "wte": _mat(rng, VOCAB, D_MODEL, 3.0 * s),
        "wpe": _mat(rng, CTX, D_MODEL, 6.0 * s),
        "layers": layers,
        "lnf_g": [1.0 + rng.uniform(-0.05, 0.05) for _ in range(D_MODEL)],
        "lnf_b": [rng.uniform(-0.02, 0.02) for _ in range(D_MODEL)],
    }


def build_prompt(seed):
    rng = random.Random(seed ^ 0x5A5A5A)
    return [rng.randrange(VOCAB) for _ in range(PROMPT_LEN)]


def _layernorm(x, g, b):
    n = len(x)
    m = sum(x) / n
    var = sum((v - m) ** 2 for v in x) / n
    inv = 1.0 / math.sqrt(var + 1e-5)
    return [(x[i] - m) * inv * g[i] + b[i] for i in range(n)]


def _matvec(mat, x):
    # mat is rows x cols; computes x @ mat (x has len rows)
    rows = len(x)
    cols = len(mat[0])
    return [sum(x[r] * mat[r][c] for r in range(rows)) for c in range(cols)]


def reference_generate(weights, prompt, n_gen):
    """Incremental greedy decode. Returns (tokens, min_logit_margin)."""
    dh = D_MODEL // N_HEADS
    layers = weights["layers"]
    kcache = [[] for _ in layers]
    vcache = [[] for _ in layers]
    tokens = list(prompt)
    out = []
    min_margin = float("inf")
    pos = 0
    while len(out) < n_gen:
        t = tokens[pos]
        x = [weights["wte"][t][i] + weights["wpe"][pos][i] for i in range(D_MODEL)]
        for li, lw in enumerate(layers):
            h = _layernorm(x, lw["ln1_g"], lw["ln1_b"])
            q = _matvec(lw["wq"], h)
            k = _matvec(lw["wk"], h)
            v = _matvec(lw["wv"], h)
            kcache[li].append(k)
            vcache[li].append(v)
            attn_out = [0.0] * D_MODEL
            for head in range(N_HEADS):
                lo = head * dh
                scores = []
                for kk in kcache[li]:
                    scores.append(
                        sum(q[lo + i] * kk[lo + i] for i in range(dh))
                        / math.sqrt(dh)
                    )
                mx = max(scores)
                exps = [math.exp(s - mx) for s in scores]
                z = sum(exps)
                for j, vv in enumerate(vcache[li]):
                    w = exps[j] / z
                    for i in range(dh):
                        attn_out[lo + i] += w * vv[lo + i]
            proj = _matvec(lw["wo"], attn_out)
            x = [x[i] + proj[i] for i in range(D_MODEL)]
            h = _layernorm(x, lw["ln2_g"], lw["ln2_b"])
            ff = _matvec(lw["w1"], h)
            ff = [max(0.0, ff[i] + lw["b1"][i]) for i in range(D_FF)]
            ff2 = _matvec(lw["w2"], ff)
            x = [x[i] + ff2[i] + lw["b2"][i] for i in range(D_MODEL)]
        if pos >= len(prompt) - 1:
            hf = _layernorm(x, weights["lnf_g"], weights["lnf_b"])
            logits = [
                sum(hf[i] * weights["wte"][tok][i] for i in range(D_MODEL))
                for tok in range(VOCAB)
            ]
            best = max(range(VOCAB), key=logits.__getitem__)
            ordered = sorted(logits, reverse=True)
            min_margin = min(min_margin, ordered[0] - ordered[1])
            out.append(best)
            tokens.append(best)
        pos += 1
    return out, min_margin
