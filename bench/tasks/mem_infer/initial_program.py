"""Correct baseline that retains every intermediate in float32."""


def _layer(rt, layer, x, state, conv, kc, vc, pos, history):
    h = rt.rmsnorm(x, layer["ln1"]); history.append(h)
    q = rt.linear(h, layer["wq"]); history.append(q)
    k = rt.linear(h, layer["wk"]); history.append(k)
    v = rt.linear(h, layer["wv"]); history.append(v)
    if state is not None:
        decay = rt.linear(h, layer["wdecay"]); history.append(decay)
        beta = rt.linear(h, layer["wbeta"]); history.append(beta)
        mixed = rt.delta_step(q, k, v, decay, beta, state, conv,
                              layer["conv"], layer["a_log"])
        history.append(mixed)
    else:
        q, k = rt.qk_norm_rope(q, k, pos)
        history.append(q); history.append(k)
        rt.cache_write(kc, pos, k)
        rt.cache_write(vc, pos, v)
        mixed = rt.attention(q, kc, vc, pos + 1); history.append(mixed)
    gate = rt.silu(rt.linear(h, layer["wgate"])); history.append(gate)
    mixed = rt.mul(mixed, gate); history.append(mixed)
    x = rt.add(x, rt.linear(mixed, layer["wo"])); history.append(x)
    h = rt.rmsnorm(x, layer["ln2"]); history.append(h)
    up = rt.linear(h, layer["wup"]); history.append(up)
    gate = rt.silu(rt.linear(h, layer["wgate_ff"])); history.append(gate)
    ff = rt.mul(up, gate); history.append(ff)
    x = rt.add(x, rt.linear(ff, layer["wdown"])); history.append(x)
    return x


def generate(rt, weights, prompt, n_tokens):
    state = rt.zeros((4, 16, 16))
    conv = rt.zeros((3, 192))
    kc = rt.zeros((128, 1, 16))
    vc = rt.zeros((128, 1, 16))
    tokens = list(prompt)
    output = []
    history = [state, conv, kc, vc]
    for pos in range(len(prompt) + n_tokens - 1):
        x = rt.embed(weights["wte"], weights["wpe"], tokens[pos], pos)
        history.append(x)
        x = _layer(rt, weights["delta"], x, state, conv, None, None, pos,
                   history)
        x = _layer(rt, weights["attn"], x, None, None, kc, vc, pos, history)
        if pos >= len(prompt) - 1:
            h = rt.rmsnorm(x, weights["lnf"]); history.append(h)
            token = rt.argmax_vocab(h, weights["wte"])
            output.append(token)
            tokens.append(token)
    return output
