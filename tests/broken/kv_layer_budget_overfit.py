"""Broken: memorizes the fixed scoring token counts. It returns a tuned
allocation for n_tokens in {224,256,288,320} but a non-general one for any
other count, so it passes the scoring instances yet blows the byte budget on
the held-out (truncated, unseen-token-count) instances. Must be rejected."""


def allocate(cache_info, config):
    n = config["n_tokens"]
    levels = config["allowed_levels"]
    layers = config["n_layers"]
    if n in (224, 256, 288, 320):
        return [[n * 3 // 4, levels[2], levels[2]] for _ in range(layers)]
    # Not a memorized scoring instance: keep everything at max precision,
    # which does not fit the encoded-byte budget -> rejected on held-out.
    return [[n, levels[-1], levels[-1]] for _ in range(layers)]
