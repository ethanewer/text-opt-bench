"""Tiny Qwen3.5-inspired hybrid decoder and its metered Torch runtime.

Candidates never receive Torch itself.  They operate on opaque ``Tensor``
handles through ``Runtime`` so both live storage and algorithmic work are
deterministic, backend-independent quantities.  Torch is only the fast CPU
engine underneath those operations.
"""

from __future__ import annotations

import math


VOCAB = 96
D_MODEL = 48
D_FF = 96
N_Q_HEADS = 4
N_KV_HEADS = 1
HEAD_DIM = 16
N_DELTA_HEADS = 4
DELTA_KEY_DIM = 16
DELTA_VALUE_DIM = 16
CONV_KERNEL = 4
DELTA_CHANNELS = (2 * N_DELTA_HEADS * DELTA_KEY_DIM
                  + N_DELTA_HEADS * DELTA_VALUE_DIM)
MAX_CONTEXT = 128
PROMPT_LEN = 96
N_GEN = 16
WORK_BUDGET = 18_000_000
LOGIT_ATOL = 0.035

DTYPES = {
    "float32": (4, "float32"),
    "float16": (2, "float16"),
    "bfloat16": (2, "bfloat16"),
    "qint8": (1, "int8"),
}


class WorkExceeded(RuntimeError):
    pass


class Tensor:
    """Opaque candidate-visible tensor handle."""

    __slots__ = ("_tensor", "_scale", "_runtime", "_bytes", "_alive", "_owned")

    def __init__(self, tensor, runtime=None, owned=False, scale=None):
        self._tensor = tensor
        self._scale = scale
        self._runtime = runtime
        self._bytes = (tensor.numel() * tensor.element_size()
                       + (0 if scale is None else scale.numel() * scale.element_size()))
        self._alive = True
        self._owned = owned

    @property
    def shape(self):
        return tuple(self._tensor.shape)

    @property
    def dtype(self):
        if self._scale is not None:
            return "qint8"
        return str(self._tensor.dtype).rsplit(".", 1)[-1]


class Runtime:
    """Small, metered tensor API used by candidate programs."""

    def __init__(self, torch, budget=WORK_BUDGET):
        self.torch = torch
        self.budget = int(budget)
        self.work = 0
        self.live_bytes = 0
        self.peak_bytes = 0
        self._logit_trace = []
        self._token_trace = []

    def _charge(self, amount):
        self.work += int(amount)
        if self.work > self.budget:
            raise WorkExceeded(
                f"deterministic work budget exceeded ({self.work} > {self.budget})"
            )

    def _check(self, value):
        if type(value) is not Tensor or not value._alive:
            raise TypeError("operation requires a live tensor handle")
        if value._owned and value._runtime is not self:
            raise ValueError("candidate tensor belongs to a different runtime")
        return value._tensor

    def _require_owned(self, value):
        self._check(value)
        if not value._owned or value._runtime is not self:
            raise ValueError("mutation target must be owned by this runtime")
        return value._tensor

    def _float(self, value):
        tensor = self._check(value)
        if value._scale is None:
            return tensor.float()
        shape = (tensor.shape[0],) + (1,) * (tensor.ndim - 1)
        self._charge(2 * tensor.numel())
        return tensor.float() * value._scale.reshape(shape)

    def _store(self, value, tensor):
        target = self._require_owned(value)
        if value._scale is None:
            target.copy_(tensor.to(target.dtype))
            return
        flat = tensor.float().reshape(tensor.shape[0], -1)
        scale = flat.abs().amax(dim=1).clamp_min(1e-8) / 127.0
        quant = self.torch.round(flat / scale[:, None]).clamp(-127, 127)
        target.copy_(quant.reshape(target.shape).to(self.torch.int8))
        value._scale.copy_(scale)
        self._charge(4 * target.numel())

    def _track_peak(self, extra=0):
        peak = self.live_bytes + int(extra)
        if peak > self.peak_bytes:
            self.peak_bytes = peak

    def _wrap(self, tensor, scale=None):
        result = Tensor(tensor, self, True, scale)
        self.live_bytes += result._bytes
        self._track_peak()
        return result

    def _output(self, tensor, out):
        if out is None:
            return self._wrap(tensor.clone())
        target = self._require_owned(out)
        if tuple(target.shape) != tuple(tensor.shape):
            raise ValueError("output buffer has the wrong shape")
        self._store(out, tensor)
        return out

    def zeros(self, shape, dtype="float32"):
        if (type(shape) not in (tuple, list)
                or not shape or not all(type(x) is int and x > 0 for x in shape)):
            raise ValueError("shape must contain positive plain integers")
        if dtype not in DTYPES:
            raise ValueError("unsupported dtype")
        torch_dtype = getattr(self.torch, DTYPES[dtype][1])
        self._charge(math.prod(shape))
        tensor = self.torch.zeros(tuple(shape), dtype=torch_dtype)
        scale = (self.torch.ones(shape[0], dtype=self.torch.float32)
                 if dtype == "qint8" else None)
        return self._wrap(tensor, scale)

    def free(self, value):
        self._check(value)
        self._require_owned(value)
        value._alive = False
        self.live_bytes -= value._bytes

    def copy(self, value, dtype=None, out=None):
        x = self._check(value)
        if dtype is None:
            dtype = out.dtype if out is not None else value.dtype
        if dtype not in DTYPES:
            raise ValueError("unsupported dtype")
        if out is not None and out.dtype != dtype:
            raise ValueError("output buffer dtype does not match requested dtype")
        self._charge(x.numel())
        y = self._float(value)
        if out is not None:
            return self._output(y, out)
        if dtype == "qint8":
            # The base copy charge covers allocation, as for every other
            # destination dtype; quantization itself adds 4N work.
            result = self._wrap(
                self.torch.zeros(tuple(x.shape), dtype=self.torch.int8),
                self.torch.ones(x.shape[0], dtype=self.torch.float32),
            )
            self._store(result, y)
            return result
        return self._wrap(y.to(getattr(self.torch, DTYPES[dtype][1])).clone())

    def embed(self, wte, wpe, token, position, out=None):
        a, b = self._check(wte), self._check(wpe)
        if type(token) is not int or type(position) is not int:
            raise TypeError("token and position must be plain integers")
        self._charge(2 * D_MODEL)
        return self._output(a[token] + b[position], out)

    def rmsnorm(self, x, gain, out=None):
        a, g = self._float(x), self._float(gain)
        if tuple(a.shape) != tuple(g.shape):
            raise ValueError("rmsnorm input and gain must have identical shapes")
        self._charge(5 * a.numel())
        y = a * self.torch.rsqrt((a * a).mean() + 1e-5) * g
        return self._output(y, out)

    def linear(self, x, weight, out=None):
        a, w = self._float(x), self._float(weight)
        if a.ndim != 1 or w.ndim != 2 or w.shape[1] != a.shape[0]:
            raise ValueError("linear expects vector and [out,in] matrix")
        self._charge(2 * w.numel())
        self._track_peak(w.shape[0] * 4)
        y = self.torch.mv(w, a)
        return self._output(y, out)

    def add(self, a, b, out=None):
        x, y = self._float(a), self._float(b)
        if tuple(x.shape) != tuple(y.shape):
            raise ValueError("add operands must have identical shapes")
        self._charge(x.numel())
        return self._output(x + y, out)

    def mul(self, a, b, out=None):
        x, y = self._float(a), self._float(b)
        if tuple(x.shape) != tuple(y.shape):
            raise ValueError("multiply operands must have identical shapes")
        self._charge(x.numel())
        return self._output(x * y, out)

    def silu(self, x, out=None):
        a = self._float(x)
        self._charge(5 * a.numel())
        return self._output(self.torch.nn.functional.silu(a), out)

    def sigmoid(self, x, out=None):
        a = self._float(x)
        self._charge(5 * a.numel())
        return self._output(self.torch.sigmoid(a), out)

    def delta_step(self, q, k, v, decay, beta, state, conv_state,
                   conv_weight, a_log, out=None, tile_size=0):
        tq, tk, tv = self._float(q), self._float(k), self._float(v)
        td, tb = self._float(decay), self._float(beta)
        ts = self._float(state)
        tc = self._float(conv_state)
        cw, alog = self._float(conv_weight), self._float(a_log)
        self._require_owned(state)
        self._require_owned(conv_state)
        expected = (N_DELTA_HEADS, DELTA_KEY_DIM, DELTA_VALUE_DIM)
        if tuple(ts.shape) != expected:
            raise ValueError("DeltaNet state has the wrong shape")
        if tuple(tc.shape) != (CONV_KERNEL - 1, DELTA_CHANNELS):
            raise ValueError("DeltaNet convolution state has the wrong shape")
        if (type(tile_size) is not int or tile_size < 0
                or tile_size > DELTA_KEY_DIM):
            raise ValueError("invalid DeltaNet tile size")
        current = self.torch.cat((tq, tk, tv))
        sequence = self.torch.cat((tc, current.unsqueeze(0)), dim=0)
        mixed = self.torch.nn.functional.silu((sequence.transpose(0, 1) * cw).sum(1))
        self._store(conv_state, sequence[1:])
        q_end = N_DELTA_HEADS * DELTA_KEY_DIM
        k_end = 2 * q_end
        qf = mixed[:q_end].reshape(N_DELTA_HEADS, DELTA_KEY_DIM)
        kf = mixed[q_end:k_end].reshape(N_DELTA_HEADS, DELTA_KEY_DIM)
        vf = mixed[k_end:].reshape(N_DELTA_HEADS, DELTA_VALUE_DIM)
        qf = self.torch.nn.functional.normalize(qf, dim=-1) / math.sqrt(DELTA_KEY_DIM)
        kf = self.torch.nn.functional.normalize(kf, dim=-1)
        # Qwen-style negative exponential decay. Prediction uses the decayed
        # state, then the gated delta correction is applied.
        rate = self.torch.nn.functional.softplus(td).reshape(N_DELTA_HEADS)
        df = self.torch.exp(-self.torch.exp(alog) * rate).reshape(N_DELTA_HEADS, 1, 1)
        bf = self.torch.sigmoid(tb).reshape(N_DELTA_HEADS, 1, 1)
        decayed = df * ts
        pred = self.torch.bmm(kf.unsqueeze(1), decayed).squeeze(1)
        err = vf - pred
        updated = decayed + bf * kf.unsqueeze(2) * err.unsqueeze(1)
        self._store(state, updated)
        restored = self._float(state)
        result = self.torch.bmm(qf.unsqueeze(1), restored).squeeze(1)
        result = result * self.torch.rsqrt((result * result).mean(-1, keepdim=True) + 1e-5)
        result = result.reshape(-1)
        h, kdim, vdim = N_DELTA_HEADS, DELTA_KEY_DIM, DELTA_VALUE_DIM
        tile = tile_size or kdim
        tiles = (kdim + tile - 1) // tile
        # Smaller state tiles model a lower-scratch recurrent kernel, with a
        # deterministic launch/control cost per tile. This makes kernel
        # scheduling a real memory/work tradeoff rather than a free flag.
        self._charge(8 * h * kdim * vdim + 8 * DELTA_CHANNELS
                     + 4096 * tiles)
        self._track_peak((h * kdim * vdim + h * tile * vdim
                          + (CONV_KERNEL + 1) * DELTA_CHANNELS) * 4)
        return self._output(result, out)

    def cache_write(self, cache, position, value):
        c, v = self._check(cache), self._float(value)
        self._require_owned(cache)
        if (type(position) is not int
                or math.prod(c.shape[1:]) != v.numel()):
            raise ValueError("cache write shape mismatch")
        self._charge(v.numel())
        row = v.reshape(c.shape[1:])
        if cache._scale is None:
            c[position].copy_(row.to(c.dtype))
        else:
            scale = row.abs().max().clamp_min(1e-8) / 127.0
            c[position].copy_(self.torch.round(row / scale).clamp(-127, 127).to(self.torch.int8))
            cache._scale[position] = scale
            self._charge(4 * row.numel())

    def qk_norm_rope(self, q, k, position, out_q=None, out_k=None):
        """Per-head Q/K RMS normalization plus partial rotary embedding."""
        tq, tk = self._float(q), self._float(k)
        if (type(position) is not int or position < 0
                or tq.numel() != N_Q_HEADS * HEAD_DIM
                or tk.numel() != N_KV_HEADS * HEAD_DIM):
            raise ValueError("invalid Q/K tensors or rotary position")

        def transform(flat, heads):
            x = flat.reshape(heads, HEAD_DIM)
            x = x * self.torch.rsqrt((x * x).mean(-1, keepdim=True) + 1e-5)
            # Qwen3.5 uses a 0.25 partial-RoPE factor. Frequencies are fixed
            # model constants, so no evaluator-owned tensor is exposed.
            rotary = HEAD_DIM // 4
            half = rotary // 2
            idx = self.torch.arange(half, dtype=self.torch.float32)
            angle = float(position) / (10000.0 ** (2.0 * idx / rotary))
            c, s = self.torch.cos(angle), self.torch.sin(angle)
            first, second = x[:, :half], x[:, half:rotary]
            rotated = self.torch.cat(
                (first * c - second * s, first * s + second * c,
                 x[:, rotary:]), dim=1)
            return rotated.reshape(-1)

        rq = transform(tq, N_Q_HEADS)
        rk = transform(tk, N_KV_HEADS)
        self._charge(14 * (tq.numel() + tk.numel()))
        self._track_peak(4 * (tq.numel() + tk.numel()))
        return self._output(rq, out_q), self._output(rk, out_k)

    def attention(self, q, key_cache, value_cache, length, out=None,
                  block_size=0):
        tq = self._float(q)
        keys_raw, values_raw = self._check(key_cache), self._check(value_cache)
        if type(length) is not int or not 1 <= length <= keys_raw.shape[0]:
            raise ValueError("invalid attention length")
        if type(block_size) is not int or block_size < 0:
            raise ValueError("block_size must be a nonnegative integer")
        qh = tq.reshape(N_Q_HEADS, HEAD_DIM)

        def cache_slice(handle, raw, lo, hi):
            part = raw[lo:hi].float()
            if handle._scale is not None:
                part = part * handle._scale[lo:hi].reshape(hi - lo, 1, 1)
                self._charge(2 * part.numel())
            return part[:, 0, :]

        if block_size == 0 or block_size >= length:
            kh = cache_slice(key_cache, keys_raw, 0, length)
            vh = cache_slice(value_cache, values_raw, 0, length)
            scores = self.torch.matmul(qh, kh.transpose(0, 1)) / math.sqrt(HEAD_DIM)
            probs = self.torch.softmax(scores, dim=-1)
            result = self.torch.matmul(probs, vh).reshape(-1)
            workspace = ((2 * length * N_KV_HEADS * HEAD_DIM
                          + 2 * N_Q_HEADS * length + N_Q_HEADS * HEAD_DIM) * 4)
            blocks = 1
        else:
            m = self.torch.full((N_Q_HEADS,), -self.torch.inf)
            denom = self.torch.zeros(N_Q_HEADS)
            acc = self.torch.zeros((N_Q_HEADS, HEAD_DIM))
            blocks = 0
            for lo in range(0, length, block_size):
                hi = min(length, lo + block_size)
                kh = cache_slice(key_cache, keys_raw, lo, hi)
                vh = cache_slice(value_cache, values_raw, lo, hi)
                scores = self.torch.matmul(qh, kh.transpose(0, 1)) / math.sqrt(HEAD_DIM)
                new_m = self.torch.maximum(m, scores.max(dim=1).values)
                old_scale = self.torch.exp(m - new_m)
                probs = self.torch.exp(scores - new_m[:, None])
                acc = acc * old_scale[:, None] + self.torch.matmul(probs, vh)
                denom = denom * old_scale + probs.sum(dim=1)
                m = new_m
                blocks += 1
            result = (acc / denom[:, None]).reshape(-1)
            b = min(block_size, length)
            workspace = ((2 * b * N_KV_HEADS * HEAD_DIM
                          + 2 * N_Q_HEADS * b + N_Q_HEADS * (HEAD_DIM + 3)) * 4)
        self._charge(4 * N_Q_HEADS * length * HEAD_DIM
                     + 8 * N_Q_HEADS * length + 256 * blocks)
        self._track_peak(workspace)
        return self._output(result, out)

    def argmax_vocab(self, x, embedding):
        a, w = self._float(x), self._float(embedding)
        self._charge(2 * w.numel())
        self._track_peak(VOCAB * 4)
        logits = self.torch.mv(w, a)
        self._logit_trace.append(logits.clone())
        token = int(self.torch.argmax(logits).item())
        self._token_trace.append(token)
        return token


def _randn(torch, generator, *shape, scale=0.12):
    return torch.randn(shape, generator=generator, dtype=torch.float32) * scale


def build_weights(torch, seed):
    """Build compact deterministic weights; returned tensors are unscored inputs."""
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    raw = {
        "wte": _randn(torch, g, VOCAB, D_MODEL, scale=0.25),
        "wpe": _randn(torch, g, MAX_CONTEXT, D_MODEL, scale=0.08),
        "delta": {}, "attn": {},
        "lnf": torch.ones(D_MODEL),
    }
    for name, out_dim in (
        ("wq", N_DELTA_HEADS * DELTA_KEY_DIM),
        ("wk", N_DELTA_HEADS * DELTA_KEY_DIM),
        ("wv", N_DELTA_HEADS * DELTA_VALUE_DIM),
        ("wdecay", N_DELTA_HEADS), ("wbeta", N_DELTA_HEADS),
        ("wgate", N_DELTA_HEADS * DELTA_VALUE_DIM),
        ("wo", D_MODEL), ("wup", D_FF), ("wgate_ff", D_FF),
        ("wdown", D_MODEL),
    ):
        in_dim = (N_DELTA_HEADS * DELTA_VALUE_DIM if name == "wo"
                  else D_FF if name == "wdown" else D_MODEL)
        raw["delta"][name] = _randn(torch, g, out_dim, in_dim)
    raw["delta"]["ln1"] = torch.ones(D_MODEL)
    raw["delta"]["ln2"] = torch.ones(D_MODEL)
    raw["delta"]["conv"] = _randn(
        torch, g, DELTA_CHANNELS, CONV_KERNEL, scale=0.18)
    raw["delta"]["a_log"] = torch.linspace(-0.7, 0.3, N_DELTA_HEADS)
    for name, out_dim in (
        ("wq", N_Q_HEADS * HEAD_DIM), ("wk", N_KV_HEADS * HEAD_DIM),
        ("wv", N_KV_HEADS * HEAD_DIM), ("wgate", N_Q_HEADS * HEAD_DIM),
        ("wo", D_MODEL), ("wup", D_FF), ("wgate_ff", D_FF),
        ("wdown", D_MODEL),
    ):
        in_dim = (N_Q_HEADS * HEAD_DIM if name == "wo"
                  else D_FF if name == "wdown" else D_MODEL)
        raw["attn"][name] = _randn(torch, g, out_dim, in_dim)
    raw["attn"]["ln1"] = torch.ones(D_MODEL)
    raw["attn"]["ln2"] = torch.ones(D_MODEL)

    def wrap(value):
        if type(value) is dict:
            return {k: wrap(v) for k, v in value.items()}
        return Tensor(value, None, False)
    return wrap(raw)


def build_prompt(seed):
    # Independent arithmetic sequence; no PRNG state is exposed to candidates.
    return [int((seed * 17 + i * i * 13 + i * 29) % VOCAB)
            for i in range(PROMPT_LEN)]


def reference_generate(torch, weights, prompt, n_tokens, forced_tokens=None):
    """Canonical float32 execution, optionally on a supplied decode path."""
    rt = Runtime(torch, budget=10**12)
    state = rt.zeros((N_DELTA_HEADS, DELTA_KEY_DIM, DELTA_VALUE_DIM))
    conv = rt.zeros((CONV_KERNEL - 1, DELTA_CHANNELS))
    kc = rt.zeros((MAX_CONTEXT, N_KV_HEADS, HEAD_DIM))
    vc = rt.zeros((MAX_CONTEXT, N_KV_HEADS, HEAD_DIM))
    tokens = list(prompt)
    out = []
    for pos in range(PROMPT_LEN + n_tokens - 1):
        x = rt.embed(weights["wte"], weights["wpe"], tokens[pos], pos)
        x = _reference_layer(rt, weights["delta"], x, state, conv, None, None, pos)
        x = _reference_layer(rt, weights["attn"], x, None, None, kc, vc, pos)
        if pos >= PROMPT_LEN - 1:
            h = rt.rmsnorm(x, weights["lnf"])
            predicted = rt.argmax_vocab(h, weights["wte"])
            step = len(out)
            token = (forced_tokens[step] if forced_tokens is not None
                     else predicted)
            out.append(predicted)
            tokens.append(token)
    return out, rt._logit_trace


def _reference_layer(rt, layer, x, state, conv, kc, vc, pos):
    h = rt.rmsnorm(x, layer["ln1"])
    q = rt.linear(h, layer["wq"])
    k = rt.linear(h, layer["wk"])
    v = rt.linear(h, layer["wv"])
    if state is not None:
        decay = rt.linear(h, layer["wdecay"])
        beta = rt.linear(h, layer["wbeta"])
        mixed = rt.delta_step(q, k, v, decay, beta, state, conv,
                              layer["conv"], layer["a_log"])
    else:
        q, k = rt.qk_norm_rope(q, k, pos)
        rt.cache_write(kc, pos, k)
        rt.cache_write(vc, pos, v)
        mixed = rt.attention(q, kc, vc, pos + 1)
    gate = rt.silu(rt.linear(h, layer["wgate"]))
    mixed = rt.mul(mixed, gate)
    x = rt.add(x, rt.linear(mixed, layer["wo"]))
    h = rt.rmsnorm(x, layer["ln2"])
    up = rt.linear(h, layer["wup"])
    gate = rt.silu(rt.linear(h, layer["wgate_ff"]))
    x = rt.add(x, rt.linear(rt.mul(up, gate), layer["wdown"]))
    return x
