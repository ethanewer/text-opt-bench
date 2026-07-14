# Task: mem_infer — hybrid LLM decode under memory and work limits

Run a batch-one, token-streaming CPU decode for a compact Qwen3.5-inspired
hybrid model while minimizing peak logical tensor memory. CPU Torch executes
the kernels, but neither elapsed time nor allocator behavior affects the
score: tensor storage, kernel scratch, and work are charged deterministically
from shapes and selected kernel schedules.

## Required API

```python
def generate(rt, weights, prompt, n_tokens):
    """Return the n_tokens greedy token ids as a plain list[int]."""
```

`weights` is a nested dictionary of read-only opaque tensor handles. `prompt`
is a plain list of token ids. Programs cannot import numerical libraries or
inspect the Torch tensors; numerical work goes through `rt`.

## Model and workload

There are exactly two pre-norm residual blocks so both hybrid mechanisms are
present without paying for redundant copies of the same optimization problem:

1. a gated DeltaNet recurrent mixer with causal depthwise convolution;
2. a gated grouped-query softmax-attention mixer.

Each block has RMSNorm, a SiLU output gate, and a SwiGLU MLP. A final RMSNorm
feeds a tied embedding/output head.

- vocabulary 96, model width 48, MLP width 96
- DeltaNet: 4 heads, key/value dimensions 16, state `[4,16,16]`
- causal Q/K/V convolution: kernel 4, history `[3,192]`
- GQA: 4 query heads, 1 KV head, head dimension 16
- per-head Q/K RMS normalization and partial RoPE
- maximum context 128; prompt 96; generate 16 tokens

The DeltaNet kernel decays its state before prediction, applies a gated delta
correction, and reads from the updated state. Its normalized Q/K vectors,
structured negative-exponential decay, causal convolution, and per-head output
RMS normalization mirror the important state behavior of current hybrid LLMs.
The attention block uses a shared KV head and causal scaled-dot-product
softmax. This is a small systems proxy, not a claim of weight compatibility
with a released Qwen checkpoint.

## Runtime API

Tensor handles expose only read-only `.shape` and `.dtype`.

```text
rt.zeros(shape, dtype="float32")
rt.free(tensor)
rt.copy(tensor, dtype=None, out=None)
rt.embed(wte, wpe, token, position, out=None)
rt.rmsnorm(x, gain, out=None)       # x and gain have identical shapes
rt.linear(x, weight, out=None)
rt.add(a, b, out=None)             # identical shapes; no broadcasting
rt.mul(a, b, out=None)             # identical shapes; no broadcasting
rt.silu(x, out=None)
rt.sigmoid(x, out=None)
rt.delta_step(q, k, v, decay, beta, state, conv_state,
              conv_weight, a_log, out=None, tile_size=0)
rt.qk_norm_rope(q, k, position, out_q=None, out_k=None)
rt.cache_write(cache, position, value)
rt.attention(q, key_cache, value_cache, length, out=None, block_size=0)
rt.argmax_vocab(x, embedding) -> int
```

Supported storage dtypes are `float32`, `float16`, `bfloat16`, and `qint8`.
Kernels accumulate in float32. `qint8` uses one float32 symmetric scale per
row (the first dimension); both its int8 values and scales count toward the
score. Quantization error is real and can compound in recurrent storage.

`out=` reuses a live candidate-owned buffer of the exact required shape.
`delta_step` mutates its state and convolution history; `cache_write` mutates
its cache. Evaluator-owned inputs are read-only, and freed handles cannot be
used again.

Two schedule controls expose realistic kernel tradeoffs:

- `delta_step(..., tile_size=t)` selects the number of key rows processed per
  recurrent-state tile. Zero means all 16 rows. Smaller tiles use less scratch
  but add deterministic work per tile.
- `attention(..., block_size=b)` runs stable online softmax over KV blocks.
  Zero means one full block. Smaller blocks reduce scratch but add work.

## Correctness

Six deterministic instances are scored. `generate` must call
`argmax_vocab` exactly 16 times and return exactly those greedy token ids.
For every step, the complete 96-way logits must be within **0.035 absolute
error** of float32 execution conditioned on the candidate's own preceding
tokens. Conditioning both executions on the same decode path prevents an
innocent near-tie from turning into an unrelated autoregressive cascade, while
still rejecting inaccurate kernels, truncated context, and fabricated output.

## Score and work constraint

The score is the maximum, over the six instances, of candidate-owned live
tensor bytes plus the current kernel's logical scratch. Lower is better.
Weights are read-only inputs and do not count. Work must not exceed
**18,000,000 units per instance**; this is the task's work limit. The process
CPU timeout is only a runaway guard.

Base work charges are:

```text
zeros / copy-into / cache_write  N
embed                            2 * d_model
rmsnorm                          5 * N
linear [m,n]                     2 * m * n
add / multiply                   N
SiLU / sigmoid                   5 * N
Q/K norm + quarter-head RoPE     14 * (Nq + Nk)
DeltaNet                         8 * heads * key_dim * value_dim
                                   + 8 * convolution_channels
                                   + 4096 * state_tiles
attention at length T            4 * q_heads * T * head_dim
                                   + 8 * q_heads * T + 256 * KV_blocks
vocabulary projection            2 * vocabulary * d_model
qint8 read / write overhead       2 * N / 4 * N
```

Thus an allocating float-to-qint8 copy costs `5 * N`: the base copy plus
qint8-write work. When `out=` is supplied, its dtype must match `dtype` (or it
determines the dtype when that argument is omitted).

Logical scratch is also exact: linear and vocabulary projection use one
float32 output vector; Q/K normalization uses float32 Q and K; DeltaNet uses a
full float32 state plus one selected state tile and convolution scratch;
attention uses the selected float32 K/V and score blocks plus stable-softmax
accumulators. The formulas are implemented directly in `Runtime`, and the
reported metrics include per-instance work and peak bytes.

The optimization surface includes buffer liveness/reuse, cache capacity,
state/cache precision, quantization, recurrent tiling, attention blocking, and
their accuracy/work interactions. Recomputing the entire prefix cannot fit
under the work ceiling.

## Rules

- No imports or class definitions. Available builtins are: `abs`, `all`,
  `any`, `bool`, `dict`, `enumerate`, `filter`, `float`, `int`, `isinstance`,
  `len`, `list`, `map`, `max`, `min`, `print`, `range`, `reversed`, `round`,
  `set`, `slice`, `sorted`, `str`, `sum`, `tuple`, and `zip`, plus ordinary
  exception types.
- Do not access runtime/tensor private attributes or metric-control state.
- Return a plain list containing plain integers, deterministically.
- Do not read or reconstruct sealed evaluator data or hardcode instance
  outputs.
