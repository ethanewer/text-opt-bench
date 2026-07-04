# Task: mem_infer — LLM decode step under minimal peak memory

Run greedy token generation for a tiny GPT-style decoder using as little
**peak memory** as possible. This is a perfect-information task: the
score is exactly the quantity being optimized — there is nothing to
overfit.

## Required API (module-level function in program.py)

```python
def generate(weights, prompt, n_tokens):
    """weights: dict of plain Python lists (see below).
    prompt: list[int] of token ids. n_tokens: int.
    Return list[int]: the next n_tokens greedy-argmax tokens."""
```

## Model (weights are given; architecture is fixed)

Tiny decoder-only transformer, all values plain Python floats:
vocab 64, d_model 48, 4 heads, d_ff 96, 2 layers, context 20
(prompt 8 + 12 generated).

- `weights["wte"]` (64x48 token embedding, also the tied output head),
  `weights["wpe"]` (20x48 positions)
- `weights["layers"]` — list of dicts with `wq wk wv wo` (48x48),
  `w1` (48x96), `b1`, `w2` (96x48), `b2`, and layer-norm params
  `ln1_g ln1_b ln2_g ln2_b`
- Per layer: `x += wo @ attention(layernorm1(x))` with causal multi-head
  attention (scores scaled by 1/sqrt(12), softmax with max subtraction),
  then `x += w2 @ relu(w1 @ layernorm2(x) + b1) + b2`
- Final layer norm (`lnf_g`, `lnf_b`), logits = hidden @ wte^T, next
  token = argmax (ties: lowest index; margins are comfortably large).

Your output must match the reference greedy decode token-for-token. Small
float differences from reordering are fine in practice (argmax margins
>= 0.02), but keep the math faithful. (The reference is computed by the
evaluator; you must reproduce it from `weights` and `prompt` yourself —
importing the evaluator's helper modules is forbidden, see Rules.)

## Scoring (lower is better)

`generate` is called on three instances (two with visible seeds, one
loaded from held-out data), each measured separately with `tracemalloc`.
Score = the **maximum peak traced bytes** over the three calls. The
tracemalloc window opens before your module is imported, so import-time
allocations and precomputation count toward the first peak. The input
`weights` themselves are allocated outside the window — reading them is
free; copying them is not.

All three outputs must match the reference exactly or the run is invalid.

## Rules

- Python 3.12 stdlib only. **Forbidden** (checked): `os`, `io`, `open`,
  `mmap`, `ctypes`, `socket`, `subprocess`, `multiprocessing`,
  `threading`, `tempfile`, `pathlib`, `shutil`, `sqlite3`, `dbm`,
  `shelve`, `importlib`, `__import__`, `sys`, `tracemalloc`, `resource`,
  and `model` / `bench` (the evaluator's own modules — they hold the
  reference decoder and the scorer). `math`, `array`, `struct`, in-place
  buffer reuse, recomputation instead of caching — all fair game.
  `generate` must return a plain `list` of ints.
- Must be deterministic.
- Off-limits: decoding `data/heldout_validation.bin` or precomputing
  outputs for specific instances (the held-out instance's peak is part of
  the score precisely so that hardcoding cannot win).
- CPU guard 240 s: recomputing to save memory is allowed and feasible,
  but mind pure-Python speed.
