> **PROVENANCE — drafted pre-hardening (2026-07-05); superseded in part by the hardening redesign (design §2/§2.5; hardening report §2.1 + Addendum A).**
> - `initial_program.py` is now the exact first-step-P(S) coordinate-ascent optimum (prefix-KV-cached, deterministic) — NOT `[min(target_set)]*8`; the floor sits at the CA rung (~0.72–0.78 per instance), not ~0.82.
> - K = {train 16, val **128**, test 512} (k_val 48→128; visible SE 0.0162→0.0095). Splits 10/6/8 are rejection-sampled on a measured ≥0.03 (=3.2 SE) CA-vs-deep-policy headroom gate.
> - New broken fixture `prompt_steer_slow.py`; governance v1.1 limits cpu_s 400 / timeout_s 900 (was 600).
> - Status: **hardened_needs_pilot** (frozen 6-instance k=200 ladder aggregate + gate acceptance rate + codex pilot pending; consider gate 0.035–0.04). NOTE: the type-1 `prompt_steer` was REJECTED at the hardening gate — this file specs only the surviving type-2 twin.

# Task: prompt_steer — steer a tiny transformer's sampler by prompt design

You are given a family of tiny GPT-style decoders (VOCAB 64, D_MODEL 48,
4 heads, D_FF 96, 2 layers, CTX 20 — the mem_infer architecture) and, per
instance, a **target token set S** of 6 token ids. Your job is to design an
8-token prompt that steers temperature-1.0 sampled continuations toward the
target tokens. The model is fixed; the prompt is your only lever.

## Required API (module-level function in program.py)

```python
def make_prompt(weights, target_set, n_gen):
    """weights: dict of model parameters for one instance (same schema as
    the weights you can build yourself from data/train.jsonl — see below).
    target_set: sorted list of 6 ints (the target token set S).
    n_gen: number of tokens the evaluator will sample (always 8).
    Return a plain list of exactly 8 plain ints, each in [0, 64)."""
```

`make_prompt` runs inside the evaluator, once per instance, with that
instance's freshly built weights. It may spend a few CPU-seconds per
instance on its own computation (budget ~6 s/instance as a guideline). It
emits tokens only — **the evaluator owns all sampling**.

## Scoring (lower is better)

For each instance the evaluator feeds your 8-token prompt to the model and
ancestrally samples `n_gen = 8` tokens `x_0..x_7` from
`softmax(logits / 1.0)` (logits rounded to 9 decimal places before softmax).
A rollout is worth

```
V = sum_{j=0}^{7} 0.6**j * [x_j in S]        VMAX = sum_j 0.6**j ≈ 2.458010
```

Per instance, V is averaged over k independent rollouts; per split,

```
score = round(1 - (sum_i mean_k V_i) / (n_instances * VMAX), 6)   # in [0, 1)
```

Splits:

- **train**: 10 instances, `data/train.jsonl` (plaintext: `{"id", "wseed",
  "S"}`), k = 16. Reported as `train_score` in metrics; never the score.
- **val**: 6 sealed instances (`data/heldout_val.bin`), k = 48. **The
  reported score is `val_score`.**
- **test**: 8 sealed instances (`data/heldout_test.bin`), k = 512. Graded
  only with `--final`; your session-best program (by visible val score) is
  regraded this way and the official number is `test_score`.

`--train-only` scores/reports train only (fast self-test).

Train instances are fully visible: build their weights from `wseed` and
experiment offline as much as you like — that is the intended workflow.
Only an algorithm that transfers to unseen weights/target sets earns the
val/test score.

## Randomness contract

Your visible grade is a k=48-per-instance Monte Carlo estimate of a fixed
expectation — expect roughly ±0.015 (1 SE) wiggle between submissions of
the same program. Each submission's rollout seed is drawn fresh by the
harness, recorded, and replayable (`verify --rescore`); you cannot choose
or predict it. Running `evaluate.py` yourself without `--seed` uses a fixed
default seed and is bit-exact — re-running locally does NOT give fresh
draws. Rollout RNGs are private to the evaluator; reseeding or
monkeypatching the global `random` module has no effect on sampling.

`make_prompt` must be deterministic. The evaluator courtesy-seeds the
global `random` module with a fixed per-instance constant before calling
you, but seed any RNG you use yourself with a constant to be safe.

## Validation

The return value must satisfy `type(prompt) is list`, `len(prompt) == 8`,
every element `type(t) is int` and `0 <= t < 64`. Anything else — wrong
type, subclasses, generators, out-of-range tokens — or any exception fails
the whole evaluation. Failures on held-out instances are reported without
naming the instance. The evaluator rebuilds each instance's weights before
sampling, so mutating the `weights` dict you are handed changes nothing.

## Rules

- Python 3.12 stdlib only. **Forbidden** (checked): `model`, `bench`,
  `sys`, `os`, `io`, `open`, `pathlib`, `mmap`, `ctypes`, `socket`,
  `subprocess`, `multiprocessing`, `threading`, `tempfile`, `shutil`,
  `sqlite3`, `dbm`, `shelve`, `importlib`, `__import__`, `tracemalloc`,
  `resource`. `random`, `math`, `copy`, `itertools` are allowed.
- Importing `model` is forbidden inside your program — reimplement any
  forward pass you need yourself. Outside the evaluator (in your own
  scripts), you may use the shipped `model.py` builder with the train
  `wseed`s to reproduce the train weights exactly.
- Decoding `data/heldout_*.bin`, importing anything under `bench.*`, or
  invoking the evaluator with self-chosen `--seed` values to average or
  shop the val draw is **cheating** and fails audit. Train-side local
  experimentation is unrestricted and encouraged.
- The whole evaluation must fit in 400 s of CPU time (16 `make_prompt`
  calls plus 448 evaluator rollouts on the default grade).

## Baseline

`initial_program.py` returns `[min(target_set)] * 8` — legal, weak.
Expected val score around **0.82**. Untargeted prompts land near 0.9; the
gap below the baseline is real and reachable. A useful first fact: the
exact next-token distribution after any prompt is one forward pass
(~9 ms in pure Python at this size).
