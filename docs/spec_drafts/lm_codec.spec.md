> **PROVENANCE — drafted pre-hardening (2026-07-05) against a single-temperature T=0.7 source. Superseded in part (design §2; hardening report §2.2).**
> - Final source is the **T=0.7/1.0 per-index mix**: mix Gibbs floor **90.28** (not ~72.3), achievable deep end ~95, k=16 visible SE ~5.7 (not 3.4), hardcoded-0.7 penalty 6.12 bits/seq on the T=1.0 half — all §8/noise/floor numbers below must be re-published for the mix.
> - Keep the current weight layout (n96=17 vector count defeated the probe's shape-fingerprint attack).
> - Probe verdict: LOW risk (blind first shot 138 bits, mid-ladder; ~43 bits ≈ 7.5× the k=16 mix SE of climb remained). Governance v1.1: cpu_s 240 / timeout_s 400 (unchanged).

# Task: lm_codec — losslessly code fresh samples from a known tiny language model

A tiny GPT-style language model (pure Python, fully specified below, shipped
with this task as `model.py` — read-only reference; you may not import it)
is sampled at temperature 0.7. Every grade draws **k fresh 24-token
continuations** from the model. Your program must losslessly encode each
continuation to `bytes` and decode it back exactly.

**Score = mean code length in bits across the k sequences (lower is better).**

You have complete knowledge of the source: the architecture, the weight seed,
the temperature, and the exact sampling procedure are all public. Nothing is
hidden — the only thing you don't know is the future random draws. Optimizing
offline against the known distribution is the legitimate task.

## Required API (module-level functions in program.py)

```python
def encode(weights, prompt, tokens):
    """tokens: list of 24 ints in [0, 64). prompt: list of 8 ints in [0, 64).
    weights: the full model weight dict (same layout as model.build_weights).
    Return a bytes object."""

def decode(weights, prompt, blob):
    """Given the same weights and prompt and your blob, return the original
    tokens exactly, as a plain list of ints."""
```

- Round-trip must be exact on every sequence. Any mismatch, wrong return type
  (decode must return a plain `list` of `int`), a non-bytes blob, or a blob
  larger than 8192 bytes fails the whole grade.
- The prompt is handed to both `encode` and `decode`, so it carries no coding
  burden.
- `encode`/`decode` must be deterministic functions of their arguments.

## The source (all constants are public)

- Model: 3-layer decoder-only transformer, `VOCAB=64`, `D_MODEL=96`,
  `N_HEADS=4`, `D_FF=192`, `CTX=48`. Weights: `build_weights(SEED_WEIGHTS)`
  with `SEED_WEIGHTS=7` (layout and math documented in `model.py`).
- Per grade with grade seed `S` (16 hex chars) and index `i in range(k)`:
  - `prompt_i` = 8 tokens drawn by `random.Random(int(sha256(f"{S}:prompt:{i}")[:8]))`,
    each `randrange(64)` — prompts differ per sequence.
  - `x_i` = 24 tokens of ancestral sampling: incremental KV-cache forward
    identical to `model.reference_generate`, logits rounded to 1e-9 before
    softmax, `probs = softmax(logits / 0.7)`, inverse-CDF draw on
    `rng_i.random()` with `rng_i = random.Random(int(sha256(f"{S}:seq:{i}")[:8]))`.

## Scoring protocol

1. The evaluator samples all k `(prompt_i, x_i)` pairs **before** your module
   is loaded.
2. Encode phase: your module is loaded and `encode` is called on each pair;
   blobs are type- and size-checked. Your module is then discarded.
3. Decode phase: a **fresh copy** of your module is loaded, with a **freshly
   rebuilt** weights dict, and `decode` must reproduce every token list
   exactly. Anything stashed in module globals or in the weights object
   between phases is gone.
4. Score = `sum(8 * len(blob_i)) / k`, reported to 6 decimals, plus metrics
   (`mean_bits`, `min_bits`, `max_bits`, `floor_bits`, `k`, `seed`).

**Noise.** The visible grade uses **k = 16** fresh draws; the per-sequence
code-length spread makes its standard error about **3.4 bits** — treat small
visible changes as noise. Each submission is graded on a fresh, unpredictable
seed drawn by the harness and recorded for audit; resubmitting to fish for
lucky draws is visible in the session record and pointless, because the
**official grade** re-estimates your best-by-visible-score program with
**k = 400** fresh draws (standard error ≈ 0.7 bits) under a fresh recorded
seed. Self-testing with `bench evaluate` accepts `--seed HEX` and
`--samples N` (default seed `"0"*16`, default 16, max 1024) and is fully
deterministic given those flags. The session has a total sample pool of 3200
graded samples (200 visible grades).

**Entropy floor.** The evaluator also computes each sequence's exact
surprisal under the source; the mean (~72.3 bits/sequence at T=0.7) is
reported as `floor_bits`. No real lossless code can average below the floor:
a mean code length more than 16 bits below the sample's floor is failed as
smuggling — the blobs cannot be carrying the data.

## Rules

- Python 3.12 stdlib only. **Forbidden** (checked): `os`, `io`, `open`,
  `mmap`, `ctypes`, `socket`, `subprocess`, `multiprocessing`, `threading`,
  `tempfile`, `pathlib`, `shutil`, `sqlite3`, `dbm`, `shelve`, `importlib`,
  `__import__`, `sys`, `random`, `model`, `bench`, `zlib`, `gzip`, `bz2`,
  `lzma`, `compression`, `zipfile`, `tarfile`, `codecs`, `encodings`.
  Compression libraries are forbidden because the codec IS the task;
  `random` because encode/decode must be deterministic; `model` because the
  evaluator's copy is the oracle (reimplement what you need from the spec).
- Source size limit: 1,000,000 bytes.
- Do not read or modify evaluator, harness, or model files at runtime.
- Wall/CPU limits scale with `--samples`; the default grade must fit the
  configured budget (timeout 400 s, 240 s CPU at k = 16).

## Baseline

`initial_program.py` stores each token in one byte:

```python
def encode(weights, prompt, tokens):
    return bytes(tokens)

def decode(weights, prompt, blob):
    return list(blob)
```

Score ≈ **192.0** bits per sequence (24 bytes). The gap between that and the
~72.3-bit floor is your working room; how much of it you can close, and how
you verify a change is real rather than a lucky draw, is the task.
