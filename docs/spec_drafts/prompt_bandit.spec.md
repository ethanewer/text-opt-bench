> **PROVENANCE — drafted pre-hardening (2026-07-05). No mechanics changed, but hardening added gates and context (design §2/§2.5; hardening report §2.2, Addendum C).**
> - SNR pilot gate: **zero-oracle strategies must lose by ≥1 sd** before freeze.
> - The deep end (~1.49) is an **extrapolation** — the 13×-budget skyline probe (1.5988 ± 0.0356) did not separate from the reference (1.5557 ± 0.0153); the pilot must confirm the deep end exists before any headroom claims go in GOAL.md. The ladder is narrow (span ~0.22–0.28, rungs 1–2× visible noise), so the sealed test k=256 grade stays non-negotiable.
> - Probe: LOW (borderline) — three blind draws (1.5994 / 1.562 / 1.6222) straddled the verdict line; judge one-shot risk by the best draw. Governance v1.1: cpu_s 300 / timeout_s **700**.

# Task: prompt_bandit — black-box prompt search against a metered stochastic rollout API

Find prompts that steer a hidden language model's random continuations toward
a target set — using only a strictly limited number of sampled rollouts per
instance. Lower score is better.

## The system

Each *instance* is a tiny GPT-style decoder (vocab 64, d_model 48, 4 heads,
2 layers) with **hidden weights**, plus a public *target set* `S` of 8 tokens.
Given an 8-token prompt, the model samples an 8-token continuation by ancestral
sampling at temperature 1.0 (each next token drawn from the softmax over
logits). The architecture and the scoring formula are fully disclosed; the
weights are not — your only channel to an instance is the metered oracle
described below. Every continuation is a fresh random draw.

## Required API (module-level function in program.py)

```python
def search(rollout, rng, n_slots, targets, budget):
    """Find a good prompt for ONE instance.

    rollout(prompt) -> list[int]: samples ONE fresh 8-token continuation of
        `prompt` and spends 1 call from the budget. `prompt` must be a list or
        tuple of exactly n_slots plain ints in [0, 64); invalid arguments
        raise ValueError and spend NO budget. Calling after the budget is
        exhausted raises RuntimeError. The oracle is permanently closed
        (raises) once search() returns — do not stash it.
    rng: a random.Random instance — your ONLY permitted source of randomness.
    n_slots: 8 (prompt length).
    targets: frozenset of 8 ints in [0, 64). Target tokens ARE allowed in
        prompts.
    budget: 200 (oracle calls for this instance).

    Return a plain list of n_slots plain ints in [0, 64)."""
```

`search` is called once per instance, sequentially over train, then val, then
(at final grading) test instances. No split label is passed; instances are
independent and the budget cannot be pooled or carried across instances.

## Objective and score (exact)

For a sampled continuation `t_0..t_7` and target set `S`:

```
value = sum over positions j of 0.5**j  where t_j is in S
        AND t_j is the FIRST occurrence of that target in the continuation
```

Each **distinct** target scores at most once, at its first position; repeats
of an already-seen target score nothing. Maximum value ≈ 1.9922. Worked
example: with `S = {5, 9, ...}` and continuation `[5, 5, 9, 3, ...]`, token 5
scores 0.5**0 = 1.0 at position 0 (its second occurrence scores nothing), and
9 scores 0.5**2 = 0.25 — value 1.25.

After `search()` returns, the evaluator grades your prompt on `k` **fresh**
scoring rollouts you never see (drawn from a stream independent of the oracle
stream):

```
instance_score = 2.0 - mean(value over k fresh rollouts)
split score    = mean over the split's instances, rounded to 6 decimals
```

| split | instances | oracle budget | scoring k | visibility |
|-------|-----------|---------------|-----------|------------|
| train | 3 (seeds published in `data/train_instances.json`) | 200 | 32 | diagnostic only |
| val   | 4 (sealed in `data/heldout_val.bin`) | 200 | 48 | **score = val_score** |
| test  | 6 (sealed in `data/heldout_test.bin`) | 200 | 256 | reported only with `--final` |

Score range is [0.0078, 2.0]; lower is better. `--train-only` runs and scores
the train split only.

## Seeds and measurement

- `evaluate.py` accepts an optional `--seed <16 hex chars>`; without it a fixed
  default seed is used, making local runs and self-tests bit-exact and
  deterministic.
- Every random stream (oracle draws, your `rng`, scoring draws) is derived
  from the master seed by domain separation; given (program, seed) the entire
  evaluation is reproducible bit-for-bit.
- Each official submission is graded under a **fresh harness-drawn seed** that
  you cannot choose; it is recorded so the run can be rescored exactly.
- Reported grades are Monte-Carlo estimates of an expectation: the visible
  val_score carries a standard error of roughly 0.03 from scoring noise alone,
  plus run-to-run variation from your own search's luck. **Resubmitting an
  unchanged program is seed-fishing: it cannot improve your expected official
  grade and wastes ~96 s of evaluator time per attempt.** Improve the program,
  not the seed.

## Rules

- Python 3.12 stdlib only. **Forbidden** (checked): `os`, `io`, `open`,
  `mmap`, `ctypes`, `socket`, `subprocess`, `multiprocessing`, `threading`,
  `tempfile`, `pathlib`, `shutil`, `sqlite3`, `dbm`, `shelve`, `importlib`,
  `__import__`, `sys`, `resource`, `tracemalloc`, `model`, `hashlib`, and —
  entropy confiscation — `random`, `secrets`, `uuid`, `time`, `datetime`.
  The passed `rng` is your only randomness.
- Do NOT read or decode `data/heldout_val.bin` / `data/heldout_test.bin`;
  doing so is cheating by rule.
- Do not access the oracle's or evaluator's internals (`__closure__`,
  `__globals__`, etc. are rejected); do not keep oracle references past
  `search()`'s return.
- `search()` must return a plain `list` of plain `int`s, validated
  immediately; exceptions on held-out instances fail the run.
- Keep per-call overhead modest: the whole evaluation must fit in 300 s of
  CPU (evaluator work alone is ~96 s for a final grade).

## Baseline

`initial_program.py` draws 10 uniform-random candidate prompts, spends 20
oracle calls estimating each, and returns the one with the best sample mean.
Its expected val score is about **1.59**, with sizable run-to-run variation.

## Self-test

```
python3 evaluate.py program.py --train-only        # fast diagnostic on the 3 public train instances
python3 evaluate.py program.py                     # full visible grade (train + val)
python3 evaluate.py program.py --seed <16 hex>     # reproduce/vary a specific run
```

The train instance seeds are published: you may rebuild those instances (or
new ones like them) locally and study your algorithm offline with as many
simulated rollouts as you like — only the sealed val/test instances are
off-limits. Useful facts: a single rollout's value has a standard deviation of
roughly 0.4, so 20 pulls of one candidate give a mean with SE ≈ 0.09; token
order in the continuation matters (earlier hits are worth exponentially more);
and each distinct target scores only once.