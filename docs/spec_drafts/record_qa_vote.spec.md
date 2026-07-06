> **PROVENANCE — drafted pre-hardening (2026-07-05); describes v1. Superseded in part by the hardening round (design §2/§2.5; hardening report §3 + Addendum A v2 redesign, hardened_needs_pilot).**
> - Recalibration gate: the **allocation axis must be the scored artifact** (measured allocation margins 0.000 and 0–0.033 resisted one-shotting; prompt-axis first-shot variance spans 0.28–0.52 — one blind draw cracked the v1 prompt ladder to 0.2833).
> - Addendum A v2 supersedes the mechanics below: `initial_program.py` = the copy-line + parser-side-compute program (baseline ~0.537); mix rebalanced onto selection-hard kinds (val M=270); **n ∈ 1..5**; UNIT budgets 120/240 / 270/540 / 540/1080 (ratio 2.0) with token-tiered cost (1 unit if max_new_tokens ≤ 48, else 2).
> - Governance v1.1: timeout_s **3600** (this draft's 1500 would kill legitimate grades at queue depth 4); cpu_s 120; LM gens metered exactly at the unit budget.

# Task: record_qa_vote — budgeted question answering over a small LM

A small language model answers questions about warehouse order logs. Your
program never sees the log. You engineer three things: (a) the prompt
template wrapped around the log, (b) how many samples (1–3) to buy per
question out of a global sample budget, and (c) the parser that turns the
sampled texts into one final answer. Score = **error rate** on a hidden
validation split (lower is better). Correctness is an exact string match —
there is no judge model. The LM's sampling randomness is part of the task:
the same prompt can succeed on one draw and fail on the next.

## Grading LM

`LiquidAI/LFM2.5-230M`, fp32, single-threaded, served by an
evaluator-managed local HTTP server. Sampling parameters are **fixed by the
evaluator**: temperature 0.7, top_k 50, no repetition penalty. You cannot
change them. During development you may query the server directly on the
training split (roughly 0.5–2 s of wall time per sample, depending on how
many tokens your prompt makes the model generate).

## Instance family (fully published; only the concrete draws are sealed)

Each instance is a `(doc, question, gold)` triple.

- **doc**: `"Warehouse order log:"` followed by ~10 bulleted records of the
  form `"{Name} ordered {qty} {item} on {Month} {day}."`
- **gold**: a canonical digit string (no leading zeros).
- **question kinds** (validation mix of 150 instances):

| kind | n@150 | what varies |
|---|---|---|
| qty-digits | 27 | plain quantity lookup |
| qty-wordnum | 33 | doc quantities written as English words ("sixty-seven") |
| qty-distract | 23 | all 10 records are the same item; names differ |
| qty-coref | 12 | "The next day the same customer also ordered…" follow-on record |
| day | 22 | "on what day of the month…" |
| maxq | 18 | "largest number of X in a single order" |
| total | 15 | two orders by the same person; the answer is their sum |

The four `qty-*` kinds share one question surface (their difficulty lives in
the sealed doc); `day`, `maxq`, and `total` are recognizable from the
question text alone. Each kind has ~6 question phrasings; the training split
exposes only ~3 of them, and validation/test also use names and items that
never appear in training. The generator script is committed, but the
production seed is not: the sealed draws cannot be regenerated.

Splits:

- `data/train.jsonl` — 120 rows `{id, doc, question, gold, kind, style}`,
  fully visible. This is your experimentation bench.
- `data/heldout_val.bin` — 150 sealed instances (the visible grade).
- `data/heldout_test.bin` — 400 sealed instances (the official grade,
  scored once at `--final`).

## Required API (module-level functions in program.py)

```python
def build(question: str) -> dict:
    """Return {"system": str,   # <= 600 chars
               "pre": str, "post": str,   # len(pre)+len(post) <= 2000 chars
               "n": int,                  # samples to buy, 1..3
               "max_new_tokens": int}     # optional, 8..80, default 48
    """

def parse(texts: list[str], question: str) -> str:
    """texts: the n sampled completions. Return the final answer (digits)."""
```

The evaluator builds the single user message as `pre + doc + post` — the doc
is inserted exactly once, by the evaluator; you never receive it. An
instance is correct iff `str(parse(...)).strip() == gold`.

Failure handling:

- `parse` raising → that instance is wrong.
- `build` raising or returning an invalid shape → the DEFAULT template
  (`{"system": "", "pre": "", "post": "\n" + question}`) is used with n=1,
  and the sample is still spent — deliberate failure banks nothing.
- Both functions must be **deterministic**. The evaluator calls
  `random.seed(0)` before every call to them; any residual nondeterminism
  breaks `verify --rescore` and invalidates the run.

## Budget walk — exact scoring protocol

Budgets (samples / instances): train-only 180/120, val 225/150,
test 600/400 — a 1.5× ratio everywhere.

```
remaining = B
for i, inst in enumerate(instances):          # fixed file order
    later = M - 1 - i
    tpl, n_req = safe_build(inst.question)    # invalid/raise -> (DEFAULT, 1)
    n     = max(1, min(3, n_req))             # bool rejected; non-int -> 1
    n_eff = min(n, remaining - later)         # reserve 1 for every later instance
    remaining -= n_eff
    texts = [generate(tpl, inst.doc, seed(i, j)) for j in range(n_eff)]
    ans = safe_parse(texts, inst.question)
    wrong += not (isinstance(ans, str) and ans.strip() == inst.gold)
score = wrong / M
```

Invariant: before instance i, `remaining >= M - i`, so `1 <= n_eff <= 3`
always — every instance gets at least one sample, and the total spent never
exceeds B. Sample j of instance i uses a seed that does not depend on any
other instance's n, so a given sample is identical under any allocation.

## Seeds and what you see

- Each grading draws a fresh `eval_seed`; per-request seeds are
  `sha256(f"{eval_seed}|{split}|{instance_id}|{j}")`-derived, and the server
  seeds the sampler per request. Same eval_seed ⇒ bit-identical transcripts
  on the same machine; `bench verify --rescore` replays and checks this via
  a recorded transcript hash.
- Visible metrics per grade: `val_score` (your score), `samples_used`,
  `budget`, `n_hist`, `gen_tokens`, `transcript_sha256`. Per-kind validation
  accuracy is recorded but **hidden** from you.
- `--final` grades the 400-instance test split (budget 600) plus a
  forced-n=1 diagnostic pass; test numbers are sealed until the run ends.
  A full final pass takes roughly 12–25 minutes, once per run.

A visible grade costs real wall time: about 150 s for a terse n=1 program,
up to ~470 s for a full-budget, verbose-prompt program. The sample budget
and the wall clock are two separate costs — plan your experiments.

## Rules

- Python 3.12 stdlib only inside `build`/`parse`. **Forbidden** (checked):
  `socket`, `ssl`, `http`, `urllib`, `xmlrpc`, `ftplib`, `poplib`,
  `imaplib`, `smtplib`, `asyncio`, `selectors`, `subprocess`,
  `multiprocessing`, `threading`, `time`, plus the standard benchmark
  blocklist (`os`, `io`, `open`, `pathlib`, `mmap`, `ctypes`, `importlib`,
  `__import__`, …). Your program talks to the LM only through the
  evaluator's budget walk.
- Templates are sanitized: literal chat special-token strings are stripped
  from your fields before templating, and the character caps above are
  enforced. Oversized/invalid templates deterministically fall back to the
  DEFAULT template for that instance.
- Answers come from parsing sampled text (plus any computation you do in
  `parse` over what the model actually said). Hardcoding answers cannot
  work: gold is a function of the sealed doc, question strings never repeat
  across splits, and the sealed draws are un-regenerable.
- Whole evaluation must fit the task time limits (`timeout_s` 1500,
  candidate CPU 120 s).

## Baseline

`initial_program.py` uses the bare default prompt (doc + question, no system
message), takes the first integer in the single sampled completion, and buys
n=1 everywhere. It scores around **0.45–0.47** validation error and takes
about 150 s per grade.

## Self-test

Run `python3 evaluate.py program.py --train-only` to grade against the
visible training split (budget 180/120) with the same clamp and seed scheme;
it reports `train_score` and the same diagnostics. Training golds are
visible in `data/train.jsonl`, so treat the train score as a development
guide, not the score — the graded number is always the sealed validation
(and, at `--final`, test) split.
