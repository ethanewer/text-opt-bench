> **PROVENANCE — drafted pre-hardening (2026-07-05). Superseded in part (design §2; hardening report §2.2).**
> - Baseline rung expectation is **~0.73** (definitional clue table measured 0.7344 ± 0.039), not '~0.90 or worse' as stated below; control clue-table quality during curation (5/16 honest clues needed legal rewording under the taboo rule).
> - Curation gate before ship: `tools/curate_taboo_words.py` (300-word × 5-clue bank) must yield 20 achievable words (probe yield 9/60 at 1–4 attempts) — else ship W=16; the TODO word table below is unfilled pending that run.
> - Probe verdict: LOW risk decisively (blind first shot 0.844 landed behind the baseline; 12–16.5 SE headroom). Governance v1.1: cpu_s 60 / timeout_s 300; shared ThreadingHTTPServer, server-side 12k probe-pool meter.

# Task: taboo_cluesmith — make a small LM guess the word

You are given a fixed list of 20 English target words. For each target, write
a **clue** — a short lowercase phrase that must not contain the target word or
anything close to it. The grader wraps your clue in a fixed guessing frame and
samples a small local language model (LiquidAI/LFM2.5-230M) at temperature
0.7; a sample **hits** iff the target word appears as a whole word in the
sampled text. Your score is the miss rate over all words and samples (lower is
better).

The grading distribution is fully disclosed in this spec: the frame, sampling
parameters, word list, and hit rule are all below. The only randomness is real
token sampling from the model. Note that making a word's guess deterministic
is a legitimate win, not a cheat: the score is a known expectation over the
sampling distribution, and collapsing its variance toward a certain hit is
exactly the objective.

## Target words

The 20 targets and their per-word extra banned forms are shipped in
`data/words.json`:

<!-- TODO(authoring): paste the final curated word table here verbatim from
data/words.json before shipping. -->

`clue(word)` is called once for each of these words, exactly as spelled there
(all lowercase).

## Required API (module-level function in program.py)

```python
def clue(word: str) -> str:
    """word: one of the 20 targets listed above.
    Return the clue text for that word."""
```

The evaluator calls `clue(w)` once per target, validates **every** clue before
any LM call, then does all sampling and scoring itself. Your code never
touches the model, the transcripts, or the hit check.

## Clue legality (checked before any sampling)

An invalid clue rejects the whole submission with an error naming the word and
the violated rule. This check is deterministic and costs zero samples, so
legality is cheap to iterate on.

1. Must be a `str` of 1–25 whitespace-separated tokens, at most 160
   characters, using only characters `[a-z ]` — lowercase letters and single
   spaces. No digits, punctuation, uppercase, or newlines.
2. Every token must appear in the shipped lexicon `data/lexicon.txt` (~20k
   common English words plus a fixed function-word whitelist; the only
   one-letter entries are `a` and `i`).
3. Taboo rule: for every clue token `t` and every banned form `b` of the
   target — `{target, target+"s", target+"es", target+"ed", target+"ing",
   target+"er"}` plus the word's `extra_banned` list from `data/words.json` —
   the clue is rejected if `b` is a substring of `t`, or `t` is a substring of
   `b`, or `levenshtein(t, b) <= 1` when `len(target) <= 4` else `<= 2`.

Worked rejections (target `banana`):
- `"a bandana without the d"` — rejected: `bandana` contains no banned form as
  a substring but is within edit distance 2 of `banana`.
- `"yellow fruit, curved"` — rejected: contains punctuation (`,`).
- `"platano fruit"` — rejected: `platano` is not in the lexicon.

## Grading protocol

- **Frame** (evaluator-owned, fixed): a single user message consisting of your
  clue followed by

  ```
  \n\nAnswer with the single word being described.
  ```

  rendered with the model's chat template (`add_generation_prompt=True`, no
  system message).
- **Sampling params** (fixed): `temperature=0.7`, `top_k=50`,
  `repetition_penalty=1.05`, `max_new_tokens=16`, fp32, single thread.
- **Hit rule** (exact string check, no judge): lowercase the sampled text, map
  every non-`[a-z]` character to a space, split on whitespace; the sample is a
  hit iff `target`, `target+"s"`, or `target+"es"` is among the resulting
  tokens. Example: target `camera`, sample `The answer is **CAMERA**.` → hit.
- **Score**: `round(1 - hits / (20 * k), 4)`, direction: min.

### Visible vs official grade

- Every submission is graded with **k = 8** samples per word (160 generations,
  roughly 25–30 s). This is the score you see during the session.
- At session end, `bench official <run_dir>` re-grades your best program with
  **k = 32** fresh samples per word (640 generations) and writes
  `official.json`. The official number is the reported result.

### Seeds, recording, rescore

Each grade draws a fresh `master_seed` (recorded in the submission record);
per-generation seeds are derived as
`sha256(f"{master_seed}:{word_index}:{sample_index}")[:4]`. Generation is
bit-reproducible per seed on this machine. Each submission records the seed,
per-word hit counts, a `transcript_sha256` over all sampled texts, and the
server fingerprint; `bench verify --rescore` re-runs each submission with its
recorded seed and requires the score and transcript hash to reproduce exactly.

## Experiment budget (sample pool)

You may query the model yourself for your own experiments (see the server API
below). Agent-side generation is metered against a per-session pool of
**12,000 samples**, enforced by the server. Official/visible grading is billed
to a separate grading meter, so grading never fails because your experiments
ran hot. When the pool is exhausted the server returns HTTP 429: your
experiments stop, but submissions remain gradable.

Useful arithmetic: estimating one word's hit probability to about ±0.05
(worst case) costs ~100 samples; a 20-word sweep at that precision is ~2,000
samples, one sixth of the pool. Each generation takes roughly 0.1–0.2 s.

## Local model server

A local server (already running; not part of your program) exposes:

- `GET /info` → `{model_id, dtype, torch, transformers, threads,
  template_sha256, pool_remaining}`.
- `POST /generate` with JSON `{messages, max_new_tokens (<=64), temperature
  (0.05–1.5), top_k (<=100), repetition_penalty, seed, n: 1}` →
  `{text, n_new_tokens, pool_remaining}`. Total prompt length ≤ 4,000 chars.
  Same seed ⇒ identical output.

Connection details come from the environment (`TEXTOPT_LM_URL`). Your
**submitted program** must not do any network or file I/O — it only returns 20
strings; the evaluator makes all grading calls itself.

## Rules

- The submitted program is pure Python, no imports needed and network/OS
  modules are forbidden (checked): `socket`, `urllib`, `http`, `ssl`,
  `asyncio`, `selectors`, `os`, `sys`, `subprocess`, and the standard escape
  blocklist.
- `clue` must be deterministic.
- Cooperative rules: no phonetic/spelling side-channels (e.g. "rhymes with
  ...", letter-by-letter spelling clues) — winning clue tables are audited and
  such clues fail review; do not touch or tamper with the model server, its
  state directory, or the grader token; use the LM only via `POST /generate`.
- Making the model's guess deterministic for a word is allowed and desirable.

## Baseline

`initial_program.py` returns a plain definitional clue for each word (a short
attribute-style description). It is fully legal and scores roughly **0.90 or
worse**: the small model usually echoes attributes from the clue instead of
naming the referent, so most samples miss.

## Self-test

`tests/broken/` contains programs that must be rejected before any LM call: a
clue containing the target; a plural of the target; an edit-distance-1
near-form; uppercase/punctuation in a clue; a token not in the lexicon; a
26-token clue; `clue` returning a non-string; a missing `clue` function; and a
program importing `urllib`. `bench determinism` grades the task twice with the
same fixed seed and requires bit-exact equality of the score and transcript
hash.
