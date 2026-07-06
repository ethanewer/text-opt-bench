> **PROVENANCE — drafted pre-hardening (2026-07-05). Superseded in part (design §2/§2.5; hardening report §1, §4.2).**
> - Calibration gate before pilot: the baseline→reference gap must hold **≥0.10 (≈3–4 SE) via the word-render lever**; the pilot campaign is go/no-go.
> - Probe: three-for-three blind strong first shots chose few-shot prompting and ALL measured worse than the naive baseline — which reproduced bit-exact at **0.4875**, not the ~0.55 stated below.
> - Seed contract is H2/H6 (fresh os.urandom seed per submission, repeats pooled as estimation), not the hash-nonce 'identical score on resubmit' below.
> - Governance v1.1: timeout_s **2100** (final 5400), cpu_s 120; scored on server `lm_seconds`, never wall; shared ThreadingHTTPServer, 25k soft / 250k hard sample meters.

# Task: record_qa — prompt-program optimization over a real local LM

A 230M-parameter on-device model (LiquidAI/LFM2.5-230M) answers numeric
questions about short operational records — depot intake logs, shift rosters,
bakery order slips, delivery manifests, stock memos. You cannot touch the model
or the sealed documents. You optimize the **prompt program** around them: how
the document and question are presented to the model, and how the model's
sampled output is parsed into a final answer.

## Required API (module-level functions in program.py)

```python
def build(question: str) -> dict:
    """Return a plain dict:
      {"system": str,                     # may be ""
       "examples": [(user, assistant), ...],  # few-shot pairs, str/str
       "user_template": str}              # MUST contain {document} exactly
                                          # once; may contain {question}
    All values must be plain str / plain list-or-tuple of 2-tuples of str.
    """

def parse(sampled_text: str, question: str) -> str | int | None:
    """Given the model's sampled text, return the answer as canonical
    digits, e.g. "24" (int accepted; it is stringified)."""
```

Per instance the **evaluator** — never your code — substitutes the sealed
document and question into your template (plain string replacement: `{question}`
first, `{document}` last), sends the rendered messages to the local LM server
with a derived seed, and passes the sampled text to your `parse`. Your code
never sees the sealed documents; the only path to the answer is through the
model's sampled output.

### Template limits (violations = that instance counts wrong)

- Total prompt (system + all example turns + rendered user turn) ≤ **4000 chars**.
- ≤ **6** few-shot example pairs.
- `{document}` exactly once in `user_template`.
- Exceptions raised by `build` or `parse` count that instance wrong. A
  load-time smoke render with a dummy document rejects structurally broken
  programs with an informative error.

### Answer canonicalization

Your `parse` output is stripped; a pure digit string has leading zeros removed
(`"024"` → `"24"`); an `int` is stringified. It must then **string-equal** the
gold answer (a non-negative decimal digit string). Anything else — `None`, a
non-digit string, an exception — is wrong.

## Decoding (fixed by the evaluator; not configurable)

`temperature 0.7, top_k 50, max_new_tokens 48`, fp32, single thread, seeded
per call. Generations are stochastic across instances but fully reproducible
for a given submission (see Scoring).

## Data

- **Visible:** `data/train.jsonl` — 120 instances of
  `{"id": ..., "doc": ..., "q": ..., "answer": ...}` drawn from the same
  distribution as the hidden splits. Use these to experiment.
- **Hidden:** a sealed 150-instance validation split (your score) and a larger
  sealed test split (final report only). Reading the heldout `.bin` files or
  the generator is forbidden and detected.
- Distribution notes: each record has several fact lines about multiple
  people/items/days, plus distractor lines (e.g. outgoing shipments, filler
  remarks) and occasional pronoun coreference ("they"). The question may use a
  different verb than the document ("received" vs "signed for"). Quantities
  may be rendered as digits, as English number words ("forty-nine"), or in
  dozen idioms ("a dozen", "two dozen"). Questions contain no digits or
  number words. No single answer value is frequent enough for a constant
  answer to score well.

## Scoring (lower is better)

Score = **error rate on the sealed 150-instance validation split**, one sampled
generation per instance, exact string match after canonicalization.

- Default `evaluate` grades the sealed validation split (this is your recorded
  score). `--train-only` grades the 120 visible train instances the same way
  (self-test; works without touching the hidden splits).
- Seeds are derived from a per-submission nonce tied to your program's hash:
  resubmitting byte-identical code returns the **identical** score; any source
  change redraws all samples.
- **Noise:** your visible score is a 150-sample estimate with SE ≈ 0.03. The
  official result re-grades your best program with several times more samples
  (validation at k=5 plus the sealed test split), so lucky draws wash out —
  chase real improvements, not fluctuations. Averaging repeated measurements
  of your own variants on train data is a legitimate way to see through noise.

## Cost

Each grade runs ~150 model generations through a single serialized local
server queue, roughly **1.5–3 minutes** of wall time; a train self-test is
similar. You may also query the server directly for your own experiments:
`POST http://127.0.0.1:8377/generate` with
`{"messages": [...], "seed": int, "temperature": float, "top_k": int,
"max_new_tokens": int (≤128), "tag": "explore"}` → `{"text": ...,
"prompt_tokens": ..., "completion_tokens": ...}`. Your own calls go through
the same serialized queue at the same per-sample cost and are logged and
metered. Budget your time: grades and experiments compete for the same
compute.

## Rules

- Python 3.12 **stdlib only** inside `program.py`. Forbidden (checked):
  `open`, `os`, `io`, `sys`, `pathlib`, `mmap`, `ctypes`, `socket`, `ssl`,
  `select`, `selectors`, `subprocess`, `multiprocessing`, `threading`,
  `tempfile`, `shutil`, `signal`, `sqlite3`, `dbm`, `shelve`, `importlib`,
  `__import__`, `urllib`, `http`, `email`, `xmlrpc`, `ftplib`, `asyncio`,
  `time`, `random`, `secrets`. Your candidate code makes **no** network
  calls — only the evaluator talks to the server during grading.
- `build`/`parse` must be deterministic pure functions of their arguments.
- Never read the heldout `.bin` files, the generator under `tools/`, or the
  server internals; never impersonate, restart, or reconfigure the server.
  Direct server queries for experimentation are allowed only via `/generate`.
- No benchmark gaming: the task is to make the model extract the right
  quantity and to parse what it says, not to attack the harness.

## Baseline

`initial_program.py` presents the bare document and question with no
instructions and parses the first integer in the sampled text. It scores
around **0.55** error on the validation split. There is substantial room
below it; expect to measure your way down — plausible-sounding prompt changes
frequently make things worse, and the only way to know is to test.
