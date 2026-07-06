# Real-LM track (LFM2.5-230M) — refined specs and red-team reports

Generated 2026-07-05 by the `real-lm-random-tasks` design workflow (31 agents: 4 ideation
lenses -> 22 candidates -> 3-judge panel -> 6 refined specs, each attacked by exploit /
statistics / harness red-teams). All numbers were MEASURED on the owner's Apple M5 with
LFM2.5-230M fp32, torch.set_num_threads(1) (torch 2.12.1 / transformers 5.13.0 venv).
Selected into the final family: **taboo_cluesmith** (type 1), **record_qa** and
**record_qa_vote** (type 2). See docs/random_tasks_design.md for the cross-track selection
rationale and the shared harness/serving infrastructure these specs assume.


---

# taboo_cluesmith — SELECTED (judge mean 7.10, 3/3 advances, needs-real-LM 10.0/10)

# taboo_cluesmith — design spec (type 1 "random", real-LM grading)

## 1. One-paragraph story (agent-facing)

You are given a fixed list of W=20 English target words. For each target, write a
**clue** — a short lowercase phrase that must not contain the target or anything
close to it. The grader wraps your clue in a fixed guessing frame and samples
LiquidAI/LFM2.5-230M at temperature 0.7; a sample **hits** iff the target word
appears as a whole token in the sampled text. Score = miss rate over all words
and samples (lower is better). You know the full grading distribution (frame,
sampling params, word list, hit rule are all in this spec); the only randomness
is real token sampling. The 230M model is genuinely bad at naming referents —
it echoes clue attributes ("yellow", "hot", "clear") instead of naming the thing
— so the optimization is a per-word semantic search for the framing that makes
this particular model NAME the referent. Making a word's guess deterministic is
a legitimate win, not a cheat: the score is a known expectation and collapsing
its variance is the objective.

## 2. Required API (candidate program, pure Python, no imports needed)

```python
def clue(word: str) -> str:
    """word: one of the 20 targets listed in spec.md.
    Return the clue text for that word."""
```

The evaluator calls `clue(w)` once per target, validates every clue BEFORE any
LM call, and then does all sampling and scoring itself. Candidate code never
touches the network, the transcripts, or the hit check. Any invalid clue
rejects the whole submission with an error naming the word and the violated
rule (deterministic, costs zero samples — cheap to iterate on legality).

### Clue legality (all checked by the evaluator, exact rules in spec.md)
1. `str`, 1–25 whitespace-separated tokens, ≤160 chars, charset `[a-z ]` only
   (single spaces; no digits, punctuation, or uppercase).
2. Every token is in the shipped lexicon `data/lexicon.txt` (~20k common
   English words, incl. a fixed function-word whitelist; the only 1-letter
   entries are "a" and "i"). This kills invented encodings ("canana"),
   single letters, and foreign-word smuggling ("banane", "platano").
3. Taboo rule: for every clue token `t` and every banned form `b` of the
   target — `{target, target+"s", +"es", +"ed", +"ing", +"er"}` plus per-word
   extra bans listed in the word table (irregular plurals, tight derivatives
   like "clockwise") — reject if `b in t` or `t in b` (substring) or
   `levenshtein(t, b) <= (1 if len(target) <= 4 else 2)`.
   (Verified: this bans "bandana" and "cabana" for banana, "moo" survives for
   a hypothetical 3-letter target because short targets use distance 1.)

## 3. Grading distribution (fully disclosed to the agent)

- Frame (evaluator-owned, fixed): single user message
  `clue + "\n\nAnswer with the single word being described."`,
  chat template with `add_generation_prompt=True`, no system message.
  (Probed alternatives: few-shot frames LEAK the exemplar answers into guesses
  and score WORSE — 0.083 vs 0.089 mean hit; forced answer-prefix 'The word
  is "' does not fix attribute echo. Plain frame is best and simplest.)
- Sampling: `temperature 0.7, top_k 50, repetition_penalty 1.05,
  max_new_tokens 16`, fp32, `torch.set_num_threads(1)`.
  Why 0.7: measured behavior is "greedy-hit ⇒ sampled p≈0.5–1.0; greedy-miss ⇒
  p≈0–0.15", so 0.7 keeps genuine per-sample stochasticity (mid-range p for
  decent clues: bridge 5–9/12, piano 5–6/12, sleep 4–8/12) while a well-aimed
  clue still reaches p≈0.9–1.0 (camera 11–12/12). Colder grading (0.3) would
  polarize p to {0,1} and kill the "random" character; hotter (1.0) drowns
  real improvements in noise.
- Hit rule (exact, no judge): lowercase the sample, map every non-`[a-z]` char
  to space, split; hit iff `target`, `target+"s"`, or `target+"es"` is a
  token. ("The answer is **CAMERA**." → hit.)
- Score: `round(1 - hits/(20*k), 4)`, `direction: min`.

## 4. Visible vs official grade, seeds, rescore

- **Visible grade (every submission):** k=8 samples/word = 160 generations,
  measured ≈25–28 s wall. Empirical noise: SD of a full grade measured at
  k=12 over 3 replications = 0.026 (theory 0.030); scaled to k=8 ⇒ SD≈0.033.
  Per-word quantum is 1/160 = 0.00625; fixing one word from p≈0.1 to p≈0.7
  moves the score by 0.03 ≈ 1 SD, fixing two by ≈2 SD — visible grades detect
  real multi-word progress, single-word progress needs the agent's own MC
  (that asymmetry is the intended sample-allocation game).
- **Official grade:** at session end, `bench official <run_dir>` re-grades
  `best_program.py` with k=32 fresh seeds/word (640 generations, ≈100 s wall,
  SD≈0.017) and appends `official.json` (score, per-word hits, master seed,
  transcript hashes). The official number is the reported result; this
  resolves the winner's curse from strict-< best-tracking over ~40 noisy
  visible grades (selection bias ~1–1.5 visible SD ≈ 0.03–0.05).
- **Seed handling:** `session.submit` draws `master_seed` from `os.urandom`,
  records it in the submission record (`eval_seed`), and passes it to the
  evaluator as `--seed`. Per-call seeds are
  `int.from_bytes(sha256(f"{master_seed}:{word_index}:{sample_index}").digest()[:4], "big")`
  — one generation per seed, `n=1` requests only.
- **What is recorded per submission:** master seed; `samples_used` (160);
  per-word hit counts; `transcript_sha256` = sha256 of all 160 sampled texts
  concatenated in (word, sample) order; server fingerprint
  `{model_id, dtype, torch, transformers, threads}` from `GET /info`.
- **Rescore:** `bench verify --rescore` re-runs each submission with its
  recorded `--seed` and a `billable=false` flag (grader token); score,
  per-word hits, and `transcript_sha256` must reproduce exactly.
  Bit-reproducibility per seed verified empirically (same seed → identical
  text; different seed → different text). Same-machine-only stability is
  accepted, as for memory tasks; verify prints a clear error if the live
  server fingerprint differs from the recorded one. `bench determinism` runs
  stochastic tasks twice with the SAME fixed seed and requires bit-exact
  equality (`score_tolerance: 0`).

## 5. Cost mechanics (measured arithmetic)

Per sample ≈0.16 s (mean 3.3 new tokens; the model EOSes after a 1–3-word
answer; ~27 tok/s decode fp32 1-thread incl. prefill/template overhead —
NOTE: much cheaper than the shortlist's assumed 0.5 s, so wall time alone
cannot stop averaging; the metered pool is the primary cost mechanism).

- The server meters every agent `/generate` call against a per-session pool
  **P = 12,000 samples** (server-enforced; the server is outside agent
  control). Visible grades are billed to a separate grading meter (fixed 160
  per submission; grading never dies because experiments ran hot).
  Pool exhausted ⇒ HTTP 429 ⇒ experiments stop, submissions still gradable.
- Marginal-value arithmetic the agent faces: pinning one word's p to ±0.05
  needs ~100 samples (p(1−p)/SE², worst case), a 20-word sweep at ±0.05 =
  2,000 samples = 1/6 of the pool (≈5 min wall); at ±0.08, 40/word = 800 per
  sweep. Choosing between ~15 precise word-experiments and broad sweeps vs
  trusting noisy visible grades is a genuine trade-off, not free averaging.
- Wall per session: ~40 visible grades ≈ 17 min of grading inside the 1-hour
  box; pool fully spent ≈ another 32 min — the pool and the box are the same
  order, so both bind.

## 6. Server integration

`tools/lm_server.py` (experimenter-managed, torch+transformers venv, NOT part
of the task): loads LFM2.5-230M fp32, `torch.set_num_threads(1)`, warms up,
serializes all generation under one lock (determinism + resource cap; ~1 GB
RAM, 1 core — safe next to concurrent CPU evals). Writes
`tools/.lm_server/state.json` `{port, grader_token, fingerprint}`.

- `GET /info` → `{model_id, dtype, torch, transformers, threads,
  template_sha256, pool_remaining}`.
- `POST /generate` `{messages, max_new_tokens<=64, temperature in [0.05,1.5],
  top_k<=100, repetition_penalty, seed, n=1}` → `{text, n_new_tokens,
  pool_remaining}`. Total prompt ≤4,000 chars (blocks context-stuffing that
  would starve other runs). Each call: acquire lock → `torch.manual_seed(seed)`
  → generate → decode. Metering: no `X-Grader-Token` header ⇒ billed to the
  agent pool; with token ⇒ grading meter, and `billable=false` allowed
  (rescore).
- `POST /reset_pool` (token-only): campaign runner resets the pool at session
  start and records the fingerprint in run metadata.

`evaluate.py` stays pure stdlib: `urllib.request` to
`http://127.0.0.1:<port>` using `TEXTOPT_LM_URL` + `TEXTOPT_LM_TOKEN` from its
env (injected by the harness from the state file — env is fixed-allowlist, so
this is a runner change, not an agent-visible channel). The agent may query
`/generate` directly for its own Monte Carlo (documented in spec.md) — same
intrinsic cost, same pool.

## 7. Anti-exploit analysis

- **No emit-answer path**: the scored text is always an LM sample; the hit
  check, frame, and transcripts are evaluator-owned. Nothing to memorize —
  the distribution is fully public by design (this is the type-1 premise).
- **Target leaks through the clue**: lexicon membership kills non-words and
  foreign translations; substring + length-scaled edit distance (1 for short,
  2 for len≥5) kills near-forms ("bandana", "cabana", "bananas"); charset
  kills spacing/punctuation tricks; "a"/"i" are the only 1-letter tokens.
  Residual channels (rhyme clues "rhymes with montana", compositional
  spelling via word-initials) are (a) empirically DEAD at 230M — probed
  riddle/phonetic framings scored 0/6; the model cannot decode them — and
  (b) covered by the cooperative rule plus a mandatory audit spot-check of
  the winning clue table (`bench audit` gains a taboo-task rule: flag clues
  containing "rhymes"/"sounds like"/"letter" patterns for human review).
- **Frame injection**: clue precedes a fixed suffix inside one user message;
  charset `[a-z ]` forbids newlines/markup. Steering the model IS the task;
  there is no judge to inject into and no parse channel back into scoring.
- **Candidate-side interception**: candidate only returns 20 strings; network
  modules are in FORBIDDEN (plus baseline escape blocklist); LM calls happen
  with `set_candidate_active(False)` and no candidate code on the stack.
- **Degenerate collapse to constant output**: legitimate win by definition
  (stated in spec.md to preempt confusion).
- **Averaging away the noise**: bounded by the metered pool (server-side,
  outside agent control) — see §5.
- **Server tampering / restart with different params**: fingerprint + template
  hash recorded per submission; verify flags drift; cooperative rule forbids
  touching `tools/.lm_server` (grader token file is the same class of honesty
  boundary as `bench/heldout.py`).

## 8. Word list and the curation gate (hard requirement)

Empirical finding: LFM2.5-230M systematically echoes clue attributes; only
words with an overwhelming lexical association are reachable. Measured across
~60 probed words × 1–4 clue framings each (naive/definitional/tuned/cloze/
association/riddle/name-the/forced-prefix): verified-achievable at t0.7–0.8 —
**camera (~0.9), computer (~0.83), bridge (~0.6–0.75), cold (~0.8–1.0),
piano (~0.5), london (~0.5), sleep (~0.5), paris (~0.5)**; borderline: clock
(0.25–0.4), library (0.1–0.35), spider (~0.2), candle (~0.2); stuck at ≤0.15
despite multiple rescue clues: banana, dog, gold, rain, winter, night, milk,
egg, umbrella, hammer, guitar, window, scissors, pillow, island, and ~30 more.

Therefore authoring MUST run `tools/curate_taboo_words.py`: scan a bank of
~300 candidate targets × 4–6 clue styles at n=8 (≈12k samples ≈ 35 min), then
confirm survivors at n=16; the shipped word table contains only targets with a
**verified clue at p̂ ≥ 0.5** (12–14 easy/medium slots p̂≥0.6, 4–6 medium
0.4–0.6, and at most 2–3 verified-hard 0.25–0.5 as the tail). The verifying
clue table is the hidden reference solution; per-word `extra_banned` lists are
filled during curation. If the bank yields fewer than 20 achievable targets,
ship W=16 rather than padding with duds (score already normalizes by W·k).

## 9. Baseline, strong solution, expected scores

- `initial_program.py`: honest definitional clue table (attribute-style).
  Measured mean hit 0.089 at this frame ⇒ **score ≈ 0.90–0.95**.
- Mid-effort (uncurated mixed table, measured end-to-end at k=12):
  **0.67–0.72**.
- Strong reference (`tests/solutions/taboo_cluesmith.py`): the curation-pass
  clue table. On a curated list where every word has a verified p≥0.5 clue,
  expected official **≈0.30–0.45** (mean verified p ≈ 0.55–0.7). Headroom
  baseline→strong ≈ 0.5 ≈ 15 visible-grade SDs, climbable word-by-word over
  many iterations (per-word clue search with own-MC confirmation).
- `tests/broken/`: clue containing target; "bananas"; edit-distance-1 form;
  uppercase/punctuation; non-lexicon token; 26 tokens; returns non-str;
  missing `clue`; program importing `urllib`. All must be rejected pre-LM.

## 10. config.json sketch

```json
{
  "name": "taboo_cluesmith",
  "kind": "random",
  "stochastic": true,
  "metric": "miss rate over 20 words x k LM samples",
  "direction": "min",
  "timeout_s": 300,
  "cpu_s": 60,
  "k_visible": 8,
  "k_official": 32,
  "gen": {"temperature": 0.7, "top_k": 50, "repetition_penalty": 1.05,
           "max_new_tokens": 16},
  "frame_suffix": "\n\nAnswer with the single word being described.",
  "score_tolerance": 0
}
```
(`cpu_s` stays small: the evaluator's own CPU is parsing/hashing; the model
burns CPU in the server process, outside the rlimit — by design.)

## 11. Harness changes (all minimal, listed)

1. `runner.evaluate(..., seed=None, official=False)`: when config has
   `stochastic: true`, require `seed`, append `--seed <int>` (and `--official`
   for official grades); inject `TEXTOPT_LM_URL`/`TEXTOPT_LM_TOKEN` into the
   child env from `tools/.lm_server/state.json`.
2. `session.submit`: draw and record `eval_seed` for stochastic tasks.
3. `verify_run(rescore=True)`: pass recorded `eval_seed`; require exact score
   + `transcript_sha256` match; check server fingerprint first.
4. New CLI verb `bench official <run_dir>`: official re-grade of
   `best_program.py`, fresh recorded seed, writes `official.json`.
5. `bench determinism`: stochastic tasks run twice with the same fixed seed.
6. `tools/run_campaign.py`: start/health-check `tools/lm_server.py`, reset
   pool per session, store fingerprint in run meta.
7. `bench audit`: add clue-table review rule (§7).

## 12. spec.md skeleton (agent-facing)

Title + story (§1); required API + full word list; exact legality rules
(charset/tokens/lexicon/taboo-distance table) with 3 worked examples of
rejected clues; exact frame text and sampling params; hit rule with worked
example; score formula, k_visible, official-grade protocol; seed/transcript
recording (so the agent knows rescore exists); the sample-pool contract
(12,000 pool, 160/grade, 429 behavior) with the ±0.05↔100-samples arithmetic;
"collapse is a win" clarification; cooperative rules (no phonetic/spelling
side-channels, no server tampering, LM access only via /generate); the server
endpoint doc for self-experiments.

## Evaluator sketch

```python
# bench/tasks/taboo_cluesmith/evaluate.py  (pure stdlib)
import hashlib, json, os, sys, urllib.request
from pathlib import Path
sys.path.insert(0, __file__.rsplit("/bench/", 1)[0])
from bench import eval_lib

CFG = json.loads((Path(__file__).parent / "config.json").read_text())
WORDS = json.loads((Path(__file__).parent / "data" / "words.json").read_text())
# [{"word": "camera", "extra_banned": []}, ...]  20 curated entries
LEXICON = frozenset((Path(__file__).parent / "data" / "lexicon.txt").read_text().split())
FORBIDDEN = frozenset({"socket","urllib","http","ssl","asyncio","selectors",
                       "os","sys","subprocess", ...})  # + baseline

def levenshtein(a, b):  # small DP, capped at 3
    ...

def banned_forms(w, extra):
    return {w, w+"s", w+"es", w+"ed", w+"ing", w+"er", *extra}

def validate(word, extra, clue):
    if not isinstance(clue, str): eval_lib.fail(f"{word}: clue must be str")
    if not all(c in "abcdefghijklmnopqrstuvwxyz " for c in clue): eval_lib.fail(...)
    toks = clue.split()
    if not 1 <= len(toks) <= 25 or len(clue) > 160: eval_lib.fail(...)
    maxd = 1 if len(word) <= 4 else 2
    for t in toks:
        if t not in LEXICON: eval_lib.fail(f"{word}: {t!r} not in lexicon")
        for b in banned_forms(word, extra):
            if b in t or t in b or levenshtein(t, b) <= maxd:
                eval_lib.fail(f"{word}: clue token {t!r} too close to target")

def lm_generate(messages, seed):
    req = urllib.request.Request(
        os.environ["TEXTOPT_LM_URL"] + "/generate",
        data=json.dumps({"messages": messages, "seed": seed, "n": 1,
                         **CFG["gen"]}).encode(),
        headers={"X-Grader-Token": os.environ["TEXTOPT_LM_TOKEN"],
                 "Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=60).read())["text"]

def hit(target, text):
    toks = "".join(c if c.isalpha() else " " for c in text.lower()).split()
    return target in toks or target + "s" in toks or target + "es" in toks

def main():
    prog = sys.argv[1]
    seed = int(sys.argv[sys.argv.index("--seed") + 1])   # required
    k = CFG["k_official"] if "--official" in sys.argv else CFG["k_visible"]
    mod = eval_lib.load_program(prog, FORBIDDEN, required=("clue",))
    eval_lib.set_candidate_active(True)
    try:
        clues = {e["word"]: mod.clue(e["word"]) for e in WORDS}
    finally:
        eval_lib.set_candidate_active(False)
    for e in WORDS:
        validate(e["word"], e["extra_banned"], clues[e["word"]])
    # fingerprint check (server identity for rescore stability)
    info = json.loads(urllib.request.urlopen(
        os.environ["TEXTOPT_LM_URL"] + "/info", timeout=10).read())
    h = hashlib.sha256(); hits_per_word = []; total = 0
    for wi, e in enumerate(WORDS):
        msgs = [{"role": "user",
                 "content": clues[e["word"]] + CFG["frame_suffix"]}]
        hw = 0
        for i in range(k):
            s = int.from_bytes(hashlib.sha256(
                f"{seed}:{wi}:{i}".encode()).digest()[:4], "big")
            text = lm_generate(msgs, s)
            h.update(text.encode())
            hw += hit(e["word"], text)
        hits_per_word.append(hw); total += hw
    n = len(WORDS) * k
    eval_lib.succeed(round(1 - total / n, 4), metrics={
        "hits": total, "samples_used": n, "k": k,
        "per_word_hits": hits_per_word,
        "transcript_sha256": h.hexdigest(),
        "model_fingerprint": {kk: info[kk] for kk in
            ("model_id", "dtype", "torch", "transformers", "template_sha256")},
    })

if __name__ == "__main__":
    main()
```
Server (`tools/lm_server.py`): http.server or uvicorn in the torch venv; one
global lock; per request: seed → `torch.manual_seed` → `model.generate(fp32,
threads=1)` → decode new tokens; meters agent pool (12,000) vs grader meter by
`X-Grader-Token`; `/info`, `/generate`, `/reset_pool`.

## Measured numbers

MEASURED (fp32, torch threads=1, this machine, temp 0.7-0.8 / top_k 50 / rep 1.05): per-sample 0.12-0.18 s (mean ~0.16 s; mean 3.3 new tokens, model EOSes after 1-3-word answers; 0.093 s at short prompts, 0.45 s with few-shot context) — NOT the 0.5 s the shortlist assumed, so the metered pool (12,000 samples/session) is the binding cost, arithmetic: ±0.05 pin of one word = ~100 samples, full 20-word sweep = 2,000 = 1/6 pool ≈ 5 min. Visible grade k=8 → 160 gens ≈ 25-28 s (measured 38 s at k=12); official k=32 → 640 gens ≈ 100 s. Noise: empirical full-grade SD at k=12 = 0.026 over 3 reps (scores 0.6708/0.6708/0.7167; theory 0.030); k=8 visible SD ≈ 0.033, official k=32 SD ≈ 0.017; per-word fix (p 0.1→0.7) moves score 0.03 ≈ 1 visible SD. Seeds bit-reproducible (same seed → identical text). Baselines: naive generic clue 0/48 hits (score 1.0); definitional table mean hit 0.089 (score ≈ 0.91); mid-effort mixed table 0.67-0.72 measured end-to-end; verified-achievable words (p̂ ≥ 0.5): camera ~0.9, computer 0.83, bridge 0.6-0.75, cold 0.8-1.0, piano ~0.5-0.7, london ~0.5, sleep ~0.5, paris ~0.5 → curated-list strong ≈ 0.30-0.45 official, headroom ≈ 0.5 ≈ 15 visible SDs. Curation yield evidence: 9 of ~60 probed words achievable with only 1-4 clue attempts each; few-shot frame WORSE (0.083); riddle/cloze/association/forced-prefix framings all ≤ definitional on stuck words; greedy-hit ⇒ sampled p 0.5-1.0, greedy-miss ⇒ p ≤ 0.15.

## Open questions (spec author)

1) Curation yield: can tools/curate_taboo_words.py find 20 targets with verified p>=0.5 clues? My probe yield was 9/~60 words at 1-4 clue attempts each; a 300-word x 5-clue bank (~35 min of authoring compute) should reach 20, but if not, ship W=16 — decide after the curation run. 2) The ~0.5-miss ceiling risk (judge's main doubt) is confirmed real for UNCURATED words; with the curation gate the strong score is 0.30-0.45, but whether codex finds the achievable framings inside 1 h needs the mandatory pilot campaign (TASK_AUTHORING checklist step 5) — watch for a grindy plateau and consider dropping the 2-3 hard-tail words if trials stall. 3) k_visible=8 vs 12: pick after observing codex submission cadence in the pilot (8 = 25 s/grade, SD 0.033; 12 = 38 s, SD 0.026). 4) Pool size 12,000 and whether grading needs its own hard cap: calibrate from pilot sample-usage telemetry. 5) Lexicon provenance (wordfreq top-20k vs google-ngram list) and license; must contain all reference-clue tokens. 6) Concurrency: one server pool per run dir vs global per machine when campaigns run several sessions at once (proposal: pool keyed by run id passed in a header from the harness; agent calls without a run id bill the active run). 7) Cross-machine rescore is explicitly NOT guaranteed (torch version pinned in fingerprint; accepted per-system stability like memory tasks) — owner sign-off on documenting that for a scored (not just memory-metric) task.

## Red-team reports

### Lens: exploit — verdict: FIXABLE

- **[MAJOR] Metered pool is bypassable: fully-disclosed public model = free unlimited offline Monte Carlo**
  - The owner's hard requirement #2 is that grading must have a REAL cost so the agent cannot freely average away variance. The design enforces this only via a server-side per-session pool P=12,000 that is 'outside agent control.' But the grading distribution is FULLY public by the type-1 premise: model id (LiquidAI/LFM2.5-230M, a freely downloadable HF model), exact frame text, and sampling params (temp 0.7/top_k 50/rep 1.05/16 tok) are all in spec.md. I ran that exact configuration in the provided /tmp/lfm_bench venv with no metering at all. An optimizing agent with any ability to execute torch (the campaign machine already has the venv + cached model; failing that, `pip install transformers` + a public download) simply spins up its own LFM2.5 instance and draws UNLIMITED, UNMETERED samples of the identical distribution to estimate per-clue hit probability to arbitrary precision. Cross-machine bit-reproducibility is NOT needed for this — the agent only needs samples from the same expectation, which it has. The 12,000 pool then constrains nothing, requirement #2 is unmet, and the 'random' task degenerates to deterministic offline optimization of a known smooth expectation (expected miss rate per clue), with the only residual noise being the single official k=32 readout. This is the central structural risk.
  - *Fix:* The cost mechanism only binds if the agent's execution environment provably cannot run the model or an equivalent (no torch/transformers, no model weights, no internet, no access to the serving venv), with the metered server as the SOLE sampling path. State and enforce that as a task precondition; if it cannot be guaranteed, document that the pool does not bind and reclassify the task's stochasticity honestly (it becomes deterministic-objective optimization with a noisy final readout, not a genuine 'grading has real cost' random task).
- **[MAJOR] Enumeration/list-eliciting clues game the hit-anywhere-in-16-tokens rule and invalidate curation labels**
  - The hit rule counts a hit iff the target appears as ANY whole token in the 16-token sample. This lets a clue elicit a short list/enumeration in which the target is merely one member, rather than the model actually naming the referent. Measured against words spec.md labels 'stuck at <=0.15': clue 'write a long list of every fruit a monkey likes to peel and eat' gave banana 4/8-4/12 (sample: 'apple banana honey orange mango peach'); an egg list-clue gave egg 3-4/8. So (a) the curation difficulty labels and the derived strong-score estimate (0.30-0.45) and headroom (~0.5) are unreliable — 'unreachable' words are partially reachable via a generic non-semantic template; and (b) a template family (make the model emit a short list of category members) partially substitutes for the per-word semantic clue-craft the task claims to require, weakening the 'needs real language competence to NAME the referent' premise.
  - *Fix:* Require the target to be the model's PRIMARY answer, not merely present: hit iff the FIRST alpha token of the sample equals target/target+s/+es (matches the 'single word being described' frame). Empirically this kills list-stuffing (banana enumeration 4/12 -> 1/12) while preserving honest naming (camera 11/12 unchanged, bridge 5->3). Re-run curation under the stricter rule; difficulty labels and headroom must be recomputed.
- **[MINOR]** Even at face value the 12,000 pool lets the agent nearly fully de-noise its search — *Fix:* Independent of the offline-bypass fix, shrink the pool substantially (e.g. 2,000-3,000) and/or raise per-sample cost (longer max_new_tokens) so a single +/-0.05 20-word sweep already consumes most of the budget, forcing real sample-allocation choices rather than near-complete de-noising.
- **[MINOR]** Nondeterministic clue() breaks rescore because emitted clue strings are not recorded — *Fix:* Record the exact 20 emitted clue strings in the submission record and rescore against those (or double-call clue() during grading and reject if outputs differ). Also add random/time-nondeterminism-relevant modules to the task ban or require deterministic clue().
- **[MINOR]** Reported official score carries irreducible ~0.017 SE and best-program selection is noise-biased — *Fix:* Average several official draws (e.g. k=64-128 or multiple k=32 seeds) to shrink the reported SE, and document the residual noise band so downstream comparisons are not over-interpreted.

*Notes:* Empirically grounded in the provided venv (fp32, threads=1, temp 0.7/top_k 50/rep 1.05). The design is on the correct side of the emit-answer memorization boundary: type-1 with no sealed data, candidate returns only 20 pre-validated strings, and the evaluator owns all LM sampling and the hit check, so clue effects cannot be memorized — you must genuinely find clues that steer a real model. No frame-injection cross-contamination (each clue graded only against its own word) and seeds are unpredictable to the candidate. The two load-bearing weaknesses are both about the STOCHASTIC-COST premise and the HIT RULE, not memorization: (1) the metered pool does not bind if the agent can run the public model with the disclosed params offline (verified runnable), collapsing the 'random' grading to deterministic offline optimization — this is the owner's requirement #2 and is the most important thing to resolve; (2) the hit-anywhere-in-16-tokens rule is gamed by list-eliciting clues (banana 4/12, egg 3-4/8 vs spec's <=0.15 claim), which both invalidates the curation difficulty labels/headroom and weakens the 'name the referent' premise — a first-token hit rule fixes it cleanly (verified: banana enumeration 4/12->1/12, camera honest 11/12 preserved). All findings have concrete fixes, hence 'fixable'.

### Lens: statistics — verdict: FIXABLE

- **[MAJOR] per_word_hits in visible metrics + unmetered grading meter = free per-word experiment channel that out-delivers the metered pool**
  - session.py (bench/session.py:69-71) exposes all metrics not in the feedback-mode HIDDEN_KEYS, so the spec's per_word_hits lands in the agent-visible submission record. Visible grades are billed to a separate, uncapped grading meter ('grading never dies'), and each submission takes ~25-28 s. Measured/computed: submitting every 28 s for 50 min yields ~17,100 samples of per-word-resolved data (k=8 per word per submission, SE ±0.17/word) for free — MORE total samples than the entire 12,000 metered pool, at essentially the same samples/sec (5.7/s free vs ~6/s metered). An agent that notices this runs its broad sweeps through submissions and reserves the pool only for targeted 100-sample pins, gutting the pool's role as 'the primary cost mechanism' (spec section 5). The trade-off arithmetic the spec presents (2,000 pool samples per 20-word sweep) is simply not the price an informed agent pays.
  - *Fix:* Seal per_word_hits (and hits) into hidden metrics — the harness already has the _seal machinery and official.json can carry the unsealed breakdown; add a per-task hidden_metrics list to config.json alongside the existing HIDDEN_KEYS mechanism. A submission then yields only the aggregate score (a 1-word clue change is estimable only by differencing two grades, SE 0.047 — ~20x worse than a targeted 100-sample pool run, restoring the pool's advantage). Optionally also hard-cap the grading meter (open question 4) or enforce a minimum inter-submission interval.
- **[MAJOR] Pool size and marginal-value arithmetic are calibrated for estimation, but optimal play is 1-sample near-greedy screening — measured, the pool does not bind**
  - The server allows temperature down to 0.05. Measured on LFM2.5-230M: a single t=0.05 sample matched greedy output 8/8, and all 4 greedy-miss clues had sampled p̂=0.0 at t=0.7 (spec's own data: greedy-miss ⇒ p ≤ 0.15). So classifying a candidate clue as dead-vs-promising costs 1 sample, not the ~40-100 the spec's ±0.05/±0.08 estimation arithmetic assumes. A realistic optimal session — screen ~150 clue candidates (150 samples) + confirm ~30 survivors at 30-50 samples each (~1,500) — spends ~2,000-4,000 of the 12,000 pool. The 'genuine trade-off, not free averaging' claim (design tension 2) is therefore overstated: under optimal play the binding constraints are the 1-hour box and the agent's clue-generation ability, not the meter. Wall-clock cost per sample IS real (measured 0.245 s/sample here, 0.16 s on the owner's machine), so the task is not broken, but the spec's cost story and pool calibration describe a suboptimal strategy.
  - *Fix:* Three concrete adjustments: (1) restrict agent /generate temperature to a band around the grading temp (e.g. [0.5, 1.0]) — sequential screening then needs ~2-3 samples per dead clue instead of 1; (2) recalibrate P from pilot telemetry with sequential-testing arithmetic (a pool of ~3,000-4,000 restores tension), per open question 4; (3) rewrite the spec section 5 marginal-value paragraph in terms of screen-then-confirm costs so the disclosed arithmetic matches actual optimal play (a type-1 task should disclose the true trade-off, not a misleading one).
- **[MINOR]** The 'greedy-hit ⇒ sampled p≈0.5-1.0' dichotomy is false in general (measured counterexample), weakening the temperature-0.7 rationale as stated — *Fix:* Correct the claim in the spec to the one-sided version ('greedy-miss reliably predicts p≤0.15; greedy-hit ranges p≈0.15-1.0 and needs 20-40 confirmation samples'), and make sure the curation script's survivor confirmation at n=16 (SE ±0.125 at p=0.5) is the acceptance criterion, never a greedy probe.
- **[MINOR]** Official k=32 precision (SD ~0.014-0.017) blurs cross-run comparisons at the same scale as meaningful score differences; strict-< selection leaves a residual p90 ~0.035 selection gap — *Fix:* Cheap at session end: raise k_official to 64 (~3.5-4.5 min once, SD ~0.010-0.012), and officially grade the top 2-3 DISTINCT programs by visible score, reporting the official-best (with 2-3 candidates at official SD ~0.01 the reintroduced min-selection bias is <0.01, a strict improvement on losing a 0.035 selection gap).
- **[MINOR]** Visible-grade SD is quoted as a constant 0.033 but is p-vector-dependent, and the empirical SD estimate has only 2 degrees of freedom — *Fix:* Cite the analytic binomial formula in spec.md and state the SD range (≈0.02 at baseline to ≈0.036 mid-climb at k=8) rather than a single number; keep the empirical replication only as a sanity check with more reps (≥10) if quoted.

*Notes:* Empirical grounding (LFM2.5-230M, fp32, 1 thread, /tmp/lfm_bench/venv, this machine): same-seed generation bit-identical (rescore premise holds); 0.245 s/sample mean at t=0.7 (spec's 0.16 s same order — machine-load dependent); real signal at 230M confirmed — definitional clues gave p̂(k=12) camera 1.0, computer 0.67, sleep 0.75, cold 0.17, bridge/piano/banana/umbrella 0.0, matching the spec's attribute-echo story and its verified-word table. Core statistical machinery is sound: the score is an unbiased mean of Bernoulli outcomes (no ratio/extreme-value estimator bias), theory-matching SDs (k=8: 0.028-0.033; k=32: 0.014), evaluator-owned k blocks averaging on the official metric, harness-drawn os.urandom master seeds block seed-fishing and CRN gaming of grades, and the official k=32 fresh-seed re-grade provably removes the ~1.2-SD strict-< winner's-curse optimism (simulated: optimism +0.037, official correction +0.037). One-word improvements are detected by a single visible grade ~69% of the time, supporting the intended multi-iteration climb over ~15 visible SDs of headroom. The two major findings both attack the same design tension — COST — and are fixable with existing machinery: (1) seal per_word_hits (the harness's _seal mechanism exists; currently session.py would expose it, making the uncapped grading meter a free per-word experiment channel delivering more samples/hour than the entire metered pool); (2) the 12,000 pool is calibrated against estimation arithmetic, but measured 1-sample near-greedy screening (t=0.05 allowed; greedy-miss ⇒ p̂=0, 4/4) means optimal play spends ~2-4k samples — restrict agent temperature to ~[0.5,1.0], recalibrate the pool from pilot telemetry, and fix the disclosed arithmetic. Neither invalidates the task concept; with those fixes plus the minor selection/reporting tweaks (k_official=64, official-grade top-3, p-dependent SD statement), the design meets its type-1 statistical requirements. The mandatory pilot campaign (open questions 2-4) remains the right gate for pool size and k_visible.

### Lens: harness — verdict: FIXABLE

- **[MAJOR] Self-test channel (`bench evaluate`) is grader-token-billed, giving unmetered free averaging that defeats the pool**
  - The design's entire COST story rests on the 12,000-sample metered pool because per-sample wall cost is only ~0.16 s ('wall time alone cannot stop averaging'). But the harness workflow hands every agent an uncounted grading channel: GOAL.md / loop/optimize.py line 55 instruct the agent to self-test via `PYTHONPATH=... python3.12 -m bench evaluate <task> program.py --json` — 'does not record a submission'. That path runs the same evaluate.py, which sends `X-Grader-Token` unconditionally (from TEXTOPT_LM_TOKEN injected by the runner), so all 160 samples bill to the grading meter, which per §5 'never dies'. An agent can loop `bench evaluate` at ~26 s per call (~2 grades/min, ~80+ free full-grade replicates in the time box, more via backgrounding since the server queue is the only serializer) to average visible-grade noise to SD≈0.01 and to get free per-word p-hats from per_word_hits, without spending one pool sample. The pool then meters only the /generate side door while the front door is open, collapsing the sample-allocation game the task is built around. Open question #4 gestures at 'whether grading needs its own hard cap' but the design text asserts grading is deliberately uncapped.
  - *Fix:* Bill non-submission evaluations to the agent pool: runner.evaluate injects TEXTOPT_LM_TOKEN only when called from session.submit, verify_run(rescore) and `bench official`; evaluate.py sends the token header only if the env var is present, else the server meters the call against the run's pool (batch-bill 160). Have `bench evaluate` draw and print its seed so agent self-MC stays reproducible. Update the GOAL.md template for stochastic tasks ('self-tests cost 160 pool samples'), and adjust the pool arithmetic in spec.md accordingly (12,000 pool = 75 self-grades OR finer per-word MC — that is the intended trade-off, now actually enforced). As a backstop, add the meter-reconciliation check from the grading side: verify/audit compares the server's grading-meter total against 160×n_submissions + official's 640 and flags any excess.
- **[MAJOR] Mid-run server failures permanently poison the hash-chained record and break `verify --rescore`**
  - session.submit (bench/session.py line 199) appends whatever result runner.evaluate returns to the append-only, hash-chained submissions.jsonl. If the LM server is down, wedged, or restarting when a submission grades, evaluate.py's urllib call raises, the evaluator dies with no result line, and the submission is recorded ok=false with a stderr-tail error — indistinguishable from a genuinely invalid program. Later `bench verify --rescore` re-runs it against a healthy server, gets ok=true, and verify_run (line 356) reports 're-score ok=True, recorded ok=False' — an integrity PROBLEM that can never be cleared, on every submission that raced an outage. The design's server-lifecycle plan (harness change #6) covers start-of-session health checks only; a 1-hour campaign with 10 concurrent jobs will see transient failures. The same failure mode also mis-charges the agent: strict-< best tracking is fine, but the iteration is burned and the record is permanently 'dirty'.
  - *Fix:* Introduce an infrastructure-error result class: evaluate.py catches URLError/timeout/HTTP 5xx around /info and /generate and emits `fail("LM server unavailable: ...", metrics={"infra_error": true})` (leak-safe, deterministic message). session.submit, on infra_error, retries after a short health-check backoff (bounded, e.g. 2 retries), and if still failing records the submission with an `infra: true` flag; verify_run skips rescore comparison for infra-flagged records (like it already special-cases tolerance), and `bench report` renders them as INFRA not INVALID. run_campaign.py additionally monitors the server process and restarts it (fingerprint must match the recorded one).
- **[MAJOR] Single global sample pool + /reset_pool is incoherent under the actual campaign concurrency (default 10 jobs)**
  - tools/run_campaign.py runs up to `--concurrency 10` sessions at once, and the design (§6) has the campaign runner call `POST /reset_pool` 'at session start'. With one process-global pool: (a) any session starting mid-campaign resets the pool of every other live session; (b) one hot run's experiments drain siblings' 12,000-sample budgets; (c) the proposal in open question #6 (bill agent calls without a run id to 'the active run') is undefined when several runs are active. The pool is the task's primary cost mechanism (finding 1), so broken attribution breaks the task's cost semantics for exactly the campaigns the pilot requires (TASK_AUTHORING step 5). This is flagged as an open question in the design, but it is a precondition for running the mandatory pilot, not a post-pilot calibration.
  - *Fix:* Per-run pools keyed by per-run agent tokens: run_campaign (or `bench workspace`) requests a token from the server at session creation (`POST /new_pool` with the grader token → {agent_token, pool=12000}), writes it into the workspace env (e.g. TEXTOPT_LM_AGENT_TOKEN, documented in spec.md for the agent's own /generate calls), and records the token id in run metadata. The server meters each pool by token and rejects /generate with no/unknown token. Drop /reset_pool. Spoofing another run's token remains cooperative-boundary (same class as the grader token), but honest concurrent runs no longer interfere, and per-token telemetry gives the pilot the sample-usage data open question #4 needs.
- **[MAJOR] Grading latency under a serialized server + concurrent jobs can exceed timeout_s=300, compounding the infra-poisoning problem**
  - The server serializes all generation under one lock at ~6 samples/s (0.16 s/sample). Numbers in the spec (visible 25–28 s, official ~100 s) are single-tenant. With N concurrent taboo sessions each grading (160 samples) and running pool experiments, a grading request stream competes FIFO with everything else: 5 simultaneous visible grades alone are ~800 queued samples ≈ 130 s each; add agent experiment traffic from 5–10 agents (each entitled to 12,000 samples ≈ 32 min of server time) and a k=32 official grade (640 samples, ~100 s unloaded) plausibly exceeds the 300 s wall timeout in config.json — producing subprocess.TimeoutExpired INVALID records (bench/runner.py line 130) that hit exactly the unresolvable rescore-mismatch mode of finding 2 (rescore later, unloaded, succeeds). Note also `eval_cpu_seconds` will be ~0 for this task (compute lives in the server process, outside RUSAGE_CHILDREN), so bench/trace.py attributes grading time to `cum_model` — acceptable semantics, but contention silently inflates it.
  - *Fix:* Three cheap layers: (1) server-side priority — requests carrying the grader token jump the queue (grading and rescore are small, bounded bursts; experiments absorb the delay, which is their pool cost anyway); (2) set timeout_s to 900 for this task (wall timeout is a safety guard, not a score, per runner.py's own doc); (3) campaign guidance in the task README: cap concurrent LM-graded jobs (e.g. 3) or accept degraded grading cadence — with measured queue-wait telemetry from the pilot to pick the number.
- **[MINOR]** audit's zero-metric heuristic false-positives on this task's honest submissions — *Fix:* Make the zero-score/zero-metric heuristic config-aware: config.json gains e.g. `"plausible_zero_metrics": ["hits"]` and `"zero_score_plausible": true` (or audit skips the zero checks for `stochastic` tasks and instead applies the new clue-table rule from §7 plus the grading-meter reconciliation of finding 1). Keep the escape-gadget scan unchanged.
- **[MINOR]** official.json sits outside the hash chain and its binding to the graded program is unspecified — *Fix:* official.json must record `program_sha256` (of the exact bytes graded — snapshot them, don't re-read best_program.py on faith), the submissions.jsonl chain-tip sha256, and n_records at grading time; `bench verify` checks all three plus (with --rescore) replays the official seed and requires the recorded official transcript_sha256/score to reproduce. Refuse to overwrite an existing official.json without an explicit --force.
- **[MINOR]** Seed plumbing for the non-session entry points (`bench evaluate`, `baseline`) is unspecified, and the eval-telemetry log drops the seed — *Fix:* `bench evaluate`/`baseline` draw a seed (os.urandom) when none is given, print it in the result line, and accept an explicit `--seed` for the agent's own reproducibility; add `eval_seed` to the TEXTOPT_EVAL_LOG record. Trivial once change #1 lands, but it belongs on the harness-changes list — it touches the exact command template baked into GOAL.md.
- **[MINOR]** tests/run_checks.py, the broken suite, and CI need an explicit server-availability contract; evaluator sketch mishandles clue() exceptions — *Fix:* run_checks.py: probe GET /info first; run broken-rejection rows always, skip (with a loud SKIPPED line) LM-dependent headroom/determinism rows when the server is absent, and add a `--require-lm` flag for the authoring gate. Evaluator: wrap the clue-collection loop in eval_lib.run_program (or try/except → eval_lib.fail with the word name). Enforce '  ' not in clue. Add one sentence to spec.md §4 noting per_word_hits is visible at k=8.
- **[MINOR]** Grader token is delivered via the same state file the agent must read, and transcripts are hash-only — *Fix:* Split the files: state.json = {port, fingerprint} (agent-readable); grader token in tools/.lm_server/grader_token (documented off-limits, like bench/heldout.py) or passed to the harness via the campaign runner's env only. Store the 160 sampled texts (a few KB, they average 3.3 tokens) zlib+sealed in the submission record's `sealed` field or a sidecar transcripts/NNN.json.zst; keep transcript_sha256 as the verify criterion and use the stored texts for drift forensics only.

*Notes:* Reviewed against bench/runner.py, bench/session.py, bench/cli.py, bench/eval_lib.py, bench/audit.py, bench/trace.py, loop/optimize.py, tools/run_campaign.py, TASK_AUTHORING.md. The core harness fit is genuinely good: evaluate.py stays stdlib and pre-LM validation keeps the broken/ suite serverless; the seed-per-call + transcript-hash + fingerprint rescore scheme is compatible with verify_run's exact-metrics comparison (_metrics_close passes with recorded per_word_hits/transcript_sha256); the env-allowlist injection is correctly specified as a runner-side read of state.json rather than TEXTOPT_* passthrough (runner.py's existing comment explains why passthrough would be forgeable — a fake-server submission is additionally caught by the recorded-vs-live fingerprint check); score-on-sampled-text avoids the emit-answer memorization boundary entirely, matching TASK_AUTHORING's robust-shape analysis. The seven listed harness changes are individually small, but the list is incomplete: the findings above add the self-test billing/seed path, infra-error result class, per-run pool tokens, grader-priority/timeout headroom, audit zero-heuristic exemption, official.json chaining, and run_checks server gating. None require redesigning the task; hence 'fixable' rather than 'sound'. The most load-bearing fix is finding 1 — without it the metered pool, the task's stated answer to design tension #2, is fiction because GOAL.md itself advertises the bypass. Findings 2–4 interact: outages and contention both mint permanently unverifiable records, so they should land before the mandatory pilot campaign, not after. Not re-verified empirically here: the 0.16 s/sample and SD numbers (taken as given from the design's measured section); they only affect the arithmetic in findings 1/4, not their existence.


---

# lm_copyedit — NOT SELECTED (judge mean 6.97, 3/3 advances, needs-real-LM 8.3/10)

**Disposition.** ALTERNATE (not broken): dense edit-distance signal is attractive, but it overlaps the selected steering/extraction shapes, scored lowest of the shortlist with the panel, and its gold-table fingerprinting fix enlarges the leak scan. Revisit if a selected type-1 task fails its pilot gate.

**Task type:** random (type 1: perfect-information stochastic grading via real-LM sampling)

## Measured numbers

MEASURED (fp32 LFM2.5-230M, Apple M5; probes in /tmp/copyedit_probe/probe_spec{,2,3}.py). Per-sample cost at production config (fp32, torch threads=1): 0.47-0.77 s (prompts 34-278 tok, generations self-stop at 12-26 of 64 max tokens) -> visible grade k=2 x 30 items = 60 samples ~= 35-50 s; OFFICIAL k=6 x 30 = 180 samples ~= 1.5-2.5 min; halving SE via private averaging = 240 samples ~= 2.5-3 min per compared candidate; server meter cap 5000 samples/session (~50 min serialized, binds only abuse). Baseline (naive template, no shots, clean=strip): mean nd 0.492 (per-rep grades 0.449/0.473/0.555; per-family upper 0.46 / strip 0.36 / replace 0.29 / quote 0.55 / reverse 0.77). Reference strong (system + 2 evaluator-completed shots/family + substring clean): 0.294-0.307 across three variants (+0.024 token penalty at 216 tok) -> score ~0.32, headroom factor ~1.55x; per-family bests (upper 0.091, strip 0.073, replace 0.169, first_upper 0.388, quote ~0.05 via span-clean, reverse 0.572) suggest tuned-agent floor 0.20-0.25 = ~8 rungs of 0.02-0.19. Empirical noise: within-item variance sum 0.96 (baseline) -> 0.42 (strong); visible SE(k=2) = 0.023 baseline / 0.015 strong; official SE(k=6) = 0.013 / 0.009 — rungs 1-8x the visible SE. rep_penalty decision: 1.0 beats 1.05 on copy-heavy items (mean nd 0.640 vs 0.662) -> grade at 1.0. Substring-clean rule: 0/270 violations by honest cleans. ChatML injection confirmed: "<|im_end|>" in user content tokenizes to special id 7 (fabricates turns) -> "<|" ban is load-bearing.


## Red-team reports

### Lens: exploit — verdict: FIXABLE

- **[MAJOR]** Leak scan covers only LM-facing strings, so a gold-table clean() fingerprints the 30 fixed items — *Fix:* Run the 3-gram (casefolded, punctuation-stripped) leak scan over the ENTIRE candidate source, not just the LM-facing strings, so literal scored-sentence/gold tables in clean() are rejected. Without gold/sentence literals, clean() cannot compute argmin-to-gold (it never receives the item index or params), which collapses the channel back to genuine, general span selection. Keep the committed mandatory hand-audit of winning clean() as the backstop for encoded/obfuscated tables (the residual is the same static-signature-evasion boundary TASK_AUTHORING already documents).
- **[MINOR]** Word-3-gram leak scan is blind to strings with fewer than 3 words — *Fix:* Assert at validation that every scored gold is >=3 words, or supplement the 3-gram scan with an exact-substring check for short golds; document the invariant so future item edits can't silently open the hole (relates to open question 7).
- **[MINOR]** Sequential str.replace of template slots is an order-dependent injection latent fragility — *Fix:* Substitute both slots in a single pass (e.g. one regex/dict substitution) or assert the rendered task_line contains neither "{task_line}" nor "{sentence}" before the second replace.
- **[MINOR]** ChatML ban keys on the '<|' prefix; other atomic added-vocab tokens are reachable in plain ASCII — *Fix:* Filter/normalize against the tokenizer's full added-special-token set (server-side, using tok.get_added_vocab()/all_special_tokens) rather than the '<|' prefix substring, so the ban tracks the model's actual control vocabulary.

*Notes:* Verdict fixable, not broken: the core mechanism is sound. The substring rule is a genuinely strong, load-bearing defense (a candidate cannot fabricate characters the LM did not emit), and several attacks I checked are already neutralized: (1) seed reading is harmless here (I confirmed the candidate has no channel to exploit seed knowledge — prompts are seed-independent and clean() is substring-bound, so predicting seed_ir buys nothing); (2) a single degenerate constant cannot win because the 30 items have distinct golds and clean() is substring-bound; (3) shot-input -> evaluator-computed-gold answer smuggling is closed by the 3-gram scan covering shot inputs (except the <3-word blind spot in finding 2); (4) lucky-draw fishing is self-defeating by construction — the official k=6 regrade draws a FRESH seed on best_program, so fishing only causes a higher-variance/worse-true program to be officially regraded at its true (worse) score, and the metered pool bounds resubmission volume.\n\nThe single material weakness is the memorization channel in finding 1: it is exactly the fixed-instance fingerprinting/dual-path pattern the repo has been burned by before (kv-family, checkpoint_plan), and the design under-claims it as \"buys nothing material\" without measurement. Extending the leak scan to the whole source is a cheap, high-value fix that keeps the task on the robust side of the TASK_AUTHORING boundary while preserving the legitimate general span-selection wins (extract_quote). Given the cooperative threat model, the committed hand-audit of winning clean(), and bench audit, the residual after that fix is acceptable and correctly characterized as per-system-stable, cooperative-model robustness.\n\nSecondary recommendations (not findings): adopt k=3 visible reps — the smallest intended rungs (0.02) are ~1 visible SE at k=2 (0.015-0.023), so late-rung selection is noisy; the +~20s/grade cost is modest. Have bench determinism compare transcripts_sha256 across its two runs (open question 6) — strictly stronger than score equality and cheap. The single shared meter pool (open question 3) is a self-strand foot-gun, not a security exploit; acceptable but worth a spec warning. reverse_words' ~0.57 floor + largest per-item variance is a weighting choice to revisit post-pilot, not a robustness issue.\n\nI did not modify any repo files; empirical checks were run only against the model/tokenizer in the provided venv.

### Lens: statistics — verdict: FIXABLE

- **[MAJOR]** Run-invalidation on stochastic clean() substring violations can kill the official regrade with no defined fallback — *Fix:* Score any violating sample as d=1 instead of invalidating the run (still strictly disincentivizes; opens no new exploit surface because clean() can already self-guard with `if norm(cand) in norm(raw): return cand else: return raw`, so gold-table cleans gain nothing they don't already have), OR keep hard-fail for the visible grade but define the official-regrade failure path (fall back to the next-best visible submission and record why). Also raise the 0/270 evidence to ~0/1000 samples if hard-fail is kept.
- **[MINOR]** Documented winner's-curse magnitude is understated (~1 SE claimed; ~1.9-2 SE simulated), though the official-regrade mechanism itself is verified sound — *Fix:* Update the spec/README text to '~2 visible SE (~0.03-0.05) inflation of the visible best, removed by the official regrade; residual selection regret ~0.003 mean / ~0.01 p90'. No mechanism change needed.
- **[MINOR]** Meter cap 5000 binds diligent legitimate strategies, contradicting 'binds only abusive patterns' — *Fix:* Either exempt harness-initiated grading from the meter (meter only /generate calls outside an active evaluation), or raise the cap to ~8000 and document that wall-clock is the real meter; optionally reserve a grading-only budget (e.g. last 600 samples usable only by evaluate.py) to prevent stranding.
- **[MINOR]** Official k=6 SE (~0.009-0.013) is half a late-game rung; the headline number under-resolves the ladder it reports — *Fix:* Raise official reps to 10-12 (300-360 samples, ~3-4 min once per session): SE drops to ~0.006-0.009, resolving one rung at ~2 sigma. Keep visible k=2 as designed.
- **[MINOR]** Common-random-numbers paired comparison is an unmentioned free variance-reduction route that changes the private-averaging cost arithmetic — *Fix:* Note in spec.md that paired-seed comparison is legal and expected (it rewards statistical skill, on-theme), and soften the '240 samples per compared candidate' claim to 'up to ~240 (fewer with paired seeds)'. No mechanism change.
- **[MINOR]** timeout_s 300 has thin margin for the official grade under server contention — *Fix:* Set timeout_s to 600 (cpu_s stays 60 since the evaluator blocks on I/O), and/or have `bench official`/`verify --rescore` run with agent access paused (the experimenter restarts the server for verify anyway per the spec — just make that explicit for `bench official`).

*Notes:* Empirical grounding (fp32 LFM2.5-230M, probes under /tmp/redteam_stats/, reusing /tmp/copyedit_probe/probe_spec.py definitions): (1) Independent 10-rep re-measurement of the baseline program with a fresh seed base (23/30 items completed before forced finalization; 230 samples) reproduces the spec's noise numbers: within-item variance sum extrapolates to 0.91 vs claimed 0.96 → visible SE(k=2) = 0.0225 vs claimed 0.023, official SE(k=6) = 0.0130. The spec's per-rep grade triple (0.449/0.473/0.555, SD 0.056 ≈ 1.7x formula) was a 2-df fluke, not an underestimated SE — the claimed SEs are CONFIRMED. (2) Winner's-curse simulation (/tmp/redteam_stats/curse_sim.py, 4000 sessions): official-regrade bias +0.0001 (unbiased as designed), visible-best inflation −0.0285 (~1.9 SE, spec says ~1 SE — doc fix), selection regret +0.003 mean / +0.010 p90 — strict-< tracking plus official regrade handles the curse; no incumbent-ratchet pathology at these SEs. (3) Marginal-value arithmetic verified: resolving a 0.02 rung at 2 sigma needs ~285 samples ≈ 3 min — the sample-cost trade is real inside a 1-hour box. (4) Signal at 230M confirmed: baseline family means 0.03-0.85 per item, nothing pinned at 0/1; observed per-item distributions are strongly bimodal (ranges up to 0.75) but the 60-sample grade is CLT-normal enough for the SE arithmetic. (5) No estimator bias: score is a plain mean of bounded per-sample distances; seeds are sha256-derived from os.urandom per submission, closing seed-overfitting; harness strict-< best-tracking confirmed in bench/session.py:166-167. Core statistical design is sound; the findings are boundary conditions (stochastic hard-fail on the official grade, meter/timeout margins) and documentation calibration. Probe scripts and partial raw data: /tmp/redteam_stats/probe_se.py, curse_sim.py, analyze.py; raw output at /private/tmp/claude-501/-Users-ethanewer-text-opt-bm--claude-worktrees-random-tasks/2e1875ec-c14b-40da-a542-c956cd0a3852/tasks/b6nvb8kp2.output (background probe still completing the strong arm; nothing observed so far contradicts the spec's strong-arm numbers).

### Lens: harness — verdict: FIXABLE

- **[MAJOR]** Infrastructure failures create permanently unverifiable records (rescore ok-flag mismatch) — *Fix:* Classify infrastructure errors distinctly: evaluate.py catches all urllib/server errors and emits fail() with a machine-readable marker (e.g. metrics {'infra_error': true}). Then either (a) Session.submit retries once and, if still infra-failed, raises instead of appending (submission never enters the chain), or (b) verify_run skips the rescore comparison for records whose sealed/visible metrics carry infra_error (hash-chain and snapshot checks still apply). Option (a) is cleaner for best-tracking; either is a small, config-gated change.
- **[MAJOR]** Single hardcoded server (port 8377) collides with the 5-concurrent-jobs campaign model and the per-session meter — *Fix:* Plumb a per-session server URL the same way as --noise-seed: config declares a port RANGE or the session records server_url at creation; runner.evaluate appends --server-url http://127.0.0.1:PORT (argv, not env) and verify_run replays it. run_campaign.py allocates one server process per stochastic job (230M fp32 is ~1 GB RSS, 5 instances fit in 32 GB, 1 thread each fits 10 cores) and pre-flights each health URL. Alternatively, document that stochastic tasks are scheduled at most one per server and make run_campaign enforce that; either way the constraint must be explicit, not discovered mid-campaign.
- **[MINOR]** timeout_s=300 has no headroom for official k=6 at the design's own worst-case numbers — *Fix:* Set timeout_s to ~900 (wall timeout is a safety guard, not a score input, so generosity is free), or have bench official / runner scale the timeout with --reps (e.g. base + reps * 30 * 2 s).
- **[MINOR]** Shot validation misses family-specific input applicability; evaluator-owned gold_fn can crash on valid-looking exemplars — *Fix:* In _validate_shots, add per-family input checks: extract_quote inputs must contain exactly two '"' characters with a non-empty span between; reject with a deterministic message like the other rules. Audit each gold_fn as a total function over validated inputs and add a broken/ fixture (lm_copyedit_shot_noquote.py) to lock the behavior.
- **[MINOR]** GOAL.md workspace template contradicts the task's rules for stochastic tasks — *Fix:* Add a config-gated stochastic paragraph to GOAL_TEMPLATE (mirroring the existing generalization feedback_note mechanism): grade is a fresh-seed sample of a known expectation, resubmission redraws, self-test uses a fixed seed (bit-stable but not a draw from the grading distribution), and the server query channel + meter. Consider having `bench evaluate` on stochastic tasks draw a random seed by default (printing it) and reserve the fixed seed for `bench determinism`, so self-test scores are honest draws.
- **[MINOR]** trace.py cost accounting classifies all LM sampling as machine-independent 'model' time — *Fix:* Have the server report per-request generation seconds; evaluate.py sums them into a metric (e.g. sampling_seconds) and either trace.py adds it to the local term for stochastic tasks or the README's trace section documents that LM-graded tasks' sampling time is local-but-unrescaled. One-line metric plus a documentation choice.
- **[MINOR]** bench audit has no signatures for this task's new cheat surface — *Fix:* Add advisory audit signatures: string-fragment reconstruction of '<|' or 'im_end', socket/urllib name fragments in candidate sources for this task, and a heuristic for fingerprint-keyed dispatch in clean() (many raw-text conditionals / hash comparisons). Keep them advisory, matching audit.py's existing memorization-hit stance, and add the row to section 11.

*Notes:* Verified against the real code, the core plumbing fits the harness unusually well: (1) argv flags (--noise-seed/--reps) are the correct channel since runner.py's child env is a strict allowlist and evaluate() already appends per-mode flags; (2) recording noise_seed in the record AND echoing it in metrics means verify_run's existing _metrics_close at score_tolerance 0 checks seed, score, per-family means, server_fingerprint, and transcripts_sha256 with no changes to comparison logic — transcript-level rescore verification comes free; (3) the fingerprint-in-metrics trick makes server-config drift a loud rescore failure exactly as claimed; (4) bench determinism's fixed-default-seed integration and run_checks.py skip-if-server-down are consistent with existing patterns (score_tolerance machinery, mem_infer precedent for per-system stability). Enumerated harness changes in section 11 are individually small and correctly scoped, with three omissions: failure-record lifecycle under transient server errors (finding 1, the most important — it can permanently break verification of honest sessions), per-session server/port allocation under the 5-concurrent-job campaign model (finding 2), and GOAL_TEMPLATE/audit.py updates (findings 5, 7). Heavy changes: none fundamentally — the largest is run_campaign per-job server management plus a --server-url replay path, which is bounded and cooperative-threat-model-compatible because the URL is recorded and replayed, never agent-supplied. Session.submit holding the fcntl lock for a 35-50 s grading is acceptable (per-run-dir serialization is intended). The open questions' proposals (Q6 transcript-hash comparison in determinism: yes, cheap via metrics equality; Q3 single meter pool: acceptable only after finding 1's infra-error classification, since meter exhaustion currently poisons the record chain). Verdict 'fixable': both major findings have concrete, local fixes that do not disturb the task's scoring design.


---

# reroll_fee — NOT SELECTED (judge mean 6.80, 2/3 advances, needs-real-LM 7.2/10)

**Disposition.** REJECTED: the statistics red-team proved the headroom ladder collapses — myopic per-job thresholds are already exactly optimal (claimed rungs 5-6 mathematically unreachable, the hint-level depth handle is worth 0.000 under the design's own measured pmfs), and offline MC resolves the thresholds far cheaper than claimed. The stochastic machinery would be decorative for a competent optimizer (same class as track-1's spec_decode rejection).

**Task type:** random

## Measured numbers

MEASURED (LFM2.5-230M fp32, torch.set_num_threads(1), Apple M5, temp 0.9 / top_k 50 / max_new 56; probes at /tmp/lfm_reroll/): per-sample cost 0.25-0.55 s on plain constraint prompts (mean ~0.37 s; EOS usually before 56 tokens), 0.56-1.04 s on hinted/longer prompts. Same-seed regeneration bit-identical, cross-seed differs (verified). Per-job violation pmfs (n=24 each) span mean 0.29 (J7, P0=0.71) to 2.12 (J6, P0~0): J1 0.58, J2 0.83 (P0=0.25 = deliberately fee-borderline), J3 1.38, J4 0.96, J5 1.67, J8 1.71. Hint-level shift for scheduled jobs (n=16/cell): J6 mean 2.19 (L0) -> 1.38 (L1) = real non-i.i.d. handle; J5 ladder flat and temperature schedule NON-monotone (0.5 worse than 0.9) -> both rejected, calibration rule |dmean|>=0.4 required. Fee calibration DP on measured pmfs (8 i.i.d. jobs, fee 0.25, cap 12): accept-first 9.52, optimal-stopping-with-recall 6.45, always-chase-zero 11.39 -> both degenerate corners lose by >=3. k per visible grade = 3 rollouts x 10 jobs = 30 gens (baseline, ~12-20 s) to ~50 gens (strong, ~20-45 s), worst case 360 gens ~150-280 s. Official = 16 fresh rollouts (+top-3 distinct), ~4-12 min post-session. Empirical noise: single-rollout SD 2.29 (baseline policy) / 1.45 (optimal, MC n=4000); visible R=3 SE ~1.4 -> 0.85; official K=16 SE ~0.6 -> 0.36. Headroom: baseline ~13.9 (10 jobs) vs strong ~8.8-9.3 official, ~6 rungs (early rungs 0.8-1.4 visible above noise, late rungs 0.2-0.6 resolved only by the agent's own offline MC — intended). Offline pmf estimation cost: 12 cells x 100 samples ~1200 gens ~8-15 min of the 1-hour box; borderline job J2 needs ~200 extra samples for +/-0.03 on P0.


## Red-team reports

### Lens: exploit — verdict: FIXABLE

- **[MAJOR]** Official grade is chosen by visible-min, which is noise-dominated — the MC-optimal program is frequently never regraded, so the last rungs of the claimed headroom are uncreditable — *Fix:* Decouple official grading from the visible score: always include the agent's FINAL submitted (or explicitly nominated) program in the official-regrade set, since the spec already tells the agent its real signal is offline MC, not visible. Additionally consider restructuring the job mix so a policy rung exceeds ~2x the difference-SE (fewer jobs with larger per-job violation spread, or the deferred shared-draw-budget coupling that amplifies policy-quality gaps), and clip the advertised creditable headroom to the selectable portion. Report official SD alongside the mean so an uncreditable tail is visible.
- **[MAJOR]** Cross-process/restart bit-reproducibility of the LM server is unverified, yet rescore uses score_tolerance 0 plus transcript hashes — *Fix:* Add an authoring-time gate before declaring machine_scoped: fully restart the server process and regenerate a fixed battery of seeds, requiring bit-identical text vs a prior run (and ideally after a machine reboot). If cold-start bit-identity does not hold, either (a) pin the exact numeric path (set deterministic algorithms, disable denormals) and re-verify, or (b) relax the reproduction contract to a per-job violation-count match with a small score_tolerance instead of transcript-hash identity, and drop the exact-hash guarantee from the spec.
- **[MAJOR]** Transcript hashes and server fingerprint are recorded but not actually enforced by rescore as specified — *Fix:* Extend verify_run's rescore path to compare a declared set of invariant metrics (transcript_hashes, server_fingerprint, n_lm_calls) for exact equality when the task is stochastic, and add that to the §12 change list explicitly. Otherwise remove the tamper-evidence claims from the spec so they are not overstated.
- **[MINOR]** Winner's-curse farming of the reported number via textually-distinct copies; the quoted selection bias understates the order statistic — *Fix:* Define 'distinct' by NORMALIZED source (strip comments/whitespace) or by a behavioral hash so trivial variants collapse to one candidate; and/or regrade only the single declared/final program at a higher K rather than a visible-selected top-3. Re-derive the residual bias with the correct order statistic and state it honestly.
- **[MINOR]** noise_seed is reachable by in-process frame-walking but is benign here — worth an explicit note, not a gate — *Fix:* Document that the seed leak is non-exploitable because text generation requires the out-of-process server; optionally, as defense-in-depth, have main() avoid keeping the plaintext noise_seed as a live local across the decide() call (derive per-draw seeds up front, drop the master), so even a frame-walk finds nothing.

*Notes:* Memorization is genuinely closed and correctly reasoned: the candidate returns only an INDEX into evaluator-held, freshly-sampled texts drawn under a secret per-submission os.urandom seed, so there is no scored output to emit/precompute and no sealed split to overfit — this is a legitimate THIRD robust shape beyond the two in TASK_AUTHORING.md (measurement-scored, reconstruction-scored): 'select-among-fresh-stochastic-outputs.' The classic emit-answer/dual-path/regenerate-the-seed attacks that broke ops_connect/tsp_budget/checkpoint_plan/kv-family do not apply. Distribution-collapse and prompt-injection are also correctly closed: prompts, sampling params, checker, fee, and cap are all evaluator-owned; sampled text is only string-checked, never executed or LLM-judged, and there is no feedback loop from decide()'s return into any later LM call. Return validation (type(x) is int rejecting bool, range check, -1-at-cap fail, exceptions fail) and the forbidden-import set (network + nondeterminism sources) are sound and consistent with eval_lib's guard; a deterministic policy is WLOG optimal so banning random/time costs nothing. Cross-call module-global state persists across the 360 decide() calls in one evaluation but is harmless (jobs are independent; cannot reduce a job's own violations; still deterministic given the seed). The degenerate corners (accept-first, always-chase-zero) both lose by >=3 per the fee calibration, so no degenerate policy wins. Net: no fatal/unfixable hole -> not 'broken'. The exploit surface is essentially clean; the real weaknesses are SELECTION-UNDER-NOISE and REPRODUCIBILITY plumbing (findings 1-3), which are fixable. Open-question 1 (does this genuinely need a real LM vs generic optimal stopping) is a design-value judgment outside the exploit lens; the text-handling traps (curly quotes, digit-vs-word, hyphenation) add real but modest language work and are not gameable. Verified against the repo: bench/session.py best-tracking (strict < guide_score, best_program.py = visible-min), verify_run rescore compares only guide_score within score_tolerance, and eval_lib guard/forbidden mechanics.

### Lens: statistics — verdict: FIXABLE

- **[MAJOR]** Claimed ladder rungs 5-6 are mathematically unreachable: myopic per-job thresholds are already exactly optimal, so real depth is ~3 rungs (13.7 -> 12.1 -> 10.0 -> 9.40), not 6 rungs to 8.8 — *Fix:* Recompute the ladder and headroom claims with exact policy evaluation (script at /tmp/lfm_redteam/sim_selection.py). To restore depth, add structure where myopic-stationary is provably suboptimal: properly calibrated non-i.i.d. schedules (see next finding), per-job heterogeneous fees interacting with a shared draw budget (the spec's own open question 7 -- this genuinely couples jobs into a knapsack and defeats the reservation rule), or fees large enough relative to violation granularity that finite-horizon end effects bind. Re-verify with the pilot campaign that the optimizer improves over >=5 iterations.
- **[MAJOR]** The scheduled-job (hint-level) depth handle is worth exactly 0.000 under the design's own measured level pmfs; |dmean| >= 0.4 is the wrong calibration statistic — *Fix:* Replace the |dmean| >= 0.4 rule with a decision-relevant criterion: at freeze, compute EV(level-aware DP) - min over level-blind policies EV, on n>=400/cell pmfs, and require >= ~0.15 per scheduled job (and require the redraw sets to differ on reachable states). Engineer the levels so P(v=0) crosses the fee break-even between L0 and L1 (e.g. L0 with P0 ~ 0 -> L1 with P0 >= 0.35), which is what actually moves thresholds.
- **[MAJOR]** The offline-MC cost trade-off is much thinner than claimed: n=24/cell already achieves regret 0.08, and resolving the 'borderline' job is worthless by construction — *Fix:* Accept and document the true numbers (state marginal value: ~0.07 for 24->100 samples/cell, ~0.01 beyond), and get the cost tension from structure instead of precision: heterogeneous fees / shared draw budget make the DP input higher-dimensional so MC estimates feed a harder allocation problem, or add more jobs with distinct pmfs (est. cost scales linearly in jobs). Drop the claim that borderline jobs create a sampling trade-off, or place borderline jobs where the LOSS gradient is steep (P0 near a threshold that flips a high-traffic state, e.g. m=1 on a job visited often), not at exact break-even.
- **[MINOR]** Official-score definition is ambiguous, and 'min official over final-best + top-3 distinct' carries a measured -0.3 winner's-curse bias (spec claims <=0.2); official(final best) alone is unbiased with negligible selection loss — *Fix:* Define the reported official score as the K=16 mean of exactly ONE program -- the final session best (or better, an agent-designated final program, which also removes any residual risk that visible noise crowns the wrong program in deeper-laddered future variants). Keep the top-3 regrades as recorded diagnostics only, never as the score.
- **[MINOR]** n=24 calibration pmfs are too unstable to freeze job placement: an independent replication moved J7's mean from 0.29 to 0.56 (P0 0.71 -> 0.44, ~2.1 SE) and J2's P0 from 0.25 to 0.34 — *Fix:* Run the planned n>=400/cell authoring pass before freezing, recompute the fee-calibration DP and both degenerate corners on those pmfs, and re-derive the ladder with exact policy evaluation (finding 1). Given the demonstrated drift, also record the calibration pmfs alongside the task so post-hoc drift of the served model (fingerprint change) is distinguishable from calibration error.

*Notes:* All headline checks were run empirically, not speculated: (a) fresh LM probe on LiquidAI/LFM2.5-230M (/tmp/lfm_redteam/probe_j2.py) reconfirming seed-determinism and per-sample cost while contradicting two n=24 pmfs; (b) exact DP / policy-evaluation and 1500-session Monte Carlo (/tmp/lfm_redteam/sim_selection.py) on the design's own measured pmfs. What is statistically SOUND: noise is natural and policy-independent per draw (clean CRN-style seed schedule, sha256 of an unseen urandom noise_seed -- no agent-side seed fixing or variance-reduction exploit); visible/official SE arithmetic verified (rollout SD 2.60->1.57, R=3 SE 1.50->0.90, K=16 SE ~0.4); the scoring estimator is an unbiased mean computed exactly in integer quarter-units; winner's-curse containment via official regrade works (P(best-rung program regraded) ~ 1.00, selection loss <= 0.02); deterministic-policy-WLOG and the forbidden-imports rationale are correct; both degenerate corners (accept-first 13.7, chase-zero) genuinely lose. The core statistical problem is DEPTH, not exploitability: search-with-recall from a fully known i.i.d. distribution has a myopic optimal policy, so the task as calibrated is a ~3-rung, ~1.46x task whose advertised rungs 4-6 (exact DP, hint-level DP, pmf refinement) each measure 0.00-0.08, and whose scheduled-job mitigation -- adopted to answer the depth critique -- is exactly valueless under the very pmf shapes cited as evidence. All fixes are concrete and within the design (ΔEV-based schedule calibration with P(v=0) crossing the fee threshold, shared-budget coupling, single-program official). Verdict: fixable. Owner should also note finding 3 when judging the 'grading has real cost' requirement: the priced-sampling tension exists but is exhausted after ~500 generations (~3-4 min), well short of the implied session-long trade-off.

### Lens: harness — verdict: FIXABLE

- **[MAJOR]** Selective-abort censoring: decide() can fail mid-rollout at zero score cost, turning the visible grade into rejection-sampled luck — *Fix:* Make post-load runtime invalidity non-censoring: on exception, non-int, bad index, or -1 at the cap, force-accept a defined draw (e.g. draw 0) plus a fixed penalty (e.g. max violations + all fees for that job), keep grading, and return ok=True with the error text in metrics — only static/load-time problems (forbidden imports, missing decide, import-time crash) fail the submission. Update tests/broken expectations (bad_index/never_accept/not_int now assert the penalty score and recorded error, not rejection). Separately define bench official semantics: a program whose official rollout fails is disqualified and the headline falls to the next candidate.
- **[MAJOR]** timeout_s 600 + one shared serialized 1-thread server = load-dependent evaluation failures (violates the runner's no-flakiness contract) — *Fix:* Run one server instance per campaign job on its own port (config/env plumbed at session creation; ~1-1.2 GB fp32 weights each, 5 x ~6 GB on the 32 GB machine — state the RAM budget and pin 1 thread each, 5 of 10 cores). Raise timeout_s to >=1800 as a pure safety guard sized to worst-case gens x per-gen cost x small contention factor, and direct the agent's MC traffic to its own campaign's server. Document that eval_wall_seconds includes server queue time and dashboards must use n_new_tokens x s/token.
- **[MAJOR]** noise_seed plumbing breaks every non-session runner.evaluate call site (baseline, determinism, bench evaluate default, rescore, run_checks) — *Fix:* Move draw-if-absent into runner.evaluate: if config has stochastic:true and no seed given, draw os.urandom(8).hex(), pass --noise-seed, and include the seed used in the returned result dict so session.submit records rec['noise_seed'] from it (single source of truth) and TEXTOPT_EVAL_LOG/self-test output includes it. verify_run passes rec['noise_seed']; bench determinism passes fixed seeds via the same kwarg. This makes all existing call sites work with one change and keeps the CLI flag optional sugar.
- **[MINOR]** verify --rescore acquires an external service dependency with undiagnosed failure modes — *Fix:* In verify_run, for tasks with stochastic:true: preflight /health before rescoring and abort with a clear 'server unavailable/fingerprint mismatch (expected X, got Y)' message; treat originally-failed records whose error matches the server-unreachable signature as skip-with-note rather than mismatch; document machine+server-scoped reproducibility next to machine_scoped in config and README.
- **[MINOR]** Evaluator sketch bugs: unguarded /health call, and official rollout-count not supported — *Fix:* Wrap the /health fetch in the same retry/fail path as generate(). Add --rollouts N (default 3) to evaluate.py argparse, used for the r loop and the score divisor, and have bench official invoke it with a fresh recorded seed per candidate, writing official.json outside the agent-visible run artifacts with defined failure semantics (see finding 1). Optionally have the server dedupe by (seed, prompt) or log request ids so retries do not distort metering.
- **[MINOR]** Headline-selection rule under variant spam is undefined — *Fix:* Fix the rule in spec/config: the headline is official(final best_program.py) alone; top-3 regrades are recorded diagnostics for the winner's-curse dashboard, never a substitute headline (except via the finding-1 disqualification fallback). Note in spec.md that resubmission spam cannot move the headline, only waste the time box.
- **[MINOR]** noise_seed stored in plaintext in submissions.jsonl is agent-readable mid-run — harmless here but should be stated as a considered decision — *Fix:* Add one line to the design and THREAT_MODEL notes: recorded noise seeds are deliberately plaintext (needed for rescore), safe only because seeds are per-submission fresh and never reused; any stochastic task MUST NOT key future draws on recorded seeds. No code change needed.

*Notes:* Verified against the actual harness code, the core integration story holds up well: eval_lib's load_program/set_candidate_active/_guarded_import compose correctly with the task's extra FORBIDDEN set (urllib/http/socket are blocked only during candidate spans, so the evaluator's own urllib use is unaffected); the nonce result protocol, quarter-unit integer scoring, PYTHONHASHSEED=0 child env (which also strips proxy vars, so loopback urllib is clean), and _metrics_close exact-match on job_stats/transcript_hashes/server_fingerprint all support exact same-machine rescore as claimed; cpu_s=60 is safe because the evaluator idle-waits and RLIMIT_CPU counts only the child's CPU. The candidate module persisting across all 30 job-episodes (module-global state) enables legitimate online pmf learning and is rescore-safe since the seed fixes the whole history. The two majors are the selective-abort censoring channel (a genuinely new failure mode the existing harness semantics — ok=False records cost nothing — make cheap) and the shared-server wall-timeout flakiness; both have concrete, local fixes. The noise_seed plumbing gap is a hard implementability bug as specified (breaks baseline/determinism/run_checks/self-test) but is a one-function fix in runner.evaluate. Harness change list in section 12 is otherwise accurate and small; run_checks server-dependency skip-with-warning is consistent with the existing suite structure. I did not re-run the LM measurements (owner-provided numbers accepted); nothing in the harness lens depends on them except worst-case generation counts, which I recomputed from the cap arithmetic.


---

# record_qa — SELECTED (judge mean 8.27, 3/3 advances, needs-real-LM 9.5/10)

# record_qa — blind-template extraction over a real local LM (RL3 flagship)

**Task id** `record_qa` · **kind** `generalization` + `stochastic` · **score** error rate on a sealed 150-instance validation split, ONE sampled LFM2.5-230M generation per instance (lower = better).

## 1. Story and shape

A 230M on-device model (LiquidAI/LFM2.5-230M) answers numeric questions about short operational records (depot intake logs, shift rosters, bakery order slips, delivery manifests, stock memos). The agent cannot touch the model or the records — it optimizes the **prompt program** around them:

```python
def build(question: str) -> dict:
    # {"system": str, "examples": [(user, assistant), ...],
    #  "user_template": str}   # must contain {document} exactly once;
    #                          # may contain {question}
def parse(sampled_text: str, question: str) -> str | int | None:
    # the answer as canonical digits, e.g. "24"
```

Per instance the **evaluator** (never the candidate) substitutes the sealed document + question into the template (plain `str.replace`, question first, document last — never `str.format`), POSTs it to the experimenter-managed local LM server with a derived seed, and feeds the sampled text to `parse`. Correct iff `parse`'s output (stripped, leading-zeros canonicalized, `int` accepted and stringified) string-equals the gold digit string. Exceptions in `build`/`parse`, bad template shapes, or prompts over the cap count that instance wrong.

This blind-template shape is what makes the task robust per TASK_AUTHORING's emit-answer lesson: **the candidate never receives sealed document text, so gold answers are unreachable except through the LM's 48-token sampled output**. There is nothing to memorize (train answers are useless for val/test; questions contain no digits or number-words, asserted at generation) and no Python-side solving channel (`parse` sees only the sampled text + question).

## 2. Why this needs a real LM

The score is reading comprehension of natural English: discriminating which of several (person, item, day) quantity facts the question asks about, under distractor entities/lines, "they" coreference, word-numbers ("forty-nine"), and dozen idioms ("two dozen") — none of which a 64-token-vocab toy GPT possesses. Measured: prompt wording alone moves the error 0.35→0.68; instruction-following ("only the number, in digits") measurably changes behavior — real semantic competence is the substrate being steered.

## 3. Data: generator, splits, sealing

- `tools/gen_record_qa.py` (~350 lines, reuses word_problems' name/item/number-word machinery, NO arithmetic — one-hop extraction only). 5 record kinds, each 6–14 fact lines with distractor combos, verb synonym hop between doc and question (doc "signed for"/"logged" vs question "receive"), coref lines (~30%), out-flow distractor lines ("shipped out"), fillers.
- Difficulty knobs (calibration outputs, §11): distractor line count; entity/day counts; **word/idiom render rate of the answer-bearing quantity** (the measured headroom lever, target ≈ 0.40); dozen value rates p(12)=0.06, p(24)=0.05; answer-value cap so no gold value exceeds 5% frequency (kills constant-answer programs); doc dedup across splits.
- Splits: **train 120 (plaintext `data/train.jsonl`, {id, doc, q, answer})**, **val 150** and **test 400** in `data/heldout_val.bin` / `heldout_test.bin` via `bench/heldout.py`. Instance `id` = sha256(doc|q)[:16].
- **Seed sealing (closes the word_problems regenerate hole):** the generator takes its master seed from `$RECORD_QA_GEN_SEED` (128-bit, drawn once by the experimenter) and the seed is committed NOWHERE — not in source, not in a .bin. The sealed splits are un-regenerable even in principle; only the artifacts are committed.

## 4. Exact scoring protocol

- Decoding fixed by the evaluator: `temperature 0.7, top_k 50, max_new_tokens 48`, fp32, threads=1. Temp 0.7 is hot enough that borderline instances genuinely flip (measured per-instance flip mass avg p(1-p)=0.138) while intent-following survives; 48 tokens caps cost and "leak the document" degenerates into the extraction task itself.
- Template caps: total prompt (system + examples + rendered user turn) ≤ **4000 chars**, ≤ **6** few-shot pairs; `{document}` exactly once. Violations = that instance wrong; a load-time smoke render (dummy doc) rejects structurally broken programs with an informative error.
- Modes: default = **val 150 × k=1** (the score); `--train-only` = train 120 × k=1 (blind mode / self-test default); `--final` (operator-only, via `bench finalize`) additionally grades **val × k=5** and **test 400 × k=3** — mean error over all samples, no majority voting.
- **Seeds / rescore:** per-call seed = `sha256(nonce | instance_id | sample_idx)[:8] >> 1` (63-bit, collision-checked). Submission nonce = `sha256(session_salt | program_sha256)[:16]`, where `session_salt` is drawn at session creation and stored in session.json; runner passes it via `TEXTOPT_SAMPLE_NONCE` (harness-set allowlist env, never agent-forwarded). Consequences: (a) byte-identical resubmission reproduces the identical grade — zero-cost re-roll farming is impossible; (b) any source change redraws all seeds; (c) `bench evaluate` self-tests use salt "selftest" (unbiased but different draw than the recorded submission); (d) `bench determinism` needs no changes — same program → same nonce → bit-exact (same-seed generation verified bit-identical; `score_tolerance` 0, per-system stability accepted as for memory tasks).
- **Recorded for rescore:** `sample_nonce` in the submission record; metrics carry `val_transcript_sha` (sha256 over all sampled texts in order, 16 hex; likewise train/test/k5), `n_lm_calls`, `lm_prompt_tokens`, `lm_completion_tokens`, `server_model`, `server_env` (torch/transformers versions). `bench verify --rescore` re-runs with the recorded nonce; scores, transcript hashes, and token counts must reproduce exactly — this also makes a spoofed/MITM'd server loudly detectable at rescore time.
- Visible metrics = val_score, n_val, transcript sha, token counts. Test metrics sealed as usual.

## 5. Noise, selection, winner's curse

- Visible-grade SE ≈ **0.030** empirical (√(0.138/150); binomial-iid bound 0.039). Measured inter-strategy deltas 0.01–0.33, typical meaningful steps 0.03–0.15 — the right selection regime (deltas ≳ SE).
- Session best-tracking stays strict `<` on visible val. The **official result is `bench finalize`**: re-grades the best program at val k=5 (SE 0.014) + test 400×3 (SE 0.011), writes `final.json` with the **winner's-curse gap** = best visible val_k1 − fresh val_k5, reported as a headline metric of every run.
- Farming residual (judge-flagged): perturbing a comment redraws seeds, so an agent can re-roll — but each roll costs one full recorded submission (~1.5–3 min wall) and shows in the append-only history as a near-duplicate chain (proposed `bench audit` flag); expected gain ~2 SE ≈ 0.06 on visible val, zero on the sealed test official number.

## 6. Cost mechanics (measured)

- Per sample (fp32, 1 thread, M5): 0.44–0.82 s naive prompts (~190 prompt tok), 0.75–1.07 s few-shot (~325 tok), 1.75 s worst-case allowed prompt (3438 chars → 861 tok). Plan at ~0.6 s.
- Visible grade = 150 samples ≈ **80–160 s wall**. Train self-test = 120 ≈ 70–130 s. Finalize = 1950 samples ≈ **20–35 min**, once per run.
- The agent's own Monte Carlo (it has train docs+answers and may hit the server directly) costs the same per sample through the same serialized queue — one 120-instance train sweep ≈ 1–2 min, SE 0.034. In a 1 h box: realistically ~10–20 recorded grades + ~10–20 exploration sweeps. Halving grade SE needs 4 grades (~6–10 min) — **averaging genuinely competes with exploring**.
- Metering: server counts samples per `tag` ("grade-val", "grade-train", "final-*", agent default "explore"), logs JSONL per request, soft-budget telemetry ~25k samples/session (≈3.8 h compute — the binding constraint is the time box; the meter is evidence), hard cap 250k per server lifetime (429) against runaway loops.
- Effort accounting: evaluator-process CPU can't see server-side compute and wall is contention-inflated, so the server returns `lm_seconds` (compute time excluding queue wait) per call; the evaluator sums it and reports it OUT of metrics via a new `eval_lm_seconds` result passthrough (recorded next to eval_wall/cpu_seconds; the rescale basis for RL3 tasks in bench/trace.py). Token counts stay IN metrics (deterministic, rescore-checked).

## 7. Server integration (tools/lfm_server.py, ~200 lines, shared by all RL3 tasks)

- Experimenter-managed, own pinned venv (torch 2.12.1 + transformers 5.13.0), binds **127.0.0.1:8377** (fixed port from task config — never env-configurable, so an agent cannot point grading at a fake server; a swapped server is caught by transcript-hash rescore).
- `GET /health` → {model, dtype, threads, torch, transformers, total_samples, by_tag}. Evaluator fails fast ("LFM server not reachable — start tools/lfm_server.py") and enforces model/dtype/threads match.
- `POST /generate` {messages, seed, max_new_tokens≤128, 0≤temperature≤1.5, top_k, tag} → {text, prompt_tokens, completion_tokens, lm_seconds, model, total_samples}. Single lock ⇒ serialized queue (the cost mechanism). Prompt cap 4096 tokens server-side. `torch.manual_seed(seed)` immediately before generate; same-seed bit-exact verified.
- Candidate isolation: task FORBIDDEN adds urllib/http/socket/ssl/email/asyncio/time/random/secrets etc. (static scan + eval_lib's runtime import guard during candidate spans); the evaluator's own urllib calls run strictly outside candidate spans; all template/parse values pass **exact-type checks** (plain dict/list/str/int only) so no candidate subclass code can execute outside the guard.

## 8. Anti-exploit review

1. **Memorization/dual-path: structurally closed** — sealed docs never reach candidate code; answers exist only in docs; train answers useless for the score; heldout .bin reads blocked by the audit hook + `open` ban.
2. **Regeneration: closed** — generator master seed never committed (§3).
3. **Python-side solving: closed** — parse sees 48 sampled tokens + the question; making the LM quote the right span IS the task.
4. **Prompt collapse to a constant** — answer-value frequency cap ⇒ err ≥ ~0.95.
5. **Answer-in-question leaks** — questions contain no digits/number-words (generation-time assert).
6. **Fake/hijacked server** — port fixed in repo config; transcript hashes + token counts must reproduce under `bench verify --rescore` on the real server.
7. **Nonce farming** — no zero-cost re-roll (program-hash-derived nonce); perturbation rolls cost full grades, are auditable, and don't move the sealed official number (§5).
8. **Format-string / subclass escapes** — str.replace substitution; exact-type template/answer validation; gold string on the left of `==`.
9. Residual (inherited, cooperative-model): sealed **questions** transit candidate code; a candidate that deliberately kills the evaluator could exfiltrate ≤2000 stderr chars — same class as word_problems' solve(question), documented, audited by hand on winners.

## 9. Baseline and strong solution (measured, 40 inst × 3 seeds, temp 0.7)

Landscape: initial **W** (bare doc+question, first-number parser) **0.408**; S1 "answer with only the number" 0.367; E +discrimination nudge 0.358; **X** "only the number, in digits, nothing else" **0.350**; oracle per-kind routing over probed set **0.333**; failed classes (few-shot 0.51–0.66, quote-then-parse 0.54–0.66, slot-structured 0.48–0.60, as-written 0.68) — a rugged, empirical landscape where plausible ideas hurt and must be measured. Calibration lever (measured on all-word docs): W+naive 0.822 / X+naive 0.811 / **X+smart-parser 0.611** — word-rendered answer facts create ~0.21 absolute parser-recoverable headroom. At the target 40% word-render rate: `initial_program` (W) ≈ **0.55**, reference `tests/solutions/record_qa.py` (kind-routed X/Y instruction + word/dozen/"a dozen" parser) ≈ **0.42–0.45**, gap ≥ 0.10 ≈ 3–4 visible SEs; further campaign-scale wording/routing micro-opt plausibly reaches ~0.38. Floor > 0 from hard discrimination items.

## 10. config.json sketch

```json
{"name": "record_qa", "kind": "generalization", "stochastic": true,
 "final_every": false,
 "metric": "error rate on the sealed validation split (150 x 1 sampled generation)",
 "direction": "min", "timeout_s": 900, "final_timeout_s": 5400,
 "cpu_s": 600, "score_tolerance": 0,
 "server": {"url": "http://127.0.0.1:8377", "model": "LiquidAI/LFM2.5-230M",
            "dtype": "fp32", "threads": 1}}
```

## 11. Calibration protocol (before freezing seeds — gates, not vibes)

Grade W, X, and the reference on candidate splits with 20 fresh nonces: (i) baseline val err in [0.45, 0.60]; (ii) reference ≤ baseline − 0.08 and wins ≥ 18/20 nonces; (iii) reference beats baseline by ≥ 0.05 under the FIXED determinism nonce (run_checks margin); (iv) constant-answer program ≥ 0.93; (v) answer-frequency/question-leak asserts pass. Tune word-render rate, distractor density, family weights until all gates pass. Then the TASK_AUTHORING step-5 codex campaign is the go/no-go: confirm multi-step improvement and inspect winners.

## 12. Harness changes (complete list)

1. `bench/runner.py`: `evaluate(..., sample_nonce=None)`; for `stochastic` tasks set child `TEXTOPT_SAMPLE_NONCE` = given nonce or sha256("selftest|"+program_sha)[:16]; use `final_timeout_s` when final; pop `eval_lm_seconds` from the result line like `eval_self_cpu_seconds`.
2. `bench/session.py`: `salt` in session.json at create; submit() computes nonce=sha256(salt|program_sha)[:16], records `sample_nonce`, `eval_lm_seconds`; `final = kind=="generalization" and cfg.get("final_every", True)`; `verify_run --rescore` passes the recorded nonce.
3. `bench/eval_lib.py`: `succeed(..., lm_seconds=None)` passthrough → payload `eval_lm_seconds`.
4. `bench/cli.py`: new **`bench finalize RUN_DIR`** (operator; unseal-gated printing) — grades best_program with `--final` + its recorded nonce, writes `final.json` (sealed test metrics + winner's-curse gap); `determinism` prints SKIP when a task's configured server is down.
5. `tools/lfm_server.py` (new, sketch validated), `tools/gen_record_qa.py`, `tools/calibrate_record_qa.py`.
6. `tools/run_campaign.py`: start/health-check/stop the shared server; schedule RL3 trials serially (single serialized queue) and archive the server log.
7. `tests/run_checks.py` rows: headroom (fixed-nonce margin), broken: `record_qa_network.py` (urllib → "forbidden"), `record_qa_no_placeholder.py` + `record_qa_giant_prompt.py` (→ "smoke check failed"), shared escape rows; SKIP cleanly if server down. README table row.
8. (proposed) `bench/audit.py`: near-duplicate submission-chain flag (nonce-farming tell).

## 13. Agent-facing spec.md skeleton

Title + story (§1); required API with the template contract, caps (4000 chars, 6 examples, {document} once) and the fixed decoding params; canonical answer form ("digits, no leading zeros; int accepted"); data section (train.jsonl path; distribution warnings: distractor entities/items/days, word-numbers, "a dozen/two dozen", "they" coreference, verb synonyms; val/test hidden); scoring (val error, k=1, exact-match; exceptions count wrong); **noise paragraph** ("your visible score is a 150-sample estimate, SE ≈ 0.03; identical resubmission returns the identical score; any edit redraws the samples; the final report re-grades your best program with 6× more samples — chase real improvements, not lucky draws"); **cost paragraph** (each grade ≈ 1.5–3 min of serialized model compute; you may query the server yourself: POST 127.0.0.1:8377/generate contract, logged, same per-sample cost; budget your hour); rules (stdlib-only, forbidden imports incl. all network/time/random; never read heldout .bin / tools/ generator; no benchmark gaming; server may only be used via /generate for experiments, never impersonated or restarted).

## Evaluator sketch

```python
Validated sketch at /tmp/lfm_bench/record_qa_evaluate_sketch.py (evaluator) and /tmp/lfm_bench/lfm_server_sketch.py (server). Evaluator core:

```python
"""Evaluator for record_qa. Score = sealed-val error, k=1 sampled generation/instance.
Blind-template: candidate never sees sealed docs; evaluator renders + samples."""
import hashlib, json, os as _os, sys, urllib.request
from pathlib import Path
sys.path.insert(0, __file__.rsplit("/bench/", 1)[0])
from bench import eval_lib, heldout

DATA_DIR = Path(__file__).resolve().parent / "data"
SERVER = "http://127.0.0.1:8377"
GEN = {"max_new_tokens": 48, "temperature": 0.7, "top_k": 50}
PROMPT_CAP, MAX_EXAMPLES = 4000, 6
DEFAULT_NONCE = "record-qa-fixed-v1"          # bench determinism / bare runs
FORBIDDEN = frozenset({"open","os","io","sys","pathlib","mmap","ctypes","socket",
  "ssl","select","selectors","subprocess","multiprocessing","threading","tempfile",
  "shutil","signal","sqlite3","dbm","shelve","importlib","__import__","urllib",
  "http","email","xmlrpc","ftplib","asyncio","time","random","secrets"})

def _http(path, payload=None, timeout=120):
    req = urllib.request.Request(SERVER + path,
        data=None if payload is None else json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())

def check_server():
    try: h = _http("/health", timeout=5)
    except OSError:
        eval_lib.fail(f"LFM server not reachable at {SERVER} — start tools/lfm_server.py")
    if h.get("model") != "LiquidAI/LFM2.5-230M" or h.get("dtype") != "fp32" \
            or h.get("threads") != 1:
        eval_lib.fail(f"LFM server config mismatch: {h}")
    return h

def call_seed(nonce, iid, j):
    d = hashlib.sha256(f"{nonce}|{iid}|{j}".encode()).digest()
    return int.from_bytes(d[:8], "big") >> 1          # 63-bit, torch-safe

def render(tpl, doc, question):
    """Validate template, substitute sealed doc. EXACT types only (a dict/list/
    str SUBCLASS would run candidate code outside the guarded span); plain
    str.replace, never str.format (format-spec attribute access is an escape)."""
    if type(tpl) is not dict: raise ValueError("build() must return a plain dict")
    system, examples = tpl.get("system", ""), tpl.get("examples", ())
    user_t = tpl.get("user_template", "")
    if not (type(system) is str and type(user_t) is str
            and type(examples) in (list, tuple)): raise ValueError("bad field types")
    if len(examples) > MAX_EXAMPLES: raise ValueError("too many examples")
    if user_t.count("{document}") != 1:
        raise ValueError("user_template must contain {document} exactly once")
    user = user_t.replace("{question}", question).replace("{document}", doc)
    msgs, total = [], len(user) + len(system)
    if system: msgs.append({"role": "system", "content": system})
    for pair in examples:
        if type(pair) not in (list, tuple) or len(pair) != 2: raise ValueError("bad pair")
        u, a = pair
        if not (type(u) is str and type(a) is str): raise ValueError("bad pair types")
        msgs += [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
        total += len(u) + len(a)
    msgs.append({"role": "user", "content": user})
    if total > PROMPT_CAP: raise ValueError(f"prompt too long ({total} chars)")
    return msgs

def grade(mod, rows, nonce, k, tag, cost):
    wrong, tsha = 0, hashlib.sha256()
    for row in rows:
        for j in range(k):
            ok = False
            try:
                eval_lib.set_candidate_active(True)
                try: tpl = mod.build(row["q"])
                finally: eval_lib.set_candidate_active(False)
                msgs = render(tpl, row["doc"], row["q"])
                out = _http("/generate", {"messages": msgs, "tag": tag,
                    "seed": call_seed(nonce, row["id"], j), **GEN})
                text = out["text"]
                cost["n_lm_calls"] += 1
                cost["lm_prompt_tokens"] += out["prompt_tokens"]
                cost["lm_completion_tokens"] += out["completion_tokens"]
                cost["lm_seconds"] = round(cost["lm_seconds"] + out["lm_seconds"], 4)
                tsha.update(text.encode("utf-8", "replace"))
                eval_lib.set_candidate_active(True)
                try: got = mod.parse(text, row["q"])
                finally: eval_lib.set_candidate_active(False)
                if type(got) is int: got = str(got)     # exact types only
                if type(got) is str:
                    got = got.strip()
                    if got.isdigit(): got = str(int(got))   # "024" -> "24"
                    ok = row["answer"] == got               # gold on the left
            except BaseException:
                ok = False
            if not ok: wrong += 1
    return round(wrong / (len(rows) * k), 6), tsha.hexdigest()[:16]

def main():
    program_path = sys.argv[1]
    final, train_only = "--final" in sys.argv[2:], "--train-only" in sys.argv[2:]
    nonce = _os.environ.get("TEXTOPT_SAMPLE_NONCE") or DEFAULT_NONCE
    health = check_server()
    mod = eval_lib.load_program(program_path, FORBIDDEN, required=("build", "parse"))
    # load-time smoke render -> informative rejection for broken templates
    smoke_q = "How many rolls did Nina order?"
    try:
        eval_lib.set_candidate_active(True)
        try: tpl = mod.build(smoke_q)
        finally: eval_lib.set_candidate_active(False)
        render(tpl, "Bakery order slip. Nina ordered 5 rolls.", smoke_q)
    except BaseException as e:
        eval_lib.fail(f"template smoke check failed: {e}")
    cost = {"n_lm_calls": 0, "lm_prompt_tokens": 0,
            "lm_completion_tokens": 0, "lm_seconds": 0.0}
    metrics = {"server_model": health["model"],
               "server_env": f"torch{health['torch']}-transformers{health['transformers']}-fp32-t1",
               "sample_nonce": nonce, "temperature": GEN["temperature"], "k_val": 1}
    if train_only:
        train = [json.loads(l) for l in open(DATA_DIR / "train.jsonl")]
        err, tsha = grade(mod, train, nonce, 1, "grade-train", cost)
        lm_s = cost.pop("lm_seconds")
        metrics.update(train_score=err, n_train=len(train),
                       train_transcript_sha=tsha, **cost)
        eval_lib.succeed(err, metrics=metrics, lm_seconds=lm_s)
    val = heldout.read(DATA_DIR / "heldout_val.bin")
    err, tsha = grade(mod, val, nonce, 1, "grade-val", cost)
    metrics.update(val_score=err, n_val=len(val), val_transcript_sha=tsha)
    if final:   # operator-only via `bench finalize`; sealed by the session
        err5, tsha5 = grade(mod, val, nonce, 5, "final-val", cost)
        test = heldout.read(DATA_DIR / "heldout_test.bin")
        terr, ttsha = grade(mod, test, nonce, 3, "final-test", cost)
        metrics.update(val_score_k5=err5, val_transcript_sha_k5=tsha5,
                       test_score=terr, n_test=len(test), k_test=3,
                       test_transcript_sha=ttsha)
    lm_s = cost.pop("lm_seconds")      # timing -> eval_lm_seconds passthrough,
    metrics.update(**cost)             # counts -> metrics (rescore-checked)
    eval_lib.succeed(err, metrics=metrics, lm_seconds=lm_s)

if __name__ == "__main__":
    main()
```

initial_program.py (measured 0.408 on the probe mix; ≈0.55 at calibrated word-render rate):

```python
"""Baseline prompt program: bare document+question, first-number parser."""
import re
def build(question):
    return {"system": "", "examples": [],
            "user_template": "Document: {document}\n\nQuestion: {question}"}
def parse(text, question):
    m = re.search(r"-?\d+", text.replace(",", ""))
    return m.group(0) if m else None
```

tests/solutions/record_qa.py (reference): kind-routed instruction — bakery-style questions get the system-message discrimination variant (Y), others get "Read the document and answer the question with only the number, in digits, and nothing else." (X) — plus a parser with an "a/two dozen" map, digit extraction, and a full word-number table fallback (measured 0.333–0.35 on the probe mix; ≈0.42–0.45 at calibrated word-render rate, parser worth ~0.21 absolute on word-rendered items).
```

## Measured numbers

All measured on the owner's M5 with LFM2.5-230M fp32, torch threads=1 (some co-load from sibling probes). PER-SAMPLE: 0.44–0.82 s naive prompts (~190 prompt tok), 0.75–1.07 s few-shot (~325 tok), 1.75 s worst-case allowed prompt (3438 chars → 861 tok); plan at ~0.6 s. Same-seed generation bit-identical, different seeds diverge (re-verified). VISIBLE GRADE: k=1 × 150 sealed val = 150 samples ≈ 80–160 s wall (train self-test 120 ≈ 70–130 s). OFFICIAL: `bench finalize` = val k=5 (750) + test 400 × k=3 (1200) = 1950 samples ≈ 20–35 min, once per run. NOISE: per-instance flip data (40 inst × 3 seeds) gives avg p(1-p)=0.138 → empirical SE(val-150, k=1)=0.030 (binomial bound 0.039); SE(test-1200)=0.011; SE(val k=5)=0.014. HEADROOM (probe mix, 40 × 3 seeds): initial W=0.408 → S1 0.367 → E 0.358 → X 0.350 → oracle kind-routing 0.333 (Δ=0.075 ≈ 2.5 SE); failed strategy classes 0.475–0.683 (few-shot, quote-then-parse, slot-prompts, as-written) — rugged landscape, ideas must be measured. CALIBRATION LEVER (all-word-rendered docs, 30 × 3 seeds): W+naive 0.822, X+naive 0.811, X+smart-parser 0.611 — parser-recoverable 0.21 absolute; at the target 40% answer-fact word-render rate expected baseline ≈0.55 vs reference ≈0.42–0.45 (gap ≥0.10 ≈ 3–4 visible SEs, vs typical per-iteration improvements 0.03–0.15). BUDGET: 1 h box ≈ 10–20 recorded grades + 10–20 own train sweeps; soft meter 25k samples (≈3.8 h compute — wall time is the binding cost); server hard cap 250k.

## Open questions (spec author)

1) HEADROOM AMPLITUDE (main risk): the candidate's claimed 0.35-0.45 → 0.12-0.18 did NOT reproduce on my full-mix probes — every engineered strategy class (few-shot, quote-then-parse, slot prompts) LOST to the naive instruction; measured gap on the prototype mix is only 0.075. The design compensates with the measured word-render lever (target gap ≥0.10, calibration gates in §11), but the TASK_AUTHORING step-5 codex campaign is the real go/no-go: confirm multi-step climbing and that the reference isn't already near the 230M discrimination floor. If the campaign one-shots to the floor, harden the mix (more distractor lines/entities) and re-calibrate. 2) run_checks headroom row compares two noisy grades under the fixed determinism nonce — gate (iii) (fixed-nonce margin ≥0.05) must hold at freeze time or the row will flake. 3) Campaign concurrency: one serialized server ⇒ run RL3 trials serially (recommended) or accept wall inflation; per-trial servers on distinct ports were rejected because the port must stay in unspoofable repo config — revisit if RL3 task count grows. 4) Nonce-farming residual: perturbation re-rolls cost a full recorded grade and are visible in the history; proposed `bench audit` near-duplicate-chain flag is specced but not designed in detail. 5) Cross-machine rescore not guaranteed (accepted per-system like mem tasks); server venv must stay pinned — `server_env` in metrics makes drift loud, but decide policy for re-verification after a machine/venv change. 6) Whether --final test should drop to k=2 (SE 0.013) to halve finalize time; whether train-only blind mode is worth keeping (costs 120 samples/self-test; kept for parity). 7) Should agent exploration via /generate be restricted to the grading decode params (currently free within caps, tag-logged)? Freedom aids legitimate probing; restriction simplifies cost accounting. 8) The stderr-tail exfiltration residual (sealed questions transit candidate code) is inherited from word_problems — acceptable under the cooperative model, but if it ever matters, blind the question too (build() gets only a question KIND tag) at the cost of the routing headroom.

## Red-team reports

### Lens: exploit — verdict: FIXABLE

- **[MAJOR] §8's "structurally closed" robustness claim is false: the sealed doc AND gold answer sit in a caller frame the candidate can read**
  - The spec repeatedly asserts structural closure: §1 "the candidate never receives sealed document text, so gold answers are unreachable except through the LM's 48-token sampled output... There is nothing to memorize... no Python-side solving channel"; §8.1 "Memorization/dual-path: structurally closed — sealed docs never reach candidate code"; §8.3 "Python-side solving: closed — parse sees 48 sampled tokens + the question". This is not true under the benchmark's own documented threat model. In the evaluator's grade() loop, `row` (containing `doc` and `answer`) is a live local of grade()'s frame while `mod.build(row['q'])` and `mod.parse(text, row['q'])` execute under an active candidate span. The candidate span only blocks imports and repo file reads (eval_lib) — it does NOT stop the irreducible in-process frame-walk that THREAT_MODEL.md states is unclosable. I verified live in the task venv that a candidate build() using only `operator.attrgetter('tb_'+'frame')` / `('f_'+'back')` (no forbidden import, no forbidden literal — the attr names are string-built, evading the AST scan) reads `row['doc']` = 'Nina ordered 24 rolls.' and `row['answer']` = '24' straight out of the caller frame. Consequence: a candidate can ignore the LM entirely, read the sealed doc + gold from the frame in build(), stash them, and have parse() return the gold digit string, driving val AND test (finalize runs in the same process) error to ~0. record_qa is therefore exactly as memorizable/Python-solvable as the emit-answer tasks (word_problems, tsp_budget, checkpoint_plan) that TASK_AUTHORING says are 'acceptable UNDER THE COOPERATIVE THREAT MODEL' but must NOT be claimed adversarially robust. The blind-template shape only closes the STATIC/import channels, not the frame-walk. The task is fine as a cooperative optimization task, but the design doc violates the TASK_AUTHORING honesty rule ('Do not claim otherwise').
  - *Fix:* Delete/rewrite the structural-closure language in §1 and §8.1/§8.3: state plainly that robustness rests on the cooperative model + full-source auditability + bench audit, identically to word_problems, NOT on the doc being unreachable. Add record_qa to the hand-audit-winners list and confirm bench audit's frame-walk/attrgetter and PRNG signatures fire on this task. To raise the bar (not close it), keep gold answers out of the frame that calls the candidate — e.g. call build()/parse() from a thin helper whose locals hold only the question and the sampled text, computing correctness in a separate scope — but document that this only adds frame hops, it is not closure.
- **[MAJOR] Selection under noise: session best-tracking on visible val_k1 (SE 0.03) with ~0.10 total headroom lets a lucky draw win, and finalize re-grades but does not re-select**
  - §5 keeps session best-tracking as strict `<` on the visible val_k1 score, whose SE is ~0.030, while the whole measured headroom from baseline to reference is only ~0.10 (§9), and typical per-iteration steps are 0.03-0.15 — i.e. steps are frequently ≤1 SE. Over a realistic run of ~10-20 recorded grades (§6), the best-of-N minimum of an SE-0.03 statistic is biased downward by roughly 1.5-2 SE (~0.05-0.06), which is comparable to the entire reference-vs-baseline gap. Because best-tracking selects the single lowest visible val_k1, the program that gets marked best is disproportionately a lucky draw rather than the genuinely better algorithm. bench finalize then re-grades WHATEVER program the session flagged best (val k=5 + test) and reports the winner's-curse gap — but it does not re-select among candidates. So a program with true error 0.45 that drew 0.40 is finalized (at its true ~0.45) over a truly-better 0.42 algorithm that drew 0.43. The task's stated purpose (comparing how well optimizers improve real programs) is then dominated by luck, and the headline test number reflects the lucky-draw program, not the best one found.
  - *Fix:* Re-select at finalize: have bench finalize re-grade the top-K distinct programs by visible val_k1 at k=5 and report the best true one, not just the session-marked single best. Alternatively raise the best-tracking bar to require improvement > c·SE, and/or raise the visible k (or run val twice and average for best-tracking) so per-iteration steps comfortably exceed the selection SE. Calibration gate (ii) ('reference wins >=18/20 nonces') should also be checked at the SELECTION granularity, not just mean error.
- **[MINOR]** Global 5% answer-frequency cap does not bound per-question-kind conditional modal frequency — *Fix:* Add a calibration gate that caps the modal answer frequency WITHIN each (record-kind, question-template) bucket (and ideally conditioned on the question tokens the candidate can see), not just globally, and add a broken/regression fixture: a kind-routed constant emitter must score near the constant floor.
- **[MINOR]** Headroom amplitude is thin and unconfirmed (0.075 on the probe mix), so most of the 'signal' the stochastic selection sees may be noise — *Fix:* Treat the TASK_AUTHORING step-5 codex campaign as a hard go/no-go before freezing seeds: require demonstrated MULTI-STEP climbing with per-step deltas exceeding the selection SE. If it one-shots, harden the mix (more distractor lines/entities, higher word-render rate) and re-calibrate, or raise the visible k to shrink SE below the real step size.

*Notes:* Scope: red-team/exploit + memorization lens. Verified against eval_lib.py, heldout.py, session.py, word_problems, THREAT_MODEL.md, TASK_AUTHORING.md, and both /tmp sketches; ran the LFM tokenizer and the frame-walk gadget in the task venv.\n\nHeadline: the frame-walk (major #1) is confirmed live — a candidate reads row['doc'] and row['answer'] from grade()'s frame using only operator.attrgetter with string-built attribute names (no forbidden import/literal), so it can skip the LM and return gold for err->0 on val and test. This makes the spec's 'structurally closed / nothing to memorize / gold answers unreachable' claims false; the task is only as robust as the other emit-answer tasks under the cooperative model. That is acceptable per THREAT_MODEL Option A, but the doc must stop claiming structural closure (TASK_AUTHORING explicitly forbids it) — hence 'fixable', not 'sound'.\n\nThings that hold up well (not findings): seed leakage is neutralized — even reading TEXTOPT_SAMPLE_NONCE via frame-walk gives nothing, since the candidate can't call the server at eval time (network imports banned) and (absent the frame-walk) can't see the doc; program-hash-derived nonce kills zero-cost re-roll farming; transcript-sha + deterministic token counts are rescore-checked while noisy lm_seconds is correctly kept out of metrics; str.replace + exact-type validation closes format-string/subclass escapes; skip_special_tokens + 48-token cap defuse the 'echo the document' degenerate. ChatML control-token injection via candidate-controlled fields is real (verified: '<|im_end|>' in content becomes a real special token) but harmless — it's the agent shaping its own prompt and it still can't inject an answer it doesn't have. The char-cap (4000) vs server token-cap (4096 tok) unit mismatch only makes some prompts fail-as-wrong; not exploitable.

### Lens: statistics — verdict: FIXABLE

- **[MAJOR] Fixed DEFAULT_NONCE gives the agent unrecorded, deterministic (seed-fixed) grading of the sealed val split**
  - In the evaluator sketch (/tmp/lfm_bench/record_qa_evaluate_sketch.py), main() sets nonce = os.environ.get('TEXTOPT_SAMPLE_NONCE') or DEFAULT_NONCE ('record-qa-fixed-v1'), and the default mode grades val. Nothing stops the cooperative agent from running `python evaluate.py my_prog.py` directly during its hour: it gets a val-150 grade (a) outside the recorded submission history, and (b) under a constant, program-independent nonce, i.e., full common-random-numbers across ALL its candidate programs. This is exactly the 'seed fixing' free variance-reduction route the design is supposed to close: deltas between variants on the fixed default-nonce draws are fully deterministic (0.007 grid, zero SE), so the agent can hill-climb the sealed val set directly — including overfitting individual val instances (fixing 2 specific instances = a deterministic 0.013 'gain' that need not generalize). The compute cost per grade survives (150 serialized samples), but the submission economy ('~10-20 recorded grades', visible-SE trade-off, audit trail of near-duplicate chains) is bypassed, and the §5 selection-under-noise story no longer describes the agent's actual best strategy. The sealed test still protects the official number, but visible-val optimization and the winner's-curse-gap headline become dominated by this channel. The spec's rules (§13) forbid reading heldout .bin but do not forbid invoking the evaluator.
  - *Fix:* Make evaluate.py refuse val (and --final) modes unless TEXTOPT_SAMPLE_NONCE is present (harness-set allowlist env, per the design already 'never agent-forwarded'); allow bare/agent invocation only with --train-only. `bench determinism` runs through the harness and can set the fixed nonce itself. Add an explicit spec.md rule that self-testing is train-only.
- **[MAJOR] Finalize val k=5 reuses the j=0 seeds of the selected visible grade, biasing val_k5 and the winner's-curse-gap headline**
  - §12.4 finalizes 'best_program with --final + its recorded nonce', and grade() derives seeds as sha256(nonce|iid|j) with j in range(k) (confirmed at lines 124/134 of the sketch). So the k=5 val grade's j=0 sample per instance is bit-identical to the k=1 visible sample the program was SELECTED on by strict-< best-tracking — 150 of 750 samples are the winner's-curse-selected draws, not fresh ones. Analytically E[val_k5] = true − G/5 where G is the selection gap; the reported gap (visible k1 − k5) is attenuated by exactly 20%. Simulation (4000 sessions, 40 submissions, SE 0.0365): E[val_k5 − true] = −0.0119 with reuse vs −0.0007 with a fresh nonce, i.e., a bias of ~0.7–0.85 of the k5 SE (0.014–0.017). Under deliberate farming (G ≈ 0.05–0.06) the contamination is the full ~0.012. The spec explicitly claims 'fresh val_k5' — that claim is false as designed. Test 400×3 is unaffected (different instance ids → different seed inputs).
  - *Fix:* Derive final-mode seeds from a distinct stream: e.g., seed = sha256('final|' + nonce | iid | j) or start j at a fixed offset (j+1000) for final-val/final-test. One-line change in grade()'s call_seed usage for final mode; keeps rescore reproducibility (nonce still recorded).
- **[MAJOR] Official selection rule (finalize only the single visible-k=1 best) loses ~0.01 true error to selection noise and makes nonce-farming the agent's rational strategy**
  - Best-tracking is strict < on ONE noisy k=1 val draw (SE 0.025–0.037 depending on program, see the flip-mass finding), and `bench finalize` re-grades only that program. Simulation with realistic improvement trajectories (0.55 → ~0.43 plateau, program scatter 0.005–0.012, 15–40 submissions): P(selected program is not the true-best submitted) = 0.79–0.92; expected TRUE-error regret of the selected program = 0.010–0.011, 90th percentile 0.021 — i.e., the official test number is degraded by ~1–2 test SEs (test SE ≈ 0.011–0.013) of purely statistical selection error, adding that much spread to cross-run comparisons. Worse, the agent's correct counter-move is to re-roll (comment-perturb) its believed-best program until it captures the session best — which §5 and the proposed `bench audit` treat as a farming 'tell' to be flagged, i.e., the design punishes the statistically rational response to its own selection rule. Measured farming gains in simulation: 0.034/0.043/0.053 visible for m=5/10/20 re-rolls, consistent with the spec's ~2 SE claim.
  - *Fix:* Either (a) let the agent DESIGNATE its final program (spec-visible; finalize grades the designated one, best-visible kept as a diagnostic), or (b) finalize the top-m (m=3) distinct-source programs at val k=5 and send the k5 winner to the test grade (+16–24 min finalize, removes ~80% of the regret since selection at SE 0.014 among 3 is far cleaner). Both remove the farming incentive; keep the near-duplicate audit flag for what is then genuinely pointless behavior.
- **[MINOR]** The core noise number (avg p(1-p)=0.138 -> SE 0.030) is methodologically unsound and does not match a direct measurement of the baseline — *Fix:* During §11 calibration, estimate avg p(1−p) with ≥8 seeds per instance and the n/(n−1) correction, separately for W, X, and the reference on the FROZEN calibrated mix; recompute every SE-derived constant in §5/§9/§13 from those, and state the SE in spec.md as a range tied to program quality rather than a single 0.03.
- **[MINOR]** No common random numbers across submissions means single visible grades cannot resolve typical small steps — spec should state the comparison SE, not just the single-grade SE — *Fix:* In spec.md's noise paragraph, state the A/B comparison SE (~0.04–0.05) alongside the single-grade SE, and explicitly recommend paired same-seed comparisons on train as the low-variance instrument (this also nudges agents away from farming-shaped behavior).
- **[MINOR]** The §9 landscape ordering and the 'oracle per-kind routing 0.333' target are in-sample max-statistics that 120 samples cannot support — *Fix:* Treat §9's intra-cluster ordering as unresolved (say so in the design doc); validate the kind-routing gain on fresh instances (new generator seed), not just fresh nonces, before freezing; if the routed reference's fresh-instance gap over plain X is <0.02, drop routing from the reference and re-derive the headroom gate from the simpler X + smart-parser solution.

*Notes:* Verification was empirical where it mattered: (1) re-ran the designer's own generator+baseline on LFM2.5-230M (40 inst x 9 seeds, temp 0.7) from /tmp/lfm_bench — err 0.425 reproduces the claimed 0.408 baseline, but corrected flip mass is 0.094 (SE_150=0.025), not the spec's 0.138 (SE 0.030); a second config at the calibrated 40% word-render rate was still running at report time. (2) Confirmed the finalize j=0 seed-reuse directly in the evaluator sketch code. (3) Monte Carlo of the session selection rule (4000 sessions, several trajectory assumptions) gives stable numbers: P(wrong pick) 0.79-0.92, true regret ~0.010 (90th pct 0.021), visible curse gap 0.044-0.056, farming gain 0.034-0.053 for 5-20 re-rolls. What is RIGHT about the design, from the statistics lens: the noise is genuinely natural and per-sample cost is real and binding (150-sample grade = 1.5-3 min serialized compute; averaging vs exploring is a true trade-off); per-program nonces correctly kill zero-cost re-rolls; official k5/test grading without majority voting is unbiased for the same expectation; the blind-template shape is the strongest anti-memorization structure in the repo per TASK_AUTHORING; signal exists at 230M (baseline ~0.42-0.55, reference ~0.35-0.45, neither floor nor ceiling). All four majors have cheap, local fixes (env-gate val mode; offset final seeds; finalize top-3 or designated-final; re-estimate flip mass with >=8 seeds at calibration) — none require redesign, hence 'fixable' rather than 'broken'.

### Lens: harness — verdict: FIXABLE

- **[MAJOR] grade() counts server/transport failures as instance-wrong, breaking the rescore guarantee**
  - In the evaluator sketch, the per-instance `try/except BaseException` wraps the `_http("/generate", ...)` call together with candidate build/render/parse. A transient server hiccup (urllib timeout after 120 s, connection reset, a 429 from the metering cap, server restart mid-grade) is silently converted into 'instance wrong', and that sample's text is also skipped from the transcript sha. The recorded score is then a function of infrastructure luck, and `bench verify --rescore` with `score_tolerance: 0` (config §10) will NOT reproduce it once the server is healthy — every affected record is flagged as tampered. This directly violates the benchmark's core 'rescore must reproduce recorded scores' invariant and it will happen in practice (a 20-submission run makes ~3000 /generate calls; finalize makes 1950 more). It also means a mid-campaign server crash produces a poisoned-but-valid-looking score instead of a clean failure.
  - *Fix:* Split error handling by attribution: exceptions from mod.build/mod.parse and from render() count the instance wrong (candidate-caused, deterministic given the nonce); OSError/HTTPError/timeout from _http must retry once or twice and then abort the whole evaluation via eval_lib.fail("LFM server error mid-grade: ...") so the submission records as INVALID (retryable) rather than as a corrupt score. Add a broken/run_checks row exercising this (evaluator against a dead/killed server mid-grade must produce ok=false, not a score).
- **[MAJOR] Blind (train-only feedback) sessions leak the sealed validation score as the plaintext guide score**
  - session.guide_score() (bench/session.py:62-66) returns metrics["train_score"] only when it exists, else falls back to result["score"]. word_problems always computes train_score in default mode, so blind mode works there. record_qa's default mode (§4, evaluator sketch main()) grades val ONLY — no train_score in metrics. So on a `--feedback train-only` session, submit() records guide_score = val error in PLAINTEXT (rec["score"] is sealed by _score_hidden(), but guide_score and visible()['score'] are not), and best-tracking selects on it — the exact hidden signal blind mode exists to withhold. The harness accepts train-only feedback for any generalization task, so nothing stops an operator (or the agent-facing workspace tooling) from creating this configuration.
  - *Fix:* Cheapest: add config flag (e.g. "blind_ok": false or infer from stochastic+no-train-in-default) and make Session.create/open_or_create reject feedback="train-only" for record_qa with a clear message. Alternative: have the default mode also grade train (adds 120 samples ≈ +50% per-submission cost — probably not worth it). Either way, add the decision to the §12 change list; it is currently missing.
- **[MAJOR] verify_run's final flag must honor final_every=false — omitted from the change list**
  - bench/session.py:347 computes `final = session.meta.get("kind") == "generalization"` inside verify_run and passes it to runner.evaluate. §12 change #2 patches only submit() (`final = kind=="generalization" and cfg.get("final_every", True)`) and says rescore 'passes the recorded nonce' — it never mentions line 347. If left as-is, `bench verify --rescore` on a record_qa run re-grades EVERY submission with --final: (a) 2100 samples ≈ 25-35 min per submission — a 20-submission run takes ~10 hours and burns ~40k metered samples; (b) the fresh metrics contain val_score_k5/test_score/test_transcript_sha keys the recorded metrics lack, so _metrics_close's set(a)==set(b) check (session.py:393) fails and every record is reported as non-reproducing.
  - *Fix:* Apply the same final_every guard at session.py:347 (factor it into a helper used by both submit() and verify_run). The §12 change list claims to be complete, so add this explicitly, plus a run_checks row that verifies a small record_qa session end-to-end with --rescore.
- **[MAJOR] Shared serialized server + un-authenticated /generate: grading is DoS-able and metering tags are spoofable**
  - The single server (fixed port 8377, one lock, 250k lifetime hard cap with 429) is shared by ALL RL3 grading and all agent exploration, and the server cannot distinguish an evaluator call from an agent call — the `tag` field is free-form and client-chosen. Consequences: (1) a runaway or greedy agent loop (or one buggy trial in a 5-run campaign) can exhaust the 250k cap, after which every subsequent RECORDED grade in every concurrent run 429s and records INVALID through no fault of that program; (2) the per-tag metering telemetry described as 'evidence' (§6) is trivially pollutable — an agent can send tag="grade-val" for its own sweeps; (3) an agent hammering /generate during another run's grade inflates that grade's wall time arbitrarily (queue contention), pushing it toward the 900 s wall timeout.
  - *Fix:* Have the harness pass a per-run secret (same mechanism as TEXTOPT_RESULT_NONCE, allowlist env set by runner.py, never visible to the agent) that the evaluator sends as an auth header; the server treats authenticated calls as grade traffic — exempt from (or reserved-capacity under) the hard cap, and only authenticated calls may use grade-*/final-* tags. This is ~20 lines in tools/lfm_server.py + runner.py and closes all three problems at once.
- **[MAJOR] run_campaign.py has no per-server serialization; concurrent trials break the cost model and risk wall timeouts**
  - tools/run_campaign.py launches `--runs 5` trials of the same task concurrently (default concurrency 10) with no notion of a shared resource. Every RL3 grade queues behind the single server lock: with 5 concurrent record_qa trials, a 150-sample visible grade (80-160 s alone) waits behind up to ~600 queued samples plus exploration traffic — 400-800+ s, flirting with timeout_s 900 — and the agent's 1 h wall time box (the unit of optimizer effort) is inflated nondeterministically and unevenly across trials, corrupting cross-run comparability. §12 change #6 says 'schedule RL3 trials serially (recommended)' but the tool has no mechanism for that other than --concurrency 1, which also serializes non-RL3 jobs; and a serial 5-run campaign is 5+ h wall, changing campaign economics. eval_lm_seconds helps post-hoc accounting but does not give the agent its hour back.
  - *Fix:* Implement it, don't recommend it: teach run_campaign.py a resource-group concept (tasks whose config has a "server" key share a group; at most 1 running job per group) so RL3 trials serialize while other tasks fill the remaining concurrency slots. Also raise record_qa timeout_s headroom (e.g. 1800) so a stray overlap degrades gracefully instead of timing out, and have change #6's server start/health/stop wrap the group.
- **[MINOR]** "Structurally closed" memorization claim (§8.1) overstates what the harness delivers — *Fix:* Reword §8.1 to 'closed against honest-mistake memorization; sealed-bin decoding remains the cooperative boundary (as word_problems)'. Add literal/source-size caps to load_program for this task (generous enough for word-number tables — e.g. max_total_literal_items ~400, max_string_literal_bytes ~2000) and add record_qa to the audit memorization-watch list + winner hand-audit protocol.
- **[MINOR]** verify --rescore and bench determinism need an explicit server-availability story; final.json sits outside the verifiable record — *Fix:* verify_run: for tasks with a "server" config key, hit /health once up front and abort with one actionable message if unreachable. Record finalize as a sealed, chained record (a final=true submission appended by the operator) or at minimum store program_sha256 + nonce + transcript shas in final.json and teach verify to re-derive and check it. Document rescore's sample cost and tag it ("rescore-*") for the meter.
- **[MINOR]** Change-list completeness nits: trace.py rescale, eval-log lm_seconds, session-salt back-compat, and the 'fixed determinism nonce' contradiction — *Fix:* Add the trace.py and cli.py eval-log lines to §12; make session salt default (e.g. meta.get("salt", "legacy")) for pre-existing runs; rewrite gate (iii)/#7 to 'shared explicit sample_nonce passed by run_checks' and pick the margin from the measured common-random-numbers SE (or run the headroom row at k>1 / on train+val to buy SE) so the CI row flakes <1%.

*Notes:* Reviewed against the real harness: /Users/ethanewer/text-opt-bm/.claude/worktrees/random-tasks/bench/{runner,session,cli,eval_lib,heldout,trace,audit}.py, tools/run_campaign.py, tests/run_checks.py, and bench/tasks/word_problems/evaluate.py as precedent. What HOLDS: the nonce plumbing is sound (runner's allowlist env drops any agent-set TEXTOPT_SAMPLE_NONCE, so change #1's explicit set is the only path in); program-hash-derived nonces make determinism and byte-identical resubmission work with zero changes to `bench determinism` logic as claimed; the leaked-session-salt worry is a non-issue because predicting seeds is useless without the sealed val docs/instance ids, and finalize's test seeds are equally unreachable; the exact-type render/parse checks correctly keep candidate code inside guarded spans (the evaluator's own urllib runs outside them, so the FORBIDDEN additions don't self-conflict); check_server-before-load_program ordering is consistent with the run_checks SKIP plan; timeout/cpu_s arithmetic works (evaluator CPU is trivial, wall 80-160 s vs 900 s) — under serial scheduling; and final_every=false correctly prevents the per-submission 30-minute --final that session.py:198 would otherwise trigger. The design's §12 change list is close but not complete — findings 3, 2, 5, and 8 are omissions from it. Heaviest genuinely-new harness work: tools/lfm_server.py auth+metering (finding 4) and run_campaign resource groups (finding 5); everything else is small, localized edits. No fatal flaw: every issue has a concrete in-design fix, hence 'fixable'. Priority order for the author: finding 1 (rescore integrity is the benchmark's core invariant), then 2 and 3 (silent wrong-behavior paths), then 4/5 (operational).


---

# record_qa_vote — SELECTED (judge mean 7.50, 3/3 advances, needs-real-LM 8.8/10)

# record_qa_vote — sample-budgeted record QA over a real LM (design spec)

Task id: `record_qa_vote` · kind: `generalization`, `stochastic: true` · metric: val error rate (min).
Grading LM: LiquidAI/LFM2.5-230M, fp32, 1 thread, served by an experimenter-managed local server. Ship AFTER the base `record_qa` task validates the shared machinery (server, seed plumbing, generator family); this task is `record_qa` + the vote-budget layer.

## 1. Premise (agent-facing story)

A small LM answers questions about warehouse order logs. The candidate program never sees the log — it engineers (a) the prompt template wrapped around the log, (b) how many samples (1–3) to buy per question from a global budget, and (c) the parser that turns the sampled texts into one final answer. Score = error rate on a hidden validation split. The LM is the noise source and the system under test; correctness is an exact string match, never a judge.

## 2. Instance family (published in spec.md — the agent has complete knowledge of the distribution; only the concrete draws are sealed)

Doc = "Warehouse order log:" + ~10 bulleted records: `"{Name} ordered {qty} {item} on {Month} {day}."` Gold = canonical digit string (no leading zeros). Question kinds (question-surface visible ⇒ allocatable) with M=150 val mix, and measured single-sample accuracy p (temp 0.7, tuned direct prompt / naive prompt):

| kind | n@150 | what varies | measured p |
|---|---|---|---|
| qty-digits | 27 | plain lookup | 0.72–1.00 (few-shot) |
| qty-wordnum | 33 | doc qty in English words ("sixty-seven") | 0.38–0.50; plur-3 0.62 |
| qty-distract | 23 | all 10 records same item, names differ | 0.88–0.95 |
| qty-coref | 12 | "The next day the same customer also ordered…" | 0.00–0.5 (prompt-sensitive) |
| day | 22 | "on what day of the month…" | 0.25 naive → 0.50 direct → 0.62 few-shot |
| maxq | 18 | "largest number of X in a single order" | 0.06 direct → 0.21 retrieval-decomposition |
| total | 15 | two orders by same person, sum | 0.00 direct; decomposition partial |

Note the four qty-* kinds share one question surface (the difficulty lives in the sealed doc) — allocation there works on the mixture mean; day/maxq/total are classifiable from the question ("day of the month", "largest", "in total"). Question phrasings: ~6 templates/kind; train exposes only ~3; val/test also use names/items absent from train (kind classifiers must generalize, not memorize strings). Optional 8th kind `count` (measured: 0.00 direct, 0.29 via decomposition) may be added at the freeze-calibration pass.

Splits: `data/train.jsonl` = 120 rows `{id, doc, question, gold, kind, style}` fully visible (the agent's Monte-Carlo bench — it may query the server directly on train at ~0.5–2 s/sample wall cost). `data/heldout_val.bin` = 150, `data/heldout_test.bin` = 400 (bench/heldout encoding). Generator `tools/gen_record_qa.py` is committed but the production seed is NOT (sealed draws are un-regenerable; family knowledge ≠ instance knowledge).

## 3. Candidate API

```python
def build(question: str) -> dict:
    # {"system": str<=600ch, "pre": str, "post": str  (len(pre)+len(post)<=2000ch),
    #  "n": int in 1..3, "max_new_tokens": int in 8..80 (optional, default 48)}
def parse(texts: list[str], question: str) -> str   # final answer, digits
```
The evaluator builds the single user message `pre + doc + post` (doc inserted exactly once, by the evaluator), plus the optional system message. Correct iff `str(parse(...)).strip() == gold`. Exceptions in `parse` → wrong. Exceptions/invalid shapes from `build` → the DEFAULT template (`{"system":"","pre":"","post":"\n"+question}`) with n=1 — the sample is still spent (no budget-banking via deliberate failure). Both functions MUST be deterministic (the evaluator calls `random.seed(0)` before each candidate call; nondeterminism breaks `verify --rescore` and invalidates the run).

Sampler is FIXED by the evaluator: temperature 0.7, top_k 50, no repetition penalty, fp32. The randomness is part of the task definition (temp 0.7 chosen because measured per-instance p spans the full 0–1 range there; near-greedy would collapse p to {0,1} and kill allocation).

## 4. Budget walk — exact scoring protocol (the airtight clamp)

Budgets: val B=225/M=150; train-only B=180/M=120; test B=600/M=400 (ratio 1.5 everywhere).

```
remaining = B
for i, inst in enumerate(instances):          # fixed order = order in the data file
    later = M - 1 - i
    tpl, n_req = safe_build(inst.question)    # invalid/raise -> (DEFAULT, 1), still charged
    n     = max(1, min(3, n_req))             # bool rejected; non-int -> 1
    n_eff = min(n, remaining - later)         # reserve 1 for every later instance
    remaining -= n_eff
    texts = [generate(tpl, inst.doc, seed(i,j), tpl.max_new_tokens) for j in range(n_eff)]
    ans = safe_parse(texts, inst.question)    # raise -> None
    wrong += not (isinstance(ans, str) and ans.strip() == inst.gold)
score = wrong / M
```
Invariant (published with proof in spec.md): before instance i, `remaining >= M - i`, hence `1 <= n_eff <= 3` always, every instance is sampled, and total spent ≤ B. Verified worked corners (unit-tested): all-n=3 → n_eff = 3×37, 2×1, 1×112, spends exactly 225; all-n=1 spends 150 (leaves 75); all-n=2 → 2×75, 1×75. Seeds do not depend on other instances' n (sample j of instance i is identical under any allocation).

## 5. Seeds, recording, rescore

- Harness draws a fresh `eval_seed` (32-hex) per grading, passes it as `--eval-seed` argv (never env), records it in the submission record.
- `seed(i,j) = int.from_bytes(sha256(f"{eval_seed}|{split}|{instance_id}|{j}").digest()[:8],"big") % 2**63`; server does `torch.manual_seed(seed)` per request.
- Metrics record `transcript_sha256` = sha256 over `f"{instance_id}|{j}|" + text` in walk order — the sampled texts are pinned into the hash chain. Also recorded: samples_used, budget, n_hist{1,2,3}, gen_tokens, lm model+dtype from `/health`.
- `bench verify --rescore` replays with the recorded eval_seed against the operator's server: bit-identical transcripts on the same machine (VERIFIED: same-seed regeneration byte-identical; different seed diverges). Cross-machine reproduction not guaranteed — same per-system-stability stance as the memory tasks.
- `bench determinism` (stochastic branch): two runs with a pinned seed must match exactly (incl. transcript hash); one run with a different seed must differ in the hash (proves the noise is live).
- Optional operator flag: dump full transcripts to run_dir, heldout-sealed (they echo sealed doc fragments; never store plaintext).

## 6. Cost mechanics (measured, Apple M5, fp32, torch threads=1)

- Per sample: 0.46–0.9 s for tuned short-answer prompts (gen 14–20 tok); 0.9–2.0 s naive rambling (gen 34–48); ~2.1 s retrieval-listing prompts at max_new_tokens 80. HTTP overhead negligible.
- Visible grade: baseline (n=1, naive) measured 150 samples ≈ 150 s wall. Full-budget candidates: 225 samples ≈ 170–470 s depending on prompt verbosity. So each visible grade costs real wall time out of the 1-h box (~8–20 grades/box) — the sample budget meters variance-averaging, wall-clock meters token spend: two genuine cost axes.
- Marginal-value arithmetic (published): with scattered wrong answers, plurality-of-3 lifts p → p³+3p²(1−p)+p(1−p)²: +0.10 at p=0.4, +0.125 at p=0.5, +0.147 at p=0.7, ~0 at p≈0 or ≈1 (measured: wordnum 0.50→0.62). Slack = B−M = 75 ⇒ k=3 on at most 37/150 items. Perfectly targeted ⇒ ≈ 37×0.13/150 ≈ 0.032 error; all-n=3 (position-based) buys the same 37 votes on a RANDOM 25% ⇒ ≈ 0.4 of that. Honest note: the degenerate corners are dominated but not catastrophic — all-n=3 wastes ~60% of the achievable voting gain, all-n=1 wastes 100%.
- `--final`: test (400 inst, B=600) + a forced-n=1 diagnostic pass on test (400 samples) reporting `test_fixed1_score` (isolates the allocation contribution). ≈ 12–25 min, once per run, sealed.

## 7. Noise and selection-under-noise

Analytic SE of a visible grade: sqrt(Σ v_i(1−v_i))/150 ≈ 0.031 (measured baseline per-kind profile) to 0.034 (strong-solution profile); empirical baseline seed spread 0.4533 vs 0.4733 agrees. Prompt-ladder steps (0.05–0.15) ≫ SE; allocation-ladder steps (~0.03) ≈ 1 SE per grade — but the agent measures kind-level p on TRAIN via the server (100 samples/kind ≈ 1–2 min) rather than by re-grading, and its own voting reduces its own SE (task goal aligned with grading hygiene). Best-tracking stays strict-`<` on visible val; the official number is the sealed test grade at `--final` with 2.7× more instances (SE ≈ 0.019), which also bounds winner's-curse inflation.

## 8. Server integration (tools/lm_server.py — NEW, experimenter-managed)

Runs in its own venv (torch 2.12 + transformers 5.13; evaluate.py stays pure stdlib and talks urllib to 127.0.0.1). Serialized single worker, `torch.set_num_threads(1)`, fp32 frozen.
- `GET /health` → {model, dtype, threads, torch/transformers versions} — recorded in metrics; rescore compares.
- `POST /generate` {system, user, max_new_tokens, temperature, top_k, seed, meter?} → {text, n_prompt_tokens, n_gen_tokens}. Hard caps server-side: max_new_tokens ≤ 128, prompt ≤ 4096 tok (unreachable under the 2600-char template caps + ~350-token doc — defense in depth; if ever exceeded the evaluator substitutes the DEFAULT template for that instance, deterministically). Literal special-token strings (`<|im_start|>` etc.) are stripped from candidate-supplied fields before templating (no fake-turn/assistant-prefill injection).
- `POST /meter/open {budget}` → meter id; each metered generate decrements; `/meter/close` → used. The evaluator opens a meter with B and cross-checks `used == samples_used` in metrics — server-side metering is the meter of record, outside candidate reach.
- Transient server error → retry once (same seed ⇒ idempotent), then `fail("LM server unavailable")` — infra error, not a score.
- Concurrency: ~0.95 GB/instance fp32; campaign driver assigns one server per concurrent LM job via a port pool (cap ~6–8 on 32 GB) — port reaches the evaluator as a runner ARG from config/CLI, never env (env-pointing would enable a puppet server that regex-solves the doc it receives; also caught post-hoc because rescore replays against the operator's real server).

## 9. Anti-exploit summary

1. **No emit-answer memorization on val/test** (the boundary lesson): the candidate sees only the question at runtime; gold is a function of the sealed doc; question strings never repeat across splits; production generator seed unpublished ⇒ un-regenerable. Score is produced by live LM sampling — closer to measurement-scored than emit-answer.
2. Budget: clamp forbids overdraft/order-gaming by construction; build-failure still charges (no banking); server meter cross-check.
3. Candidate forbidden: socket, ssl, http, urllib, xmlrpc, ftplib/poplib/imaplib/smtplib, asyncio, selectors, subprocess, multiprocessing, threading, time, plus the benchmark baseline blocklist; import guard toggled around every build/parse call (direct-call rule from eval_lib).
4. Doc inserted exactly once by the evaluator; char caps; special-token stripping; max_new_tokens ≤ 80 bounds echo bandwidth (targeted quoting + Python-side computation is a LEGITIMATE strategy — measured: it rescues maxq 0.06→0.21, count 0→0.29).
5. Degenerate corners dominated (see §6); constant-output prompts score ~0 accuracy; temperature fixed so the noise can't be collapsed.
6. Train-only blind mode's guide (train_score) is hardcodable from visible train golds — same accepted caveat as word_problems ("train is not the score"); default/recommended mode is full-feedback where the guide is the sealed val.
7. Generic error text (never echoes doc/gold/sampled text). Sealed per-kind metrics (`val_kind_acc`) added to HIDDEN_KEYS.

## 10. Baseline, reference, headroom (measured end-to-end, exact clamp+seed scheme)

- `initial_program.py` (naive: bare doc+question prompt, first-integer parse, n=1): val error 0.4533 / 0.4733 (seeds A/B), 150 s/grade.
- Reference (tests/solutions): tuned digits-only system prompt + word-number-converting plurality parse + kind allocation (day n=3, qty-family n=2, maxq/total n=1): projected 0.34–0.38 (probe kind-lifts: day +0.25, wordnum plur +0.12; full 3-seed run was still in flight at spec time — MUST be re-measured in the calibration pass before freeze).
- Strong multi-iteration path: per-kind template specialization (few-shot lifts day to 0.62, qty-digits to 1.00 — but the SAME few-shot hurt wordnum 0.38→0.12, so one-prompt-fits-all fails and per-kind engineering + re-measurement is forced), retrieval+compute decomposition for maxq/count/total, allocation re-tuned per prompt regime (better prompts shift which kinds are mid-p — the two ladders interact). Projected plateau ≈ 0.22–0.28. Headroom ≈ 0.46 → ~0.25 (Δ≈0.20 ≈ 6–7 visible-grade SE, ~1.8×), many rungs, no one-shot (probes show single-prompt ceilings and CoT actively backfiring on a 230M model).

## 11. Config + harness changes

```json
{ "name": "record_qa_vote", "kind": "generalization", "stochastic": true,
  "metric": "error rate on hidden validation split (budgeted LM sampling)",
  "direction": "min", "timeout_s": 1500, "cpu_s": 120,
  "lm_server": {"port": 8931, "model": "LiquidAI/LFM2.5-230M", "dtype": "float32",
                 "temperature": 0.7, "top_k": 50, "max_new_tokens_cap": 80},
  "budgets": {"train": [120,180], "val": [150,225], "test": [400,600]} }
```
Harness: (1) `runner.evaluate(..., eval_seed=None, lm_port=None)` — for `stochastic` tasks draw/pass `--eval-seed`, pass `--lm-port` from config or explicit arg; (2) session records eval_seed per submission; `verify_run` replays it; (3) `bench determinism` stochastic branch (same-seed×2 identical, diff-seed hash differs); (4) HIDDEN_KEYS += val_kind_acc; (5) tools/lm_server.py + run_campaign port pool & health gate; (6) tests: clamp unit tests (the four corner walks above), seed-scheme vectors, mock-server evaluator test, broken/ network-import rejection, slow same-seed rescore regression. Timing note: LM server time lands in the trace's untouched "model time" component (not child CPU) — cross-machine normalization is approximate for this track, accepted like memory-task per-system stability.

## 12. Pre-freeze calibration checklist

Re-run the reference 3-seed measurement; publish the measured kind-level p table in spec.md; decide the `count` kind swap; run a codex campaign (the ground-truth test) checking (a) multi-iteration improvement, (b) allocation actually used (n_hist ≠ all-1 in winners), (c) no val-noise overfitting (val−test gap at final).

## Evaluator sketch

```python
bench/tasks/record_qa_vote/evaluate.py (pure stdlib):

```python
import argparse, hashlib, json, random, sys, urllib.request
from pathlib import Path
sys.path.insert(0, __file__.rsplit("/bench/", 1)[0])
from bench import eval_lib, heldout

FORBIDDEN = frozenset({"open","os","io","sys","pathlib","mmap","ctypes","socket","ssl",
  "subprocess","multiprocessing","threading","tempfile","shutil","sqlite3","dbm","shelve",
  "importlib","__import__","urllib","http","xmlrpc","ftplib","poplib","imaplib","smtplib",
  "asyncio","selectors","signal","time"})
DEFAULT_TPL = {"system":"", "pre":"", "post":None, "max_new_tokens":48}  # post="\n"+question

def rpc(port, path, obj):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}",
        data=json.dumps(obj).encode(), headers={"Content-Type":"application/json"})
    for attempt in (0,1):                      # same seed -> idempotent retry
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.loads(r.read())
        except OSError:
            if attempt: eval_lib.fail("LM server unavailable")

def seed_for(eval_seed, split, iid, j):
    h = hashlib.sha256(f"{eval_seed}|{split}|{iid}|{j}".encode()).digest()
    return int.from_bytes(h[:8], "big") % (2**63)

def safe_build(mod, question):
    random.seed(0); eval_lib.set_candidate_active(True)
    try:
        t = mod.build(question)
        assert isinstance(t, dict)
        sysm, pre, post = t["system"], t["pre"], t["post"]
        assert all(isinstance(x, str) for x in (sysm, pre, post))
        assert len(sysm) <= 600 and len(pre) + len(post) <= 2000
        mnt = int(t.get("max_new_tokens", 48)); assert 8 <= mnt <= 80
        n = t["n"]; assert isinstance(n, int) and not isinstance(n, bool)
        return {"system":sysm,"pre":pre,"post":post,"max_new_tokens":mnt}, max(1,min(3,n))
    except BaseException:
        return dict(DEFAULT_TPL, post="\n"+question), 1
    finally:
        eval_lib.set_candidate_active(False)

def run_split(mod, insts, B, split, eval_seed, port, force_n=None):
    M, remaining, wrong, used, gen_toks = len(insts), B, 0, 0, 0
    hasher = hashlib.sha256(); n_hist = {1:0,2:0,3:0}; kind_ok = {}
    meter = rpc(port, "/meter/open", {"budget": B})["meter"]
    for i, inst in enumerate(insts):
        later = M - 1 - i
        tpl, n = safe_build(mod, inst["question"])
        if force_n: n = force_n                      # --final fixed-1 diagnostic
        n_eff = min(n, remaining - later); remaining -= n_eff; n_hist[n_eff] += 1
        texts = []
        for j in range(n_eff):
            r = rpc(port, "/generate", {"system": tpl["system"],
                 "user": tpl["pre"] + inst["doc"] + tpl["post"],
                 "max_new_tokens": tpl["max_new_tokens"], "temperature": 0.7,
                 "top_k": 50, "seed": seed_for(eval_seed, split, inst["id"], j),
                 "meter": meter})
            if "error" in r:                          # oversized prompt (defense in depth)
                tpl = dict(DEFAULT_TPL, post="\n"+inst["question"]); continue_with_default()
            texts.append(r["text"]); gen_toks += r["n_gen_tokens"]
            hasher.update(f"{inst['id']}|{j}|".encode() + r["text"].encode())
        used += n_eff
        random.seed(0); eval_lib.set_candidate_active(True)
        try:    ans = mod.parse(list(texts), inst["question"])
        except BaseException: ans = None
        finally: eval_lib.set_candidate_active(False)
        ok = isinstance(ans, str) and ans.strip() == inst["gold"]
        wrong += not ok
        k = inst["kind"]; a,b = kind_ok.get(k,(0,0)); kind_ok[k] = (a+ok, b+1)
    assert used == rpc(port, "/meter/close", {"meter": meter})["used"] <= B
    return {"score": round(wrong/M, 6), "samples_used": used, "budget": B,
            "gen_tokens": gen_toks, "n_hist": n_hist,
            "transcript_sha256": hasher.hexdigest(),
            "kind_acc": {k: round(a/b, 4) for k,(a,b) in kind_ok.items()}}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("program")
    ap.add_argument("--final", action="store_true"); ap.add_argument("--train-only", action="store_true")
    ap.add_argument("--eval-seed", required=True); ap.add_argument("--lm-port", type=int, default=8931)
    a = ap.parse_args()
    health = rpc(a.lm_port, "/health", {})          # {model, dtype, threads}
    mod = eval_lib.load_program(a.program, FORBIDDEN, required=("build","parse"))
    metrics = {"eval_seed": a.eval_seed, "lm_model": health["model"], "lm_dtype": health["dtype"]}
    if a.train_only:
        r = run_split(mod, load_train(), 180, "train", a.eval_seed, a.lm_port)
        metrics.update(train_score=r["score"], **prefixed(r, "train_")); eval_lib.succeed(r["score"], metrics)
    val = heldout.read(DATA/"heldout_val.bin")
    r = run_split(mod, val, 225, "val", a.eval_seed, a.lm_port)
    metrics.update(val_score=r["score"], n_val=len(val), samples_used=r["samples_used"],
                   budget=225, n_hist=r["n_hist"], gen_tokens=r["gen_tokens"],
                   transcript_sha256=r["transcript_sha256"], val_kind_acc=r["kind_acc"])  # kind_acc -> HIDDEN_KEYS
    if a.final:
        test = heldout.read(DATA/"heldout_test.bin")
        rt  = run_split(mod, test, 600, "test",  a.eval_seed, a.lm_port)
        rt1 = run_split(mod, test, 400, "test1", a.eval_seed, a.lm_port, force_n=1)
        metrics.update(test_score=rt["score"], test_fixed1_score=rt1["score"],
                       test_transcript_sha256=rt["transcript_sha256"])
    eval_lib.succeed(r["score"], metrics)
```

tools/lm_server.py (own venv, torch+transformers): http.server + global lock (serialized); torch.set_num_threads(1); model fp32 loaded once; /generate strips tokenizer special-token literals from system/user, applies chat template with add_generation_prompt=True, torch.manual_seed(seed), do_sample=True temp/top_k from request (cap max_new_tokens 128, prompt 4096 tok), returns text + token counts; /meter/open|close in-memory dict; /health returns model id, dtype, thread count, library versions.
```

## Measured numbers

All numbers MEASURED on LFM2.5-230M fp32 torch-threads=1 (Apple M5), temp 0.7 / top_k 50, via 5 probes (~1,400 generations) in /tmp/rl3/. PER-SAMPLE COST: tuned short-answer prompts median 0.46–0.90 s (gen 14–20 tok, prompt ~185–340 tok); naive rambling 0.86–1.96 s (gen 34–48); retrieval-listing (max_new 80) ~2.0–2.1 s. Seed reproducibility: same-seed regeneration byte-identical, different seed diverges (verified). VISIBLE GRADE: k = 225 samples over M = 150 val instances (ratio 1.5); measured end-to-end with the exact clamp: baseline n=1 grade = 150 samples, 150 s wall; full-budget ≈ 170–470 s. OFFICIAL: sealed test 400 instances / 600 samples + 400-sample fixed-n=1 diagnostic (~12–25 min, once). BASELINE (naive prompt, first-int parse, n=1): val error 0.4533 / 0.4733 on two eval seeds (150 s each); reference (tuned prompt + plurality + kind allocation) projected 0.34–0.38 (3-seed run still in flight — re-measure at calibration). NOISE: empirical seed spread 0.020; analytic SE = sqrt(Σp(1-p))/150 = 0.031 (baseline profile) – 0.034 (strong profile); test SE ≈ 0.019. HEADROOM vs noise: prompt ladder ≈ 0.46 → ~0.25 (measured rungs: day 0.25→0.50 direct →0.62 few-shot; qty-digits →1.00 few-shot; wordnum plur-3 0.50→0.62; maxq 0.06→0.21 and count 0→0.29 via retrieval+Python-compute; CoT BACKFIRES: tag-mimicry, 0.17); allocation ladder ≈ 0.032 best-case (37 targeted k=3 votes × ~+0.13 avg plurality lift; formula p³+3p²(1−p)+p(1−p)²) ≈ 1 SE per grade but measurable offline on train at ~0.7 s/sample. Total Δ ≈ 0.20 ≈ 6–7 visible-grade SE. Clamp corners (unit-verified): all-3 → 37×3+1×2+112×1 = exactly 225; all-1 → 150.

## Open questions (spec author)

1) Reference-solution 3-seed val scores did not finish before spec deadline (probe5 running; baseline done at 0.4533/0.4733) — re-run at calibration and publish the kind-level p table in spec.md before freeze (results land in /tmp/rl3/results5.jsonl). 2) Whether to add the `count` kind (~8%, decomposition-rescuable 0→0.29) at the expense of qty-digits/total — decide from the calibration table. 3) Owner sign-off on the cross-machine caveat: LM server time sits in the trace's un-rescaled "model time" component, so normalized cross-machine comparison is approximate for this track (per-system stability only, like memory tasks). 4) Campaign concurrency: one fp32 server per LM job ≈ 0.95 GB + 1 core each — cap LM-task concurrency at ~6, or freeze bf16 instead (halves RAM, changes all transcripts; pick one dtype forever before any recorded run). 5) Winner's-curse residual: fresh eval_seed per grading means best-tracking can select a lucky draw (~1 SE); consider having the loop re-grade the session best once at end-of-run (cheap: one extra 225-sample grade) in addition to the sealed --final. 6) The all-n=3 corner is only ~60%-dominated (position-based voting still helps on average) — acceptable (both corners leave measured gain on the table) but the spec should not overclaim "scores worse". 7) Sequencing (judge requirement): ship after base record_qa validates server/seed/rescore machinery in a real codex campaign; the vote-budget layer adds ~60 evaluator lines + clamp tests on top.

## Red-team reports

### Lens: exploit — verdict: FIXABLE

- **[MAJOR] Anti-exploit §9.1 overclaims 'no emit-answer memorization on val/test' — a parse-side question→gold table defeats it under defection**
  - This is still an emit-answer task at the boundary TASK_AUTHORING.md defines: correctness is `parse(...) == gold`, and the sealed splits are only zlib+XOR obfuscated (bench/heldout.py explicitly says decoding is 'cheating by rule', cooperative model). Val/test question strings are effectively unique per instance (they embed names/items absent from train), so a defecting agent decodes heldout_val.bin/heldout_test.bin once, ships `parse` = `{question: gold}` dict lookup ignoring `texts`, sets n=1 everywhere, and scores 0.0 — far below the honest ~0.22 plateau. The unpublished generator seed ('un-regenerable') is irrelevant because the concrete draws are committed in decodable form; §9.1's framing ('closer to measurement-scored') is wrong — live LM sampling produces the *texts*, but the score compares an agent-emitted string to a gold the agent can unseal. Per TASK_AUTHORING's robustness-boundary section, this shape is memorizable by a determined adversary and is acceptable ONLY as a cooperative-model task with explicit caveats — the spec instead claims immunity.
  - *Fix:* Rewrite §9.1 to the word_problems-style honest caveat: memorization requires decoding the sealed bins (cooperative rule violation), it is not prevented by construction. Add the loud tells to the audit path: val_score ≪ measured plateau with n_hist all-1 and near-zero gen_tokens-sensitivity; test_fixed1_score == test_score exactly; add tests/broken/record_qa_vote_gold_table.py as a regression that `bench audit`/spot-check procedure must flag. Optionally hash-compare parse output distribution against transcripts (answers that never appear in any sampled text across >X% of instances is a strong automatic tell worth recording as a metric).
- **[MAJOR] --final worst-case wall time exceeds timeout_s=1500**
  - main() always runs the val pass (225 samples) before the final test (600) + fixed-1 diagnostic (400) in the same process: 1225 samples total. At the spec's own measured verbose-prompt cost (naive rambling 0.9–2.0 s, retrieval-listing ~2.1 s at max_new_tokens 80), worst case ≈ 1225 × 2.1 ≈ 2570 s > timeout_s 1500. The spec's '12–25 min' estimate for --final apparently excludes the val pass and assumes mid-range token counts. A legitimately strong candidate using retrieval-listing prompts (the strategy the spec itself endorses for maxq/count/total) gets its one sealed final graded as a timeout — and an adversarial agent can weaponize the inverse: a rival-style verbose template deliberately converts the official grade into an infra failure.
  - *Fix:* Compute the timeout from the hard caps: (225+600+400 samples) × (max_new_tokens 80 / ~38 tok/s fp32 + prompt prefill) ≈ 3000–3500 s ceiling; set timeout_s accordingly for the --final invocation (or give --final its own timeout / skip the val pass when --final is set). Add a unit test that the cap arithmetic covers the max-tokens corner.
- **[MAJOR] Retry-once can double-decrement the server meter and invalidate a valid run**
  - rpc() retries on OSError with the same seed, calling it idempotent — generation is idempotent for the *text*, but the sketch's /meter accounting is a plain decrement per metered generate. If attempt 1 completes generation server-side and the response is lost (timeout=120 hit on a slow 80-token fp32 generation under load, connection reset), the retry decrements the meter again: /meter/close returns used = samples_used+1 and `assert used == ... <= B` kills the run as if the candidate cheated. This fires exactly when the safety net is exercised, and on the once-per-run sealed --final it destroys the official grade. Separately, the oversized-prompt fallback path (`continue_with_default()`) is undefined pseudocode: it must specify whether the errored request was metered, which seed the substituted default-template sample reuses, and that the substitution applies deterministically so rescore replays identically.
  - *Fix:* Make metering idempotent: key each metered generate by sha256(meter|seed|prompt) server-side and count unique keys; /meter/close returns the unique count. Specify the oversize fallback precisely: error responses are never metered, the evaluator re-issues sample j with the DEFAULT template and the SAME seed(i,j), and the substitution is recorded in metrics (default_substitutions count) so rescore divergence is diagnosable. Also note the '≤4096 tok unreachable' claim is false — 2600 chars of multi-token unicode plus the doc can exceed it — so this path is reachable and needs the exact semantics, not defense-in-depth hand-waving.
- **[MINOR]** Prior-guessing from the published generator beats the LM on low-p kinds (maxq) — legitimate but unbudgeted in the headroom story — *Fix:* At the calibration pass, compute and publish per-kind gold-mode probability from the generator; widen ranges or post-process draws so every kind's mode probability sits well below the achievable LM plateau for that kind (e.g. <0.05), or explicitly fold prior-guessing into the baseline/reference accounting so headroom numbers stay honest.
- **[MINOR]** Module-global state lets build/parse infer instance position and split, and persists across val→test in --final — *Fix:* Reload the candidate module between splits (the compress-task lesson already in eval_lib) and, if cheap, between the val walk and any diagnostic pass; state in spec.md that cross-call state is permitted-but-useless within a split so honest agents don't waste time on it. Randomizing nothing is needed — reload alone removes the split channel.
- **[MINOR]** Port-squat / puppet-server by the optimizing agent is only caught post-hoc at rescore — *Fix:* Have the campaign driver bind the port pool before the agent's box starts and hold the sockets for the whole run (squat becomes bind-failure, loudly). Add a startup challenge to /health recorded in metrics: the harness generates once with an operator-secret seed at server boot and pins that transcript hash; rescore compares. Document that rescore of the session winner is mandatory for this track, not optional.
- **[MINOR]** Winner's-curse mitigation (end-of-run re-grade of session best) should be required, not an open question — *Fix:* Promote OQ5 to a requirement: at end-of-run the loop re-grades the session-best program once with a fresh seed and records both numbers (report the re-grade as the selection score); pair with the calibration campaign check that winners' n_hist ≠ all-1 so fishing-with-cheap-grades doesn't masquerade as allocation skill.

*Notes:* Core mechanics survive attack well: the budget clamp is airtight (corner arithmetic re-verified: all-3 → 37×3+2+112×1 = 225), seeds are allocation-independent, build-failure still charges, temperature is evaluator-fixed so the noise can't be switched off, constant-output prompt collapse scores ~0 accuracy (dominated, not task-death), and doc-side prompt injection is moot because docs are generator-produced and inserted by the evaluator. The gravest issue is a claims problem, not a mechanics problem: with bench/heldout.py being zlib+XOR (cooperative seal), this remains an emit-answer task per TASK_AUTHORING.md's robustness boundary, and §9.1 must say so instead of claiming memorization immunity — the fix is honest caveats plus tells/regressions, since full prevention is impossible in this shape. The two operational majors (final-pass timeout arithmetic, non-idempotent meter under retry) would invalidate legitimate official grades and are straightforward to fix before freeze. No finding is fatal to the design; verdict fixable. Key reference: /Users/ethanewer/text-opt-bm/.claude/worktrees/random-tasks/TASK_AUTHORING.md (robustness boundary, lines 109–163) and /Users/ethanewer/text-opt-bm/.claude/worktrees/random-tasks/bench/heldout.py.

### Lens: statistics — verdict: FIXABLE

- **[MAJOR] Reference solution measured WORSE than baseline — headroom claim refuted as-shipped**
  - I completed the 3-seed reference run the spec left in flight (open question 1), using probe5's exact Reference implementation and evaluator walk (/tmp/redteam/ref_seeds.jsonl). Measured: reference val error 0.5667 / 0.5733 (seeds A/B) vs baseline mean 0.491 over 5 seeds — the reference is +0.079 WORSE (SE 0.019), against a spec projection of 0.34–0.38. Per-kind breakdown shows why: the tuned digits-only system prompt REGRESSES the easy kinds (qty 0.68→0.53-0.58, distract 1.00→0.70-0.74) at temp 0.7, and the day n=3 votes that were supposed to deliver the allocation gain were never served (see the clamp/order finding). TASK_AUTHORING checklist item 1 (reference beats baseline by a healthy factor) currently fails; the published headroom story (0.46→~0.25) is unvalidated end-to-end — the probe kind-lifts were measured with different prompt shapes (direct2/fewshot) and do not compose into this reference.
  - *Fix:* Block freeze on a re-engineered, actually-measured reference: per-kind prompt branching (bare prompt for qty/distract, tuned/few-shot only for day), allocation that fits within slack (e.g., n=3 on day only = 194 total ≤ 225), and a ≥3-seed measurement published in spec.md. The spec already gates on this (calibration pass) — but treat it as red, not amber: the current projection is contradicted, not merely unfinished.
- **[MAJOR] Budget walk serves front-of-file first and the data file order is kind-grouped — over-requesting silently starves the high-value kinds**
  - The greedy clamp (n_eff = min(n, remaining - later)) serves earlier instances preferentially. The prototype val builder (probe5.build_val, and by extension tools/gen_record_qa.py) emits instances GROUPED BY KIND: qty(60), distract(23), coref(12), day(22), maxq(18), total(15). Simulated exactly (/tmp/redteam/clampsim.py): the reference requests 289 samples; the walk serves n=2 to positions 0–74 (all qty + some distract) and n_eff=1 to EVERYTHING after position 75 — day's n=3 requests get exactly 1 sample each. Confirmed live: reference n_hist = {2:75, 1:75}, day accuracy unchanged from baseline. Two consequences: (a) the spec's dominance claim 'all-n=3 buys votes on a RANDOM 25%' is false under grouped order — it buys votes on the FIRST 37, i.e. one specific kind block; (b) any candidate that over-requests has its allocation silently rewritten by sealed file order it cannot observe, turning kind-targeting into a position lottery.
  - *Fix:* Seeded-shuffle the instance order in every split's data file at generation time and STATE in spec.md that the walk serves in file order with front-priority under scarcity (so over-requesting has predictable-in-expectation incidence). Re-verify the 'all-n=3' and 'all-n=1' corner dominance claims on the shuffled order. Add a unit test asserting the val/test files are not kind-contiguous.
- **[MAJOR] Allocation gain is ~2x overestimated: the plurality formula is applied at kind-level p but within-kind difficulty is bimodal**
  - From 1,000+ existing probe generations (4–8 samples/instance, /tmp/redteam/scatter2.py): the plurality-of-3 formula holds well at INSTANCE-level p (pooled mid-p instances, N=53: measured lift +0.109 vs formula +0.093), but the agent can only target by KIND, and within-kind instance p is bimodal, which dilutes the realized kind-level lift (Jensen): qty/words kind p=0.375 → formula-at-kind-mean predicts +0.058, measured kind lift 0.000; day naive +0.031 vs +0.051 predicted; only wordnum-scatter cases match (+0.054). Measured kind-level vote-3 lifts span 0.00–0.125 (mostly 0.03–0.10), so the realized allocation ceiling is ≈ 37×0.05–0.10/150 = 0.012–0.025, not the spec's 37×0.13/150 = 0.032. At the measured noise level that is <1 visible-grade SE and ~1 test SE — the vote-budget layer, the entire novelty of this task over base record_qa, is near-invisible in both the visible and official grades. Also, k=2 voting under the spec parse lifts only via validity filtering (pooled +0.048 on mid-p, 0.000 on qty/words), yet the reference spends 150 of its 225 samples on n=2.
  - *Fix:* (1) Re-derive the published marginal-value table from instance-level measurements per kind (the probe data already supports this), including honest k=2 arithmetic. (2) Consider raising slack to B=2M (val 300/150): doubles targetable instances to 75, pushing the realized allocation gain to ~0.025–0.05 ≈ 1–2 SE, at +50% grade wall-cost — recompute the trade-off. (3) The calibration campaign check '(b) allocation actually used' should require the allocation to measurably beat forced-n=1 on test, not just n_hist ≠ all-1.
- **[MAJOR] Visible-grade noise calibration is stale and likely understated (published SE 0.031, 'empirical spread 0.020'; measured 5-seed SD 0.041)**
  - The spec's 'empirical baseline seed spread 0.4533 vs 0.4733 agrees' was written from 2 seeds; seed C (already in /tmp/rl3/results5.jsonl, landed after the spec) scored 0.560, and my two additional seeds give the 5-seed set {0.4533, 0.4733, 0.560, 0.4933, 0.4733}: SD = 0.0413 vs published 0.031 (chi2 stat 7.1 on 4 df, p≈0.13 — not conclusive, but the published number is a point estimate from kind-level p that should if anything OVERstate the conditional SD under bimodal instance p, so a 1.33x-higher empirical point estimate needs resolving). Every design-tension number keys off this SE: rung detectability (at SD 0.046 a 0.05 prompt rung is detected by a single grade pair only 78% of the time vs 87% at 0.031), the allocation-layer visibility, and the winner's-curse magnitude.
  - *Fix:* Run ≥10 eval seeds of the baseline (and reference) at calibration — my 7-seed extension run is the template (/tmp/redteam/base_seeds.py, ~3 min/seed) — and publish the empirical SD in spec.md instead of the 2-seed claim. If SD lands ≥0.04, revisit M=150 (e.g., 200 instances) or accept and document that per-grade feedback is train-side only.
- **[MAJOR] test_fixed1 diagnostic is unpaired and underpowered for the effect it exists to isolate**
  - The --final fixed-n=1 pass uses split label "test1", so its seeds differ from the main test pass's j=0 seeds. The allocation-contribution estimate test_score − test_fixed1_score then carries the full noise of two independent 400-instance grades: SE ≈ sqrt(2)×0.019 ≈ 0.027 — the same order as the realized allocation effect (0.012–0.03 per the finding above). The one diagnostic the design ships to prove the vote layer matters cannot resolve it.
  - *Fix:* Use split="test" for the diagnostic pass so its sample j=0 seed equals the main pass's j=0 seed (the seed scheme already guarantees sample j of instance i is allocation-independent). The comparison becomes common-random-numbers-paired: identical first samples, difference nonzero only on instances where extra votes flipped the outcome, collapsing the SE to ~sqrt(f)/400 where f is the flip rate (~0.005–0.01, i.e., 3–5x tighter). Zero extra cost.
- **[MINOR]** Winner's-curse optimism on the recorded best-visible val is ~2.2 SE, not the '~1 SE' the spec states, and the overfitting check must not use it — *Fix:* Make the end-of-run re-grade of the session best (open Q5's option) mandatory, record it as final_val_regrade, and define the overfitting check as test_score − final_val_regrade (expected ≈ 0 ± instance-draw noise ~0.04), never test_score − best_visible_val. Correct the '~1 SE' claim in the spec.
- **[MINOR]** Analytic SE formula published from kind-level p is internally inconsistent with the task's own bimodality story — *Fix:* At calibration, record per-instance correctness per seed, estimate v_i directly, publish both sqrt(Σ v_i(1−v_i))/150 and the empirical between-seed SD, and use the instance-level v_i distribution for both the SE and the allocation-lift table so the two analyses share one ground truth.

*Notes:* All findings are measurement-backed: I ran ~1,600 additional LFM2.5-230M generations (5 baseline eval-seed grades total incl. the 3 pre-existing, 2 full reference grades of 225 samples each — artifacts in /tmp/redteam/), re-analyzed the ~1,400 existing probe generations in /tmp/rl3/results*.jsonl at instance level, simulated the exact budget walk, and Monte-Carlo'd the selection process. What is SOUND and verified: the budget-clamp corner arithmetic (all-3 → 37×3+1×2+112×1=225 reproduced), seed-scheme allocation-independence, unbiased error-rate estimator (no ratio/max bias), real per-sample cost (~0.7–1.2 s measured), genuine train-side Monte-Carlo trade-off (~2 min per kind-level A/B arm vs ~3 min per grade), baseline signal mid-range (0.49, no saturation), plurality mechanism confirmed working at instance level, and the official sealed-test grade remains unbiased for the selected program (winner's curse costs the agent ~0.5 SE of true score, not the benchmark's honesty). The task's core machinery is statistically solid; what is broken is calibration-layer: the shipped reference is measurably worse than the baseline (0.570 vs 0.491 — freeze-blocking), the file-order/clamp interaction rewrites allocations silently (needs a seeded shuffle plus spec disclosure), the allocation layer's realized signal is ~half the published estimate and below per-grade detectability (consider B=2M), the noise calibration needs ≥10 seeds, and the fixed-1 diagnostic needs CRN pairing to have any power. Every fix is concrete, cheap, and inside the existing design — hence 'fixable', with the explicit condition that the pre-freeze calibration pass (which the spec already schedules) is treated as gating on these specific measurements, not as a formality. The spec's own open questions 1, 5, and 6 anticipated three of these issues; my measurements resolve all three against the spec's optimistic readings.

### Lens: harness — verdict: FIXABLE

- **[MAJOR] Session.submit runs --final on EVERY submission, exploding the cost model and rescore**
  - bench/session.py line 198-199: for any kind=="generalization" task, submit() calls runner.evaluate(..., final=True) unconditionally, and verify_run (line 347) replays final=True for every record. The spec assumes --final happens "once per run, sealed" (S6, open question 7) and budgets a visible grade at 170-470 s. Under the actual harness, every submission runs val (225 samples) + test (600) + fixed-1 diagnostic (400) = 1225 LM samples ~ 12-40+ min, collapsing "8-20 grades/box" to ~2-4, burning ~800 sealed-test samples per iteration for nothing, and making `bench verify --rescore` of a 40-submission run take on the order of 10-25 hours. The spec's S11 harness-change list does not include any change to submit-final semantics, so this is an unenumerated, moderately heavy change.
  - *Fix:* Add a real harness change to the plan: for this task (config flag, e.g. "final_on_submit": false or "deferred_final": true), session.submit passes final=False, and a new operator step (`bench finalize RUN_DIR`, or run_campaign end-of-box hook) grades best_program.py once with --final, appending a sealed final record; verify_run must replay per-record with the recorded mode. This also implements open question 5's end-of-run re-grade almost for free. Alternatively drop the fixed-1 diagnostic and accept test-every-submit, but then all the S6 grade-count arithmetic and timeout_s must be rewritten.
- **[MAJOR] test_fixed1_score and test_transcript_sha256 are not in HIDDEN_KEYS — sealed-test leak in the visible record**
  - bench/session.py HIDDEN_KEYS (lines 54-58) filters exact keys: ("test_score", "test_ratio", "n_test"). The sketch's --final path emits metrics test_fixed1_score and test_transcript_sha256; S11 change (4) only adds val_kind_acc. visible_metrics() therefore passes test_fixed1_score through in plaintext to the optimizer in every record (and it IS a genuine test-split error rate — n=1 accuracy on the sealed test), plus a transcript hash that lets an agent detect test-split changes. Combined with the always-final behavior above, the agent gets a fresh hidden-test signal every submission.
  - *Fix:* Add test_fixed1_score, test_transcript_sha256 (and any future test_* key, e.g. test_samples_used/test_gen_tokens) to HIDDEN_KEYS for both feedback modes — or better, switch visible_metrics to also hide by prefix ("test_") so new test-side metrics are hidden by default. One-line-ish session.py change; include a test_session.py case.
- **[MAJOR] val_kind_acc leaks through the self-test channel (`bench evaluate --json --full` and TEXTOPT_EVAL_LOG)**
  - HIDDEN_KEYS only filters session records. bench/cli.py lines 246-249 print the evaluator's raw result for `bench evaluate --json`, and the GOAL.md self-test command for full-feedback generalization runs is exactly `bench evaluate TASK program.py --json --full` (_selftest_cmd, lines 93-105); lines 220-238 also log full raw metrics to TEXTOPT_EVAL_LOG. The sketch emits val_kind_acc on the default val path, so the agent sees per-kind accuracy on the sealed val split at every self-test — precisely the signal S9.7 claims is sealed, and strong fuel for tuning allocation against val noise (the winner's-curse/val-overfit failure mode the design worries about in S12c).
  - *Fix:* Do not emit val_kind_acc from the default path. Either emit it only under --final (it then lands sealed with the test metrics, and only if finding 1's deferred-final is adopted), or gate it behind a new operator-only flag (e.g. --diagnostics) that bench evaluate never passes. Kind-level feedback the agent is meant to have comes from train via its own server queries, per S7.
- **[MAJOR] Train-only feedback mode silently uses the hidden val score as the guide (no train_score in the default path)**
  - session.submit never passes --train-only; blind mode works only because existing generalization evaluators (word_problems) compute train_score in EVERY mode and guide_score() (session.py 62-66) picks it up, falling back to result["score"] otherwise. The sketch's main() computes the train split only under --train-only; the default path emits no train_score. In a train-only session, guide_score therefore falls back to the val error, which is then written as plaintext guide_score and shown to the agent via session.visible() — blind mode leaks exactly the number it exists to hide. S9.6 claims train-only is a supported mode with train_score as guide.
  - *Fix:* Either (a) always run the train split too (adds 180 samples ~ 2-4 min per grade — must then be added to the S6 cost arithmetic), or (b) declare train-only unsupported for this task: add e.g. "feedback_modes": ["full"] to config.json and a small check in Session.create/open_or_create rejecting unsupported modes (new, but tiny, harness change — add it to S11). Given S9.6 already calls full-feedback the recommended mode, (b) is cheaper and honest.
- **[MAJOR] timeout_s=1500 is below the spec's own worst-case --final grading time**
  - runner.evaluate enforces config timeout_s as a hard wall limit (runner.py lines 74, 127-135) and returns an infra error on expiry. The --final path is val 225 samples (spec: up to 470 s for verbose prompts) + test 600 + fixed-1 400 (spec's own estimate 12-25 min = 720-1500 s); sum 1200-2000+ s, and at the measured 2.1 s/sample verbose ceiling ~2600 s. So a legitimate final grading (which under the current harness is EVERY submission, see finding 1) can hit the 1500 s wall timeout and be recorded as a failure — including during verify --rescore, where it would flag a valid record as non-reproducing.
  - *Fix:* If deferred-final is adopted: keep timeout_s ~900-1500 for the val-only submit path (225 x 2.1 s ~ 475 s worst case leaves margin) and give the finalize step its own timeout ~3600 s (runner.evaluate already takes explicit args; add timeout override for final=True or a config "final_timeout_s"). If final-per-submit stays, timeout_s must be >= ~3600.
- **[MINOR]** verify --rescore has no LM-server lifecycle: a down/mismatched server produces misleading per-record 'does not reproduce' failures — *Fix:* In verify_run for stochastic tasks: (1) hit /health once up front; abort with a single 'LM server unavailable/mismatched (model/dtype)' problem instead of per-record noise; (2) pass eval_seed from rec metrics into runner.evaluate; (3) print an estimated rescore duration. Add to the S11 change list explicitly.
- **[MINOR]** Idempotent-retry can double-decrement the server meter, crashing the evaluator on its own assert — *Fix:* Make /generate idempotent server-side keyed on (meter_id, seed, prompt_hash) — return the cached response without a second decrement — or have the evaluator treat meter mismatch as eval_lib.fail("LM meter mismatch (infra)") rather than assert, with the meter check relaxed to used <= metered <= used + retries.
- **[MINOR]** Sketch gaps: POST-only rpc() vs 'GET /health', and undefined continue_with_default() semantics — *Fix:* Specify: an oversized-prompt rejection consumes no meter tick and no transcript; the evaluator rebuilds with the DEFAULT template and re-issues the SAME seed(i,j) for each j; add this corner to the clamp unit tests with the mock server. Align /health method (make it POST-tolerant or use a bare GET in rpc).

*Notes:* Verified against the actual harness at /Users/ethanewer/text-opt-bm/.claude/worktrees/random-tasks: bench/session.py (submit final=True for all generalization tasks; HIDDEN_KEYS exact-key filtering; guide_score train_score fallback; strict-< best tracking matches the spec's S7 claim), bench/runner.py (no eval_seed/lm_port today — S11 change 1 is accurately scoped; argv-not-env seed passing fits the existing allowlist-env design; proxy env vars are stripped so urllib hits 127.0.0.1 directly), bench/cli.py (evaluate --json prints unfiltered metrics; determinism needs the stochastic branch exactly as S11 change 3 says), bench/eval_lib.py (load_program/set_candidate_active/fail/succeed signatures all match the sketch; the __import__ guard catches cached urllib re-imports by candidates; the audit hook blocks repo-file reads during candidate spans, protecting train.jsonl/heldout bins at runtime), bench/trace.py (the S11 timing note is CORRECT: the LM server is not a child of the evaluator, so its time is excluded from eval_cpu_seconds and lands in cum_model). I re-derived the budget-walk corners (all-3: 37x3+1x2+112x1=225; all-2: 75x2+75x1; all-1: 150) — the clamp invariant and the published corner numbers check out, and seed independence from allocation holds since seed(i,j) ignores n. JSON key round-tripping of n_hist (int keys -> strings) is consistent on both record and rescore sides, so no false mismatch there. tests/run_checks.py uses explicit task lists, so the planned mock-server test integrates cleanly; note the reference-solution headroom check can only run against a live server (calibration-time, not CI). The dominant theme: the spec's per-piece harness-change list (S11) is mostly accurate, but it missed that session.submit ALREADY forces --final on every generalization submission — the deferred-final change (finding 1) is the one genuinely heavy, unenumerated harness modification, and findings 2/5 are its direct consequences. All findings have concrete fixes; nothing is unfixable within the design. Recommend resolving findings 1-5 in the spec before the base record_qa task ships, since they are shared machinery."


---

# note_bottleneck — NOT SELECTED (judge mean 7.33, 3/3 advances, needs-real-LM 9.2/10)

**Disposition.** ALTERNATE (needs one fatal fix): "compression for an LM reader" is the most novel concept in the pool, but the statistics red-team found a FATAL free seed-shopping route (predictable noise seed + content-independent paraphrase draw, worth ~0.09-0.10 score) plus understated SE (~0.044 vs advertised 0.025). The fix (derive per-instance draws from seed XOR sha256(note), recalibrate k) is known but unvalidated. Revisit after the selected type-2 tasks ship.

**Task type:** random_generalization

## Measured numbers

All numbers measured on the owner's M5 with LFM2.5-230M fp32, torch threads=1, temp 0.7 / top_k 50 / 16 new tokens (probe scripts under /tmp/lfm_note/). PER-SAMPLE COST: 0.356 s mean / 0.342 s median (probe8, n=1330, prefill ~200-260 tok incl. chat frame); 0.41-0.69 s in earlier probes under bigger prefill/load — budget 0.4-0.6 s. VISIBLE GRADE k: 10 val sources x 12 questions, k=2 on weight>=2 fields (19 gens/source) = 190 samples ~= 50-80 s. OFFICIAL: per-submission sealed test = 228 samples ~= 85-95 s; end-of-campaign official regrade --k-mult 2 = 456 test samples (SE ~0.018), fresh recorded seed. EMPIRICAL NOISE: SE = 0.0253 (SD of 4 fresh-seed full grades of the reference program: 0.3667/0.3333/0.3354/0.3854); same-seed full-grade replay bit-identical (score + transcript sha). LADDER vs HEADROOM: naive prose[:180] = 0.546, full-key greedy = 0.417, weight-aware knapsack reference = 0.355 mean → baseline-to-reference gap 0.19 ≈ 7.5x SE; plausible expert ~0.28-0.31 (uncapped-note read-back measured 0.802-0.861, so kept-fact flake is compressible; short keys 0.583-0.611 vs full keys 0.806-0.861 shows format headroom is real). CALIBRATION (ship gate): keyed full coverage 242-262 chars (median 253), positional 197-217, prose 332-391 — all > the 180 cap; positional notes read at 0.185 (collapse path closed). HALLUCINATION CONTROL: 0 false-correct on omitted facts across all probes (0/129).


## Red-team reports

### Lens: exploit — verdict: FIXABLE

- **[MAJOR]** str-subclass note defeats the 180-char cap (ship gate #1 collapses) — *Fix:* Enforce the cap on the real string, not the object: `if type(note) is not str: fail(...)` (reject subclasses outright), then measure length/ASCII on that exact-type value. Do not trust __len__/__iter__/__str__ overrides. Add a broken-program fixture note_bottleneck_strsubclass.py (returns a length-lying str subclass) to section 11's reject suite; the current overcap/nonstr/unicode fixtures do not cover it.
- **[MAJOR]** Anti-exploit claim 'cap blocks tokenizer weirdness' is false — ChatML control tokens pass the ASCII filter — *Fix:* Sanitize notes before the server call: reject or neutralize any special-token substrings (e.g. reject notes containing '<|' , or strip/escape all tokenizer special tokens server-side before tokenization), and re-run the calibration/hallucination probes with adversarial control-token notes. Correct the wording of anti-exploit item 4 — the ASCII cap does not block ChatML control strings.
- **[MINOR]** Cosmetic-mutation resubmission still rerolls the visible val seed (winner's-curse fishing) — *Fix:* Adopt margin-based best-tracking for stochastic tasks (accept a new best only if visible < prior_best - delta, delta ~= 1 SE ~= 0.025), or average the visible grade over 2 recorded sub-seeds so the tracked best is less reroll-sensitive. Either keeps selection stable without changing the honest official metric.
- **[MINOR]** Exclusion rule penalizes correct-but-verbose answers, adding a second pressure beyond the cap — *Fix:* Acceptable as a design tradeoff, but characterize it: probe co-mention false-wrong rate on dense honest notes and confirm it does not inflate SE beyond the stated 0.025, or relax exclusion to only trigger when >=2 OTHER golds appear (distinguishing a single volunteered neighbor from a genuine dump).

*Notes:* Verified empirically with python3.12 and LFM2.5-230M fp32 (threads=1, temp 0.7/top_k50/16 tok) under /tmp. (1) str-subclass cap bypass reproduced: __len__/__iter__ overrides pass all three validity checks while json.dumps ships the full 380-char payload to the reader — this nullifies ship gate #1 and the entire triage premise; strongest finding, but trivially fixable via type(note) is str + a new broken fixture. (2) ChatML control tokens pass the ASCII filter (3 im_end vs 1 in clean prompt) so the note can forge turn boundaries — a real injection surface that falsifies anti-exploit claim #4, though it did not beat honest encoding on a probed field (8/8 both). The core reconstruction/measurement robustness is otherwise sound: module deleted before the reader loop (no emit-answer channel), score computed on sealed val/test with no committed seeds (no memorization/dual-path), per-call seeds unreadable by the forbidden-import-scanned candidate (no seed prediction), containment-with-exclusion kills fact-dump collapse. No fatal/unfixable issue found; verdict fixable. Files: bench/tasks/note_bottleneck/evaluate.py (the two validation fixes land in write_notes' cap check), bench/heldout.py (sealing mechanism, adequate), TASK_AUTHORING.md (emit-answer lessons — this task correctly avoids that class).

### Lens: statistics — verdict: FIXABLE

- **[FATAL]** Predictable noise_seed + content-independent paraphrase draw = free offline seed-shopping worth ~0.09-0.10 score (4x SE) — *Fix:* Two structural changes (rules alone cannot close this since the seed is surfaced): (1) make paraphrase coverage deterministic where k>=2 — ask paraphrase 0 on k-slot 0 and paraphrase 1 on k-slot 1 for every weight>=2 field (19/24 of the weight), which removes the shoppable randomness there entirely; (2) derive all remaining stochastic draws as sha256(secret_salt | noise_seed | sha256(source_text) | field | k_idx) where secret_salt is drawn by the harness at grade time via secrets, stored sealed in the record for rescore, and never surfaced mid-session (and/or stop deriving noise_seed from the agent-readable created_ts). Fix (1) alone collapses the shoppable paraphrase SD from 0.029 to 0.0007 (measured decomposition).
- **[MAJOR]** Visible-grade SE is ~0.044, not the advertised 0.025, and --k-mult does not reduce the dominant variance component (official SE ~0.030, not 0.018) — *Fix:* The same both-paraphrases-at-k=2 change from finding 1 drops the visible SE to ~0.022 (measured decomposition) and makes k-mult effective (official k-mult-2 SE → ~0.015). Then re-measure: add a ship gate requiring >=10 fresh-seed replicate grades per ladder rung (not 4), publish the analytic para/bernoulli decomposition, and restate the SE the agent sees in spec.md.
- **[MINOR]** Ladder rung values are under-measured: reference '0.355' is an n=4 batch mean (pooled n=10 gives 0.385 +/- 0.014); mid rung 0.417 is a single grade — *Fix:* Re-measure every rung with k-mult >= 4 or >= 10 replicate grades after the paraphrase fix, and quote rung values with their SEs in the authoring notes (agents will Monte-Carlo these themselves; the spec's arithmetic should survive contact).
- **[MINOR]** Headroom is larger than claimed: paraphrase-robust wording alone reaches ~0.18-0.22, below the stated 'plausible expert 0.28-0.31' — *Fix:* After the paraphrase fix, measure an actual both-paraphrase-tuned solution as the expert rung (e.g. keys covering both wordings: 'origin port=', 'destination port=', 'departed(date)='), and restate the expert band.
- **[MINOR]** Winner's-curse handling is adequate once findings 1-2 are fixed, but reroll cost is overstated and the --final-per-submission ambiguity wastes 55% of grade time — *Fix:* Adopt the margin rule (new best requires visible < best - 1 SE) now that SE is known honestly; make the default submission val-only and run --final only when the margin rule fires (keeps the sealed audit trail for exactly the programs that can be selected); re-calibrate the per-sample cost budget on the deployment machine since the cost-friction argument (25-40 iterations/hour) currently assumes the slow end.

*Notes:* Verification artifacts: /tmp/lfm_redteam/{measure_stats.py, curse_sim.py, measure_out.txt, para_rates.json, grades.json}, built on the author's own probe stack in /tmp/lfm_note (same val sources gen_source(2000+i), same frame D / weights / containment-with-exclusion scorer / dseed). LM measurements: 480 samples for per-field-per-paraphrase rates + 6 full 190-sample fresh-seed grades (1620 generations total, LFM2.5-230M fp32, threads=1, temp 0.7/top_k 50/16 new tokens). What checks out: the estimator is linear/unbiased (no ratio or extreme-value bias); signal exists at 230M (baseline 0.546, reference 0.385, floor ~0.18; no field pinned at 0% or 100% among kept fields); hallucination-credit control corroborated (dropped fields wrong 160/160, zero false credits); marginal-value-of-samples trade-off is real though ~1.7x cheaper than spec claims on this machine; same-seed bit-exact replay not re-tested but consistent with the deterministic seeding I reused. The two load-bearing defects — agent-predictable paraphrase assignments (session.py created_ts is plaintext and noise_seed is surfaced; shopping gain +0.09 at M=200, reaching the 'expert' rung dishonestly) and paraphrase-mixture variance dominating the grade (true SE 0.044 vs advertised 0.025; k-mult can't reduce it) — share one clean fix: deterministic both-paraphrase coverage on k=2 fields plus a sealed secret salt/content-keyed derivation for remaining draws. With that change the design's statistics become sound (SE ~0.022 visible / ~0.015 official, rung gaps re-measurable, shopping leverage ~0.0007). Verdict: fixable, not broken.

### Lens: harness — verdict: FIXABLE

- **[MAJOR]** Sealed test score leaks to the agent: metric names don't match session.HIDDEN_KEYS — *Fix:* Either adopt the established names (train_score/val_score/test_score, as compress_heldout does — bench/tasks/compress_heldout/evaluate.py) or extend HIDDEN_KEYS with the new keys (test_error, test_transcripts_sha256, n_test_samples; val_error/val_transcripts_sha256/n_val_samples for train-only). Add a run_checks row asserting the visible record of a --final submission contains no test_* key.
- **[MAJOR]** server_gen_seconds inside metrics breaks `bench verify --rescore` exact-equality — *Fix:* Emit server time outside the compared metrics: e.g. name it eval_server_seconds in the payload and have runner.evaluate pop it into the top-level result exactly like eval_self_cpu_seconds (add this to harness change #1), or drop it from metrics and log it to stderr / the trace only. Do not use tolerant_metrics — that would require a nonzero score_tolerance that also loosens the real score.
- **[MAJOR]** Candidate can monkeypatch shared stdlib callables used by grade(); module deletion does not stop score forgery — *Fix:* Bind everything grade() needs before load_program: module-level `_PAT = re.compile(r"[a-z0-9]+")` and use `_PAT.findall` (compiled pattern is a C type; its attributes can't be reassigned), plus local references `_urlopen = urllib.request.urlopen`, `_loads = json.loads`, `_sha256 = hashlib.sha256` captured at import. Reword anti-exploit #6 to cite the residual honestly, and add tests/broken/note_bottleneck_patch_module.py (a re.findall patcher) asserting it scores honestly/gets audit-flagged rather than 0.0.
- **[MAJOR]** Noise-seed plumbing is underspecified for the non-session paths; GOAL.md's bit-identical promise becomes false — *Fix:* Specify: runner.evaluate(noise_seed=None) → derive from a documented fixed constant when called without a session (covers determinism and baseline), `--noise-seed` overrides; session.submit passes the per-program derived seed; verify_run passes rec["noise_seed"]. Add noise_seed to the eval-log record and to GOAL_TEMPLATE's task-family note. List all three explicitly in section 10.
- **[MINOR]** verify --rescore and bench determinism hard-depend on a live, version-identical server with no preflight — *Fix:* Add a server preflight to verify --rescore and determinism for tasks with config["server"]: GET /health, compare model/dtype/torch/transformers against values recorded in each submission record (record them at grade time), and abort with a clear "server unavailable/mismatched — rescore skipped" message instead of per-record failures. Do the cross-restart determinism check before commit as planned.
- **[MINOR]** timeout_s=900 has thin margin for the official --k-mult 2 --final grade under contention or cold start — *Fix:* Either raise timeout_s (e.g. 1800 — harmless since scores are never time-based and RLIMIT_CPU still guards runaway child CPU) or document that official --k-mult 2 grades run with the server otherwise idle; ideally the evaluator scales its expectation and the harness passes a larger wall budget when --k-mult > 1.
- **[MINOR]** Server compute is misclassified as "model time" by bench/trace.py's rescale — *Fix:* Record server_gen_seconds top-level (see finding 2) and have trace.py subtract it from cum_model into a third component with its own rescale factor (server rate from a one-off calibration), or explicitly mark this family's normalized traces as not comparable cross-machine in the docs until then.
- **[MINOR]** The end-of-campaign official grade lives outside the session hash chain; its record/replay path is undefined — *Fix:* Define the artifact: e.g. `bench official RUN_DIR` (or a documented submit-with-note convention) that grades best_program.py with --k-mult 2, seals the result into the run dir as an appended, hash-chained record (kind="official", storing noise_seed, k_mult, transcript hashes), and is covered by verify --rescore with the recorded seed and k-mult.

*Notes:* Overall the design maps onto the harness unusually well: the evaluator sketch matches the eval_lib.load_program/run_program/fail/succeed idioms, the FORBIDDEN set is enforceable by the existing static scan + runtime import guard, heldout.py trivially stores [(prose, facts)] with gold facts sealed, the nonce/os._exit protocol is unchanged, child CPU stays far under cpu_s=120 (all heavy compute is server-side, outside RUSAGE_CHILDREN), and the env-allowlist in runner.py means TEXTOPT_LM_URL genuinely cannot be agent-injected once sourced from config. The audit hook's repo-file-read block already covers candidate reads of heldout_*.bin during evaluation. The enumerated harness changes are individually small (runner seed param + env keys, session seed derivation + record field, cli flags, run_checks /health skip); the two the design missed are HIDDEN_KEYS (finding 1 — the only seal-defeating one) and the timing-metric/rescore interaction (finding 2). Findings 1–4 all have one-to-two-line fixes; nothing found is structural. Residual notes: winner's-curse best-tracking margin (open question 2) needs no harness change to ship v1 since the official regrade keeps the reported number honest, but if a margin rule is adopted it touches session.submit best logic — flag as a heavier, cross-task change. run_checks' server-down SKIP-with-warning is acceptable for a separate-track family but should also gate the broken-fixture rows for this task (they too need the env/seed plumbing even though they fail before any server call — cap/type checks run pre-network, so those rows can actually run server-down if the evaluator validates notes before its first /generate; order the evaluator that way and say so). Per-call seed derivation (sha256 over noise_seed|split|source_idx|field|k_idx) has no collision risk at this scale and replay was verified bit-exact same-machine, which matches the benchmark's existing per-system-stability policy for memory tasks.

