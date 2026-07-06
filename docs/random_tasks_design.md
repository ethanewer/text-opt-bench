# Random & random-generalization tasks — final design (2026-07-05, hardened 2026-07-06)

Two new task kinds for text-opt-bm, designed and adversarially vetted by two
multi-agent workflows (79 agents total; 64 candidate designs; every finalist
passed a full spec + three-lens red-team — exploit / statistics / harness),
then **hardened by a third workflow** (29 agents): empirical one-shot-resistance
probes (blind frontier-strength first programs scored on prototype ladders),
redesigns where the probes found weakness, a live concurrency load test, and a
measured resource-governance layer — see `random_tasks_hardening_report.md`.

- **Self-contained track** (toy-GPT noise source, pure stdlib): 42 candidates
  → 8 shortlisted (all prototyped end-to-end in /tmp) → 5 selected, 3 rejected
  at the design-time final gate (a 4th, `prompt_steer`, was subsequently
  rejected at the 2026-07-06 hardening gate — the current toy-track set is 4
  tasks; see §2.5). Full synthesis: `random_tasks_toylm_report.md`.
- **Real-LM track** (LFM2.5-230M as the stochastic grader, owner-approved):
  22 candidates → 6 refined specs → 3 selected, 1 rejected, 2 alternates.
  Full specs + red-team reports: `random_tasks_reallm_specs.md`.

Every number below was **measured**, not estimated (toy-GPT prototypes in
/tmp; LFM2.5-230M fp32 / torch-threads=1 on the M5, ~1,400+ probe
generations).

## 1. The two task kinds

**`random` (type 1)** — works like a perfect-information task, but the grading
function is stochastic. The noise is *natural* (token sampling from an LM at
temperature > 0 — never synthetic noise added to a deterministic score). The
agent has complete knowledge of the score-defining distribution; nothing is
hidden except future random draws. The grade the agent sees is a small-sample
estimate; the **official** grade is the same expectation estimated with more
samples. Grading must carry a real cost, or averaging-away variance must be a
genuine trade-off.

**`random_generalization` (type 2)** — same noisy-grading mechanics plus
train/val/test structure: train fully visible (still noisily graded), val/test
sealed (`bench/heldout.py`), official score on test with more samples.

Config encoding: `kind` stays `perfect`/`generalization`; a new
`"stochastic": true` flag marks the family (H11 lists all new config keys).

## 2. Final selection

### Type 1 — `random`

| task | substrate | one line | status |
|---|---|---|---|
| ~~`prompt_steer`~~ | toy GPT (96d/3L) | ~~16-token prompt steers a fully-known sampler~~ | **REJECTED at the hardening gate (2026-07-06)**: the blind probe nearly saturated the 1-step reference rung, and BOTH measured redesign attempts (late-window objective, 8-target coverage objective) were empirically killed — deterministic analytic play reaches the effective ceiling (max experimentation headroom ≈ 1.5 SE across 18 ceiling measurements on 12 instances — candidate A: 6 curated pairs × W∈{8,12}; candidate B: 6 weight seeds — vs the required ≥4). The steering substrate survives only as type-2 `prompt_steer_heldout`. Replacement candidate if a 4th type-1 task is wanted: `lm_copyedit` (real-LM alternate; needs its own probe first). Details: hardening report Addendum B |
| `token_pursuit` | toy GPT (mem_infer size) | Interception game on the 64-token ring: the opponent's position is the LM's next sampled token (T=0.7); per-turn policy, score = capture turns over K=36 episodes. The cleanest cost story: variance reduction provably costs compute everywhere (measured: Rao-Blackwellization gives no free lunch — variance×cost at parity). | ready (probe: LOW risk twice — blind shots 629/662, below even the naive chase rung); re-pin outer ladder anchors on the real evaluator (probe measured drift 804 vs spec 1000, value-iter 493 vs 389 — suspect pursuit2.py MOVE/R defaults) |
| `lm_codec` | toy GPT (96d/3L) | Losslessly encode/decode **fresh** T=0.7/1.0 samples drawn per grade; score = mean bits over k=16. Reconstruction-scored: you cannot shrink fresh entropy by memorizing; an entropy-floor tripwire quantitatively caps every smuggling channel (~9–14 bits). | ready (probe: LOW risk, blind shot 138 bits mid-ladder, ~43 bits ≈ 7.5× the k=16 mix SE of climb); re-publish the ladder numbers in the shipped spec.md's scoring section (the sanitized draft omits them): mix floor 90.28, deep end ~95, k=16 SE ~5.7, hardcoded-0.7 penalty 6.12 bits/seq; keep the current weight layout (its n96=17 vector count defeated the probe's shape-fingerprint attack); heaviest authoring lift (reference = transformer port + arithmetic coder) |
| `taboo_cluesmith` | **LFM2.5-230M** | Write a taboo-legal clue per fixed word so the real LM *names* the referent in sampled guesses; score = miss rate. Fully-known distribution; per-word semantic prompt search — the 230M model echoes attributes ("yellow", "hot") unless the framing is right. | gated: curation run must find 20 words with achievable clues (probe yield 9/60 at 1–4 attempts; else ship W=16) + pilot campaign. Probe: LOW risk decisively — blind shot 0.844 landed *behind* the definitional baseline (0.734 measured; expect ~0.73 not ~0.91 on a curated list); control clue-table quality — the taboo rule forced legal rewording of 5/16 honest clues |

### Type 2 — `random_generalization`

| task | substrate | one line | status |
|---|---|---|---|
| `record_qa` | **LFM2.5-230M** | Flagship. Prompt template + exemplars + parser for one-hop numeric QA over generated operational records (distractor quantities, word-numbers, "a dozen", coreference — all measured at 230M competence, no multi-step arithmetic). Train 120 visible; val 150 / test 400 sealed with **non-plaintext generator seeds** (closes the word_problems regenerate hole). Score = val error under sampled decoding; official = `finalize`: sealed val k=5 (750) + sealed test 400×k=3 (1,200) = 1,950 samples. | gated: calibration must hold the baseline→reference gap ≥ 0.10 (≈3–4 SE) via the word-render lever; pilot campaign is go/no-go |
| `prompt_bandit` | toy GPT (hidden weights) | Black-box prompt search through a hard-metered 200-call rollout oracle per instance — the explore/exploit allocation under sampling noise IS the artifact; the meter is in-band and unbypassable by construction. Sealed val/test. | ready; SNR pilot gate (zero-oracle strategies must lose by ≥1 sd) |
| `prompt_steer_heldout` | toy GPT | The type-2 twin of `prompt_steer`, **redesigned after its probe (moderate risk)**: the shipped `initial_program.py` IS the analytic first-step coordinate-ascent optimum (the strongest pure-reasoning play becomes the floor); sealed instances are rejection-sampled on a measured ≥0.03 (=3.2 SE) CA-vs-deep-policy headroom gate; splits 10/6/8, K={train 16, val 128, test 512}. Post-redesign, the strongest honest first program lands at −0.9%..+1.6% of the shipped floor (vs ~70% of the old ladder). | **hardened, needs pilot**: frozen-ladder aggregate + gate acceptance rate were still generating at report deadline; codex pilot must confirm the first-submission distribution; consider raising the gate to 0.035–0.04 |
| `record_qa_vote` | **LFM2.5-230M** | The owner's cost requirement turned into the artifact: a global budget of 225 samples across 150 val questions; the program decides per-question how many samples to buy (1–3) and how to vote. Buying variance reduction exactly where p(correct) is mid-range is the skill; the budget is metered server-side inside one grade. | gated (recalibration + redesign `hardened_needs_pilot`): reference measured WORSE than baseline as-shipped — allocation-gain ladder must be re-measured, budget-walk shuffled. Probe addendum: the allocation axis resisted one-shotting in both measured first shots (margins 0.000 and 0–0.033), but prompt-axis first-shot variance spans 0.28–0.52 across blind draws — **recalibration must make the allocation axis the scored artifact**, or one-shot risk is program-luck. (Numbers in the one-liner are the v1 spec; the Addendum-A redesign that earned hardened_needs_pilot rebalances to val 270 questions / 540 budget units with token-tiered costs and n 1–5 — recalibration decides which ships, and the governance envelope must be re-derived if v2 is adopted) |

**Diversity check.** Mechanisms: adversarial pursuit, coding of fresh entropy,
semantic prompt search, program induction over a document family,
explore/exploit budgeting (×2, one in-band and one artifact-level), and
steering under sealed instances. Noise sources: toy-GPT ancestral sampling
(4 tasks) and real-LM sampling (3 tasks). One deliberate pairing remains an
experiment: `record_qa` ↔ `record_qa_vote` (same family; template quality vs
sample-budget allocation). The originally-planned `prompt_steer` ↔
`prompt_steer_heldout` information-regime pair died with the type-1 task's
rejection at the hardening gate.

## 2.5 One-shot resistance — measured (hardening round, 2026-07-06)

Method: per task, a frontier-strength agent wrote its single best program
**blind** (reasoning only, from a sanitized agent-facing spec), and the program
was scored on a spec-faithful prototype harness against the measured ladder.
Full table, replication history, and redesigns: `random_tasks_hardening_report.md`.

Headline: **8/8 probed; 6 passed outright** — blind first shots landed at
25–50% of the ladder or *below the naive baseline* (record_qa three-for-three:
every strong blind program chose few-shot prompting by pure reasoning and all
measured worse than the naive instruction; taboo_cluesmith's confident
universal framing landed behind the definitional baseline; token_pursuit's
clever Markov interceptor lost to naive chase twice). Of the two exceptions,
`prompt_steer_heldout` was **redesigned and hardened** (ship the analytic
optimum as the floor + headroom-gated sealed instances; measured accepts
3.2–7.2 SE), while `prompt_steer` was ultimately **rejected**: two independent
objective redesigns (late-window, 8-target coverage) were both empirically
killed — on this substrate deterministic analytic play reaches the effective
ceiling (max ≈1.5 SE of experimentation headroom across 18 ceiling
measurements on 12 instances). The final set is therefore **3 type-1 + 4 type-2 tasks**.

Two structural lessons the probes taught, now design rules for this family:
1. **Judge one-shot risk by the best draw, not the average.** Replicated
   probes (three blind draws each on record_qa_vote and prompt_bandit)
   straddled the verdict line — one draw cracked record_qa_vote's prompt axis
   to near the deep end (0.2833 vs deep ~0.25) while the other two stayed near
   the naive baseline (0.5222 behind it, 0.4417 just ahead, vs 0.4917). Gates must bind the
   axis the task is actually about (vote: allocation; bandit: oracle-driven
   search) so a lucky draw on a side axis can't harvest the ladder.
2. **The analytically-reachable rung must be the shipped baseline.** Wherever
   pure reasoning can reach a rung (prompt_steer's 1-step sweep,
   steer_heldout's coordinate ascent), ship that rung as `initial_program.py`
   so only the experimentation-driven climb scores. This is the
   word_problems-v4 lesson generalized.

## 3. How the family meets the requirements (and where honesty is required)

| requirement | how it is met |
|---|---|
| natural noise | All seven tasks: token sampling from an LM (toy GPT via sha256 counter-mode uniforms; LFM2.5 via seeded torch sampling). Red-teams killed every non-LM noise source that was tried (sketch hashing, SIR epidemics) as analytically shortcuttable or freely averageable. |
| visible grade = samples, official = more samples | Uniform protocol: per-submission seed drawn by the harness from `os.urandom`, recorded, replayed by `verify --rescore`; official grade = `bench official` fresh-seed high-sample regrade (H3), the **sole reportable number**. |
| cost / trade-off | Three honest regimes, stated per task (see spectrum below). |
| type 1 is not generalization | Everything defining the distribution is public (weights/word lists/frames/sampling params); overfitting-to-noise is punished by the official regrade automatically. |
| type 2 splits | Sealed via heldout.py with the new H4 rule: **seedless self-tests score train only** — val/test are computed only under a harness-supplied seed. This single rule killed the two fatal-class holes found in type-2 candidates (deterministic val oracle; free local averaging of sealed-split noise). |

**The noise-bite spectrum (honest framing, from the red-teams).** Where the
cost-of-grading tension actually binds differs by task, and the specs say so
explicitly rather than overclaiming:

1. **Binding everywhere** — `token_pursuit` (no free variance reduction
   exists), `prompt_bandit` / `record_qa_vote` (the meter is inside the scored
   episode), `record_qa` (sealed instances: local simulation impossible
   without breaking the decode rule).
2. **Binding as acquisition cost** — `lm_codec`: fresh samples cost ~0.21 s
   each wherever they are drawn; the local game is corpus-size vs overfit
   (optimizing against a cached corpus reinstates the winner's curse at the
   official fresh-seed regrade).
3. **Scoreboard-level only** — `taboo_cluesmith`: the model is public, so a
   competent agent can estimate its true score locally (LFM sampling at
   ~0.16 s/gen is cheap). The binding mechanisms are the harness-drawn
   recorded seed and the official regrade; local estimation still costs real
   wall-clock inside the time box. Track 1 rejected `spec_decode` for this
   property taken to its limit (exact score computable offline in 20 ms), and
   the hardening round ultimately rejected `prompt_steer` for the adjacent
   failure (analytic play saturates the climbable range) — taboo_cluesmith
   survives because its expectation is Monte-Carlo-only, per-word outcomes are
   unpredictable from priors (measured), and the probe put a strong blind
   first shot behind the honest baseline.

## 4. Shared harness protocol (H1–H11)

Authoritative text with rationale in `random_tasks_toylm_report.md` §2
(~150 lines in `bench/` + one CLI command, shared by all seven tasks):

- **H1** `runner.evaluate(seed=, samples=)` → child `--seed`/`--samples`;
  CPU/wall limits scale with sample count (official regrades die on SIGXCPU
  otherwise — verified against three tasks).
- **H2** Seed provenance: `Session.submit` draws `os.urandom(8).hex()` under
  the lock, records it plaintext in the hash-chained record; rescore replays
  it. Chain-derived seeds are explicitly rejected (agent can precompute them;
  a pre-known seed is catastrophic in every selected task).
- **H3** `bench official RUN_DIR` — regrades the designated-final and
  best-by-visible snapshots at `official_samples` with fresh experimenter
  seeds, writes tamper-evident `official.json`. **A run without official.json
  has no reportable score.** Per-task hooks: token_pursuit two-stage top-m;
  prompt_bandit M=8 fresh-seed mean (search-luck variance).
- **H4** Self-test seed policy: stochastic-perfect seedless self-tests draw
  fresh seeds (no hill-climbing the published default realization);
  stochastic-generalization seedless invocations score **train only**.
- **H5** `bench determinism`: fixed-seed repeat must be bit-exact.
- **H6** Session best-tracking pools records by `program_sha256` (repeats
  become estimation, not min-statistic fishing); visible ±SE surfaced.
- **H7** Audit: `random.seed/.setstate/.getstate` + class-attr assignment
  signatures (task-scoped); nested/subscript module-attr stash detection;
  **near-floor behavioral flag** (`implausible_below`) — escapes in this
  family pay off as near-floor scores, not zeros; zero-ok metric allowlist;
  near-duplicate resubmission heuristic.
- **H8** `suppress_stderr_tail` config flag (type-2 tasks): the stderr tail is
  candidate-writable — a sealed-data exfiltration channel on the error path.
- **H9** `eval_lib.run_program(leak_safe_label=...)`: candidate exceptions on
  held-out instances report a generic label, never the exception text
  (compress_heldout should adopt this too).
- **H10** GOAL.md stochastic branch: self-test = one noisy draw; recorded
  grade uses a fresh harness seed; official = high-sample fresh-seed regrade,
  so lucky draws don't survive. Per-task `specialization_policy` (without it,
  GOAL literally forbids the intended winning strategy on two tasks).
- **H11** New config keys: `stochastic`, `default_seed`, `visible_samples`,
  `official_samples`, `official_cpu_s/_timeout_s`, optional `sample_pool`,
  `suppress_stderr_tail`, `implausible_below`, `zero_ok_metrics`, and
  `resource_class: cpu|lm` (governance v1.1). resource_class tasks also get a
  conditional `RLIMIT_AS = 2 GB` in the runner's `set_limits` (measured eval
  peak is 19.5 MB — 100× headroom) — **never** applied to tracemalloc-scored
  tasks, where an address-space limit changes allocator behavior.

## 5. Real-LM serving infrastructure (R1–R6)

The three LFM tasks relax "no third-party deps / fully self-contained" into:
**the evaluator stays pure stdlib** and talks over localhost to an
experimenter-managed model server. This is a separate optional track, exactly
as `docs/task_fit_ranking.md` anticipated for LLM-dependent tasks.

**Measured feasibility (M5, torch 2.12.1 / transformers 5.13.0, fp32,
`torch.set_num_threads(1)`):** ~72 tok/s single-threaded (bf16 needs ≥2
threads for 76); short-answer generations 0.12–0.9 s, few-shot prompts up to
~1.8 s; **same-seed sampling is byte-identical, different seeds diverge**
(verified repeatedly, incl. by the workflow's own probes); ~1.25 GB steady RSS
per fp32 server process (early estimate ~0.95 GB, superseded by the load-test
measurement — see R5).

- **R1 Server**: `tools/lm_server.py` in a pinned venv (torch+transformers),
  fp32, threads=1, serialized request queue, localhost only.
  `POST /generate {messages, max_new_tokens, temperature, top_k,
  repetition_penalty, seed, n, run_token} → {texts, token_ids}`. A version
  fingerprint (model sha, torch/transformers versions, dtype, threads) is
  returned with every response and recorded in every submission.
- **R2 Auth + metering**: per-run bearer token issued at session creation;
  the server meters samples per run token server-side — unlike a CLI-side
  pool, the agent cannot bypass it (candidates are additionally forbidden
  `socket`/`urllib` by the AST scan + import guard). Retries are idempotent
  (request ids), fixing the double-decrement finding.
- **R3 Failure semantics**: transport/server failure ⇒ the grade **aborts**
  as `infra_error` (recorded, non-scoring, excluded from rescore) — never
  "instance counted wrong", which both corrupts scores and permanently breaks
  `verify --rescore` (two red-teams flagged chain poisoning independently).
- **R4 Rescore contract**: same machine + matching server fingerprint;
  recorded seeds replayed; transcript SHA-256s recorded (sealed for held-out
  splits) and compared. On fingerprint mismatch, verify reports "environment
  changed" instead of false tampering. Cross-machine comparison: LM server
  time sits in the trace's *model time* component (un-rescaled) — per-system
  stability only, same class of caveat as the memory tasks.
- **R5 Concurrency — superseded by the measured governance layer v1.1**
  (`random_tasks_hardening_report.md` §4, live load test 2026-07-06): ONE
  shared server (fp32 steady RSS measured **1.25 GB**, not the ~0.95 GB first
  estimated; startup 3.1 s) using **`ThreadingHTTPServer`** — the stdlib
  single-threaded `HTTPServer` hard-drops connections at client concurrency 8
  (measured; generation stays lock-serialized, determinism byte-identical
  across load, restarts, and independent processes). Campaign scheduler
  classes: config `resource_class: cpu|lm`; reference 10-core/32 GB box runs
  N=16 slots as **cpu_slots=12 + lm_slots=4** (independent caps = mutual
  anti-starvation). Queue latency is linear in depth (p50 1.30/4.16/7.59 s at
  depth 1/4/8), so LM `timeout_s = n_req × per_req × 4 × 1.6 × 1.25`; LM
  grades are scored on server `lm_seconds`, never wall — a timeout is
  `infra_error` (R3), never a score. Toy-eval wall inflates ×3.1 under 12 CPU
  hogs with byte-identical scores — size cpu-class `timeout_s` at ×4–5 quiet
  wall. `tools/run_campaign.py` gains `--lm-slots`, class-aware fill loop, and
  shared-server lifecycle (start once, health-poll, SIGTERM at campaign_done).
- **R6 Determinism check**: `bench determinism` on LFM tasks runs the
  fixed-seed repeat through the server and requires byte-identical
  transcripts + scores (verified achievable).

## 6. Rejected candidates (design gate and hardening gate, both tracks)

| candidate | track | judge | decisive reason |
|---|---|---|---|
| `prompt_steer` | toy | 8.00 (was the type-1 flagship) | **Rejected at the HARDENING gate (2026-07-06)**, not the design gate: blind probe nearly saturated the 1-step rung, and both measured objective redesigns (late-window; 8-target coverage, run to completion post-recovery) showed deterministic analytic play reaching the effective ceiling — max ≈1.5 SE of experimentation headroom across 18 ceiling measurements on 12 instances vs the required ≥4. |
| `spec_decode` | toy | 7.93 (highest-rated rejection) | Noise cosmetic: by the design's own Leviathan invariance the optimal draft table AND its exact official score are computable offline (measured 20 ms surrogate preserves the ladder's rank order). |
| `epidemic_heldout` | toy | 7.53 | Broken: sealed-val noise freely averaged locally (agent-runnable evaluator must decode the graphs and honor `--seed`); calibration failed replication wholesale. H4 would fix the access hole but the salvage is a deterministic-visible task — a different brief. |
| `sketch_freq` | toy | 7.33 | Optimal play derandomizes (fixed hash constants via exact conditional evaluation → SE 0.6% vs 2–10% rungs); the "provable 85 KiB floor" was falsified by an O(1)-invertible scramble. |
| `reroll_fee` | real-LM | 6.8 | Headroom collapses: myopic per-job stopping thresholds are exactly optimal (claimed deeper rungs mathematically unreachable; the hint-level handle worth 0.000 on its own measured pmfs). |
| `lm_copyedit` | real-LM | 6.97 | Alternate, not broken — lowest-scored survivor; overlaps selected shapes. First substitute if a type-1 pilot gate fails. |
| `note_bottleneck` | real-LM | 7.33 | Alternate — most novel concept ("compress for an LM reader") but one fatal (seed-shopping; fix known, unvalidated) + understated SE. Revisit after the selected type-2 tasks ship. |

## 7. Implementation plan

Recommended order (dependencies, cheapest validation first):

1. **H1–H11** as one atomic harness change (all seven tasks assume them).
   Extend `tests/test_session.py` for seed/samples records, pooled best, H3.
2. **Self-contained four** (`token_pursuit`, `lm_codec`,
   `prompt_steer_heldout`, `prompt_bandit`) — no new infra. Per task: the
   broken-fixture suite named in its spec, `bench determinism` fixed-seed row,
   headroom rows certified at high `--samples`, then the **mandatory pilot
   codex campaign** (gpt-5.5 low, TASK_AUTHORING step 5) measuring multi-rung
   climbing and the visible-vs-official curse gap before freezing config
   literals. `lm_codec` last (reference solution is the big lift).
   Hardening-round gates added per task: `prompt_steer_heldout` — generate
   the headroom-gated curated splits (~2.5 h owner-side, nice'd, resumable),
   add the `prompt_steer_slow.py` broken fixture, then pilot; `token_pursuit`
   — re-pin outer ladder anchors on the real evaluator; `prompt_bandit` — the
   pilot must confirm the extrapolated deep end exists (the 13×-budget skyline
   probe did not separate from the reference) before headroom claims go in
   GOAL.md, and the sealed test k=256 grade is non-negotiable.
   (`prompt_steer` was rejected at the hardening gate — see §2.5 and the
   hardening report Addendum B.)
3. **R1–R6 + governance v1.1** server infra + venv bootstrap + campaign
   integration: `tools/lfm_server.py` on `ThreadingHTTPServer` with startup
   pinning asserts (threads=1, fp32), 2.5 GB RSS watchdog (restart-safe:
   determinism holds across restarts; in-flight requests become `infra_error`),
   request ceilings (max_new_tokens ≤128, n ≤8, ≤2 outstanding per run token),
   per-run-token meters, `GET /health` with fingerprint; `run_campaign.py`
   `--lm-slots` + class-aware fill loop; `bench/runner.py` `resource_class`
   RLIMIT_AS + config timeout literals from the governance envelope table
   (notably record_qa 2100 s / vote 4000 s at queue depth 4 — the drafts'
   900/1500 s would kill legitimate grades under load; 4000 covers the
   formula's 3800 s worst case with margin).
4. **`record_qa`** (+ its `tools/` generator with sealed seeds) → calibration
   gates → pilot campaign (go/no-go on the ≥0.10 gap).
5. **`taboo_cluesmith`** — run `tools/curate_taboo_words.py` (300-word × 5-clue
   bank, ~35 min compute) to lock the 20-word list, then pilot.
6. **`record_qa_vote`** — only after its recalibration gate passes (reference
   must beat baseline by allocation, not prompt quality; budget-walk shuffle;
   paired diagnostics).

Every real-LM task also inherits the R-series fixes for the majors its
red-teams found (HIDDEN_KEYS additions for new sealed metrics, `--final`
cost-model corrections, train-only guide score in blind mode) — these are
enumerated per task in `random_tasks_reallm_specs.md` and must be checked off
before its pilot.

## 8. Open questions for the owner

1. **record_qa_vote's gate** — if recalibration can't separate allocation
   skill from prompt skill, ship 3 type-2 tasks (family still in-spec).
2. **taboo word count** — W=20 preferred, W=16 acceptable if curation yield
   falls short.
3. **bf16 vs fp32 server freeze** — fp32/1-thread is fastest per-thread and
   what all numbers were measured at; bf16 halves RAM but changes every
   transcript (pick before any records are created).
4. **Sample pools** — the toy-LM track rejected CLI-side pools as bypassable
   (kept only as belt-and-suspenders on `lm_codec`); the real-LM track meters
   server-side where it's enforceable. Confirm you're happy with wall-clock +
   official-regrade as the binding cost on the sole "scoreboard-level" type-1 task (`taboo_cluesmith`).
