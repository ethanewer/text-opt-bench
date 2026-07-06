> **Historical track report (2026-07-05).** This is the self-contained track's
> synthesis as written at design time. The 2026-07-06 hardening round
> (`random_tasks_hardening_report.md`) later **rejected `prompt_steer`**
> (analytic play measured at the climb ceiling across two objective redesigns)
> and **redesigned `prompt_steer_heldout`** (analytic optimum ships as the
> floor; headroom-gated sealed instances). `random_tasks_design.md` holds the
> authoritative current statuses.

# text-opt-bm stochastic task family — final design report

## 1. Executive summary

**Selected — TYPE 1 "random" (perfect information, stochastic grade):**

| task | one line |
|---|---|
| **prompt_steer** | Steer a fully-known tiny GPT's temperature-0.8 sampler with a 16-token prompt; score = Monte-Carlo miss-rate of a target token over k=25 rollouts/instance; offline analysis (incl. hardcoding found prompts) is the intended play. |
| **token_pursuit** | Interception game on a 64-node ring where the opponent's position is the tiny GPT's next sampled token; per-turn policy, score = total capture turns over K=36 episodes — the one type-1 task where variance reduction provably costs wall-clock everywhere. |
| **lm_codec** | Losslessly encode/decode fresh temperature-0.7 samples from a known tiny GPT; score = mean code length in bits over k=16 fresh draws; reconstruction-scored, entropy-floor tripwire caps every smuggling channel. |

**Selected — TYPE 2 "random generalization" (noisy grade + sealed splits):**

| task | one line |
|---|---|
| **prompt_steer_heldout** | Write a general `make_prompt(weights, targets)` search algorithm, graded by sampled discounted target-mass on sealed val/test instances it has never seen offline (score computed ON the sealed data, kv-fix applied). |
| **prompt_bandit** | Black-box prompt search against a hidden-weights tiny GPT through a hard-metered 200-call rollout oracle; explore/exploit under sampling noise IS the artifact; sealed val/test instances, in-band unbypassable sample budget. |

**Rejected:** spec_decode (noise cosmetic — optimum and score computable offline), sketch_freq (optimal play derandomizes; floor claim falsified), epidemic_heldout (broken: sealed-val noise freely averaged locally; premise unfixable in this harness).

**Shared mechanics the set relies on.** All five tasks converge on the tiny-GPT sampler as the noise source — deliberately: red-teaming showed the non-LM sources (linear sketches, SIR epidemics) admit analytic shortcuts or free local averaging that make their noise decorative, whereas LM ancestral sampling has no cheap conditional-expectation shortcut in three of the five designs, and in the other two the design is honest that noise is a scoreboard/selection phenomenon. The family shares one seed protocol (harness-drawn `os.urandom` per submission, recorded, replayed bit-exactly by `verify --rescore`; fixed `DEFAULT_SEED` when no flag so `bench determinism`/baselines need zero changes), one authority rule (**the only reportable number is the `bench official` fresh-seed high-sample regrade**; the session's visible best is a diagnostic, never a result), and one honest framing: in type-1 tasks the winner's curse lives at the submission meter and the official regrade carries the statistical design; in type-2 tasks the sealed splits additionally make local replication a rule violation rather than a physics claim.

---

## 2. Shared harness changes

All changes are one-time and shared by the whole family (~150 lines in `bench/` plus one new CLI command). Each selected task lists only per-task wiring beyond these.

**H1. `runner.evaluate` gains `seed`, `samples`, and limit-scaling.**
Add `seed: str|None = None, samples: int|None = None` to `runner.evaluate` (bench/runner.py:57); when set, append `["--seed", seed]` / `["--samples", str(samples)]` to the child cmd. When `samples` exceeds the task's `visible_samples`, scale RLIMIT_CPU and the wall timeout by `ceil(samples / visible_samples)`, capped by config `official_cpu_s`/`official_timeout_s`. This fixes the finding, confirmed independently against three tasks, that official regrades (k=400–512, 3–25 min) die on SIGXCPU under the visible-grade limits. Defaults of `None` break no existing caller (cli evaluate/baseline/determinism, session.submit, verify_run all pass nothing today).

**H2. Seed provenance: harness-drawn `os.urandom`, recorded plaintext; chain-derived seeds explicitly rejected.**
`Session.submit`, when the task config has `"stochastic": true`, draws `seed = os.urandom(8).hex()` under the fcntl lock **after** snapshotting program bytes, passes it via H1, and stores `rec["seed"]` and `rec["samples"]` in the hash-chained record. `verify_run` replays both on rescore. A chain-derived seed (`sha256(prev_sha‖program_sha‖n)`) was considered to authenticate provenance, and is **rejected**: the agent holds the chain tip and its own program bytes, so it could compute the upcoming seed *before* submitting — and in every selected task a pre-known seed is catastrophic (prompt_steer: optimize the prompt for the realized draws → score ≈ 0; token_pursuit: predict every token → capture turn 1; lm_codec: decode regenerates tokens without the blob). The residual — a hand-forged record with a locally-shopped seed passes rescore — is accepted under the cooperative model and neutralized by H3 (only `official.json`, with an experimenter-drawn seed, is trusted); THREAT_MODEL.md gains a sentence: *for stochastic tasks the hash chain + rescore authenticate consistency, not seed provenance.*

**H3. `bench official RUN_DIR` — mandatory, and the sole authoritative score.**
New CLI command (~40 lines): identifies (a) the agent's **last successful submission** (the designated final — agents are told to end on their confirmed best) and (b) the best-by-visible record; regrades both snapshots at `official_samples` with fresh experimenter-drawn recorded seeds through `runner.evaluate` (H1 limit scaling, same env allowlist — never a bare shell invocation, which loses PYTHONHASHSEED and the env hygiene); writes `official.json` containing `{program_sha256, seed(s), samples, official_scores, visible_best, curse_gap, chain_tip_sha}` beside the run, tamper-evident by embedding the submissions.jsonl chain-tip hash. Headline = min of the two official scores (adds ≤ ~0.6·SE_official of optimism, negligible vs the 1.9–2.5·SE_visible selection noise it removes — verified by two independent simulations). Per-task policy hooks: token_pursuit regrades the top-m within 2·SE two-stage; prompt_bandit averages the winner over M fresh seeds (search-luck variance). A run without `official.json` has **no reportable score** — this closes, in one rule, the "official regrade is deferrable/manual" hole that three red teams independently flagged as the single load-bearing weakness of the family.

**H4. Self-test seed policy (split by kind).**
`bench evaluate` gains `--seed`/`--samples`. For **stochastic perfect** tasks, a seedless `bench evaluate` self-test draws and echoes a *fresh* seed (recording it in TEXTOPT_EVAL_LOG telemetry), so agents cannot inadvertently hill-climb the published DEFAULT_SEED realization; `bench determinism` and `bench baseline` call `runner.evaluate` directly and stay on the bit-exact DEFAULT_SEED path. For **stochastic generalization** tasks, seedless/self-test invocations score **train only** — the evaluator computes val/test *only when `--seed` is present*, and only the harness supplies it (submit/verify/official). This closes the two independent fatal-class holes found in the type-2 candidates: the free deterministic fixed-draw val oracle, and free local averaging of sealed-split noise (the epidemic_heldout killer). Agents keep unlimited train self-tests plus the published train data for offline simulators.

**H5. `bench determinism` fixed-seed repeat.**
When config has `stochastic: true`, additionally run the initial program twice under one explicit non-default `--seed` and require bit-equality (validates the seeded path, catches honest-nondeterminism bugs at CI, not at campaign verify). The proposed "two different seeds must give different scores" nicety is dropped — it false-fails constant-output programs (lm_codec's fixed-length baseline); where a seed-sensitivity check is wanted, compare a seed-dependent metric (e.g. `floor_bits`) instead.

**H6. Session best tracking pooled per program hash; ±SE surfaced.**
For stochastic tasks, session status and best-flag ranking group records by `program_sha256` and rank by the **pooled mean** of visible scores (repeats then reduce variance instead of feeding a min-statistic ratchet), and display `±sd/√k` computed from per-record per-sample metrics. This converts the sanctioned "resubmit to average" behavior from fishing into estimation, resolving the contradiction (flagged twice) where the same behavior was both recommended and audit-flagged.

**H7. Audit additions (scoped).**
(a) Stochastic-task-scoped signatures `random.seed(`, `.setstate(`, `.getstate(`, and stdlib class-attr assignment (`Random\.\w+\s*=`) — scoped via a per-task signature bucket keyed on `config.stochastic` so tsp_budget's legitimate `random.Random(0)` idiom doesn't false-positive; `.getstate(`/`.setstate(` may go suite-wide (no honest use). (b) Extend `_module_attr_mutations` to nested attribute-chain targets and subscript stores rooted at imported modules (the `collections.OrderedDict._x` / `copyreg.dispatch_table[k]` evasions). (c) A **near-floor behavioral flag**: config may declare `implausible_below` (task floor + margin); any ok record under it is flagged for mandatory hand inspection — the stochastic analogue of the existing zero-score flag, needed because frame-walk escapes in this family pay off as *near-floor* scores, not zeros. (d) Per-task allowlist of metrics-that-may-be-zero (kills the `capped: 0` false positive class). (e) Resubmission-fishing heuristic retargeted at many *near-identical-but-not-identical* sources (whitespace variants), since identical-hash repeats are now pooled by H6.

**H8. Runner stderr-tail suppression.**
Config flag `suppress_stderr_tail: true` (set on both type-2 tasks): on the no-result error path, `runner.evaluate` reports only the exit code (plus the *first* N evaluator-written bytes), not the last-2000-bytes tail, which is candidate-writable — closing the print-then-OOM exfiltration of sealed-split observations into the visible error field.

**H9. `eval_lib` leak-safe candidate-call variant.**
`run_program(..., leak_safe_label="a held-out instance")`: on candidate exception, `fail()` with the generic label only — no traceback, no exception message, and (for the strictest task) no exception type. Today `run_program` embeds the candidate-controlled exception text, which is a live exfiltration channel for any task that hands sealed data to the candidate; compress_heldout should adopt it too.

**H10. GOAL.md stochastic branch + per-task specialization note.**
The workspace GOAL template gains a stochastic section (driven by config): the self-test score is one noisy draw, not the judged number; the recorded grade uses a fresh harness seed you cannot choose; the official result is a high-sample fresh-seed regrade, so lucky draws don't survive — end your session on your confirmed best. A per-task `specialization_policy` insertion (checkpoint_plan precedent) overrides hard-rule 4 for prompt_steer ("offline-found prompts for the published instances are legitimate") and whitelists copying the task's `model.py` into offline scripts where intended. Without this, GOAL literally forbids the intended winning strategy on two tasks.

**H11. Config keys consumed.**
`stochastic`, `default_seed` (single source of truth: evaluate.py reads it from config — no duplicated literal), `visible_samples`, `official_samples`, `official_cpu_s`/`official_timeout_s`, optional `sample_pool`, `suppress_stderr_tail`, `implausible_below`, `zero_ok_metrics`. Documented rescore costs: a full `verify --rescore` of a stochastic campaign is 15 min–1.5 h experimenter-side (stated per task; optional `--sample N` spot-check mode, full rescore remains certification).

---

## 3. Selected type-1 tasks

### 3.1 prompt_steer — steer a known sampler with its prompt

**Design (red-team fixes folded in).** Unchanged core: 3 public `(weight_seed, target)` instances on a 96d/4H/192ff/3L/ctx48 decoder; `make_prompt(weights, target) -> list[int]` (1..16 tokens); evaluator samples k=25 24-token continuations per instance at T=0.8 via sha256 counter-mode uniforms; score = mean miss-rate. Fixes now in the design, not appended:

1. **Honest cost story.** The spec now names the sequential-survival (Rao-Blackwell) estimator — accumulate `S *= (1 − p_t(target))` while sampling from the renormalized non-target distribution — as an intended analysis rung alongside the exact 1-step lookahead. Measured 50–3200× variance reduction at equal per-sample cost means local grading is near-free for a competent agent; the §5 "local MC costs minutes per variant" claims are deleted. Tension 2 is met by disclosure: the binding anti-variance mechanisms are the harness-drawn per-submission seed and the official regrade, and the spec says so. The visible grade is a scoreboard, and the scoreboard's statistics are protected by H3/H6, not by compute cost.
2. **Evaluator ordering (load-bearing).** `main()` now: builds instances → seeds global `random` with the constant 0xC0FFEE → loads program → collects/validates/detaches all 3 prompts → closes the candidate span → **only then** parses `--seed` from argv and samples. During candidate execution no frame local holds the seed; the only residual channel is `sys.argv` via a post-escape gadget (forbidden + audit-flagged). The spec's former claims ("gadgets reach globals only", "seed safe as a local") are deleted and replaced with the honest boundary statement.
3. **Weight copy pinned by test, not prose.** `_copy_weights` is a hand-written schema-exact copier reaching every innermost list of the mem_infer weight dict (wte/wpe rows, per-layer wq..w2 rows, bias/ln vectors). New broken fixture `prompt_steer_mutate_weights_inplace.py` does `weights['wte'][target][:] = [1e9]*D_MODEL` and the run_checks score-ratio block asserts it scores ≥ 2× the reference — the sharpest in-place-aliasing case, previously a silent audit-invisible score→0.
4. **Tamper check by known value.** `_sha256(b"\x00").hexdigest() == <precomputed>` and `_exp(1.0) == <precomputed>` before and after sampling (a coordinated double-patch of both the module attr and the captured binding can no longer pass); `set_candidate_active(True)` is held across the sampling loop (no downside — the candidate isn't called), and the incorrect "gc.disable stops finalizers" justification is removed (gc stays disabled for determinism of allocation order only).
5. **Official grading** per H3: designated-final + best-by-visible, k=400 each, fresh seeds, `official_cpu_s: 1600 / official_timeout_s: 2000`, `--samples` capped at 1000. run_checks headroom/ratio rows are certified at `--samples 256` (SE ≈ 0.013) with the k=25 DEFAULT_SEED row kept only for bit-exactness.
6. `rec["samples"]` recorded and replayed (k=25→40 retune knob no longer breaks rescore of old runs); GOAL specialization note (H10); fresh-seed self-tests (H4).

**Evaluator sketch (delta view).**

```python
def main():
    insts = [(_build_weights(ws), t) for ws, t in INSTANCES]     # pre-load
    random.seed(0xC0FFEE)                                        # constant, not seed-derived
    mod = eval_lib.load_program(path, FORBIDDEN, required=("make_prompt",))
    eval_lib.set_candidate_active(True); gc.disable()
    prompts = [validate_detach(eval_lib.run_program(
        mod.make_prompt, _copy_weights(w), t)) for w, t in insts]
    # candidate span stays OPEN through sampling; guard active
    _tamper_known_values()                                       # sha256/exp probe vs constants
    seed_hex, k = _parse_args_after_candidate(sys.argv)          # seed enters a frame only NOW
    hits = [_sample_hits(w, p, t, seed_hex, i, k)                # sha256(seed:i:j:t) uniforms,
            for i, ((w, t), p) in enumerate(zip(insts, prompts))]#  logits round(,9), early-exit
    _tamper_known_values(); eval_lib.set_candidate_active(False); gc.enable()
    eval_lib.succeed(round(sum(1 - h/k for h in hits)/3, 6),
        metrics={"seed": seed_hex, "k": k, "hits_per_instance": hits, ...})
```

**Numbers.**

| quantity | value |
|---|---|
| per full-rollout sample | 0.2–0.45 s (324 ms under load); RB sample same cost, 50–3200× lower variance (disclosed); exact 1-step eval 152 ms |
| visible grade | k=25 × 3 = 75 samples ≈ 10–24 s; SE ≈ 0.058 (declared knob k→40, SE 0.046) |
| official grade | k=400 × 3, fresh seed, SE 0.0144, ~4–10 min under scaled limits |
| ladder (measured, seed 1) | random 0.883 → `[t]*16` 0.517 → 1-step sweep ≈ 0.30; hard target: 0.933 → 0.900 → 0.773 |
| headroom | baseline ≈ 0.6–0.75 → reference ≈ 0.4–0.5 (1.5–1.8×); deeper methods plausibly 0.25–0.35; early rungs 4–6 SE, late rungs 0.3–1 SE |
| pool | none — metering rejected; wall-clock + official regrade are the meter |

**Anti-exploit.** Expectation-of-known-distribution shape: memorization is the task, so the boundary collapses to protocol manipulation, each surface closed — candidate never participates in sampling (typed, detached prompts); weights recursively copied (fixture-pinned); stdlib scoring path uses captured bindings verified against known values; seed enters no frame until the candidate is done; MT19937/class-patch audit signatures. **Remaining accepted risks:** a frame-walk gadget that reads `sys.argv` mid-candidate defeats both visible and official grades — detected-not-prevented, flagged by existing signatures, with a strong spot-check tell (a dynamic seed-reader must carry a full sampler inside `make_prompt`) and the H7 near-floor flag (`implausible_below ≈ 0.1`); forged-seed records pass rescore (H2 residual, neutralized by H3); same-machine libm contract (logit quantization as hygiene).

---

### 3.2 token_pursuit — intercept a sampling LM on a ring

**Design (fixes folded in).** Core unchanged: `move(weights, history, my_pos)` per turn, MOVE=2, RADIUS=1, T=0.7, 64-turn cap, K=36 episodes over weight seeds 3/5/9, score = total capture turns. Fixes:

1. **Determinism hardening.** `time`, `datetime`, `secrets` added to FORBIDDEN; spec states `move()` must be a pure function of its arguments (wall-clock-adaptive planning fails verification, and now fails the static scan). Logits quantized `round(l, 9)` before softmax; baselines re-measured post-quantization; the rescore contract is stated as *stronger* than mem_infer's: pinned interpreter + same machine, because sampling has no argmax margin and one ulp cascades a whole trajectory.
2. **Official grading two-stage (H3 policy hook).** Every distinct program within 2·SE (~120 sum-units) of the visible best is officially regraded at k=504 under scaled limits (`official_cpu_s ≈ 3600`); selection is by official score; the selected program gets one final fresh-seed estimate (select-then-fresh-estimate is unbiased). **Official-fail = disqualified, fall to next candidate** — this makes the adaptive-censoring exploit (module state hard-failing unlucky grades so only lucky draws are recorded) strictly self-defeating. No pooling of selection-triggering visible grades into official numbers.
3. **State isolation.** Fresh candidate module per weight seed (kills the cross-episode running-total censor at 3× import cost); the instance index (0–2) is passed as a fourth argument to `move` so per-weight-seed tuning needs no weight fingerprinting.
4. **Audit/metrics hygiene.** `capped` metric dropped (derivable from the recorded `turns` list) — no zero-metric false positive; stratified SE formula (`sqrt(12·Σ_ws var_ws)`) published in spec and used in the curse arithmetic; spec scopes the "SE ≈ improvement" claim to the chase→value→lookahead rungs and tells agents to rank close variants by local paired simulation (measured: Rao-Blackwellization gives no free lunch here — variance×cost at parity — so local estimation costs real wall-clock, the family's cleanest tension-2 story).
5. GOAL branch whitelists copying `model.py` for offline simulation (or ships `simulate.py` in the workspace); argv parsing fails cleanly; `--samples` capped at 5040; combined CPU budget stated honestly (~75 ms/turn candidate share of cpu_s 240, since the evaluator's own forward passes consume ~58 s worst case).

**Evaluator sketch (delta view).** As submitted, with: `FORBIDDEN += {time, datetime, secrets}`; per-weight-seed `load_program`; `move(w_cand, hist_copy, my_pos, ws_index)`; `round(l,9)` in `forward_probs`; validated argv; metrics without `capped`.

**Numbers.**

| quantity | value |
|---|---|
| per LM turn / episode | ~25 ms; 200–800 ms per episode (better policy ⇒ cheaper) |
| visible grade | K=36 ≈ 7–30 s evaluator + candidate compute (reference ≈ 20 s, depth-2 ≈ 105 s; fits cpu_s 240) |
| SE (visible) | ≈ 48–70 sum-units (1.35–1.9 turns/ep, stratified) — early rungs 4–10×, late rungs 0.3–1× |
| official | top-m at k=504 each, ~2–25 min/program under scaled limits; SE ≈ 15 sum-units; final fresh estimate reported |
| ladder (measured, replicated by red team) | drift 1000 → chase ~509 → value-iter ~389 → depth-2 ~350 → tuned ~310–330 (≈3.1×); greedy-mass trap at ~837 |
| local variance reduction | none free (RB conditional MC: sd halves, cost ×10) — tension 2 genuinely holds |

**Anti-exploit.** Policy-scored expectation, no emittable instance; opponent trajectory is policy-independent (no herding, no noise collapse); episode RNG evaluator-local with `random` forbidden statically; weights private-copy; hard-fail legality checks in-window (`type is int`, ring distance) with no penalty arithmetic to game; censoring closed by disqualify-on-official-fail + per-seed module reload + behavioral audit on failed-grade/partial-score correlation. **Remaining accepted risks:** frame-walk to the seed/rng = perfect floor (score = K) — elevated payoff class, mitigated by hard-treated `.setstate/.getstate` signatures, H7 near-floor flag (`implausible_below ≈ 60`), and spot-check; forged-seed records (H2 residual); full-campaign rescore is 30–120 min (documented); cross-platform rescore unsupported (explicitly stronger caveat than mem_infer).

---

### 3.3 lm_codec — lossless coding of fresh tiny-LM samples

**Design (fixes folded in).** Core unchanged: `encode(weights, prompt, tokens) -> bytes`, `decode(weights, prompt, blob) -> list[int]`, exact round-trip on k fresh 24-token samples at the scaled 96d model; score = mean bits. Fixes:

1. **Prompt bug fixed and re-measured.** `rng_p = _rng(seed, "prompt", i)` hoisted out of the comprehension (the submitted sketch drew 8 identical tokens — 64 possible prompts, invalidating the anti-precompute purpose and the calibration basis). Entropy/sd/floor re-measured on the corrected distribution before freezing config; a CI unit check asserts within-prompt token diversity under DEFAULT_SEED.
2. **Tripwire restated against the right bound and tightened.** Blobs are externally framed (length is free side information), so the binding bound is the one-to-one (Wyner) bound `E[L] ≥ H − log2(H+1) − log2 e ≈ 64.7`, ~7.6 bits below the Gibbs floor — a 6–8-bit margin below `floor_bits` would false-fail a legitimate length-exploiting endgame codec. New rule: `fail` if `mean_bits < one_to_one_bound(sample) − 6·sqrt(16/k)` (≈ floor − 13.6 at k=16, ≈ floor − 8.8 at k=400). Smuggling cap shrinks from ~20 bits to ~9–14 bits below the floor while the legitimate frontier (~68–73 with length exploitation) stays clear; any official grade within 3 bits of the tripwire triggers mandatory hand inspection (H7); the module-attr detector extension (H7b) covers the nested/subscript stash evasions; `min_bits = 0` (empty blob for the modal continuation) is declared legitimate and exempted from zero-metric heuristics. The spec advertises length-exploiting coding as the true endgame — which also *widens* post-port headroom, the judges' main worry.
3. **Endgame widened, honesty about caching.** The T=0.7/T=1.0 mixture ships in v1 (k/2 sequences each, temperature passed to encode/decode) — doubles the endgame surface at zero harness cost. The cost story is rewritten: the metered quantity is fresh-sample *acquisition* (0.21 s/seq, amortizable into a cached corpus with cached per-step probability vectors); the local game is corpus-size vs overfit (corpus SE = 13.7/√N; optimizing hard against a fixed corpus reinstates the curse against the official fresh-seed regrade). CRN comparisons are named as legitimate.
4. **Official authority.** `samples_official = 512` (power of two — kills the exact-float nit and the misleading config note), via H3 with `limits_scale_with_samples` capped (agent-invokable self-tests ≤ 128 samples; only `bench official` exceeds); official.json is the sole reportable number; top-3-by-visible also regraded for reporting fidelity.
5. Fresh module **and** fresh `build_weights` for phase D locked in by two broken fixtures (`weights_stash`, `framewalk_regen` — decode ignores blob, regenerates, pads to the tripwire); strict-type deep-walk inside a guarded span before `json`-independent type checks (bytes/bytearray only, plain int list out); `_parse_args` never raises; pool (3200) kept but demoted to belt-and-suspenders with pool-remaining echoed per record; GOAL stochastic branch + fresh-seed self-tests (H4); the fixed-seed-repeat determinism check (H5) replaces the dropped two-seeds-differ nicety.

**Evaluator sketch (delta view).** As submitted with: hoisted `rng_p`; phase D = fresh module + fresh weights (unchanged, now fixture-pinned); tripwire vs one-to-one bound with k-scaled margin; validated argv; metrics `{mean/min/max_bits, floor_bits, one_to_one_bound, k, seed, temps}`.

**Numbers.**

| quantity | value |
|---|---|
| per sample | generation 208 ms/seq; strong codec ≈ +0.35 s enc + 0.35 s dec |
| visible grade | k=16 (8 per temperature): 3.5 s (baseline) – 15 s (strong); SE ≈ 3.4 bits (re-measured post-fix) |
| official | k=512 fresh seed, SE ≈ 0.6 bits, ~8 min under scaled limits; sole reportable number |
| floors | Gibbs 72.3 bits; one-to-one ≈ 64.7; tripwire ≈ floor − 13.6 (k=16) / − 8.8 (k=400) |
| ladder | 192 → 144 (zero-variance rungs, disclosed) → ~125 → ~110 → ~77 → length-exploiting endgame ~68–73; early jumps 15–48 bits ≫ SE, endgame 2–5 bits ≈ 0.6–1.5 SE |
| pool | 3200 samples (≈200 grades), belt-and-suspenders only |

**Anti-exploit.** Reconstruction-scored on fresh per-grade entropy — the most robust shape in the suite: nothing fixed to memorize, evaluator draws the prompts (no low-entropy prompt selection), sampling completes before `load_program`, `random/sys/os` forbidden, phase isolation fixture-pinned, and — uniquely — the tripwire gives a *quantitative cap* on the payoff of any escape, including frame-walk. **Remaining accepted risks:** frame-walk regeneration up to the tripwire (~9–14 bits, flagged by near-floor inspection rule); same-interpreter/same-machine rescore only (a libm ulp flips a *sampled token*, failing round-trip cross-platform — contract stated in config); reference solution is the heaviest authoring lift in the repo (transformer port + integer arithmetic coder) and doubles as the headroom proof — budgeted explicitly; post-port headroom remains modest even with the mixture (accepted for a first stochastic reconstruction task).

---

## 4. Selected type-2 tasks

### 4.1 prompt_steer_heldout — general prompt search, scored on sealed instances

(Renamed from the type-2 "prompt_steer" to avoid collision with the type-1 flagship; the pair is deliberate — same substrate, opposite information model — and the spec of each cross-references the other.)

**Design (fixes folded in).** Core unchanged: `make_prompt(weights, target_set, n_gen) -> 8 ints`; discounted target mass V with γ=0.6, normalized score; 10 plaintext train / 6 sealed val / 8 sealed test instances; score = val_score; official = sealed test. Fixes:

1. **Evaluator correctness (both sketch-breaking bugs).** All split data (train.jsonl and both heldout bins, test only when `--final`) is loaded into memory **before** the first `set_candidate_active(True)` — the submitted sketch deadlocked on its own audit hook. Held-out candidate calls use the H9 leak-safe direct-call pattern: `run_program`'s traceback path never runs on val/test; failures say exactly "make_prompt failed on a held-out instance" (a probe submission can no longer exfiltrate weights/S through the visible error field — new broken fixture `exfil_raise` asserts this).
2. **The sealed splits are actually sealed from self-tests (H4).** Seedless invocations score train only; val/test are computed only under a harness-supplied `--seed`. This kills the free deterministic CRN val oracle and, combined with H8 stderr suppression, the black-box val-specialization and print-exfil channels on the agent-runnable path. Every *recorded* grade still reports train+val (and seals test), per the word_problems shape.
3. **Cost model redone for reality.** `Session.submit` always passes `--final` on generalization tasks, so a recorded grade = 10×16 + 6×48 + 8×256 rollouts + 24 make_prompt calls ≈ **190 s** (not 72–90 s); k_test lowered to 256 (test SE 0.006, still ≪ final gaps), cpu_s raised to 500; fishing arithmetic restated (best-of-N fishing now costs ~3 min/attempt and is erased by the official test number — *less* attractive, and the spec says so). §3's "experimenter reruns --final" is corrected: the official number is the unsealed `test_score` of the H3-designated records — no extra run needed, `bench official` just unseals, verifies against the chain, and writes official.json.
4. **Determinism of the candidate.** FORBIDDEN += `time, datetime, secrets, uuid`; `forbidden_attrs += SystemRandom`; the evaluator calls `make_prompt` twice on one train instance and fails on mismatch (+4 s/grade); search guidance restated in iteration counts, not seconds.
5. **Honest robustness statement + hardening.** The claim "prompts must be computed from weights never seen" is corrected: at call time the candidate holds the sealed instance's weights, so a fingerprint→prompt table works *iff* the agent decodes the heldout bins — the compress_heldout cooperative boundary, no stronger. The heldout-decode and agent-side `--seed` audit signatures therefore ship **with** the task (not deferred); instance evaluation order is shuffled per grade seed and a fresh module is loaded per split (kills the call-counter split/phase inference, which evaded the module-attr detector); per-grade sealed weight perturbation is documented as the escalation option if spot-checks ever find a table.
6. Statistics hygiene: k_train = 48 in `--train-only` mode (blind-mode guide SE 0.013, matching the calibrated val SE); curation requires the val-vs-test gap < 2·SE for **three** ladder rungs plus a deliberately val-greedy variant (not just the reference); expected ≈2·SE selection optimism in the val-vs-test comparison documented so campaign reports don't misread it as generalization failure; H6 pooled best tracking.

**Evaluator sketch (delta view).**

```python
def main():
    path, final, train_only, seed_hex, seeded = _args_validated()
    train = load_train(); val = heldout.read(...) if seeded and not train_only else None
    test  = heldout.read(...) if seeded and final else None      # all I/O BEFORE any guard
    mod = eval_lib.load_program(path, FORBIDDEN, required=("make_prompt",))
    _determinism_probe(mod, train[0])                            # call twice, must match
    tr = score_split(mod, train, "train", seed_hex, hide=False)
    if train_only or not seeded: succeed(tr, metrics={...})      # H4: seedless = train-only
    mod_v = fresh_load(path)                                     # fresh module per split
    vs = score_split(mod_v, shuffled(val, seed_hex), "val", seed_hex, hide=True)   # leak-safe calls
    ... test under --final ...
```

**Numbers.**

| quantity | value |
|---|---|
| rollout / exact forward | 17–21 ms / 9 ms; reference make_prompt ≈ 4–5 s/instance |
| recorded grade | ≈ 190 s (train 10×16 + val 6×48 + test 8×256 + 24 searches); ≤ ~15–18 submissions/hour |
| visible val SE | 0.015 (k=48 × 6); no search-luck term (deterministic make_prompt) |
| official | sealed test_score, 8 × k=256, SE ≈ 0.006; designated-final + best-by-val per H3 |
| ladder (measured) | 0.906 random → 0.82 baseline `[minS]*8` → 0.712 first-step CA → strong 0.63–0.68; early rungs 6–7 SE, refinement 1–3 SE; headroom ≈ 19× SE |

**Anti-exploit.** Emit-a-policy scored ON sealed unrecognizable instances (the kv-family fix); RNG isolation via closure-local sha256-derived streams (reseeding/monkeypatching global `random` inert — fixture-pinned); weights rebuilt before sampling (mutation inert — fixture-pinned); in-window typed validation; leak-safe held-out failures; sealed splits never executed on agent-runnable paths. **Remaining accepted risks:** heldout bins are reversibly obfuscated — a rule-breaking decode + fingerprint table beats the task and evades static audit (compact-table residual); defense = cooperative rule + shipped audit signatures + hand spot-check, honestly stated; ~2·SE val-selection optimism in the visible best (corrected at report time by H3); same-machine libm contract; frame-walk residual as everywhere.

---

### 4.2 prompt_bandit — metered black-box prompt search

**Design (fixes folded in).** Core unchanged and unique in the pool: hidden-weights instances, public 8-token target sets, `search(rollout, rng, n_slots, targets, budget)` with a hard in-band 200-call oracle budget per instance; distinct-first-hit discounted objective; sealed val/test; the explore/exploit trade under sampling noise is the artifact. Fixes:

1. **Self-test boundary sealed (the two majors).** Per H4, seedless self-tests are train-only — the deterministic DEFAULT_SEED val oracle and the plaintext leak of `val_oracle_calls` through `bench evaluate --json` both die at the root. Additionally, oracle-call-count metrics are emitted **only under `--final`** (so they exist solely inside recorded submissions, where HIDDEN_KEYS seals them) — the covert-channel closure now lives at the correct layer.
2. **Gadget surface.** FORBIDDEN += `operator`, `functools` (honest search needs neither; with getattr/eval/exec already name-banned this closes the known no-literal-dunder access gadgets, e.g. `attrgetter("__closure__")(rollout)`); the oracle closure no longer holds `weights` directly (one level of indirection through an evaluator-side container — a bar-raiser, not a proof); the spec's security claim is corrected to "literal dunder access is scan-rejected; string-hidden access is audit-flagged, not prevented"; new broken fixture `attrgetter_grab` proves the import ban. Fresh module per instance kills the call-counter split/mode inference.
3. **SNR and incentives retuned.** `initial_program.py` becomes the zero-oracle `sorted(targets)` heuristic (~1.48–1.58) so the measured headroom is genuinely oracle-driven; the pilot codex campaign has a hard gate — *if any zero-oracle strategy remains within 1 visible-sd of the best oracle strategy, raise BUDGET/targets contrast and re-calibrate before shipping*. The "≈0 held-out oracle calls ⇒ suspicious" heuristic is dropped (it false-positives the honest zero-oracle family) and replaced by the H7 near-floor flag (`implausible_below ≈ 0.5`; the frame-walk payoff here is the 0.0078 floor). Val widened 4→8 instances at k_val=24 (search-luck on the val mean falls from ~0.055 to ~0.032; scoring SE 0.043 — visible sd ≈ 0.05, still ~1× a rung); spec documents that the visible signal mildly rewards variance and that only the official number counts.
4. **Official grade honest about search luck.** Single-realization test_score carries ~0.046 sd (search luck dominates the quoted 0.011 scoring SE). Official = mean of `test_score` over **M=8 fresh seeds** of the winning program (H3 hook), ~13 min experimenter-side, SE ≈ 0.017; the recorded per-submission test numbers remain the selection-audit trail.
5. **Local-replication claim reframed.** "Impossible at any compute price" is deleted: the bins are XOR-decodable and direct `evaluate.py --final` runs exist — the protections are the cooperative rule (stated in the agent rules: do not decode `data/*.bin`, do not run `--final`), H4 train-only self-tests, H8 stderr suppression, and spot-check. Mechanical gaps closed: argv length-checked, `~15 s CPU per search() call` stated, cpu_s 400, rollout-accepts-tuple / must-return-list asymmetry documented, HIDDEN_KEYS namespace claim commented, rescore cost (~1–1.5 h/campaign) documented.

**Evaluator sketch (delta view).** As submitted with: per-instance `fresh_load`; oracle closure indirection; call-count metrics gated on `final`; `FORBIDDEN += {operator, functools}`; val = 8 instances shuffled per seed; H9 label style for held-out failures (exception type dropped too — generic label only).

**Numbers.**

| quantity | value |
|---|---|
| rollout / budget | 21.6 ms; 200 calls/instance = 4.3 s of evidence, metered in-band (unbypassable, unpoolable) |
| recorded grade | 3 train×(200+32) + 8 val×(200+24) + 6 test×(200+256) ≈ 5150 rollouts ≈ 112 s + candidate compute ⇒ ~2 min/submission |
| visible val noise | scoring SE 0.043 ⊕ search-luck ~0.032 ⇒ sd ≈ 0.05 ≈ 1× a rung |
| official | mean test_score over M=8 fresh seeds, SE ≈ 0.017; single-run sd honestly stated as ~0.046 |
| ladder | zero-oracle sorted-targets 1.48–1.58 (new baseline) → raced pairs/quads 1.35–1.42 → reference halving+refine 1.38–1.42 → skyline 1.30–1.33; campaign gate enforces oracle-driven separation |

**Anti-exploit.** The budget is enforced by construction inside the evaluator (closure counter, disarm-before-guard-close, stashed references die — fixture set: overbudget, stash_reenter, lazy_prompt, attrgetter_grab, import_random, bad_tokens); scoring streams are sha256-domain-separated from oracle and candidate streams (unreachable without inverting SHA-256); val⊥test given the program, so best-by-val selection cannot bias the official test number; prompts for sealed instances cannot be precomputed offline without breaking the decode rule. **Remaining accepted risks:** frame-walk → floor 0.0078 (near-floor flag + spot-check); heldout decode by rule-breakers (cooperative boundary, honestly stated); same-machine libm with no sampling margin (same strengthened contract as token_pursuit); the SNR retune rests on the pilot campaign gate — if the gate fails, the task ships only after recalibration.

---

## 5. Rejected at the final gate

**spec_decode** (judge 7.93 — highest-rated rejection). Decisive: the noise never bites a capable optimizer. By the Leviathan invariance the design itself invokes, the visited-context distribution is the *public* target model's own generative distribution, so the expected verification-call count is a deterministic functional of (q, γ) and the public weights — the optimal table *and its exact official score* are computable offline, and the red team's measured cached-corpus surrogate scores arbitrary variants in ~20 ms with the real ladder's rank order preserved. Every stochastic mechanism (per-sample cost, winner's curse, metering) becomes cosmetic; the exploit lens's own recommendation is to demote it to a warm-up and build the sealed-weights type-2 sibling — which is unspecced and unmeasured, so nothing shippable survives this gate. Secondary confirmations that the sketch was never run against eval_lib (safe_builtins would reject every legitimate program; import_budget misread by ~6 orders of magnitude) reinforced the decision. The sealed-weights sibling is the natural next candidate for a future round.

**sketch_freq** (judge 7.33). Decisive: the stochastic machinery is vestigial under optimal play, and two calibration pillars were falsified. (1) The "provable ≥85 KiB information floor" was broken by the O(1)-invertible multiply-shift scramble — a legal dense count-by-rank table scores ~0.26 vs the reference's 2.28, invalidating every calibrated number. (2) Spec-legitimate derandomization (fixed hash constants selected by *exact* 0.3 s conditional evaluation) removes the dominant noise component, leaving visible SE ~0.6% against 2–10% rungs — noise decision-irrelevant. (3) The shared-tracemalloc-window gate drifts with sample count (visible-pass/official-fail cliff, ~2.8 KB run-to-run peak wobble vs the claimed 59 B), and the 21× tracemalloc cost asymmetry falsifies the "same intrinsic compute wherever it runs" story. Each is individually fixable, but together they amount to a redesign of a task whose own statistics red team concluded the tiny-LM source fits the type better than any linear sketch can.

**epidemic_heldout** (judge 7.53). Rejected on an unresolved **broken** verdict: the marquee claim — sealed val/test noise "cannot be reduced locally at any price" — is structurally false in this harness, because the agent-runnable evaluator must both decode the sealed graphs and honor `--seed` for rescore, so val_score is freely averaged to arbitrary precision in ~10 min of the box (and `--final` prints the official answer key). Every proposed fix abandons the noisy-visible-grade premise. The statistics replication also failed the calibration wholesale (baseline CV ~1% vs claimed 18%; percolation-tipping bimodality relocated rather than removed; 3-graph splits carry 20–40% family variance that k cannot reduce; resubmission-curse bias exceeding a real rung), and the harness lens showed the pool/cost model ignored the always-`--final` submit path. Note: the H4 seedless-train-only rule invented for the surviving type-2 tasks would repair the *access* hole here too — but the calibration failures and the fact that the noise would then be experimenter-side-only mean the salvage is a different (deterministic-visible) task than the brief asked for. Worth revisiting as a plain generalization task; not as a random one.

---

## 6. Funnel appendix

| candidate | type | judge mean | advances | red-team verdicts (exploit / stats / harness) | outcome |
|---|---|---|---|---|---|
| prompt_steer | random | 8.00 | 3 | fixable / fixable / fixable | **SELECTED** (flagship; RB-honest cost story, ordering + copy + tamper fixes) |
| spec_decode | random | 7.93 | 3 | fixable / fixable / fixable | REJECTED (noise cosmetic: optimum + score computable offline; sealed sibling unspecced) |
| lm_codec | random | 7.60 | 3 | fixable / fixable / fixable | **SELECTED** (prompt-RNG fix, one-to-one-bound tripwire, T-mixture, official k=512) |
| epidemic_heldout | random_gen | 7.53 | 3 | **broken** / fixable / fixable | REJECTED (fatal: sealed-val noise freely averaged; premise unfixable; calibration failed replication) |
| token_pursuit | random | 7.37 | 3 | fixable / fixable / fixable | **SELECTED** (two-stage official, censor closure, time-forbidden, quantized logits) |
| sketch_freq | random | 7.33 | 3 | fixable / fixable / fixable | REJECTED (derandomized optimal play + falsified floor + gate drift ⇒ redesign-scale rework) |
| prompt_steer (T2) | random_gen | 7.30 | 2 | fixable / fixable / fixable | **SELECTED as prompt_steer_heldout** (train-only self-tests, leak-safe calls, real cost model) |
| prompt_bandit | random_gen | 7.20 | 3 | fixable / fixable / fixable | **SELECTED** (self-test seal, operator ban, M-seed official, SNR gate) |

**Pre-ship checklist carried by every selected task:** H1–H11 landed atomically; broken-fixture suites as listed per task (including the new behavioral score-ratio blocks in run_checks); headroom rows certified at high `--samples` plus a DEFAULT_SEED bit-exact row; a pilot codex campaign (gpt-5.5 low) per task confirming multi-rung climbs, measuring the visible-vs-official curse gap, and exercising each task's declared tuning knob (prompt_steer k, token_pursuit T, lm_codec temperature mix, prompt_steer_heldout γ/k_val, prompt_bandit BUDGET + SNR gate) before the config literals are frozen.