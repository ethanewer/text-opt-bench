# text-opt-bm stochastic family — hardening round synthesis (2026-07-06)

Inputs: 8 one-shot-resistance probes (blind strong first programs, scored on spec-faithful harnesses), 1 completed redesign (prompt_steer_heldout), 1 measured concurrency load test, 1 resource-governance spec. All numbers below are measured; artifacts under /tmp/hardening/.

---

## 1. One-shot-resistance table

Requirement (1): a strong agent's FIRST program must land well short of the deep end, and the remaining climb must require experimentation against the stochastic system.

| task | first shot (probe) | ladder position (same-harness anchors) | headroom to deep end | risk | iterativeness |
|---|---|---|---|---|---|
| **prompt_steer** | 0.4150 ± 0.0135 (k=400×3) | baseline 0.6150 ± 0.0132 → **first shot** → 1-step ref 0.3558 ± 0.0127 → deep ~0.30 (0.25–0.35). ~63% of range; ties the reference on 2/3 instances; lost only on instance 2 (0.387 vs 0.200) | 0.115 to best-guess deep end = 2.0 visible-SE (8 official-SE); only 1.1 visible-SE if deep end is 0.35 | **moderate, leaning high** — the probe was blind; with the shipped spec (full schema), a strong first program lands ~0.36, at the reference rung | Mixed. The 0.415→0.356 rung is deterministic reasoning + a 16×64 sweep. Only below 0.356 is genuinely experimental (multi-step proxies fail there), and that segment is thin: 0.01–0.10 abs = 0.2–1.8 visible-SE |
| **token_pursuit** | 662.4 ± 20.3 (K=288×8 grades) | drift 803.8 ± 45.4 → **first shot** → chase 522.2 ± 25.8 → value-iter 492.8 ± 39.9 → deep ~310–330. Below the FIRST rung; ~29% of measured range (~50% on spec's drift=1000 scale) | ~330–350 units = 5–7× visible SE; entire ladder ahead | **low** | Predominantly experimental: blind self-prediction (~450) off by >3 SE; faithful value-iter repro scored 493 not spec's 389 (details take measure/retry); greedy-mass trap measures ~837 (worse than chase); sub-value-rung gains 0.5–2 visible-SE |
| **lm_codec** (T=0.7/1.0 mix) | 138.19 ± 0.85 (k=128; T=0.7 subset 132.00 ± 1.17) | baseline 192.000 exact → 6-bit rung 144 → **first shot** → Huffman ~125 → order-1/2 ~110 → mix deep end ~95 (Gibbs floor 90.28). 49–55% by percent, but no middle rung cleared | ~43 bits ≈ 7.5× the k=16 visible SE (~5.7); even a perfect blind port hardcoding T=0.7 lands ~98 (6.12 bits/seq KL penalty on the T=1.0 half, measured 114.97 ± 1.94) | **low** (boundary moderate) | Yes: its blind fingerprint guesses were wrong in run-only-detectable ways (predicted 78–90, measured 138.2); mid-climb validation signal is measured rate only; T-mix discovery is intrinsically experimental; endgame tricks 0.35–1.5 visible-SE |
| **prompt_steer_heldout** | 0.7555 ± 0.0074 (6 fresh instances, k=200) | random 0.9082 ± 0.0045 → baseline 0.8483 ± 0.0060 → **first shot** → CA ref 0.7128 ± 0.0078 → deep 0.63–0.68. ~50% of range | 5.0–8.4 visible-SE; but spec-informed first shot expected at CA rung ~0.71 (~70% of range) | **moderate (high end in deployment)** → **REDESIGNED**, see below | Pre-redesign: mixed — CA rung pure-reasoning-reachable from shipped model.py; only 0.03–0.08 (2–5 SE) above it experimental |
| **prompt_bandit** | 1.6222 ± 0.0190 harness scale (~1.48 spec-equivalent) | baseline 1.7065 ± 0.0176 → zero-oracle 1.6634 ± 0.0166 (**first shot** only weakly beats it, t=−0.8) → reference halving 1.5557 ± 0.0153 → deep ~1.49 (extrapolated). ~39% of range; did not clear middle rungs (raced pairs/quads 1.35–1.42 spec-scale) | ~0.134 = 2.7× visible submission sd, ~7× own SE | **low** (borderline-solid) | Mixed, leaning experimental: strong author invested in a measured-wrong prior (bigram + const screening) and overestimated itself by ~0.4; remaining rungs 0.04–0.08 = 1–2× visible sd, found only via measured probes |
| **taboo_cluesmith** | 0.8438 ± 0.023 (cluster SE 0.057; replicates 0.8828/0.8047) | naive 1.0 → **definitional baseline 0.7344 ± 0.039 — the first shot is BEHIND it by ~2.4σ** → mid 0.67–0.72 → deep 0.30–0.45. Coverage-corrected (real spec discloses the word list) ~0.73–0.78: still the baseline rung | 0.39–0.54 = 12–16.5 visible-SE; own prediction 0.45, landed 0.39 worse | **low** | Strongly experimental: the confidently-reasoned universal framing was near-worthless per-word (banana 0/16, piano 1/16 vs camera 13/16); which framing unlocks which word is unpredictable from priors (spec curation: 9/60 words, specific framings only) |
| **record_qa** | 0.5625 ± 0.074 (paired same-seed) | baseline 0.4875 ± 0.077 (bit-exact repro) → reference 0.4625 → deep 0.42–0.45 → campaign-best ~0.38. **Below the bottom rung: −75% of range** | 0.10–0.18 abs = 3.3–4.7 shipping-SE (SE 0.030) to reference/deep, ~6 SE to campaign-best — after first climbing 0.075 back to baseline | **low** | Strongly experimental — three-for-three strong blind first shots (0.6875, 0.60, 0.5625) all chose few-shot by pure reasoning and ALL measured worse than the naive baseline; remaining rungs are 0.01–0.06 wording micro-deltas vs grade SE 0.03 |
| **record_qa_vote** | 0.5222 ± 0.020 (M=60/B=90 analog) | naive baseline 0.4917 ± 0.025 → deep ~0.25 analog. Below the bottom rung; **isolated allocation margin achieved: exactly 0.000 on all 3 seeds** (180 paired instances; best case +0.032) | ~0.27 = 4.2× per-grade SE, 8–9× the real 150-instance SE (0.031); prediction 0.30, missed by ~0.22 the wrong way | **low** (this program); **family worst case moderate** — a prior blind first shot (copy-the-line) hit 0.2833 on all 3 seeds on the prompt axis | Strongly iterative: confidently wrong kind-level bets (few-shot day prompts measured worse; votes spent on p≈0 kinds bought nothing, exactly as the marginal-value formula predicts). The **allocation axis resisted one-shotting in both measured first shots** (0.000 and 0–0.033) |

**Redesigned task, post-redesign row:**

| task | redesign one-liner | post-redesign first-shot position | verdict |
|---|---|---|---|
| prompt_steer_heldout | Ship the analytic first-step-CA optimum AS the initial program (floor = the strongest reasoning play), sharpen k_val 48→128 (SE 0.0162→0.0095), and rejection-sample sealed instances on a measured ≥0.03 CA-vs-deep-policy headroom gate | Strongest honest first program (exact forward + prefix-cached CA + 2-step DP + untuned n=30 MC, plus a guarded worst-case variant) lands **+0.016/−0.009 of the shipped floor** (0.7909/0.7098 vs baseline 0.7820/0.7256), i.e. ~0–15% of the gated band vs ~70% of the old ladder; sometimes below the floor (untuned MC is net-negative, measured drift to −0.061). Guaranteed band ≥0.03 = 3.2 SE at k=128; observed accepts 0.030–0.068 (3.2–7.2 SE) | **hardened_needs_pilot** |

---

## 2. Design deltas to fold into specs

### 2.1 prompt_steer_heldout (full redesign — implementation-ready)

Objective, sampling, sealing, seeds all **unchanged** (V = Σ_{j<8} 0.6^j·[x_j∈S], T=1.0 ancestral, logits round 9dp, sealed bins, os.urandom per-submission seed + `verify --rescore` replay, `--final` k=512 test regrade, courtesy-seeded candidate determinism, FORBIDDEN list, resubmission-fishing audit). Four knobs:

1. **`initial_program.py` = the exact first-step-P(S) coordinate ascent** (prefix-KV-cached, 2 sweeps, ~5–6 s/instance, deterministic — verified). spec.md states: "the shipped program already maximizes the exact first-step distribution; gains above it are multi-step properties of the sampling dynamics — measure them."
2. **`config.json` K = {train:16, val:128, test:512}** (k_val 48→128; visible SE 0.0162→0.0095 at measured sd_norm 0.24–0.275). Envelope: visible ~150 s, worst-case `--final` ~300 s at 8 s/call, under cpu_s 400. Spec documents the self-limiting budget: >~8 CPU-s/call risks failing your own official regrade (24 calls under cpu_s 400).
3. **Headroom-gated curated splits** (train 10 / val 6 / test 8; generator stays uncommitted): rejection-sample uniform (wseed, sha256-uniform S) accepting only instances where score(shipped CA) − score(derandomized champion-guarded MC-refinement policy @18k forwards) ≥ 0.03 normalized at k=320 paired rollouts. Acceptance ~10–15%; ~6 min/accepted instance owner-side, ~2.5 h for 24, nice'd, resumable. **Two hard-won implementation requirements:** (a) the gate policy carries the full CA prompt as champion and switches only on a paired big-n win — naive gates measured −0.061..+0.012 pure winner's-curse noise; (b) the gate policy is derandomized per instance exactly as courtesy-seeded candidates are, so the gate measures the realization a graded program reproduces. Rejects pin at exactly +0.0000 (guard confirmed).
4. **Support files**: tests/solutions = the tuned guarded-concentration policy; new broken fixture `prompt_steer_slow.py` (make_prompt busy-loop → cpu_s breach → fail), locking the bounded-cost requirement; no harness changes beyond the k table.

Rejected rebuilds (measured, keep in the spec's design-notes so nobody retries them): deep-horizon objective (prompt influence dies after ~3 sampled tokens; best-vs-CA gaps 0.003–0.021; ladder collapses); first-hit coverage (anti-aligned with strong self-loops, P(t|[t]*8) up to 0.58; band ~0.01; naive MC drifts to/below CA).

Pilot must confirm before freeze (evidence gaps, stated bluntly): the full 6-instance k=200 frozen-ladder aggregate and the 10–15% acceptance rate were **still generating at report deadline** — interim per-instance numbers came through the identical scoring path but the aggregate is unconfirmed; a codex pilot should confirm the realistic first-submission distribution (a lucky knob-guessing first shot could harvest part of the band on some instances); consider raising the gate to 0.035–0.04 (~2× generation cost) since gate gaps regress ~0.005–0.015 on fresh seeds. The deep rung is a lower bound (oracle search finds more), so headroom is understated, not overstated.

### 2.2 Minor deltas from probe evidence (not redesigns)

- **prompt_steer — instance-curation gate (new, required):** re-curate the shipped instances toward instance-2-like cases where the 1-step optimum is a weak proxy for the 24-token hit objective (the probe's only separation from the reference rung: 0.387 vs 0.200), and/or add a 4th hard instance. Acceptance target: spec-informed first shot (≈ the 1-step reference rung) sits ≥3–4 visible-SE above the certified deep end. Pin the deep end while doing it — the risk verdict flips on whether it is 0.30 or 0.35.
- **lm_codec — adopt the T=0.7/1.0 per-index mix** and re-publish spec §8: mix Gibbs floor 90.28 (T=0.7: 71.70 sd 13.9 — matches the old 72.3; T=1.0: 108.85 sd 11.7), achievable deep end ~95, k=16 visible SE ~5.7 (per-seq sd 22.6), hardcoded-0.7 penalty 6.12 bits/seq on the T=1.0 half (measured 114.97 ± 1.94 vs floor 108.85). Keep the current weight layout — its n96=17 vector count defeated the probe's shape-fingerprint attack (a biases-folded layout with n96=29 might not have).
- **token_pursuit — re-pin outer ladder anchors on the real evaluator:** probe reproduces chase (~522 vs spec ~509) but measures drift 803.8 vs spec's ~1000 and value-iter 492.8 vs spec's ~389; likely a prototype-config mismatch (pursuit2.py defaults MOVE=3/R=2 vs task MOVE=2/R=1). Placement conclusions hold on either scale.
- **taboo_cluesmith — re-pin the baseline rung expectation:** an honest definitional clue table measures 0.7344 ± 0.039 on the achievable-word-enriched proxy list, not the spec's ~0.91; the shipped 20-word curated list should expect the baseline rung nearer 0.7 unless clue-table quality is controlled. Also note the taboo rule bites honest clues (5/16 baseline entries needed legal rewording).
- **prompt_bandit — document ladder narrowness:** the full baseline→deep-end span is ~0.22–0.28 with per-rung steps 1–2× visible noise; the sealed test k=256 grade is what makes late-campaign rung separation meaningful — keep it non-negotiable. The deep end itself is an extrapolation (the 13×-budget skyline probe, 1.5988 ± 0.0356, did not separate from the reference 1.5557 ± 0.0153); the pilot should confirm the deep end exists before headroom claims go in GOAL.md.

---

## 3. Drops and demotions

**Nothing is dropped.** All eight tasks pass or can pass requirement (1); six passed outright with blind first shots at ~25–50% of range or below the bottom rung.

- **prompt_steer: demote from "ready to implement" to "gated"** (instance-curation gate, §2.2). Blunt assessment: this is the one task where the evidence for requirement (1) is genuinely weak. The blind probe reads "moderate," but blind is the wrong condition — the real task ships the full schema, and the probe showed a schema-blind program already ties the 1-step reference on 2/3 instances. A spec-informed strong agent's first program should be expected at ~0.36, leaving an experimental segment of only 0.2–1.8 visible-SE, and if the true deep end is 0.35 the first shot is within 1.1 visible-SE of it. The hardening lever (curate instances where 1-step optima are weak proxies) is proven on its type-2 twin but **unmeasured for the type-1 task** — gate it on that measurement, don't ship on the twin's evidence.
- **prompt_steer_heldout: hardened_needs_pilot** — treat as gated on (a) completion of the frozen 6-instance k=200 ladder + acceptance-rate confirmation, (b) a codex pilot confirming first-submission distribution.
- **record_qa_vote: stays gated** (the pre-existing recalibration gate — reference must beat baseline via allocation, not prompt quality). New evidence cuts both ways and should be recorded in the gate: the allocation axis resisted one-shotting in both measured first shots (margins 0.000 and 0–0.033), but prompt-axis first-shot variance spans 0.2833–0.5222 across blind programs — the recalibration must ensure the scored artifact is the allocation axis, or the task's one-shot risk is program-luck.
- **taboo_cluesmith, record_qa: existing gates unchanged**; both passed one-shot resistance decisively (record_qa three-for-three below baseline; taboo 12–16.5 SE of headroom).

---

## 4. Resource-governance spec (tightened, v1.1)

Grounded in the measured load test (/tmp/hardening/loadtest/), `bench/runner.py` limit machinery, `tools/run_campaign.py`, design doc §5 R1–R6. Total load-test runtime ~9 min; cleanup verified (no stray lfm_server/hog processes).

### 4.0 Measured evidence base (stated once; everything below derives from it)

- **LFM determinism under load: PASS.** Same-seed output byte-identical (sha256[:16]=ef1584a72e26b5f2) across: quiet ×3, 8 nice CPU hogs ×3 (server compute 0.95 s → 1.51 s, ~1.5×), 4 concurrent mixed-seed clients, a server restart, and two independent concurrent server processes.
- **Queue latency is linear in depth** (64-token gen, per-request lm_seconds ~1.0–1.25 s quiet):

  | client conc | p50 wall | p95 wall | throughput |
  |---|---|---|---|
  | 1 | 1.30 s | 1.47 s | 0.80 req/s |
  | 4 | 4.16 s | 4.44 s | 0.95 req/s |
  | 8 | 7.59 s | 8.69 s | 0.99 req/s |

  Per-request client timeout ≥ depth × 1.3 s × 1.6 (hog) × 2 (safety) ≈ **depth × 4 s**; ≥70 s if ever run at raw conc-16 against one server. Clients retry connect errors.
- **Critical bug (fixed):** stdlib single-threaded `HTTPServer` (backlog 5) hard-fails at client conc 8 — dropped connect()s stall ~75 s to errno 60. Fix is one line: `ThreadingHTTPServer` (applied in /tmp/hardening/loadtest/lfm_server.py); the existing lock still serializes generation; determinism verified unchanged.
- **CPU-class contention:** toy eval (pursuit K=36) alone: score 7.777778, wall 5.0 s; under 12 nice hogs: score identical (all 36 per-episode values byte-equal), wall 15.4 s — **×3.1 wall inflation**; user CPU 4.7 s → 7.7 s — **×1.6 CPU inflation**. Therefore: size wall timeouts at ×4 quiet (×5 for margin), size `cpu_s ≥ 1.6 × quiet_cpu_worst × 1.25`, and never score the stochastic family on evaluator CPU — use deterministic score-only metrics or server `lm_seconds`.
- **RAM:** toy eval peak RSS 19.5 MB (identical alone vs loaded); LFM server ~1.25 GB steady fp32; two servers = 2.5 GB, startup 3.1 s each (HF-cached). Worst simultaneous rlimit exposure 16 × 2 GB + 1.25 GB = 33 GB theoretical, never realized (<3 GB realistic).

### 4.1 Campaign scheduler classes

Config key (H11 addition): `"resource_class": "cpu" | "lm"`. cpu = prompt_steer, token_pursuit, lm_codec, prompt_steer_heldout, prompt_bandit; lm = taboo_cluesmith, record_qa, record_qa_vote.

Slot formula for C cores / M GB, N total job slots (jobs are codex-wait-dominated; oversubscription N > C is fine — only grading bursts CPU):

- `lm_slots = min(4, max(2, floor(C/3)))` — bounds worst-case queue depth D = lm_slots, which directly sets the queue-aware timeout. D=4 keeps worst-case grade wall under ~30 min even for record_qa_vote.
- `cpu_slots = N − lm_slots`, `N ≤ min(16, C + 6)`.
- **Reference 10-core/32 GB: N=16 → cpu_slots=12, lm_slots=4.** RAM closes per §4.0.

Enforcement in run_campaign.py (~25 lines): `--lm-slots` (default 4); read `resource_class` at job build; in the fill loop launch only if `count(running, class) < class_slots`, scanning `pending` past a blocked head — the anti-starvation guarantee: independent class caps mean 8 queued LM jobs can never occupy the 12 cpu slots, and vice versa. If any lm job is in the plan: start the shared server before the first launch, health-poll, SIGTERM at campaign_done.

### 4.2 Per-task worst-case envelopes (conc-16)

LM timeout formula: `timeout_s = n_req × per_req_s_max × D × 1.6 × 1.25`, D = 4. LM grades are scored on server `lm_seconds`, never wall; a timeout is `infra_error` (R3, non-scoring) — generous LM timeouts are pure runaway guards, not score inputs.

| task | class | quiet visible wall | conc-16 wall (worst legit) | CPU-s/grade (quiet→loaded) | peak RAM | LFM gens/grade | **cpu_s** | **timeout_s** |
|---|---|---|---|---|---|---|---|---|
| prompt_steer | cpu | 10–24 s (sweeps ≤160 s) | 40–100 s, ≤640 s | ≤160 → ≤260 | ~20 MB | 0 | **300** | **700** (was 600) |
| token_pursuit | cpu | 7–30 s; ≤163 s w/ depth-2 | ≤650 s | ≤163 → ≤260 | ~25 MB | 0 | **300** (was 240) | **700** (was 400 — would kill legit depth-2 grades under load) |
| lm_codec | cpu | 3.5–15 s (strong ≤60 s) | ≤240 s | ≤80 → ≤130 | ~30 MB | 0 | **240** | **400** |
| prompt_steer_heldout | cpu | ~150 s at k_val=128; ~300 s w/ --final | ≤800 s | ≤200 → ≤320 | ~25 MB | 0 | **400** | **900** (was 600) |
| prompt_bandit | cpu | ~96 s (2736 rollouts w/ --final) | ≤400 s | ≤100 → ≤160 | ~25 MB | 0 | **300** | **700** |
| taboo_cluesmith | lm | 25–28 s | 160×0.17×4×1.6×1.25 ≈ 220 s | ~5 (client parses only) | ~30 MB + shared server | **160** (12k probe pool/run) | **60** | **300** |
| record_qa | lm | 80–160 s | 150×1.75×4×1.6×1.25 ≈ 2100 s | ≤60 | ~30 MB (+server) | **150** (25k soft / 250k hard) | **120** (draft's 600 was CPU the client never uses) | **2100** (was 900 — too tight at D=4); `final_timeout_s` 5400 |
| record_qa_vote | lm | 150–470 s | 225×2.1×4×1.6×1.25 ≈ 3800 s | ≤60 | ~30 MB (+server) | **≤225** (metered exact) | **120** | **4000** (was 1500; covers the 3800 s formula worst case with margin); final 12–25 min quiet |

Official regrades use H1 limit scaling capped by `official_cpu_s/official_timeout_s` (prompt_steer 1600/2000, token_pursuit ~3600, lm_codec ~1600, per specs). Note prompt_steer_heldout's redesign updated its quiet wall (k_val 128) — envelope above already reflects it.

### 4.3 Server governance (one shared server)

Shared beats per-job on the measurements: per-job = 4 × 1.25 GB churn + 3.1 s startup × job turnover for zero determinism benefit (byte-identical across independent processes and restarts); one queue serves lm_slots=4 comfortably (throughput holds ~1 req/s to depth 8).

- **`ThreadingHTTPServer` mandatory** (§4.0 bug); generation stays lock-serialized.
- **Startup pinning, refuse-to-start otherwise:** `torch.set_num_threads(1)`, fp32 asserted (`model.dtype == torch.float32`); fingerprint (model sha, torch/transformers, dtype, threads) in every response and `GET /health` (R1/R4). Launch under `nice` (measured: nice + hogs still byte-identical).
- **RSS watchdog:** self-check every 30 s; restart at **2.5 GB** (2× the 1.25 GB steady state — leak signal, not load signal). Safe: determinism holds across restarts; an in-flight request during restart surfaces as transport failure ⇒ `infra_error` under R3 — recorded, non-scoring, excluded from rescore — never a wrong score. **No RLIMIT_AS on the server** (torch allocator); watchdog + `--max-total-samples` are its caps.
- **Request ceilings** (server rejects, 400/429): `max_new_tokens ≤ 128`; prompt ≤ 4096 tokens; `n ≤ 8`; per-run-token meters — taboo 12,000-sample probe pool; record_qa 25k/session soft telemetry + 250k lifetime hard; record_qa_vote exact `/meter/open{budget}` decrement. Fairness: max 2 outstanding requests per run token (a runaway agent loop cannot starve another job's grade in the FIFO queue).
- **Lifecycle:** `GET /health → {fingerprint, total_samples, by_tag, rss_mb, uptime}`; run_campaign starts once if any lm job (poll /health, ~3.1 s warm start), records fingerprint in launcher.jsonl, SIGTERM at campaign_done; evaluators fail fast on unreachable server ⇒ infra_error.

### 4.4 Official-regrade scheduling

Officials run experimenter-side, after the campaign, on a quiet machine (quiet-wall numbers, no inflation). Policy: **cpu-class officials max 2 concurrent; lm-class strictly serial** (one queue; serial keeps quiet per-request cost). `bench official --queue` FIFO under these caps; schedule overnight.

| task | per snapshot | per run (2 snapshots unless noted) | 5-run campaign |
|---|---|---|---|
| prompt_steer (k=400×3) | 4–9.5 min | 8–19 min | 40–95 min |
| token_pursuit (k=504, two-stage top-m≤3) | 2–7 min | 6–21 min | 30–105 min |
| lm_codec (k=400) | ~6 min | ~12 min | ~60 min |
| prompt_steer_heldout (k=512 test) | ~3–5 min | 6–10 min | 30–50 min |
| prompt_bandit (M=8 fresh seeds × 96 s) | ~13 min | ~26 min (winner only) | ~65 min |
| taboo_cluesmith (640 gens) | ~100 s | ~3.5 min | ~17 min |
| record_qa (`finalize`, 1950 samples) | 20–35 min | 20–35 min (best only) | 100–175 min |
| record_qa_vote (test 600+400 samples) | 12–25 min | 12–25 min (best only) | 60–125 min |

60-run campaign bound (~7.5 runs/task): Σ ≈ 13–21 h serial; at 2-way cpu parallel + serial lm ≈ **8–13 h** — one overnight + one off-day window. Bounded and predictable because every official's sample count is a config literal (H3) and lm officials run at quiet per-sample cost.

### 4.5 Hard guarantees ("no single grade can exceed…" — each bound has a named enforcement mechanism)

| task | wall | CPU | RAM | LFM gens | enforced by |
|---|---|---|---|---|---|
| prompt_steer | 700 s | 300 s | 2 GB | 0 | wall: `subprocess.run(timeout=timeout_s)`; CPU: `RLIMIT_CPU` preexec; RAM: **new** `RLIMIT_AS=2 GB` in `set_limits` for `resource_class` tasks (**never** for tracemalloc-scored tasks); gens: n/a |
| token_pursuit | 700 s | 300 s | 2 GB | 0 | same |
| lm_codec | 400 s | 240 s | 2 GB | 0 (pool 3200/session) | same + Session pool (belt-and-suspenders) |
| prompt_steer_heldout | 900 s | 400 s | 2 GB | 0 | same |
| prompt_bandit | 700 s | 300 s | 2 GB | 0 (200-call in-band oracle) | same + evaluator-side budget closure |
| taboo_cluesmith | 300 s | 60 s | 2 GB | 160/grade, 12k probe pool/run | wall/CPU/RAM as above; gens: server meter per run token |
| record_qa | 2100 s | 120 s | 2 GB | 150/grade, 25k soft / 250k hard | server meter (tag-billed) |
| record_qa_vote | 3600 s | 120 s | 2 GB | 225/grade exact | server `/meter/open` decrement, cross-checked in metrics |
| — LFM server (not a grade) | — | 1 core (threads=1) | restart @ 2.5 GB RSS | `--max-total-samples` lifetime | startup pinning + RSS watchdog + meter |

RLIMIT_AS sizing note: 2 GB is 100× headroom over the 19.5 MB measured eval peak; 1 GB is also safe if a tighter bound is wanted. Campaign-level: ≤12 cpu + ≤4 lm concurrent (class slots); worst simultaneous RAM 33 GB theoretical / <3 GB realistic; every timeout/infra failure is non-scoring (R3), so no governance mechanism can corrupt a score or the rescore chain.

**Repo deltas implied** (implementation pass, not applied): `bench/runner.py` — read `resource_class`, conditional `RLIMIT_AS` in `set_limits`, pass `eval_lm_seconds` through; `tools/run_campaign.py` — `--lm-slots`, class-aware fill loop, server lifecycle; `tools/lfm_server.py` — ThreadingHTTPServer, pinning asserts, watchdog, meters, /health; config literals per §4.2 (revised from drafts: token_pursuit cpu_s 240→300 and timeout_s 400→700; prompt_steer timeout_s 600→700; prompt_steer_heldout 600→900; record_qa 900→2100 + final 5400; record_qa_vote 1500→3600).

---

## 5. What changed vs docs/random_tasks_design.md (editor's checklist)

1. **§2 type-1 table, prompt_steer row:** status "ready to implement" → "**gated:** instance re-curation toward cases where the 1-step optimum is a weak proxy (probe: blind first shot 0.4150 ± 0.0135 nearly saturated the 1-step reference 0.3558 on 2/3 instances; spec-informed first shots expected ~0.36) + deep-end pin (0.30 vs 0.35 flips the risk verdict)."
2. **§2 type-1 table, lm_codec row:** adopt the T=0.7/1.0 per-index mix; update the one-liner ("fresh T=0.7/1.0 samples") is already consistent, but re-publish the ladder numbers in its spec §8: mix floor 90.28, deep end ~95, k=16 visible SE ~5.7, hardcoded-0.7 penalty 6.12 bits/seq. Note: keep the current weight layout (shape-fingerprint defense, measured).
3. **§2 type-2 table, prompt_steer_heldout row:** replace the one-liner with the redesigned task ("shipped initial program IS the analytic first-step optimum; sealed instances headroom-gated ≥0.03 = 3.2 SE at k_val=128; splits 10/6/8, K={16,128,512}"); status "ready to implement" → "**hardened, needs pilot** (frozen-ladder aggregate + acceptance rate + codex pilot pending)."
4. **§2 type-2 table, record_qa_vote row:** append to gate text: "recalibration must produce a measured positive allocation margin (both hardening first shots measured 0.000 and 0–0.033); prompt-axis first-shot variance spans 0.28–0.52, so allocation must be the scored artifact."
5. **New section (suggest §2.5): one-shot-resistance results** — insert the table from §1 above; headline: 8/8 probed, 6 pass outright (token_pursuit, lm_codec, prompt_bandit, taboo_cluesmith, record_qa, record_qa_vote), prompt_steer demoted to gated, prompt_steer_heldout redesigned and hardened.
6. **§5 R5: replace entirely.** Old text (one server per LM job, ~0.95 GB, cap LM concurrency ~6) is superseded by the measured governance layer: one shared **ThreadingHTTPServer** (stdlib HTTPServer drops connections at conc 8 — measured), lm_slots=4 / cpu_slots=12 on the reference box, queue-aware timeouts (latency linear in depth: p50 1.30/4.16/7.59 s at depth 1/4/8), scores on `lm_seconds` never wall. Also correct the RSS figure: fp32 server measures **1.25 GB** steady (doc says ~0.95 GB).
7. **§4 H11:** add `resource_class` to the config-key list; note conditional `RLIMIT_AS=2 GB` for resource_class tasks only (never tracemalloc-scored tasks).
8. **New section (or appendix): Resource-Governance Layer v1.1** — paste §4 above verbatim (slot formula, envelope table with the four revised timeout literals + token_pursuit cpu_s, server governance incl. 2.5 GB RSS watchdog and request ceilings, official-regrade scheduling with the 8–13 h 60-run bound, hard-guarantees table).
9. **§7 implementation plan:** step 2 — add per-task gates: prompt_steer instance-curation measurement before its pilot; prompt_steer_heldout curated-split generation (~2.5 h owner-side, resumable) + `prompt_steer_slow.py` broken fixture + pilot. Step 3 — fold in the server fixes (ThreadingHTTPServer, pinning asserts, watchdog, /health, meters). Add: token_pursuit anchor re-pin (probe: drift 803.8 vs spec 1000, value-iter 492.8 vs 389 — suspect pursuit2.py MOVE=3/R=2 defaults) when the real evaluator lands.
10. **§2 taboo_cluesmith row / its spec:** baseline (definitional) rung expectation on the shipped curated list is ~0.73, not ~0.91 (measured 0.7344 ± 0.039 on the achievable-word-enriched proxy list); control clue-table quality during curation.
11. **Header claim update:** the family now has measured one-shot-resistance evidence in addition to the red-teams — record_qa is three-for-three blind strong first shots at or below the naive baseline; record_qa_vote's allocation axis resisted one-shotting in both measured attempts.

**Weak-evidence flags (do not paper over in the doc):** prompt_steer_heldout's post-redesign numbers are partly interim (pipeline still generating at deadline); prompt_bandit's deep end is extrapolated and its skyline probe did not separate from the reference; token_pursuit's outer spec anchors don't reproduce on the probe harness; prompt_steer's "moderate" is a blind-condition number that overstates real resistance; record_qa_vote's family-level one-shot verdict rests on two first shots with opposite outcomes.

---

# Addendum A — full redesigns recovered from the run journal

The synthesis above embeds the prompt_steer_heldout redesign summary; the full
structured outputs of both completed refiners are preserved here verbatim.

## record_qa_vote — hardening redesign (from the first resume)

**Verdict: hardened_needs_pilot**

# record_qa_vote v2 — hardening redesign

Root cause of the probe failure: "copy the line, compute in the parser" is pure-reasoning-reachable, and 71% of the val mass (single-record lookups) sits at ~0.9 accuracy under it, so the prompt ladder (0.46→0.28) collapsed in one shot and only ~1 visible-SE of allocation remained. The fix: make that insight the *published starting point*, move the scored mass onto kinds where the 230M model's measured weakness (selection/transformation across records) puts the achievable band mid-range and unpredictable, and deepen the allocation economics so budget targeting is worth several SE and only measurable empirically.

## Design changes (knobs / objective / instances)

1. **Baseline promotion (ladder re-basing).** `initial_program.py` IS the copy-line + parser-side-compute program (the program that one-shotted v1), with its per-kind accuracy table published in spec.md. The recall rung no longer scores; the visible ladder starts at 0.537 under the new mix.
2. **Mix rebalance onto selection-hard kinds** (val M=270): lookup 16, numword 27, distract 16, day 21, coref 43, **maxq 60, count 43 (new kind, from gen2), total 44**. Aggregates = 54%, +coref = 70% of mass. All aggregate answers are functions of *multiple* records; measured on LFM2.5-230M, the model transcribes verbatim well but cannot filter or transform: 8 plausible strategies measured — copy-all-matching p .36/.15/.50 (maxq/count/total), few-shot copy-all .36/.30/.43, largest-first .45(A-noise: pooled .36), digit-listing .04, name-listing .00, compressed transcription .00 ("SANDPUP"/"PYLOWELLS" garbling), count-direct .10, copy+COUNT .20. Neither the level nor the *ranking* is reasoning-predictable, and single-seed small-sample winners did not replicate across eval seeds (V9, V3) — replication on the 120-row train bench is forced.
3. **Allocation deepened**: n ∈ 1..5 (was 1..3); budgets ratio 2.0 in UNITS: train 120/240, val 270/540, test 540/1080.
4. **Token-tiered sample cost (new knob)**: a sample costs **1 unit if max_new_tokens ≤ 48, else 2 units** (published). Clamp reserves 1 unit per later instance; if an instance can't afford its template's cost the evaluator substitutes the DEFAULT template (cost 1), deterministically — still charged. Consequences (all measured): long-output retrieval prompts (the only working aggregate rescue, 80 tok) compete with votes for slack; budget-naive vote spreading self-destructs (the naive first-shot variant lost 16/60 instances to DEFAULT-fallbacks, scoring 0.417 vs 0.333 budget-sane); hash-subset vote sizing on the sealed instance set must be conservative; and worst-case grade wall is *capped* (≤ B/2 long samples ≈ 570 s at val — old worst was 470 s, same envelope).
5. **Scoring objective unchanged** (val error rate, min; strict-< best-tracking): the allocation axis now shows up in the score because vote targeting at the measured mid-p kinds is worth ~2 SE and mis-targeting (votes on p≈0, small-answer-space collisions on count where errors cluster at k−1) actively loses.
6. **Noise mechanics & anti-exploit preserved verbatim**: eval_seed drawn/recorded per grade, sha256 seed(i,j) scheme (sample j identical under any allocation — paired diagnostics intact), transcript hash chain, server-side meter, `verify --rescore` replay, `--final` = sealed test (540/1080) + forced-n=1 diagnostic; temp 0.7/top_k 50 fixed; forbidden-import guard, doc-inserted-once, special-token stripping, 80-token output cap (now doubly load-bearing: it truncates whole-log dumps — keep docs ≥10 records), generic errors, val_kind_acc hidden.
7. **Published in spec.md**: the tiered-cost walk with worked corners, the plurality formula, baseline per-kind p table, and an explicit warning that per-kind p under *new* prompts must be re-measured on train (the spec's own decomposition numbers did not transfer to new prompt wordings).

## Why the climb is experimentation-shaped (evidence from this validation)
- Round-1/2/3 variant search: plausible prompts scored 0.00–0.45 with no a-priori pattern; transcription collapses, selection fails, verbatim copying wins — discoverable only by measurement.
- Selection-under-noise is real: seed-S n=1 winners (V9 largest-first, V3 few-shot-count) regressed on seed A; the pooled level is what a disciplined train-bench measurement finds.
- The demonstrated deep rung's biggest single gain is a pure **error-pattern discovery**: count answers cluster at k−1 because the 80-token dump truncates mid-list → parser adds 1 when the text ends mid-line. Measured: seed A 2/10→5/10, held-out seed S 1/10→4/10, few-shot family A 0/10→2/10. No reasoning path reaches "+1".
- Remaining measured/spec-documented levers below the demonstrated rung (vote targeting at official-scale slack 140+ units, few-shot easy-kind lifts that are kind-dependent — helps day/digits, hurts wordnum —, cost-1 aggregate prompt engineering, corrector-class discoveries on maxq/total) support a conservative plateau floor of ~0.336.

## Validation protocol (all measured this session; 468 LFM generations ≤ 500, fp32, torch threads=1, niced; /tmp/lfm_bench/venv python 3.12/torch 2.12.1/transformers 5.13.0 = spec server env)
Scaled probe M=60/B=120 units (mix 5/7/5/6/9/11/10/7), exact tiered clamp + sha256 seed scheme, instances/ids shared with the prior probe where kinds overlap (byte-identical regeneration verified). Artifacts: /tmp/hardening/record_qa_vote/redesign/ (mix3.py, harness3.py, harness3c.py tiered clamp, firstshot.py, deep2.py, variants*.py, probe_*.py, gencache.json, results_full_*.json, dryrun_clamp.py).

### Measured numbers
Probe scale (M=60/B=120 units, eval seed A, end-to-end under the exact tiered clamp): baseline (published copy-line program) 0.4667 (96 units, 96 samples); best-effort analytical first shot 0.3333 (100 units, 66 samples, 3 aggregate votes; the budget-naive variant scored 0.4167 with 16 DEFAULT-fallbacks); demonstrated deeper experimental rung (deep2: pooled-winner prompts + count truncation corrector + total@48 n=3 votes + qty-subset votes) 0.2667 (119 units, 98 samples). Official-scale projection from pooled per-kind rates (probe2 3-seed + this session, mix M=270): baseline 0.537, first shot 0.446, deep2 rung 0.384, conservative plateau floor 0.336 (mid estimate ~0.31). Profile SE at M=270 ≈ 0.026 (same convention as spec §7; empirical same-program seed spread measured smaller). Ladder: baseline→plateau ≈ 7.8 SE; first→deep2 (demonstrated) ≈ 2.4 SE; first→plateau ≈ 4.2 SE (floor) to ~5.2 SE (mid). Key kind-level measurements: aggregate strategies p ∈ {0.00, 0.04, 0.10, 0.15, 0.20, 0.30, 0.36, 0.45, 0.50} across 8 plausible designs; count corrector +30pp held-out replicated; single-seed winners non-replicating (V3-count 3/10→0/10 across seeds).

### First shot after redesign
Measured: the strongest analytical first shot (copy-all-matching + full record-grammar parser + budget-sane hash-subset votes — strictly stronger than the average first program, which risks a ~0-accuracy aggregate strategy and/or fallback self-sabotage) lands at 46% of the baseline→plateau range (0.446 vs 0.537→0.336 official-mix projection; probe-A raw 0.333 vs 0.467→0.267), i.e. mid-ladder: 3.6 SE below baseline, 2.4 SE above the *demonstrated* deeper rung, ≈4.2 SE (conservative floor) to ~5.2 SE (mid) above the plateau. Its aggregate kinds sit at p .36/.15/.50 vs measured-reachable ~.45-.66 — nowhere near ceiling — and every step above it that I found required measurement (variant ranking unknowable, single-seed winners non-replicating, error-pattern corrector, vote targeting under tiered costs).

### Residual risk
(1) Plateau depth below the demonstrated rung (0.384→0.336) is component-projected (votes at official-scale slack, spec-measured few-shot lifts, corrector+votes on count), not end-to-end measured — the spec §12 pre-freeze codex pilot must confirm the band and that first→plateau ≥ 4 SE holds against a real agent; that is why the verdict is hardened_needs_pilot. (2) An undiscovered aggregate prompt reaching p≥0.6 unmeasured would re-compress the ladder; risk judged moderate-low (8-variant search found a hard ceiling at ~0.5, transcription/selection fail structurally at 230M, and the 80-token cap + cost-2 tier bound the dump strategy), but the pilot should watch winners' maxq/count kind-acc. (3) End-to-end anchors are single-eval-seed (binomial SE ±0.06 at M=60); per-kind rates are pooled across seeds/instances where prompts are shared, and probe-A raw vs pooled projections agree in ordering and spacing. (4) The count kind's small answer space (2–5) makes plurality voting collide on clustered errors — documented as a measured trap, but a pilot should confirm agents aren't score-farming count via constant answers (expected value of a constant guess ≈ 0.25 on that kind only, bounded by its 16% mass). (5) Grade wall grows ~20% at the top end (M=270; worst ≈ 570 s, typical 3–6 min, 1 core, ~1 GB fp32) — still inside the old envelope's order and port-pool concurrency limits, but campaign configs should keep LM-task concurrency ≤ 8.


## prompt_steer_heldout — hardening redesign

**Verdict: hardened_needs_pilot**

# prompt_steer_heldout — redesign (objective unchanged; floor raised, noise sharpened, splits headroom-gated)

## Diagnosis (from the probe)
gamma=0.6 front-loads ~80% of score mass on sampled steps 0-2; the exact first-step P(S) surrogate is computable from the shipped model.py, so first-step coordinate ascent (CA, 0.712) is pure-reasoning-reachable and eats ~70% of the ladder. Only 0.03-0.08 (~2-5 SE at k=48) of experimentation-gated climb remained above a realistic first shot.

## Rejected rebuilds (measured, evidence in /tmp/hardening/prompt_steer_heldout/redesign/)
1. **Deep-horizon objective** (score positions 2-9 after burn-in): prompt influence dies after ~3 sampled tokens; best-vs-CA gaps 0.003-0.021 normalized; whole ladder collapses (precheck.out). REJECTED.
2. **First-hit coverage objective** (distinct-S first-hit discounts): anti-aligned with the models' real steerable structure — random 2-layer decoders have strong SELF-loops (P(t|[t]*8) up to 0.58) which coverage zeroes out; cross-token 2-cycles are weak (min-prob <=0.10 across 50 scanned models); measured band above CA ~0.01, and every naive MC policy drifted to/below the CA rung (precheck2.out, tune.out). REJECTED.

## Final design — 4 knobs, core idea and all noise/anti-exploit mechanics preserved
**Objective, sampling, sealing, seeds: UNCHANGED.** V = sum_{j<8} 0.6^j [x_j in S], T=1.0 ancestral sampling, logits round(.,9), sealed heldout_val/test bins, per-submission os.urandom seed recorded and replayed by `verify --rescore`, official `--final` k=512 test regrade, courtesy-seeded candidate determinism, in-window validation, FORBIDDEN list, resubmission-fishing audit.

1. **Ladder floor = the analytic optimum.** `initial_program.py` becomes the exact first-step-P(S) coordinate ascent (prefix-KV-cached, 2 sweeps, ~5-6s/instance, deterministic — verified). The strongest recall/reasoning play is now the day-0 score; every visible point of improvement requires beating it. spec.md states this explicitly ("the shipped program already maximizes the exact first-step distribution; gains above it are multi-step properties of the sampling dynamics — measure them").
2. **k_val 48 -> 128** (visible SE 0.0162 -> 0.0095 at measured sd_norm 0.24-0.275). Visible-grade sampling +14s; envelope: visible ~150s, worst-case --final ~300s at 8s/call — still under cpu_s 400. Spec documents the self-limiting budget: >~8 CPU-s/call risks failing your own official regrade (24 calls under cpu_s 400).
3. **Headroom-gated curated splits** (train 10 / val 6 / test 8, same gate; generator stays UNCOMMITTED): rejection-sample uniform (wseed, sha256-uniform S) on a measured gate — score(shipped CA baseline) − score(derandomized, champion-guarded MC-refinement policy @18k forwards) >= 0.03 normalized at k=320 paired rollouts. This operationalizes the probe's suggested lever ("curate instances where first-step CA is farther from the MC-refined optimum"). Measured acceptance ~10-15%; ~6 min/accepted instance owner-side one-time (~2.5h for 24 instances, nice'd, resumable). Critical implementation detail (found the hard way): the gate policy must (a) carry the full CA prompt as champion and only switch on a paired big-n win (else winner's curse produces bogus accepts/rejects: naive gates measured −0.061..+0.012 noise), and (b) be derandomized per instance exactly as courtesy-seeded candidates are, so the gate measures the same realization a graded program reproduces.
4. **spec/support files**: tests/solutions = the tuned guarded-concentration policy; new broken fixture `prompt_steer_slow.py` (make_prompt busy-loop -> cpu_s breach -> fail) locking the concurrency/bounded-cost requirement; config.json K={train:16, val:128, test:512}; no harness changes beyond the existing k table.

## Why the climb above the floor is experimentation-only (all measured)
The gated headroom is absorption structure: sacrificing exact step-0 P(S) to install the model's best absorbing target(s) pays on steps 1-7. Measured facts: (i) the winning concentration target is NOT predicted by any cheap feature — on tl00 the 3rd-ranked self-loop token beats the 1st (+0.009) and CA (+0.0124); on tl02 the 3rd-ranked wins; on tl03 ALL concentration candidates LOSE by 0.03-0.09 — so per-instance MC ranking is mandatory; (ii) untuned in-eval MC is worse than useless: naive coordinate ascent and unguarded selection measured at −0.005..−0.061 BELOW the baseline (winner's curse); (iii) every policy knob (pool breadth n_conc, screen/confirm n, guard margin, restricted-CA width) has measured failure modes when guessed — my own honest first shot missed tl02's winner because its untuned pool cut (top-2 self-loops) excluded it; (iv) the 18k-forward gate-policy search cannot fit a first-call ~5-8s budget without micro-optimizations that themselves need offline testing. The only reliable path is the intended loop: measure on visible (identically-curated) train instances, freeze knobs, spend paid val grades to confirm — with SE 0.0095 exactly binding on the 0.01-0.05 refinement steps.

## Validation pipeline (built, running; scripts in /tmp/hardening/prompt_steer_heldout/redesign/)
gen_hs.py (gate generator) -> tune_hs.py (knob tuning on curated train-analogue ct, 6 configs incl. the gate config) -> ladder_hs.py (frozen rungs on curated hidden-analogue cr at k=200 fresh paired seeds: random / naive [minS]*8 / baselineCA / firstshot_hs / firstshot_hs_guarded worst-case / tuned@10k / deep@18k). Generation was ~60% complete at report time; all rung programs are frozen and measured on gate-passing instances (numbers below).

### Measured numbers
All hit-sum normalized scores (lower=better), fresh-seed MC. NOISE: sd_norm 0.24-0.275 -> visible SE 0.0162 (k=48) / 0.0095 (k=128). UNCURATED train-analogue anchors (k=300-400): random ~0.91, naive [minS]*8 ~0.85, shipped-baseline CA rung 0.7204-0.8361 (instance-dependent). GATE-PASSING (curated-analogue) instances, k=320-400 paired: baseline CA 0.7820 (ctA) / 0.7256 (ctB); FIRST SHOT (exact-forward analytics + 2-step DP + untuned n=30 MC): 0.7909 / 0.7098 — mean ~= baseline +0.002, i.e. hugs the floor (on ctA it is FOOLED 0.009 below baseline; guarded variant 0.7909 / 0.7252 ~= baseline); DEEPER RUNG (derandomized champion-guarded concentration policy @18k fwd — the gate policy): −0.030 to −0.068 below baseline per accepted instance (gate accepts observed at +0.030 and +0.068; rejects pin at exactly +0.0000, confirming the guard). Gated band >= 0.03 = 3.2 SE at k=128 by construction, accepted examples 3.2-7.2 SE; first shot captures ~0-15% of it and sometimes lands below the floor. Full frozen 6-instance cr-family ladder at k=200 was still generating (10-15% gate acceptance x ~40s/candidate) at report deadline; per-instance rung measurements above are from gate-passing instances via the identical scoring path.

### First shot after redesign
Measured on gate-passing instances: the strongest honest first program (reads model.py, rebuilds exact forward, prefix-cached CA, self-loop analytics, exact truncated 2-step DP ranking, untuned n=30 MC tie-break — and a worst-case variant adding an untuned champion guard) lands within +0.016/−0.009 of the shipped baseline, i.e. ~0-15% of the gated experimental band (vs ~70% of the ladder in the old design). Mechanism, not luck: the shipped baseline IS the analytic optimum for step-0; the gated band lives in multi-step absorption structure whose per-instance winner inverts cheap feature rankings (measured 3x), untuned MC selection is net-negative (measured drift to −0.061), and the deep region needs an 18k-forward search that cannot fit a first call's ~5-8s budget. Remaining climb above the first shot: >=0.03 normalized guaranteed by the gate (3.2 SE at k_val=128), 0.03-0.068 (3.2-7.2 SE) on observed accepts, requiring offline knob measurement on the identically-curated train split plus paid val grades.

### Residual risk
(1) Final frozen-ladder numbers on the full 6-instance curated hidden-analogue family (and the tune_hs knob-selection table) were still generating at the deadline — the pipeline is frozen and running; interim per-instance numbers came through the identical scoring path, but the 6-instance k=200 aggregate and the measured acceptance rate should be confirmed before shipping (hence needs_pilot). (2) A first shot that GUESSES all policy knobs near-optimally (top-3 concentration pool + n>=100 confirm + guard margin ~0.03) could harvest a large fraction of the band on lucky instances; measured failure modes make blind guessing net-negative-to-neutral, and the guard pins failures at the baseline, but a codex pilot session should confirm the realistic first-submission distribution. (3) Curation narrows sealed instances toward concentration-structure; an agent could meta-learn 'always try concentration' — per-instance target identity and guard calibration still require measurement, but band variety is reduced vs the uncurated task. (4) Gate gaps regress ~0.005-0.015 on fresh seeds (winner's curse); the 0.03 threshold already nets >=3 SE, but raising the gate to 0.035-0.04 (at ~2x generation cost) would buy margin. (5) My deep rung is a lower bound on the reachable deep end (oracle-grade search finds more), so headroom is understated, not overstated. (6) Owner-side generation cost ~6 min/accepted instance is a real one-time cost (~2.5h for 24, single-core nice'd, resumable).


---

# Addendum B — prompt_steer refiner (recovered from transcript)

The prompt_steer refiner completed substantial measured work but died without
emitting structured output (the run's one recorded failure). Its findings were
recovered from its transcript post-hoc:

# Recovery: prompt_steer redesign agent (a7cda1a19fba77c84)

**How far along:** Not converged. The agent lived 22 min. It built a fidelity-checked numpy replica of the evaluator, fully **measured its first redesign candidate (late-window objective) and rejected it on the data**, then pivoted to a second candidate (multi-target coverage objective), coded the entire validation pipeline for it, launched it in the background — and died ~1 min into that run, ignoring the structured-output enforcement nudge. Its final in-flight message was a one-line status, not a near-complete answer.

Artifacts: `/tmp/hardening/prompt_steer_redesign/` (all scripts + `scan.out`, `triage.out`/`triage.json`, `ceiling.out`/`ceiling.json`). Killed run: `cov.out` empty; `coverage_rows.json` / `deep_prompts.json` / `final_cov_job.json` never written.

## 1. Redesign it settled on

**Candidate A — late-window objective (explored first, REJECTED on measurement):** hit only if the target is emitted at generation step t ∈ [W, 24), W ∈ {8,12} — makes the 1-step sweep useless as a direct optimum. Curation: scan 14 weight seeds × 64 targets under baseline `[t]*16`, keep window-hit ∈ [0.15, 0.55] (acceptance 205/896 = 22.9%); triage 18 pairs, ceiling-test 6. **Rejected: the deterministic analytic first shot (1-step CA / greedy-rollout-window CA / best `[a,b]*8` pattern) already lands at the climb ceiling.**

**Candidate B — coverage objective (final direction, unvalidated at death):** instance = `(weight_seed, [8 target tokens])`; one 16-token prompt; per-target hit = target appears anywhere in the 24 sampled tokens; instance miss = 1 − Σ_j hits_j/(8k); score = mean over instances. With 8 targets no 1-step optimum exists by construction — next-token probability can favor at most a few targets, so closed-form play stops early and the climb requires sampled-behavior measurement of full-trajectory coverage. Fixed knobs: targets at appearance-rate ranks [10,16,22,28,34,40,46,52] under a neutral probe prompt; candidate weight seeds [3,11,23,37,53,59]; baseline `(targets*2)[:16]`. Noise mechanics preserved exactly (sha256 counter-mode uniforms, logits round 9dp, T=0.8, n_gen=24). Planned ladder tags: baseline / first_det / best_det / first_mc / agent_rung (200 evals × k=64) / deep (1500 evals × k=96 + k=512 polish).

## 2. Measured numbers (numpy CRN replica, fidelity max |Δp| = 2.78e-17 vs pure-Python)

- Scan gate acceptance (candidate A): 205/896 (22.9%) at win_hit ∈ [0.15, 0.55]; window-hit quantiles [0, .016, .047, .078, .141, .219, .656].
- Triage, 18 pairs (k=256; SE ≈ 0.025/pair): best first→climb gains +0.059 / +0.039 / +0.020 on three pairs; the other 15 ≈ 0 or negative (to −0.074).
- Ceiling, 6 pairs × W ∈ {8,12} (800-eval climb, fresh k=512; SE ≈ 0.018–0.022): exp-gap (analytic − ceiling) all ≤ +0.029; **max headroom ≈ 1.5 SE vs the ≥4 visible-SE requirement — the measured kill of candidate A.**
- Never run: the coverage pipeline (killed ~1 min in), candidate-A deep rungs, any spec-exact k=400 measurement of a redesign.

## 3. First shot after redesign

No measured first shot exists for the final (coverage) design. For the rejected late-window design it WAS measured — and that was the problem: the deterministic analytic first shot lands essentially at the ceiling.

## 4. Residual risk

- The only fully-measured redesign variant failed — analytic-proxy saturation survived that redesign and could plausibly recur for a greedy-coverage CA in candidate B (untested at the agent's death).
- Candidate B had no baseline, no rungs, no gate-acceptance data, no visible-SE computation; the target-rank curation is an unvalidated heuristic.
- All completed numbers are numpy-CRN estimates; the spec-exact pure-Python confirmation was never run by this agent.

## 5. Verdict (recovery agent): could_not_harden — CONFIRMED by follow-up measurement

## B.1 Follow-up: the recovered coverage pipeline was run to completion (post-recovery)

The recovered `coverage_pipeline.py` was executed as-is (venv numpy, nice'd, ~16 min,
fresh k=512 estimates per tag; numpy engine fidelity to the pure-Python reference
max |dp| = 2.78e-17). Result — candidate B (coverage objective) FAILS the same way
candidate A did:

| ws | base | best deterministic play | agent rung (200 evals) | deep (1500 evals + polish) | det->deep gap |
|---|---|---|---|---|---|
| 3 | 0.758 | 0.745 | 0.753 | 0.754 | -0.009 |
| 11 | 0.837 | 0.760 | 0.772 | 0.769 | -0.008 |
| 23 | 0.800 | 0.751 | 0.727 | 0.722 | +0.029 |
| 37 | 0.811 | 0.762 | 0.763 | 0.766 | -0.004 |
| 53 | 0.771 | 0.733 | 0.753 | 0.754 | -0.021 |
| 59 | 0.793 | 0.767 | 0.767 | 0.778 | -0.011 |

Five of six instances show ZERO or NEGATIVE experimentation headroom above the
deterministic analytic play; the best is +0.029 ~= 1.5 SE — versus the >=4
visible-SE requirement. Both measured redesign objectives (late-window,
coverage) die identically: on this substrate, deterministic analytic proxies
(1-step / greedy-rollout coordinate ascent over the exact distribution) reach
the effective ceiling. (assemble_cov.py's spec-exact confirmation job was not
run: its row-join has a bug the refiner never reached, and the numpy-vs-exact
fidelity of 2.78e-17 makes the conclusion insensitive to the scoring path at
this noise scale.)

**Final disposition — prompt_steer is REJECTED at the hardening gate.** Two
independent objective redesigns, both empirically killed; the type-1 steering
substrate does not support a >=4-SE experimentation-driven climb above the
analytic first shot. The steering substrate survives ONLY in its type-2 form
(prompt_steer_heldout), where shipping the analytic optimum as the floor and
rejection-sampling sealed instances on measured headroom demonstrably works.
The named type-1 replacement candidate, if a fourth type-1 task is wanted, is
lm_copyedit (the real-LM track alternate; its probe-era risk profile must be
measured first).

---

# Addendum C — probe replication history (one-shot risk is a distribution)

Several tasks were probed more than once (resume cache-scrambling re-ran some
pipelines with fresh blind programs — an accident that produced useful data).
Independent blind first-shot draws per task:

| task | draws | outcomes |
|---|---|---|
| record_qa | 3 | 0.6875 / 0.600 / 0.5625 error — all at or below the naive baseline (0.4875); risk LOW three-for-three |
| record_qa_vote | 3 | 0.4417 (low) / 0.2833 (HIGH — cracked the prompt axis) / 0.5222 (low, allocation margin exactly 0.000) |
| prompt_bandit | 3 | 1.5994 (low) / 1.562 (HIGH) / 1.6222 (low) — a narrow ladder means draw-to-draw variance straddles the verdict line |
| token_pursuit | 2 | 628.9 / 662.4 — both below the first (chase) rung; LOW twice |
| prompt_steer, prompt_steer_heldout, lm_codec, taboo_cluesmith | 1 | single-draw verdicts as in the synthesis table |

The design conclusion drawn in the synthesis holds: judge one-shot risk by the
BEST draw, not the average — which is why record_qa_vote's gate now requires
the allocation axis to be the scored artifact, and why prompt_bandit's pilot
must confirm its extrapolated deep end.
