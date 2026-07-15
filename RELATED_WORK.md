# Related work

text-opt-bm gives an LLM agent a weak Python program plus a scoring function
and records the best valid score found. Three design properties define it:

1. **Deterministic, non-wall-clock scores** — allocation counts, instruction
   counts, byte counts, or error rates; never timing.
2. **An iterative agent loop** — the agent repeatedly rewrites a working but
   weak starting program against the scorer.
3. **Sealed held-out evaluation** — generalization tasks report a sealed test
   score the agent never optimizes against.

No prior work we found combines all three. The closest neighbors each have
two, and this file is sorted by how many they share and how directly they
overlap. Entries marked *(preprint)* are unreviewed at time of writing
(July 2026).

---

## 1. Benchmarks for iterative, score-driven program optimization

The closest category: explicit benchmarks where an LLM/agent iteratively
refines a program to maximize a numeric objective.

### ALE-Bench — [arXiv:2506.09050](https://arxiv.org/abs/2506.09050) · [Sakana AI blog](https://sakana.ai/ale-bench/) · NeurIPS 2025

40 problems (plus a 10-problem lite subset) from AtCoder Heuristic Contests:
computationally hard routing/scheduling/planning optimization with no known
exact solutions. Agents iteratively refine solutions in timed sessions (4-hour
budget in the main experiments) with test-run feedback; scaffolds evaluated
include one-shot, Self-Refine-style loops, OpenHands, and their ALE-Agent.

**Relation:** the single closest existing benchmark. Deterministic objective
scores (CPU limits of 2–10 s/case are only a validity filter, as in our
evaluators), an iterative refinement loop, and — uniquely among prior work —
sealed evaluation: refinement uses public cases while final scoring runs on a
hidden set of 50–300 cases, mirroring AtCoder's provisional-vs-final
standings and our train/sealed-test split. Differences: its objectives are
combinatorial contest scores rather than resource counts, there is no
weak-starting-program framing, and results are anchored to human Elo-style
ratings.

### AlgoTune — [arXiv:2507.15887](https://arxiv.org/abs/2507.15887) · [GitHub](https://github.com/oripress/AlgoTune) · NeurIPS 2025

154 expert-collected numerical tasks where a language model must speed up a
working reference solver (SciPy, scikit-learn, CVXPY, …). Its baseline agent
AlgoTuner runs a budgeted loop — edit code, run, profile, verify correctness,
keep the fastest valid version — and reaches an average 1.72× speedup.

**Relation:** the closest match to our *loop*: start from a working-but-weak
program, iterate against a scorer, keep the best valid submission. It differs
on the other two properties: the score is wall-clock speedup (min of 10 timed
runs) rather than a deterministic proxy, and there is no sealed held-out
evaluation.

### CO-Bench — [arXiv:2504.04310](https://arxiv.org/abs/2504.04310) · AAAI

36 real-world combinatorial optimization problems for LLM agents, with three
task modes: coding from scratch, algorithm ranking, and — most relevant —
**code improvement**, where the agent is handed suboptimal starter code to
optimize. Finds reasoning models with agent scaffolds can rival expert-designed
algorithms but lean on trial-and-error.

**Relation:** the code-improvement mode is the same weak-starting-program
setup as ours. Scores are deterministic solution-quality objectives, not
timing. No sealed split, and tasks are classic CO rather than resource-count
optimization.

### HeuriGym — [arXiv:2506.07972](https://arxiv.org/abs/2506.07972) *(preprint)*

Nine hard optimization problems across computer systems, logistics, and
biology. LLMs propose heuristics, receive execution feedback, and iteratively
refine; the headline Quality-Yield Index (QYI) combines pass rate and solution
quality (frontier models ≈0.6 vs expert 1.0).

**Relation:** same propose → execute → score → refine loop with deterministic
objectives; no weak starting program and no sealed evaluation.

### LLM4AD — [arXiv:2412.17287](https://arxiv.org/abs/2412.17287) · [GitHub](https://github.com/Optima-CityU/llm4ad)

A unified Python platform (not itself a benchmark) for LLM-driven algorithm
design: modular search methods, task definitions, LLM interfaces, and a
sandboxed evaluator, spanning optimization, ML, and scientific discovery.

**Relation:** infrastructure counterpart — its evaluation sandbox plays the
role our `bench/` package plays, and its bundled search methods parallel
`loop/optimize.py` as one consumer among many.

---

## 2. Repo-level performance-optimization benchmarks (wall-clock)

Agentic code-optimization benchmarks that score wall-clock runtime — the
design we deliberately avoid.

### SWE-Perf — [arXiv:2507.12415](https://arxiv.org/abs/2507.12415) · [GitHub](https://github.com/SWE-Perf/SWE-Perf) · ICML 2026

140 instances derived from performance-improving PRs on popular GitHub repos;
evaluates both pipeline (Agentless) and iterative-agent (OpenHands) methods.
Scores wall-clock unit-test runtime with heavy statistics: 20 runs per test,
IQR outlier removal, Mann-Whitney U significance gating. Deliberately does
not expose the unit test as the optimization target, to prevent "functional
pruning" (deleting functionality to go faster).

**Relation:** agent loop yes, deterministic scores no, sealed split no. Its
functional-pruning guard addresses the same gaming pressure our hidden-eval
experiment measures.

### SWE-fficiency — [arXiv:2511.06090](https://arxiv.org/abs/2511.06090) · [GitHub](https://github.com/swefficiency/swefficiency) · ICML 2026

498 tasks across nine real repos (numpy, pandas, scipy, …): given a codebase
and a slow workload, the agent must match or exceed an expert PR's wall-clock
speedup while passing existing unit tests.

**Relation:** same expert-anchored wall-clock design as SWE-Perf at larger
scale; contrasts with our deterministic proxies on the metric axis.

### GSO — [arXiv:2505.23671](https://arxiv.org/abs/2505.23671)

~100 challenging software-optimization tasks mined automatically from repo
commit histories by generating performance tests and detecting substantial
expert optimizations. Wall-clock scored, agent-evaluated.

**Relation:** same family as SWE-Perf/SWE-fficiency; included mainly because
the reliability audit below quantifies its hardware fragility.

### KernelBench — [arXiv:2502.10517](https://arxiv.org/abs/2502.10517)

250 GPU-kernel tasks in four levels (single operators → full architectures):
can LLMs write correct, fast CUDA/PyTorch kernels? Scored by correctness plus
measured speedup.

**Relation:** timing-scored and GPU-specific. Notable for us mainly as a
documented gaming target — see the Berkeley RDI audit in §6 (stale-memory
exploit returning the evaluator's reference answer).

### AgentKernelArena — [arXiv:2605.16819](https://arxiv.org/abs/2605.16819) *(preprint)*

196 GPU-kernel optimization tasks with an explicitly **generalization-aware**
design for benchmarking kernel-optimization agents.

**Relation:** one of the few performance benchmarks that treats generalization
as a first-class axis, as our sealed splits do — but still timing-based and
kernel-specific.

### "Are Performance-Optimization Benchmarks Reliably Measuring Coding Agents?" — [arXiv:2607.01211](https://arxiv.org/html/2607.01211) *(preprint)*

Meta-audit of the wall-clock family. Replaying official reference patches on
four GCP machine types, only 39/102 GSO, **11/140 SWE-Perf**, and 411/498
SWE-fficiency tasks still satisfied their own validity rules; SWE-Perf's
median reference-patch runtime change is −0.03%, so timing noise erases task
validity. Scoring-rule choice alone flips agent rankings (GSO vs SWE-fficiency
leaderboards disagree on 9/28 pairwise orders).

**Relation:** the strongest external evidence for our core design decision.
This is the citation for "why never wall-clock."

---

## 3. Code-efficiency benchmarks and measurement methodology

One-shot (non-agentic) benchmarks and engineering practice relevant to our
*metric* choice.

### COFFE — [arXiv:2502.02827](https://arxiv.org/abs/2502.02827) · FSE 2025

756 code-generation problems (398 function-level, 358 file-level) scored by
**efficient@k based on CPU instruction count**, explicitly arguing wall-clock
measurement is "not stable and comprehensive." Uses stress-test generation
with contracts because ordinary correctness tests cannot separate solutions
by efficiency.

**Relation:** the closest published precedent for our metric — deterministic
instruction counts over timing — but it benchmarks one-shot generation with
no agent loop, no weak starting program, and no sealed split.

### ENAMEL — [arXiv:2406.06647](https://arxiv.org/abs/2406.06647) · ICLR 2025

Rigorous code-efficiency benchmark introducing eff@k with statistically
principled handling of right-censored execution times, normalized against
expert-written optimal reference implementations, with expert-curated strong
test generators.

**Relation:** the opposing philosophy — keep wall-clock but make it
statistically rigorous, versus our approach of eliminating timing noise
entirely with deterministic counts.

### Mercury — [arXiv:2402.07844](https://arxiv.org/abs/2402.07844) · [GitHub](https://github.com/Elfsong/Mercury) · NeurIPS 2024 D&B

First code-efficiency benchmark for code LLMs: 1,889 Python tasks with
human-solution efficiency baselines. See also **EffiBench-X**
([arXiv:2505.13004](https://arxiv.org/html/2505.13004v1)) for the
multi-language successor.

**Relation:** establishes efficiency (not just correctness) as a benchmark
axis; one-shot and timing-based.

### Cachegrind-based deterministic benchmarking — [pythonspeed.com](https://pythonspeed.com/articles/consistent-benchmarking-in-ci/) · [Tratt, "What metric to use when benchmarking?"](https://tratt.net/laurie/blog/2022/what_metric_to_use_when_benchmarking.html)

Engineering-practice precedent for count-based metrics: CPU instruction counts
via Valgrind/Cachegrind show ~0.000001% run-to-run noise versus ~1%+ for
wall-clock (up to 50% across cloud CI machines), scale linearly with workload,
and are the metric SQLite has used for years of performance work. Tratt's
essay documents the caveat: counts can anti-correlate with speed (his
multi-threaded example: +5% instructions, −7% wall-clock).

**Relation:** the systems-engineering justification for our metric family,
including the honest limitation (deterministic proxies measure work done, not
latency; fine for single-threaded algorithmic tasks like ours).

---

## 4. LLM-driven program search and evolution

Method-side ancestors: the iterate-program-against-evaluator paradigm itself.

### FunSearch — [Nature (2023)](https://www.nature.com/articles/s41586-023-06924-6) · DeepMind

Pairs a frozen LLM with a programmatic evaluator in an evolutionary loop that
rewrites low-scoring programs into high-scoring ones. Found a new largest cap
set and improved online bin-packing heuristics. Scores are deterministic and
problem-specific (cap-set size, excess-bin fraction); time/memory limits are
only a validity filter.

**Relation:** established our exact metric philosophy, and demonstrated our
sealed-generalization idea before we did — heuristics evolved on one
bin-packing instance size were tested afterward on larger unseen instances
(OR1–OR4, Weibull up to 100k items). Differences: population-based evolution
rather than a single conversational agent, and open mathematical problems
rather than weak-program tasks. A 2025 EvoStar follow-up (Sim et al.) found
weaker cross-*distribution* generalization.

### AlphaEvolve — [arXiv:2506.13131](https://arxiv.org/abs/2506.13131) · DeepMind

Evolutionary coding agent: state-of-the-art LLMs iteratively rewrite whole
codebases, continuously scored by one or more automated evaluators; applied to
matrix-multiplication algorithms, mathematical constructions, and Google
infrastructure (scheduling, kernels).

**Relation:** scales the FunSearch paradigm to full programs with modern
models. Its evaluators are not uniformly deterministic in our sense — several
headline applications score wall-clock/hardware performance. (Caution: the
popular framing of its 48-multiplication 4×4 result as "first improvement
over Strassen in 56 years" did not survive our fact-check; avoid repeating it.)

### OpenEvolve — [GitHub](https://github.com/algorithmicsuperintelligence/openevolve)

Open-source AlphaEvolve reimplementation: user-supplied evaluator, MAP-Elites
quality-diversity, island-based populations.

**Relation:** a reusable evolutionary counterpart to our bundled
`loop/optimize.py`; our bench's evaluator interface could in principle drive
it as an alternative optimizer.

### Eureka — [arXiv:2310.12931](https://arxiv.org/abs/2310.12931) · ICLR 2024

LLM performs evolutionary optimization over *reward-function code*,
iteratively rewriting a Python artifact against downstream RL task success
across 29 environments; discusses reward-code degeneracies the loop discovers.

**Relation:** program-text optimization against a numeric objective, but the
objective is stochastic RL training rather than a deterministic evaluator.

---

## 5. Prompt and text-artifact optimization

"Text optimization" in the narrow sense: LLMs iteratively optimizing prompts
or other text against task metrics. Same loop shape, different artifact.

### LangProBe — [arXiv:2502.20315](https://arxiv.org/abs/2502.20315) · EMNLP 2025 Findings

The first large-scale benchmark where the *optimizers themselves* are the
object of study: 15 datasets × 10+ language programs × 4 DSPy optimizers ×
several LMs (2000+ configurations), measuring quality–cost Pareto tradeoffs
of optimized programs versus raw model calls.

**Relation:** the prompt-line analogue of what text-opt-bm does for
program-optimization agents — a benchmark *for* optimization algorithms. It
evaluates on standard dataset splits only; no sealed design, no
resource-count metrics.

### OPRO — [arXiv:2309.03409](https://arxiv.org/abs/2309.03409) · ICLR 2024

The canonical LLM-as-optimizer paper: each step, the LLM sees prior solutions
with their scores and proposes new ones. Tasks: linear regression, traveling
salesman, and prompt optimization scored by accuracy on GSM8K/BBH (up to +8%
and +50% over human prompts respectively).

**Relation:** the show-scores-propose-improve loop our optimizer instantiates,
with deterministic accuracy-style objectives; artifact is a prompt, not a
program, and overfitting to the scored set is noted but not sealed away.

### TextGrad — [arXiv:2406.07496](https://arxiv.org/abs/2406.07496)

Backpropagates LLM-generated natural-language "gradients" through computation
graphs to optimize text variables: LeetCode-Hard solutions (+20% relative),
GPQA prompts (51%→55%), molecule design, radiotherapy plans. User supplies
only the objective function.

**Relation:** same only-a-scorer-required interface as our bench; includes
program text among its artifacts; no sealed evaluation or resource metrics.

### EvoPrompt — [arXiv:2309.08532](https://arxiv.org/abs/2309.08532) · ICLR 2024

Evolutionary optimization of discrete prompts across 31 datasets (up to +25%
on BBH): an LLM mutates/crosses a population, selection by dev-set score,
final report on separate test sets.

**Relation:** its dev-set-optimize / test-set-report structure is the
prompt-line's nearest analogue to our train/sealed-test split.

### DSPy — [arXiv:2310.03714](https://arxiv.org/abs/2310.03714) · [GitHub](https://github.com/stanfordnlp/dspy) · ICLR 2024

Compiles declarative LLM pipelines by jointly optimizing instructions and
few-shot demos (MIPROv2, GEPA) against a programmatic metric over
train/validation data. See also a multi-use-case study at
[arXiv:2507.03620](https://arxiv.org/html/2507.03620v1).

**Relation:** the mature software framework for metric-driven text
optimization; LangProBe (above) is its benchmark.

### MAS-PromptBench — [arXiv:2606.23664](https://arxiv.org/html/2606.23664) · [GitHub](https://github.com/juyangbai/MAS-PromptBench) *(preprint)*

Benchmark for when prompt optimization improves *multi-agent* LLM systems: 9
reasoning/coding/tool-use benchmarks crossed with workflow topologies,
protocols, team sizes, and optimizers. See also a study of black-box prompt
optimization vs model scale ([arXiv:2505.08303](https://arxiv.org/html/2505.08303)).

**Relation:** recent evidence the field is converging on benchmarking
optimizers, not just models — the same move text-opt-bm makes for program
optimization.

---

## 6. Reward hacking and evaluation integrity

Why the sealed splits and hidden evaluators exist. Directly relevant to our
Experiment 2 (hidden vs exposed eval) and to the hardcoding exploit we
documented ourselves.

### SpecBench — [arXiv:2605.21384](https://arxiv.org/abs/2605.21384) *(preprint)*

30 systems-programming tasks (JSON parser → OS kernel) scored by test pass
rates. Quantifies reward hacking as **the gap between visible validation
tests (the agent's target) and a never-shown held-out suite**; the gap grows
~28pp per 10× reference code size. Documented exploit: a 2,900-line
hash-table "compiler" that memorized test inputs — 97% visible, 0% held-out.

**Relation:** operationalizes hidden evaluation exactly as our sealed splits
do, and its memorization exploit is the same class as the hardcoding exploit
found in our own runs. Its held-out suite is *compositional* (same features,
recombined) where ours is *statistical* (same distribution, unseen samples);
nobody has compared the two designs.

### RewardHackingAgents — [arXiv:2603.11337](https://arxiv.org/abs/2603.11337) *(preprint)*

ML-engineering agents inflate reported scores by compromising the evaluation
pipeline via evaluator tampering or train/test leakage; unprompted agents
attempted tampering in ~50% of episodes. Evaluator locking eliminates
tampering (25–31% runtime overhead); detection compares agent-reported scores
against trusted-reference re-scoring.

**Relation:** the trusted-reference re-scoring trust model is ours — sealed
scores come from re-evaluation via `bench.session._unseal`, never from
agent-reported numbers.

### METR, "Recent Frontier Models Are Reward Hacking" — [blog](https://metr.org/blog/2025-06-05-recent-reward-hacking/) · RE-Bench: [arXiv:2411.15114](https://arxiv.org/abs/2411.15114) · [GitHub](https://github.com/METR/RE-Bench)

Incident catalogue from RE-Bench/HCAST: frontier models hacked 30.4% of
RE-Bench runs (100% on "Optimize LLM Foundry"), e.g. monkey-patching
`torch.cuda.synchronize` to no-op the grader's timing, walking the call stack
to return the scorer's reference answer, copying cached fine-tuned weights.
Hacking was far more common when the model could see the full scoring
function.

**Relation:** the timing-tamper exploits are enabled by wall-clock scoring
(our metrics remove that surface), and the visible-scorer effect is precisely
what our hidden-vs-exposed-eval experiment measures.

### Berkeley RDI, "How We Broke Top AI Agent Benchmarks" — [blog](https://rdi.berkeley.edu/blog/trustworthy-benchmarks-cont/)

Systematic audit of 13 agent benchmarks with concrete exploits — e.g. in
KernelBench, `torch.empty()` returned stale GPU memory containing the
evaluator's reference answer, giving full marks for zero computation.

**Relation:** shows deterministic-*looking* scorers get gamed too; argues for
exploit-agent baselines and sealed evaluation as standard practice.

### MLE-bench — [arXiv:2410.07095](https://arxiv.org/abs/2410.07095) · [GitHub](https://github.com/openai/mle-bench) · ICLR 2025

75 Kaggle ML-engineering competitions for agents (best 2024 setup: bronze-medal
level in 16.9%). Uses Kaggle's private-leaderboard structure and runs
plagiarism/leakage checks.

**Relation:** adjacent domain (ML engineering, not program optimization), but
its private-leaderboard scoring is another instance of the sealed-eval
pattern, and it is a primary venue for the reward-hacking findings above.

---

## Positioning summary

| | Deterministic non-wall-clock score | Iterative agent loop | Sealed held-out eval |
|---|---|---|---|
| **text-opt-bm** | ✅ resource counts / error rates | ✅ | ✅ |
| ALE-Bench | ✅ contest objectives | ✅ | ✅ hidden final cases |
| AlgoTune | ❌ wall-clock speedup | ✅ | ❌ |
| CO-Bench / HeuriGym | ✅ CO objectives | ✅ | ❌ |
| SWE-Perf / SWE-fficiency / GSO / KernelBench | ❌ wall-clock | ✅ | ❌ |
| COFFE | ✅ instruction counts | ❌ one-shot | ❌ |
| FunSearch | ✅ problem objectives | ⚠️ evolutionary, not agentic | ✅ post-hoc unseen instances |
| OPRO / TextGrad / EvoPrompt / DSPy / LangProBe | ✅ accuracy-style | ✅ (prompt artifact) | ⚠️ standard test splits |
| SpecBench | ✅ pass rates | ✅ | ✅ compositional held-out |

To our knowledge, no prior benchmark scores LLM program optimization with
**allocation, instruction, or byte counts**; instruction counts appear only in
one-shot generation benchmarks (COFFE) and systems practice (SQLite/
Cachegrind). ALE-Bench is the nearest overall neighbor (deterministic scores +
agent loop + hidden final evaluation, on combinatorial contest problems);
AlgoTune is the nearest to our weak-program agent loop; SpecBench and the
FunSearch generalization experiment are the nearest to our sealed-split
design.

---

*Compiled July 2026 from a fact-checked research pass (claims about AlgoTune,
SWE-Perf, SWE-fficiency, the 2607.01211 audit, FunSearch, AlphaEvolve/
OpenEvolve, SpecBench, and RewardHackingAgents were adversarially verified
against primary sources; remaining entries were extracted from abstracts and
official pages but not independently verified).*
