# Task Fit Ranking for `text-opt-bm`

Date: 2026-07-03

This ranks the task families found in the literature sweep by how well they fit the current benchmark design: local Python programs, deterministic numeric scores, short CPU-only evaluation, reproducible data, and no required external APIs.

Hidden validation is useful for studying overfitting, but it is not a strict requirement. A task can be an excellent fit as a **perfect-information task** when the reported score is exactly the deployment objective and benchmark gaming is self-limiting, for example resident memory, compressed bytes, or bytecode instructions on a fixed workload. A task can also be an excellent fit as a **generalization task** when train/validation/test splits are central to the behavior being studied.

## Fit Rubric

Scores are qualitative:

- **Determinism**: stable across repeated runs without LLM judges, web access, or wall-clock timing.
- **Runtime**: feasible as a per-iteration benchmark; ideally seconds to a few minutes.
- **Reproducibility**: data and scoring can be generated locally from seeds or shipped compactly.
- **Information model fit**: works cleanly as either perfect-information or train/validation/test generalization.
- **Cheat resistance**: hardcoding/memorization is self-penalizing, irrelevant because the fixed workload is the deployment target, or catchable with optional validation.
- **Implementation fit**: can be expressed as a single-file `program.py` with a small API.

## Tier 1: Best Fit

These are closest to the current benchmark and should be top candidates for expansion.

| Rank | Task family | Literature examples | Fit | Why it fits |
|---:|---|---|---|---|
| 1 | **Online bin packing / bin packing heuristics** | FunSearch, EoH/ReEvo, HeuriGym, CO-Bench | Excellent | Deterministic streams, compact inputs, simple validity, scalar score, very cheap runtime. Works as perfect-information fixed-stream optimization or as train/heldout stream generalization. |
| 2 | **Traveling Salesman / routing under deterministic budget** | OPRO, EoH/ReEvo, CO-Bench, HeuriGym, ALE-Bench, current `tsp_budget` | Excellent | Already present and proven. Can score fixed deployment instances directly, enforce bytecode budgets, and optionally add unseen-instance checks. |
| 3 | **Classic graph algorithms / dynamic connectivity / shortest paths / matching variants** | current `ops_connect`; CO/algorithm-design suites | Excellent | Fully deterministic, cheap, correctness checkable, strong algorithmic headroom. Good for bytecode-instruction metrics. Less directly represented in APO literature, which makes it distinctive. |
| 4 | **Lossless compression on synthetic structured corpora** | current `compress`, `compress_heldout`; adjacent code optimization literature | Excellent | Deterministic byte score, no timing, and supports both fixed-corpus perfect-information scoring and train/heldout generalization. External literature overlap is weaker, but benchmark quality is high. |
| 5 | **Memory-efficient data structures** | current `mem_kv`, `mem_index`; adjacent systems/code optimization | Excellent | Traced-memory scores are deterministic on a platform and correctness is exact. This is a natural perfect-information task when the measured workload is the deployment target. |
| 6 | **Small fixed-model inference optimization** | current `mem_infer`; code/systems optimization literature | Excellent | Deterministic outputs and memory metrics. Good systems flavor. Requires careful platform pinning but already works in the harness. |
| 7 | **Continuous black-box optimizer generation on BBOB-style functions** | BLADE, LLaMEA | Very good | Strong official benchmark ecosystem, reproducible functions, scalar scores, short local evals possible. Needs careful API design so generated optimizers are not too large/slow. |
| 8 | **Knapsack / set cover / orienteering / vehicle routing micro-instances** | ReEvo, EoH, CO-Bench, HeuriGym | Very good | Deterministic objective and feasibility checks. Works with fixed instance suites or heldout generated instances; use small instances and instruction budgets to avoid long runtimes. |
| 9 | **SAT / MaxSAT / graph coloring micro-heuristics** | HeuriGym, NPHardEval-style CO tasks | Very good | Exact deterministic scoring and validation. Need a carefully designed objective so there is optimization headroom beyond pass/fail correctness. |
| 10 | **Synthetic math word-problem solver** | GSM8K-family prompt optimization; current `easy_word_problems` | Very good | The synthetic version avoids public-data leakage and works without LLM calls. Keep train small relative to distribution diversity. |

## Tier 2: Good Fit With Careful Design

These are compatible with the benchmark, but need tighter constraints to avoid flakiness, long runtimes, or overfitting.

| Rank | Task family | Literature examples | Fit | Main caveat |
|---:|---|---|---|---|
| 11 | **AtCoder Heuristic Contest-style optimization** | ALE-Bench | Good | Excellent conceptual fit, but many tasks are long-horizon and may be too heavy. Use mini-AHC-style tasks rather than full contest workloads. |
| 12 | **Code efficiency on small programming tasks** | EvalPerf, PerfForge, EffiBench-X | Good | Very relevant, but wall-clock runtime is flaky. Prefer bytecode/instruction counts or fixed operation counters where possible. |
| 13 | **Repository-level performance optimization** | SWE-Perf | Medium-good | Strong real-world relevance, but dependency setup, runtime variance, and multi-file edits conflict with current single-file/local simplicity. |
| 14 | **ML engineering / Kaggle-style code optimization** | AIDE, MLE-bench, MLE-STAR | Medium-good | Score-based and iterative, but often expensive, stochastic, dependency-heavy, and dataset-heavy. Better as a separate benchmark track. |
| 15 | **Matrix multiplication / algebraic algorithm discovery** | AlphaEvolve, CodeEvolve | Medium-good | Prestigious and deterministic, but hard to make lightweight, accessible, and robust against specialized hardcoding. |
| 16 | **Geometry / packing / covering problems** | AlphaEvolve, optimize_anything | Medium-good | Deterministic objective possible, but floating-point tolerances and local minima require careful evaluation and validation. |
| 17 | **Molecule optimization via SMILES** | TextGrad, Feedback Descent | Medium | Text artifact fits, but reliable scoring often depends on external chemistry packages/docking models and may be slow. |
| 18 | **SVG / visual artifact optimization** | Feedback Descent, optimize_anything | Medium | Artifact is text, but evaluation usually needs VLM judges or perceptual metrics, weakening determinism. |
| 19 | **Harness/context-management code around an LLM** | Meta-Harness, FAPO, optimize_anything | Medium | Very relevant to agent systems, but it introduces external model calls, cost, nondeterminism, and API drift. Better as an optional expensive track. |
| 20 | **Agent prompt/pipeline optimization** | PROMST, MASPO, SPEAR, FAPO | Medium | Task structure is relevant, but scoring depends on LLM task models or judges. SPEAR's code-analysis loop is conceptually close, but not local/evaluator-only. |

## Tier 3: Weak Fit For This Benchmark, Strong Fit For Prompt-Optimizer Papers

These dominate APO papers, but they are not a natural fit for the current repo because the optimized artifact is usually a prompt for an external LLM.

| Rank | Task family | Literature examples | Fit | Why lower |
|---:|---|---|---|---|
| 21 | **GSM8K / public math QA prompt optimization** | APE, OPRO, PromptBreeder, TextGrad, PE2, PromptWizard | Weak-medium | Highly shared, but public leakage is severe and scoring requires LLM calls unless converted to a non-LLM solver task. Current synthetic `easy_word_problems` is the better form. |
| 22 | **BBH / reasoning prompt optimization** | APE, OPRO, PromptBreeder, EvoPrompt, GEPA, PrefPO | Weak-medium | Shared benchmark value is high, but model/API dependence and prompt sensitivity make it poor for deterministic local scoring. |
| 23 | **Instruction Induction / Natural Instructions** | APE, GrIPS, PromptBreeder, MOP, PLUM | Weak-medium | Good train/test protocol, but requires an LLM task model and often uses heterogeneous subsets. |
| 24 | **Sentiment/topic/question classification prompt optimization** | GrIPS, ProTeGi, EvoPrompt, DAPO, AMPO | Weak-medium | Cheap and reproducible datasets, but the benchmark measures prompt-task-model interaction rather than standalone program optimization. |
| 25 | **Multi-hop QA / RAG prompt optimization** | DSPy, MIPRO, GEPA, Meta-Harness | Weak-medium | Important for LM systems, but retrieval corpora, LLM calls, and judge/model variance complicate deterministic scoring. |
| 26 | **Medical QA / biomedical NLP prompt optimization** | PromptAgent, AMPO, PromptWizard, StraGo | Weak | High leakage and model dependence; high-stakes domain also raises evaluation burden. |
| 27 | **LLM instruction-following / IFEval-style tasks** | GEPA, PrefPO | Weak | Mostly evaluates a task model's obedience to prompts; deterministic local program scoring is not natural. |
| 28 | **LLM-as-judge industrial tasks** | SPEAR, FAPO | Weak | Useful in production, but judge drift, proprietary data, and external calls make it hard to reproduce as an open local benchmark. |

## Detailed Sorted Inventory

This expands the ranking to named tasks and datasets from the literature sweep. The ranking is by fit to this repo, not by popularity in the prompt-optimization literature.

| Fit | Named tasks / datasets | Recommended mode | Notes |
|---|---|---|---|
| Excellent | `mem_kv`, `mem_index`, compact key/value stores, inverted indexes | Perfect information | Current repo already validates this pattern. The deployment objective is memory on the measured workload. |
| Excellent | `compress`, fixed-corpus compression, structured logs/JSON/prose/CSV compression | Perfect information | Byte count is deterministic and self-contained. Fixed-corpus optimization is legitimate when the corpus is the deployment target. |
| Excellent | `compress_heldout`, train/validation/test compression | Generalization | Best when studying overfitting or dictionary generalization. |
| Excellent | `ops_connect`, dynamic connectivity, union/find, graph reachability streams | Perfect information | Strong deterministic instruction-count benchmark. |
| Excellent | `tsp_budget`, TSP, routing, VRP micro-instances | Perfect information or generalization | Fixed instance suites are valid; generated heldout instances are optional. |
| Excellent | Online bin packing, bin packing, packing heuristics | Perfect information or generalization | Best new task candidate. Used by FunSearch and heuristic-design papers. |
| Excellent | `mem_infer`, tiny transformer decoding, small fixed-model inference | Perfect information | Natural memory/systems task; keep model small and deterministic. |
| Very good | BBOB, MA-BBOB, SBOX-COST, small black-box optimizer generation | Perfect information or generalization | Strong benchmark lineage from BLADE/LLaMEA. Needs small budgets. |
| Very good | Knapsack, multidimensional knapsack, set cover, orienteering, prize-collecting routing | Perfect information or generalization | Deterministic objectives and exact references for small cases. |
| Very good | SAT, MaxSAT, graph coloring, clique/independent-set-style micro-instances | Perfect information or generalization | Good if scored by quality under budget rather than pass/fail only. |
| Very good | Synthetic GSM8K-style word problems, `easy_word_problems` | Generalization | Use synthetic generation to avoid public leakage and keep evaluation local. |
| Good | AtCoder Heuristic Contest-derived tasks: routing, scheduling, production planning, power-grid balancing | Perfect information or generalization | ALE-Bench is a strong reference; shrink tasks for per-iteration runtime. |
| Good | Code-efficiency microbenchmarks, EvalPerf-style tasks, PerfForge-style stress tests | Perfect information | Replace wall-clock runtime with bytecode/instruction/operation counts where possible. |
| Good | Matrix multiplication, algebraic algorithm discovery, sorting/circuit simplification kernels | Perfect information | Deterministic but harder to make lightweight and hardcoding-resistant. |
| Medium-good | Geometry packing, covering, cap-set-style constructive math tasks | Perfect information | Strong FunSearch/AlphaEvolve lineage; floating-point and validator design need care. |
| Medium-good | Repository-level performance tasks, SWE-Perf-style PR optimization | Perfect information | Realistic but conflicts with single-file simplicity and fast reproducibility. |
| Medium-good | Kaggle/MLE-bench/AIDE-style ML engineering tasks | Generalization | Score-based and iterative, but costly, stochastic, and dependency-heavy. |
| Medium | Molecule optimization, SMILES, DOCKSTRING-like tasks | Perfect information or generalization | Text artifact fits, but scoring usually needs chemistry dependencies or docking models. |
| Medium | SVG/image artifact optimization | Perfect information if using deterministic metrics; weak if judged by VLM | Avoid VLM judges if reliability is the goal. |
| Medium | Harness/context optimization, Meta-Harness, FAPO-style pipelines | Generalization | Very relevant to agents, but model calls and API drift make it a separate track. |
| Medium | Multi-agent prompt optimization, MASPO/PROMST/SPEAR-style tasks | Generalization | Conceptual fit, but scoring depends on external LLM behavior. |
| Weak-medium | GSM8K, MultiArith, AddSub, SVAMP, SingleEq, AQuA-RAT, MAWPS | Generalization | Popular shared tasks, but public leakage and LLM-call dependence weaken fit. Synthetic local variants fit much better. |
| Weak-medium | BBH tasks: object counting, word sorting, navigate, date understanding, logical deduction, hyperbaton, snarks, etc. | Generalization | Excellent for prompt papers; poor for local deterministic program scoring unless converted into synthetic solvers. |
| Weak-medium | Instruction Induction, BIG-Bench Instruction Induction, Natural Instructions, Super-NaturalInstructions | Generalization | Good prompt-optimizer benchmark, but requires a task LLM and inconsistent subsets across papers. |
| Weak-medium | Sentiment/topic/classification: SST-2/5, CR, MR, AGNews, TREC, Subj, DBPedia, MPQA, Yelp, IMDB, Amazon, Flipkart | Generalization | Cheap datasets, but mostly evaluate prompt-model interaction. |
| Weak-medium | GLUE/SuperGLUE/NLI: MNLI, SNLI, QNLI, RTE, MRPC, QQP, CoLA, CB, WSC, COPA, WiC, Winogrande, ANLI | Generalization | Reproducible data, but LLM prompt scoring is not local deterministic artifact scoring. |
| Weak-medium | Multi-hop QA/RAG: HotPotQA, HoVer, Open-SQuAD, QReCC, Natural Questions | Generalization | Useful if the benchmark grows an LLM-harness track; otherwise too model-dependent. |
| Weak-medium | Summarization/NLG/style transfer: XSUM, CNN/DailyMail, SAMSum, WebNLG, E2E NLG, ASSET, Yelp style transfer, Shakespeare style transfer | Generalization | Often needs reference metrics or judges; less reliable than exact-score tasks. |
| Weak-medium | Translation/dialog/state tracking: IWSLT, en-de/en-es/en-fr translation, MultiWOZ, DSTC7, Ubuntu Dialog, MuTual | Generalization | Requires model calls and metrics with more variance. |
| Weak | Medical/biomedical QA and NLP: MedQA, MedMCQA, PubMedQA, NCBI disease NER, biomedical similarity | Generalization | Domain importance is high, but contamination, licensing, and evaluation burden are high. |
| Weak | Safety/factuality/toxicity/jailbreak: ETHOS, LIAR, sarcasm, jailbreak detection, TruthfulQA | Generalization | Relevant to prompt optimization, but often judge/model dependent and not a good local program benchmark. |
| Weak | IFEval/IFBench/IFEval-Hard, instruction-following constraints | Generalization | Good for prompt adherence, weak fit for standalone Python artifact optimization. |
| Weak | LLM-as-judge industrial tasks, proprietary preference tasks | Generalization | Hard to reproduce openly; judge drift is a core reliability problem. |

## Recommended Additions

If adding tasks to `text-opt-bm`, the highest-return additions are:

1. **`bin_pack_online`**: implement online bin packing over fixed seeded item streams. Score = bins used or waste. It can be perfect-information on a fixed deployment stream suite, or generalized with train/validation/test stream families.
2. **`bbob_tiny` or `optimizer_bbob`**: optimize a pure-Python black-box optimizer over a small seeded suite of deterministic functions. Score = area over convergence curve or final regret under instruction budget.
3. **`knapsack_budget`**: choose items under capacity across fixed or generated instances. Score = negative value gap or approximation gap, with exact DP reference for small instances.
4. **`graph_coloring_budget` or `maxsat_budget`**: produce high-quality feasible solutions under bytecode budget. Score = colors/unsatisfied clauses.
5. **`code_efficiency_micro`**: optimize small functions for deterministic instruction counts, not wall-clock runtime. This imports the spirit of EvalPerf without its runtime flakiness.
6. **`mini_ahc`**: one small AtCoder-Heuristic-style scheduling/routing task with generated instances and visualization optional, not required.

## Practical Ranking Summary

Best direct fit:

`bin packing`, `TSP/routing`, `graph algorithms`, `compression`, `memory data structures`, `tiny inference`, `BBOB optimizer generation`, `knapsack/set cover/VRP micro-instances`, `SAT/graph coloring`, `synthetic word problems`.

Good but likely separate track:

`ALE-style long-horizon contests`, `code efficiency benchmarks`, `SWE-Perf`, `ML/Kaggle engineering`, `matrix multiplication`, `geometry packing`.

Poor fit for the current local deterministic harness:

`public prompt-optimization NLP tasks`, `RAG/multi-hop QA prompts`, `medical QA prompts`, `LLM instruction-following`, `LLM-as-judge industrial prompt tasks`, `visual/VLM-judged artifacts`.
