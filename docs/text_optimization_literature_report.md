# Text Optimization Literature and Benchmark Task Report

Date: 2026-07-03

## Scope used for this report

This project defines text optimization as iterative improvement of a text artifact, usually source code, using an evaluator that returns a numeric score. The existing benchmark is closest to the "LLM as optimizer / executable artifact search" line of work, not to parameter training, continuous soft prompts, or ordinary code-generation benchmarks.

I therefore scoped "relevant papers" to work that optimizes a discrete text-representable artifact: prompts, instructions, LM-program prompts, source code, harness code, heuristics, molecules-as-strings, or other textual design objects. I included soft-prompt/RL papers only where they are major baselines in automatic prompt optimization (APO).

## Current Project Summary

`text-opt-bm` is a benchmark for iterative program-text optimization by LLM/code-agent loops.

- Artifact optimized: a single Python `program.py`.
- Objective: lower a deterministic numeric score.
- Harness API: `bench.runner.evaluate(task, program_path)` returns one JSON result with `ok`, `score`, `metrics`, and `error`.
- Metrics intentionally avoid wall-clock timing. Current scoring uses traced memory, compressed bytes, bytecode instruction counts, tour length under instruction budget, and validation error rate.
- Optimization loop: `loop/optimize.py` implements a greedy hill-climb that repeatedly asks Codex to edit `program.py`, scores it, and accepts strict improvements.
- Anti-overfitting design: tasks are labeled as perfect-information or generalization tasks; generalization tasks expose train/validation and hide final test. Several tasks also validate unseen data to reject hardcoded benchmark gaming.

Current tasks:

| Task | Kind | Optimized artifact behavior | Metric | Closest literature family |
|---|---:|---|---|---|
| `mem_kv` | perfect | compact key/value store | resident traced bytes | program/code optimization, data-structure design |
| `mem_index` | perfect | compact inverted index | serving peak traced bytes | IR/data-structure optimization |
| `mem_infer` | perfect | tiny transformer decoding | peak traced bytes | systems/program optimization |
| `compress` | perfect | lossless compressor | compressed bytes | code artifact optimization; compression challenge |
| `ops_connect` | perfect | dynamic connectivity | bytecode instructions | algorithm design / heuristic search |
| `tsp_budget` | perfect | TSP under instruction budget | tour length | combinatorial optimization / heuristic design |
| `word_problems` | generalization | rule-based math word-problem solver | hidden validation error | prompt-optimization-like train/val/test generalization, GSM8K-like |
| `compress_heldout` | generalization | compressor with train/val/test | hidden validation bytes | generalization-aware artifact optimization |

## Most Relevant Paper Families

### 1. LLM-as-optimizer and prompt optimization

These papers optimize natural-language prompts or instructions. They are useful because they establish the dominant benchmark pattern: optimize on a training/validation split, evaluate downstream task accuracy on held-out tasks, and report task clusters such as GSM8K, BBH, instruction induction, Natural Instructions, and sentiment/classification datasets.

| Paper | Artifact optimized | Optimization signal | Benchmarked tasks | Paper | Code |
|---|---|---|---|---|---|
| Automatic Prompt Engineer / APE, "Large Language Models Are Human-Level Prompt Engineers" (Zhou et al., 2022/ICLR 2023) | task instructions | candidate generation + task score | 24 Instruction Induction tasks; 21 BIG-Bench Hard tasks; TruthfulQA/informativeness style applications | https://arxiv.org/abs/2211.01910 | https://github.com/keirp/automatic_prompt_engineer |
| GrIPS, "Gradient-free, Edit-based Instruction Search" (Prasad et al., 2022/2023) | human-written instructions | edit search over phrases | 8 Natural Instructions classification tasks | https://arxiv.org/abs/2203.07281 | https://github.com/archiki/GrIPS |
| RLPrompt, "Optimizing Discrete Text Prompts with Reinforcement Learning" (Deng et al., EMNLP 2022) | discrete prompt tokens | RL reward from task performance | classification and text-style transfer | https://aclanthology.org/2022.emnlp-main.222/ | https://github.com/mingkaid/rl-prompt |
| APO/ProTeGi, "Automatic Prompt Optimization with Gradient Descent and Beam Search" (Pryzant et al., EMNLP 2023) | prompts | textual gradients from error cases + beam search | Jailbreak detection, LIAR, Sarcasm, ETHOS | https://aclanthology.org/2023.emnlp-main.494/ | Microsoft LMOps repo: https://github.com/microsoft/LMOps |
| OPRO, "Large Language Models as Optimizers" (Yang et al., ICLR 2024) | numeric solutions and prompts | LLM proposes new candidates from prior candidate/score history | linear regression, TSP, GSM8K, BBH, MultiArith, AQuA | https://arxiv.org/abs/2309.03409 | https://github.com/google-deepmind/opro |
| PromptBreeder (Fernando et al., 2023/2024) | task prompts and mutation prompts | evolutionary prompt mutation | GSM8K, MultiArith, AddSub, SVAMP, SingleEq, AQuA-RAT, CSQA, StrategyQA, ETHOS, Instruction Induction | https://arxiv.org/abs/2309.16797 | no official repo found; community implementations include https://github.com/vaughanlove/PromptBreeder |
| EvoPrompt (Guo et al., 2023/2024) | prompts | GA/DE-style evolutionary operators | SST-2/5, CR, MR, AGNews, TREC, Subj, SAMSum, ASSET, BBH reasoning tasks | https://arxiv.org/abs/2309.08532 | https://github.com/beeevita/EvoPrompt |
| PromptAgent (Wang et al., ICLR 2024) | prompts | MCTS / strategic planning over prompt edits | 12 tasks: BBH subset, biomedical NER/QA/similarity, TREC, Subj, CB | https://openreview.net/forum?id=22pyNMuIoa | https://github.com/xinyuanwangcs/PromptAgent |
| PE2, "Prompt Engineering a Prompt Engineer" (Ye et al., 2023/2024) | prompts | stronger meta-prompted prompt engineer | MultiArith, GSM8K, Instruction Induction, BBH, counterfactual and production prompts | https://arxiv.org/abs/2311.05661 | no official repo found |
| PROMST, "PRompt Optimization in Multi-Step Tasks" (Chen et al., EMNLP 2024) | agent prompts | score model + human feedback + heuristic sampling | WebArena, ALFWorld, ScienceWorld, BoxNet, BoxLift, Warehouse, Gridworld, Blocksworld, Logistics | https://aclanthology.org/2024.emnlp-main.226/ | https://github.com/yongchao98/PROMST |
| DSPy, "Compiling Declarative Language Model Calls into Self-Improving Pipelines" (Khattab et al., 2023/2024) | LM program prompts/demos | compiler optimizes prompts, demos, and sometimes finetuning | math word problems, multi-hop retrieval/QA, complex QA, agent loops | https://arxiv.org/abs/2310.03714 | https://github.com/stanfordnlp/dspy |
| MIPRO, "Optimizing Instructions and Demonstrations for Multi-Stage Language Model Programs" (Opsahl-Ong et al., EMNLP 2024) | instructions and demonstrations per LM-program module | Bayesian / surrogate-guided prompt configuration search | HotPotQA, Iris, Heart Disease, ScoNe, HoVer and other LM programs | https://arxiv.org/abs/2406.11695 | in DSPy: https://github.com/stanfordnlp/dspy |
| TextGrad (Yuksekgonul et al., 2024) | text variables in computation graphs | LLM-generated textual gradients | Google-proof QA, LeetCode-Hard code solutions, MMLU subsets, BBH object counting/word sorting, GSM8K, DOCKSTRING molecules, radiotherapy planning | https://arxiv.org/abs/2406.07496 | https://github.com/zou-group/textgrad |
| GEPA, "Reflective Prompt Evolution Can Outperform Reinforcement Learning" (Agrawal et al., 2025/ICLR 2026 oral) | textual components of LM systems | reflection on trajectories + Pareto/evolutionary search | HotpotQA, IFBench, HoVer, PUPA, AIME-2025, LiveBench-Math; also code-optimization case study | https://arxiv.org/abs/2507.19457 | https://github.com/gepa-ai/gepa |
| Feedback Descent (Lee et al., 2025/2026) | prompts, code, molecules, SVG-like artifacts | pairwise preference + structured textual feedback | prompt optimization, code/text artifacts, DOCKSTRING molecule discovery, SVG/image optimization | https://arxiv.org/abs/2511.07919 | https://github.com/JaredJoss/feedback-descent |
| PrefPO, "Optimizing Prompts through Preferences" (Distyl, 2026) | prompts | pairwise preferences, with and without labels | BBH subset and IFEval-Hard, compared to GEPA/MIPRO/TextGrad | https://www.distyl.ai/blog/research/prefpo-optimizing-prompts-through | linked from blog |

### 2. Source-code, harness, and algorithm discovery

These are closest to `text-opt-bm`, because they ask an LLM loop to edit executable text and score it with a programmatic evaluator.

| Paper | Artifact optimized | Optimization signal | Benchmarked tasks | Paper | Code |
|---|---|---|---|---|---|
| FunSearch, "Mathematical discoveries from program search with LLMs" (Romera-Paredes et al., Nature 2024) | Python functions | evolutionary code search with evaluator | cap set constructions; online bin packing heuristics | https://www.nature.com/articles/s41586-023-06924-6 | https://github.com/google-deepmind/funsearch |
| EoH, "Evolution of Heuristics" (Liu et al., 2024) | heuristic thoughts + code | LLM + evolutionary computation | automatic heuristic design; commonly reported on combinatorial optimization such as TSP, bin packing, or routing variants depending on setup | https://arxiv.org/abs/2401.02051 | https://github.com/FeiLiu36/EoH |
| ReEvo, "Large Language Models as Hyper-Heuristics with Reflective Evolution" (Ye et al., NeurIPS 2024) | heuristic code | evolutionary search plus LLM reflection | TSP, VRP, Orienteering, MKP, BPP, EDA | https://arxiv.org/abs/2402.01145 | project/code linked from OpenReview: https://openreview.net/forum?id=483IPG0HWL |
| AlphaEvolve (Novikov et al., 2025) | algorithms/source code | evolutionary coding agent with one or more evaluators | Google datacenter scheduling, hardware circuit simplification, LLM training kernels/procedures, matrix multiplication, geometry/packing, other math/CS discovery problems | https://arxiv.org/abs/2506.13131 | official system not public; open-source clone: https://github.com/algorithmicsuperintelligence/openevolve |
| OpenEvolve (software, 2025) | source code | open implementation inspired by AlphaEvolve | user-defined evaluator tasks; examples replicate some AlphaEvolve-style tasks | software page: https://github.com/algorithmicsuperintelligence/openevolve | https://github.com/algorithmicsuperintelligence/openevolve |
| Meta-Harness (Lee et al., 2026) | harness code around a fixed model | agentic proposer reads prior code, scores, traces via filesystem | online text classification, retrieval-augmented IMO/math reasoning, TerminalBench-2 agentic coding | https://arxiv.org/abs/2603.28052 | https://github.com/stanford-iris-lab/meta-harness |
| optimize_anything / GEPA API (2026) | arbitrary text parameters: code, prompts, configs, SVGs, agent architectures | evaluator-driven GEPA-style text evolution | AI agent discovery, cloud scheduling, CUDA kernel generation, geometric packing, math optimization tasks | https://arxiv.org/html/2605.19633v1 | https://github.com/gepa-ai/gepa and artifact: https://github.com/gepa-ai/optimize-anything-artifact |

### 3. Benchmark-suite papers to treat as direct competitors or sources of task ideas

These explicitly try to provide standard evaluation sets. They should be highlighted if this project wants to position itself as an official benchmark.

| Benchmark paper | What it benchmarks | Task set | Why it matters here | Paper | Code/data |
|---|---|---|---|---|---|
| BLADE, "Benchmark suite for LLM-driven Automated Design and Evolution of iterative optimisation heuristics" (van Stein et al., GECCO 2025) | LLM-driven automated algorithm design for continuous black-box optimization | MA-BBOB, SBOX-COST and other continuous optimization problem collections; includes instance generators, textual descriptions, logging, IOH analysis integration | Explicit official benchmark suite for LLM-driven algorithm design; strong model for logging/reproducibility | https://arxiv.org/abs/2504.20183 | https://github.com/XAI-liacs/BLADE |
| CO-Bench, "Benchmarking Language Model Agents in Algorithm Search for Combinatorial Optimization" (Sun et al., AAAI 2026) | LLM agents that develop algorithms for combinatorial optimization | 36 real-world CO problems with structured formulations and curated data | Official benchmark suite for algorithm-search agents; overlaps with this repo's `tsp_budget`/`ops_connect` spirit | https://arxiv.org/html/2504.04310v1 | https://github.com/sunnweiwei/CO-Bench and https://sunnweiwei.github.io/co-bench |
| HeuriGym, "An Agentic Benchmark for LLM-Crafted Heuristics in Combinatorial Optimization" (Chen et al., ICLR 2026) | agentic, code-driven heuristic design | practical scientific/engineering CO problems; reports stage-wise metrics such as SOLVEs@i and QYI | Official benchmark for iterative heuristic construction; relevant to agent interaction and not just final score | https://arxiv.org/abs/2506.07972 | https://github.com/cornell-zhang/heurigym and https://cornell-zhang.github.io/heurigym/ |
| ALE-Bench, "A Benchmark for Long-Horizon Objective-Driven Algorithm Engineering" (Imajuku et al., NeurIPS 2025 Datasets & Benchmarks) | score-based algorithm engineering agents | AtCoder Heuristic Contest-derived optimization tasks such as routing, scheduling, production planning, and power-grid balancing | One of the closest official benchmarks to this repo's long-horizon iterative score optimization, but larger and contest-oriented | https://arxiv.org/abs/2506.09050 | https://github.com/SakanaAI/ALE-Bench and https://huggingface.co/datasets/SakanaAI/ALE-Bench |
| promptolution (2025/2026) | modular prompt-optimization methods | framework-level benchmark experiments over NLP tasks and APO algorithms | More of a reproducibility framework than a new task benchmark, but useful for standardizing prompt optimizer comparisons | https://arxiv.org/html/2512.02840v2 | https://github.com/automl/promptolution |
| MIPRO / DSPy benchmark release | prompt optimization for multi-stage LM programs | seven diverse LM programs; common tasks include HotPotQA, HoVer, ScoNe, Iris, Heart Disease | Not branded as a broad benchmark suite, but it released reusable optimizer infrastructure and benchmark programs in DSPy | https://arxiv.org/abs/2406.11695 | https://github.com/stanfordnlp/dspy |
| PROMST multistep task suite | prompt optimization for long-horizon agents | WebArena, ALFWorld, ScienceWorld, BoxNet/BoxLift/Warehouse/Gridworld/Blocksworld/Logistics | Not an official generic benchmark, but a compact task suite for prompt optimization beyond single-step NLP | https://aclanthology.org/2024.emnlp-main.226/ | https://github.com/yongchao98/PROMST |

## Benchmark Tasks Used Across Papers

### Common NLP prompt-optimization task clusters

| Task cluster | Papers using it | Task description | Notes for `text-opt-bm` |
|---|---|---|---|
| GSM8K / grade-school math | APE, OPRO, PromptBreeder, PE2, PromptWizard, PRewrite, OIRL, TextGrad, DSPy-style papers, PrefPO comparisons | Solve natural-language arithmetic word problems, usually exact numeric answer accuracy | Current `word_problems` intentionally mirrors this but uses synthetic unseen splits to avoid frontier-model memorization. This is one of the most shared task families. |
| MultiArith / AddSub / SVAMP / SingleEq / AQuA-RAT / MAWPS | OPRO, PromptBreeder, PE2, PromptWizard, OIRL, Boosted Prompting | Smaller arithmetic word-problem datasets with varying equation extraction and multi-step reasoning demands | Useful if adding public baselines, but leakage/memorization risk is high. |
| BIG-Bench Hard (BBH) | APE, OPRO, PromptBreeder, EvoPrompt, PromptAgent, PE2, PromptWizard, SCULPT, StraGo, TextGrad, PrefPO | 23 challenging BIG-Bench reasoning tasks: symbolic, logical, commonsense, arithmetic, tracking, etc. | The dominant shared benchmark for prompt optimizers. It tests prompt quality, not executable code optimization. |
| Instruction Induction / BIG-Bench Instruction Induction | APE, PromptBreeder, PromptWizard, PACE, MOP, AutoHint | Given examples, infer the hidden instruction; tasks span spelling, syntax, morphology, lexical semantics, phonetics, knowledge, semantics, style | Closest to "optimize instructions from examples"; less close to `program.py` optimization. |
| Natural Instructions / Super-NaturalInstructions | GrIPS, PLUM, MOP, SAMMO | Many instruction-following tasks, often classification or structured generation | Good for broad prompt generalization but large and LLM-output-dependent. |
| Sentiment/topic/question classification | GrIPS, ProTeGi, EvoPrompt, PromptAgent, Random Separators, AMPO, DAPO, COPLE, PRewrite | SST-2/5, CR, MR, AGNews, TREC, Subj, DBPedia, MPQA | Very common APO tasks. Most evaluate label accuracy on fixed train/test splits. |
| Factuality / misinformation / toxicity / safety classification | ProTeGi, PromptBreeder, PREFER, PromptWizard, SOS | LIAR, ETHOS, jailbreak/safety datasets, sarcasm datasets | Useful for multi-objective or safety-aware prompt optimization; less relevant to deterministic program score. |
| Multi-hop QA and retrieval | DSPy, MIPRO, GEPA, Meta-Harness, TextGrad variants | HotPotQA, HoVer, RAG over math/IMO-style corpora | Strong fit if `text-opt-bm` ever adds harness optimization around retrieval/code agents. |
| Medical QA / biomedical NLP | PromptAgent, AMPO, PromptWizard, StraGo, UNIPROMPT | MedQA, MedMCQA, PubMedQA, NCBI disease NER, biomedical sentence similarity | High-stakes and model-leakage-prone; useful mainly as examples of domain transfer. |
| LLM instruction-following | GEPA, PrefPO, IFBench/IFEval-Hard | Follow format/content constraints exactly | Could inspire deterministic format-compliance tasks with hidden tests. |

### Code, algorithm, and optimization task clusters

| Task cluster | Papers using it | Task description | Notes for `text-opt-bm` |
|---|---|---|---|
| Traveling Salesman Problem | OPRO, LMEA, EoH/ReEvo-family papers, CO-Bench/HeuriGym-type suites, current `tsp_budget` | Find short tours, often under compute/search budget | A direct shared task with this repo. Your instruction-budget variant is a distinctive deterministic alternative to time budgets. |
| Online bin packing / bin packing | FunSearch, EoH/ReEvo/HSEvo-family papers, CO-Bench/HeuriGym | Design a heuristic to pack online item streams into few bins | Strong candidate for a future `text-opt-bm` task: simple correctness, scalar score, clear generalization splits. |
| Dynamic graph / connectivity-like algorithms | Current `ops_connect`; related CO/algorithm-design benchmarks include graph optimization tasks | Maintain/query graph structure efficiently | Less common in prompt optimizer papers; valuable differentiator. |
| Matrix multiplication / algorithmic discovery | AlphaEvolve, CodeEvolve/OpenEvolve-style replications | Search over algorithmic decompositions or code to reduce scalar multiplications / improve performance | High-prestige but hard to make lightweight and cheat-resistant. |
| Compression | Current `compress` and `compress_heldout`; not common in APO literature | Write compressor/decompressor optimized for bytes and generalization | Distinctive contribution. It is easy to score deterministically and exposes train/heldout overfitting cleanly. |
| LeetCode / coding tasks | TextGrad, Meta-Harness/TerminalBench adjacent papers | Improve or solve code problems, often pass tests or improve runtime | Strongly related, but public tests can be gamed; hidden tests and deterministic resource metrics are key. |
| Harness/context-management code | Meta-Harness, optimize_anything | Optimize the wrapper around an LLM: retrieval, memory, prompt composition, tool usage | Very close philosophically to this repo's agent-loop benchmark, but usually expensive to run. |
| Continuous black-box optimization heuristics | BLADE, LLaMEA-family work | Generate iterative optimizers/metaheuristics for BBOB-like functions | Adjacent: code artifact optimization with standardized benchmark logging. |
| Molecule optimization | TextGrad, Feedback Descent | Optimize SMILES/string-like molecules against binding/druglikeness objectives such as DOCKSTRING | Demonstrates non-code text artifacts with scalar scientific objectives. |
| SVG/image/textual design optimization | Feedback Descent, optimize_anything | Optimize text descriptions or SVG/code artifacts under preference/scoring functions | Useful if extending beyond Python programs. |

## Papers Sharing the Same Tasks

These are the clearest reuse clusters. If this project wants comparable external baselines, these are the strongest task bridges.

1. GSM8K / math word problems:
   - APE, OPRO, PromptBreeder, PE2, PromptWizard, TextGrad, PRewrite, OIRL, DSPy, GEPA-adjacent prompt evaluations, PrefPO-style comparisons.
   - Current project analogue: `word_problems`.
   - Caution: public GSM8K is heavily contaminated in modern models; this repo's synthetic generator is a sensible replacement.

2. BBH:
   - APE, OPRO, PromptBreeder, EvoPrompt, PromptAgent, PE2, PromptWizard, SCULPT, TextGrad, StraGo, PrefPO, GEPA comparisons.
   - Shared role: general reasoning prompt optimization.
   - Caution: prompt optimizers often report accuracy only, not artifact cost, run budget, or overfitting curves.

3. Instruction Induction / Natural Instructions:
   - APE, GrIPS, PromptBreeder, PromptWizard, MOP, PLUM, SAMMO, AutoHint.
   - Shared role: optimizing instructions from examples.
   - Caution: many papers use different task subsets, so headline averages are not always comparable.

4. TSP / combinatorial optimization:
   - OPRO uses TSP as a toy optimizer task; EoH/ReEvo/HeuriGym/CO-Bench/BLADE-adjacent work uses CO tasks more seriously; this repo has `tsp_budget`.
   - This is the strongest bridge between prompt-optimizer literature and executable-code optimization.

5. HotPotQA / HoVer / multi-hop QA:
   - DSPy, MIPRO, GEPA, Meta-Harness use these for LM-program or harness optimization.
   - Relevant if this project evolves from single `program.py` tasks toward optimizing multi-file harnesses or retrieval workflows.

6. Sentiment/classification datasets:
   - GrIPS, ProTeGi, EvoPrompt, PromptAgent, DAPO, AMPO, COPLE, PRewrite, Random Separators.
   - Shared role: cheap prompt optimization benchmark.
   - Less aligned with the project because the optimized artifact is usually a prompt for a task model, not a standalone program.

7. Bin packing:
   - FunSearch and many heuristic-design papers use it.
   - Strong candidate future task because scoring is deterministic, solutions are executable heuristics, and train/heldout streams are easy to generate.

## Broader APO Paper Inventory

The table below captures additional APO papers and task sets reported in the 2025 automatic prompt optimization survey. These are less directly aligned with this repo than executable-artifact search, but they are relevant if the project positions itself against the full prompt-optimization literature.

| Paper / method | Reported benchmark tasks |
|---|---|
| GPS (Xu et al., 2022) | 10 unseen T0 tasks: ANLI R1/R2/R3, CB, RTE, WSC, Winogrande, COPA, HellaSwag, WiC |
| Instruction Induction (Honovich et al., 2022) | 24 instruction-induction tasks spanning spelling, syntax, morpho-syntax, lexical semantics, phonetics, knowledge, semantics, style |
| TEMPERA (Zhang et al., 2022) | Classification |
| AELP (Hsieh et al., 2024) | Big-Bench Hard |
| AutoHint (Sun et al., 2023) | BB instruction-induction tasks including epistemic reasoning, logical fallacy detection, implicatures, hyperbaton, causal judgment, Winowhy |
| BDPL (Diao et al., 2022) | MNLI, QQP, SST-2, MRPC, CoLA, QNLI, RTE, CitationIntent, SciERC, RCT, HyperPartisan |
| Boosted Prompting (Pitis et al., 2023) | GSM8K, AQuA |
| BPO (Cheng et al., 2024) | Dolly Eval, Vicuna Eval, Self-Instruct Eval |
| Directional Stimulus Prompting (Li et al., 2023) | MultiWOZ |
| DLN / joint prompt optimization of stacked LLMs (Sordoni et al., 2023) | MPQA, TREC, Subj, Leopard Disaster, Leopard Airline, BBH Hyperbaton/Navigate/Date/Logic |
| DSP (Khattab et al., 2022) | Open-SQuAD, HotPotQA, QReCC |
| GATE (Joko et al., 2024) | LAPS: content recommendation, moral reasoning, email verification |
| GPO (Li et al., 2023) | Yelp, Flipkart, IMDB, Amazon sentiment; MNLI/ANLI/RTE; SocialIQA; DSTC7, Ubuntu Dialog, MuTual; DROP |
| PACE (Dong et al., 2024) | BBH, 24 instruction-induction tasks, translation en-de/en-es/en-fr |
| PREFER (Zhang et al., 2024) | SNLI, MNLI, QNLI, RTE, ETHOS, LIAR, ArSarcasm |
| PromptBoosting (Hou et al., 2023) | Text classification |
| Random Separators (Lu et al., 2024) | SST-2, SST-5, DBPedia, MR, CR, MPQA, Subj, TREC, AGNews |
| ABO (Yang et al., 2024) | BBH object counting, navigate, snarks, question selection |
| Adv-ICL (Long et al., 2024) | XSUM, CNN/DailyMail, WebNLG, E2E NLG, LIRO, TED Talks, Yelp-5, WSC, GSM8K, SVAMP |
| AMPO (Yang et al., 2024) | TREC, SST-5, RACE, MedQA, MedMCQA |
| APEER (Jin et al., 2024) | Passage reranking |
| APOHF (Lin et al., 2024) | InstructZero user-instruction optimization, text-to-image, response optimization |
| BATPrompt (Shi et al., 2024) | Language understanding, summarization, simplification |
| COPLE (Zhan et al., 2024) | GLUE SST-2/CoLA/MNLI/QNLI/RTE/MRPC/QQP; MMLU STEM/Humanities/Social Sciences/Other |
| CRISPO (He et al., 2025) | Summarization and QA |
| DAPO (Yang et al., 2024) | Sentiment, topic classification, news, TREC, Subj, Logic Five, Hyperbaton, Disambiguation, Salient, translation |
| DRPO (Amini et al., 2024) | Alignment benchmark |
| FIPO (Lu et al., 2025) | GSM8K, BBH, PiQA, CosmosQA, MMLU |
| LMEA (Liu et al., 2023) | Traveling Salesman Problems |
| MOP (Wang et al., 2025) | 50 tasks from Instruction Induction, Super-NaturalInstructions, BBH |
| MORL-Prompt (Jafari et al., 2024) | Shakespearean style transfer, IWSLT2017 machine translation |
| OIRL (Sun et al., 2024) | GSM8K, MAWPS, SVAMP |
| PIN (Choi et al., 2024) | SST-2 and related classification; Yelp style transfer; MSCOCO/LAION textual inversion |
| PLUM (Pan et al., 2024) | Natural-Instructions v2.6 |
| PRewrite (Kong et al., 2024) | AG News, SST-2, Natural Questions, GSM8K |
| PromptWizard (Agarwal et al., 2024) | BBII, GSM8K, AQuA-RAT, SVAMP, BBH, MMLU, ETHOS, PubMedQA, MedQA |
| Reprompting (Xu et al., 2024) | BBH, GSM8K, MATH |
| SAMMO (Schnabel and Neville, 2024) | BigBench zero-shot classification; GeoQuery, SMCalFlow, Overnight; Super-NaturalInstructions |
| SCULPT (Kumar et al., 2024) | BBH, RAI |
| SOS / Survival of the Safest (Sinha et al., 2024) | Sentiment, orthography, taxonomy, disambiguation QA, Logical Five, color reasoning |
| SPRIG (Zhang et al., 2024) | 47 task types across MMLU, BBH, TruthfulQA, XCOPA, SocKET and related multilingual/domain tasks |
| StraGo (Wu et al., 2024) | BBH subset, SST-5, TREC, MedQA, MedMCQA, internal personalized intent query |
| UNIPROMPT (Juneja et al., 2024) | ETHOS, ARC, MedQA, GSM8K, real-world search query intent |

## Additional Sweep Addendum

This addendum records papers and benchmark resources found in a second sweep. They were not in the first core table because they are either older gradient/soft-prompt work, newer framework papers, or code-efficiency benchmarks adjacent to this repo.

### Early and soft/black-box prompt optimization

| Work | Why it is relevant | Tasks / benchmarks | Paper | Code |
|---|---|---|---|---|
| AutoPrompt, "Eliciting Knowledge from Language Models with Automatically Generated Prompts" (Shin et al., EMNLP 2020) | Early discrete trigger-token optimization using gradients; important ancestor of APO | fact retrieval and text classification/probing tasks | https://aclanthology.org/2020.emnlp-main.346/ | https://github.com/ucinlp/autoprompt |
| Prefix-Tuning (Li and Liang, 2021) | Optimizes continuous prefix parameters rather than text; adjacent but not directly comparable to this repo | generation tasks including table-to-text and summarization | https://arxiv.org/abs/2101.00190 | code linked from paper/project pages |
| Prompt Tuning / P-Tuning family | Continuous prompt-vector optimization; important background, but outside discrete text-artifact search | GLUE/SuperGLUE, NLU and generation tasks depending on paper | survey entry: https://arxiv.org/abs/2502.11560 | various |
| BBT / Black-Box Tuning (Sun et al., ICML 2022) | Derivative-free soft prompt optimization for LM-as-a-service | common language-understanding tasks; BBTv2 extends to more-label and larger models | https://arxiv.org/abs/2201.03514 | https://github.com/txsun1997/Black-Box-Tuning |
| InstructZero (Chen et al., ICML 2024) | Black-box instruction optimization, closer to discrete APO | broad downstream instruction-following/NLP tasks; reports outperforming auto-instruction baselines | https://proceedings.mlr.press/v235/chen24e.html | https://github.com/lichang-chen/instructzero |
| BBT-RGB (Sun et al., LREC-COLING 2024) | Improves black-box prompt tuning generalization | black-box prompt tuning tasks over classification/entailment style datasets | https://aclanthology.org/2024.lrec-main.956/ | https://github.com/QiushiSun/BBT-RGB |

### Additional prompt-optimization frameworks and methods

| Work | Why it is relevant | Tasks / benchmarks | Paper | Code |
|---|---|---|---|---|
| CAPO, "Cost-Aware Prompt Optimization" (Zehle et al., 2025) | Explicitly optimizes prompt performance under evaluation/token cost; useful for this repo's future score-vs-budget reporting | 15 prompt-optimization cases across diverse datasets/LLMs | https://arxiv.org/abs/2504.16005 | https://github.com/finitearth/capo |
| Promptomatix (Murthy et al., 2025) | Framework paper combining lightweight meta-prompting with DSPy/MIPROv2-style compilation | prompt optimization over LLM task descriptions; paper reports modular APO experiments | https://arxiv.org/abs/2507.14241 | https://github.com/salesforceairesearch/promptomatix |
| Prompt Optimization as a State-Space Search Problem (Taneja, 2025) | Directly frames prompts as graph search over text transformations | five NLP tasks: sentiment, QA, summarization, reasoning, NLI | https://arxiv.org/abs/2511.18619 | https://github.com/MaanasTaneja/PromptOptimiser |
| System Prompt Optimization with Meta-Learning (2025) | Optimizes system prompts across source tasks for generalization | multi-task user-prompt/source-task setup | https://arxiv.org/html/2505.09666v1 | no official repo found |
| AutoPDL, "Automatic Prompt Optimization for LLM Agents" (2025) | Agent prompt optimization; relevant to optimizing scaffolds rather than single prompts | LLM-agent tasks; paper cites CrewAI-style agent orchestration | https://arxiv.org/abs/2504.04365 | no official repo found |
| MPCO, "Tuning LLM-based Code Optimization via Meta-Prompting" (2025) | Optimizes prompts used for code optimization workflows | industrial Artemis code optimization/validation setting | https://arxiv.org/html/2508.01443v1 | no official repo found |
| CoolPrompt (2025) | Open prompt optimization framework | framework-level APO experiments | https://openreview.net/forum?id=XGECnjDEcS | https://github.com/CTLab-ITMO/CoolPrompt |
| PROPANE, "Prompt Design as an Inverse Problem" (2023) | Prompt reconstruction/inversion from outputs; adjacent to prompt design, less directly benchmark comparable | prompt reconstruction/style recovery tasks | https://arxiv.org/abs/2311.07064 | community/forked code appears under `propane` repositories |
| MASPO, "Joint Prompt Optimization for LLM-based Multi-Agent Systems" (Wang et al., 2026) | Jointly optimizes role-specific prompts across interacting agents, addressing local/global credit assignment | 6 multi-agent/collaborative tasks; reports average improvement over existing prompt optimizers | https://arxiv.org/abs/2605.06623 | https://github.com/wangzx1219/MASPO |
| SPEAR, "Code-Augmented Agentic Prompt Optimization" (Lu et al., 2026) | Free-form prompt optimizer with `evaluate`, `python`, `set_prompt`, `finish` tools; optimizer writes analysis code over eval data | 13 industrial LLM-as-judge tasks, 7 BBH tasks, GSM8K; compares to GEPA/TextGrad | https://arxiv.org/abs/2605.26275 | no official repo found in sweep |
| FAPO, "Fully Autonomous Prompt Optimization of Multi-Step LLM Pipelines" (Cisco Foundation AI, 2026) | Uses a coding agent to optimize prompts, skills, parameters, and pipeline structure with step-level failure attribution | six benchmarks across three task models; compared against GEPA | https://arxiv.org/abs/2606.19605 | https://github.com/cisco-foundation-ai/fully-automated-prompt-optimization |

### Additional code, algorithm-engineering, and performance-optimization benchmarks

| Work | Why it is relevant | Tasks / benchmarks | Paper | Code/data |
|---|---|---|---|---|
| LLaMEA, "A Large Language Model Evolutionary Algorithm for Automatically Generating Metaheuristics" (van Stein and Back, 2024/2025) | LLM evolves metaheuristic code; strong direct analogue for `text-opt-bm`'s program optimization | BBOB / black-box optimization functions; 24 BBOB functions in public examples | https://arxiv.org/abs/2405.20132 | https://github.com/XAI-liacs/LLaMEA |
| LLM4AD platform (Liu et al., 2024/2025) | Unifies LLM-assisted algorithm design methods and tasks | optimization, ML, scientific-discovery algorithm design tasks; integrates EoH/ReEvo-style methods | https://arxiv.org/abs/2412.17287 | https://github.com/Optima-CityU/llm4ad |
| CodeEvolve (2025/2026) | Open-source evolutionary coding agent evaluated on AlphaEvolve/EoH-style tasks | AlphaEvolve benchmark subset and EoH benchmark suite | https://arxiv.org/html/2510.14150v5 | https://github.com/inter-co/science-codeevolve |
| AIDE, "AI-Driven Exploration in the Space of Code" (Jiang et al., 2025) | Frames ML engineering as code optimization/tree search | Kaggle tasks and research-oriented ML engineering benchmarks; third-party use on NAS and kernel optimization | https://arxiv.org/abs/2502.13138 | https://github.com/WecoAI/aideml |
| MLE-bench (Chan et al., 2024) | Official benchmark for end-to-end ML engineering agents; code is optimized through iterative experiments | 75 offline Kaggle competitions | https://arxiv.org/abs/2410.07095 | https://github.com/openai/mle-bench |
| EvalPerf / DPE (Liu et al., COLM 2024) | Evaluates efficient code generation and uses CPU instruction counts as one reliability tool, directly relevant to this repo's deterministic metric philosophy | 118-121 performance-challenging coding tasks derived from existing coding benchmarks | https://arxiv.org/abs/2408.06450 | https://github.com/evalplus/evalplus |
| SWE-Perf (He et al., 2025) | Repository-level code performance optimization benchmark | real GitHub repositories and performance-improvement pull requests; correctness plus performance | https://arxiv.org/abs/2507.12415 | https://github.com/SWE-Perf/SWE-Perf |
| EffiBench-X (2025) | Multi-language efficiency benchmark for LLM-generated code | competitive programming tasks in Python, C++, Java, JavaScript, Ruby, Go; runtime/memory efficiency | https://arxiv.org/abs/2505.13004 | https://github.com/EffiBench/EffiBench-X |
| PerfForge / WEDGE (2025) | Generates performance-stressing tests for code-efficiency evaluation | CodeContests subset with adversarial/stress tests exposing performance bottlenecks | https://arxiv.org/abs/2505.23471 | https://github.com/UChiSeclab/perfforge |

### Additional surveys and curated lists

| Resource | Use |
|---|---|
| "A Survey of Automatic Prompt Engineering: An Optimization Perspective" (Li et al., 2025), https://arxiv.org/abs/2502.11560 | Broader than the APO survey used above; includes discrete, continuous, and hybrid prompt spaces plus multimodal prompting. |
| "A Systematic Survey on Large Language Models for Algorithm Design" / LLM4Opt list, https://github.com/FeiLiu36/LLM4Opt | Useful for tracking LLM-driven algorithm design and heuristic-design papers. |
| Awesome LLM Prompt Optimization, https://github.com/jxzhangjhu/Awesome-LLM-Prompt-Optimization | Tracks newer APO papers such as CAPO, PROPANE, and related gradient-free methods. |
| Awesome FM4CO, https://github.com/ai4co/awesome-fm4co | Tracks combinatorial-optimization benchmarks and LLM heuristic-design papers, including CO-Bench, HeuriGym, ALE-Bench, EoH/ReEvo-family work. |

## Design Takeaways for `text-opt-bm`

1. The project is differentiated by deterministic resource metrics.
   - Most prompt optimization papers use task accuracy or LLM-judge scores, which can be noisy and model-dependent.
   - This repo's bytecode, memory, and byte-count metrics are unusually clean for agent-loop comparisons.

2. The strongest external comparison category is not classic APO; it is executable artifact search.
   - FunSearch, AlphaEvolve/OpenEvolve, Meta-Harness, optimize_anything, BLADE, CO-Bench, and HeuriGym are the closest relatives.
   - APO papers are still useful for train/validation/test protocol and task reuse.

3. Official benchmark positioning should emphasize:
   - stable local scoring;
   - no third-party services required for evaluation;
   - visible train vs hidden validation/test support;
   - artifact history and failed-attempt inspectability;
   - cheat-resistance checks on unseen data.

4. Task gaps worth filling:
   - Add an online bin packing task to connect directly to FunSearch and heuristic-design papers.
   - Add a CO-Bench-like small routing/scheduling task, but keep deterministic instruction counts rather than wall-clock time.
   - Add a harness/context-management task only if the project is willing to depend on model calls; otherwise it would violate the current cheap/local evaluator design.
   - Add a public "prompt-only" task only if it can avoid relying on volatile external model APIs. A local non-LLM solver task, like current `word_problems`, is more consistent.

5. Reporting suggestions:
   - For every run, plot score-at-iteration and score-vs-agent-token/cost, not just best final score.
   - For generalization tasks, always log train, validation, and hidden test every iteration.
   - For official benchmark credibility, publish task specs, initial programs, evaluator code, validation strategy, and reference improved solutions, as this repo already mostly does.

## Source Index

Primary sources used heavily:

- Project README: `README.md`
- Project analysis: `ANALYSIS.md`
- Automatic Prompt Optimization survey: https://arxiv.org/html/2502.16923v2
- APE: https://arxiv.org/abs/2211.01910 and https://github.com/keirp/automatic_prompt_engineer
- GrIPS: https://arxiv.org/abs/2203.07281 and https://github.com/archiki/GrIPS
- APO/ProTeGi: https://aclanthology.org/2023.emnlp-main.494/
- OPRO: https://arxiv.org/abs/2309.03409 and https://github.com/google-deepmind/opro
- PromptBreeder: https://arxiv.org/abs/2309.16797
- EvoPrompt: https://arxiv.org/html/2309.08532v3 and https://github.com/beeevita/EvoPrompt
- PromptAgent: https://openreview.net/forum?id=22pyNMuIoa and https://github.com/xinyuanwangcs/PromptAgent
- DSPy: https://arxiv.org/abs/2310.03714 and https://github.com/stanfordnlp/dspy
- MIPRO: https://arxiv.org/abs/2406.11695
- TextGrad: https://arxiv.org/abs/2406.07496 and https://github.com/zou-group/textgrad
- GEPA: https://arxiv.org/abs/2507.19457 and https://github.com/gepa-ai/gepa
- Feedback Descent: https://arxiv.org/abs/2511.07919 and https://github.com/JaredJoss/feedback-descent
- FunSearch: https://www.nature.com/articles/s41586-023-06924-6 and https://github.com/google-deepmind/funsearch
- AlphaEvolve: https://arxiv.org/abs/2506.13131
- OpenEvolve: https://github.com/algorithmicsuperintelligence/openevolve
- EoH: https://arxiv.org/abs/2401.02051 and https://github.com/FeiLiu36/EoH
- ReEvo: https://arxiv.org/html/2402.01145v1 and https://openreview.net/forum?id=483IPG0HWL
- Meta-Harness: https://arxiv.org/abs/2603.28052 and https://github.com/stanford-iris-lab/meta-harness
- optimize_anything: https://arxiv.org/html/2605.19633v1 and https://github.com/gepa-ai/optimize-anything-artifact
- BLADE: https://arxiv.org/abs/2504.20183 and https://github.com/XAI-liacs/BLADE
- CO-Bench: https://arxiv.org/html/2504.04310v1 and https://github.com/sunnweiwei/CO-Bench
- HeuriGym: https://arxiv.org/abs/2506.07972 and https://github.com/cornell-zhang/heurigym
- ALE-Bench: https://arxiv.org/abs/2506.09050 and https://github.com/SakanaAI/ALE-Bench
- AutoPrompt: https://aclanthology.org/2020.emnlp-main.346/ and https://github.com/ucinlp/autoprompt
- InstructZero: https://proceedings.mlr.press/v235/chen24e.html and https://github.com/lichang-chen/instructzero
- CAPO: https://arxiv.org/abs/2504.16005 and https://github.com/finitearth/capo
- Promptomatix: https://arxiv.org/abs/2507.14241 and https://github.com/salesforceairesearch/promptomatix
- MASPO: https://arxiv.org/abs/2605.06623 and https://github.com/wangzx1219/MASPO
- SPEAR: https://arxiv.org/abs/2605.26275
- FAPO: https://arxiv.org/abs/2606.19605 and https://github.com/cisco-foundation-ai/fully-automated-prompt-optimization
- LLaMEA: https://arxiv.org/abs/2405.20132 and https://github.com/XAI-liacs/LLaMEA
- LLM4AD: https://arxiv.org/abs/2412.17287 and https://github.com/Optima-CityU/llm4ad
- AIDE: https://arxiv.org/abs/2502.13138 and https://github.com/WecoAI/aideml
- MLE-bench: https://arxiv.org/abs/2410.07095 and https://github.com/openai/mle-bench
- EvalPerf: https://arxiv.org/abs/2408.06450 and https://github.com/evalplus/evalplus
- SWE-Perf: https://arxiv.org/abs/2507.12415 and https://github.com/SWE-Perf/SWE-Perf
- EffiBench-X: https://arxiv.org/abs/2505.13004 and https://github.com/EffiBench/EffiBench-X
- PerfForge: https://arxiv.org/abs/2505.23471 and https://github.com/UChiSeclab/perfforge
- PROMST: https://aclanthology.org/2024.emnlp-main.226/ and https://github.com/yongchao98/PROMST
- promptolution: https://arxiv.org/html/2512.02840v2 and https://github.com/automl/promptolution
