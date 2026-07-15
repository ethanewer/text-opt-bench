# Auto-extracted prose/content from the previous blogpost (2026-07-12).
# Edited by hand where figure semantics changed. HTML fragments.

HEADER_HTML = "<p class=\"eyebrow\">text-opt-bm</p>\n<h1>Benchmarking LLMs on text optimization</h1>\n<p class=\"sub\">text-opt-bm gives an LLM agent a weak Python program and a scoring function.\nThe agent repeatedly edits the program, and the benchmark records the best valid score it finds.</p>\n<ul class=\"fam-ul\">\n<li><span class=\"tag\">scores</span> Metrics include logical memory, compressed bytes, utility regret, and error rate.</li>\n<li><span class=\"tag\">perfect information</span> Two tasks are scored on the same workload the agent optimizes.</li>\n<li><span class=\"tag\">generalization</span> Six tasks optimize on non-final feedback and are judged on a sealed test: tagging, compression, LLM routing, optimizer transfer, and SLM compression at two storage budgets.</li>\n<li><span class=\"tag\">plots</span> Curves step at accepted submissions. Select any task card for protocol details.</li>\n</ul>"
OFFICIAL_HEADER_HTML = "<p class=\"eyebrow\">text-opt-bm · alpha</p>\n<h1>Official alpha task results</h1>\n<p class=\"sub\">Results for the four tasks designated as the official alpha set: LLM routing, optimizer generalization, and SLM compression at two storage budgets.</p>\n<ul class=\"fam-ul\">\n<li><span class=\"tag\">status</span> Every task on this page is official. Historical and legacy results are in <a href=\"blogpost-all.html\">the complete results post</a>.</li>\n<li><span class=\"tag\">generalization</span> Online optimization feedback is separated from sealed-test evaluation.</li>\n<li><span class=\"tag\">plots</span> Curves step at accepted submissions and use optimizer-active time.</li>\n</ul>"
FOOTER_HTML = ""

SECTIONS = {
 "experiment-1": {
  "sect_n": "Experiment 1",
  "h2": "Official and legacy task results",
  "fam_html": "<li><span class=\"tag\">scope</span> This complete post includes four official alpha tasks and four legacy tasks retained for historical comparison.</li>\n<li><span class=\"tag\">models</span> Five independent one-hour trials per complete model/task series. A line is omitted unless all five trials are complete.</li>\n<li><span class=\"tag\">feedback</span> Perfect-information tasks optimize their final metric directly. Generalization tasks optimize reusable feedback and are evaluated on a sealed test.</li>\n<li><span class=\"tag\">figure</span> The overview normalizes each task to 1 at the starter and 0 at the best score found across complete plotted series.</li>\n<li><span class=\"tag\">time axis</span> Optimizer-active time from 0 to 60 minutes; evaluator-queue waits and campaign pauses are not charged.</li>",
  "hds": []
 },
 "experiment-1a": {
  "sect_n": "Experiment 2 · historical",
  "h2": "Historical reasoning-effort and model sweep",
  "fam_html": "<li><span class=\"tag\">tasks</span> The two retained perfect-information task families under their historical protocols.</li>\n<li><span class=\"tag\">settings</span> gpt-5.5 with high, low, and no reasoning effort; grok-4.5 with xhigh reasoning through Cursor.</li>\n<li><span class=\"tag\">runs</span> Five one-hour runs per task and setting.</li>\n<li><span class=\"tag\">scope</span> Separate gpt-5.5 xhigh saturation probes are excluded from these plots.</li>\n<li><span class=\"tag\">figure</span> Scores are normalized per task: 1 is the seed program, 0 is the best score found within the plotted runs. Bands show ±1 standard deviation across five aggregates formed by pairing runs with the same trial index across tasks.</li>\n<li><span class=\"tag\">time axis</span> Optimizer-active time from 0 to 60 minutes.</li>",
  "hds": [
   "Two perfect-information tasks, averaged by model and reasoning setting",
   "Every retained perfect-information task, 20 runs each (5 per setting)"
  ]
 },
 "experiment-1b": {
  "sect_n": "Experiment 3 · historical",
  "h2": "Historical training feedback and sealed test",
  "fam_html": "<li><span class=\"tag\">tasks</span> The two retained original generalization tasks.</li>\n<li><span class=\"tag\">grading</span> During optimization, the visible training set supplies the score.</li>\n<li><span class=\"tag\">test</span> The same selected programs are evaluated afterward on a sealed test.</li>\n<li><span class=\"tag\">settings</span> gpt-5.5 high, low, and none; grok-4.5 xhigh through Cursor.</li>\n<li><span class=\"tag\">scope</span> compress_heldout curves use the repaired plain-buffer scorer and a complete offline rescore of every featured submission.</li>\n<li><span class=\"tag\">figure</span> Left panel: training score. Right panel: sealed-test score. Aggregate curves are normalized per task and averaged across the two tasks.</li>\n<li><span class=\"tag\">runs</span> Five independent one-hour runs per task and model setting.</li>\n<li><span class=\"tag\">time axis</span> Optimizer-active time from 0 to 60 minutes.</li>",
  "hds": [
   "Two retained original generalization tasks, train set versus sealed test",
   "Per task, train (graded) and sealed test, raw scores"
  ]
 },
 "experiment-2": {
  "sect_n": "Experiment 4 · historical",
  "h2": "Historical visible training feedback versus hidden validation",
  "fam_html": "<li><span class=\"tag\">tasks</span> The same two retained generalization tasks as the training-feedback comparison.</li>\n<li><span class=\"tag\">visible</span> The graded set is the full visible training set.</li>\n<li><span class=\"tag\">hidden</span> The agent sees 5 sequences for tagging or 4 documents for compression, and receives only an aggregate score on a separate hidden validation set.</li>\n<li><span class=\"tag\">figure</span> Both conditions use the same sealed test. The three panels show train score, validation score, and sealed test score.</li>\n<li><span class=\"tag\">runs</span> Five independent one-hour runs per task and feedback condition.</li>\n<li><span class=\"tag\">time axis</span> Optimizer-active time from 0 to 60 minutes.</li>",
  "hds": [
   "Two retained generalization tasks, train, validation, and sealed test",
   "Per task, hidden validation (graded) and sealed test (raw scores)"
  ]
 },
 "experiment-3": {
  "sect_n": "Experiment 5 · historical",
  "h2": "Historical training-set size sweep",
  "fam_html": "<li><span class=\"tag\">tasks</span> The two retained original generalization tasks with fixed distributions and fixed sealed tests.</li>\n<li><span class=\"tag\">sizes</span> Visible train-to-test ratios are <b style=\"color:#0d9488\">1:4</b>, <b style=\"color:#7c3aed\">1:8</b>, and <b style=\"color:#ea580c\">1:16</b>.</li>\n<li><span class=\"tag\">figure</span> Left panel: training score. Right panel: sealed-test score for the submitted programs.</li>\n<li><span class=\"tag\">runs</span> Five independent one-hour runs per task and training-set size.</li>\n<li><span class=\"tag\">time axis</span> Optimizer-active time from 0 to 60 minutes.</li>",
  "hds": [
   "Two retained original generalization tasks, train-set size sweep",
   "Per task, train (graded) and sealed test, one line per training-set size (raw scores)"
  ]
 }
}

BACKS = {
 "experiment-1a": {
  "mem_index": {
   "tag": "memory",
   "html": "<ul class=\"bl\"><li>Index ~20,000 short documents so each query returns the sorted ids of docs containing a word.</li><li>Peak is sampled during serving. Retained input strings and per-query decompression both count.</li><li>Scored by peak traced bytes while serving queries (lower is better).</li></ul><div class=\"bh\">select to return to chart</div>"
  },
  "mem_str": {
   "tag": "memory",
   "html": "<ul class=\"bl\"><li>Store ~100,000 heavily-duplicated, prefix-sharing strings; return any one exactly by its index.</li><li>Peak counts each get()&#x27;s transients, so dedupe and share prefixes yet rebuild one string cheaply.</li><li>Scored by peak traced bytes while serving retrievals (lower is better).</li></ul><div class=\"bh\">select to return to chart</div>"
  },
  "mem_infer": {
   "tag": "memory",
   "html": "<ul class=\"bl\"><li><b>Historical experiment:</b> these curves used the original tiny-GPT/tracemalloc task.</li><li>The current benchmark replaces it with a CPU-Torch hybrid decoder containing one gated DeltaNet layer and one grouped-query attention layer.</li><li>It minimizes logical state, KV-cache, activation, and kernel-scratch bytes under an 18M deterministic-work ceiling, while checking every generated vocabulary-logit vector against float32 inference.</li><li>Its scores are not comparable to this archived campaign.</li></ul><div class=\"bh\">select to return to chart</div>"
  },
  "ops_connect": {
   "tag": "algorithms",
   "html": "<ul class=\"bl\"><li>Answer a stream of union and connectivity queries on an undirected graph, one boolean each.</li><li>The metric counts executed bytecode. Efficient submissions use an incremental connectivity structure.</li><li>Scored by Python bytecode instructions executed (lower is better).</li></ul><div class=\"bh\">select to return to chart</div>"
  }
 },
 "experiment-1b": {
  "easy_word_problems": {
   "tag": "generalization",
   "html": "<ul class=\"bl\"><li><b>Historical easy-only protocol:</b> parse a grade-school math word problem and output its exact numeric answer.</li><li>These curves predate the current <code>word_problems</code> task, which pools this 500/2000 regime with the deeper 600/2400 hard regime under one score.</li><li>The sealed test uses broader wording than the training set. The program must parse quantities, word-numbers, and pronouns.</li><li>Scored by error rate (lower is better).</li><li><b>Archived dataset sizes</b>: train 500 (visible, graded); validation none; sealed test 2000.</li></ul><div class=\"bh\">select to return to chart</div>"
  },
  "tag_seq": {
   "tag": "generalization",
   "html": "<ul class=\"bl\"><li>Train on token/tag sequences, then tag every token of unseen sequences with a letter A to F.</li><li>Random stems do not repeat across splits. The target includes context rules such as last-match-wins behavior.</li><li>Scored by per-token tag error rate (lower is better).</li><li><b>Dataset sizes</b>: train 500 seq (visible, graded); validation none; sealed test 2000 seq.</li><li>The train set is graded during optimization. The sealed test is evaluated afterward.</li></ul><div class=\"bh\">select to return to chart</div>"
  },
  "compress_heldout": {
   "tag": "generalization",
   "html": "<ul class=\"bl\"><li>Pure-Python lossless compression (no codec libraries) shrinking text documents to the fewest bytes.</li><li>During optimization, the score is total compressed bytes across four visible training documents. The same program is later evaluated on four unseen documents; lower is better.</li><li><b>Dataset sizes</b>: train 4 documents, about 50 KB each (visible, graded); validation none; sealed test 4 documents, about 200 KB each.</li></ul><div class=\"bh\">select to return to chart</div>"
  }
 },
 "experiment-2": {
  "easy_word_problems": {
   "tag": "generalization",
   "html": "<ul class=\"bl\"><li><b>Historical easy-only protocol:</b> this restricted-information experiment predates the current combined <code>word_problems</code> task.</li><li>Parse a grade-school math word problem and output its exact numeric answer.</li><li><b>Archived dataset sizes</b>: train 5 (visible); validation 495 (hidden, graded); sealed test 2000.</li><li><b style=\"color:#ea580c\">visible</b>: graded on the fully visible train set. <b style=\"color:#0d9488\">hidden</b>: graded on aggregate validation score with only 5 examples visible.</li></ul><div class=\"bh\">select to return to chart</div>"
  },
  "tag_seq": {
   "tag": "generalization",
   "html": "<ul class=\"bl\"><li>Train on token/tag sequences, then tag every token of unseen sequences with a letter A to F.</li><li>Random stems do not repeat across splits. The target includes context rules such as last-match-wins behavior.</li><li>Scored by per-token tag error rate (lower is better).</li><li><b>Dataset sizes</b>: train 5 seq (visible); validation 495 seq (hidden, graded); sealed test 2000 seq.</li><li><b style=\"color:#ea580c\">visible</b>: graded on the fully visible train set. <b style=\"color:#0d9488\">hidden</b>: graded on aggregate validation score with only 5 examples visible.</li></ul><div class=\"bh\">select to return to chart</div>"
  },
  "compress_heldout": {
   "tag": "generalization",
   "html": "<ul class=\"bl\"><li>Pure-Python lossless compression (no codec libraries) shrinking text documents to the fewest bytes.</li><li><b>Dataset sizes</b>: train 4 documents, about 5 KB each; validation 4 documents, about 50 KB each; sealed test 4 documents, about 200 KB each.</li><li><b style=\"color:#ea580c\">visible</b>: graded on the four fully visible training documents. <b style=\"color:#0d9488\">hidden</b>: the visible documents are for development only; grading uses the aggregate score on four hidden validation documents.</li></ul><div class=\"bh\">select to return to chart</div>"
  }
 },
 "experiment-3": {
  "easy_word_problems": {
   "tag": "generalization",
   "html": "<ul class=\"bl\"><li><b>Historical easy-only protocol:</b> this size sweep predates the current combined <code>word_problems</code> task.</li><li>Parse a grade-school math word problem and output its exact numeric answer.</li><li><b>Archived dataset sizes</b>: train 500 / 250 / 125 (1:4 / 1:8 / 1:16); validation none; sealed test 2000.</li><li>Same task and sealed test; only the visible training-set size shrinks (1:4 to 1:8 to 1:16 of the test).</li></ul><div class=\"bh\">select to return to chart</div>"
  },
  "tag_seq": {
   "tag": "generalization",
   "html": "<ul class=\"bl\"><li>Train on token/tag sequences, then tag every token of unseen sequences with a letter A to F.</li><li>Random stems do not repeat across splits. The target includes context rules such as last-match-wins behavior.</li><li>Scored by per-token tag error rate (lower is better).</li><li><b>Dataset sizes</b>: train 500 / 250 / 125 seq (1:4 / 1:8 / 1:16); validation none; sealed test 2000 seq.</li><li>Same task and sealed test; only the visible training-set size shrinks (1:4 to 1:8 to 1:16 of the test).</li></ul><div class=\"bh\">select to return to chart</div>"
  },
  "compress_heldout": {
   "tag": "generalization",
   "html": "<ul class=\"bl\"><li>Pure-Python lossless compression (no codec libraries) shrinking text documents to the fewest bytes.</li><li>During optimization, the score is total compressed bytes across the visible training documents. The same program is later evaluated on unseen documents; lower is better.</li><li><b>Dataset sizes</b>: 4 training documents of about 50, 25, or 12.5 KB each (1:4 / 1:8 / 1:16); validation none; 4 sealed-test documents of about 200 KB each.</li></ul><div class=\"bh\">select to return to chart</div>"
  }
 }
}

# The current split reuses only unchanged historical task descriptions.
BACKS["experiment-1"] = {
 "mem_index": BACKS["experiment-1a"]["mem_index"],
 "tag_seq": BACKS["experiment-1b"]["tag_seq"],
 "compress_heldout": BACKS["experiment-1b"]["compress_heldout"],
 "llm_routing": {
  "tag": "routing v7",
  "html": "<ul class=\"bl\"><li>Build a cost-aware router over 11 pinned models. The task reuses recorded prompt-level quality and cost outcomes from <a href=\"https://arxiv.org/abs/2601.07206\" target=\"_blank\" rel=\"noreferrer\">LLMRouterBench</a>; it does not call models during scoring.</li><li>This is not a reproduction of an LLMRouterBench or <a href=\"https://arxiv.org/abs/2508.12631\" target=\"_blank\" rel=\"noreferrer\">Avengers-Pro</a> result. The model set, preprocessing, splits, cost settings, and ranked score are specific to text-opt-bm.</li><li>For each dataset and cost setting, the score measures utility lost relative to choosing the best recorded model for every prompt. It is normalized so 0 is that oracle and 1 is always choosing the worst model. The online score averages 21 cost settings and gives each development dataset equal weight.</li><li>The sealed test uses 33 different cost settings and weights familiar and test-only datasets 50/50. LiveCodeBench, SWE-Bench, and τ2 appear only in the test-only half.</li><li>The dashed references show the benchmark's global starter and two locally rerun paper-derived baselines: the 40-neighbor cosine KNN configuration from <a href=\"https://arxiv.org/abs/2403.12031\" target=\"_blank\" rel=\"noreferrer\">RouterBench</a>, and the 64-cluster Avengers-Pro configuration reproduced by LLMRouterBench. The KNN uses this task's pinned embeddings, and Avengers-Pro maps each cost setting to a performance coefficient using fit data only. The displayed values use this task's score and splits; they are not numbers copied from the papers.</li></ul><div class=\"bh\">select to return to chart</div>"
 },
 "optimizer_generalization": {
  "tag": "optimizer v9",
  "html": "<ul class=\"bl\"><li>Write one deterministic first-order optimizer as source code for small image and text neural networks. Five architecture families are used online; the sealed test adds three unseen families.</li><li>This is a custom screening task, not a reproduction of a published optimizer table. At 17 checkpoints, validation loss is compared with the best result from local SGD-and-Adam sweeps. This adapts the empirical-best normalization used by <a href=\"https://arxiv.org/abs/2002.11887\" target=\"_blank\" rel=\"noreferrer\">TaskSet</a>.</li><li>The normalized loss curve is integrated over training steps, and architecture/track groups receive equal weight. Lower is better. The sealed score gives 50% weight to familiar architectures and 50% to unseen architectures.</li><li>The workloads use MNIST, Fashion-MNIST, Shakespeare, and <i>Alice's Adventures in Wonderland</i>. The task is motivated by TaskSet and <a href=\"https://arxiv.org/abs/2211.09760\" target=\"_blank\" rel=\"noreferrer\">VeLO</a>, but general-purpose claims still need a larger suite such as <a href=\"https://arxiv.org/abs/2306.07179\" target=\"_blank\" rel=\"noreferrer\">AlgoPerf</a>.</li><li>Heavy-ball SGD, RMSProp, Adam, NAdamW, Schedule-Free AdamW, and Shampoo were tuned and evaluated locally. Their implementations follow the original <a href=\"https://arxiv.org/abs/1412.6980\" target=\"_blank\" rel=\"noreferrer\">Adam</a>, <a href=\"https://arxiv.org/abs/2405.15682\" target=\"_blank\" rel=\"noreferrer\">Schedule-Free</a>, and <a href=\"https://proceedings.mlr.press/v80/gupta18a.html\" target=\"_blank\" rel=\"noreferrer\">Shampoo</a> equations. Baseline results are local measurements; the charts show agent runs.</li></ul><div class=\"bh\">select to return to chart</div>"
 },
 "mem_infer": {
  "tag": "ML systems · memory",
  "html": "<ul class=\"bl\"><li>Run batch-one CPU-Torch decoding for a compact Qwen3.5-inspired hybrid model with one gated DeltaNet block and one grouped-query-attention block.</li><li>Optimize deterministic logical tensor storage plus kernel scratch while staying under an 18M-work-unit ceiling per instance.</li><li>The surface includes buffer reuse, cache/state precision, quantization, recurrent tiling, attention blocking, and accuracy/work interactions.</li><li>Every generated 96-value logit vector must stay within 0.035 absolute error of the float32 reference; lower peak logical bytes is better.</li></ul><div class=\"bh\">select to return to chart</div>"
 },
 "slm_compression_3_5bpw": {
  "tag": "ML systems · compression · 3.5 BPW",
  "html": "<ul class=\"bl\"><li>Compress a fixed revision of <a href=\"https://huggingface.co/LiquidAI/LFM2.5-230M\" target=\"_blank\" rel=\"noreferrer\">LiquidAI's LFM2.5-230M</a> model into the benchmark's QWeight file format. The complete file must use at most 3.5 bits per original model parameter (BPW). The starter uses round-to-nearest 3-bit weight quantization (RTN W3).</li><li>This is a custom task inspired by the small-model comparison in <a href=\"https://aclanthology.org/2025.findings-emnlp.645/\" target=\"_blank\" rel=\"noreferrer\">Zhou et al.</a> It uses a different model, format, data mixture, and score, so it does not reproduce that paper's results.</li><li>The data are subsets of <a href=\"https://arxiv.org/abs/2311.12022\" target=\"_blank\" rel=\"noreferrer\">GPQA Diamond</a>, <a href=\"https://arxiv.org/abs/2507.02833\" target=\"_blank\" rel=\"noreferrer\">IFBench</a>, <a href=\"https://gorilla.cs.berkeley.edu/leaderboard.html\" target=\"_blank\" rel=\"noreferrer\">BFCL v4</a>, <a href=\"https://arxiv.org/abs/2110.14168\" target=\"_blank\" rel=\"noreferrer\">GSM8K</a>, and <a href=\"https://arxiv.org/abs/2406.01574\" target=\"_blank\" rel=\"noreferrer\">MMLU-Pro</a>. GSM8K answers are converted to deterministic four-choice questions.</li><li>The uncompressed bfloat16 model (BF16) is the behavioral reference. The score is the mean rate of changed choices or tool calls, failed instructions, and failed or truncated generated answers. It is not benchmark accuracy; lower is better.</li><li><b>Data sizes:</b> 128 unscored calibration conversations, then 20 BF16-passing examples per source online and a disjoint 20 per source in the sealed test. The evaluator supports deterministic CUDA and Apple MPS execution. All plotted results were measured on CUDA; no cross-device equality is claimed. Dashed fixed-method scores are local, not paper results.</li></ul><div class=\"bh\">select to return to chart</div>"
 },
 "slm_compression_4_5bpw": {
  "tag": "ML systems · compression · 4.5 BPW",
  "html": "<ul class=\"bl\"><li>Compress a fixed revision of <a href=\"https://huggingface.co/LiquidAI/LFM2.5-230M\" target=\"_blank\" rel=\"noreferrer\">LiquidAI's LFM2.5-230M</a> model into the benchmark's QWeight file format. The complete file must use at most 4.5 bits per original model parameter (BPW). The starter uses round-to-nearest 4-bit weight quantization (RTN W4).</li><li>This is a custom task inspired by the small-model comparison in <a href=\"https://aclanthology.org/2025.findings-emnlp.645/\" target=\"_blank\" rel=\"noreferrer\">Zhou et al.</a> It uses a different model, format, data mixture, and score, so it does not reproduce that paper's results.</li><li>The data are subsets of <a href=\"https://arxiv.org/abs/2311.12022\" target=\"_blank\" rel=\"noreferrer\">GPQA Diamond</a>, <a href=\"https://arxiv.org/abs/2507.02833\" target=\"_blank\" rel=\"noreferrer\">IFBench</a>, <a href=\"https://gorilla.cs.berkeley.edu/leaderboard.html\" target=\"_blank\" rel=\"noreferrer\">BFCL v4</a>, <a href=\"https://arxiv.org/abs/2110.14168\" target=\"_blank\" rel=\"noreferrer\">GSM8K</a>, and <a href=\"https://arxiv.org/abs/2406.01574\" target=\"_blank\" rel=\"noreferrer\">MMLU-Pro</a>. GSM8K answers are converted to deterministic four-choice questions.</li><li>The uncompressed bfloat16 model (BF16) is the behavioral reference. The score is the mean rate of changed choices or tool calls, failed instructions, and failed or truncated generated answers. It is not benchmark accuracy; lower is better.</li><li><b>Data sizes:</b> 128 unscored calibration conversations, then 20 BF16-passing examples per source online and a disjoint 20 per source in the sealed test. The evaluator supports deterministic CUDA and Apple MPS execution. All plotted results were measured on CUDA; no cross-device equality is claimed. Dashed fixed-method scores are local, not paper results.</li></ul><div class=\"bh\">select to return to chart</div>"
 }
}
