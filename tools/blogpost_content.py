# Auto-extracted prose/content from the previous blogpost (2026-07-12).
# Edited by hand where figure semantics changed. HTML fragments.

HEADER_HTML = "<p class=\"eyebrow\">text-opt-bm</p>\n<h1>Benchmarking LLMs on text optimization</h1>\n<p class=\"sub\">text-opt-bm gives an LLM agent a weak Python program and a scoring function.\nThe agent repeatedly edits the program, and the benchmark records the best valid score it finds.</p>\n<ul class=\"fam-ul\">\n<li><span class=\"tag\">scores</span> Metrics include traced memory, compressed bytes, executed bytecode, and error rate.</li>\n<li><span class=\"tag\">perfect information</span> Four tasks are scored on the same workload the agent optimizes.</li>\n<li><span class=\"tag\">generalization</span> Six tasks optimize on non-final feedback and are judged on a sealed test. Three use visible training data; three apply the same principle to LLM routing, optimizer transfer, and SLM compression. Historical plots below predate the merger of the easy and hard word-problem protocols.</li>\n<li><span class=\"tag\">plots</span> Curves step at accepted submissions. Select any task card for protocol details.</li>\n</ul>"
FOOTER_HTML = ""

SECTIONS = {
 "experiment-1": {
  "sect_n": "Experiment 1",
  "h2": "The current ten-task benchmark",
  "fam_html": "<li><span class=\"tag\">tasks</span> Ten retained tasks: four perfect-information systems problems and six generalization problems.</li>\n<li><span class=\"tag\">models</span> Five independent one-hour trials per complete model/task series. A line is omitted unless all five trials are complete.</li>\n<li><span class=\"tag\">feedback</span> Perfect-information tasks optimize their final metric directly. Generalization tasks optimize reusable feedback and are evaluated on a sealed test.</li>\n<li><span class=\"tag\">figure</span> The overview normalizes each task to 1 at the starter and 0 at the best score found across complete plotted series. Task cards retain raw units.</li>\n<li><span class=\"tag\">status</span> The ten-task gpt-5.6-sol high campaign is complete. The gpt-5.5 ten-task mean stays hidden until every one of its ten N=5 task series is complete.</li>\n<li><span class=\"tag\">time axis</span> Optimizer-active time from 0 to 60 minutes; evaluator-queue waits and campaign pauses are not charged.</li>",
  "hds": []
 },
 "experiment-1a": {
  "sect_n": "Experiment 2 · historical",
  "h2": "Historical reasoning-effort and model sweep",
  "fam_html": "<li><span class=\"tag\">tasks</span> The four then-retained perfect-information task protocols.</li>\n<li><span class=\"tag\">settings</span> gpt-5.5 with high, low, and no reasoning effort; grok-4.5 with xhigh reasoning through Cursor.</li>\n<li><span class=\"tag\">runs</span> Five one-hour runs per task and setting.</li>\n<li><span class=\"tag\">scope</span> Separate gpt-5.5 xhigh saturation probes are excluded from these plots.</li>\n<li><span class=\"tag\">figure</span> Scores are normalized per task: 1 is the seed program, 0 is the best score found within the plotted runs. Bands show one standard deviation of the benchmark aggregate over independent per-task run choices.</li>\n<li><span class=\"tag\">time axis</span> Optimizer-active time from 0 to 60 minutes.</li>",
  "hds": [
   "Four perfect-information tasks, averaged by model and reasoning setting",
   "Every retained perfect-information task, 20 runs each (5 per setting)"
  ]
 },
 "experiment-1b": {
  "sect_n": "Experiment 3 · historical",
  "h2": "Historical training feedback and sealed test",
  "fam_html": "<li><span class=\"tag\">tasks</span> The three original, campaign-tested generalization tasks.</li>\n<li><span class=\"tag\">grading</span> During optimization, the visible training set supplies the score.</li>\n<li><span class=\"tag\">test</span> The same selected programs are evaluated afterward on a sealed test.</li>\n<li><span class=\"tag\">settings</span> gpt-5.5 high, low, and none; grok-4.5 xhigh through Cursor.</li>\n<li><span class=\"tag\">scope</span> compress_heldout curves use the repaired plain-buffer scorer and a complete offline rescore of every featured submission.</li>\n<li><span class=\"tag\">figure</span> Left panel: train error. Right panel: sealed test error. Aggregate curves are normalized per task and averaged across the three tasks.</li>\n<li><span class=\"tag\">runs</span> Five independent one-hour runs per task and model setting.</li>\n<li><span class=\"tag\">time axis</span> Optimizer-active time from 0 to 60 minutes.</li>",
  "hds": [
   "Three original generalization tasks, train set versus sealed test",
   "Per task, train (graded) and sealed test, raw scores"
  ]
 },
 "experiment-2": {
  "sect_n": "Experiment 4 · historical",
  "h2": "Historical visible training feedback versus hidden validation",
  "fam_html": "<li><span class=\"tag\">tasks</span> The same three generalization tasks as the training-feedback comparison.</li>\n<li><span class=\"tag\">visible</span> The graded set is the full visible training set.</li>\n<li><span class=\"tag\">hidden</span> The agent sees five examples and receives only an aggregate validation score on unseen data.</li>\n<li><span class=\"tag\">figure</span> Both conditions use the same sealed test. The three panels show train score, validation score, and sealed test score.</li>\n<li><span class=\"tag\">runs</span> Five independent one-hour runs per task and feedback condition.</li>\n<li><span class=\"tag\">time axis</span> Optimizer-active time from 0 to 60 minutes.</li>",
  "hds": [
   "Three original generalization tasks, train, validation, and sealed test",
   "Per task, hidden validation (graded) and sealed test (raw error)"
  ]
 },
 "experiment-3": {
  "sect_n": "Experiment 5 · historical",
  "h2": "Historical training-set size sweep",
  "fam_html": "<li><span class=\"tag\">tasks</span> The three original generalization tasks with fixed distributions and fixed sealed tests.</li>\n<li><span class=\"tag\">sizes</span> Visible train-to-test ratios are <b style=\"color:#0d9488\">1:4</b>, <b style=\"color:#7c3aed\">1:8</b>, and <b style=\"color:#ea580c\">1:16</b>.</li>\n<li><span class=\"tag\">figure</span> Left panel: train error. Right panel: sealed test error for the submitted programs.</li>\n<li><span class=\"tag\">runs</span> Five independent one-hour runs per task and training-set size.</li>\n<li><span class=\"tag\">time axis</span> Optimizer-active time from 0 to 60 minutes.</li>",
  "hds": [
   "Three original generalization tasks, train-set size sweep",
   "Per task, train (graded) and sealed test, one line per training-set size (raw error)"
  ]
 },
 "harder-tasks": {
  "sect_n": "",
  "h2": "Archived ML-systems reference study",
  "fam_html": "<li><span class=\"tag\">study</span> Fixed compression-method checks run while validating the revised behavioral LFM2.5 protocol.</li>\n<li><span class=\"tag\">scope</span> These are aggregate method results, not one-hour agent trajectories; the completed agent campaign is reported in Experiment 1.</li>\n<li><span class=\"tag\">metric</span> Macro-average BF16 behavioral regression across GPQA, IFBench, and single-turn BFCL; lower is better.</li>",
  "hds": []
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
   "html": "<ul class=\"bl\"><li>Train on token/tag sequences, then tag every token of unseen sequences with a letter A to F.</li><li>Random stems do not repeat across splits. The target includes context rules such as last-match-wins behavior.</li><li>Scored by per-token tag error rate (lower is better).</li><li><b>Dataset sizes</b>: train 500 seq (visible, graded); validation none; sealed test 2000 seq.</li><li>The train set is graded during optimization. The sealed sealed test is evaluated afterward.</li></ul><div class=\"bh\">select to return to chart</div>"
  },
  "compress_heldout": {
   "tag": "generalization",
   "html": "<ul class=\"bl\"><li>Pure-Python lossless compression (no codec libraries) shrinking text documents to the fewest bytes.</li><li>Scoring uses hidden documents. File-specific dictionaries from the visible set are not sufficient.</li><li>Scored by compressed bytes on hidden documents (lower is better).</li><li><b>Dataset sizes</b>: train 4 docs ~50 KB (visible, graded); validation none; sealed test 4 docs ~200 KB.</li><li>The train set is graded during optimization. The sealed sealed test is evaluated afterward.</li></ul><div class=\"bh\">select to return to chart</div>"
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
   "html": "<ul class=\"bl\"><li>Pure-Python lossless compression (no codec libraries) shrinking text documents to the fewest bytes.</li><li>Scoring uses hidden documents. File-specific dictionaries from the visible set are not sufficient.</li><li>Scored by compressed bytes on hidden documents (lower is better).</li><li><b>Dataset sizes</b>: train 4 docs ~5 KB (visible); validation 4 docs ~50 KB (hidden, graded); sealed test 4 docs ~200 KB.</li><li><b style=\"color:#ea580c\">visible</b>: graded on the fully visible train set. <b style=\"color:#0d9488\">hidden</b>: graded on aggregate validation score with only 5 examples visible.</li></ul><div class=\"bh\">select to return to chart</div>"
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
   "html": "<ul class=\"bl\"><li>Pure-Python lossless compression (no codec libraries) shrinking text documents to the fewest bytes.</li><li>Scoring uses hidden documents. File-specific dictionaries from the visible set are not sufficient.</li><li>Scored by compressed bytes on hidden documents (lower is better).</li><li><b>Dataset sizes</b>: train 4 docs ~50 / 25 / 12.5 KB (1:4 / 1:8 / 1:16); validation none; sealed test 4 docs ~200 KB.</li><li>Same task and sealed test; only the visible training-set size shrinks (1:4 to 1:8 to 1:16 of the test).</li></ul><div class=\"bh\">select to return to chart</div>"
  }
 },
 "harder-tasks": {
  "llm_routing": {
   "tag": "routing v7",
   "html": "<ul class=\"bl\"><li>Learn a cost-aware router over 11 pinned models using precomputed LLMRouterBench outcomes.</li><li>Optimize dataset-macro normalized utility regret over 21 visible cost preferences; the sealed test uses 33 shifted preferences.</li><li>Validation is dataset-ID. The sealed test balances dataset-ID against held-out LiveCodeBench, SWE-Bench, and tau2 sources.</li><li>Paper-style AvgAcc, Gain@B, Gap@O, PerfGain, and CostSave diagnostics are retained alongside the primary score.</li><li>Avengers-Pro and centroid references are re-evaluated locally under this custom protocol; this is not a direct numerical reproduction of the paper.</li></ul><div class=\"bh\">select to return to chart</div>"
  },
  "optimizer_generalization": {
   "tag": "optimizer v9",
   "html": "<ul class=\"bl\"><li>Synthesize a deterministic first-order optimizer for real image and text neural workloads; analytic functions are unranked diagnostics.</li><li>Scored by TaskSet-style empirical-best-normalized validation-loss curve AUC, macro-balanced across workload cells.</li><li>Five visible architecture families support development; the sealed set adds three unseen families and separately reports ID/OOD.</li><li>Loss and gradient computation is CPU JAX-JIT accelerated, while submitted optimizers may use NumPy, JAX, or plain Python.</li><li>Adam, RMSProp, Schedule-Free AdamW, NAdamW, and block/diagonal Shampoo were locally tuned and evaluated; shape-conditional variants match the task's legal topology dispatch.</li></ul><div class=\"bh\">select to return to chart</div>"
  },
  "slm_weight_compression_lfm25": {
   "tag": "behavioral compression v4",
   "html": "<ul class=\"bl\"><li>Submit an expressive QWeight bundle for LiquidAI/LFM2.5-230M under one measured 3.5-bit-per-original-parameter storage ceiling.</li><li>Graded by BF16 behavioral regression on capabilities the native checkpoint passes: GPQA choice likelihood, IFBench instruction following, and BFCL single-turn tool calls.</li><li><b>Data sizes</b>: 128 unscored calibration conversations; 20 examples from each behavior family in online validation and a disjoint 20 per family in sealed test.</li><li>Generation is greedy, requires EOS, and uses a BF16-response-relative token cap. All weights are decoded by the trusted QWeight implementation and inferred in BF16 on MPS with fallback disabled.</li><li>This aggregate method study is protocol context, not an agent optimization campaign; the completed agent results appear in Experiment 1.</li></ul><div class=\"bh\">select to return to chart</div>"
  }
 }
}

# The current split deliberately reuses the stable task descriptions above,
# while overriding protocols that changed after the historical campaigns.
BACKS["experiment-1"] = {
 "mem_index": BACKS["experiment-1a"]["mem_index"],
 "mem_str": BACKS["experiment-1a"]["mem_str"],
 "ops_connect": BACKS["experiment-1a"]["ops_connect"],
 "tag_seq": BACKS["experiment-1b"]["tag_seq"],
 "compress_heldout": BACKS["experiment-1b"]["compress_heldout"],
 "llm_routing": BACKS["harder-tasks"]["llm_routing"],
 "optimizer_generalization": BACKS["harder-tasks"]["optimizer_generalization"],
 "mem_infer": {
  "tag": "ML systems · memory",
  "html": "<ul class=\"bl\"><li>Run batch-one CPU-Torch decoding for a compact Qwen3.5-inspired hybrid model with one gated DeltaNet block and one grouped-query-attention block.</li><li>Optimize deterministic logical tensor storage plus kernel scratch while staying under an 18M-work-unit ceiling per instance.</li><li>The surface includes buffer reuse, cache/state precision, quantization, recurrent tiling, attention blocking, and accuracy/work interactions.</li><li>All generated logits are checked against float32 inference; lower peak logical bytes is better.</li></ul><div class=\"bh\">select to return to chart</div>"
 },
 "word_problems": {
  "tag": "generalization",
  "html": "<ul class=\"bl\"><li>Solve synthetic arithmetic word problems with exact numeric answers using deterministic pure Python.</li><li>The current task combines the former easy and hard regimes and macro-averages their error rates 50/50.</li><li><b>Data sizes</b>: 1,100 visible graded questions (500 easy, 600 hard); 4,400 sealed questions (2,000 easy, 2,400 hard).</li><li>The hard regime adds multi-step inventory, rates, schedules, ratios, percentages, geometry, and production chains. Lower error is better.</li></ul><div class=\"bh\">select to return to chart</div>"
 },
 "slm_weight_compression_lfm25": {
  "tag": "ML systems · compression",
  "html": "<ul class=\"bl\"><li>Compress LiquidAI/LFM2.5-230M into a trusted QWeight bundle under a measured 3.5-bit-per-original-parameter ceiling.</li><li>Optimize BF16 behavioral regression across GPQA choice likelihood, IFBench instruction following, and BFCL single-turn tool calls.</li><li><b>Data sizes</b>: 128 unscored calibration conversations; 20 examples per behavior family online and a disjoint 20 per family in the sealed test.</li><li>Greedy MPS inference requires EOS and uses BF16-response-relative token caps. Lower macro regression rate is better.</li></ul><div class=\"bh\">select to return to chart</div>"
 }
}
