# Auto-extracted prose/content from the previous blogpost (2026-07-12).
# Edited by hand where figure semantics changed. HTML fragments.

HEADER_HTML = "<p class=\"eyebrow\">text-opt-bm</p>\n<h1>Benchmarking LLMs on text optimization</h1>\n<p class=\"sub\">text-opt-bm gives an LLM agent a weak Python program and a scoring function.\nThe agent repeatedly edits the program, and the benchmark records the best valid score it finds.</p>\n<ul class=\"fam-ul\">\n<li><span class=\"tag\">scores</span> Metrics include traced memory, compressed bytes, executed bytecode, recompute cost, and error rate.</li>\n<li><span class=\"tag\">perfect information</span> Eight tasks are scored on the same workload the agent optimizes.</li>\n<li><span class=\"tag\">generalization</span> Five tasks expose a training set and keep a larger sealed test from the same distribution.</li>\n<li><span class=\"tag\">harder tasks</span> Three ML-systems tasks add routing, optimizer transfer, and physically constrained model compression with validation-guided selection and sealed tests.</li>\n<li><span class=\"tag\">plots</span> Curves step at accepted submissions. Select any task card for its objective, metric, and exact data split.</li>\n</ul>"
FOOTER_HTML = "<b>Experiment scope.</b> Experiments 1a–3 use gpt-5.5 at high, low, and no reasoning effort plus grok-4.5 xhigh where shown. Experiment 4 uses gpt-5.6-sol with high reasoning on routing-v6, optimizer-v8, and LFM-compression-v2. Every plotted condition has five independent trials and a one-hour optimization budget. Aggregate bands show one standard deviation over independent run choices. Separate saturation probes are excluded. text-opt-bm is a work in progress."

SECTIONS = {
 "experiment-1a": {
  "sect_n": "Experiment 1a",
  "h2": "Reasoning effort and model sweep",
  "fam_html": "<li><span class=\"tag\">tasks</span> The eight perfect-information tasks.</li>\n<li><span class=\"tag\">note</span> checkpoint_plan has already reached its known offline optimum, so it functions as a solved control.</li>\n<li><span class=\"tag\">settings</span> gpt-5.5 with high, low, and no reasoning effort; grok-4.5 with xhigh reasoning through Cursor.</li>\n<li><span class=\"tag\">runs</span> Five one-hour runs per task and setting.</li>\n<li><span class=\"tag\">scope</span> Separate gpt-5.5 xhigh saturation probes are excluded from these plots.</li>\n<li><span class=\"tag\">figure</span> Scores are normalized per task: 1 is the seed program, 0 is the best score found within the plotted runs. Bands show one standard deviation of the benchmark aggregate over independent per-task run choices.</li>\n<li><span class=\"tag\">time axis</span> Elapsed wall-clock time from 0 to 60 minutes.</li>",
  "hds": [
   "Eight perfect-information tasks, averaged by model and reasoning setting",
   "Every perfect-information task, 20 runs each (5 per setting)"
  ]
 },
 "experiment-1b": {
  "sect_n": "Experiment 1b",
  "h2": "Training feedback and sealed test",
  "fam_html": "<li><span class=\"tag\">tasks</span> The five generalization tasks.</li>\n<li><span class=\"tag\">grading</span> During optimization, the visible training set supplies the score.</li>\n<li><span class=\"tag\">test</span> The same selected programs are evaluated afterward on a sealed test.</li>\n<li><span class=\"tag\">settings</span> gpt-5.5 high, low, and none; grok-4.5 xhigh through Cursor.</li>\n<li><span class=\"tag\">scope</span> Grok compress_heldout r4 and r5 are excluded here because the current scorer accepted impossible nonpositive byte scores.</li>\n<li><span class=\"tag\">figure</span> Left panel: train error. Right panel: sealed test error. Aggregate curves are normalized per task and averaged across the five tasks.</li>\n<li><span class=\"tag\">runs</span> Five independent one-hour runs per task and model setting.</li>\n<li><span class=\"tag\">time axis</span> Elapsed wall-clock time from 0 to 60 minutes.</li>",
  "hds": [
   "Five generalization tasks, train set versus sealed test",
   "Per task, train (graded) and sealed test, raw scores"
  ]
 },
 "experiment-2": {
  "sect_n": "Experiment 2",
  "h2": "Visible training feedback versus hidden validation",
  "fam_html": "<li><span class=\"tag\">tasks</span> The same five generalization tasks as Experiment 1b.</li>\n<li><span class=\"tag\">visible</span> The graded set is the full visible training set.</li>\n<li><span class=\"tag\">hidden</span> The agent sees five examples and receives only an aggregate validation score on unseen data.</li>\n<li><span class=\"tag\">figure</span> Both conditions use the same sealed test. The three panels show train score, validation score, and sealed test score.</li>\n<li><span class=\"tag\">runs</span> Five independent one-hour runs per task and feedback condition.</li>\n<li><span class=\"tag\">time axis</span> Elapsed wall-clock time from 0 to 60 minutes.</li>",
  "hds": [
   "Five generalization tasks, train, validation, and sealed test",
   "Per task, hidden validation (graded) and sealed test (raw error)"
  ]
 },
 "experiment-3": {
  "sect_n": "Experiment 3",
  "h2": "Training-set size sweep",
  "fam_html": "<li><span class=\"tag\">tasks</span> The five generalization tasks with fixed distributions and fixed sealed tests.</li>\n<li><span class=\"tag\">sizes</span> Visible train-to-test ratios are <b style=\"color:#0d9488\">1:4</b>, <b style=\"color:#7c3aed\">1:8</b>, and <b style=\"color:#ea580c\">1:16</b>.</li>\n<li><span class=\"tag\">figure</span> Left panel: train error. Right panel: sealed test error for the submitted programs.</li>\n<li><span class=\"tag\">runs</span> Five independent one-hour runs per task and training-set size.</li>\n<li><span class=\"tag\">time axis</span> Elapsed wall-clock time from 0 to 60 minutes.</li>",
  "hds": [
   "Five generalization tasks, train-set size sweep",
   "Per task, train (graded) and sealed test, one line per training-set size (raw error)"
  ]
 },
 "harder-tasks": {
  "sect_n": "Experiment 4 · harder tasks",
  "h2": "Research-oriented ML systems optimization",
  "fam_html": "<li><span class=\"tag\">selection</span> Visible training or calibration data may be used to build a method, while acceptance is validation-guided. Sealed tests cannot affect acceptance or later agent prompts.</li>\n<li><span class=\"tag\">budget</span> Runs receive one hour of active optimization time. Evaluation-capacity intervals are excluded from charged time, and final sealed-test work drains after optimization.</li>\n<li><span class=\"tag\">reporting</span> Lower is better for all three primary metrics. Test results belong to validation-selected programs, not test-selected checkpoints.</li>\n<li><span class=\"tag\">agent</span> gpt-5.6-sol with high reasoning; five independent trials per task.</li>\n<li><span class=\"tag\">time axis</span> Active optimization time from 0 to 60 minutes; evaluator queueing is refunded.</li>",
  "hds": []
 }
}

BACKS = {
 "experiment-1a": {
  "mem_kv": {
   "tag": "memory",
   "html": "<ul class=\"bl\"><li>Store 150k string key-value pairs; answer 40k exact-match lookups.</li><li>Peak is measured while serving. The program must stay compact at rest and avoid large per-query decompression.</li><li>Scored by peak traced bytes while serving lookups (lower is better).</li></ul><div class=\"bh\">select to return to chart</div>"
  },
  "mem_index": {
   "tag": "memory",
   "html": "<ul class=\"bl\"><li>Index ~20,000 short documents so each query returns the sorted ids of docs containing a word.</li><li>Peak is sampled during serving. Retained input strings and per-query decompression both count.</li><li>Scored by peak traced bytes while serving queries (lower is better).</li></ul><div class=\"bh\">select to return to chart</div>"
  },
  "mem_intset": {
   "tag": "memory",
   "html": "<ul class=\"bl\"><li>Index ~150,000 distinct integers (universe ~5,000,000) to answer exact membership queries.</li><li>Peak resets after build and then charges per-query transients. The target is compact, low-allocation serving.</li><li>Scored by peak traced bytes while serving queries (lower is better).</li></ul><div class=\"bh\">select to return to chart</div>"
  },
  "mem_str": {
   "tag": "memory",
   "html": "<ul class=\"bl\"><li>Store ~100,000 heavily-duplicated, prefix-sharing strings; return any one exactly by its index.</li><li>Peak counts each get()&#x27;s transients, so dedupe and share prefixes yet rebuild one string cheaply.</li><li>Scored by peak traced bytes while serving retrievals (lower is better).</li></ul><div class=\"bh\">select to return to chart</div>"
  },
  "mem_infer": {
   "tag": "memory",
   "html": "<ul class=\"bl\"><li>Given a tiny GPT&#x27;s weights, a prompt, and n, greedily decode the next n token ids.</li><li>Peak includes import-time allocations. The target is buffer reuse with minimal copied state.</li><li>Scored by peak traced bytes across three decode runs (lower is better).</li></ul><div class=\"bh\">select to return to chart</div>"
  },
  "compress": {
   "tag": "compression",
   "html": "<ul class=\"bl\"><li>Losslessly compress four redundant files: logs, JSON, prose, and CSV. The decoder must reconstruct each byte exactly.</li><li>Standard codecs are banned. The compressor must use task-specific structure and pure Python code.</li><li>Scored by total compressed bytes across four documents (lower is better).</li></ul><div class=\"bh\">select to return to chart</div>"
  },
  "ops_connect": {
   "tag": "algorithms",
   "html": "<ul class=\"bl\"><li>Answer a stream of union and connectivity queries on an undirected graph, one boolean each.</li><li>The metric counts executed bytecode. Efficient submissions use an incremental connectivity structure.</li><li>Scored by Python bytecode instructions executed (lower is better).</li></ul><div class=\"bh\">select to return to chart</div>"
  },
  "checkpoint_plan": {
   "tag": "algorithms",
   "html": "<ul class=\"bl\"><li>Given per-layer memory/compute costs and a budget, output checkpoint indices minimizing backward recompute.</li><li>Peak memory is the checkpoint sum plus only the largest segment. The task is to choose feasible checkpoint indices under that constraint.</li><li>Scored by total recompute forward-cost across profiles (lower is better).</li></ul><div class=\"bh\">select to return to chart</div>"
  }
 },
 "experiment-1b": {
  "word_problems": {
   "tag": "generalization",
   "html": "<ul class=\"bl\"><li>Parse a grade-school math word problem and output its exact numeric answer.</li><li>The sealed test uses broader wording than the training set. The program must parse quantities, word-numbers, and pronouns.</li><li>Scored by error rate (lower is better).</li><li><b>Dataset sizes</b>: train 500 (visible, graded); validation none; sealed test 2000.</li><li>The train set is graded during optimization. The sealed sealed test is evaluated afterward.</li></ul><div class=\"bh\">select to return to chart</div>"
  },
  "normalize": {
   "tag": "generalization",
   "html": "<ul class=\"bl\"><li>Parse a messy free-form duration string into total whole seconds.</li><li>The training set emphasizes common formats. The sealed test includes rarer duration formats.</li><li>Scored by exact-match error rate (lower is better).</li><li><b>Dataset sizes</b>: train 500 (visible, graded); validation none; sealed test 2000.</li><li>The train set is graded during optimization. The sealed sealed test is evaluated afterward.</li></ul><div class=\"bh\">select to return to chart</div>"
  },
  "rule_list": {
   "tag": "generalization",
   "html": "<ul class=\"bl\"><li>Learn from labeled 8-feature rows; classify unseen rows into four classes.</li><li>Labels depend on within-row rank and comparison structure rather than raw magnitudes.</li><li>Scored by error rate (lower is better).</li><li><b>Dataset sizes</b>: train 1200 (visible, graded); validation none; sealed test 4800.</li><li>The train set is graded during optimization. The sealed sealed test is evaluated afterward.</li></ul><div class=\"bh\">select to return to chart</div>"
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
  "word_problems": {
   "tag": "generalization",
   "html": "<ul class=\"bl\"><li>Parse a grade-school math word problem and output its exact numeric answer.</li><li>The sealed test uses broader wording than the training set. The program must parse quantities, word-numbers, and pronouns.</li><li>Scored by error rate (lower is better).</li><li><b>Dataset sizes</b>: train 5 (visible); validation 495 (hidden, graded); sealed test 2000.</li><li><b style=\"color:#ea580c\">visible</b>: graded on the fully visible train set. <b style=\"color:#0d9488\">hidden</b>: graded on aggregate validation score with only 5 examples visible.</li></ul><div class=\"bh\">select to return to chart</div>"
  },
  "normalize": {
   "tag": "generalization",
   "html": "<ul class=\"bl\"><li>Parse a messy free-form duration string into total whole seconds.</li><li>The training set emphasizes common formats. The sealed test includes rarer duration formats.</li><li>Scored by exact-match error rate (lower is better).</li><li><b>Dataset sizes</b>: train 5 (visible); validation 495 (hidden, graded); sealed test 2000.</li><li><b style=\"color:#ea580c\">visible</b>: graded on the fully visible train set. <b style=\"color:#0d9488\">hidden</b>: graded on aggregate validation score with only 5 examples visible.</li></ul><div class=\"bh\">select to return to chart</div>"
  },
  "rule_list": {
   "tag": "generalization",
   "html": "<ul class=\"bl\"><li>Learn from labeled 8-feature rows; classify unseen rows into four classes.</li><li>Labels depend on within-row rank and comparison structure rather than raw magnitudes.</li><li>Scored by error rate (lower is better).</li><li><b>Dataset sizes</b>: train 5 (visible); validation 1195 (hidden, graded); sealed test 4800.</li><li><b style=\"color:#ea580c\">visible</b>: graded on the fully visible train set. <b style=\"color:#0d9488\">hidden</b>: graded on aggregate validation score with only 5 examples visible.</li></ul><div class=\"bh\">select to return to chart</div>"
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
  "word_problems": {
   "tag": "generalization",
   "html": "<ul class=\"bl\"><li>Parse a grade-school math word problem and output its exact numeric answer.</li><li>The sealed test uses broader wording than the training set. The program must parse quantities, word-numbers, and pronouns.</li><li>Scored by error rate (lower is better).</li><li><b>Dataset sizes</b>: train 500 / 250 / 125 (1:4 / 1:8 / 1:16); validation none; sealed test 2000.</li><li>Same task and sealed test; only the visible training-set size shrinks (1:4 to 1:8 to 1:16 of the test).</li></ul><div class=\"bh\">select to return to chart</div>"
  },
  "normalize": {
   "tag": "generalization",
   "html": "<ul class=\"bl\"><li>Parse a messy free-form duration string into total whole seconds.</li><li>The training set emphasizes common formats. The sealed test includes rarer duration formats.</li><li>Scored by exact-match error rate (lower is better).</li><li><b>Dataset sizes</b>: train 500 / 250 / 125 (1:4 / 1:8 / 1:16); validation none; sealed test 2000.</li><li>Same task and sealed test; only the visible training-set size shrinks (1:4 to 1:8 to 1:16 of the test).</li></ul><div class=\"bh\">select to return to chart</div>"
  },
  "rule_list": {
   "tag": "generalization",
   "html": "<ul class=\"bl\"><li>Learn from labeled 8-feature rows; classify unseen rows into four classes.</li><li>Labels depend on within-row rank and comparison structure rather than raw magnitudes.</li><li>Scored by error rate (lower is better).</li><li><b>Dataset sizes</b>: train 1200 / 600 / 300 (1:4 / 1:8 / 1:16); validation none; sealed test 4800.</li><li>Same task and sealed test; only the visible training-set size shrinks (1:4 to 1:8 to 1:16 of the test).</li></ul><div class=\"bh\">select to return to chart</div>"
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
  "llm_routing_v2": {
   "tag": "routing v6",
   "html": "<ul class=\"bl\"><li>Learn a cost-aware router over 11 pinned models and 21 cost preferences from precomputed LLMRouterBench outcomes.</li><li>Scored by dataset-macro normalized utility regret across ten source datasets.</li><li><b>Dataset sizes</b>: fit 5,801; visible train 1,235; hidden validation 2,500; sealed test 2,354 prompts.</li><li>The test plot evaluates submitted programs on sealed prompts; it cannot guide acceptance or subsequent agent reasoning.</li></ul><div class=\"bh\">select to return to chart</div>"
  },
  "optimizer_generalization_v2": {
   "tag": "optimizer v8",
   "html": "<ul class=\"bl\"><li>Synthesize a deterministic first-order optimizer for real image and text neural workloads.</li><li>Scored by normalized validation-loss curve area at 17 checkpoints, macro-averaged over family and track cells.</li><li><b>Workload sizes</b>: train 120; hidden validation 320; sealed test 656. The ranked test contains 48 ID and 48 OOD neural workloads.</li><li>ID preserves development families and architectures; OOD introduces held-out data, widths, horizons, and a residual MLP architecture.</li></ul><div class=\"bh\">select to return to chart</div>"
  },
  "slm_weight_compression_lfm25": {
   "tag": "compression v2",
   "html": "<ul class=\"bl\"><li>Submit an expressive QWeight bundle for LiquidAI/LFM2.5-230M under a measured 3.5-bit-per-original-parameter storage ceiling.</li><li>Graded by conversation-mean <code>max(NLL compressed − NLL native, 0)</code> on assistant tokens.</li><li><b>Data sizes</b>: 128 unscored calibration, 128 ID validation, 128 ID test, and 128 OOD test conversations; 187,803 tokens total.</li><li>Calibration is not a scored train split, so the development panel is correctly shown as ID validation. Test-ID and test-OOD remain separately visible.</li></ul><div class=\"bh\">select to return to chart</div>"
  }
 }
}
