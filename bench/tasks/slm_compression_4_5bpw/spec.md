# Task: LFM2.5-230M behavioral-regression compression at 4.5 BPW

This task uses the pinned `LiquidAI/LFM2.5-230M` checkpoint, calibration
corpus, trusted `qweight-1` submission format, deterministic MPS/CUDA execution,
and a measured 4.5 whole-model bits-per-parameter ceiling.

The only objective change is scoring. The supplied 128 conversations remain
unscored quantization calibration. Online feedback is the macro-average
behavioral regression rate across 20 BF16-passing examples from each of GPQA
Diamond, IFBench, single-turn BFCLv4, a deterministic multiple-choice
derivation of GSM8K, and MMLU-Pro. The deferred test uses disjoint 20-example
subsets of the same five benchmarks. Lower is better.

GPQA regression means the four-way continuation-likelihood choice differs
from BF16. IFBench regression means the response fails the pinned loose
instruction verifier. BFCL regression means the parsed function name or
arguments differ from the accepted call. Generation is greedy and
deterministic. GSM8K uses its exact numeric answer and three deterministic,
distinct numeric distractors; GSM8K and MMLU-Pro are scored by one-token
answer-label likelihood. For generated responses, each cap is
`min(original_limit, round_up_to_16(BF16_response_tokens) * 1.25)`, and a
response that reaches its cap without EOS is always a regression. This avoids
granting artificial passes to truncated prefixes.

The producer receives `--model`, `--calibration`, `--output`, `--targets`, and
`--device`, and writes a `4.500/` QWeight bundle. The complete bundle,
including metadata, must use at most 4.5 bits per base-model parameter.
`auto` selects CUDA when available and otherwise MPS; campaigns pin an
explicit backend for reproducibility.
