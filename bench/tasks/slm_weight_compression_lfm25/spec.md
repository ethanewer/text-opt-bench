# Task: LFM2.5-230M behavioral-regression weight compression

This task uses the pinned `LiquidAI/LFM2.5-230M` checkpoint, calibration
corpus, trusted `qweight-1` submission format, Apple-MPS execution contract,
and a measured 3.5 whole-model bits-per-parameter ceiling.

The only objective change is scoring. The supplied 128 conversations remain
unscored quantization calibration. Online feedback is the macro-average
behavioral regression rate across 20 BF16-passing GPQA Diamond questions, 20
BF16-passing IFBench tasks, and 20 BF16-passing single-turn BFCLv4 calls. The
deferred test uses disjoint 20-example subsets of the same three benchmarks.
Lower is better.

GPQA regression means the four-way continuation-likelihood choice differs
from BF16. IFBench regression means the response fails the pinned loose
instruction verifier. BFCL regression means the parsed function name or
arguments differ from the accepted call. Generation is greedy and
deterministic. Each cap is
`min(original_limit, round_up_to_16(BF16_response_tokens) * 1.25)`, and a
response that reaches its cap without EOS is always a regression. This avoids
granting artificial passes to truncated prefixes.

The producer receives `--model`, `--calibration`, `--output`, `--targets`, and
`--device`, and writes a `3.500/` QWeight bundle. The complete bundle,
including metadata, must use at most 3.5 bits per base-model parameter.
