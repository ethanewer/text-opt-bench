# Task: LFM2.5-230M constrained weight compression

Submit a deterministic Python weight producer for the pinned
`LiquidAI/LFM2.5-230M` checkpoint. It receives `--model`, `--calibration`,
`--output`, `--targets`, and `--device`, and writes a `3.500/` QWeight bundle.
The complete bundle, including metadata, must use at most 3.5 bits per base
model parameter. Any trusted `qweight-1` dense, affine, codebook, block-float,
or bounded tensor decode-graph representation is accepted; submitted decoder
code is not. This is a fixed-budget task, not a Pareto-frontier task: every
submission competes under the same 3.5-BPW feasibility constraint. Future BPW
operating points must be separate runs of the same protocol with a different
cap, never mixed into this scalar.

The supplied 128 diverse conversations are calibration-only and are never
scored. Online feedback is the mean across 128 sealed ID validation
conversations of `max(NLL_compressed - NLL_native, 0)`, measured on assistant
tokens. Final ID and OOD sets contain 128 conversations each. Lower is better.
Sealed reports retain per-conversation positive/signed deltas, domain scores,
family-stratified bootstrap intervals, and absolute native/compressed NLL as
unranked diagnostics. The exact calibration/validation/test bytes and the
pinned checkpoint, tokenizer, configuration, and chat template are SHA-256
attested before evaluation and re-attested after producer execution.
Quantization, calibration, decoding, and grading run on Apple MPS with CPU
fallback disabled. The starter is symmetric groupwise three-bit RTN with
40-weight groups.
