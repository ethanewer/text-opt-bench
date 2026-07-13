# Task: Qwen3.5 constrained weight compression

Submit a deterministic Python weight producer for the pinned text-only,
nonthinking `Qwen/Qwen3.5-0.8B` checkpoint. The producer receives `--model`,
`--calibration`, `--output`, `--targets`, and `--device` arguments and must
write `3.125/` and `4.125/` QWeight bundles beneath the output directory.
These emitted weights—not a declared policy—are the object being graded.

Each bundle must contain `manifest.json` in the safe `qweight-1` schema and
referenced safetensors payloads. Every base state tensor must be represented
exactly once by a dense, packed affine, codebook, block-float, bounded decode-
graph, or alias record.
This supports dense BF16/FP16, FP8 and NVFP4-style block floating point,
GPTQ/AWQ/HQQ-style affine groups, and GGUF-style uniform, K-quant, IQ, or
importance-matrix block layouts. The graph language is tensor-only and
whitelists unpacking, bitfields, lookup, reshape/permutation, slicing,
concatenation, repetition, and arithmetic; it cannot call submitted code.
A quantizer may use arbitrary optimization logic while producing
the bundle; the evaluator accepts no submitted executable decoder or kernel.

The hard constraints are **whole submitted weight-bundle** sizes of 3.125 and
4.125 bits per unique base-model parameter. All files count byte for byte:
packed values, scales, zero points, codebooks, permutations/group indices,
aliases, padding, safetensors headers, and the JSON manifest. The architecture,
tokenizer, and evaluator's trusted generic decoder are fixed and do not count.

A bundle may alternatively contain one losslessly wrapped, authenticated
Qwen3.5 `.gguf` payload through a trusted `native_gguf` record. Every GGUF byte
plus the QWeight manifest counts toward the cap. The evaluator reverses the
fixed llama.cpp Qwen3.5 tensor transforms and accepts every quantization type
supported by pinned `gguf==0.19.0`; it never runs code from the payload.
Consequently an agent can submit GGUF-style weights without transcoding their
packed blocks, provided the complete bundle fits the requested operating point.

The producer may calibrate only with the supplied 128 training conversations.
They are never scored. During optimization, both checkpoints are graded on the
same sealed 64-conversation ID validation set using assistant-token signed
delta-NLL (compressed minus the runtime uncompressed reference), macro-averaged
through domains and operation-template clusters. Final ID and OOD curves each
contain 64 disjoint conversations. All model loading, calibration transforms,
generic QWeight decoding, and inference use Apple MPS with PyTorch CPU fallback
disabled. Native GGUF parsing/dequantization uses the pinned trusted CPU
importer before the exact reconstructed weights are moved to MPS; ordinary
JSON parsing and byte serialization may also use the CPU.
