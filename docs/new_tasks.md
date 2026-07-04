# New Benchmark Tasks

This document summarizes the tasks added on the `review/current-benchmark`
branch, why they fit the benchmark, how they are graded, and the research
work they are intended to connect to.

## Acceptance Criteria Used

Each accepted task is:

- CPU-only at benchmark runtime.
- Deterministic, with no wall-clock timing in the score.
- Fast enough for repeated evaluation.
- Algorithmic rather than knowledge-recall based.
- Valuable when improved: better scores correspond to lower serving memory,
  lower quantization error, or lower expected inference cost.
- Hardened under the benchmark's cooperative threat model: imports,
  filesystem/process access, frame introspection, large literals, invalid
  numeric outputs, and obvious state-smuggling paths are rejected where
  relevant.

No branch-added task was removed in the final review. Two fixes were made:
`spec_decode_plan` now deep-copies nested acceptance traces before calling a
candidate, and `spec_tree_select` now generates token-tree edge probabilities
as a proper subdistribution.

## KV Cache Compression

### `kv_quant`

**Goal.** Compress multi-layer real-model KV-cache slices and answer attention
queries from the compressed representation.

**Input.** Four selected layer/head slices from the open-weight
`vijaymohan/gpt2-tinystories-from-scratch-10m` GPT-2-style model, with
Q/K/V values generated from public-domain text. The candidate sees keys,
values, and observation-window importance scores.

**API.**

```python
def encode(cache, config): ...
def attend(encoded, queries, config): ...
```

**Grading.** Lower is better:

```text
encoded_storage_bytes + error_weight * layer_weighted_attention_MSE
    + instruction_weight * bytecode_instructions
```

The evaluator reloads the candidate module between `encode` and `attend`, so
the encoded object must be self-contained.

**Why it is valuable.** KV cache memory is a major long-context serving cost.
Better algorithms reduce memory while preserving attention outputs.

**Related work.**

- [H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models](https://arxiv.org/abs/2306.14048)
- [SnapKV: LLM Knows What You are Looking for Before Generation](https://arxiv.org/abs/2404.14469)
- [KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache](https://arxiv.org/abs/2402.02750)

### `kv_fixed_budget`

**Goal.** Maximize attention fidelity under a fixed encoded KV-cache byte cap.

**Input and API.** Same real-model KV slices and `encode`/`attend` API as
`kv_quant`.

**Grading.** The encoded object must fit `config["max_encoded_bytes"]`.
Lower is better:

```text
error_weight * layer_weighted_attention_MSE
    + instruction_weight * bytecode_instructions
```

**Why it is valuable.** This matches deployment settings where the memory
budget is fixed and the algorithmic question is how to spend it across
tokens, layers, keys, and values.

**Related work.**

- [KIVI](https://arxiv.org/abs/2402.02750)
- [SnapKV](https://arxiv.org/abs/2404.14469)
- [H2O](https://arxiv.org/abs/2306.14048)

### `kv_layer_budget`

**Goal.** Allocate retention and quantization budgets across layers. The
evaluator owns the compression algorithm; the candidate only returns layer
budgets.

**API.**

```python
def allocate(cache_info, config): ...
```

Each layer budget is `[keep_tokens, key_levels, value_levels]`.

**Grading.** The evaluator applies a fixed heavy-hitter/recent/sink compressor
and quantizer from the candidate budgets. Lower is better:

```text
error_weight * layer_weighted_attention_MSE
    + instruction_weight * allocate_instructions
```

The encoded cache must stay under a fixed byte cap.

**Why it is valuable.** It isolates the layer-allocation decision studied in
layerwise KV-cache methods, reducing room for arbitrary implementation tricks.

**Related work.**

- [PyramidKV: Dynamic KV Cache Compression based on Pyramidal Information Funneling](https://arxiv.org/abs/2406.02069)
- [SnapKV](https://arxiv.org/abs/2404.14469)
- [H2O](https://arxiv.org/abs/2306.14048)

## Weight Quantization

### `weight_quant`

**Goal.** Compress real transformer linear weight slices while preserving
layer outputs on held-out real activation rows.

**Input.** Selected trained linear weight slices, biases, calibration
inputs/outputs, and activation RMS statistics from
`vijaymohan/gpt2-tinystories-from-scratch-10m`. Calibration activations and
held-out activations come from separate text passages.

**API.**

```python
def compress(layers, config): ...
def infer(encoded, inputs, config): ...
```

**Grading.** Lower is better:

```text
encoded_storage_bytes + error_weight * normalized_MSE
    + instruction_weight * bytecode_instructions
```

The evaluator reloads the candidate module between `compress` and `infer`, so
the encoded object must be self-contained.

**Why it is valuable.** Post-training weight quantization reduces model memory
and bandwidth pressure. Improvements in this task correspond to better
compression/fidelity tradeoffs for real transformer weights.

**Related work.**

- [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323)
- [AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](https://arxiv.org/abs/2306.00978)
- [SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models](https://arxiv.org/abs/2211.10438)
- [AQLM: Extreme Compression of Large Language Models via Additive Quantization](https://arxiv.org/abs/2401.06118)
- [QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks](https://arxiv.org/abs/2402.04396)

## Speculative Decoding

### `spec_decode_plan`

**Goal.** Choose adaptive draft lengths for speculative decoding requests.

**Input.** Synthetic but structured request traces with output lengths,
per-position draft acceptance probabilities, and draft/verifier cost
multipliers.

**API.**

```python
def plan(requests, config): ...
```

The candidate returns exactly one valid draft length per generated-token
position.

**Grading.** The evaluator computes exact expected serving cost under the
lossless speculative decoding progress model. Lower is better:

```text
mean expected speculative-decoding cost across traces
```

**Why it is valuable.** Choosing draft length is a real serving tradeoff:
longer drafts can reduce verifier rounds when accepted, but waste draft and
verification work when rejected.

**Related work.**

- [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192)
- [EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](https://arxiv.org/abs/2401.15077)
- [Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads](https://arxiv.org/abs/2401.10774)

### `spec_tree_select`

**Goal.** Select prefix-closed speculative token trees for parallel
verification.

**Input.** Synthetic candidate token trees with parent links, depth, rank,
edge probability, and path probability. Sibling edge probabilities form a
proper subdistribution.

**API.**

```python
def select(trees, config): ...
```

The candidate returns one selected node-id list per tree. Selections must be
unique, under `config["max_nodes"]`, and prefix-closed.

**Grading.** Lower is better:

```text
mean verifier cost / expected generated tokens
```

Selected nodes add expected accepted draft tokens; cost increases with node
count and tree depth to model verifier/tree-attention overhead.

**Why it is valuable.** Tree-shaped speculative verification is central to
multi-token and multi-branch speculative decoding systems. Better algorithms
choose enough likely nodes to improve expected progress without overpaying
for wide or deep verification.

**Related work.**

- [SpecInfer: Accelerating Generative Large Language Model Serving with Tree-based Speculative Inference and Verification](https://arxiv.org/abs/2305.09781)
- [Medusa](https://arxiv.org/abs/2401.10774)
- [EAGLE](https://arxiv.org/abs/2401.15077)
