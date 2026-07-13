# Candidate-task proof-of-concept results

These are feasibility measurements, not benchmark leaderboards. Runs used an
Apple M5 Mac with 34 GB unified memory. Downloaded models and datasets were
kept outside the repository.

This file is historical. The CPU/CUDA/backend-comparison SLM measurements and
short-context KV experiments are inadmissible under the active protocol; all
current SLM generation, calibration, compression, and scoring is strict MPS
under the shared device lease.

## Results

| Proof of concept | Backend and input | Wall time | Main result |
| --- | --- | ---: | --- |
| SLM compression | PyTorch MPS, Qwen2.5-0.5B, 510 tokens per split, six transformations | 9.8 s | Groupwise 4-bit quantization ranked first; all methods scored by held-out perplexity. |
| SLM compression | PyTorch CPU, Qwen2.5-0.5B, 127 tokens per split | 26.8 s | Same six-method ordering as MPS at the same sample size. |
| Tiny-LM compression | PyTorch MPS, TinyStories-1M, 256 tokens per split | 3.2 s | Same ranking as CPU; backend NLL differences were about 2e-5. |
| KV eviction | PyTorch MPS, TinyStories-1M, 96 tokens, 24-token cache | 6.6 s | Full-cache PPL 4.57; sink+recent 5.45; recent 5.84; H2O-style 6.42. |
| KV eviction | PyTorch CPU, same run | 1.8 s | Same ranking as MPS. |
| KV eviction, longer context | PyTorch CPU, 192 tokens, 48-token cache | 2.9 s | H2O-style PPL 5.65; sink+recent 6.15; recent 6.43, reversing the short-context policy ordering. |
| Optimizer generalization | PyTorch CPU, eight workload instances, five optimizers, 150 steps | 1.3 s | Momentum, RMSprop, Adam, SGD, Lion ordering on normalized validation-loss AUC. |
| Optimizer generalization | PyTorch MPS, same run | 4.1 s | Same ordering; one ill-conditioned workload produced a roughly 1.5% aggregate score difference. |
| Offline LLM routing | CPU, official RouterBench outcomes, 36,497 prompts | 37.9 s | A prompt router improved the low-cost frontier over best-single; the oracle gap remained large. |
| ANN indexing | NumPy CPU, 12k vectors, 160 queries | 0.17 s | IVF recall@10 0.574 under 55,840 exact distance calls; LSH 0.317 under 38,400 calls. |

## Design conclusions

- Use held-out negative log-likelihood or perplexity as the official score for
  language-model compression. Calibration KL and reconstruction MSE are useful
  diagnostics only.
- Use three disjoint roles: visible development/calibration instances, a
  hidden validation distribution for iterative feedback, and a sealed test
  distribution for final ranking. Corpus families, model seeds, and operating
  budgets should all vary across the split.
- Score several context lengths and cache budgets. The KV proof of concept
  changed its winning compressed-cache policy when context and budget doubled.
- Treat small-model compression as its own workload. Do not claim that its
  method ordering predicts 1B-8B ordering without an empirical transfer study.
- Make CPU the canonical numerical scorer. The same PyTorch implementation can
  run on CPU, CUDA, or MPS, but accelerator outputs should be compared with
  tolerances rather than required to be bit-identical.
- Do not include wall-clock time in a cross-platform local score. Count
  evaluator-owned operations, bytes, steps, or candidate evaluations instead.
- Compact the routing data to prompts, correctness values, costs, and stable
  identifiers. The proof of concept spends much of its time loading a 95 MB
  dataframe containing fields the scorer does not need.
