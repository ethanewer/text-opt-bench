# Candidate-task proofs of concept

These scripts test evaluation mechanics for proposed benchmark tasks. They are
not benchmark tasks themselves. They deliberately keep downloaded models and
datasets outside the repository.

These are historical prototypes, not admissible benchmark evaluators. The old
compression and KV scripts are retired; the active SLM tasks and paper-native
diagnostics require Apple MPS, disable CPU fallback, and share one global MPS
lease.

Create an environment:

```bash
uv venv /tmp/text-opt-bm-poc-venv --python python3.12
uv pip install --python /tmp/text-opt-bm-poc-venv/bin/python \
  torch transformers safetensors
```

The experiments:

- `compression_ppl_poc.py`: **retired** pre-protocol pruning/quantization POC;
  it exits without model compute.
- `kv_ppl_poc.py`: **retired from the active suite** after long-context KV
  evaluation proved incompatible with the evaluation-time budget.
- `optimizer_poc.py`: compare update rules by fixed-step normalized
  loss-curve area across hidden workload families.
- `ann_poc.py`: compare vector indices by recall versus exact distance calls
  and index bytes.
- `router_poc.py`: evaluate routers on precomputed model outcomes using
  accuracy/cost Pareto metrics, with no model calls during scoring.
- `hpo_taskset_poc.py`: exercise transfer portfolios and multi-fidelity curve
  decisions through an evaluator-owned synthetic table before official HPO-B
  and TaskSet artifacts are compacted.
- `gradient_compression_poc.py`: compare dense, sign, top-k, and error-feedback
  top-k training by validation quality versus exact transmitted bits.

All scripts print one JSON object so their runtimes and rankings can be
captured mechanically.

Measured results and benchmark-design conclusions are in `RESULTS.md`.
