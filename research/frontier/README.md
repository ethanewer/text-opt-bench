# Local published-frontier validation

Every numeric point used by the blog must be produced by the local evaluator.
Paper-native results are context, not coordinates on a different metric.

Status labels:

- `faithful`: the defining method and paper metric are represented locally.
- `api-adapted`: executable, but a documented task API mismatch changes the method.
- `blocked-by-api`: the defining method cannot be represented by the current task.

The adapters in this directory retain source comments describing any mismatch.
`results.json` is generated from actual evaluator output and is the plot's only
source for numeric frontier markers.

## Reproduction commands

```bash
PYTHONPATH=. /tmp/text-opt-bm-poc-venv/bin/python -m bench evaluate optimizer_synthesis research/frontier/optimizer_schedule_free_adamw.py --json --full
PYTHONPATH=/tmp/fboptimizers:. /tmp/text-opt-bm-poc-venv/bin/python research/frontier/optimizer_shampoo_diagnostic.py
PYTHONPATH=. /tmp/text-opt-bm-poc-venv/bin/python research/frontier/gradient_ef21_diagnostic.py
PYTHONPATH=. /tmp/text-opt-bm-poc-venv/bin/python research/frontier/kv_snap_adakv_diagnostic.py --split val
PYTHONPATH=. /tmp/text-opt-bm-poc-venv/bin/python research/frontier/routing_avengers_pro_diagnostic.py
PYTHONPATH=. /tmp/text-opt-bm-poc-venv/bin/python -m bench evaluate hpo_taskset research/baselines/hpo_taskset_transfer.py --json --full
```

The Shampoo command uses an unmodified checkout of Meta's official
`facebookresearch/optimizers` package. The routing adapter uses the official
semantic-cluster mechanism but replaces the external embedding API with local
TF-IDF. The KV diagnostic uses per-head eager-attention masks instead of the
official CUDA-only flattened-cache kernel, making the mechanism portable across
CPU, CUDA, and MPS.

## Validation conclusions

- Distributed Shampoo is the strongest measured optimizer at `-3.71478249`.
- Ada-SnapKV beats equal-head SnapKV on validation (`0.17352195` versus
  `0.18220913`), but their order reverses on the tiny train split.
- EF21-HB (`0.80031065`) and EF21 (`0.84123770`) both lose to classical top-k
  error feedback (`0.642184`), invalidating a paper-frontier interpretation of
  the current gradient workload.
- The SLM score's zero clamp saturated during optimization. It must be replaced
  with a signed paired estimate and more held-out tokens before AWQ comparisons.
- FSBO and AWQ cannot be assigned honest local coordinates with the current
  anonymized data/API and Qwen3.5 hybrid architecture, respectively.
