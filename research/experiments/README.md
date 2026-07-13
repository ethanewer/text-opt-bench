# Research experiment runner

This is an archived launcher for the remaining CPU-only prototype studies. Its
manifest no longer contains the retired KV-cache or pre-protocol SLM jobs. It
must not be used to produce SLM evidence for the active benchmark.

The prototype environment needs the dependencies used by the selected CPU
studies, including NumPy, pandas, SciPy, and scikit-learn.

```bash
/tmp/text-opt-bm-poc-venv/bin/python research/experiments/run_experiments.py --dry-run
/tmp/text-opt-bm-poc-venv/bin/python research/experiments/run_experiments.py
```

The default resource envelope is two CPU slots and 70% of physical memory.
Every remaining job consumes one CPU slot, so two can run concurrently.

CPU library thread counts are reduced for every child process. `--dry-run`
displays the selected historical commands without requiring their data assets.

The HPO-B + TaskSet program currently validates the interactive table and
multi-fidelity scoring mechanics on a deterministic synthetic suite. Replacing
that generator with compact official HPO-B and TaskSet artifacts is the next
data-preparation step; its output is labeled as synthetic to prevent accidental
comparison with published scores.

Measured smoke-test results are recorded in `RESULTS.md`.
