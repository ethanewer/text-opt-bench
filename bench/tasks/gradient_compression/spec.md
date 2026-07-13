# Task: communication-budgeted gradient compression

Compress distributed worker gradients while the evaluator owns data,
gradients, residual error feedback, aggregation, model updates, and validation.

```python
def init(parameter_count, worker_count, workload_info): ...
def encode(corrected_gradient, max_items, worker_info, state):
    return [indices, values, new_state]
```

At most `max_items` unique coordinates may be transmitted. Each value costs 32
bits, each index costs `ceil(log2(parameter_count))` bits, and every message has
a 16-bit header. Values are actually rounded through IEEE float32 before the
evaluator applies them. The evaluator subtracts the decoded message from the
corrected gradient to maintain residual error feedback. Fabricated values are
allowed as compressed update rules, but their real downstream effect is scored.

The metric combines normalized validation-loss curve area with the exact
fraction of dense communication, then adds a worst-workload term. Metrics also
report final validation accuracy and transmitted-bit fraction. Hidden workloads
change model dimension, worker count, IID/non-IID skew, feature scale, noise,
and bandwidth. This follows gradient-compression papers' quality-versus-bits
evaluation rather than gradient MSE.

Candidate code is deterministic, import-free safe Python and limited to 32 KB.
