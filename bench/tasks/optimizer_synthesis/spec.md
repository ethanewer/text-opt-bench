# Task: generalizing first-order optimizer synthesis

Implement a bounded-state optimizer while the evaluator owns parameters,
gradients, training loops, and validation losses:

```python
def init(task_info, parameter_count): ...
def update(parameters, gradients, state, step, task_info):
    return [new_parameters, new_state]
```

Workloads include rotated ill-conditioned quadratics, scaled logistic
regression, and low-rank matrix factorization. `task_info` reveals only the
family, dimension/scale bucket, and a coarse conditioning descriptor. Entire
instances and family/scale combinations are held out.

Every workload runs for 120 steps. The score is the mean area under
`log(validation_loss / initial_loss)` plus 0.25 times the worst-quartile area;
lower is better. This is a fixed-step TaskSet/AlgoPerf-style metric and never
uses wall time. Metrics include final loss ratio and worst-quartile behavior.

The candidate is deterministic, import-free safe Python, limited to 32 KB,
and may keep arbitrary JSON-like state. It cannot execute the workload or
compute validation loss itself.
