# Task: transfer HPO over HPO-B and TaskSet curves

Implement a sequential optimizer evaluated by table lookup on compact artifacts
derived from the official HPO-B-v3 dataset and the original TaskSet paper's
1,000-configuration, five-replica learning-curve archives.

```python
def prepare(meta_tasks): ...
def suggest(task_info, configurations, observations, remaining_budget, state): ...
```

`suggest` returns `[configuration_index, fidelity_index]`. An observation is
`[index, fidelity, loss, cost]`. HPO-B tasks have one fidelity; TaskSet tasks
have four curve fidelities costing 0.125, 0.25, 0.5, and 1 full-training
equivalent. Repeating a pair is invalid.

The score is normalized-regret AUC at budgets 1, 2, 4, and 8, plus 0.25 times
the worst-quartile task score. This follows HPO-B's incumbent normalized-regret
protocol and TaskSet's validation-loss-versus-trial-budget analysis. Metrics
separately report both source datasets.

Meta-training tasks are supplied to `prepare`. Ranked task families and opaque
coordinate permutations are disjoint across training, validation, and sealed
test. Each split loads a fresh candidate module and calls `prepare` again, so
mutable state cannot cross split boundaries. The candidate is deterministic,
import-free safe Python and limited to 32 KB.
