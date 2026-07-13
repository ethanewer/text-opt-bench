# Task: offline cost-aware LLM routing

Learn a router from precomputed RouterBench outcomes. No LLM is called during
scoring. The evaluator owns all validation/test correctness and API-cost
outcomes and executes the selected model choice.

Implement:

```python
def fit(training_rows): ...
def route(prompt, model_stats, cost_penalty, state): ...
```

Each training row is `[prompt, quality, cost]`; `quality` and `cost` are lists
for five opaque models. `model_stats[i]` is `[mean_train_accuracy,
relative_mean_cost]`. `route` must return a model index. The evaluator calls it
at five cost penalties and scores the mean gap from the per-prompt oracle
utility `quality - penalty * normalized_cost`. Lower is better. Metrics also
report RouterBench-style average accuracy, mean cost, and oracle gap.

The visible fit rows, scored training rows, validation rows, and sealed test
rows are disjoint by stable sample ID. Dataset/source names are not supplied.
The program is limited to 32 KB, deterministic, import-free Python using the
safe builtins listed in the repository README. It may learn from the supplied
training rows but must not read files or emit outcomes.
