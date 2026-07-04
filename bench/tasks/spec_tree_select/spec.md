# Task: spec_tree_select - speculative token-tree selection

Select prefix-closed draft-token trees for one-step parallel
verification. Each input tree contains candidate draft nodes with parent
links, depth, rank, edge probability, and path probability.

## Required API

```python
def select(trees, config):
    """Return one list of selected node ids per tree."""
```

Selections must be prefix-closed: if a node is selected, every ancestor
must also be selected. At most `config["max_nodes"]` nodes may be
selected per tree.

## Scoring

The evaluator computes expected verifier cost per generated token. A
selected node contributes its path probability to expected accepted
draft tokens. Verification cost increases with selected node count and
tree depth, capturing tree-attention and verifier overhead.

Lower is better:

`mean verifier cost / expected generated tokens`

This models Medusa/EAGLE/SpecInfer-style decisions about which token
tree to verify in parallel. Useful approaches include best-first
frontier expansion, prefix-closed knapsack, dynamic tree depth control,
and cost-aware pruning of low-probability branches.

## Rules

Pure Python 3.12 stdlib only. Imports, file/network/process access,
introspection, large literals, and benchmark internals are forbidden.
The run must be deterministic and CPU-only.
