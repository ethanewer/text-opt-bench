# Task: spec_decode_plan - adaptive speculative decoding lengths

Choose draft lengths for speculative decoding requests. Each request
contains an output length, per-position draft acceptance probabilities,
and cost multipliers for the draft and verifier models.

## Required API

```python
def plan(requests, config):
    """Return one list of draft lengths per request.

    policies[i][pos] is the draft length to use after `pos` output tokens
    have already been produced for request i.
    """
```

## Scoring

The evaluator computes exact expected serving cost under the standard
lossless speculative-decoding progress model. A verifier round with
draft length `k` always produces at least one target token; it may also
accept a prefix of the `k` draft tokens, and if all draft tokens are
accepted it advances by `k + 1` tokens.

Lower is better:

`mean expected speculative decoding cost across generated traces`

This models the real serving decision of when a larger draft is worth
its draft-model and verifier overhead. Useful approaches include dynamic
programming, calibrated confidence thresholds, remaining-length
adjustments, and cost-aware draft-length policies.

## Rules

Pure Python 3.12 stdlib only. Imports, file/network/process access,
introspection, large literals, and benchmark internals are forbidden.
The run must be deterministic and CPU-only.
