# Task: inference_batching - CPU-stable LLM serving scheduler

Schedule prefill/decode work for a simulated LLM serving engine. The
model is architecture-agnostic: no CUDA is used, but the cost model
captures common deployment tradeoffs around prefill batching, decode
steps, KV-cache token capacity, and latency tails.

## Required API

```python
def order(requests, config):
    """requests: list[dict] with id, arrival, prompt, output, priority.
    config: dict with max_batch, max_prefill_tokens, kv_capacity.
    Return a list[int] giving request ids in admission priority order."""
```

The evaluator forms prefill batches from your order, then simulates
continuous decoding with a fixed KV-cache capacity.

## Scoring

Score is the sum over deterministic traces of:

`priority_weighted_mean_latency + 0.30 * p95_latency + 0.04 * makespan`

Lower is better. Requests must appear exactly once. Candidate import-time
code and each ordering call each have a 5,000,000 bytecode-instruction
budget. The evaluator scores an untouched copy of each trace, so mutating
request dictionaries cannot change arrivals, prompt lengths, output
lengths, priorities, or ids.
Programs are also limited to 12,000 bytes of source, no literal
container with more than 80 items, at most 300 total literal container
items, and no string/bytes literal over 2,000 bytes.

This is a perfect-information task. Validation traces check basic quality
on different seeds with the same request-count shapes used in scoring.

## Rules

- No imports. Programs run under a curated builtins subset. Available:
  `abs, all, any, bool, dict, enumerate, filter, float, int, isinstance,
  len, list, map, max, min, print, range, reversed, round, set, slice,
  sorted, str, sum, tuple, zip` plus common exception types
  (`Exception`, `ValueError`, `KeyError`, `IndexError`, `TypeError`,
  `RuntimeError`, `StopIteration`, `ZeroDivisionError`, `LookupError`,
  `BaseException`). Anything else — including class definitions and
  builtins like `getattr`, `iter`, `next`, `divmod`, `pow`, `hasattr`,
  `frozenset`, `bytes` — is unavailable and fails at run time. Use
  operators (`**`, `//`, `%`) instead of math functions.
- Forbidden (checked): filesystem/process/threading modules, `sys`,
  `bench`, `builtins`, `__builtins__`, `importlib`, `__import__`, and
  introspection/eval helpers such as `globals`, `locals`, `vars`, `dir`,
  `getattr`, `type`, `object`, `eval`, `exec`, `compile`, and traceback
  frame attributes.
- Must be deterministic.
