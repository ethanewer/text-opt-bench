# Task: kv_quant - KV-cache quantization for approximate attention

Compress a multi-layer transformer KV cache and answer attention queries
from the compressed representation. The KV tensors are fixed slices from
forward passes of the open-weight
`vijaymohan/gpt2-tinystories-from-scratch-10m` GPT-2-style model on
public-domain English text. The workload is CPU-only, but the scoring
model captures the deployment tradeoff behind long-context KV-cache
compression: smaller cache state is useful only if attention outputs stay
accurate and cheap to access.

## Required API

```python
def encode(cache, config):
    """cache: dict with layers and scale.
    Return any deterministic compressed representation."""

def attend(encoded, queries, config):
    """queries: list[layer][query][float].
    Return one output vector per layer/query using only `encoded`."""
```

`cache["layers"]` is a list of layer/head slice dictionaries. Each layer
has `keys`, `values`, and `importance`. `keys` and `values` are lists of
token-position vectors. `importance` is an accumulated attention score
from a small observation window near the end of the prompt; it is meant
to support H2O/SnapKV-style heavy-hitter retention. Instances use
selected heads from model layers 0, 2, 5, and 7, with real-text contexts
in the hundreds of tokens. `queries` are matching query-vector slices
from the same forward passes. `config` contains `n_layers`, `n_tokens`,
`key_dim`, `value_dim`, `selected_model_layers`,
`selected_model_heads`, `observation_queries`, `sink_tokens`,
`layer_weights`, `max_mse`, and scoring constants.

## Scoring

For each fixed cache/query instance, the evaluator computes exact
softmax attention for every selected layer/head from the original float cache, calls
your `encode`, estimates the storage cost of your encoded object, calls
`attend`, and computes layer-weighted mean squared error against the
exact outputs. Later layers carry larger error weights, so good solutions
can allocate compression budget unevenly across layers. The exposed
importance scores make token-selection strategies viable: retaining
attention sinks, recent tokens, and observed heavy hitters can beat
uniformly quantizing the whole cache.

Score per instance:

`encoded_storage_bytes + error_weight * mse + instruction_weight * bytecode_instructions`

Lower is better. Outputs with `mse > config["max_mse"]` are invalid. The
storage model is deterministic and intentionally simple: integers are
charged by bit length, floats cost 8 bytes, and nested containers have
small overheads. Returning the raw cache is valid but expensive. The
instruction term prevents pathological codecs that save bytes only by
using excessive per-query decompression work.

Candidate import-time code, `encode`, and `attend` each have bytecode
instruction budgets. The evaluator passes copies of all input data, so
mutating inputs cannot change the reference outputs. The candidate
module is reloaded between `encode` and `attend`, so the encoded object
must be self-contained; hidden module globals do not persist.

This is a perfect-information task. Validation instances use different
seeds and must beat a raw-ish quality gate.

## Rules

- No imports. Programs run under a curated builtins subset. Available:
  `abs, all, any, bool, dict, enumerate, filter, float, int, isinstance,
  len, list, map, max, min, print, range, reversed, round, set, slice,
  sorted, str, sum, tuple, zip` plus common exception types.
- Forbidden (checked): filesystem/process/threading modules, `sys`,
  `bench`, `builtins`, `__builtins__`, `importlib`, `__import__`, and
  introspection/eval helpers such as `globals`, `locals`, `vars`, `dir`,
  `getattr`, `type`, `object`, `eval`, `exec`, `compile`, and traceback
  frame attributes.
- Must be deterministic.
