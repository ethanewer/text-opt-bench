# Task: weight_quant - real transformer weight quantization

Compress selected linear weight matrices from an open-weight
TinyStories GPT-2-style transformer while preserving outputs on held-out
real activation rows.

## Required API

```python
def compress(layers, config):
    """Return an encoded representation of the layer weights."""

def infer(encoded, inputs, config):
    """Return one output matrix per layer for the supplied hidden inputs."""
```

`layers` contains real trained weight slices, bias vectors, calibration
inputs/outputs, and `input_rms` activation statistics. The calibration
rows come from two public-domain text passages. `inputs` contains
held-out activation rows from different passages, and those rows are not
passed to `compress`.

## Scoring

Lower is better:

`encoded_storage_bytes + error_weight * normalized_MSE + instruction_weight * bytecode_instructions`

The error is measured against full-precision matrix outputs on held-out
activation rows and normalized by output energy per layer. The evaluator
reloads the candidate module between `compress` and `infer`, so the
encoded object must be self-contained.

Useful approaches include per-group/per-channel quantization,
activation-aware scaling, mixed precision, outlier handling, low-rank or
codebook residuals, and GPTQ-like reconstruction using calibration
inputs and outputs.

## Rules

Pure Python 3.12 stdlib only. Imports, file/network/process access,
introspection, traceback/frame access, large literals, and benchmark
internals are forbidden. The run must be deterministic and CPU-only.
