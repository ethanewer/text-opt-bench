"""Deterministically generate optimizer-generalization protocol v9.

The analytic diagnostic tier exposes the same natural matrix-plus-vector
parameter surface across families. Shapes, initial RMS, first-gradient RMS,
and horizons are sampled from shared track designs rather than family-specific
distributions, so analytic-family difficulty is expressed through loss
geometry rather than a trivial initialization fingerprint. The ranked neural
tier exposes natural multi-block parameter shapes, and sealed test adds three
architectures absent from development.
"""

from __future__ import annotations

import hashlib
import gzip
import json
import math
import struct
import urllib.request
from pathlib import Path

import numpy as np

from bench import heldout


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
SCHEMA = 8
PROTOCOL = 9
DEVELOPMENT_FAMILIES = (
    "quadratic", "logistic", "robust", "factorization", "softmax",
)
TEST_ONLY_FAMILIES = (
    "nonlinear", "poisson", "quantile", "ranking", "fourier",
)
SPLITS = {
    "train": {"seed": 41_001, "id_per_family": 16, "ood_per_family": 0},
    "validation": {"seed": 41_002, "id_per_family": 24,
                   "ood_per_family": 24},
    "test": {"seed": 41_003, "id_per_family": 28,
             "ood_per_family": 28},
}

# Real-data workloads are a deliberately small canonical subset, not a claim
# to reproduce all of TaskSet or AlgoPerf.  They make the ranked score depend
# on actual neural training while the larger analytic tier remains a cheap
# development/red-team signal.
REAL_COUNTS = {
    "train": {"id": 8, "ood": 0},
    "validation": {"id": 16, "ood": 0},
    "test": {"id": 8, "ood": 8},
}
REAL_FAMILIES = ("image_mlp", "image_deep_mlp", "image_conv",
                 "image_autoencoder", "char_lm")
TEST_ONLY_REAL_FAMILIES = (
    "image_residual", "image_gated_mlp", "image_bottleneck",
)
SOURCE_SPECS = {
    "mnist.npz": (
        "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz",
        "731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1"),
    "fashion-train-images.gz": (
        "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"
        "train-images-idx3-ubyte.gz",
        "3aede38d61863908ad78613f6a32ed271626dd12800ba2636569512369268a84"),
    "fashion-train-labels.gz": (
        "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"
        "train-labels-idx1-ubyte.gz",
        "a04f17134ac03560a47e3764e11b92fc97de4d1bfaf8ba1a3aa29af54cc90845"),
    "tinyshakespeare.txt": (
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/"
        "tinyshakespeare/input.txt",
        "86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed"),
    "alice.txt": (
        "https://www.gutenberg.org/files/11/11-0.txt",
        "a3a27f8edbf7fcd9b8ba8435494440e24952deaa3e2f2d65192d4cb7ca403754"),
}
_SOURCE_CACHE = None


def _r(value, digits=8):
    if isinstance(value, np.ndarray):
        return np.round(value, digits).tolist()
    return round(float(value), digits)


def _log_uniform(rng, low, high):
    return float(math.exp(rng.uniform(math.log(low), math.log(high))))


def _horizon(rng, track):
    if track == "id":
        return int(rng.integers(96, 177))
    if rng.random() < 0.5:
        return int(rng.integers(64, 97))
    return int(rng.integers(192, 257))


def _batch_seed(rng):
    return int(rng.integers(1, 2**31 - 1))


def _softplus(value):
    return max(value, 0.0) + math.log1p(math.exp(-abs(value)))


def _interface_designs(seed, track, count):
    """Shared observable distributions, reused across all families.

    The numeric design is shared, but each family receives independently drawn
    parameter values.  Consequently every family has exactly the same finite
    shape/RMS/horizon support without duplicated parameter tensors.
    """
    # The candidate-visible part of the design is deliberately identical
    # across ID/OOD.  A separate stream changes only the hidden horizon.
    # This makes the finite realized support match, not merely the population
    # distribution from which it was sampled.
    rng = np.random.default_rng(seed ^ 0x51A7)
    horizon_rng = np.random.default_rng(
        seed ^ (0x101D if track == "id" else 0x00D5))
    return [{
        "rows": int(rng.integers(12, 21)),
        "cols": int(rng.integers(3, 6)),
        "initial_rms": _log_uniform(rng, 0.04, 0.24),
        "gradient_rms": _log_uniform(rng, 0.015, 0.35),
        "horizon": _horizon(horizon_rng, track),
    } for _ in range(count)]


def _initial_parameters(rng, design):
    rows, cols = design["rows"], design["cols"]
    matrix = rng.normal(size=(rows, cols))
    vector = rng.normal(size=cols)
    squared = float(np.square(matrix).sum() + np.square(vector).sum())
    count = matrix.size + vector.size
    scale = design["initial_rms"] / math.sqrt(squared / count)
    # Retain enough precision that the *realized* aggregate RMS, not only its
    # latent target, matches across family/track copies of an interface design.
    return [_r(matrix * scale, 14), _r(vector * scale, 14)]


def _base(family, track, design, initial, raw_anchor, rng, payload,
          batch_size):
    return {
        "family": family,
        "track": track,
        "horizon": design["horizon"],
        "shapes": [[design["rows"], design["cols"]], [design["cols"]]],
        "initial": initial,
        "reference_anchor": _r(raw_anchor, 10),
        "loss_scale": 1.0,
        "target_initial_gradient_rms": _r(design["gradient_rms"], 10),
        "batch_size": int(batch_size),
        "batch_seed": _batch_seed(rng),
        "payload": payload,
    }


def _features(rng, count, rows, condition):
    q, _ = np.linalg.qr(rng.normal(size=(rows, rows)))
    scales = np.geomspace(1.0 / math.sqrt(condition), 1.0, rows)
    return (rng.normal(size=(count, rows)) * scales) @ q.T


def _linear_targets(x, weights, bias):
    return x @ weights + bias


def _quadratic(rng, track, design, initial):
    rows, cols = design["rows"], design["cols"]
    condition = (_log_uniform(rng, 4, 120) if track == "id" else
                 _log_uniform(rng, 350, 6_000))
    truth_w = rng.normal(scale=0.65, size=(rows, cols))
    truth_b = rng.normal(scale=0.3, size=cols)
    train_x = _features(rng, 112, rows, condition)
    valid_x = _features(rng, 160, rows, condition)
    train = np.column_stack([train_x, _linear_targets(train_x, truth_w, truth_b)])
    valid = np.column_stack([valid_x, _linear_targets(valid_x, truth_w, truth_b)])
    return _base(
        "quadratic", track, design, initial, 0.0, rng,
        {"train": _r(train), "validation": _r(valid)}, 16)


def _classification_rows(rng, count, rows, cols, condition, weights, bias):
    x = _features(rng, count, rows, condition)
    logits = x @ weights + bias
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))
    labels = (rng.random((count, cols)) < probabilities).astype(float)
    return np.column_stack([x, labels])


def _binary_loss(data, rows, weights, bias):
    total = 0.0
    for record in data:
        x, labels = record[:rows], record[rows:]
        for output in range(len(bias)):
            z = bias[output] + sum(x[j] * weights[j][output]
                                   for j in range(rows))
            total += _softplus(z) - labels[output] * z
    return total / (len(data) * len(bias))


def _logistic(rng, track, design, initial):
    rows, cols = design["rows"], design["cols"]
    condition = (_log_uniform(rng, 2, 20) if track == "id" else
                 _log_uniform(rng, 40, 250))
    weights = rng.normal(size=(rows, cols)) * 1.7 / math.sqrt(rows)
    bias = rng.normal(scale=0.25, size=cols)
    train = _classification_rows(rng, 128, rows, cols, condition, weights, bias)
    valid = _classification_rows(rng, 192, rows, cols, condition, weights, bias)
    anchor = _binary_loss(valid.tolist(), rows, weights.tolist(), bias.tolist())
    return _base(
        "logistic", track, design, initial, anchor, rng,
        {"train": _r(train), "validation": _r(valid)}, 16)


def _regression_rows(rng, count, rows, cols, condition, weights, bias,
                     outlier_fraction, outlier_scale):
    x = _features(rng, count, rows, condition)
    targets = _linear_targets(x, weights, bias)
    targets += rng.normal(scale=0.10, size=targets.shape)
    number = max(1, int(round(targets.size * outlier_fraction)))
    indices = rng.choice(targets.size, number, replace=False)
    targets.flat[indices] += rng.normal(scale=outlier_scale, size=number)
    return np.column_stack([x, targets])


def _huber_loss(data, rows, weights, bias, delta):
    total = 0.0
    for record in data:
        x, targets = record[:rows], record[rows:]
        for output, target in enumerate(targets):
            error = bias[output] + sum(x[j] * weights[j][output]
                                       for j in range(rows)) - target
            absolute = abs(error)
            total += (0.5 * error * error if absolute <= delta else
                      delta * (absolute - 0.5 * delta))
    return total / (len(data) * len(bias))


def _robust(rng, track, design, initial):
    rows, cols = design["rows"], design["cols"]
    condition = _log_uniform(rng, 1, 15)
    fraction = (float(rng.uniform(0.04, 0.14)) if track == "id" else
                float(rng.uniform(0.18, 0.34)))
    outlier_scale = (_log_uniform(rng, 2, 12) if track == "id" else
                     _log_uniform(rng, 18, 70))
    delta = float(rng.uniform(0.6, 1.4))
    weights = rng.normal(scale=0.7, size=(rows, cols))
    bias = rng.normal(scale=0.25, size=cols)
    train = _regression_rows(
        rng, 128, rows, cols, condition, weights, bias, fraction, outlier_scale)
    valid = _regression_rows(
        rng, 192, rows, cols, condition, weights, bias, fraction, outlier_scale)
    anchor = _huber_loss(
        valid.tolist(), rows, weights.tolist(), bias.tolist(), delta)
    return _base(
        "robust", track, design, initial, anchor, rng,
        {"delta": _r(delta), "train": _r(train), "validation": _r(valid)},
        16)


def _factorization(rng, track, design, initial):
    nodes, width = design["rows"], design["cols"]
    left_count = int(rng.integers(5, nodes - 4))
    right_count = nodes - left_count
    effective_rank = min(2, width)
    truth = np.zeros((nodes, width))
    truth[:left_count, :effective_rank] = rng.normal(
        scale=0.8, size=(left_count, effective_rank))
    truth[left_count:, :effective_rank] = rng.normal(
        scale=0.8, size=(right_count, effective_rank))
    latent_scale = rng.normal(scale=0.12, size=width)
    target = np.zeros((left_count, right_count))
    for i in range(left_count):
        for j in range(right_count):
            target[i, j] = sum(
                truth[i, k] * truth[left_count + j, k]
                * (1.0 + latent_scale[k]) for k in range(width))
    entries = [[i, j, float(target[i, j])]
               for i in range(left_count) for j in range(right_count)]
    rng.shuffle(entries)
    degrees = effective_rank * (left_count + right_count - effective_rank)
    fraction = 0.78 if track == "id" else 0.68
    cut = max(int(round(len(entries) * fraction)),
              int(math.ceil(1.2 * degrees)))
    cut = min(cut, max(1, len(entries) - max(3, len(entries) // 8)))
    train, valid = entries[:cut], entries[cut:]
    covered_left = {row[0] for row in train}
    covered_right = {row[1] for row in train}
    if (len(covered_left) != left_count or
            len(covered_right) != right_count):
        raise RuntimeError("factorization training graph has an isolated node")
    # A randomized evaluator-side edge cover uses at most nodes-1 records and
    # then fills to this size.  It remains a proper minibatch for every graph.
    batch_size = min(cut - 1, max(nodes, 16))
    return _base(
        "factorization", track, design, initial, 0.0, rng,
        {"left_count": left_count,
         "train": [[i, j, _r(y)] for i, j, y in train],
         "validation": [[i, j, _r(y)] for i, j, y in valid]},
        batch_size)


def _multiclass_rows(rng, count, rows, classes, condition, weights, bias,
                     class_scales):
    x = _features(rng, count, rows, condition)
    logits = (x @ weights + bias) * class_scales
    logits -= logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    draws = rng.random(count)
    labels = np.array([np.searchsorted(np.cumsum(row), draw)
                       for row, draw in zip(probabilities, draws)])
    return np.column_stack([x, labels])


def _softmax_loss(data, weights, bias, class_scales):
    total = 0.0
    for record in data:
        x, label = record[:-1], int(record[-1])
        logits = [class_scales[c] * (
            bias[c] + sum(x[j] * weights[j][c] for j in range(len(x))))
                  for c in range(len(bias))]
        maximum = max(logits)
        total += maximum + math.log(sum(math.exp(z - maximum) for z in logits))
        total -= logits[label]
    return total / len(data)


def _softmax(rng, track, design, initial):
    rows, classes = design["rows"], design["cols"]
    condition = (_log_uniform(rng, 2, 15) if track == "id" else
                 _log_uniform(rng, 30, 180))
    weights = rng.normal(size=(rows, classes)) * 1.5 / math.sqrt(rows)
    weights -= weights.mean(axis=1, keepdims=True)
    bias = rng.normal(scale=0.25, size=classes)
    bias -= bias.mean()
    if track == "id":
        class_scales = np.exp(rng.uniform(
            math.log(0.55), math.log(1.8), size=classes))
    else:
        class_scales = np.exp(rng.uniform(
            math.log(0.2), math.log(3.5), size=classes))
    train = _multiclass_rows(
        rng, 144, rows, classes, condition, weights, bias, class_scales)
    valid = _multiclass_rows(
        rng, 208, rows, classes, condition, weights, bias, class_scales)
    anchor = _softmax_loss(
        valid.tolist(), weights.tolist(), bias.tolist(), class_scales.tolist())
    return _base(
        "softmax", track, design, initial, anchor, rng,
        {"class_scales": _r(class_scales),
         "train": _r(train), "validation": _r(valid)}, 18)


def _nonlinear_rows(rng, count, rows, weights, bias, output, input_scale,
                    noise):
    x = rng.normal(size=(count, rows)) * input_scale
    hidden = np.tanh(x @ weights + bias)
    targets = hidden @ output
    if noise:
        targets += rng.normal(scale=noise, size=count)
    return np.column_stack([x, targets])


def _nonlinear(rng, track, design, initial):
    rows, hidden = design["rows"], design["cols"]
    # A saturated teacher makes this a genuinely unseen neural-objective
    # geometry instead of a near-linear regression problem.  ID and OOD share
    # the public interface statistics; OOD shifts only activation saturation.
    input_scale = (_log_uniform(rng, 1.0, 2.5) if track == "id" else
                   _log_uniform(rng, 4.0, 8.0))
    weights = rng.normal(scale=3.2 / math.sqrt(rows), size=(rows, hidden))
    bias = rng.normal(scale=1.0, size=hidden)
    output = rng.choice([-1.0, 1.0], size=hidden) / math.sqrt(hidden)
    train = _nonlinear_rows(
        rng, 144, rows, weights, bias, output, input_scale, 0.0)
    valid = _nonlinear_rows(
        rng, 208, rows, weights, bias, output, input_scale, 0.0)
    return _base(
        "nonlinear", track, design, initial, 0.0, rng,
        {"output": _r(output), "train": _r(train),
         "validation": _r(valid)}, 18)


def _poisson_rows(rng, count, rows, condition, weights, bias):
    x = _features(rng, count, rows, condition)
    log_rates = x @ weights + bias
    rates = np.exp(log_rates)
    counts = rng.poisson(rates).astype(float)
    return np.column_stack([x, counts])


def _poisson_loss(data, rows, weights, bias):
    total = 0.0
    for record in data:
        x, counts = record[:rows], record[rows:]
        for output, count in enumerate(counts):
            z = bias[output] + sum(
                x[j] * weights[j][output] for j in range(rows))
            total += math.exp(z) - count * z
    return total / (len(data) * len(bias))


def _poisson(rng, track, design, initial):
    rows, outputs = design["rows"], design["cols"]
    condition = (_log_uniform(rng, 2, 18) if track == "id" else
                 _log_uniform(rng, 45, 260))
    weight_scale = 0.55 if track == "id" else 0.9
    weights = rng.normal(
        scale=weight_scale / math.sqrt(rows), size=(rows, outputs))
    bias = rng.uniform(-0.35, 0.65 if track == "id" else 1.0, size=outputs)
    train = _poisson_rows(
        rng, 144, rows, condition, weights, bias)
    valid = _poisson_rows(
        rng, 208, rows, condition, weights, bias)
    anchor = _poisson_loss(
        valid.tolist(), rows, weights.tolist(), bias.tolist())
    return _base(
        "poisson", track, design, initial, anchor, rng,
        {"train": _r(train), "validation": _r(valid)}, 18)


def _quantile_rows(rng, count, rows, outputs, condition, weights, bias,
                   quantile, noise_scale):
    x = _features(rng, count, rows, condition)
    targets = _linear_targets(x, weights, bias)
    negative = rng.random((count, outputs)) < quantile
    magnitudes = rng.exponential(scale=noise_scale, size=(count, outputs))
    noise = np.where(negative, -magnitudes, magnitudes)
    return np.column_stack([x, targets + noise])


def _quantile_loss(data, rows, weights, bias, quantile):
    total = 0.0
    for record in data:
        x, targets = record[:rows], record[rows:]
        for output, target in enumerate(targets):
            prediction = bias[output] + sum(
                x[j] * weights[j][output] for j in range(rows))
            error = target - prediction
            total += (quantile * error if error >= 0.0 else
                      (quantile - 1.0) * error)
    return total / (len(data) * len(bias))


def _quantile(rng, track, design, initial):
    rows, outputs = design["rows"], design["cols"]
    if track == "id":
        condition = _log_uniform(rng, 1, 18)
        quantile = float(rng.uniform(0.3, 0.7))
        noise_scale = _log_uniform(rng, 0.08, 0.5)
    else:
        condition = _log_uniform(rng, 45, 300)
        quantile = float(rng.choice([0.1, 0.15, 0.85, 0.9]))
        noise_scale = _log_uniform(rng, 1.5, 6.0)
    weights = rng.normal(scale=0.7, size=(rows, outputs))
    bias = rng.normal(scale=0.3, size=outputs)
    train = _quantile_rows(
        rng, 144, rows, outputs, condition, weights, bias,
        quantile, noise_scale)
    valid = _quantile_rows(
        rng, 208, rows, outputs, condition, weights, bias,
        quantile, noise_scale)
    anchor = _quantile_loss(
        valid.tolist(), rows, weights.tolist(), bias.tolist(), quantile)
    return _base(
        "quantile", track, design, initial, anchor, rng,
        {"quantile": _r(quantile), "train": _r(train),
         "validation": _r(valid)}, 18)


def _ranking_rows(rng, count, rows, outputs, condition, weights, bias,
                  temperature):
    features = _features(rng, count * 2, rows, condition)
    differences = features[:count] - features[count:]
    logits = temperature * (differences @ weights + bias)
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))
    labels = (rng.random((count, outputs)) < probabilities).astype(float)
    return np.column_stack([differences, labels])


def _ranking_loss(data, rows, weights, bias, temperature):
    total = 0.0
    for record in data:
        x, labels = record[:rows], record[rows:]
        for output, label in enumerate(labels):
            z = temperature * (bias[output] + sum(
                x[j] * weights[j][output] for j in range(rows)))
            total += _softplus(z) - label * z
    return total / (len(data) * len(bias))


def _ranking(rng, track, design, initial):
    rows, outputs = design["rows"], design["cols"]
    if track == "id":
        condition = _log_uniform(rng, 2, 20)
        temperature = _log_uniform(rng, 0.65, 1.5)
    else:
        condition = _log_uniform(rng, 50, 320)
        temperature = (_log_uniform(rng, 0.2, 0.45)
                       if rng.random() < 0.5 else
                       _log_uniform(rng, 2.2, 4.0))
    weights = rng.normal(size=(rows, outputs)) * 1.5 / math.sqrt(rows)
    bias = rng.normal(scale=0.25, size=outputs)
    train = _ranking_rows(
        rng, 144, rows, outputs, condition, weights, bias, temperature)
    valid = _ranking_rows(
        rng, 208, rows, outputs, condition, weights, bias, temperature)
    anchor = _ranking_loss(
        valid.tolist(), rows, weights.tolist(), bias.tolist(), temperature)
    return _base(
        "ranking", track, design, initial, anchor, rng,
        {"temperature": _r(temperature), "train": _r(train),
         "validation": _r(valid)}, 18)


def _fourier_rows(rng, count, rows, weights, bias, output, input_scale):
    x = rng.normal(size=(count, rows)) * input_scale
    targets = np.sin(x @ weights + bias) @ output
    return np.column_stack([x, targets])


def _fourier(rng, track, design, initial):
    rows, hidden = design["rows"], design["cols"]
    input_scale = (_log_uniform(rng, 0.4, 1.4) if track == "id" else
                   _log_uniform(rng, 2.5, 6.0))
    frequency = (_log_uniform(rng, 0.8, 2.0) if track == "id" else
                 _log_uniform(rng, 3.0, 7.0))
    weights = rng.normal(
        scale=frequency / math.sqrt(rows), size=(rows, hidden))
    bias = rng.uniform(-math.pi, math.pi, size=hidden)
    output = rng.choice([-1.0, 1.0], size=hidden) / math.sqrt(hidden)
    train = _fourier_rows(
        rng, 144, rows, weights, bias, output, input_scale)
    valid = _fourier_rows(
        rng, 208, rows, weights, bias, output, input_scale)
    return _base(
        "fourier", track, design, initial, 0.0, rng,
        {"output": _r(output), "train": _r(train),
         "validation": _r(valid)}, 18)


BUILDERS = {
    "quadratic": _quadratic,
    "logistic": _logistic,
    "robust": _robust,
    "factorization": _factorization,
    "softmax": _softmax,
    "nonlinear": _nonlinear,
    "poisson": _poisson,
    "quantile": _quantile,
    "ranking": _ranking,
    "fourier": _fourier,
}


def _flatten(blocks):
    values = []
    for block in blocks:
        if block and isinstance(block[0], list):
            values.extend(value for row in block for value in row)
        else:
            values.extend(block)
    return values


def _calibrate_initial_gradient(task):
    """Give every family the same sampled first-gradient RMS distribution."""
    from bench.tasks.optimizer_generalization import evaluate

    gradient = evaluate.training_gradient(task, task["initial"], 1)
    values = _flatten(gradient)
    rms = math.sqrt(sum(value * value for value in values) / len(values))
    if not math.isfinite(rms) or rms <= 1e-12:
        raise RuntimeError(f"zero/nonfinite initial gradient for {task['family']}")
    target = float(task.pop("target_initial_gradient_rms"))
    scale = target / rms
    task["loss_scale"] = _r(scale, 15)
    task["reference_anchor"] = _r(task["reference_anchor"] * scale, 10)
    # Verify the public evaluator sees the requested scale and a valid anchor.
    scaled = _flatten(evaluate.training_gradient(task, task["initial"], 1))
    observed = math.sqrt(sum(value * value for value in scaled) / len(scaled))
    if abs(observed / target - 1.0) > 2e-7:
        raise RuntimeError("initial-gradient calibration drifted")
    denominator = (evaluate.validation_loss(task, task["initial"])
                   - task["reference_anchor"])
    if not math.isfinite(denominator) or denominator <= 1e-7:
        raise RuntimeError(f"invalid normalized denominator for {task['family']}")


def _source_dir():
    path = Path.home() / ".cache" / "text-opt-bm" / "optimizer-v7"
    path.mkdir(parents=True, exist_ok=True)
    # Reuse the preparation cache used in this repository's development runs.
    fallback = Path("/tmp/optimizer_data")
    for name, (url, expected) in SOURCE_SPECS.items():
        target = path / name
        if not target.exists() and (fallback / name).exists():
            target.write_bytes((fallback / name).read_bytes())
        if not target.exists():
            urllib.request.urlretrieve(url, target)
        if _sha(target) != expected:
            target.unlink(missing_ok=True)
            raise RuntimeError(f"source hash mismatch for {name}")
    return path


def _downsample(images):
    images = np.asarray(images, dtype=np.float64).reshape(-1, 28, 28)
    # 7x7 mean-pooled pixels retain spatial structure while keeping a complete
    # evaluation comfortably below the local runtime budget.
    result = images.reshape(-1, 7, 4, 7, 4).mean(axis=(2, 4)) / 255.0
    return result.reshape(-1, 49)


def _idx(path):
    raw = gzip.decompress(path.read_bytes())
    magic, count = struct.unpack(">II", raw[:8])
    if magic == 2049:
        return np.frombuffer(raw, dtype=np.uint8, offset=8, count=count)
    if magic != 2051:
        raise RuntimeError("bad Fashion-MNIST IDX magic")
    rows, cols = struct.unpack(">II", raw[8:16])
    return np.frombuffer(raw, dtype=np.uint8, offset=16).reshape(count, rows, cols)


def _real_sources():
    global _SOURCE_CACHE
    if _SOURCE_CACHE is not None:
        return _SOURCE_CACHE
    source = _source_dir()
    mnist = np.load(source / "mnist.npz")
    fashion_x = _idx(source / "fashion-train-images.gz")
    fashion_y = _idx(source / "fashion-train-labels.gz")
    # Fixed standardization is estimated from training data only.
    mnist_train = _downsample(mnist["x_train"])
    mnist_test = _downsample(mnist["x_test"])
    fashion = _downsample(fashion_x)
    vocab_chars = sorted(set(
        (source / "tinyshakespeare.txt").read_text(errors="replace") +
        (source / "alice.txt").read_text(errors="replace")))
    # Collapse rare Unicode and control characters into a stable printable
    # vocabulary.  Both corpora are naturally generated public-domain text.
    vocab_chars = [ch for ch in vocab_chars if ch == "\n" or 32 <= ord(ch) < 127]
    vocab = {ch: index for index, ch in enumerate(vocab_chars)}

    def encode(path):
        text = path.read_text(errors="replace")
        return np.asarray([vocab[ch] for ch in text if ch in vocab], dtype=np.int64)

    _SOURCE_CACHE = {
        "mnist_train": (mnist_train, np.asarray(mnist["y_train"], dtype=np.int64)),
        "mnist_test": (mnist_test, np.asarray(mnist["y_test"], dtype=np.int64)),
        "fashion": (fashion, np.asarray(fashion_y, dtype=np.int64)),
        "shakespeare": encode(source / "tinyshakespeare.txt"),
        "alice": encode(source / "alice.txt"),
        "vocab_size": len(vocab),
    }
    return _SOURCE_CACHE


def _xavier(rng, rows, columns):
    return rng.normal(0.0, math.sqrt(2.0 / (rows + columns)),
                      size=(rows, columns))


def _real_initial(rng, family, hidden, vocab_size=None, embedding=12):
    if family in ("image_mlp", "image_autoencoder"):
        outputs = 10 if family == "image_mlp" else 49
        return [_r(_xavier(rng, 49, hidden), 7), [0.0] * hidden,
                _r(_xavier(rng, hidden, outputs), 7), [0.0] * outputs]
    if family == "image_deep_mlp":
        return [_r(_xavier(rng, 49, hidden), 7), [0.0] * hidden,
                _r(_xavier(rng, hidden, hidden), 7), [0.0] * hidden,
                _r(_xavier(rng, hidden, hidden), 7), [0.0] * hidden,
                _r(_xavier(rng, hidden, 10), 7), [0.0] * 10]
    if family == "image_conv":
        filters = max(4, hidden // 2)
        return [_r(_xavier(rng, 9, filters), 7), [0.0] * filters,
                _r(_xavier(rng, filters, 10), 7), [0.0] * 10]
    if family == "image_residual":
        return [_r(_xavier(rng, 49, hidden), 7), [0.0] * hidden,
                _r(_xavier(rng, hidden, hidden), 7), [0.0] * hidden,
                _r(_xavier(rng, hidden, 10), 7), [0.0] * 10]
    if family == "image_gated_mlp":
        return [_r(_xavier(rng, 49, hidden), 7), [0.0] * hidden,
                _r(_xavier(rng, 49, hidden), 7), [0.0] * hidden,
                _r(_xavier(rng, hidden, 10), 7), [0.0] * 10]
    if family == "image_bottleneck":
        bottleneck = max(4, hidden // 2)
        return [_r(_xavier(rng, 49, hidden), 7), [0.0] * hidden,
                _r(_xavier(rng, hidden, bottleneck), 7), [0.0] * bottleneck,
                _r(_xavier(rng, bottleneck, hidden), 7), [0.0] * hidden,
                _r(_xavier(rng, hidden, 10), 7), [0.0] * 10]
    recurrent = _xavier(rng, hidden, hidden)
    recurrent *= 0.8 / max(1e-8, np.linalg.norm(recurrent, ord=2))
    return [_r(rng.normal(0.0, 0.05, size=(vocab_size, embedding)), 7),
            _r(_xavier(rng, embedding, hidden), 7), _r(recurrent, 7),
            [0.0] * hidden, _r(_xavier(rng, hidden, vocab_size), 7),
            [0.0] * vocab_size]


def _sample_image_rows(rng, split, track, count):
    sources = _real_sources()
    if track == "ood":
        x, y = sources["fashion"]
        start, stop = 40_000, len(y)
    elif split == "test":
        x, y = sources["mnist_test"]
        start, stop = 0, len(y)
    else:
        x, y = sources["mnist_train"]
        # Training and validation pools are disjoint by construction.
        start, stop = ((0, 40_000) if split == "train" else (40_000, 60_000))
    indices = rng.choice(np.arange(start, stop), size=count, replace=False)
    return x[indices], y[indices], indices


def _sample_text_rows(rng, split, track, count, context):
    sources = _real_sources()
    corpus = sources["alice"] if track == "ood" else sources["shakespeare"]
    if track == "ood" or split == "test":
        low, high = len(corpus) * 3 // 4, len(corpus) - context - 1
    elif split == "validation":
        low, high = len(corpus) // 2, len(corpus) * 3 // 4
    else:
        low, high = 0, len(corpus) // 2
    # Non-overlapping contexts eliminate exact or partial train/validation
    # sequence leakage inside a workload.
    candidates = np.arange(low, high, context + 1)
    starts = rng.choice(candidates, size=count, replace=False)
    x = np.stack([corpus[index:index + context] for index in starts])
    y = np.asarray([corpus[index + context] for index in starts])
    return x, y, starts


def _baseline_reference(task):
    """Empirical best-loss envelope from tuned SGD and Adam.

    References are computed on the task's validation data, never from planted
    parameters.  This mirrors TaskSet's empirical best-known normalization and
    is intentionally allowed to be beaten by new methods.
    """
    from bench.tasks.optimizer_generalization import real_workloads

    best = real_workloads.validation_loss(task, task["initial"])
    checkpoints = set(max(1, round(task["horizon"] * i / 8)) for i in range(1, 9))
    for method in ("sgd", "adam"):
        for lr in (0.003, 0.01, 0.03, 0.1, 0.3):
            params = [np.asarray(block, dtype=np.float64) for block in task["initial"]]
            first = [np.zeros_like(value) for value in params]
            second = [np.zeros_like(value) for value in params]
            for step in range(1, task["horizon"] + 1):
                gradients = [np.asarray(value) for value in
                             real_workloads.training_gradient(task, params, step)]
                if method == "sgd":
                    first = [0.9 * m + gradient for m, gradient in zip(first, gradients)]
                    params = [value - lr * m for value, m in zip(params, first)]
                else:
                    first = [0.9 * m + 0.1 * gradient
                             for m, gradient in zip(first, gradients)]
                    second = [0.999 * v + 0.001 * gradient * gradient
                              for v, gradient in zip(second, gradients)]
                    params = [value - lr * (m / (1.0 - 0.9 ** step)) /
                              (np.sqrt(v / (1.0 - 0.999 ** step)) + 1e-8)
                              for value, m, v in zip(params, first, second)]
                if step in checkpoints:
                    loss = real_workloads.validation_loss(task, params)
                    if math.isfinite(loss):
                        best = min(best, loss)
    return float(best)


def _real_task(family, split, track, index, seed):
    rng = np.random.default_rng(seed + index * 7919)
    hidden = ((8, 12, 16, 20)[index % 4] if track == "id"
              else (6, 24, 28, 32)[index % 4])
    activation = ("tanh" if index % 2 == 0 else "relu")
    horizon = ((160, 224, 288, 352)[index % 4] if track == "id"
               else (128, 256, 384, 512)[index % 4])
    if family in ("image_mlp", "image_deep_mlp", "image_conv", "image_autoencoder",
                  "image_residual", "image_gated_mlp", "image_bottleneck"):
        all_x, all_y, all_ids = _sample_image_rows(rng, split, track, 512)
        train_x, validation_x = all_x[:256], all_x[256:]
        train_y, validation_y = all_y[:256], all_y[256:]
        train_ids, validation_ids = all_ids[:256], all_ids[256:]
        if family == "image_autoencoder":
            train_y = np.zeros(len(train_x), dtype=np.int64)
            validation_y = np.zeros(len(validation_x), dtype=np.int64)
        initial = _real_initial(rng, family, hidden)
        payload = {"dataset": ("fashion_mnist" if track == "ood" else "mnist"),
                   "activation": activation,
                   "train_x": _r(train_x, 6), "train_y": train_y.tolist(),
                   "train_example_ids": train_ids.tolist(),
                   "validation_x": _r(validation_x, 6),
                   "validation_y": validation_y.tolist(),
                   "validation_example_ids": validation_ids.tolist()}
    else:
        context = (8, 12, 16, 24)[index % 4]
        all_x, all_y, all_ids = _sample_text_rows(
            rng, split, track, 768, context)
        train_x, validation_x = all_x[:384], all_x[384:]
        train_y, validation_y = all_y[:384], all_y[384:]
        train_ids, validation_ids = all_ids[:384], all_ids[384:]
        vocab_size = _real_sources()["vocab_size"]
        initial = _real_initial(rng, family, hidden, vocab_size)
        payload = {"dataset": ("alice" if track == "ood" else "shakespeare"),
                   "context": context, "train_x": train_x.tolist(),
                   "train_y": train_y.tolist(),
                   "train_example_ids": train_ids.tolist(),
                   "validation_x": validation_x.tolist(),
                   "validation_y": validation_y.tolist(),
                   "validation_example_ids": validation_ids.tolist()}
    task = {"suite": "real", "family": family, "track": track,
            "initial": initial,
            "shapes": [list(np.asarray(block).shape) for block in initial],
            "horizon": horizon, "batch_size": 32,
            "batch_seed": _batch_seed(rng), "loss_scale": 1.0,
            "payload": payload,
            "reference_provenance": "empirical tuned SGD/Adam envelope"}
    task["reference_anchor"] = _r(_baseline_reference(task), 10)
    initial_loss = __import__(
        "bench.tasks.optimizer_generalization.real_workloads",
        fromlist=["validation_loss"]).validation_loss(task, task["initial"])
    if initial_loss - task["reference_anchor"] <= 1e-7:
        raise RuntimeError("real workload baseline envelope made no progress")
    return task


def _real_split(name):
    tasks = []
    families = REAL_FAMILIES + (TEST_ONLY_REAL_FAMILIES if name == "test" else ())
    for family_index, family in enumerate(families):
        for track in ("id", "ood"):
            for index in range(REAL_COUNTS[name][track]):
                tasks.append(_real_task(
                    family, name, track, index,
                    SPLITS[name]["seed"] + 1_000_000 * (family_index + 1) +
                    (500_000 if track == "ood" else 0)))
    return tasks


def build_split(name):
    settings = SPLITS[name]
    families = list(DEVELOPMENT_FAMILIES)
    if name == "test":
        families.extend(TEST_ONLY_FAMILIES)
    designs = {
        "id": _interface_designs(
            settings["seed"], "id", settings["id_per_family"]),
        "ood": _interface_designs(
            settings["seed"], "ood", settings["ood_per_family"]),
    }
    tasks = []
    for family_index, family in enumerate(families):
        # Family-specific geometry gets an independent stream; observables come
        # exclusively from the shared designs above.
        rng = np.random.default_rng(settings["seed"] + 10_000 * (family_index + 1))
        for track in ("id", "ood"):
            for design in designs[track]:
                initial = _initial_parameters(rng, design)
                failure = None
                for _attempt in range(16):
                    task = BUILDERS[family](rng, track, design, initial)
                    task["suite"] = "analytic"
                    try:
                        _calibrate_initial_gradient(task)
                        failure = None
                        break
                    except RuntimeError as exc:
                        failure = exc
                if failure is not None:
                    raise RuntimeError(
                        f"could not generate stable {family}/{track} workload") from failure
                tasks.append(task)
    tasks.extend(_real_split(name))
    order_rng = np.random.default_rng(settings["seed"] ^ 0x5EED)
    order_rng.shuffle(tasks)
    for index, task in enumerate(tasks):
        task["task_id"] = f"{name}-{index:04d}"
    return {"schema": SCHEMA, "protocol": PROTOCOL,
            "split": name, "tasks": tasks}


def _block_rms(blocks):
    values = _flatten(blocks)
    return math.sqrt(sum(value * value for value in values) / len(values))


def observable_signature_redteam(payload, digits=10):
    """Measure finite-support label leakage from public scalar signatures.

    Initial tensors themselves must differ across workloads, so this red team
    intentionally considers only the aggregate observables an optimizer can
    cheaply dispatch on: block shapes, initial RMS, and first-gradient RMS.
    RMS values are rounded at a precision far finer than the generated ranges.
    For each signature, the Bayes accuracy is the accuracy of an oracle lookup
    table fit to this same finite split; zero advantage is therefore a strong
    balance check rather than an out-of-sample classifier estimate.
    """
    from bench.tasks.optimizer_generalization import evaluate

    # Shape and scale balance is an anti-tag requirement for the standardized
    # analytic tier.  Real neural workloads intentionally expose natural model
    # topology; treating architecture as secret would invalidate useful
    # layer-aware optimizer research.
    tasks = [task for task in payload["tasks"]
             if task.get("suite", "analytic") == "analytic"]
    records = []
    for task in tasks:
        shape = tuple(tuple(value) for value in task["shapes"])
        initial_rms = round(_block_rms(task["initial"]), digits)
        gradient_rms = round(_block_rms(
            evaluate.training_gradient(task, task["initial"], 1)), digits)
        records.append({
            "shape": shape,
            "shape_initial_rms": (shape, initial_rms),
            "shape_initial_gradient_rms": (
                shape, initial_rms, gradient_rms),
            "family": task["family"],
            "track": task["track"],
            "family_track": task["family"] + "/" + task["track"],
        })

    def classification(signature_name, target_name):
        overall = {}
        buckets = {}
        for record in records:
            label = record[target_name]
            overall[label] = overall.get(label, 0) + 1
            signature = record[signature_name]
            counts = buckets.setdefault(signature, {})
            counts[label] = counts.get(label, 0) + 1
        majority = max(overall.values()) / len(records)
        oracle = sum(max(counts.values()) for counts in buckets.values()) / len(records)
        return {
            "labels": len(overall),
            "majority_accuracy": _r(majority, 10),
            "signature_oracle_accuracy": _r(oracle, 10),
            "advantage": _r(oracle - majority, 10),
        }

    levels = {}
    for signature_name in (
            "shape", "shape_initial_rms", "shape_initial_gradient_rms"):
        bucket_sizes = {}
        for record in records:
            signature = record[signature_name]
            bucket_sizes[signature] = bucket_sizes.get(signature, 0) + 1
        levels[signature_name] = {
            "unique_signatures": len(bucket_sizes),
            "minimum_bucket_size": min(bucket_sizes.values()),
            "maximum_bucket_size": max(bucket_sizes.values()),
            "family": classification(signature_name, "family"),
            "track": classification(signature_name, "track"),
            "family_track": classification(signature_name, "family_track"),
        }
    maximum_advantage = max(
        result["advantage"]
        for level in levels.values()
        for result in (level["family"], level["track"], level["family_track"]))
    return {
        "rms_decimal_places": digits,
        "n_workloads": len(records),
        "levels": levels,
        "maximum_oracle_advantage": maximum_advantage,
        "passed": maximum_advantage <= 1e-10,
    }


def real_architecture_signature_audit(payload, digits=10):
    """Quantify, rather than pretend to hide, legal topology dispatch.

    Natural neural parameter shapes reveal architecture.  The protocol deals
    with this by matching baseline expressiveness and scoring unseen
    architectures separately; this audit makes the available dispatch signal
    explicit in every prepared split.
    """
    from bench.tasks.optimizer_generalization import real_workloads

    tasks = [task for task in payload["tasks"] if task.get("suite") == "real"]
    if not tasks:
        return {"n_workloads": 0, "levels": {}}
    records = []
    for task in tasks:
        shape = tuple(tuple(value) for value in task["shapes"])
        initial_rms = round(_block_rms(task["initial"]), digits)
        gradient_rms = round(_block_rms(
            real_workloads.training_gradient(task, task["initial"], 1)), digits)
        records.append({
            "shape": shape,
            "shape_initial_rms": (shape, initial_rms),
            "shape_initial_gradient_rms": (shape, initial_rms, gradient_rms),
            "family": task["family"],
            "generalization_cell": (
                "unseen" if task["family"] in TEST_ONLY_REAL_FAMILIES
                else "known"),
        })

    def classification(signature_name, target_name):
        overall, buckets = {}, {}
        for record in records:
            label = record[target_name]
            overall[label] = overall.get(label, 0) + 1
            counts = buckets.setdefault(record[signature_name], {})
            counts[label] = counts.get(label, 0) + 1
        majority = max(overall.values()) / len(records)
        oracle = sum(max(values.values()) for values in buckets.values()) / len(records)
        return {"labels": len(overall), "majority_accuracy": _r(majority, 10),
                "signature_oracle_accuracy": _r(oracle, 10),
                "advantage": _r(oracle - majority, 10)}

    levels = {}
    for signature_name in (
            "shape", "shape_initial_rms", "shape_initial_gradient_rms"):
        buckets = {}
        for record in records:
            buckets[record[signature_name]] = buckets.get(
                record[signature_name], 0) + 1
        levels[signature_name] = {
            "unique_signatures": len(buckets),
            "minimum_bucket_size": min(buckets.values()),
            "maximum_bucket_size": max(buckets.values()),
            "family": classification(signature_name, "family"),
            "generalization_cell": classification(
                signature_name, "generalization_cell"),
        }
    return {
        "n_workloads": len(records),
        "interpretation": (
            "topology dispatch is legal and expected; conditional baselines "
            "must receive the same information"),
        "levels": levels,
    }


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_artifacts(output=DATA):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    train = build_split("train")
    validation = build_split("validation")
    test = build_split("test")
    (output / "train.json").write_text(
        json.dumps(train, separators=(",", ":")) + "\n")
    heldout.write(output / "heldout_val.bin", validation)
    heldout.write(output / "heldout_test.bin", test)
    paths = [output / "train.json", output / "heldout_val.bin",
             output / "heldout_test.bin"]
    manifest = {
        "schema": SCHEMA,
        "protocol": PROTOCOL,
        "seeds": {name: settings["seed"] for name, settings in SPLITS.items()},
        "development_families": list(DEVELOPMENT_FAMILIES),
        "test_only_families": list(TEST_ONLY_FAMILIES),
        "real_families": list(REAL_FAMILIES),
        "test_only_real_families": list(TEST_ONLY_REAL_FAMILIES),
        "sealed_test_only_families": len(TEST_ONLY_FAMILIES),
        "sealed_test_known_architecture_score_weight": 0.5,
        "sealed_test_unseen_architecture_score_weight": 0.5,
        "tracks": {"train": ["id"],
                   "validation": ["id", "ood"],
                   "test": ["id", "ood"]},
        "counts": {"train": len(train["tasks"]),
                   "validation": len(validation["tasks"]),
                   "test": len(test["tasks"])},
        "real_counts": {name: {
            track: REAL_COUNTS[name][track] * (
                len(REAL_FAMILIES) +
                (len(TEST_ONLY_REAL_FAMILIES) if name == "test" else 0))
            for track in ("id", "ood")} for name in REAL_COUNTS},
        "analytic_per_family_track": {"validation": 24, "test": 28},
        "real_per_family_track": {"validation": 16, "test": 8},
        "parameter_interface": "matrix-plus-vector for every workload",
        "real_parameter_interface": "natural multi-block neural parameters",
        "real_sources": {name: {"url": url, "sha256": digest}
                         for name, (url, digest) in SOURCE_SPECS.items()},
        "reference_policy": {
            "analytic": "protocol-6 fixed references (diagnostic tier)",
            "real": "empirical best validation loss over tuned SGD and Adam",
        },
        "factorization_minibatches": (
            "deterministic randomized edge cover plus random fill"),
        "softmax_parameterization": (
            "hidden positive per-workload class logit scales"),
        "observable_signature_redteam": {
            "train": observable_signature_redteam(train),
            "validation": observable_signature_redteam(validation),
            "test": observable_signature_redteam(test),
        },
        "real_architecture_signature_audit": {
            "train": real_architecture_signature_audit(train),
            "validation": real_architecture_signature_audit(validation),
            "test": real_architecture_signature_audit(test),
        },
        "sha256": {path.name: _sha(path) for path in paths},
    }
    (output / "data_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


if __name__ == "__main__":
    print(json.dumps(write_artifacts(), indent=2, sort_keys=True))
