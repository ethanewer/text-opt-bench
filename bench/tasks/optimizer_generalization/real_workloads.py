"""Small, real-data neural workloads for optimizer protocol 9.

The models are deliberately implemented with NumPy rather than an autograd
framework.  This keeps the gradient contract identical on CPU, CUDA, and MPS,
and makes every stochastic choice evaluator-owned and reproducible.
"""

from __future__ import annotations

import math

import numpy as np


REAL_FAMILIES = frozenset((
    "image_mlp", "image_deep_mlp", "image_conv", "image_autoencoder", "char_lm",
    "image_residual", "image_gated_mlp", "image_bottleneck"))


def _arrays(blocks):
    return [np.asarray(block, dtype=np.float64) for block in blocks]


def _softmax(logits):
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    values = np.exp(np.clip(shifted, -60.0, 0.0))
    return values / np.sum(values, axis=1, keepdims=True)


def _indices(task, step):
    count = len(task["payload"]["train_y"])
    size = min(int(task["batch_size"]), count)
    # Sampling with replacement is the TaskSet convention and avoids exposing
    # epoch boundaries to candidates.
    rng = np.random.default_rng(
        (int(task["batch_seed"]) + 0x9E3779B9 * step) & 0xFFFFFFFF)
    return rng.integers(0, count, size=size)


def _activation(value, kind):
    if kind == "relu":
        return np.maximum(value, 0.0)
    return np.tanh(value)


def _activation_grad(pre, activated, kind):
    if kind == "relu":
        return (pre > 0.0).astype(np.float64)
    return 1.0 - activated * activated


def _classifier_loss(blocks, x, y, activation):
    w1, b1, w2, b2 = _arrays(blocks)
    hidden = _activation(x @ w1 + b1, activation)
    probabilities = _softmax(hidden @ w2 + b2)
    return float(-np.log(np.maximum(probabilities[np.arange(len(y)), y],
                                    1e-30)).mean())


def _classifier_gradient(blocks, x, y, activation):
    w1, b1, w2, b2 = _arrays(blocks)
    pre = x @ w1 + b1
    hidden = _activation(pre, activation)
    probabilities = _softmax(hidden @ w2 + b2)
    probabilities[np.arange(len(y)), y] -= 1.0
    probabilities /= len(y)
    gw2 = hidden.T @ probabilities
    gb2 = probabilities.sum(axis=0)
    dh = (probabilities @ w2.T) * _activation_grad(pre, hidden, activation)
    return [x.T @ dh, dh.sum(axis=0), gw2, gb2]


def _deep_forward(blocks, x, activation):
    w0, b0, w1, b1, w2, b2, output, output_bias = _arrays(blocks)
    pre0 = x @ w0 + b0
    hidden0 = _activation(pre0, activation)
    pre1 = hidden0 @ w1 + b1
    hidden1 = _activation(pre1, activation)
    pre2 = hidden1 @ w2 + b2
    hidden2 = _activation(pre2, activation)
    probabilities = _softmax(hidden2 @ output + output_bias)
    return (pre0, hidden0, pre1, hidden1, pre2, hidden2, probabilities)


def _deep_loss(blocks, x, y, activation):
    *_, probabilities = _deep_forward(blocks, x, activation)
    return float(-np.log(np.maximum(
        probabilities[np.arange(len(y)), y], 1e-30)).mean())


def _deep_gradient(blocks, x, y, activation):
    w0, b0, w1, b1, w2, b2, output, output_bias = _arrays(blocks)
    pre0, hidden0, pre1, hidden1, pre2, hidden2, probabilities = (
        _deep_forward(blocks, x, activation))
    probabilities[np.arange(len(y)), y] -= 1.0
    probabilities /= len(y)
    goutput = hidden2.T @ probabilities
    goutput_bias = probabilities.sum(axis=0)
    dh2 = (probabilities @ output.T) * _activation_grad(
        pre2, hidden2, activation)
    gw2, gb2 = hidden1.T @ dh2, dh2.sum(axis=0)
    dh1 = (dh2 @ w2.T) * _activation_grad(pre1, hidden1, activation)
    gw1, gb1 = hidden0.T @ dh1, dh1.sum(axis=0)
    dh0 = (dh1 @ w1.T) * _activation_grad(pre0, hidden0, activation)
    return [x.T @ dh0, dh0.sum(axis=0), gw1, gb1, gw2, gb2,
            goutput, goutput_bias]


def _residual_forward(blocks, x):
    w0, b0, w1, b1, output, output_bias = _arrays(blocks)
    first = np.tanh(x @ w0 + b0)
    residual_pre = first @ w1 + b1
    hidden = first + 0.5 * np.tanh(residual_pre)
    return first, residual_pre, hidden, _softmax(hidden @ output + output_bias)


def _residual_loss(blocks, x, y):
    _, _, _, probabilities = _residual_forward(blocks, x)
    return float(-np.log(np.maximum(
        probabilities[np.arange(len(y)), y], 1e-30)).mean())


def _residual_gradient(blocks, x, y):
    w0, b0, w1, b1, output, output_bias = _arrays(blocks)
    first, residual_pre, hidden, probabilities = _residual_forward(blocks, x)
    probabilities[np.arange(len(y)), y] -= 1.0
    probabilities /= len(y)
    goutput = hidden.T @ probabilities
    goutput_bias = probabilities.sum(axis=0)
    dhidden = probabilities @ output.T
    dresidual = 0.5 * dhidden * (1.0 - np.tanh(residual_pre) ** 2)
    gw1 = first.T @ dresidual
    gb1 = dresidual.sum(axis=0)
    dfirst = dhidden + dresidual @ w1.T
    dfirst_pre = dfirst * (1.0 - first * first)
    return [x.T @ dfirst_pre, dfirst_pre.sum(axis=0), gw1, gb1,
            goutput, goutput_bias]


def _gated_forward(blocks, x):
    value_weight, value_bias, gate_weight, gate_bias, output, output_bias = (
        _arrays(blocks))
    value = np.tanh(x @ value_weight + value_bias)
    gate_pre = x @ gate_weight + gate_bias
    gate = 1.0 / (1.0 + np.exp(np.clip(-gate_pre, -60.0, 60.0)))
    hidden = value * gate
    return value, gate, hidden, _softmax(hidden @ output + output_bias)


def _gated_loss(blocks, x, y):
    _, _, _, probabilities = _gated_forward(blocks, x)
    return float(-np.log(np.maximum(
        probabilities[np.arange(len(y)), y], 1e-30)).mean())


def _gated_gradient(blocks, x, y):
    value_weight, value_bias, gate_weight, gate_bias, output, output_bias = (
        _arrays(blocks))
    value, gate, hidden, probabilities = _gated_forward(blocks, x)
    probabilities[np.arange(len(y)), y] -= 1.0
    probabilities /= len(y)
    goutput = hidden.T @ probabilities
    goutput_bias = probabilities.sum(axis=0)
    dhidden = probabilities @ output.T
    dvalue = dhidden * gate * (1.0 - value * value)
    dgate = dhidden * value * gate * (1.0 - gate)
    return [x.T @ dvalue, dvalue.sum(axis=0),
            x.T @ dgate, dgate.sum(axis=0), goutput, goutput_bias]


def _bottleneck_forward(blocks, x):
    w0, b0, down, down_bias, up, up_bias, output, output_bias = _arrays(blocks)
    first = np.tanh(x @ w0 + b0)
    middle = np.tanh(first @ down + down_bias)
    up_pre = middle @ up + up_bias
    hidden = first + 0.5 * np.tanh(up_pre)
    return first, middle, up_pre, hidden, _softmax(hidden @ output + output_bias)


def _bottleneck_loss(blocks, x, y):
    *_, probabilities = _bottleneck_forward(blocks, x)
    return float(-np.log(np.maximum(
        probabilities[np.arange(len(y)), y], 1e-30)).mean())


def _bottleneck_gradient(blocks, x, y):
    w0, b0, down, down_bias, up, up_bias, output, output_bias = _arrays(blocks)
    first, middle, up_pre, hidden, probabilities = _bottleneck_forward(blocks, x)
    probabilities[np.arange(len(y)), y] -= 1.0
    probabilities /= len(y)
    goutput = hidden.T @ probabilities
    goutput_bias = probabilities.sum(axis=0)
    dhidden = probabilities @ output.T
    dup = 0.5 * dhidden * (1.0 - np.tanh(up_pre) ** 2)
    gup, gup_bias = middle.T @ dup, dup.sum(axis=0)
    dmiddle = (dup @ up.T) * (1.0 - middle * middle)
    gdown, gdown_bias = first.T @ dmiddle, dmiddle.sum(axis=0)
    dfirst = dhidden + dmiddle @ down.T
    dfirst_pre = dfirst * (1.0 - first * first)
    return [x.T @ dfirst_pre, dfirst_pre.sum(axis=0),
            gdown, gdown_bias, gup, gup_bias, goutput, goutput_bias]


def _image_patches(x):
    images = x.reshape(-1, 7, 7)
    return np.stack([images[:, i:i + 3, j:j + 3].reshape(len(x), 9)
                     for i in range(5) for j in range(5)], axis=1)


def _conv_forward(blocks, x):
    kernel, bias, output, output_bias = _arrays(blocks)
    patches = _image_patches(x)
    pre = patches @ kernel + bias
    hidden = np.maximum(pre, 0.0)
    pooled = hidden.mean(axis=1)
    return patches, pre, pooled, _softmax(pooled @ output + output_bias)


def _conv_loss(blocks, x, y):
    _, _, _, probabilities = _conv_forward(blocks, x)
    return float(-np.log(np.maximum(
        probabilities[np.arange(len(y)), y], 1e-30)).mean())


def _conv_gradient(blocks, x, y):
    kernel, bias, output, output_bias = _arrays(blocks)
    patches, pre, pooled, probabilities = _conv_forward(blocks, x)
    probabilities[np.arange(len(y)), y] -= 1.0
    probabilities /= len(y)
    goutput = pooled.T @ probabilities
    goutput_bias = probabilities.sum(axis=0)
    dpool = probabilities @ output.T
    dpre = (dpool[:, None, :] / pre.shape[1]) * (pre > 0.0)
    gkernel = np.einsum("npi,npf->if", patches, dpre)
    return [gkernel, dpre.sum(axis=(0, 1)), goutput, goutput_bias]


def _autoencoder_loss(blocks, x, activation):
    w1, b1, w2, b2 = _arrays(blocks)
    hidden = _activation(x @ w1 + b1, activation)
    reconstruction = hidden @ w2 + b2
    return float(0.5 * np.square(reconstruction - x).mean())


def _autoencoder_gradient(blocks, x, activation):
    w1, b1, w2, b2 = _arrays(blocks)
    pre = x @ w1 + b1
    hidden = _activation(pre, activation)
    reconstruction = hidden @ w2 + b2
    dr = (reconstruction - x) / reconstruction.size
    gw2 = hidden.T @ dr
    gb2 = dr.sum(axis=0)
    dh = (dr @ w2.T) * _activation_grad(pre, hidden, activation)
    return [x.T @ dh, dh.sum(axis=0), gw2, gb2]


def _lm_forward(blocks, contexts):
    embedding, input_weight, recurrent_weight, bias, output, output_bias = _arrays(blocks)
    states = [np.zeros((len(contexts), recurrent_weight.shape[0]))]
    inputs = []
    for position in range(contexts.shape[1]):
        encoded = embedding[contexts[:, position]]
        inputs.append(encoded)
        states.append(np.tanh(encoded @ input_weight +
                              states[-1] @ recurrent_weight + bias))
    return inputs, states, _softmax(states[-1] @ output + output_bias)


def _lm_loss(blocks, contexts, targets):
    _, _, probabilities = _lm_forward(blocks, contexts)
    return float(-np.log(np.maximum(
        probabilities[np.arange(len(targets)), targets], 1e-30)).mean())


def _lm_gradient(blocks, contexts, targets):
    embedding, input_weight, recurrent_weight, bias, output, output_bias = _arrays(blocks)
    inputs, states, probabilities = _lm_forward(blocks, contexts)
    probabilities[np.arange(len(targets)), targets] -= 1.0
    probabilities /= len(targets)
    goutput = states[-1].T @ probabilities
    goutput_bias = probabilities.sum(axis=0)
    dh = probabilities @ output.T
    gembedding = np.zeros_like(embedding)
    ginput = np.zeros_like(input_weight)
    grecurrent = np.zeros_like(recurrent_weight)
    gbias = np.zeros_like(bias)
    for position in range(contexts.shape[1] - 1, -1, -1):
        dz = dh * (1.0 - states[position + 1] ** 2)
        ginput += inputs[position].T @ dz
        grecurrent += states[position].T @ dz
        gbias += dz.sum(axis=0)
        np.add.at(gembedding, contexts[:, position], dz @ input_weight.T)
        dh = dz @ recurrent_weight.T
    return [gembedding, ginput, grecurrent, gbias, goutput, goutput_bias]


def validation_loss(task, blocks):
    payload = task["payload"]
    x = np.asarray(payload["validation_x"], dtype=(
        np.int64 if task["family"] == "char_lm" else np.float64))
    y = np.asarray(payload["validation_y"], dtype=np.int64)
    if task["family"] == "image_mlp":
        raw = _classifier_loss(blocks, x, y, payload["activation"])
    elif task["family"] == "image_deep_mlp":
        raw = _deep_loss(blocks, x, y, payload["activation"])
    elif task["family"] == "image_residual":
        raw = _residual_loss(blocks, x, y)
    elif task["family"] == "image_gated_mlp":
        raw = _gated_loss(blocks, x, y)
    elif task["family"] == "image_bottleneck":
        raw = _bottleneck_loss(blocks, x, y)
    elif task["family"] == "image_conv":
        raw = _conv_loss(blocks, x, y)
    elif task["family"] == "image_autoencoder":
        raw = _autoencoder_loss(blocks, x, payload["activation"])
    elif task["family"] == "char_lm":
        raw = _lm_loss(blocks, x, y)
    else:
        raise ValueError("unknown real optimizer workload family")
    return raw * float(task.get("loss_scale", 1.0))


def training_gradient(task, blocks, step):
    payload = task["payload"]
    chosen = _indices(task, step)
    x = np.asarray(payload["train_x"], dtype=(
        np.int64 if task["family"] == "char_lm" else np.float64))[chosen]
    y = np.asarray(payload["train_y"], dtype=np.int64)[chosen]
    if task["family"] == "image_mlp":
        result = _classifier_gradient(blocks, x, y, payload["activation"])
    elif task["family"] == "image_deep_mlp":
        result = _deep_gradient(blocks, x, y, payload["activation"])
    elif task["family"] == "image_residual":
        result = _residual_gradient(blocks, x, y)
    elif task["family"] == "image_gated_mlp":
        result = _gated_gradient(blocks, x, y)
    elif task["family"] == "image_bottleneck":
        result = _bottleneck_gradient(blocks, x, y)
    elif task["family"] == "image_conv":
        result = _conv_gradient(blocks, x, y)
    elif task["family"] == "image_autoencoder":
        result = _autoencoder_gradient(blocks, x, payload["activation"])
    elif task["family"] == "char_lm":
        result = _lm_gradient(blocks, x, y)
    else:
        raise ValueError("unknown real optimizer workload family")
    scale = float(task.get("loss_scale", 1.0))
    return [(gradient * scale).tolist() for gradient in result]


def parameter_count(task):
    return sum(math.prod(shape) for shape in task["shapes"])
