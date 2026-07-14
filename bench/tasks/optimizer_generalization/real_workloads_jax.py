"""CPU-only JAX kernels for optimizer protocol 9 real workloads.

Only evaluator-owned loss and gradient math is compiled. Candidate optimizers
remain ordinary independent programs and are never traced by JAX.
"""

from __future__ import annotations

import os

# Pin the backend before importing JAX. This task must never acquire MPS/CUDA.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "true")

import jax
import jax.numpy as jnp
import numpy as np


jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platforms", "cpu")

_TASK_ARRAYS = {}


def _cross_entropy(logits, targets):
    return -jnp.mean(jax.nn.log_softmax(logits, axis=-1)[
        jnp.arange(targets.shape[0]), targets])


def _mlp_loss(params, x, y, relu):
    w1, b1, w2, b2 = params
    pre = x @ w1 + b1
    hidden = jnp.maximum(pre, 0.0) if relu else jnp.tanh(pre)
    return _cross_entropy(hidden @ w2 + b2, y)


def _deep_loss(params, x, y, relu):
    w0, b0, w1, b1, w2, b2, output, output_bias = params
    activation = lambda value: jnp.maximum(value, 0.0) if relu else jnp.tanh(value)
    hidden = activation(x @ w0 + b0)
    hidden = activation(hidden @ w1 + b1)
    hidden = activation(hidden @ w2 + b2)
    return _cross_entropy(hidden @ output + output_bias, y)


def _autoencoder_loss(params, x, _y, relu):
    w1, b1, w2, b2 = params
    pre = x @ w1 + b1
    hidden = jnp.maximum(pre, 0.0) if relu else jnp.tanh(pre)
    reconstruction = hidden @ w2 + b2
    return 0.5 * jnp.mean(jnp.square(reconstruction - x))


def _conv_loss(params, x, y):
    kernel, bias, output, output_bias = params
    images = x.reshape((-1, 7, 7, 1))
    filters = kernel.reshape((3, 3, 1, kernel.shape[1]))
    hidden = jax.nn.relu(jax.lax.conv_general_dilated(
        images, filters, (1, 1), "VALID",
        dimension_numbers=("NHWC", "HWIO", "NHWC")) + bias)
    pooled = jnp.mean(hidden, axis=(1, 2))
    return _cross_entropy(pooled @ output + output_bias, y)


def _residual_loss(params, x, y):
    w0, b0, w1, b1, output, output_bias = params
    first = jnp.tanh(x @ w0 + b0)
    hidden = first + 0.5 * jnp.tanh(first @ w1 + b1)
    return _cross_entropy(hidden @ output + output_bias, y)


def _gated_loss(params, x, y):
    value_weight, value_bias, gate_weight, gate_bias, output, output_bias = params
    value = jnp.tanh(x @ value_weight + value_bias)
    gate = jax.nn.sigmoid(x @ gate_weight + gate_bias)
    return _cross_entropy((value * gate) @ output + output_bias, y)


def _bottleneck_loss(params, x, y):
    w0, b0, down, down_bias, up, up_bias, output, output_bias = params
    first = jnp.tanh(x @ w0 + b0)
    middle = jnp.tanh(first @ down + down_bias)
    hidden = first + 0.5 * jnp.tanh(middle @ up + up_bias)
    return _cross_entropy(hidden @ output + output_bias, y)


def _lm_loss(params, contexts, targets):
    embedding, input_weight, recurrent_weight, bias, output, output_bias = params
    encoded = embedding[contexts].swapaxes(0, 1)
    initial = jnp.zeros((contexts.shape[0], recurrent_weight.shape[0]),
                        dtype=embedding.dtype)

    def recurrent(state, value):
        state = jnp.tanh(value @ input_weight + state @ recurrent_weight + bias)
        return state, None

    final, _ = jax.lax.scan(recurrent, initial, encoded)
    return _cross_entropy(final @ output + output_bias, targets)


_MLP_LOSS = jax.jit(_mlp_loss, static_argnums=(3,))
_MLP_GRAD = jax.jit(jax.grad(_mlp_loss), static_argnums=(3,))
_DEEP_LOSS = jax.jit(_deep_loss, static_argnums=(3,))
_DEEP_GRAD = jax.jit(jax.grad(_deep_loss), static_argnums=(3,))
_AUTO_LOSS = jax.jit(_autoencoder_loss, static_argnums=(3,))
_AUTO_GRAD = jax.jit(jax.grad(_autoencoder_loss), static_argnums=(3,))
_CONV_LOSS = jax.jit(_conv_loss)
_CONV_GRAD = jax.jit(jax.grad(_conv_loss))
_RESIDUAL_LOSS = jax.jit(_residual_loss)
_RESIDUAL_GRAD = jax.jit(jax.grad(_residual_loss))
_GATED_LOSS = jax.jit(_gated_loss)
_GATED_GRAD = jax.jit(jax.grad(_gated_loss))
_BOTTLENECK_LOSS = jax.jit(_bottleneck_loss)
_BOTTLENECK_GRAD = jax.jit(jax.grad(_bottleneck_loss))
_LM_LOSS = jax.jit(_lm_loss)
_LM_GRAD = jax.jit(jax.grad(_lm_loss))


def _arrays(task):
    key = task["task_id"]
    cached = _TASK_ARRAYS.get(key)
    if cached is not None:
        return cached
    payload = task["payload"]
    x_dtype = jnp.int64 if task["family"] == "char_lm" else jnp.float64
    cached = {
        "train_x": jnp.asarray(payload["train_x"], dtype=x_dtype),
        "train_y": jnp.asarray(payload["train_y"], dtype=jnp.int64),
        "validation_x": jnp.asarray(payload["validation_x"], dtype=x_dtype),
        "validation_y": jnp.asarray(payload["validation_y"], dtype=jnp.int64),
    }
    _TASK_ARRAYS[key] = cached
    return cached


def _parameters(blocks):
    return tuple(jnp.asarray(block, dtype=jnp.float64) for block in blocks)


def _kernel(task, gradient):
    family = task["family"]
    if family == "image_mlp":
        return _MLP_GRAD if gradient else _MLP_LOSS
    if family == "image_deep_mlp":
        return _DEEP_GRAD if gradient else _DEEP_LOSS
    if family == "image_autoencoder":
        return _AUTO_GRAD if gradient else _AUTO_LOSS
    if family == "image_conv":
        return _CONV_GRAD if gradient else _CONV_LOSS
    if family == "image_residual":
        return _RESIDUAL_GRAD if gradient else _RESIDUAL_LOSS
    if family == "image_gated_mlp":
        return _GATED_GRAD if gradient else _GATED_LOSS
    if family == "image_bottleneck":
        return _BOTTLENECK_GRAD if gradient else _BOTTLENECK_LOSS
    if family == "char_lm":
        return _LM_GRAD if gradient else _LM_LOSS
    raise ValueError("unknown real optimizer workload family")


def validation_loss(task, blocks):
    arrays = _arrays(task)
    arguments = [_parameters(blocks), arrays["validation_x"],
                 arrays["validation_y"]]
    if task["family"] in ("image_mlp", "image_deep_mlp", "image_autoencoder"):
        arguments.append(task["payload"]["activation"] == "relu")
    value = _kernel(task, False)(*arguments)
    return float(value) * float(task.get("loss_scale", 1.0))


def training_gradient(task, blocks, step, indices):
    arrays = _arrays(task)
    chosen = jnp.asarray(np.asarray(indices, dtype=np.int64))
    arguments = [_parameters(blocks), arrays["train_x"][chosen],
                 arrays["train_y"][chosen]]
    if task["family"] in ("image_mlp", "image_deep_mlp", "image_autoencoder"):
        arguments.append(task["payload"]["activation"] == "relu")
    gradients = _kernel(task, True)(*arguments)
    scale = float(task.get("loss_scale", 1.0))
    return [(gradient * scale).tolist() for gradient in gradients]


def backend():
    devices = jax.devices()
    if not devices or any(device.platform != "cpu" for device in devices):
        raise RuntimeError(f"optimizer JAX backend is not CPU-only: {devices}")
    return {"framework": "jax", "version": jax.__version__,
            "platforms": sorted({device.platform for device in devices})}
