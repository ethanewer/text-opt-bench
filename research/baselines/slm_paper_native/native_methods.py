"""Audited pure-PyTorch cores for the paper-native SLM diagnostic.

The orchestration layer is MPS-only.  These small tensor kernels intentionally
remain device-agnostic so their algebra can be unit-tested without loading an
SLM.  They are ports, not imports, of the pinned upstream implementations
recorded by :mod:`qwen_native_runner`.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any


@dataclass
class GramAccumulator:
    """Full input Gram used by GPTQ and SparseGPT."""

    torch: Any
    columns: int
    device: Any

    def __post_init__(self) -> None:
        self.value = self.torch.zeros(
            (self.columns, self.columns), dtype=self.torch.float32,
            device=self.device)
        self.tokens = 0

    def add(self, inputs: Any) -> None:
        flat = inputs.detach().reshape(-1, inputs.shape[-1]).float()
        if flat.shape[-1] != self.columns:
            raise ValueError("Gram input width changed")
        self.value.addmm_(flat.T, flat)
        self.tokens += int(flat.shape[0])

    def finish(self) -> Any:
        if self.tokens <= 0:
            raise ValueError("cannot finish an empty Gram")
        # The official fixed-length implementation accumulates 2 E[x x^T].
        return self.value.mul(2.0 / self.tokens)


@dataclass
class ActivationEnergy:
    """Per-input-channel activation energy used by Wanda."""

    torch: Any
    columns: int
    device: Any

    def __post_init__(self) -> None:
        self.sum_squares = self.torch.zeros(
            self.columns, dtype=self.torch.float32, device=self.device)
        self.tokens = 0

    def add(self, inputs: Any) -> None:
        flat = inputs.detach().reshape(-1, inputs.shape[-1]).float()
        if flat.shape[-1] != self.columns:
            raise ValueError("activation-energy input width changed")
        self.sum_squares.add_(flat.square().sum(0))
        self.tokens += int(flat.shape[0])

    def finish(self) -> Any:
        if self.tokens <= 0:
            raise ValueError("cannot finish empty activation energy")
        return self.sum_squares / self.tokens


def _check_group_shape(weight: Any, group_size: int) -> None:
    if weight.ndim != 2 or weight.shape[1] % group_size:
        raise ValueError(
            f"weight shape {tuple(weight.shape)} is not divisible by "
            f"group_size={group_size}")


def asymmetric_fake_quant(weight: Any, bits: int, group_size: int = 128,
                          *, return_params: bool = False):
    """AutoAWQ/original-GPTQ asymmetric per-row group fake quantization."""
    if bits not in (4, 8):
        raise ValueError("paper-native dense quantization uses INT4 or INT8")
    _check_group_shape(weight, group_size)
    shape = weight.shape
    grouped = weight.reshape(-1, group_size)
    maximum = grouped.amax(1, keepdim=True)
    minimum = grouped.amin(1, keepdim=True)
    qmax = 2 ** bits - 1
    scales = (maximum - minimum).clamp_min(1e-5) / qmax
    zeros = (-minimum / scales).round().clamp_(0, qmax)
    quantized = ((grouped / scales).round() + zeros).clamp_(0, qmax)
    dequantized = ((quantized - zeros) * scales).reshape(shape)
    if return_params:
        groups = shape[1] // group_size
        return (dequantized, scales.reshape(shape[0], groups),
                zeros.reshape(shape[0], groups))
    return dequantized


def _hessian_inverse_factor(torch: Any, gram: Any,
                            percdamp: float = 0.01):
    """Return the upper Cholesky factor of the damped inverse Gram."""
    hessian = gram.float().clone()
    dead = hessian.diagonal() == 0
    if bool(dead.any()):
        indices = torch.arange(hessian.shape[0], device=hessian.device)
        hessian[indices[dead], indices[dead]] = 1
    damp = percdamp * hessian.diagonal().mean()
    indices = torch.arange(hessian.shape[0], device=hessian.device)
    hessian[indices, indices] += damp
    chol = torch.linalg.cholesky(hessian)
    # ``torch.cholesky_inverse`` is CUDA/CPU-only in PyTorch 2.8.  This
    # triangular-solve identity is algebraically identical and keeps the
    # complete second-order method on MPS with operator fallback disabled:
    # H^-1 = L^-T L^-1 for H = L L^T.
    identity = torch.eye(
        chol.shape[0], dtype=chol.dtype, device=chol.device)
    lower_inverse = torch.linalg.solve_triangular(
        chol, identity, upper=False)
    inverse = lower_inverse.T @ lower_inverse
    return torch.linalg.cholesky(inverse, upper=True), dead


def _group_quantize_column(torch: Any, values: Any, source_group: Any,
                           bits: int):
    zero = torch.zeros(source_group.shape[0], device=source_group.device,
                       dtype=source_group.dtype)
    minimum = torch.minimum(source_group.amin(1), zero)
    maximum = torch.maximum(source_group.amax(1), zero)
    qmax = 2 ** bits - 1
    scale = (maximum - minimum).clamp_min(1e-5) / qmax
    offset = (-minimum / scale).round().clamp_(0, qmax)
    integer = ((values / scale).round() + offset).clamp_(0, qmax)
    return (integer - offset) * scale


def gptq_compress_linear(torch: Any, layer: Any, gram: Any, bits: int,
                         *, group_size: int = 128, block_size: int = 128,
                         percdamp: float = 0.01) -> dict[str, Any]:
    """Original GPTQ column updates with asymmetric g128 fake quantization."""
    weight = layer.weight.detach().float().clone()
    _check_group_shape(weight, group_size)
    if block_size != group_size:
        raise ValueError("the pinned diagnostic requires block_size=group_size=128")
    hinv, dead = _hessian_inverse_factor(torch, gram, percdamp)
    weight[:, dead] = 0
    result = torch.zeros_like(weight)
    total_loss = torch.zeros((), device=weight.device, dtype=torch.float32)
    columns = weight.shape[1]
    for start in range(0, columns, block_size):
        stop = min(start + block_size, columns)
        local = weight[:, start:stop].clone()
        errors = torch.zeros_like(local)
        local_hinv = hinv[start:stop, start:stop]
        source_group = weight[:, start:stop]
        for offset in range(stop - start):
            values = local[:, offset]
            diagonal = local_hinv[offset, offset]
            quantized = _group_quantize_column(
                torch, values, source_group, bits)
            result[:, start + offset] = quantized
            error = (values - quantized) / diagonal
            total_loss += ((values - quantized).square() /
                           diagonal.square()).sum() / 2
            local[:, offset:] -= (
                error.unsqueeze(1) * local_hinv[offset, offset:].unsqueeze(0))
            errors[:, offset] = error
        if stop < columns:
            weight[:, stop:] -= errors @ hinv[start:stop, stop:]
    with torch.no_grad():
        layer.weight.copy_(result.to(layer.weight.dtype))
    return {
        "bits": bits,
        "group_size": group_size,
        "block_size": block_size,
        "percdamp": percdamp,
        "dead_columns": int(dead.sum().item()),
        "reconstruction_proxy_loss": float(total_loss.item()),
        "finite": bool(torch.isfinite(result).all().item()),
    }


def _exact_smallest_mask(torch: Any, values: Any, count: int):
    flat = values.reshape(-1)
    mask = torch.zeros(flat.numel(), dtype=torch.bool, device=flat.device)
    if count:
        indices = torch.topk(flat, count, largest=False, sorted=False).indices
        mask[indices] = True
    return mask.reshape(values.shape)


def sparsegpt_compress_linear(torch: Any, layer: Any, gram: Any,
                              sparsity: float, *, block_size: int = 128,
                              percdamp: float = 0.01) -> dict[str, Any]:
    """Pinned SparseGPT unstructured pruning and second-order weight update."""
    if sparsity not in (0.5, 0.75):
        raise ValueError("paper-native SparseGPT uses S50 or S75")
    weight = layer.weight.detach().float().clone()
    hinv, dead = _hessian_inverse_factor(torch, gram, percdamp)
    weight[:, dead] = 0
    columns = weight.shape[1]
    total_loss = torch.zeros((), device=weight.device, dtype=torch.float32)
    for start in range(0, columns, block_size):
        stop = min(start + block_size, columns)
        local = weight[:, start:stop].clone()
        errors = torch.zeros_like(local)
        local_hinv = hinv[start:stop, start:stop]
        importance = (local.square() /
                      local_hinv.diagonal().reshape(1, -1).square())
        prune_count = int(importance.numel() * sparsity)
        mask = _exact_smallest_mask(torch, importance, prune_count)
        for offset in range(stop - start):
            values = local[:, offset]
            diagonal = local_hinv[offset, offset]
            pruned = values.clone()
            pruned[mask[:, offset]] = 0
            weight[:, start + offset] = pruned
            error = (values - pruned) / diagonal
            total_loss += ((values - pruned).square() /
                           diagonal.square()).sum() / 2
            local[:, offset:] -= (
                error.unsqueeze(1) * local_hinv[offset, offset:].unsqueeze(0))
            errors[:, offset] = error
        if stop < columns:
            weight[:, stop:] -= errors @ hinv[start:stop, stop:]
    with torch.no_grad():
        layer.weight.copy_(weight.to(layer.weight.dtype))
    zeros = int((weight == 0).sum().item())
    return {
        "sparsity": sparsity,
        "block_size": block_size,
        "percdamp": percdamp,
        "dead_columns": int(dead.sum().item()),
        "zeros": zeros,
        "weights": weight.numel(),
        "realized_sparsity": zeros / weight.numel(),
        "reconstruction_proxy_loss": float(total_loss.item()),
        "finite": bool(torch.isfinite(weight).all().item()),
    }


def wanda_prune_linear(torch: Any, layer: Any, activation_energy: Any,
                       sparsity: float) -> dict[str, Any]:
    """Official Wanda row-wise |W| sqrt(E[x^2]) unstructured selection."""
    if sparsity not in (0.5, 0.75):
        raise ValueError("paper-native Wanda uses S50 or S75")
    weight = layer.weight.detach()
    if activation_energy.shape != (weight.shape[1],):
        raise ValueError("Wanda activation energy has the wrong width")
    metric = weight.float().abs() * activation_energy.sqrt().reshape(1, -1)
    count = int(weight.shape[1] * sparsity)
    indices = torch.topk(metric, count, dim=1, largest=False,
                         sorted=False).indices
    mask = torch.zeros_like(weight, dtype=torch.bool)
    mask.scatter_(1, indices, True)
    with torch.no_grad():
        weight[mask] = 0
    zeros = int((weight == 0).sum().item())
    return {
        "sparsity": sparsity,
        "zeros": zeros,
        "weights": weight.numel(),
        "realized_sparsity": zeros / weight.numel(),
        "finite": bool(torch.isfinite(weight).all().item()),
    }


def awq_weight_mean(torch: Any, linears: list[Any], group_size: int = 128):
    """AutoAWQ's group-normalized per-input-channel weight statistic."""
    weight = torch.cat([layer.weight.detach().float() for layer in linears], 0)
    _check_group_shape(weight, group_size)
    shape = weight.shape
    grouped = weight.reshape(-1, group_size)
    normalized = grouped.abs() / (grouped.abs().amax(1, keepdim=True) + 1e-6)
    return normalized.reshape(shape).mean(0)


def awq_scale_candidates(torch: Any, x_mean: Any, w_mean: Any,
                         n_grid: int = 20):
    """AutoAWQ duo-scaling candidates in its exact grid order."""
    for index in range(n_grid):
        ratio = index / n_grid
        scales = (x_mean.pow(ratio) /
                  (w_mean.pow(1.0 - ratio) + 1e-4)).clamp_min(1e-4)
        scales = scales / (scales.max() * scales.min()).sqrt()
        scales = torch.where(torch.isfinite(scales), scales,
                             torch.ones_like(scales))
        yield ratio, scales


def _module_tensor_output(value: Any) -> Any:
    if isinstance(value, tuple):
        value = value[0]
    if hasattr(value, "last_hidden_state"):
        value = value.last_hidden_state
    if not hasattr(value, "shape"):
        raise TypeError("AWQ reconstruction module did not return a tensor")
    return value


def _masked_squared_error(torch: Any, left: Any, right: Any,
                          token_mask: Any | None) -> tuple[Any, int]:
    difference = (left - right).float()
    if token_mask is not None:
        if difference.ndim < 2 or tuple(token_mask.shape) != tuple(
                difference.shape[:2]):
            raise ValueError("AWQ reconstruction mask does not match output")
        difference = difference[token_mask]
    return difference.square().sum(), difference.numel()


def awq_reconstruction_scale_search(
        torch: Any, module: Any, linears: list[Any],
        cases: list[tuple[Any, dict[str, Any], Any | None]], bits: int,
        *, group_size: int = 128, n_grid: int = 20) -> tuple[Any, dict[str, Any]]:
    """Search AutoAWQ's 20-point duo-scaling reconstruction objective.

    ``cases`` preserves independent variable-length conversations.  Each item
    contains the input expected by ``module``, sanitized forward kwargs, and
    an optional ``[batch, sequence]`` mask.  This is algebraically the same
    objective as AutoAWQ's fixed-length batch, without allowing right-padding
    positions to influence either activation statistics or reconstruction.
    """
    if not cases:
        raise ValueError("AWQ reconstruction search needs calibration cases")
    width = linears[0].weight.shape[1]
    if any(layer.weight.shape[1] != width for layer in linears):
        raise ValueError("all AWQ-scaled linears must share an input width")
    originals = [layer.weight.detach().clone() for layer in linears]
    x_sum = torch.zeros(width, dtype=torch.float32,
                        device=originals[0].device)
    x_tokens = 0
    references = []
    with torch.no_grad():
        for inputs, kwargs, mask in cases:
            flat = (inputs[mask] if mask is not None else
                    inputs.reshape(-1, inputs.shape[-1]))
            x_sum += flat.detach().float().abs().sum(0)
            x_tokens += int(flat.shape[0])
            references.append(
                _module_tensor_output(module(inputs, **kwargs)).detach().clone())
    if x_tokens <= 0:
        raise ValueError("AWQ reconstruction cases contain no valid tokens")
    x_mean = x_sum / x_tokens
    w_mean = awq_weight_mean(torch, linears, group_size)
    best_error = float("inf")
    best_ratio = None
    best_scales = None
    history = []
    try:
        with torch.no_grad():
            for ratio, scales in awq_scale_candidates(
                    torch, x_mean, w_mean, n_grid):
                for layer, original in zip(linears, originals):
                    scaled = original.float() * scales.reshape(1, -1)
                    candidate = asymmetric_fake_quant(
                        scaled, bits, group_size) / scales.reshape(1, -1)
                    layer.weight.copy_(candidate.to(layer.weight.dtype))
                total = torch.zeros((), dtype=torch.float32,
                                    device=originals[0].device)
                elements = 0
                for (inputs, kwargs, mask), reference in zip(cases, references):
                    candidate_output = _module_tensor_output(
                        module(inputs, **kwargs))
                    error, local_elements = _masked_squared_error(
                        torch, candidate_output, reference, mask)
                    total += error
                    elements += local_elements
                loss = float((total / elements).item())
                history.append(loss)
                if loss < best_error:
                    best_error = loss
                    best_ratio = ratio
                    best_scales = scales.detach().clone()
                for layer, original in zip(linears, originals):
                    layer.weight.copy_(original)
    finally:
        with torch.no_grad():
            for layer, original in zip(linears, originals):
                layer.weight.copy_(original)
    if best_scales is None or best_ratio is None:
        raise RuntimeError("AWQ reconstruction search did not find a scale")
    return best_scales, {
        "bits": bits,
        "group_size": group_size,
        "n_grid": n_grid,
        "calibration_tokens": x_tokens,
        "best_ratio": best_ratio,
        "best_reconstruction_mse": best_error,
        "candidate_reconstruction_mse": history,
        "finite": math.isfinite(best_error),
    }


def apply_awq_norm_scale(torch: Any, norm: Any, linears: list[Any],
                         scales: Any) -> None:
    """Function-preserving Qwen2/Qwen3 RMSNorm -> qkv or gate/up scaling."""
    with torch.no_grad():
        local = scales.to(norm.weight.device, norm.weight.dtype)
        norm.weight.div_(local)
        if getattr(norm, "bias", None) is not None:
            norm.bias.div_(local)
        for layer in linears:
            layer.weight.mul_(local.reshape(1, -1))


def apply_awq_linear_scale(torch: Any, previous: Any, following: Any,
                           scales: Any) -> None:
    """Function-preserving Qwen2/Qwen3 up_proj -> down_proj scaling."""
    with torch.no_grad():
        local = scales.to(previous.weight.device, previous.weight.dtype)
        previous.weight[-local.numel():].div_(local.reshape(-1, 1))
        if previous.bias is not None:
            previous.bias.div_(local)
        following.weight.mul_(local.reshape(1, -1))


def awq_clip_linear(torch: Any, layer: Any, sampled_inputs: Any, bits: int,
                    *, group_size: int = 128, n_grid: int = 20,
                    max_shrink: float = 0.5,
                    output_batch_size: int | None = None) -> dict[str, Any]:
    """AutoAWQ output-reconstruction clipping search (at most 512 tokens)."""
    weight = layer.weight.detach().float()
    _check_group_shape(weight, group_size)
    inputs = sampled_inputs.detach().reshape(-1, sampled_inputs.shape[-1]).float()
    if not 1 <= inputs.shape[0] <= 512:
        raise ValueError("AWQ clipping needs 1..512 sampled tokens")
    groups = weight.shape[1] // group_size
    inputs = inputs.reshape(1, inputs.shape[0], groups, group_size)
    shaped = weight.reshape(weight.shape[0], 1, groups, group_size)
    if output_batch_size is None:
        output_batch_size = 256 if weight.shape[0] % 256 == 0 else 64
    if weight.shape[0] % output_batch_size:
        raise ValueError("AWQ output batch must divide output features")
    best_values = []
    for start in range(0, weight.shape[0], output_batch_size):
        local = shaped[start:start + output_batch_size]
        original_max = local.abs().amax(-1, keepdim=True)
        best_max = original_max.clone()
        best_error = torch.full_like(original_max, float("inf"))
        reference = (inputs * local).sum(-1)
        for shrink_index in range(int(max_shrink * n_grid)):
            maximum = original_max * (1.0 - shrink_index / n_grid)
            clipped = local.clamp(-maximum, maximum)
            quantized = asymmetric_fake_quant(
                clipped.reshape(output_batch_size, -1), bits, group_size)
            quantized = quantized.reshape_as(local)
            output = (inputs * quantized).sum(-1)
            error = (output - reference).square().mean(1).reshape_as(best_error)
            better = error < best_error
            best_error[better] = error[better]
            best_max[better] = maximum[better]
        best_values.append(best_max)
    maxima = torch.cat(best_values, 0).squeeze(1)
    with torch.no_grad():
        original_shape = layer.weight.shape
        local = layer.weight.reshape(maxima.shape[0], maxima.shape[1], -1)
        local.clamp_(-maxima.to(local.dtype), maxima.to(local.dtype))
        layer.weight.data = local.reshape(original_shape)
    return {
        "bits": bits,
        "group_size": group_size,
        "sample_tokens": int(inputs.shape[1]),
        "mean_retained_max_ratio": float(
            (maxima.squeeze(-1) /
             shaped.abs().amax(-1).squeeze(1).clamp_min(1e-8))
            .mean().item()),
        "finite": bool(torch.isfinite(layer.weight).all().item()),
    }


def expected_qwen25_storage_layout() -> dict[str, int]:
    """Constants independently checked by protocol.py storage accounting."""
    hidden, intermediate, layers = 896, 4864, 24
    kv = 128
    eligible_per_layer = (
        hidden * hidden + 2 * kv * hidden + hidden * hidden +
        2 * intermediate * hidden + hidden * intermediate)
    groups_per_layer = (
        (hidden + hidden + kv + kv + intermediate + intermediate) *
        (hidden // 128) + hidden * (intermediate // 128))
    return {
        "eligible_parameters": layers * eligible_per_layer,
        "groups_128": layers * groups_per_layer,
    }
