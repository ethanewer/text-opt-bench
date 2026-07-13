#!/usr/bin/env python3
"""CPU algebra tests for the MPS-only paper-native SLM runner."""

from __future__ import annotations

from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from research.baselines.slm_paper_native.native_methods import (  # noqa: E402
    _hessian_inverse_factor,
    ActivationEnergy,
    GramAccumulator,
    apply_awq_linear_scale,
    apply_awq_norm_scale,
    asymmetric_fake_quant,
    awq_clip_linear,
    awq_reconstruction_scale_search,
    expected_qwen25_storage_layout,
    gptq_compress_linear,
    sparsegpt_compress_linear,
    wanda_prune_linear,
)
from research.baselines.slm_paper_native.protocol import (  # noqa: E402
    ELIGIBLE_PARAMETERS,
    GROUPS_128,
)


def test_accumulators_are_token_weighted() -> None:
    first = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    second = torch.tensor([[[5.0, 6.0]]])
    gram = GramAccumulator(torch, 2, torch.device("cpu"))
    energy = ActivationEnergy(torch, 2, torch.device("cpu"))
    gram.add(first)
    gram.add(second)
    energy.add(first)
    energy.add(second)
    flat = torch.cat((first.reshape(-1, 2), second.reshape(-1, 2)))
    assert gram.tokens == 3
    assert energy.tokens == 3
    torch.testing.assert_close(gram.finish(), 2 * flat.T @ flat / 3)
    torch.testing.assert_close(energy.finish(), flat.square().mean(0))


def test_mps_triangular_solve_identity_matches_upstream_cholesky_inverse(
        ) -> None:
    torch.manual_seed(101)
    values = torch.randn(17, 17, dtype=torch.float32)
    gram = values @ values.T + 0.25 * torch.eye(17, dtype=torch.float32)
    actual, dead = _hessian_inverse_factor(torch, gram, percdamp=0.01)
    assert not dead.any()
    damped = gram.clone()
    diagonal = torch.arange(gram.shape[0])
    damped[diagonal, diagonal] += 0.01 * gram.diagonal().mean()
    lower = torch.linalg.cholesky(damped)
    upstream_inverse = torch.cholesky_inverse(lower)
    expected = torch.linalg.cholesky(upstream_inverse, upper=True)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)


def test_asymmetric_fake_quant_has_exact_shape_and_group_params() -> None:
    weight = torch.tensor([
        [-2.0, -1.0, 1.0, 3.0, -4.0, -2.0, 2.0, 5.0],
        [-1.0, 0.0, 0.5, 2.0, -3.0, 1.0, 2.0, 4.0],
    ])
    quantized, scales, zeros = asymmetric_fake_quant(
        weight, 4, group_size=4, return_params=True)
    assert quantized.shape == weight.shape
    assert scales.shape == zeros.shape == (2, 2)
    assert torch.isfinite(quantized).all()
    assert ((zeros >= 0) & (zeros <= 15)).all()


def test_gptq_identity_gram_reduces_to_fixed_group_column_quantization() -> None:
    torch.manual_seed(1)
    layer = torch.nn.Linear(8, 3, bias=False)
    original = layer.weight.detach().float().clone()
    audit = gptq_compress_linear(
        torch, layer, torch.eye(8), 4, group_size=4, block_size=4,
        percdamp=0.01)
    assert audit["finite"]
    assert audit["group_size"] == audit["block_size"] == 4
    assert not torch.equal(layer.weight, original)
    # An identity Hessian has no off-diagonal error compensation.  Every
    # column therefore uses the fixed min/max parameters of its source group.
    expected = torch.empty_like(original)
    for start in (0, 4):
        source = original[:, start:start + 4]
        minimum = torch.minimum(source.amin(1), torch.zeros(3))
        maximum = torch.maximum(source.amax(1), torch.zeros(3))
        scale = (maximum - minimum).clamp_min(1e-5) / 15
        zero = (-minimum / scale).round().clamp(0, 15)
        expected[:, start:start + 4] = (
            ((source / scale[:, None]).round() + zero[:, None])
            .clamp(0, 15) - zero[:, None]) * scale[:, None]
    torch.testing.assert_close(layer.weight.float(), expected)


def test_sparsegpt_and_wanda_have_exact_row_or_blockwise_counts() -> None:
    torch.manual_seed(2)
    sparse = torch.nn.Linear(8, 4, bias=False)
    sparse_audit = sparsegpt_compress_linear(
        torch, sparse, torch.eye(8), 0.5, block_size=4)
    assert sparse_audit["finite"]
    assert sparse_audit["zeros"] == sparse.weight.numel() // 2
    assert sparse_audit["realized_sparsity"] == 0.5

    wanda = torch.nn.Linear(8, 4, bias=False)
    energy = torch.linspace(0.5, 2.0, 8)
    wanda_audit = wanda_prune_linear(torch, wanda, energy, 0.75)
    assert wanda_audit["finite"]
    assert wanda_audit["zeros"] == 3 * wanda.weight.numel() // 4
    assert (wanda.weight == 0).sum(1).tolist() == [6, 6, 6, 6]


class _DiagonalNorm(torch.nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(width))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs * self.weight


def test_awq_scale_application_is_function_preserving() -> None:
    torch.manual_seed(3)
    inputs = torch.randn(5, 8)
    scales = torch.linspace(0.5, 1.7, 8)
    norm = _DiagonalNorm(8)
    q, k, v = (torch.nn.Linear(8, width, bias=False)
               for width in (8, 4, 4))
    references = [module(norm(inputs)).detach().clone()
                  for module in (q, k, v)]
    apply_awq_norm_scale(torch, norm, [q, k, v], scales)
    for module, reference in zip((q, k, v), references):
        torch.testing.assert_close(module(norm(inputs)), reference,
                                   rtol=2e-5, atol=2e-5)

    previous = torch.nn.Linear(8, 12, bias=True)
    following = torch.nn.Linear(12, 8, bias=False)
    inner_scales = torch.linspace(0.6, 1.4, 12)
    reference = following(previous(inputs)).detach().clone()
    apply_awq_linear_scale(torch, previous, following, inner_scales)
    torch.testing.assert_close(following(previous(inputs)), reference,
                               rtol=2e-5, atol=2e-5)


def test_awq_reconstruction_search_uses_only_unmasked_tokens() -> None:
    torch.manual_seed(31)
    layer = torch.nn.Linear(8, 8, bias=False)
    inputs = torch.randn(2, 5, 8)
    mask = torch.tensor([
        [True, True, True, True, True],
        [True, True, False, False, False],
    ])
    scales, audit = awq_reconstruction_scale_search(
        torch, layer, [layer], [(inputs, {}, mask)], 4,
        group_size=4, n_grid=4)
    assert scales.shape == (8,)
    assert audit["calibration_tokens"] == 7
    assert len(audit["candidate_reconstruction_mse"]) == 4
    assert audit["finite"]


def test_awq_clip_search_is_finite_and_never_expands_ranges() -> None:
    torch.manual_seed(4)
    layer = torch.nn.Linear(8, 64, bias=False)
    inputs = torch.randn(19, 8)
    before = layer.weight.detach().abs().reshape(64, 2, 4).amax(-1)
    audit = awq_clip_linear(
        torch, layer, inputs, 4, group_size=4,
        output_batch_size=64)
    after = layer.weight.detach().abs().reshape(64, 2, 4).amax(-1)
    assert audit["finite"]
    assert 0.5 <= audit["mean_retained_max_ratio"] <= 1.0
    assert torch.all(after <= before + 1e-7)


def test_qwen25_storage_layout_matches_protocol_constants() -> None:
    layout = expected_qwen25_storage_layout()
    assert layout["eligible_parameters"] == ELIGIBLE_PARAMETERS
    assert layout["groups_128"] == GROUPS_128
