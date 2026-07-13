"""Checks that SLM calibration rows can never become scoring rows."""

import os
import sys
import json
from pathlib import Path
import runpy
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bench import runner
from bench import ml_models
from bench.cli import effective_evaluate_train_only
from bench.ml_models import (attest_model_device_dtype, choose_device,
                             choose_slm_device,
                             require_attested_mps_runtime,
                             require_fresh_torch_import,
                             require_mps_runtime,
                             validate_model_device_dtype_attestation)
from bench.slm_sft import (packed_layer_storage, prompt_ids_sha256,
                           select_online_validation)
from bench.session import Session
from tools.prepare_slm_sft_benchmark import development_layout
from tools.preflight_ml_benchmark import pin_mps_fallback_before_import
from research.slm_sft_data.generate_responses import (
    require_generation_mps, select_device as select_generation_device)


def test_fresh_process_guard_rejects_preimported_torch(monkeypatch):
    sentinel = object()
    monkeypatch.setitem(sys.modules, "torch", sentinel)
    try:
        require_fresh_torch_import("unit SLM path")
    except RuntimeError as exc:
        assert "before torch is imported" in str(exc)
    else:
        raise AssertionError("SLM entry accepted unverifiable prior torch import")


def test_compute_guard_rejects_unattested_torch():
    fake_torch = object()
    device = type("Device", (), {"type": "mps"})()
    try:
        require_attested_mps_runtime(fake_torch, device, "unit SLM compute")
    except RuntimeError as exc:
        assert "import attestation" in str(exc)
    else:
        raise AssertionError("SLM compute accepted an unattested torch module")


def test_model_attestation_is_nonvacuous_all_mps_and_dtype_exact(monkeypatch):
    fake_torch = object()
    monkeypatch.setattr(ml_models, "_ATTESTED_TORCH_MODULE_ID", id(fake_torch))
    monkeypatch.setattr(ml_models, "require_active_mps_lock", lambda _label: {})
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "0")

    class Tensor:
        def __init__(self, device="mps", dtype="torch.float32", elements=16):
            self.device = type("Device", (), {"type": device})()
            self.dtype = dtype
            self._elements = elements

        def is_floating_point(self):
            return True

        def numel(self):
            return self._elements

    class Model:
        def __init__(self, parameter=None):
            self.parameter = parameter

        def named_parameters(self):
            return ([] if self.parameter is None else
                    [("weight", self.parameter)])

        @staticmethod
        def named_buffers():
            return [("position", Tensor(elements=4))]

    device = type("Device", (), {"type": "mps"})()
    proof = attest_model_device_dtype(
        fake_torch, Model(Tensor()), device, "unit model", "torch.float32")
    assert proof["parameter_devices"] == ["mps"]
    assert proof["buffer_devices"] == ["mps"]
    assert proof["parameter_elements"] == 16
    assert validate_model_device_dtype_attestation(
        proof, "unit persisted model", "torch.float32") == proof

    for invalid, expected in (
            (Model(Tensor(device="cpu")), "all-MPS"),
            (Model(Tensor(dtype="torch.bfloat16")), "dtypes"),
            (Model(), "non-vacuous")):
        try:
            attest_model_device_dtype(
                fake_torch, invalid, device, "invalid unit model",
                "torch.float32")
        except RuntimeError as exc:
            assert expected in str(exc)
        else:
            raise AssertionError("invalid model placement attestation passed")


def main():
    dense_rtn = packed_layer_storage(
        1024, 1024, 4, 128, 0, False)
    assert dense_rtn == {
        "quantized_code_bits": 1024 * 1024 * 4,
        "sparsity_bitmap_bits": 0,
        "row_group_scale_bits": 1024 * 8 * 16,
        "channel_multiplier_bits": 0,
        "decode_header_bits": 8,
        "total_bits": 1024 * 1024 * 4 + 1024 * 8 * 16 + 8,
    }
    activation_scaled = packed_layer_storage(
        1024, 1024, 4, 128, 0, True)
    assert activation_scaled["channel_multiplier_bits"] == 1024 * 16
    assert (activation_scaled["total_bits"] - dense_rtn["total_bits"] ==
            1024 * 16)
    assert activation_scaled["decode_header_bits"] == 8
    sparse = packed_layer_storage(1024, 1024, 4, 128, 512, False)
    assert sparse["sparsity_bitmap_bits"] == 1024 * 1024
    assert sparse["quantized_code_bits"] == 1024 * 512 * 4

    # Every shipped policy must stay below the literal cap after the new
    # self-decoding byte is charged for each variable layer policy.
    def descriptor(role, depth, rows, columns):
        return (role, depth, rows, columns, 0.1, 1.0, (), ())

    def repeated_standard(blocks, shapes):
        return [descriptor(role, index / max(1, blocks - 1), rows, columns)
                for index in range(blocks)
                for role, rows, columns in shapes]

    # Exact eligible Linear shapes/multiplicities from the three pinned
    # safetensor indexes. Reading shape metadata needs no model load/compute.
    q25_shapes = (
        ("full_attention_q", 896, 896),
        ("full_attention_k", 128, 896),
        ("full_attention_v", 128, 896),
        ("full_attention_out", 896, 896),
        ("mlp_gate", 4864, 896),
        ("mlp_up", 4864, 896),
        ("mlp_down", 896, 4864),
    )
    q3_shapes = (
        ("full_attention_q", 2048, 1024),
        ("full_attention_k", 1024, 1024),
        ("full_attention_v", 1024, 1024),
        ("full_attention_out", 1024, 2048),
        ("mlp_gate", 3072, 1024),
        ("mlp_up", 3072, 1024),
        ("mlp_down", 1024, 3072),
    )
    q35_linear_shapes = (
        ("linear_attention_qkv", 6144, 1024),
        ("linear_attention_gate", 2048, 1024),
        ("linear_attention_decay_b", 16, 1024),
        ("linear_attention_decay_a", 16, 1024),
        ("linear_attention_out", 1024, 2048),
        ("mlp_gate", 3584, 1024),
        ("mlp_up", 3584, 1024),
        ("mlp_down", 1024, 3584),
    )
    q35_full_shapes = (
        ("full_attention_q", 4096, 1024),
        ("full_attention_k", 512, 1024),
        ("full_attention_v", 512, 1024),
        ("full_attention_out", 1024, 2048),
        ("mlp_gate", 3584, 1024),
        ("mlp_up", 3584, 1024),
        ("mlp_down", 1024, 3584),
    )
    q35_layers = [
        descriptor(role, index / 23, rows, columns)
        for index in range(24)
        for role, rows, columns in (
            q35_full_shapes if index % 4 == 3 else q35_linear_shapes)
    ]
    pinned_layer_sets = {
        "qwen25": repeated_standard(24, q25_shapes),
        "qwen3": repeated_standard(28, q3_shapes),
        "qwen35": q35_layers,
    }
    plan_paths = (
        ROOT / "bench/tasks/slm_compression_v2/initial_program.py",
        ROOT / "bench/tasks/slm_compression_qwen35/initial_program.py",
        ROOT / "research/baselines/slm_plans/rtn.py",
        ROOT / "research/baselines/slm_plans/awq_style.py",
        ROOT / "research/baselines/slm_plans/magnitude_sparse.py",
        ROOT / "research/baselines/slm_plans/wanda_style.py",
        ROOT / "tests/solutions/slm_compression_v2.py",
        ROOT / "tests/solutions/slm_compression_qwen35.py",
    )
    for plan_path in plan_paths:
        plan = runpy.run_path(str(plan_path))["plan"]
        for model_key, layers in pinned_layer_sets.items():
            total_weights = sum(layer[2] * layer[3] for layer in layers)
            for target in (3.125, 4.125):
                policies = plan(tuple(layers), target)
                assert len(policies) == len(layers)
                total_bits = 0
                for layer, policy in zip(layers, policies):
                    rows, columns = layer[2:4]
                    bits, group, _clip, prune, _prune_power, quant_power = policy
                    pruned = min(columns - 1, int(columns * prune)) if prune else 0
                    total_bits += packed_layer_storage(
                        rows, columns, bits, group, pruned,
                        quant_power > 0.0)["total_bits"]
                assert total_bits <= int(target * total_weights), (
                    plan_path, model_key, target, total_bits / total_weights)

    class Unavailable:
        @staticmethod
        def is_available():
            return False

    class FakeTorch:
        cuda = Unavailable()
        backends = type("Backends", (), {"mps": Unavailable()})()

        @staticmethod
        def device(value):
            return value

    class Available:
        @staticmethod
        def is_available():
            return True

    class FakeGenerationTorch:
        backends = type("Backends", (), {"mps": Available()})()

        @staticmethod
        def device(value):
            return value

    assert select_generation_device(FakeGenerationTorch, "mps") == "mps"
    try:
        require_generation_mps(type("Device", (), {"type": "cpu"})())
    except RuntimeError:
        pass
    else:
        raise AssertionError("direct SLM datagen helper accepted CPU")
    try:
        select_generation_device(FakeGenerationTorch, "cpu")
    except RuntimeError:
        pass
    else:
        raise AssertionError("SLM generator accepted a CPU backend")
    try:
        select_generation_device(FakeTorch, "mps")
    except RuntimeError:
        pass
    else:
        raise AssertionError("SLM generator silently fell back without MPS")
    prior_fallback = os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK")
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    try:
        try:
            require_generation_mps(type("Device", (), {"type": "mps"})())
        except RuntimeError:
            pass
        else:
            raise AssertionError(
                "direct SLM datagen helper accepted MPS CPU fallback")
        select_generation_device(FakeGenerationTorch, "mps")
    except RuntimeError:
        pass
    else:
        raise AssertionError("SLM generator accepted enabled MPS CPU fallback")
    finally:
        if prior_fallback is None:
            os.environ.pop("PYTORCH_ENABLE_MPS_FALLBACK", None)
        else:
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = prior_fallback

    prior_fallback = os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK")
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    try:
        assert pin_mps_fallback_before_import() is True
        assert os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] == "0"
    finally:
        if prior_fallback is None:
            os.environ.pop("PYTORCH_ENABLE_MPS_FALLBACK", None)
        else:
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = prior_fallback

    assert choose_device(FakeTorch, "cpu") == "cpu"
    try:
        choose_slm_device(FakeTorch)
    except RuntimeError:
        pass
    else:
        raise AssertionError("SLM scorer silently fell back when MPS was absent")
    try:
        choose_slm_device(FakeTorch, "cpu")
    except RuntimeError:
        pass
    else:
        raise AssertionError("SLM scorer accepted a non-MPS backend")
    try:
        choose_device(FakeTorch, "bogus")
    except ValueError:
        pass
    else:
        raise AssertionError("invalid explicit scoring device was accepted")
    calibration = [{"prompt_id": f"cal-{index}"} for index in range(128)]
    sealed_validation = [
        {"prompt_id": f"val-{index}"} for index in range(64)
    ]
    visible_validation = [dict(row) for row in sealed_validation]
    for task in ("slm_compression_v2", "slm_compression_qwen35"):
        config = runner.load_config(task)
        assert config["required_device"] == "mps"
        assert config["mps_fallback_allowed"] is False
        for forbidden in ("cpu", "cuda"):
            try:
                runner.evaluate(task, runner.initial_program(task),
                                device=forbidden)
            except ValueError:
                pass
            else:
                raise AssertionError(
                    f"{task} runner accepted forbidden {forbidden} scoring")
        assert not effective_evaluate_train_only(config)
        assert not effective_evaluate_train_only(config, full=True)
        # An explicit unsupported flag reaches the evaluator and is rejected;
        # it is never silently converted into a calibration score.
        assert effective_evaluate_train_only(config, requested=True)
    try:
        require_mps_runtime(type("Device", (), {"type": "cpu"})())
    except RuntimeError:
        pass
    else:
        raise AssertionError("low-level SLM compute accepted a CPU device")
    previous_fallback = os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK")
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    try:
        try:
            choose_slm_device(FakeTorch, "mps")
        except RuntimeError:
            pass
        else:
            raise AssertionError("SLM runtime accepted MPS CPU fallback")
    finally:
        if previous_fallback is None:
            os.environ.pop("PYTORCH_ENABLE_MPS_FALLBACK", None)
        else:
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = previous_fallback
    assert prompt_ids_sha256(calibration) == prompt_ids_sha256(
        [dict(row) for row in calibration])
    assert prompt_ids_sha256(calibration) != prompt_ids_sha256(
        list(reversed(calibration)))

    mixed_visible, mixed_sealed = development_layout(
        "mixed", sealed_validation)
    full_visible, full_sealed = development_layout(
        "full", sealed_validation)
    assert mixed_visible == [] and mixed_sealed == sealed_validation
    assert full_visible == sealed_validation and full_sealed is None
    assert {row["prompt_id"] for row in mixed_sealed} == {
        row["prompt_id"] for row in full_visible}

    mixed = select_online_validation(
        "mixed", [], sealed_validation, calibration)
    full = select_online_validation(
        "full", visible_validation, None, calibration)
    assert len(mixed) == len(full) == 64
    assert {row["prompt_id"] for row in mixed} == {
        row["prompt_id"] for row in full
    }
    calibration_ids = {row["prompt_id"] for row in calibration}
    assert not calibration_ids.intersection(
        row["prompt_id"] for row in mixed)
    assert not calibration_ids.intersection(
        row["prompt_id"] for row in full)

    invalid = (
        ("mixed", calibration, sealed_validation, calibration),
        ("mixed", [], sealed_validation[:-1], calibration),
        ("full", calibration[:64], None, calibration),
        ("full", visible_validation, sealed_validation, calibration),
    )
    for profile, visible, sealed, calibration_rows in invalid:
        try:
            select_online_validation(
                profile, visible, sealed, calibration_rows)
        except ValueError:
            pass
        else:
            raise AssertionError(
                f"invalid {profile} scoring layout was accepted")

    with tempfile.TemporaryDirectory(prefix="slm-session-contract-") as raw:
        run_dir = Path(raw)
        (run_dir / "session.json").write_text(json.dumps({
            "format": 1,
            "task": "slm_compression_v2",
            "kind": "generalization",
            "feedback": "train-only",
        }))
        try:
            Session.open(run_dir)
        except ValueError:
            pass
        else:
            raise AssertionError(
                "tampered SLM train-only session bypassed feedback policy")
    print("SLM calibration/scoring separation checks passed")


if __name__ == "__main__":
    main()
