import json
import math

import torch
from safetensors.torch import save_file

from bench.qweight import (QWeightError, bundle_bytes, decode_bundle,
                           load_manifest, validate_manifest)


def packed(values, bits):
    result = bytearray(math.ceil(len(values) * bits / 8))
    for index, value in enumerate(values):
        position = index * bits
        result[position // 8] |= (value << (position % 8)) & 255
        if position % 8 + bits > 8:
            result[position // 8 + 1] |= value >> (8 - position % 8)
    return torch.tensor(list(result), dtype=torch.uint8)


def test_affine_codebook_dense_and_alias_roundtrip(tmp_path):
    tensors = {
        "a_codes": packed([0, 1, 2, 3], 2),
        "a_scales": torch.tensor([[2.0]], dtype=torch.float16),
        "c_codes": packed([0, 1, 2, 3], 2),
        "c_scales": torch.tensor([[1.0]], dtype=torch.float16),
        "c_table": torch.tensor([-1.0, -.25, .25, 1.0], dtype=torch.float16),
        "dense": torch.tensor([7.0], dtype=torch.float16),
    }
    save_file(tensors, tmp_path / "weights.safetensors")
    manifest = {
        "format": "qweight-1", "base_model": "model", "base_revision": "rev",
        "target_bpw": 4.125, "producer": "test", "tensors": {
            "a": {"codec": "affine", "codes": "weights.safetensors:a_codes",
                  "bits": 2, "shape": [1, 4], "group_size": 4,
                  "scales": "weights.safetensors:a_scales"},
            "b": {"codec": "alias", "source": "a"},
            "c": {"codec": "codebook", "codes": "weights.safetensors:c_codes",
                  "bits": 2, "shape": [1, 4], "group_size": 4,
                  "scales": "weights.safetensors:c_scales",
                  "codebook": "weights.safetensors:c_table"},
            "d": {"codec": "dense", "tensor": "weights.safetensors:dense"},
        }}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    parsed, decoded = decode_bundle(
        tmp_path, {"a": (1, 4), "b": (1, 4), "c": (1, 4), "d": (1,)},
        "model", "rev", torch.device("cpu"))
    assert parsed["format"] == "qweight-1"
    assert torch.equal(decoded["a"], torch.tensor([[-4., -2., 0., 2.]]))
    assert torch.equal(decoded["a"], decoded["b"])
    assert torch.allclose(decoded["c"], tensors["c_table"].float()[None])
    assert decoded["d"].item() == 7
    assert bundle_bytes(tmp_path) > (tmp_path / "weights.safetensors").stat().st_size


def test_schema_expresses_research_and_interchange_families():
    # GPTQ, AWQ, and HQQ use affine records (optionally g_idx/permutation);
    # GGUF IQ/NF families use codebooks; NVFP4/FP8 use block_float; dense
    # covers BF16/FP16. Validation is independent of algorithm branding.
    shapes = {name: (2, 4) for name in ("gptq", "awq", "hqq", "gguf",
                                               "nvfp4", "fp8", "bf16", "gguf_k")}
    common = {"codes": "w.safetensors:c", "shape": [2, 4],
              "group_size": 4, "scales": "w.safetensors:s"}
    records = {
        "gptq": {"codec": "affine", "bits": 4, **common,
                 "g_idx": "w.safetensors:g"},
        "awq": {"codec": "affine", "bits": 4, **common,
                "permutation": "w.safetensors:p"},
        "hqq": {"codec": "affine", "bits": 3, **common,
                "zeros": "w.safetensors:z"},
        "gguf": {"codec": "codebook", "bits": 4, **common,
                 "codebook": "w.safetensors:t"},
        "nvfp4": {"codec": "block_float", "format": "e2m1", **common},
        "fp8": {"codec": "block_float", "format": "e4m3fn", **common},
        "bf16": {"codec": "dense", "tensor": "w.safetensors:b"},
        "gguf_k": {"codec": "graph", "shape": [2, 4], "output": "out",
                   "nodes": [
                       {"id": "blocks", "op": "payload",
                        "tensor": "w.safetensors:raw_blocks"},
                       {"id": "out", "op": "reshape", "input": "blocks",
                        "shape": [2, 4]},
                   ]},
    }
    manifest = {"format": "qweight-1", "base_model": "m",
                "base_revision": "r", "target_bpw": 4.125,
                "producer": "test", "tensors": records}
    validate_manifest(manifest, shapes, "m", "r")


def test_rejects_unknown_code_and_unaccounted_file(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({
        "format": "qweight-1", "base_model": "m", "base_revision": "r",
        "target_bpw": 3.125, "tensors": {"x": {"codec": "python"}}}))
    with __import__("pytest").raises(QWeightError):
        validate_manifest(load_manifest(tmp_path), {"x": (1,)}, "m", "r")


def test_group_padding_is_restored_per_row(tmp_path):
    save_file({
        "codes": packed([0, 1, 2, 3, 1, 2], 2),
        "scales": torch.ones((2, 2), dtype=torch.float16),
    }, tmp_path / "w.safetensors")
    manifest = {"format": "qweight-1", "base_model": "m",
                "base_revision": "r", "target_bpw": 4.125,
                "tensors": {"x": {
                    "codec": "affine", "codes": "w.safetensors:codes",
                    "bits": 2, "shape": [2, 3], "group_size": 2,
                    "scales": "w.safetensors:scales"}}}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    _, result = decode_bundle(tmp_path, {"x": (2, 3)}, "m", "r",
                              torch.device("cpu"))
    assert torch.equal(result["x"], torch.tensor([[-2., -1., 0.],
                                                   [1., -1., 0.]]))


def test_bounded_graph_decodes_packed_scale_codes(tmp_path):
    save_file({"raw": packed([0, 1, 2, 3], 2),
               "table": torch.tensor([.5, 1., 2., 4.])},
              tmp_path / "g.safetensors")
    nodes = [
        {"id": "raw", "op": "payload", "tensor": "g.safetensors:raw"},
        {"id": "idx", "op": "unpack", "input": "raw", "bits": 2, "count": 4},
        {"id": "table", "op": "payload", "tensor": "g.safetensors:table"},
        {"id": "values", "op": "lookup", "table": "table", "indices": "idx"},
        {"id": "out", "op": "reshape", "input": "values", "shape": [2, 2]},
    ]
    manifest = {"format": "qweight-1", "base_model": "m",
                "base_revision": "r", "target_bpw": 3.125,
                "tensors": {"x": {"codec": "graph", "shape": [2, 2],
                                   "nodes": nodes, "output": "out"}}}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    _, result = decode_bundle(tmp_path, {"x": (2, 2)}, "m", "r",
                              torch.device("cpu"))
    assert torch.equal(result["x"], torch.tensor([[.5, 1.], [2., 4.]]))


def test_graph_vector_lookup_expresses_additive_codebooks(tmp_path):
    save_file({
        "codes": torch.tensor([[0, 1]], dtype=torch.uint8),
        "table": torch.tensor([[1., 2.], [3., 4.]], dtype=torch.float16),
    }, tmp_path / "a.safetensors")
    nodes = [
        {"id": "codes", "op": "payload", "tensor": "a.safetensors:codes"},
        {"id": "table", "op": "payload", "tensor": "a.safetensors:table"},
        {"id": "vectors", "op": "vector_lookup", "table": "table", "indices": "codes"},
        {"id": "out", "op": "reshape", "input": "vectors", "shape": [1, 4]},
    ]
    manifest = {"format": "qweight-1", "base_model": "m",
                "base_revision": "r", "target_bpw": 3.5,
                "tensors": {"x": {"codec": "graph", "shape": [1, 4],
                                      "nodes": nodes, "output": "out"}}}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    _, result = decode_bundle(tmp_path, {"x": (1, 4)}, "m", "r",
                              torch.device("cpu"))
    assert torch.equal(result["x"], torch.tensor([[1., 2., 3., 4.]]))


def test_native_gguf_descriptor_is_safe_and_exclusive():
    manifest = {
        "format": "qweight-1", "base_model": "m", "base_revision": "r",
        "target_bpw": 4.125, "tensors": {}, "native_gguf": {
            "file": "model.gguf", "sha256": "a" * 64,
            "architecture": "qwen35",
            "importer": "transformers-5.2-gguf-0.19-qwen35-v3",
        }}
    validate_manifest(manifest, {"x": (2, 2)}, "m", "r")
    manifest["native_gguf"]["file"] = "../model.gguf"
    with __import__("pytest").raises(QWeightError):
        validate_manifest(manifest, {"x": (2, 2)}, "m", "r")
