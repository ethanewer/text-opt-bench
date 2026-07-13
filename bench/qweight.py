"""Safe, size-accounted weight bundles for the Qwen3.5 compression task.

QWeight deliberately separates the untrusted *producer* from the trusted
decoder.  A submission may use any quantization algorithm, but its result is a
JSON manifest plus safetensors payloads.  The evaluator never imports a custom
kernel or deserializer from the submission.
"""

import json
import hashlib
import math
from pathlib import Path


FORMAT = "qweight-1"
MAX_MANIFEST_BYTES = 8 << 20
CODECS = {"dense", "affine", "codebook", "block_float", "graph", "alias"}


class QWeightError(ValueError):
    pass


def bundle_bytes(directory):
    root = Path(directory)
    if not root.is_dir():
        raise QWeightError("weight bundle is not a directory")
    total = 0
    for path in root.rglob("*"):
        if path.is_symlink():
            raise QWeightError("weight bundles may not contain symlinks")
        if path.is_file():
            total += path.stat().st_size
    return total


def load_manifest(directory):
    root = Path(directory)
    path = root / "manifest.json"
    if not path.is_file() or path.stat().st_size > MAX_MANIFEST_BYTES:
        raise QWeightError("missing or oversized manifest.json")
    try:
        value = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise QWeightError(f"invalid manifest.json: {exc}") from exc
    if not isinstance(value, dict) or value.get("format") != FORMAT:
        raise QWeightError(f"manifest format must be {FORMAT!r}")
    if not isinstance(value.get("tensors"), dict):
        raise QWeightError("manifest tensors must be an object")
    allowed = {"format", "base_model", "base_revision", "target_bpw",
               "producer", "tensors", "native_gguf"}
    if set(value) - allowed:
        raise QWeightError("manifest contains unknown top-level fields")
    return value


def _shape(value, label):
    if (not isinstance(value, list) or
            any(type(x) is not int or x < 0 for x in value)):
        raise QWeightError(f"{label} must be a nonnegative integer shape")
    return tuple(value)


def validate_manifest(manifest, expected_shapes, base_model, base_revision):
    """Validate structure without allocating decoded model weights."""
    if (manifest.get("base_model") != base_model or
            manifest.get("base_revision") != base_revision):
        raise QWeightError("bundle targets the wrong pinned base model")
    target = manifest.get("target_bpw")
    if type(target) not in (int, float) or not math.isfinite(target):
        raise QWeightError("target_bpw must be finite")
    records = manifest["tensors"]
    native = manifest.get("native_gguf")
    if native is not None:
        if records:
            raise QWeightError("native GGUF bundles may not mix tensor records")
        if (not isinstance(native, dict) or
                set(native) != {"file", "sha256", "architecture", "importer"} or
                not isinstance(native.get("file"), str) or
                Path(native["file"]).name != native["file"] or
                not native["file"].endswith(".gguf") or
                not isinstance(native.get("sha256"), str) or
                len(native["sha256"]) != 64 or
                native.get("architecture") != "qwen35" or
                native.get("importer") != "transformers-5.2-gguf-0.19-qwen35-v3"):
            raise QWeightError("invalid native Qwen3.5 GGUF descriptor")
        return
    if set(records) != set(expected_shapes):
        missing = sorted(set(expected_shapes) - set(records))[:3]
        extra = sorted(set(records) - set(expected_shapes))[:3]
        raise QWeightError(f"state tensor mismatch; missing={missing}, extra={extra}")
    for name, expected in expected_shapes.items():
        record = records[name]
        if not isinstance(record, dict) or record.get("codec") not in CODECS:
            raise QWeightError(f"{name}: invalid codec")
        codec = record["codec"]
        allowed = {
            "dense": {"codec", "tensor"},
            "affine": {"codec", "codes", "bits", "shape", "group_size",
                       "scales", "zeros", "g_idx", "permutation"},
            "codebook": {"codec", "codes", "bits", "shape", "group_size",
                         "codebook", "scales", "g_idx", "permutation"},
            "block_float": {"codec", "codes", "format", "shape",
                            "group_size", "scales", "permutation"},
            "graph": {"codec", "shape", "nodes", "output"},
            "alias": {"codec", "source"},
        }[codec]
        if set(record) - allowed:
            raise QWeightError(f"{name}: unknown {codec} fields")
        if codec == "alias":
            if record.get("source") not in expected_shapes:
                raise QWeightError(f"{name}: invalid alias source")
            continue
        if codec == "dense":
            if not isinstance(record.get("tensor"), str):
                raise QWeightError(f"{name}: dense tensor name is required")
            continue
        if codec == "graph":
            if _shape(record.get("shape"), f"{name}.shape") != tuple(expected):
                raise QWeightError(f"{name}: decoded shape differs from base model")
            _validate_graph(record, name)
            continue
        if _shape(record.get("shape"), f"{name}.shape") != tuple(expected):
            raise QWeightError(f"{name}: decoded shape differs from base model")
        group = record.get("group_size")
        if type(group) is not int or group < 1 or group > 65536:
            raise QWeightError(f"{name}: invalid group_size")
        for field in ("codes", "scales"):
            if not isinstance(record.get(field), str):
                raise QWeightError(f"{name}: {field} tensor name is required")
        if codec in ("affine", "codebook"):
            bits = record.get("bits")
            if type(bits) is not int or not 1 <= bits <= 8:
                raise QWeightError(f"{name}: bits must be in [1, 8]")
        if codec == "codebook" and not isinstance(record.get("codebook"), str):
            raise QWeightError(f"{name}: codebook tensor name is required")
        if codec == "block_float" and record.get("format") not in {
                "e2m1", "e4m3fn", "e5m2"}:
            raise QWeightError(f"{name}: unsupported block-float format")


def _validate_graph(record, name):
    nodes = record.get("nodes")
    if (not isinstance(nodes, list) or not 1 <= len(nodes) <= 128 or
            not isinstance(record.get("output"), str)):
        raise QWeightError(f"{name}: graph needs 1--128 nodes and an output")
    allowed_ops = {"payload", "constant", "unpack", "reshape", "permute",
                   "slice", "lookup", "vector_lookup", "add", "sub", "mul", "div",
                   "bit_and", "shift_right", "repeat_interleave", "concat"}
    identifiers = set()
    for node in nodes:
        if (not isinstance(node, dict) or not isinstance(node.get("id"), str) or
                node["id"] in identifiers or node.get("op") not in allowed_ops):
            raise QWeightError(f"{name}: invalid graph node")
        identifiers.add(node["id"])
        if len(node) > 8:
            raise QWeightError(f"{name}: graph node has too many fields")
    if record["output"] not in identifiers:
        raise QWeightError(f"{name}: graph output is missing")


def _unpack(torch, packed, bits, count, device):
    """Unpack a little-endian contiguous bitstream entirely on device."""
    source = packed.to(device=device, dtype=torch.uint8).flatten()
    positions = torch.arange(count, device=device, dtype=torch.int64) * bits
    byte = torch.div(positions, 8, rounding_mode="floor")
    shift = positions.remainder(8)
    lo = source[byte].to(torch.int64)
    hi_index = (byte + 1).clamp_max(max(0, source.numel() - 1))
    hi = source[hi_index].to(torch.int64)
    joined = lo | (hi << 8)
    return ((joined >> shift) & ((1 << bits) - 1)).to(torch.int64)


def _payload_tensor(handles, reference):
    if not isinstance(reference, str) or ":" not in reference:
        raise QWeightError("payload references must be 'file.safetensors:tensor'")
    filename, tensor = reference.split(":", 1)
    if filename not in handles:
        raise QWeightError(f"undeclared payload file {filename!r}")
    try:
        return handles[filename].get_tensor(tensor)
    except Exception as exc:
        raise QWeightError(f"missing payload tensor {reference!r}") from exc


def _references(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {"tensor", "codes", "scales", "zeros", "g_idx",
                       "permutation", "codebook"} and isinstance(item, str):
                yield item
            else:
                yield from _references(item)
    elif isinstance(value, list):
        for item in value:
            yield from _references(item)


def _graph(torch, record, handles, device, expected_numel):
    """Evaluate a bounded tensor-only decode graph for exotic block formats."""
    values = {}
    limit = max(1_000_000, expected_numel * 8)
    for node in record["nodes"]:
        op, ident = node["op"], node["id"]
        def source(key="input"):
            value = values.get(node.get(key))
            if value is None:
                raise QWeightError(f"graph node {ident!r} has a missing input")
            return value
        if op == "payload":
            value = _payload_tensor(handles, node.get("tensor")).to(device)
        elif op == "constant":
            raw = node.get("value")
            if type(raw) not in (int, float) or not math.isfinite(float(raw)):
                raise QWeightError("graph constants must be finite scalars")
            value = torch.tensor(raw, device=device)
        elif op == "unpack":
            bits, count = node.get("bits"), node.get("count")
            if (type(bits) is not int or not 1 <= bits <= 8 or
                    type(count) is not int or not 0 <= count <= limit):
                raise QWeightError("graph unpack bounds are invalid")
            value = _unpack(torch, source(), bits, count, device)
        elif op == "reshape":
            shape = _shape(node.get("shape"), "graph reshape")
            value = source().reshape(shape)
        elif op == "permute":
            dims = node.get("dims")
            if not isinstance(dims, list) or sorted(dims) != list(range(source().ndim)):
                raise QWeightError("graph permutation is invalid")
            value = source().permute(dims)
        elif op == "slice":
            dim, start, stop = node.get("dim"), node.get("start"), node.get("stop")
            base = source()
            if any(type(x) is not int for x in (dim, start, stop)):
                raise QWeightError("graph slice bounds must be integers")
            slices = [slice(None)] * base.ndim
            slices[dim] = slice(start, stop)
            value = base[tuple(slices)]
        elif op == "lookup":
            table, indices = source("table").flatten(), source("indices").long()
            value = table[indices]
        elif op == "vector_lookup":
            table, indices = source("table"), source("indices").long()
            if table.ndim != 2 or indices.numel() * table.shape[1] > limit:
                raise QWeightError("graph vector lookup bounds are invalid")
            value = table[indices]
        elif op in {"add", "sub", "mul", "div", "bit_and", "shift_right"}:
            left, right = source("left"), source("right")
            if op == "add": value = left + right
            elif op == "sub": value = left - right
            elif op == "mul": value = left * right
            elif op == "div": value = left / right
            elif op == "bit_and": value = torch.bitwise_and(left, right)
            else: value = torch.bitwise_right_shift(left, right)
        elif op == "repeat_interleave":
            repeats, dim = node.get("repeats"), node.get("dim")
            if type(repeats) is not int or not 1 <= repeats <= 65536 or type(dim) is not int:
                raise QWeightError("graph repeat_interleave bounds are invalid")
            value = source().repeat_interleave(repeats, dim=dim)
        else:
            inputs, dim = node.get("inputs"), node.get("dim")
            if (not isinstance(inputs, list) or not inputs or
                    any(item not in values for item in inputs) or type(dim) is not int):
                raise QWeightError("graph concat inputs are invalid")
            value = torch.cat([values[item] for item in inputs], dim=dim)
        if value.numel() > limit:
            raise QWeightError("graph intermediate exceeds its allocation bound")
        values[ident] = value
    return values[record["output"]].to(dtype=torch.float32)


def _groups(torch, values, shape, group_size, scales, zeros=None, g_idx=None):
    columns = shape[-1] if len(shape) else 1
    outer = math.prod(shape[:-1]) if len(shape) > 1 else 1
    groups = math.ceil(columns / group_size)
    if g_idx is not None:
        matrix = values.reshape(outer, columns).float()
        indices = g_idx.to(device=values.device, dtype=torch.long).flatten()
        if indices.numel() != columns or indices.min() < 0:
            raise QWeightError("g_idx must assign every input column")
        scale_matrix = scales.float().reshape(outer, -1)
        if indices.max() >= scale_matrix.shape[1]:
            raise QWeightError("g_idx refers to a missing scale group")
        selected_scales = scale_matrix.index_select(1, indices)
        if zeros is not None:
            zero_matrix = zeros.float().reshape(outer, -1)
            matrix = matrix - zero_matrix.index_select(1, indices)
        return (matrix * selected_scales).reshape(shape)
    padded = outer * groups * group_size
    # Packed codes omit the unused tail of every row, not merely one tail at
    # the end of the tensor. Restore each row independently before grouping.
    values = values.reshape(outer, columns)
    values = torch.nn.functional.pad(values, (0, groups * group_size - columns))
    values = values.reshape(outer, groups, group_size)
    scales = scales.float().reshape(outer, groups, 1)
    if zeros is not None:
        values = values - zeros.float().reshape(outer, groups, 1)
    decoded = (values.float() * scales).reshape(outer, groups * group_size)
    return decoded[:, :columns].reshape(shape)


def decode_bundle(directory, expected_shapes, base_model, base_revision,
                  device):
    """Return decoded FP32 tensors using only the trusted codec implementation."""
    import torch
    from safetensors import safe_open

    root = Path(directory)
    manifest = load_manifest(root)
    validate_manifest(manifest, expected_shapes, base_model, base_revision)
    if manifest.get("native_gguf") is not None:
        return _decode_native_gguf(
            torch, root, manifest, expected_shapes, device)
    payloads = sorted({ref.split(":", 1)[0]
                       for ref in _references(manifest["tensors"])
                       if ":" in ref})
    if set(path.name for path in root.iterdir() if path.is_file()) != {
            "manifest.json", *payloads}:
        raise QWeightError("bundle contains unreferenced or missing files")
    handles = {}
    try:
        for filename in payloads:
            if Path(filename).name != filename or not filename.endswith(".safetensors"):
                raise QWeightError("payload filenames must be local .safetensors files")
            handles[filename] = safe_open(root / filename, framework="pt", device="cpu")
        decoded = {}
        pending = dict(manifest["tensors"])
        while pending:
            progressed = False
            for name, rec in list(pending.items()):
                codec = rec["codec"]
                if codec == "alias":
                    source = rec["source"]
                    if source not in decoded:
                        continue
                    value = decoded[source]
                elif codec == "dense":
                    value = _payload_tensor(handles, rec["tensor"]).to(
                        device=device, dtype=torch.float32)
                elif codec == "graph":
                    value = _graph(torch, rec, handles, device,
                                   math.prod(expected_shapes[name]))
                else:
                    shape = tuple(rec["shape"])
                    count = math.prod(shape)
                    if codec == "block_float":
                        bits = {"e2m1": 4, "e4m3fn": 8, "e5m2": 8}[rec["format"]]
                    else:
                        bits = rec["bits"]
                    codes = _unpack(torch, _payload_tensor(handles, rec["codes"]),
                                    bits, count, device)
                    scales = _payload_tensor(handles, rec["scales"]).to(device)
                    g_idx = (_payload_tensor(handles, rec["g_idx"]).to(device)
                             if rec.get("g_idx") else None)
                    if codec == "affine":
                        zeros = (_payload_tensor(handles, rec["zeros"]).to(device)
                                 if rec.get("zeros") else
                                 torch.full_like(scales, 1 << (bits - 1)))
                        value = _groups(torch, codes, shape, rec["group_size"],
                                        scales, zeros, g_idx)
                    elif codec == "codebook":
                        table = _payload_tensor(handles, rec["codebook"]).to(
                            device=device, dtype=torch.float32)
                        value = table.flatten()[codes]
                        value = _groups(torch, value, shape, rec["group_size"],
                                        scales, g_idx=g_idx)
                    else:
                        table = _block_float_table(torch, rec["format"], device)
                        value = table[codes]
                        value = _groups(torch, value, shape, rec["group_size"],
                                        scales)
                    if rec.get("permutation"):
                        perm = _payload_tensor(handles, rec["permutation"]).to(
                            device=device, dtype=torch.long)
                        value = value.index_select(-1, perm)
                if tuple(value.shape) != tuple(expected_shapes[name]):
                    raise QWeightError(f"{name}: payload has wrong decoded shape")
                if not torch.isfinite(value).all():
                    raise QWeightError(f"{name}: decoded tensor is nonfinite")
                decoded[name] = value
                del pending[name]
                progressed = True
            if not progressed:
                raise QWeightError("alias cycle in manifest")
        return manifest, decoded
    finally:
        handles.clear()


def _file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _decode_native_gguf(torch, root, manifest, expected_shapes, device):
    """Decode a losslessly wrapped GGUF through the trusted Qwen3.5 adapter."""
    descriptor = manifest["native_gguf"]
    path = root / descriptor["file"]
    files = {item.name for item in root.iterdir() if item.is_file()}
    if files != {"manifest.json", descriptor["file"]}:
        raise QWeightError("native GGUF bundle contains missing or extra files")
    if not path.is_file() or _file_sha256(path) != descriptor["sha256"]:
        raise QWeightError("native GGUF payload hash mismatch")
    try:
        from bench.gguf_qwen35 import load_model
        imported = load_model(path)
        state = imported.state_dict()
        if set(state) != set(expected_shapes):
            missing = sorted(set(expected_shapes) - set(state))[:3]
            extra = sorted(set(state) - set(expected_shapes))[:3]
            raise QWeightError(
                f"native GGUF state mismatch; missing={missing}, extra={extra}")
        decoded, aliases = {}, {}
        for name, value in state.items():
            if tuple(value.shape) != tuple(expected_shapes[name]):
                raise QWeightError(f"{name}: native GGUF tensor has wrong shape")
            identity = (value.untyped_storage().data_ptr(), value.storage_offset(),
                        tuple(value.shape), tuple(value.stride()))
            if identity not in aliases:
                aliases[identity] = value.to(
                    device=device, dtype=torch.float32)
            decoded[name] = aliases[identity]
        del state, imported, aliases
        return manifest, decoded
    except QWeightError:
        raise
    except Exception as exc:
        raise QWeightError(f"native GGUF import failed: {exc}") from exc


def _block_float_table(torch, name, device):
    """Exact finite lookup tables for common FP4/FP8 interchange encodings."""
    if name == "e2m1":
        values = [0, .5, 1, 1.5, 2, 3, 4, 6]
        values += [-x for x in values]
        return torch.tensor(values, dtype=torch.float32, device=device)
    import struct
    fmt = "e"  # E5M2 is represented through IEEE float16 high bits.
    if name == "e5m2":
        vals = [struct.unpack(fmt, bytes((0, code << 2)))[0] for code in range(256)]
    else:
        # PyTorch provides the authoritative E4M3FN cast semantics.
        raw = torch.arange(256, dtype=torch.uint8, device=device)
        vals = raw.view(torch.float8_e4m3fn).float()
        return torch.nan_to_num(vals, nan=0.0)
    return torch.tensor(vals, dtype=torch.float32, device=device)
