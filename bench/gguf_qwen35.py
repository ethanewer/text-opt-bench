"""Trusted Qwen3.5 GGUF import compatibility for Transformers 5.2.

Transformers 5.2 predates its Qwen3.5 GGUF mapping.  llama.cpp additionally
applies several reversible transformations when exporting this architecture.
This module supplies only those fixed mappings; it never executes submitted
code from a weight bundle.
"""

import re


EXPECTED_PARAMETERS = 752_393_024
IMPORTER_VERSION = 3


def install_adapter():
    import numpy as np
    import transformers.modeling_gguf_pytorch_utils as module
    from transformers.integrations import GGUF_CONFIG_MAPPING

    mapping = dict(GGUF_CONFIG_MAPPING["qwen3"])
    GGUF_CONFIG_MAPPING["qwen35"] = mapping
    module.GGUF_TO_TRANSFORMERS_MAPPING["config"]["qwen35"] = mapping
    if "qwen35" not in module.GGUF_SUPPORTED_ARCHITECTURES:
        module.GGUF_SUPPORTED_ARCHITECTURES.append("qwen35")

    class Qwen35Processor(module.TensorProcessor):
        def perform_fallback_tensor_mapping(
                self, gguf_to_hf_name_map, suffix, qual_name, hf_name):
            full_name = qual_name + hf_name
            match = re.fullmatch(
                r"model\.layers\.(\d+)\.linear_attn\.dt_bias", full_name)
            if match:
                gguf_to_hf_name_map[
                    f"blk.{match.group(1)}.ssm_dt.bias"] = full_name

        def process(self, weights, name, **kwargs):
            # llama.cpp stores the recurrent decay itself rather than HF's
            # logarithmic parameterization.
            if name.endswith(".ssm_a"):
                weights = np.log(-weights)
            if name.endswith(".ssm_conv1d.weight"):
                weights = np.expand_dims(weights, axis=1)
            # GGUF stores effective RMSNorm multipliers. Qwen3.5 HF stores
            # multiplier-minus-one except in its gated recurrent RMSNorm.
            if (name == "output_norm.weight" or
                    ("_norm.weight" in name and
                     ".ssm_norm.weight" not in name)):
                weights = weights - 1.0
            return module.GGUFTensor(weights, name, {})

    module.TENSOR_PROCESSORS["qwen35"] = Qwen35Processor
    module.tqdm = lambda iterable, **_kwargs: iterable


def load_model(path, model_path="/tmp/qwen35-08b"):
    """Load and authenticate a complete text-only Qwen3.5 GGUF state."""
    from transformers import AutoConfig, AutoModelForCausalLM

    install_adapter()
    config = AutoConfig.from_pretrained(
        model_path, local_files_only=True).text_config
    # AutoModel dispatch uses the config class; gguf-py uses this model_type
    # string to select its QWEN35 tensor map.
    config.model_type = "qwen35"
    model, info = AutoModelForCausalLM.from_pretrained(
        model_path, config=config, gguf_file=str(path),
        local_files_only=True, output_loading_info=True)
    missing = sorted(info.get("missing_keys", ()))
    unexpected = sorted(info.get("unexpected_keys", ()))
    mismatched = info.get("mismatched_keys", ())
    if missing or unexpected or mismatched:
        raise RuntimeError(
            f"incomplete GGUF import: missing={missing[:3]}, "
            f"unexpected={unexpected[:3]}, mismatched={mismatched[:3]}")
    count = sum(parameter.numel() for parameter in model.parameters())
    if count != EXPECTED_PARAMETERS:
        raise RuntimeError(f"GGUF import has {count} parameters")
    return model.eval()
