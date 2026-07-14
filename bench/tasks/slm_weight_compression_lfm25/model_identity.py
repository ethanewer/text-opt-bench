"""Independent trust root for the pinned LFM2.5-230M checkpoint."""

from pathlib import Path


MODEL_ID = "LiquidAI/LFM2.5-230M"
REVISION = "37b30cce3446f3f2e26a0d3f8c67c9167f5079d7"
MODEL_PATHS = (
    Path("/private/tmp/lfm25-230m-source"),
    Path("/tmp/lfm25-230m-source"),
)
# macOS uses /private/tmp while Linux exposes /tmp directly.  Select only
# between these fixed, hash-attested locations; no candidate-controlled path
# or environment variable participates in model loading.
MODEL_PATH = next((path for path in MODEL_PATHS if path.is_dir()), MODEL_PATHS[0])
MODEL_FILES = (
    ("chat_template.jinja",
     "6d65c8804847ad74eea912dd7eca3dc1cf7a457b53a77f47d841a14121910963"),
    ("config.json",
     "f7d0bcc454b7a30fa471b1e7b9e359e11fb25b56f5b4ffd59bb18248e3c2ea3d"),
    ("generation_config.json",
     "4f88574c47c3215f7f952e1f520d1df7387422dde0345655228fb7b3fde6858c"),
    ("model.safetensors",
     "f630da86651136c9aee893b04b7542007e90fdd718355358e57e7ecc31517cfd"),
    ("tokenizer.json",
     "df1d8d5ec5d091b460562ffd545e4a5e91d17d4a0db7ebe733be34ed374377bd"),
    ("tokenizer_config.json",
     "75c287923e252b08b0a0f1c367bbe557ab23a681d0b71c5a34e0932ddbe2f5ee"),
)


def expected_files():
    """Return a fresh mapping so callers cannot mutate the trust root."""
    return dict(MODEL_FILES)
