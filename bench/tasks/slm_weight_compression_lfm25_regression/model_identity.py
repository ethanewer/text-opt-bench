"""Re-export the unchanged pinned LFM checkpoint trust root."""

from bench.tasks.slm_weight_compression_lfm25.model_identity import (
    MODEL_FILES,
    MODEL_ID,
    MODEL_PATH,
    REVISION,
    expected_files,
)

__all__ = ("MODEL_FILES", "MODEL_ID", "MODEL_PATH", "REVISION", "expected_files")
