"""Content authentication for every tokenizer used by the SLM corpus."""

from __future__ import annotations

import hashlib
from pathlib import Path


PINNED_TOKENIZER_FILES = {
    "qwen25": {
        "tokenizer_config_sha256":
            "5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583",
        "tokenizer_json_sha256":
            "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539",
        "vocab_json_sha256":
            "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
        "merges_txt_sha256":
            "599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3",
    },
    "qwen3": {
        "tokenizer_config_sha256":
            "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101",
        "tokenizer_json_sha256":
            "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
        "vocab_json_sha256":
            "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
        "merges_txt_sha256":
            "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5",
    },
    "qwen35": {
        "tokenizer_config_sha256":
            "49e2b6e395f959f077f1e992b338919c0d4a9732fc6e613995e06557f843500c",
        "tokenizer_json_sha256":
            "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42",
        "vocab_json_sha256":
            "ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003",
        "merges_txt_sha256":
            "a9d356d7bdf1ef4949e3e748e95b8e10ad9d4e2e838eddc38a0a7b6b94d1db8d",
    },
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_pinned_tokenizer_snapshots(paths: dict[str, str]) -> dict:
    """Fail closed unless all three tokenizer/config byte snapshots match."""
    if set(paths) != set(PINNED_TOKENIZER_FILES):
        raise RuntimeError("tokenizer path set differs from the pinned SLM set")
    snapshots = {}
    for key in sorted(paths):
        root = Path(paths[key])
        try:
            actual = {
                "tokenizer_config_sha256": file_sha256(
                    root / "tokenizer_config.json"),
                "tokenizer_json_sha256": file_sha256(root / "tokenizer.json"),
                "vocab_json_sha256": file_sha256(root / "vocab.json"),
                "merges_txt_sha256": file_sha256(root / "merges.txt"),
            }
        except OSError as exc:
            raise RuntimeError(
                f"pinned tokenizer files are unavailable for {key} at {root}") from exc
        if actual != PINNED_TOKENIZER_FILES[key]:
            raise RuntimeError(
                f"tokenizer snapshot authentication failed for {key}")
        snapshots[key] = actual
    return snapshots
