"""Guard operator-only SLM corpus state at campaign boundaries."""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PRIVATE_SLM_OPERATOR_PATHS = (
    REPO_ROOT / "research" / "slm_sft_data" / "generated",
    REPO_ROOT / "research" / "slm_sft_data" / "catalog_v2",
)


def require_private_slm_operator_state_absent(paths=None):
    """Fail if plaintext corpus-generation state is still in the repository.

    Codex iteration workspaces are write-restricted, but the parent repository
    remains readable.  Generated answer keys, judge records, selections,
    operator-final score exports, and the source catalogs containing exact
    candidate prompts and references must therefore be quarantined before
    either a readiness pass or an optimization campaign starts.
    """
    if paths is None:
        paths = PRIVATE_SLM_OPERATOR_PATHS
    elif isinstance(paths, (str, Path)):
        paths = (Path(paths),)
    else:
        paths = tuple(Path(path) for path in paths)
    present = [Path(path) for path in paths if Path(path).exists()]
    if present:
        raise RuntimeError(
            "private SLM operator state is still present in the optimizer-"
            "readable repository: " + ", ".join(map(str, present)) +
            "; quarantine both research/slm_sft_data/generated/ and "
            "research/slm_sft_data/catalog_v2/ outside the repository before "
            "preflight or campaign launch, and restore them only for "
            "operator-final work"
        )
