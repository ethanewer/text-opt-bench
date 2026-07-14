"""Guard operator-only SLM corpus state at campaign boundaries."""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PRIVATE_SLM_OPERATOR_PATHS = (
    REPO_ROOT / "research" / "slm_sft_data" / "generated",
    REPO_ROOT / "research" / "slm_sft_data" / "catalog_v2",
)
PRIVATE_SLM_OPERATOR_GLOBS = (
    (REPO_ROOT / "research" / "benchmark_v2", "lfm25_behavior_data*"),
    (REPO_ROOT / "research" / "benchmark_v2", "lfm25_behavior*_results.json"),
    (REPO_ROOT / "research" / "benchmark_v2", "lfm25_gpqa*results.json"),
    (Path("/private/tmp"), "lfm25-230m-*-qweight"),
    (Path("/private/tmp"), "lfm25-capmatched-*"),
    (Path("/private/tmp"), "lfm25-gpqa-*"),
    (Path("/private/tmp"), "lfm25_behavior*"),
    (Path("/private/tmp"), "lfm25_bfcl*"),
    (Path("/private/tmp"), "lfm25_gpqa*"),
    (Path("/private/tmp"), "lfm25_ifbench*"),
    (Path("/private/tmp"), "lfm25-fast-*"),
)


def private_slm_operator_state_paths():
    """Return known readable artifacts that reveal SLM scoring state."""
    paths = set(PRIVATE_SLM_OPERATOR_PATHS)
    for parent, pattern in PRIVATE_SLM_OPERATOR_GLOBS:
        if parent.is_dir():
            paths.update(parent.glob(pattern))
    return tuple(sorted(paths, key=str))


def require_private_slm_operator_state_absent(paths=None):
    """Fail if plaintext corpus-generation state is still in the repository.

    Codex iteration workspaces are write-restricted, but the parent repository
    remains readable.  Generated answer keys, judge records, selections,
    operator-final score exports, and the source catalogs containing exact
    candidate prompts and references must therefore be quarantined before
    either a readiness pass or an optimization campaign starts.
    """
    if paths is None:
        paths = private_slm_operator_state_paths()
    elif isinstance(paths, (str, Path)):
        paths = (Path(paths),)
    else:
        paths = tuple(Path(path) for path in paths)
    present = [Path(path) for path in paths if Path(path).exists()]
    if present:
        raise RuntimeError(
            "private SLM operator state is still present in the optimizer-"
            "readable repository: " + ", ".join(map(str, present)) +
            "; quarantine generated corpora, selection records, detailed "
            "score exports, caches, and candidate bundles outside optimizer-"
            "readable paths before preflight or campaign launch"
        )
