#!/usr/bin/env python3
"""Retired pre-protocol SLM adapter runner.

This module previously offered backend-agnostic CPU/CUDA/MPS adapters against
an obsolete evaluator.  Its outputs are not admissible for the active
benchmark because the final SLM protocol is MPS-only, uses a shared global
accelerator lease, and scores 64 SFT validation conversations rather than the
old language-modeling corpus.
"""


def main():
    raise SystemExit(
        "retired SLM adapter runner: use "
        "research/baselines/slm_calibration_ablation.py for evaluator-owned "
        "adapters or research/baselines/slm_paper_native/qwen_native_runner.py "
        "for paper-native MPS diagnostics")


if __name__ == "__main__":
    main()
