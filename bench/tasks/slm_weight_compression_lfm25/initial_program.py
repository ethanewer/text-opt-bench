"""Starter: symmetric groupwise three-bit RTN under the 3.5-BPW cap."""

import argparse

from research.benchmark_v2.lfm25_3p5_starter import build


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--targets", required=True)
    parser.add_argument("--device", choices=("mps",), required=True)
    args = parser.parse_args()
    build(args.model, args.output,
          tuple(float(x) for x in args.targets.split(",")), args.device)
