"""Consolidate validated Qwen3.5 compression runs and publish the frontier."""

from __future__ import annotations

import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
GGUF = ROOT / "research/benchmark_v2/qwen35_gguf_qweight_results.json"
GPTQ = ROOT / "research/benchmark_v2/qwen35_gptq_awq_results.json"
OUTPUT = ROOT / "research/benchmark_v2/qwen35_compression_frontier.json"
TABLE = ROOT / "research/benchmark_v2/QWEN35_COMPRESSION_FRONTIER.md"
FIGURE = ROOT / "docs/qwen35_compression_pareto.svg"
LOWER_CAP, UPPER_CAP = 4.5, 5.5


def load_points():
    gguf = json.loads(GGUF.read_text())
    points = [{
        "method": row["quant"], "family": (
            "BF16" if row["quant"] == "BF16" else
            "Unsloth Dynamic" if row["quant"].startswith("UD-") else
            "IQ" if row["quant"].startswith("IQ") else "GGUF K/Q"),
        "evidence": "validated QWeight-wrapped GGUF",
        "bpw": row["bpw"], "score": row["score"], "status": "valid",
    } for row in gguf["results"] if row.get("status") == "ok"]
    # These are the two evaluator-owned whole-model RTN controls recorded in
    # the benchmark blog before the open checkpoint format replaced that task.
    points += [
        {"method": "RTN-256 low", "family": "RTN", "evidence": "local control",
         "bpw": 3.066, "score": 7.170622, "status": "valid"},
        {"method": "RTN-256 high", "family": "RTN", "evidence": "local control",
         "bpw": 4.066, "score": .329529, "status": "valid"},
    ]
    native = json.loads(GPTQ.read_text())
    for row in native.get("results", []):
        points.append({
            "method": "GPTQ W4 g128 + RTN residuals", "family": "GPTQ",
            "evidence": "validated local QWeight", "bpw": row["bpw"],
            "score": row["score"], "status": "valid",
        })
    excluded = [{**row, "status": "excluded"}
                for row in native.get("excluded_results", [])]
    return sorted(points, key=lambda row: (row["bpw"], row["score"])), excluded


def mark_frontier(points):
    best = float("inf")
    for row in points:
        row["frontier"] = row["score"] < best
        if row["frontier"]:
            best = row["score"]


def write_svg(points):
    width, height = 1280, 760
    left, right, top, bottom = 92, 360, 65, 85
    xmin, xmax, epsilon = 2.8, 16.35, 1e-5
    ymax = math.log10(max(row["score"] for row in points) + epsilon) + .05
    ymin = math.log10(epsilon)
    x = lambda v: left + (width-left-right)*(v-xmin)/(xmax-xmin)
    y = lambda v: top + (height-top-bottom)*(ymax-math.log10(max(0, v)+epsilon))/(ymax-ymin)
    colors = {"BF16":"#183b56", "Unsloth Dynamic":"#087f75", "IQ":"#d97706",
              "GGUF K/Q":"#7555a4", "GPTQ":"#be3455", "RTN":"#687573"}
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}">',
           '<rect width="100%" height="100%" fill="#fbfaf6"/>',
           '<style>text{font-family:ui-sans-serif,system-ui;fill:#263638}.s{font-size:12px}.k{font-size:11px}</style>',
           '<text x="92" y="30" font-size="21" font-weight="700">Qwen3.5 compression: total physical BPW vs validation ΔNLL</text>',
           '<text x="92" y="50" class="s">25 validated checkpoints and controls · lower-left is better · red curve is the empirical Pareto frontier</text>']
    for value in (0, .0001, .001, .01, .1, 1, 10):
        py = y(value)
        if top <= py <= height-bottom:
            out += [f'<line x1="{left}" y1="{py:.2f}" x2="{width-right}" y2="{py:.2f}" stroke="#dfe3df"/>',
                    f'<text x="{left-10}" y="{py+4:.2f}" text-anchor="end" class="s">{value:g}</text>']
    for value in range(3, 17):
        px = x(value)
        out += [f'<line x1="{px:.2f}" y1="{top}" x2="{px:.2f}" y2="{height-bottom}" stroke="#eef0ed"/>',
                f'<text x="{px:.2f}" y="{height-bottom+22}" text-anchor="middle" class="s">{value}</text>']
    for cap, label, anchor, offset in ((LOWER_CAP, "lower cap 4.5", "end", -6),
                                       (UPPER_CAP, "upper cap 5.5", "start", 6)):
        px = x(cap)
        out += [f'<line x1="{px:.2f}" y1="{top}" x2="{px:.2f}" y2="{height-bottom}" stroke="#be3455" stroke-width="2" stroke-dasharray="7 5"/>',
                f'<text x="{px+offset:.2f}" y="{top+16}" text-anchor="{anchor}" class="s" fill="#be3455">{label}</text>']
    out += [f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#596866"/>',
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#596866"/>',
            f'<text x="{(left+width-right)/2}" y="{height-28}" text-anchor="middle">total bundle bits per base-model parameter</text>',
            f'<text transform="translate(24 {(top+height-bottom)/2}) rotate(-90)" text-anchor="middle">assistant-token ΔNLL · log scale</text>']
    frontier = [row for row in points if row["frontier"]]
    out.append('<polyline fill="none" stroke="#be3455" stroke-width="3" opacity=".72" points="' +
               " ".join(f'{x(r["bpw"]):.2f},{y(r["score"]):.2f}' for r in frontier) + '"/>')
    for index, row in enumerate(points, 1):
        px, py, color = x(row["bpw"]), y(row["score"]), colors[row["family"]]
        shape = (f'<rect x="{px-6:.2f}" y="{py-6:.2f}" width="12" height="12"' if row["family"] in {"GPTQ", "RTN"}
                 else f'<circle cx="{px:.2f}" cy="{py:.2f}" r="6"')
        out.append(shape + f' fill="{color}" stroke="white" stroke-width="1.5"/>')
        out.append(f'<text x="{px+7:.2f}" y="{py-7:.2f}" class="k">{index}</text>')
    key_x = width-right+22
    for index, row in enumerate(points, 1):
        py = top + (index-1) * 24
        star = "★ " if row["frontier"] else ""
        out += [f'<circle cx="{key_x}" cy="{py}" r="4.5" fill="{colors[row["family"]]}"/>',
                f'<text x="{key_x+10}" y="{py+4}" class="k">{index}. {star}{row["method"]} · {row["bpw"]:.3f} · {row["score"]:.4g}</text>']
    out.append('</svg>')
    FIGURE.write_text("\n".join(out) + "\n")


def main():
    points, excluded = load_points()
    mark_frontier(points)
    cap_stats = {}
    for cap in (4.5, 4.625, 5.0, 5.5, 5.625, 5.75):
        eligible = [row for row in points if row["bpw"] <= cap]
        best = min(row["score"] for row in eligible)
        cap_stats[str(cap)] = {
            "eligible_points": len(eligible),
            "frontier_points": sum(row["frontier"] for row in eligible),
            "within_2x_best": sum(row["score"] <= 2 * best for row in eligible),
            "best_score": best,
        }
    payload = {"format": 1, "model": "Qwen/Qwen3.5-0.8B text-only nonthinking",
               "metric": "64-conversation ID validation assistant-token signed delta-NLL",
               "direction": "lower is better", "recommended_caps_bpw": [LOWER_CAP, UPPER_CAP],
               "threshold_statistics": cap_stats, "points": points,
               "excluded_results": excluded,
               "noncomparable_campaign_summary": {
                   "task": "slm_compression_qwen35 policy protocol",
                   "valid_unique_policies": 8, "invalid_policies": 2,
                   "whole_model_bpw_range": [7.458948, 8.146138],
                   "aggregate_two_budget_score_range": [0.432730, 2.558445],
                   "reason_not_plotted": "eligible-linear storage estimates, not emitted total-QWeight bundle bytes"
               }}
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    lines = ["# Qwen3.5 compression Pareto frontier", "",
             "All byte counts include the complete submitted bundle. Lower BPW and lower ΔNLL are better.", "",
             "| Frontier | Evidence | Method | Total BPW | Validation ΔNLL | ≤4.5 | ≤5.5 |", "|---|---|---|---:|---:|:---:|:---:|"]
    for row in points:
        lines.append(f'| {"★" if row["frontier"] else ""} | {row["evidence"]} | {row["method"]} | {row["bpw"]:.6f} | {row["score"]:.6f} | {"yes" if row["bpw"] <= LOWER_CAP else ""} | {"yes" if row["bpw"] <= UPPER_CAP else ""} |')
    lines += ["", "## Threshold evidence", "",
              "| Cap | Eligible points | Frontier points | Within 2× best | Best ΔNLL |", "|---:|---:|---:|---:|---:|"]
    for cap, stats in cap_stats.items():
        lines.append(f'| {cap} | {stats["eligible_points"]} | {stats["frontier_points"]} | {stats["within_2x_best"]} | {stats["best_score"]:.6f} |')
    lines += ["", "Recommended balanced pair: **4.5 / 5.5 BPW**. The 4.625 and 5.625 alternatives admit no additional measured method, so tighter caps win the tie. If existing-method competition is the only objective, **5.0 / 5.75 BPW** is stronger, but it weakens separation between the two tiers.", "",
              "## Excluded or noncomparable results", "",
              "- The attempted Qwen3.5 AWQ run is not a valid method baseline because its calibration changed shared linear-attention branches; it is not plotted or used to choose caps.",
              "- The earlier `slm_compression_qwen35` policy campaign produced eight unique valid policies (aggregate scores 0.432730–2.558445) and two invalid policies. Its reported total model storage was 7.458948–8.146138 BPW, but those are estimates derived from eligible-linear storage rather than emitted QWeight bundle bytes, so they are not mixed into this physical-bundle frontier.",
              "- Direct GGUF scores duplicate the 22 plotted QWeight-wrapped GGUF scores exactly and are deduplicated. Qwen2.5 paper markers and older-model tasks use different models/data and remain in their own panel.", ""]
    TABLE.write_text("\n".join(lines))
    write_svg(points)


if __name__ == "__main__":
    main()
