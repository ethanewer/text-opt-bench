#!/usr/bin/env python3
"""Generate docs/blogpost.html from recorded campaign runs.

Every figure is rendered by one chart engine so axes, ticks, colors, and
layout are consistent by construction. The x-axis for every experiment is
OPTIMIZER-ACTIVE time: each run's launch windows are read from the campaign
launcher logs (runs/_campaign/launcher.jsonl* and gen_campaign.jsonl), a run
interrupted and relaunched by the campaign is stitched at the interruption
point, and everything past 60 active minutes is excluded. This replaces the
old raw `ts - first_ts` axis that clamped relaunch-window submissions to the
60-minute mark and produced a spurious cliff at the right edge.

Experiment 1 is the current seven-task split. Its current campaign curves are
reconstructed from timestamped, validation-selected submissions. A model/task
line is rendered only when all five trials are complete. Evaluation-queue
intervals from both agent self-tests and harness submissions are removed from
wall time. The archived experiments follow it unchanged.

Usage: python3 tools/make_blogpost.py [-o docs/blogpost.html]
"""

import argparse
import glob
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bench import deferred
from bench.session import _unseal  # sealed test scores (operator-side)
import tools.blogpost_content as CONTENT
from tools.analyze_slm_trajectories import analyze as analyze_slm_trajectories
from tools.blogpost_exp4_data import HARD as PILOT_HARD

# ---------------------------------------------------------------- config

PERFECT = ["mem_index", "mem_infer"]
GEN = ["tag_seq", "compress_heldout"]
CURRENT_PERFECT = ["mem_index", "mem_infer"]
CURRENT_GEN = ["tag_seq", "compress_heldout", "llm_routing",
               "optimizer_generalization", "slm_weight_compression_lfm25"]
CURRENT_TASKS = CURRENT_PERFECT + CURRENT_GEN
POST_FIRST_SCALE = {"mem_index", "mem_infer"}

CAP = 3600.0  # one hour of active time per run

# Validated categorical palette (dataviz validator: all checks pass, light).
C_HIGH, C_LOW, C_NONE, C_GROK = "#2a78d6", "#199e70", "#4a3aa7", "#eb6834"
SETTINGS = [
    ("gpt-5.5 high", C_HIGH, lambda t, k: f"E1-r{k}-gpt-5.5-high"),
    ("gpt-5.5 low", C_LOW, lambda t, k: f"5xE-r{k}-gpt-5.5-low"),
    ("gpt-5.5 none", C_NONE, lambda t, k: _none_dir(t, k)),
    ("grok-4.5 xhigh", C_GROK,
     lambda t, k: f"GROK45-20260709-r{k}-cursor-grok-4.5-xhigh-xhigh"),
]
GEN_SETTINGS = [
    ("gpt-5.5 high", C_HIGH, lambda t, k: f"E1-r{k}-gpt-5.5-high"),
    ("gpt-5.5 low", C_LOW, lambda t, k: f"E1-r{k}-gpt-5.5-low"),
    ("gpt-5.5 none", C_NONE, lambda t, k: f"E1-r{k}-gpt-5.5-none"),
    ("grok-4.5 xhigh", C_GROK,
     lambda t, k: f"GROK45-20260709-r{k}-cursor-grok-4.5-xhigh-xhigh"),
]
# Exp 2: two feedback conditions, both gpt-5.5 low.
C_VIS, C_HID = "#2a78d6", "#eb6834"
# Exp 3: ordinal ramp light->dark = smallest->largest train set (validated).
C_R16, C_R8, C_R4 = "#86b6ef", "#2a78d6", "#104281"
# Current ML-systems/model series + reference-line hues. Keep gpt-5.5 high
# blue everywhere in the post; gpt-5.6-sol uses the distinct orange series.
C_SOL, C_55 = "#eb6834", C_HIGH
CURRENT_MODELS = [
    ("gpt-5.6-sol high", C_SOL),
    ("gpt-5.5 high", C_55),
    ("gpt-5.5 low", C_LOW),
]
# Known-invalid series stay unavailable even when their run directories and
# some sealed records exist. This also keeps partial audits out of figures.
CURRENT_EXCLUDED_SERIES = {
    ("slm_weight_compression_lfm25", "gpt-5.5 low"),
}
# Current Experiment 1 run-set mapping. Immutable pre-unification directories
# retain ``_v2`` on disk, but public task names do not.
CURRENT_RUN_SETS = {
    "gpt-5.6-sol high": {
        "campaign": "n5-main-56sol-20260713",
        "campaign_template": "n5-main-56sol-20260713-r{run}-codex-gpt-5.6-sol-high",
        "deferred_template": "v7v9-20260713-r{run}-codex-gpt-5.6-sol-high",
    },
    "gpt-5.5 high": {
        "campaign": "n5-main-55-20260713",
        "campaign_template": "n5-main-55-20260713-r{run}-codex-gpt-5.5-high",
        "deferred_template": "v9-35-gpt55-20260713-r{run}-codex-gpt-5.5-high",
        "legacy_template": "E1-r{run}-gpt-5.5-high",
    },
    "gpt-5.5 low": {
        "campaign": "n5-main-55low-20260714",
        "campaign_template": (
            "n5-main-55low-20260714-r{run}-codex-gpt-5.5-low"
        ),
        "mem_index_template": "5xE-r{run}-gpt-5.5-low",
        "legacy_template": "E1-r{run}-gpt-5.5-low",
    },
}
E4_REF_COLORS = ["#4a3aa7", "#e34948", "#d55181", "#c98500"]

EXCLUDE = set()
RESCORE_PATH = ROOT / "tools" / "blogpost_compress_heldout_rescore.json"
if RESCORE_PATH.exists():
    _rescore_payload = json.loads(RESCORE_PATH.read_text())
    for _task, _fingerprint in _rescore_payload["evaluator_fingerprints"].items():
        if deferred.benchmark_fingerprint(_task) != _fingerprint:
            raise RuntimeError(
                f"stale compress_heldout rescore for {_task}; rerun "
                "tools/rescore_compress_heldout.py"
            )
    RESCORES = {row["key"]: row for row in _rescore_payload["results"]}
else:
    RESCORES = {}

METRIC = {"mem_index": "serving peak bytes", "mem_str": "serving peak bytes",
          "mem_infer": "legacy peak bytes per instance",
          "ops_connect": "executed instructions",
          "easy_word_problems": "error rate", "tag_seq": "error rate",
          "compress_heldout": "compressed bytes"}


def recorded_task(task):
    """Historical run directories retain the pre-migration task name."""
    if task.startswith("easy_word_problems"):
        return task.replace("easy_word_problems", "word_problems", 1)
    return task


def _none_dir(task, k):
    d = f"cov-none-r{k}-gpt-5.5-none"
    if (ROOT / "runs" / task / d).exists():
        return d
    return f"CMPN-r{k}-gpt-5.5-none"


# ---------------------------------------------------------------- windows

def build_windows():
    """(task, run_dir_basename) -> sorted [(launch_t, end_t_or_None), ...]."""
    wins = {}

    def add(key, t):
        wins.setdefault(key, []).append([t, None])

    def close(key, t):
        if key in wins and wins[key] and wins[key][-1][1] is None:
            wins[key][-1][1] = t

    for f in sorted(glob.glob(str(ROOT / "runs/_campaign/launcher.jsonl*"))):
        open_key_by_task = {}
        for line in open(f):
            r = json.loads(line)
            ev, job, det = r.get("event"), r.get("job"), r.get("detail", "")
            if not job:
                continue
            task = job.split(":")[0]
            if ev == "launch" and "dir=" in det:
                d = det.split("dir=")[1].split()[0]
                add((task, d), r["t"])
                open_key_by_task[job] = (task, d)
            elif ev in ("timeout", "finish", "optimizer_error", "pause"):
                key = open_key_by_task.pop(job, None)
                if key:
                    close(key, r["t"])

    def gen_dirname(task, eff, k):
        pfx = ("E2-" if task.endswith("_e2")
               else "E3-" if task.endswith(("_r8", "_r16")) else "E1-")
        return f"{pfx}r{k}-gpt-5.5-{eff}"

    open_key_by_job = {}
    gen_log = ROOT / "runs/_campaign/gen_campaign.jsonl"
    if gen_log.exists():
        for line in open(gen_log):
            r = json.loads(line)
            ev, job = r.get("event"), r.get("job")
            if not job:
                continue
            task, eff, rk = job.split(":")
            key = (task, gen_dirname(task, eff, int(rk[1:])))
            if ev == "launch":
                add(key, r["t"])
                open_key_by_job[job] = key
            elif ev in ("timeout", "finish"):
                k2 = open_key_by_job.pop(job, None)
                if k2:
                    close(k2, r["t"])
    for v in wins.values():
        v.sort(key=lambda w: w[0])
    return wins


WINDOWS = build_windows()


# ---------------------------------------------------------------- run data

def load_run(task, dirname):
    """Submissions on the stitched active-time axis, cut at 60 minutes.

    Returns {'seed': {...}, 'subs': [{'t': minutes, 'guide', 'train', 'val',
    'test', 'ok'}, ...]} or None if the run has no usable submissions.
    """
    source_task = recorded_task(task)
    if (source_task, dirname) in EXCLUDE:
        return None
    f = ROOT / "runs" / source_task / dirname / "submissions.jsonl"
    if not f.exists():
        return None
    recs = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    recs = [r for r in recs if r.get("ts") is not None]
    recs.sort(key=lambda r: r["ts"])
    if not recs:
        return None

    wins = WINDOWS.get((source_task, dirname)) or [[recs[0]["ts"], None]]
    spans = []  # (start, hard_end, charged_span)
    for i, (s, e) in enumerate(wins):
        nxt = wins[i + 1][0] if i + 1 < len(wins) else math.inf
        hard_end = min(x for x in (e, nxt, s + CAP + 100) if x is not None)
        acts = [r["ts"] for r in recs if s <= r["ts"] < hard_end]
        span = min(max(acts) - s if acts else 0.0, CAP, hard_end - s)
        spans.append((s, hard_end, max(span, 0.0)))

    def active_offset(ts):
        base = 0.0
        for s, hard_end, span in spans:
            if s <= ts < hard_end:
                return base + (ts - s)
            base += span
        return None

    subs = []
    for r in recs:
        off = active_offset(r["ts"])
        if off is None or off > CAP:
            continue
        rescored = RESCORES.get(f"{source_task}/{dirname}/{r.get('n')}")
        if rescored is not None:
            m = rescored.get("metrics") or {}
            test = m.get("test_score")
            ok = bool(rescored.get("ok"))
            guide = rescored.get("score")
        else:
            m = r.get("metrics") or {}
            test = None
            if r.get("sealed"):
                try:
                    test = (_unseal(r["sealed"]).get("metrics") or {}).get("test_score")
                except Exception:
                    test = None
            ok = bool(r.get("ok"))
            guide = r.get("guide_score")
        subs.append({"t": off / 60.0, "ok": ok, "guide": guide,
                     "train": m.get("train_score"), "val": m.get("val_score"),
                     "test": test})
    ok = [s for s in subs if s["ok"] and s["guide"] is not None]
    if not ok:
        return None
    return {"seed": ok[0], "subs": subs}


def guide_curve(run):
    """Best-so-far graded score: [(t_min, best)], starting at the seed."""
    out, best = [], None
    for s in run["subs"]:
        if not s["ok"] or s["guide"] is None:
            continue
        if best is None or s["guide"] < best:
            best = s["guide"]
            out.append((s["t"], best))
    return out


def incumbent_curve(run, split):
    """`split` value of the graded-best incumbent over time: [(t_min, v)]."""
    out, best = [], None
    for s in run["subs"]:
        if not s["ok"] or s["guide"] is None or s[split] is None:
            continue
        if best is None or s["guide"] < best:
            best = s["guide"]
            out.append((s["t"], s[split]))
    return out


def value_at(curve, t, seed):
    v = seed
    for x, y in curve:
        if x <= t:
            v = y
        else:
            break
    return v


GRID = [i * 0.25 for i in range(241)]  # 0..60 min


def grid_mean(curves_with_seeds):
    """Pointwise mean over runs on the fixed grid -> compact step curve."""
    pts = []
    for t in GRID:
        vals = [value_at(c, t, sd) for c, sd in curves_with_seeds]
        pts.append((t, sum(vals) / len(vals)))
    return compress_steps(pts)


def compress_steps(pts):
    out = []
    for t, v in pts:
        if out and abs(out[-1][1] - v) < 1e-12:
            continue
        out.append((t, v))
    return out


# ---------------------------------------------------------------- charts

INK_MUT = "#8a8f96"
INK_LBL = "#5f6b76"
GRID_LN = "#e8e6e1"
BASE_LN = "#cfd4d8"


def nice_step(raw):
    if raw <= 0:
        return 1.0
    mag = 10 ** math.floor(math.log10(raw))
    for m in (1, 2, 2.5, 5, 10):
        if m * mag >= raw - 1e-12:
            return m * mag
    return 10 * mag


def y_axis(vmax, vmin=0.0):
    """Nice axis [lo, hi] with 4-8 round ticks; lo lifts off zero only when
    the data occupy a narrow high band (then zero would hide the signal)."""
    vmax = max(vmax, 1e-9)
    lo = 0.0
    if vmin > 0.45 * vmax:
        span_step = nice_step((vmax - vmin) / 4)
        lo = span_step * math.floor(vmin / span_step + 1e-9)
    best = None
    for n in (3, 4, 5, 6, 7):
        step = nice_step((vmax - lo) / n)
        lo2 = step * math.floor(lo / step + 1e-9)
        hi = step * math.ceil(vmax / step - 1e-9)
        nt = round((hi - lo2) / step) + 1
        if not (3 <= nt <= 8):
            continue
        waste = hi - vmax
        if best is None or (waste, step) < (best[0], best[1]):
            best = (waste, step, lo2, hi)
    if best is None:
        step = nice_step((vmax - lo) / 4)
        lo2 = step * math.floor(lo / step + 1e-9)
        hi = step * math.ceil(vmax / step - 1e-9)
        best = (0, step, lo2, hi)
    _, step, lo, hi = best
    if hi <= lo:
        hi = lo + step
    n = round((hi - lo) / step)
    ticks = [round(lo + i * step, 10) for i in range(n + 1)]
    return lo, hi, ticks, step


def fmt_val(v, step, hi):
    if v == 0:
        return "0"
    if hi >= 1e6:
        return f"{v/1e6:g}M"
    if hi >= 2e3:
        return f"{v/1e3:g}k"
    if step >= 1:
        return f"{v:g}"
    dec = min(3, max(1, -math.floor(math.log10(step))))
    return f"{v:.{dec}f}".rstrip("0").rstrip(".") if "." in f"{v:.{dec}f}" else f"{v:.{dec}f}"


class Chart:
    """One step-line chart: faint runs, bold means, bands, dashed refs."""

    def __init__(self, w, h, y_max, y_min=0.0, mini=False, y_label=None):
        self.w, self.h = w, h
        self.mini = mini
        self.ml = 44 if mini else 56
        self.mr = 10 if mini else 14
        self.mt = 8 if mini else 12
        self.mb = 30 if mini else 42
        self.y_lo, self.y_hi, self.y_ticks, self.y_step = y_axis(y_max, y_min)
        self.y_label = y_label
        self.series = []   # bold means: (name, color, curve, seed)
        self.runs = []     # faint runs: (color, curve, seed)
        self.bands = []    # (color, lo_curve, hi_curve, seeds)
        self.refs = []     # (name, value, color)

    def x(self, t):
        return self.ml + (self.w - self.ml - self.mr) * min(max(t, 0), 60) / 60.0

    def y(self, v):
        ph = self.h - self.mt - self.mb
        v = min(max(v, self.y_lo), self.y_hi)
        return self.mt + ph * (self.y_hi - v) / (self.y_hi - self.y_lo)

    def step_path(self, curve, seed):
        d = [f"M{self.x(0):.1f} {self.y(seed):.1f}"]
        for t, v in curve:
            d.append(f"H{self.x(t):.1f}")
            d.append(f"V{self.y(v):.1f}")
        d.append(f"H{self.x(60):.1f}")
        return "".join(d)

    def svg(self, hover=True):
        p = [f'<svg viewBox="0 0 {self.w} {self.h}" role="img">']
        fs = 8.5 if self.mini else 10
        # y gridlines + ticks
        for tv in self.y_ticks:
            yy = self.y(tv)
            stroke = BASE_LN if tv == self.y_ticks[0] else GRID_LN
            p.append(f'<line x1="{self.ml}" y1="{yy:.1f}" x2="{self.w-self.mr}" '
                     f'y2="{yy:.1f}" stroke="{stroke}" '
                     f'vector-effect="non-scaling-stroke"/>')
            p.append(f'<text x="{self.ml-6}" y="{yy+3:.1f}" font-size="{fs}" '
                     f'fill="{INK_MUT}" text-anchor="end">{fmt_val(tv, self.y_step, self.y_hi)}</text>')
        # x ticks
        for t in (0, 15, 30, 45, 60):
            p.append(f'<text x="{self.x(t):.1f}" y="{self.h-self.mb+14}" font-size="{fs}" '
                     f'fill="{INK_MUT}" text-anchor="middle">{t}</text>')
        p.append(f'<text x="{(self.ml+self.w-self.mr)/2:.0f}" y="{self.h-6}" '
                 f'font-size="{fs}" fill="{INK_LBL}" text-anchor="middle">Active time (min)</text>')
        if self.y_label and not self.mini:
            ym = (self.mt + self.h - self.mb) / 2
            p.append(f'<text x="12" y="{ym:.0f}" font-size="{fs}" fill="{INK_LBL}" '
                     f'text-anchor="middle" transform="rotate(-90 12 {ym:.0f})">{self.y_label}</text>')
        # bands
        for color, lo, hi, seeds in self.bands:
            up, dn = [], []
            for t in GRID:
                up.append((self.x(t), self.y(value_at(hi, t, seeds[1]))))
                dn.append((self.x(t), self.y(value_at(lo, t, seeds[0]))))
            path = "M" + "L".join(f"{a:.1f} {b:.1f}" for a, b in up) \
                 + "L" + "L".join(f"{a:.1f} {b:.1f}" for a, b in reversed(dn)) + "Z"
            p.append(f'<path d="{path}" fill="{color}" opacity="0.08"/>')
        # faint runs
        for color, curve, seed in self.runs:
            p.append(f'<path d="{self.step_path(curve, seed)}" fill="none" '
                     f'stroke="{color}" stroke-width="1" opacity="0.28" '
                     f'stroke-linejoin="round" '
                     f'vector-effect="non-scaling-stroke"/>')
        # refs
        for name, v, color in self.refs:
            if self.y_lo <= v <= self.y_hi:
                yy = self.y(v)
                p.append(f'<line x1="{self.ml}" y1="{yy:.1f}" x2="{self.w-self.mr}" '
                         f'y2="{yy:.1f}" stroke="{color}" stroke-width="1" '
                         f'stroke-dasharray="5 4" '
                         f'vector-effect="non-scaling-stroke"/>')
        # bold means (+ end dot)
        for name, color, curve, seed in self.series:
            p.append(f'<path d="{self.step_path(curve, seed)}" fill="none" '
                     f'stroke="{color}" stroke-width="2" stroke-linejoin="round" '
                     f'stroke-linecap="round" '
                     f'vector-effect="non-scaling-stroke"/>')
            endv = curve[-1][1] if curve else seed
            p.append(f'<circle cx="{self.x(60):.1f}" cy="{self.y(endv):.1f}" '
                     f'r="2.5" fill="{color}"/>')
        p.append("</svg>")
        svg = "".join(p)
        if hover and self.series:
            data = {"s": [{"n": n, "c": c,
                           "p": [[round(t, 2), round(v, 6)] for t, v in cv],
                           "s0": round(sd, 6)}
                          for n, c, cv, sd in self.series],
                    "step": self.y_step, "hi": self.y_hi,
                    "ml": self.ml, "mr": self.mr}
            return (f'<div class="ch" data-h=\'{json.dumps(data, separators=(",", ":"))}\'>'
                    f"{svg}</div>")
        return f'<div class="ch">{svg}</div>'


# ---------------------------------------------------------------- figure helpers

def norm_factory(runsets, split):
    """1 = seed, 0 = best value observed within the plotted runs (per split)."""
    seeds, best = [], None
    for run in runsets:
        c = incumbent_curve(run, split) if split != "guide" else guide_curve(run)
        sd = run["seed"][split if split != "guide" else "guide"]
        if sd is None:
            continue
        seeds.append(sd)
        for _, v in c:
            best = v if best is None else min(best, v)
    if not seeds:
        return None
    seed = sum(seeds) / len(seeds)
    if best is None or seed == best:
        return lambda v: 0.0
    return lambda v: (v - best) / (seed - best)


def aggregate_setting(task_runs, split):
    """task_runs: {task: [run, ...]}. Returns (mean_curve, band_lo, band_hi)
    on the grid, in normalized units (seed=1, best=0), plus per-run-choice
    aggregates for the std band."""
    normed = {}  # task -> [(curve, seed_normed)]
    for task, runs in task_runs.items():
        nf = norm_factory(runs, split)
        if nf is None:
            continue
        lst = []
        for run in runs:
            c = incumbent_curve(run, split) if split != "guide" else guide_curve(run)
            sd = run["seed"][split if split != "guide" else "guide"]
            if sd is None:
                continue
            lst.append(([(t, nf(v)) for t, v in c], nf(sd)))
        if lst:
            normed[task] = lst
    if not normed:
        return None
    nmax = max(len(v) for v in normed.values())
    per_choice = []
    for j in range(nmax):
        pts = []
        for t in GRID:
            vals = []
            for lst in normed.values():
                curve, seed = lst[j % len(lst)]
                vals.append(value_at(curve, t, seed))
            pts.append(sum(vals) / len(vals))
        per_choice.append(pts)
    mean_pts, lo_pts, hi_pts = [], [], []
    for i, t in enumerate(GRID):
        vals = [pc[i] for pc in per_choice]
        m = sum(vals) / len(vals)
        sd = (sum((v - m) ** 2 for v in vals) / len(vals)) ** 0.5
        mean_pts.append((t, m))
        lo_pts.append((t, m - sd))
        hi_pts.append((t, m + sd))
    return (compress_steps(mean_pts), compress_steps(lo_pts), compress_steps(hi_pts))


def legend(items):
    row = "".join(f'<span><i style="--c:{c}"></i>{n}</span>' for n, c in items)
    return f'<div class="key">{row}</div>'


def curve_key(mean_label="setting mean"):
    return (f'<div class="key key-sub"><span><i class="thin" style="--c:#7a828a"></i>'
            f'individual run</span><span><i style="--c:#7a828a"></i>{mean_label}</span></div>')


# ---------------------------------------------------------------- load everything

def load_matrix(tasks, settings):
    """{(setting_label): {task: [runs]}} keeping only loadable runs."""
    out = {}
    for label, color, dirfn in settings:
        per = {}
        for t in tasks:
            runs = []
            for k in range(1, 6):
                r = load_run(t, dirfn(t, k))
                if r:
                    runs.append(r)
            if runs:
                per[t] = runs
        out[label] = per
    return out


print("[blogpost] loading runs ...", file=sys.stderr)
M_PERF = load_matrix(PERFECT, SETTINGS)
M_GEN = load_matrix(GEN, GEN_SETTINGS)
M_E2 = load_matrix([f"{t}_e2" for t in GEN],
                   [("hidden", C_HID, lambda t, k: f"E2-r{k}-gpt-5.5-low")])
M_R8 = load_matrix([f"{t}_r8" for t in GEN],
                   [("1:8", C_R8, lambda t, k: f"E3-r{k}-gpt-5.5-low")])
M_R16 = load_matrix([f"{t}_r16" for t in GEN],
                    [("1:16", C_R16, lambda t, k: f"E3-r{k}-gpt-5.5-low")])


# ---------------------------------------------------------------- figures

AGG_W, AGG_H = 980, 320      # single wide aggregate
SINGLE_W, SINGLE_H = 980, 280 # full-width one-panel task cards
PAIR_W, PAIR_H = 480, 300    # aggregate panels + task-panel subplots (2-up)
TRI_W, TRI_H = 350, 270      # 3-up subplots (larger relative type)
MINI_W, MINI_H = 320, 210


def fig_aggregate(settings, task_runs_by_setting, split, w=PAIR_W, h=PAIR_H,
                  y_label="normalized score (1 = seed, 0 = best)"):
    aggs = {}
    ymax = 1.0
    for label, color, _ in settings:
        a = aggregate_setting(task_runs_by_setting[label], split)
        if a:
            aggs[label] = (color, a)
            ymax = max(ymax, max(v for _, v in a[2]))
    ch = Chart(w, h, ymax, y_label=y_label)
    for label, (color, (mean, lo, hi)) in aggs.items():
        ch.bands.append((color, lo, hi, (lo[0][1], hi[0][1])))
    for label, (color, (mean, lo, hi)) in aggs.items():
        ch.series.append((label, color, mean, mean[0][1]))
    return ch.svg()


def fig_task_panel(task, runs_by_series, splits, split_titles, y_label):
    """One task: len(splits) subplots sharing one y-scale.
    runs_by_series: [(label, color, [runs])]."""
    ymax = 0.0
    per = {}
    for sp in splits:
        for label, color, runs in runs_by_series:
            for run in runs:
                c = incumbent_curve(run, sp) if sp != "guide" else guide_curve(run)
                sd = run["seed"][sp if sp != "guide" else "guide"]
                if sd is None:
                    continue
                ymax = max(ymax, sd, *(v for _, v in c)) if c else max(ymax, sd)
                per.setdefault(sp, {}).setdefault(label, []).append((c, sd, color))
    w, h = (TRI_W, TRI_H) if len(splits) == 3 else (PAIR_W, PAIR_H)
    cells = []
    for sp, title in zip(splits, split_titles):
        ch = Chart(w, h, ymax, y_label=y_label)
        for label, color, _ in runs_by_series:
            entries = per.get(sp, {}).get(label, [])
            if not entries:
                continue
            for c, sd, col in entries:
                ch.runs.append((col, c, sd))
            mean = grid_mean([(c, sd) for c, sd, _ in entries])
            ch.series.append((label, color, mean, mean[0][1]))
        cells.append(f'<div><div class="ct">{title}</div>{ch.svg()}</div>')
    return cells


def fig_mini(task, settings, task_runs_by_setting):
    axis_values = []
    entries = []
    for label, color, _ in settings:
        runs = task_runs_by_setting[label].get(task, [])
        if not runs:
            continue
        cs = []
        for run in runs:
            c = guide_curve(run)
            sd = run["seed"]["guide"]
            visible = [v for _, v in c[1:]] if task in POST_FIRST_SCALE else []
            axis_values.extend(visible or [sd, *(v for _, v in c)])
            cs.append((c, sd))
        entries.append((label, color, cs))
    ch = Chart(MINI_W, MINI_H, max(axis_values), y_min=min(axis_values), mini=True)
    for label, color, cs in entries:
        for c, sd in cs:
            ch.runs.append((color, c, sd))
    for label, color, cs in entries:
        mean = grid_mean(cs)
        ch.series.append((label, color, mean, mean[0][1]))
    return ch.svg()


# ---- Experiment 1b ML systems

E4_MODELS = [("gpt-5.6-sol high", C_SOL), ("gpt-5.5 high", C_55)]
E4_TASK_DIR = {
    # Historical run directories are immutable and keep their old names.
    "routing": "llm_routing_v2",
    "optimizer": "optimizer_generalization_v2",
    "lfm": "slm_weight_compression_lfm25",
}


def _current_run_dir(task, model, run):
    base = ROOT / "runs" / E4_TASK_DIR[task]
    if model == "gpt-5.6-sol high":
        if task == "lfm":
            return None  # archived traces below already use the charged axis
        name = f"v7v9-20260713-r{run}-codex-gpt-5.6-sol-high"
    else:
        name = f"v9-35-gpt55-20260713-r{run}-codex-gpt-5.5-high"
    return base / name


def _metric(metrics, task, split):
    keys = {
        "routing": {"val": "val_score", "test": "test_dataset_macro_normalized_utility_regret"},
        "optimizer": {"val": "val_score", "test": "test_reference_normalized_curve_auc",
                      "id": "test_id_auc", "ood": "test_ood_auc"},
        "lfm": {"val": "validation_score", "test": "test_score",
                "id": "test_id_score", "ood": "test_ood_score"},
    }
    return metrics.get(keys[task][split])


def _queue_intervals(run_dir):
    """Recover trusted evaluator-wait intervals retained in run telemetry.

    Agent self-test records timestamp the start of grading, immediately after
    their queue wait. Harness submission timestamps precede their queue wait.
    Taking the interval union matches the launcher's queue-refund accounting
    without stretching any run to an artificial terminal x-coordinate.
    """
    intervals = []
    for path in Path(run_dir).glob("iter_*/evals.jsonl"):
        for line in path.read_text().splitlines():
            rec = json.loads(line)
            wait = float(rec.get("eval_queue_seconds") or 0.0)
            start = float(rec["ts"])
            if wait > 0:
                intervals.append((start - wait, start))
    submissions = Path(run_dir) / "submissions.jsonl"
    for line in submissions.read_text().splitlines():
        rec = json.loads(line)
        wait = float(rec.get("eval_queue_seconds") or 0.0)
        start = float(rec["ts"])
        if wait > 0:
            intervals.append((start, start + wait))
    return sorted(intervals)


def _interval_union_seconds(intervals, start, end):
    total = 0.0
    covered_until = start
    for left, right in intervals:
        left, right = max(left, start), min(right, end)
        if right <= left:
            continue
        if left > covered_until:
            total += right - left
            covered_until = right
        elif right > covered_until:
            total += right - covered_until
            covered_until = right
    return total


def _charged_minutes(rec, baseline_ts, queue_intervals):
    if rec.get("n") == 0:
        return 0.0
    # All panels, including deferred sealed tests, use the timestamp at which
    # the program/checkpoint was submitted. Evaluation completion time must not
    # move a checkpoint later on the optimization trajectory.
    submitted = float(rec["ts"])
    refunded = _interval_union_seconds(
        queue_intervals, baseline_ts, submitted)
    return max(0.0, (submitted - baseline_ts - refunded) / 60.0)


def _session_curve(task, model, run, split):
    run_dir = _current_run_dir(task, model, run)
    records = [json.loads(line) for line in
               (Path(run_dir) / "submissions.jsonl").read_text().splitlines()
               if line.strip()]
    records = [r for r in records if r.get("best")]
    t0 = records[0]["ts"]
    queue_intervals = _queue_intervals(run_dir)
    holdouts = {}
    holdout_path = Path(run_dir) / "holdouts.jsonl"
    if holdout_path.exists():
        for line in holdout_path.read_text().splitlines():
            if not line.strip():
                continue
            sealed = json.loads(line)
            try:
                payload = _unseal(sealed["sealed"])
            except Exception:
                continue
            if payload.get("ok"):
                holdouts[payload["n"]] = payload.get("metrics") or {}
    out = []
    for rec in records:
        x = _charged_minutes(rec, t0, queue_intervals)
        if x > 60.0 + 1e-6:
            continue
        if split == "val":
            value = rec.get("guide_score")
        else:
            metrics = holdouts.get(rec["n"])
            if metrics is None:
                continue
            value = _metric(metrics, task, split)
        if value is not None:
            out.append([round(x, 3), round(float(value), 8)])
    # Deferred evaluation intentionally coalesces old incumbents.  A sealed
    # curve therefore changes only when a scored incumbent is available.
    if out and out[0][0] > 0:
        seed = _seed_score(task, split)
        if seed is not None:
            out.insert(0, [0.0, seed])
    return out


def _seed_score(task, split):
    if task == "lfm":
        if split == "val":
            return PILOT_HARD["traces"]["lfm"][0][0][1]
        extra = PILOT_HARD["extra"]["lfm"]
        return extra.get(split + "Seed")
    if task == "routing":
        method = json.loads((ROOT / "bench/tasks/llm_routing/baseline_results.json").read_text())["methods"]["global"]
        part = "validation" if split == "val" else "test"
        return method[part]["dataset_macro_normalized_utility_regret"]
    # The optimizer starter is Adam-like.  Use the published-method Adam
    # evaluation as the sealed starting reference when the deferred queue did
    # not retain a score for submission zero.
    method = json.loads((ROOT / "bench/tasks/optimizer_generalization/baseline_results.json").read_text())["methods"]["adam"]
    if split == "val":
        return method["validation"]["score"]
    return method["test"][{"test": "score", "id": "id_auc", "ood": "ood_auc"}[split]]


def e4_runs(task, split, model):
    if task == "lfm" and model == "gpt-5.6-sol high":
        val = PILOT_HARD["traces"]["lfm"]
        if split == "val":
            return val
        extra = PILOT_HARD["extra"]["lfm"]
        observed = PILOT_HARD.get("observed", {}).get("lfm", {}).get(split)
        if observed:
            return [([[val[i][0][0], extra[split + "Seed"]]] + curve
                     if curve[0][0] > 0.1 else curve)
                    for i, curve in enumerate(observed)]
        return [[[run[0][0], extra[split + "Seed"]],
                 [run[-1][0], extra[split + "Final"][i]]]
                for i, run in enumerate(val)]
    return [_session_curve(task, model, i, split) for i in range(1, 6)]


def _e4_refs(task, split):
    if task == "routing":
        methods = json.loads((ROOT / "bench/tasks/llm_routing/baseline_results.json").read_text())["methods"]
        part = "validation" if split == "val" else "test"
        field = "dataset_macro_normalized_utility_regret"
        return [("Avengers-Pro", methods["avengers_pro_llmrouterbench_adapter"][part][field]),
                ("centroid", methods["avengers_style_centroid"][part][field]),
                ("global router", methods["global"][part][field])]
    if task == "optimizer":
        methods = json.loads((ROOT / "bench/tasks/optimizer_generalization/baseline_results.json").read_text())["methods"]
        part = "validation" if split == "val" else "test"
        field = {"val": "score", "test": "score", "id": "id_auc", "ood": "ood_auc"}[split]
        return [("Adam", methods["adam"][part][field]),
                ("RMSProp + shape LR", methods["rmsprop_shape_conditional"][part][field]),
                ("Shampoo + shape LR", methods["shampoo_shape_conditional"][part][field])]
    rows = json.loads((ROOT / "research/benchmark_v2/lfm25_capmatched_3p5_results.json").read_text())["results"]
    field = {"val": "validation", "test": "test_all", "id": "id_test", "ood": "ood_test"}[split]
    refs = [("RTN W3", _seed_score("lfm", split))]
    labels = {"GPTQModel": "GPTQ cap-matched", "HQQ": "HQQ cap-matched", "AQLM": "AQLM cap-matched"}
    for row in rows:
        for prefix, label in labels.items():
            if row["name"].startswith(prefix):
                refs.append((label, row["delta_nll"][field]))
    return refs


def fig_e4(task, split, title):
    refs = [(name, value, E4_REF_COLORS[i % len(E4_REF_COLORS)])
            for i, (name, value) in enumerate(_e4_refs(task, split))]
    model_runs = [(label, color, e4_runs(task, split, label))
                  for label, color in E4_MODELS]
    vals = [v for _, _, runs in model_runs for run in runs for _, v in run]
    vals += [v for _, v, _ in refs]
    ch = Chart(PAIR_W, PAIR_H, max(vals), y_min=min(vals),
               y_label="score (lower is better)")
    for label, color, runs in model_runs:
        curves = []
        for run in runs:
            if not run:
                continue
            curve = [(p[0], p[1]) for p in run]
            ch.runs.append((color, curve, curve[0][1]))
            curves.append((curve, curve[0][1]))
        if curves:
            mean = grid_mean(curves)
            ch.series.append((label, color, mean, mean[0][1]))
    ch.refs = refs
    key = ('<div class="key key-sub">' +
           "".join(f'<span><i class="dash" style="--c:{c}"></i>{n}</span>'
                   for n, _, c in refs) + "</div>")
    return f'<div><div class="ct">{title}</div>{ch.svg()}{key}</div>'


# ---- Current seven-task benchmark

CURRENT_METRIC = {
    "mem_index": "serving peak bytes",
    "mem_infer": "peak logical bytes",
    "tag_seq": "error rate",
    "compress_heldout": "compressed bytes",
    "llm_routing": "normalized utility regret",
    "optimizer_generalization": "normalized curve AUC",
    "slm_weight_compression_lfm25": "behavioral regression rate",
}

_CURRENT_CAMPAIGN_STATE = {}
for _campaign in {row["campaign"] for row in CURRENT_RUN_SETS.values()}:
    _state_path = ROOT / "runs/_campaign/benchmarks" / _campaign / "state.json"
    if _state_path.exists():
        _state = json.loads(_state_path.read_text())
        _CURRENT_CAMPAIGN_STATE[_campaign] = {
            (job["task"], int(job["run"])): job["status"]
            for job in _state.get("jobs", [])
        }


def _current_source(task, model, run):
    """Return (run_dir, loader, campaign) for one current-split trial.

    Some complete current protocols were run before the unified campaign and
    retain immutable historical directory names. ``legacy`` uses the repaired
    historical loader (notably the compress_heldout offline rescore), while
    ``session`` uses current queue-refunded session telemetry.
    """
    run_set = CURRENT_RUN_SETS[model]
    if model == "gpt-5.6-sol high":
        if task in ("llm_routing", "optimizer_generalization"):
            source_task = task + "_v2"
            name = run_set["deferred_template"].format(run=run)
            return ROOT / "runs" / source_task / name, "session", None
        name = run_set["campaign_template"].format(run=run)
        return (ROOT / "runs" / task / name, "session",
                run_set["campaign"])

    if model == "gpt-5.5 low":
        if (task, model) in CURRENT_EXCLUDED_SERIES:
            raise ValueError(f"excluded current series: {task} / {model}")
        if task in ("mem_infer", "llm_routing", "optimizer_generalization"):
            name = run_set["campaign_template"].format(run=run)
            return (ROOT / "runs" / task / name, "session",
                    run_set["campaign"])
        template = (run_set["mem_index_template"] if task == "mem_index"
                    else run_set["legacy_template"])
        name = template.format(run=run)
        return ROOT / "runs" / task / name, "legacy", None

    if task in ("mem_infer", "slm_weight_compression_lfm25"):
        name = run_set["campaign_template"].format(run=run)
        return ROOT / "runs" / task / name, "session", run_set["campaign"]
    if task in ("llm_routing", "optimizer_generalization"):
        source_task = task + "_v2"
        name = run_set["deferred_template"].format(run=run)
        return ROOT / "runs" / source_task / name, "session", None
    name = run_set["legacy_template"].format(run=run)
    return ROOT / "runs" / task / name, "legacy", None


def _campaign_trial_complete(campaign, task, run):
    if campaign is None:
        return True
    return _CURRENT_CAMPAIGN_STATE.get(campaign, {}).get((task, run)) == "complete"


def _current_holdout_metric(task, metrics, split):
    if task == "llm_routing":
        return metrics.get("test_dataset_macro_normalized_utility_regret")
    if task == "optimizer_generalization":
        return metrics.get("test_reference_normalized_curve_auc")
    if task == "slm_weight_compression_lfm25":
        return metrics.get("test_score")
    return metrics.get("test_score")


def _current_seed_score(task, split):
    alias = {"llm_routing": "routing",
             "optimizer_generalization": "optimizer"}.get(task)
    return _seed_score(alias, "val" if split == "online" else "test") if alias else None


def _current_active_minutes(rec, run_dir, queue_intervals, fallback_start):
    """Stitch campaign launch windows and refund evaluator-queue intervals."""
    if rec.get("n") == 0:
        return 0.0
    submitted = float(rec["ts"])
    key = (Path(run_dir).parent.name, Path(run_dir).name)
    windows = WINDOWS.get(key)
    if not windows:
        refunded = _interval_union_seconds(
            queue_intervals, fallback_start, submitted)
        return max(0.0, submitted - fallback_start - refunded) / 60.0
    active = 0.0
    for start, end in windows:
        hard_end = float(end) if end is not None else math.inf
        if start <= submitted <= hard_end:
            refunded = _interval_union_seconds(
                queue_intervals, start, submitted)
            return max(0.0, active + submitted - start - refunded) / 60.0
        if end is not None and submitted > end:
            refunded = _interval_union_seconds(queue_intervals, start, end)
            active += max(0.0, end - start - refunded)
    return None


def _session_current_curve(task, run_dir, split):
    submissions = Path(run_dir) / "submissions.jsonl"
    if not submissions.exists():
        return None
    records = [json.loads(line) for line in submissions.read_text().splitlines()
               if line.strip()]
    records = [r for r in records if r.get("ok") and r.get("best")]
    if not records:
        return None
    t0 = records[0]["ts"]
    queue_intervals = _queue_intervals(run_dir)
    holdouts = {}
    holdout_path = Path(run_dir) / "holdouts.jsonl"
    if holdout_path.exists():
        for line in holdout_path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                payload = _unseal(json.loads(line)["sealed"])
            except Exception:
                continue
            if payload.get("ok"):
                holdouts[payload["n"]] = payload.get("metrics") or {}

    curve = []
    for rec in records:
        x = _current_active_minutes(rec, run_dir, queue_intervals, t0)
        if x is None:
            continue
        if x > 60.0 + 1e-6:
            continue
        if split == "online":
            value = rec.get("guide_score")
        elif rec["n"] in holdouts:
            value = _current_holdout_metric(task, holdouts[rec["n"]], split)
        elif rec.get("sealed"):
            try:
                payload = _unseal(rec["sealed"])
                value = ((payload.get("metrics") or {}).get("test_score")
                         if payload.get("ok", True) else None)
            except Exception:
                value = None
        else:
            value = None
        if value is not None:
            curve.append((round(x, 3), float(value)))

    if not curve:
        return None
    seed = curve[0][1] if curve[0][0] <= 1e-6 else _current_seed_score(task, split)
    if seed is None:
        return None
    if curve[0][0] > 1e-6:
        curve.insert(0, (0.0, seed))
    return curve, seed


def _legacy_current_curve(task, run_dir, split):
    run = load_run(task, Path(run_dir).name)
    if not run:
        return None
    if split == "online":
        return guide_curve(run), run["seed"]["guide"]
    seed = run["seed"].get("test")
    curve = incumbent_curve(run, "test")
    return (curve, seed) if seed is not None and curve else None


_CURRENT_RUN_CACHE = {}


def current_runs(task, model, split="online"):
    """Five complete (curve, seed) trials, or [] if the line is incomplete."""
    key = (task, model, split)
    if key in _CURRENT_RUN_CACHE:
        return _CURRENT_RUN_CACHE[key]
    if (task, model) in CURRENT_EXCLUDED_SERIES:
        _CURRENT_RUN_CACHE[key] = []
        return []
    runs = []
    for run in range(1, 6):
        run_dir, loader, campaign = _current_source(task, model, run)
        if not _campaign_trial_complete(campaign, task, run):
            _CURRENT_RUN_CACHE[key] = []
            return []
        item = (_legacy_current_curve(task, run_dir, split)
                if loader == "legacy" else
                _session_current_curve(task, run_dir, split))
        if item is None:
            _CURRENT_RUN_CACHE[key] = []
            return []
        runs.append(item)
    _CURRENT_RUN_CACHE[key] = runs
    return runs


_CURRENT_SLM_AUDIT = None


def current_slm_audit():
    """Return the complete all-submission audit, never a partial summary."""
    global _CURRENT_SLM_AUDIT
    if _CURRENT_SLM_AUDIT is not None:
        return _CURRENT_SLM_AUDIT
    task = "slm_weight_compression_lfm25"
    audit_models = [model for model, _ in CURRENT_MODELS
                    if (task, model) not in CURRENT_EXCLUDED_SERIES]
    run_dirs = [_current_source(task, model, run)[0]
                for model in audit_models for run in range(1, 6)]
    try:
        audit = analyze_slm_trajectories(run_dirs)["all"]
    except (OSError, KeyError, RuntimeError, ValueError):
        return None
    if (audit["complete_runs"] != len(run_dirs) or
            audit["valid_submissions"] != audit["scored_submissions"]):
        return None
    _CURRENT_SLM_AUDIT = audit
    return audit


def _current_aggregate(model, normalizers):
    per_choice = []
    for j in range(5):
        pts = []
        for t in GRID:
            task_values = []
            for task in CURRENT_TASKS:
                curve, seed = current_runs(task, model)[j]
                nf = normalizers[task]
                task_values.append(nf(value_at(curve, t, seed)))
            pts.append((t, sum(task_values) / len(task_values)))
        per_choice.append(pts)
    mean, lo, hi = [], [], []
    for i, t in enumerate(GRID):
        vals = [choice[i][1] for choice in per_choice]
        avg = sum(vals) / len(vals)
        sd = (sum((v - avg) ** 2 for v in vals) / len(vals)) ** 0.5
        mean.append((t, avg))
        lo.append((t, avg - sd))
        hi.append((t, avg + sd))
    return compress_steps(mean), compress_steps(lo), compress_steps(hi)


def fig_current_aggregate(w=AGG_W, h=AGG_H):
    eligible = [(label, color) for label, color in CURRENT_MODELS
                if all(current_runs(task, label) for task in CURRENT_TASKS)]
    if not eligible:
        return '<p class="tip">No complete seven-task N=5 model series yet.</p>'
    normalizers = {}
    for task in CURRENT_TASKS:
        all_runs = [item for label, _ in eligible
                    for item in current_runs(task, label)]
        seeds = [seed for _, seed in all_runs]
        best = min(v for curve, _ in all_runs for _, v in curve)
        seed = sum(seeds) / len(seeds)
        normalizers[task] = ((lambda v, b=best, s=seed: (v - b) / (s - b))
                             if seed != best else (lambda v: 0.0))
    aggregates = [(label, color, _current_aggregate(label, normalizers))
                  for label, color in eligible]
    ymax = max(1.0, max(v for _, _, (_, _, hi) in aggregates for _, v in hi))
    ch = Chart(w, h, ymax, y_label="normalized score (1 = starter, 0 = best)")
    for label, color, (mean, lo, hi) in aggregates:
        ch.bands.append((color, lo, hi, (lo[0][1], hi[0][1])))
        ch.series.append((label, color, mean, mean[0][1]))
    return ch.svg()


def fig_current_task(task):
    splits = ["online"]
    titles = ["Online feedback · graded"]
    if task in CURRENT_GEN:
        splits.append("test")
        titles.append("Sealed test · selected incumbent")
    by_split = {}
    values = []
    for split in splits:
        by_split[split] = []
        for label, color in CURRENT_MODELS:
            runs = current_runs(task, label, split)
            if runs:
                by_split[split].append((label, color, runs))
                for curve, seed in runs:
                    visible = ([v for _, v in curve[1:]]
                               if task in POST_FIRST_SCALE else [])
                    values.extend(visible or [seed, *(v for _, v in curve)])
    ymax, ymin = max(values), min(values)
    cells = []
    # Perfect-information cards are the only full-width single-panel cards;
    # every generalization task has a paired online/sealed view.
    chart_w, chart_h = ((SINGLE_W, SINGLE_H) if task in CURRENT_PERFECT
                        else (PAIR_W, PAIR_H))
    for split, title in zip(splits, titles):
        if task in POST_FIRST_SCALE:
            title += " · post-first scale (starter clipped)"
        ch = Chart(chart_w, chart_h, ymax, y_min=ymin,
                   y_label=CURRENT_METRIC[task])
        for label, color, runs in by_split[split]:
            for curve, seed in runs:
                ch.runs.append((color, curve, seed))
            mean = grid_mean(runs)
            ch.series.append((label, color, mean, mean[0][1]))
        cells.append(f'<div><div class="ct">{title}</div>{ch.svg()}</div>')

    shown = []
    for label, color in CURRENT_MODELS:
        if any(current_runs(task, label, split) for split in splits):
            shown.append((label, color))
    key = legend(shown) + curve_key("N=5 mean")
    if task == "slm_weight_compression_lfm25":
        audit = current_slm_audit()
        if audit:
            changes = audit["accepted_validation_improvement_test_changes"]
            cells_mean = audit["selected_test_cells"]
            key += (
                '<div class="tip"><b>All-submission overfitting audit:</b> '
                f'all {audit["valid_submissions"]} valid submissions have sealed '
                f'scores (Spearman ρ = {audit["spearman_validation_test"]:.2f}). '
                f'The mean validation gain was {audit["mean_validation_improvement"]:.3f}, '
                f'while {audit["mean_test_improvement"]:.3f} transferred to the sealed '
                f'test. Validation-selected incumbents averaged '
                f'{audit["mean_selection_regret"]:.3f} more regression than the '
                f'per-run sealed oracle. Across accepted validation improvements, '
                f'{changes["improved"]} improved sealed score, {changes["same"]} tied, '
                f'and {changes["worsened"]} worsened it. Selected-incumbent BFCL '
                f'regression remained {cells_mean["bfcl"]:.2f}.</div>')
    return cells, key


def current_completion_note():
    parts = []
    for label, _ in CURRENT_MODELS:
        count = sum(bool(current_runs(task, label)) for task in CURRENT_TASKS)
        parts.append(f"<b>{label}</b>: {count}/{len(CURRENT_TASKS)} complete N=5 task series")
    return " · ".join(parts)


# ---------------------------------------------------------------- HTML

CSS = """
 :root{
  --ink:#18212b;--mut:#5f6b76;--faint:#82909b;--line:#dfe4e8;
  --bg:#f6f7f8;--card:#ffffff;--soft:#f1f4f6;
  --accent:#3b4a5a;--chip-bg:#eef1f4;--chip-ink:#44515e;
  --sans:Inter,-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Arial,sans-serif;
  --mono:"SFMono-Regular",Consolas,"Liberation Mono",Menlo,monospace;
  --radius:12px;--shadow:0 4px 16px rgba(24,33,43,.05)
 }
 *{box-sizing:border-box}
 html{scroll-behavior:smooth}
 body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--sans);
  font-size:15px;line-height:1.58;-webkit-font-smoothing:antialiased}
 .wrap{width:min(1160px,100%);margin:0 auto;padding:44px 28px 88px}
 header{background:var(--card);border:1px solid var(--line);border-radius:16px;
  padding:34px 38px 30px;box-shadow:var(--shadow)}
 .eyebrow,.sect-n{font-size:.72rem;font-weight:750;letter-spacing:.14em;
  text-transform:uppercase;color:var(--accent)}
 .eyebrow{margin:0 0 11px}
 h1{font-size:clamp(2rem,4vw,2.8rem);font-weight:740;line-height:1.08;
  letter-spacing:-.025em;margin:0 0 13px}
 h2{font-size:clamp(1.4rem,2.3vw,1.8rem);font-weight:710;line-height:1.2;
  letter-spacing:-.02em;margin:5px 0 10px}
 .sub{display:block;color:var(--mut);font-size:1.02rem;line-height:1.55;
  max-width:780px;margin:0 0 23px}
 p{margin:.85rem 0}
 b,strong{font-weight:680}
 code{font-family:var(--mono);font-size:.86em;background:#eaeef1;color:#28333c;
  padding:.1rem .34rem;border-radius:4px;overflow-wrap:anywhere}
 .experiment{padding:48px 0 8px;border-top:1px solid var(--line)}
 .sect-n{display:block;margin:0 0 9px}
 .lead{max-width:900px;color:var(--mut);font-size:.94rem;line-height:1.6;margin:7px 0 17px}
 .fam-ul{list-style:none;padding:0;display:grid;
  grid-template-columns:repeat(2,minmax(0,1fr));gap:8px;margin:16px 0 20px}
 header .fam-ul{gap:8px 22px;margin:0;border:0}
 header .fam-ul li{display:block;border:0;background:none;padding:2px 0}
 header .fam-ul .tag{margin-right:8px;vertical-align:1px}
 .fam-ul li{display:grid;grid-template-columns:88px 1fr;align-items:start;gap:9px;
  padding:10px 12px;background:var(--card);border:1px solid var(--line);
  border-radius:9px;font-size:.82rem;line-height:1.5;color:var(--mut)}
 .fam-ul li:last-child:nth-child(odd){grid-column:1/-1}
 .tag{display:inline-flex;align-items:center;justify-content:center;min-height:21px;
  background:var(--chip-bg);color:var(--chip-ink);border-radius:5px;
  padding:1px 7px;font-size:.63rem;font-weight:720;letter-spacing:.05em;
  text-transform:uppercase;white-space:nowrap}
 .card,.panel,.mini{background:var(--card);border:1px solid var(--line);
  box-shadow:var(--shadow);border-radius:var(--radius)}
 .card{padding:18px 20px;margin:16px 0}
 .hd{font-size:.7rem;font-weight:760;letter-spacing:.075em;text-transform:uppercase;
  color:var(--mut);margin:0 0 13px}
 svg{display:block;width:100%;height:auto}
 svg text{font-family:var(--sans);font-variant-numeric:tabular-nums}
 .sub1,.sub2,.sub3{display:grid;gap:12px}
 .sub1{grid-template-columns:1fr}
 .sub2{grid-template-columns:repeat(2,minmax(0,1fr))}
 .sub3{grid-template-columns:repeat(3,minmax(0,1fr))}
 .sub1>div,.sub2>div,.sub3>div{min-width:0;background:#fcfdfd;border:1px solid #e9edf0;
  border-radius:9px;padding:10px 10px 6px}
 .ct{min-height:18px;margin:0 0 5px;padding:0 4px;font-size:.7rem;font-weight:720;
  letter-spacing:.025em;color:#47535e;line-height:1.3}
 .base-row{display:grid;grid-template-columns:minmax(0,1fr) 64px;gap:10px;
  padding:7px 5px;border-top:1px solid #e9edf0;font-size:.75rem;color:#47535e}
 .base-row:first-of-type{border-top:0}
 .base-row b{font-family:var(--mono);font-variant-numeric:tabular-nums;
  text-align:right;color:#26323c}
 .panels{display:grid;grid-template-columns:1fr;gap:14px;margin-top:14px}
 .panel{overflow:hidden;min-width:0}
 .ph{display:flex;align-items:center;gap:10px;flex-wrap:wrap;padding:11px 15px;
  background:var(--soft);border-bottom:1px solid var(--line)}
 .pname{font-family:var(--mono);font-size:.88rem;font-weight:700;color:#26323c;
  overflow-wrap:anywhere}
 .pgap{margin-left:auto;font-size:.66rem;font-weight:690;letter-spacing:.055em;
  text-transform:uppercase;color:var(--mut)}
 .chip{font-size:.62rem;font-weight:720;letter-spacing:.05em;text-transform:uppercase;
  color:var(--faint);background:var(--card);border:1px solid var(--line);
  border-radius:5px;padding:2px 7px;white-space:nowrap}
 .pbody{padding:13px 14px 14px}
 .grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px;margin-top:13px}
 .grid>*{min-width:0}
 .mini{padding:9px 11px 7px}
 .mh{display:flex;align-items:center;gap:8px;margin:1px 2px 4px}
 .mt{font-family:var(--mono);font-size:.76rem;font-weight:700;color:#34414b;
  margin-right:auto}
 .flip{cursor:pointer;transition:border-color .14s ease,box-shadow .14s ease}
 .flip:hover,.flip:focus-visible{border-color:#9db4c6;
  box-shadow:0 8px 24px rgba(37,84,120,.12);outline:none}
 .flip .back{display:none}
 .flip.flipped .front{display:none}
 .flip.flipped .back{display:block}
 .back{padding:4px 6px 8px}
 .panel .back{padding:16px 19px}
 .bt{display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin-bottom:6px}
 .bt b{font-family:var(--mono);font-size:.88rem}
 .bl{margin:8px 0 0;padding-left:18px}
 .bl li{margin:6px 0;color:#35414b;font-size:.79rem;line-height:1.48}
 .mini .bl{padding-left:15px}
 .mini .bl li{font-size:.72rem;margin:4px 0}
 .bh{margin-top:12px;font-size:.61rem;font-weight:700;letter-spacing:.06em;
  text-transform:uppercase;color:var(--faint)}
 .key{display:flex;flex-wrap:wrap;align-items:center;gap:6px 18px;
  color:var(--mut);font-size:.72rem;margin:10px 2px 2px}
 .key-sub{margin:6px 4px 0;font-size:.68rem}
 .key span{display:inline-flex;align-items:center;white-space:nowrap}
 .key i{display:inline-block;width:19px;height:3px;border-radius:2px;margin-right:7px;
  background:var(--c)}
 .key i.thin{height:1px;opacity:.45}
 .key i.dash{height:0;border-top:2px dashed var(--c);background:none;border-radius:0}
 .tip{max-width:980px;margin:11px 2px 0;color:var(--mut);font-size:.75rem;line-height:1.5}
 .tip b{color:var(--accent)}
 figure{margin:0}
 .ch{position:relative}
 .ch .xh{position:absolute;top:0;bottom:0;width:1px;background:#9aa4ad;opacity:0;
  pointer-events:none}
 .ch .tt{position:absolute;min-width:120px;background:#fff;border:1px solid var(--line);
  border-radius:7px;box-shadow:0 6px 18px rgba(24,33,43,.14);padding:7px 10px;
  font-size:.7rem;line-height:1.5;color:var(--ink);opacity:0;pointer-events:none;
  z-index:5;font-variant-numeric:tabular-nums}
 .tt b{display:block;color:var(--mut);font-weight:650;margin-bottom:2px}
 .tt .r{display:flex;align-items:center;gap:6px;white-space:nowrap}
 .tt .r i{width:10px;height:3px;border-radius:2px;background:var(--c)}
 .tt .r em{font-style:normal;margin-left:auto;padding-left:10px}
 footer{margin-top:48px;padding:17px 19px;background:var(--card);
  border:1px solid var(--line);border-radius:10px;color:var(--mut);
  font-size:.74rem;line-height:1.5}
 @media(max-width:920px){
  .sub3{grid-template-columns:1fr}
 }
 .only-m{display:none}
 @media(max-width:760px){
  .only-d{display:none}
  .only-m{display:block}
  .wrap{padding:22px 14px 64px}
  header{padding:25px 20px 22px}
  .fam-ul,.grid,.sub1,.sub2,.sub3{grid-template-columns:1fr}
  .fam-ul li:last-child:nth-child(odd){grid-column:auto}
  .experiment{padding-top:38px}
  .card{padding:14px 12px}
 }
"""

HOVER_JS = """
document.addEventListener('click',e=>{const f=e.target.closest('.flip');
 if(f&&!e.target.closest('a'))f.classList.toggle('flipped')});
function fmtv(v,st,hi){if(hi>=1e6)return (v/1e6).toFixed(2)+'M';
 if(hi>=2e3)return (v/1e3).toFixed(1)+'k';if(st>=1)return v.toFixed(0);
 return v.toFixed(Math.min(4,Math.max(2,1-Math.floor(Math.log10(st)))))}
document.querySelectorAll('.ch[data-h]').forEach(ch=>{
 const d=JSON.parse(ch.dataset.h),svg=ch.querySelector('svg');
 const vb=svg.viewBox.baseVal,ml=d.ml,mr=d.mr;
 const xh=document.createElement('div');xh.className='xh';
 const tt=document.createElement('div');tt.className='tt';
 ch.appendChild(xh);ch.appendChild(tt);
 function at(s,t){let v=s.s0;for(const p of s.p){if(p[0]>t)break;v=p[1]}return v}
 ch.addEventListener('mousemove',e=>{
  const r=ch.getBoundingClientRect(),sc=r.width/vb.width;
  const x0=ml*sc,x1=r.width-mr*sc,px=e.clientX-r.left;
  if(px<x0||px>x1){xh.style.opacity=0;tt.style.opacity=0;return}
  const t=60*(px-x0)/(x1-x0);
  xh.style.left=px+'px';xh.style.opacity=.6;
  let rows='';for(const s of d.s)rows+=`<span class="r"><i style="--c:${s.c}"></i>`
   +`${s.n}<em>${fmtv(at(s,t),d.step,d.hi)}</em></span>`;
  tt.innerHTML=`<b>${t.toFixed(1)} min</b>${rows}`;
  tt.style.opacity=1;
  const left=Math.min(Math.max(px+12,0),r.width-tt.offsetWidth-4);
  tt.style.left=left+'px';
  tt.style.top=Math.max(6,e.clientY-r.top-tt.offsetHeight-14)+'px'});
 ch.addEventListener('mouseleave',()=>{xh.style.opacity=0;tt.style.opacity=0})});
"""


def back_html(sid, name):
    b = CONTENT.BACKS.get(sid, {}).get(name)
    if not b:
        return ""
    return (f'<div class="face back"><div class="bt"><b>{name}</b>'
            f'<span class="tag">{b["tag"]}</span></div>{b["html"]}</div>')


def fam_html(sid):
    s = CONTENT.SECTIONS[sid]["fam_html"]
    # Corrected time-axis notes (the old ones described raw wall-clock and
    # produced a spurious cliff at 60 min from relaunched runs). Experiment 4
    # keeps its own note: its harness already charges active time directly.
    if sid not in ("harder-tasks", "experiment-1"):
        new_axis = ("<li><span class=\"tag\">time axis</span> Optimizer-active time "
                    "from 0 to 60 minutes. A run the campaign launcher interrupted and "
                    "relaunched is stitched at the interruption point, so improvements "
                    "appear at the active minute they occurred; nothing is clamped to "
                    "the 60-minute mark.</li>")
        s = re.sub(r'<li><span class="tag">time axis</span>.*?</li>', new_axis, s,
                   flags=re.S)
    # Recolor the Experiment 3 size markers to the ordinal ramp.
    s = s.replace("#0d9488", C_R4).replace("#7c3aed", C_R8).replace("#ea580c", C_R16)
    return wrap_li_bodies(s)


def wrap_li_bodies(s):
    """Wrap everything after the tag chip in one <span> so inline elements
    inside a bullet don't become separate grid items."""
    def fix(m):
        return (f'<li>{m.group(1)}<span class="li-body">{m.group(2).strip()}'
                f'</span></li>')
    return re.sub(r'<li>\s*(<span class="tag">.*?</span>)(.*?)</li>', fix, s,
                  flags=re.S)


def panel(sid, name, gap, cells, key_html):
    grid = "sub3" if len(cells) == 3 else ("sub1" if len(cells) == 1 else "sub2")
    front = (f'<div class="face front"><div class="ph"><span class="pname">{name}</span>'
             f'<span class="pgap">{gap}</span><span class="chip">details ↗</span></div>'
             f'<div class="pbody"><div class="{grid}">{"".join(cells)}</div>'
             f'{key_html}</div></div>')
    return (f'<figure class="panel flip" title="select for task details">{front}'
            f'{back_html(sid, name)}</figure>')


def section_open(sid):
    s = CONTENT.SECTIONS[sid]
    label = f'<span class="sect-n">{s["sect_n"]}</span>' if s["sect_n"] else ""
    heading = (f'<section class="experiment" id="{sid}">'
               f'{label}<h2>{s["h2"]}</h2>')
    details = fam_html(sid) if s["fam_html"] else ""
    return (heading + f'<ul class="fam-ul">{details}</ul>'
            if details else heading)


def build():
    parts = []

    # ---------- Experiment 1: current seven-task split
    parts.append(section_open("experiment-1"))
    aggregate_items = [(label, color) for label, color in CURRENT_MODELS
                       if all(current_runs(task, label) for task in CURRENT_TASKS)]
    parts.append('<div class="card"><div class="hd">All seven current tasks · '
                 'complete N=5 model series only</div>')
    parts.append('<div class="only-d">' + fig_current_aggregate()
                 + '</div><div class="only-m">'
                 + fig_current_aggregate(w=560, h=360) + '</div>')
    parts.append(legend(aggregate_items))
    parts.append('<div class="tip"><b>Completeness:</b> '
                 f'{current_completion_note()}. The all-task mean is shown only '
                 'when one model has all seven complete task series. Bands are ±1 '
                 'standard deviation across the five trial-index aggregates.'
                 '</div></div>')
    parts.append('<div class="card"><div class="hd">Every current task · raw '
                 'scores</div><div class="panels">')
    for task in CURRENT_TASKS:
        cells, key_html = fig_current_task(task)
        kind = ("perfect information" if task in CURRENT_PERFECT else
                "generalization · online to sealed")
        parts.append(panel("experiment-1", task, kind, cells, key_html))
    parts.append('</div><div class="tip"><b>Complete series only:</b> every plotted '
                 'model/task line contains five trials. The SLM sealed trajectories '
                 'come from the post-run evaluation of every valid submission; '
                 'optimizer-active time remains unchanged.</div></div>')
    parts.append('</section>')

    # ---------- Experiment 2: archived model/reasoning sweep
    parts.append(section_open("experiment-1a"))
    set_items = [(l, c) for l, c, _ in SETTINGS]
    parts.append('<div class="card"><div class="hd">Two perfect-information '
                 'tasks, averaged by model and reasoning setting</div>')
    parts.append('<div class="only-d">'
                 + fig_aggregate(SETTINGS, M_PERF, "guide", w=AGG_W, h=AGG_H)
                 + '</div><div class="only-m">'
                 + fig_aggregate(SETTINGS, M_PERF, "guide", w=560, h=360)
                 + '</div>')
    parts.append(legend(set_items))
    parts.append('<div class="tip"><b>Reading the bands:</b> shaded regions are '
                 '±1 standard deviation of the benchmark aggregate over '
                 'independent per-task run choices.</div></div>')
    parts.append('<div class="card"><div class="hd">Every perfect-information '
                 'task, 20 runs each (5 per setting), raw scores</div>')
    parts.append(legend(set_items) + curve_key())
    parts.append('<div class="grid">')
    for t in PERFECT:
        scale_note = ('<span class="pgap">post-first scale · starter clipped</span>'
                      if t in POST_FIRST_SCALE else '')
        front = (f'<div class="face front"><div class="mh"><span class="mt">{t}</span>'
                 f'{scale_note}<span class="chip">details ↗</span></div>'
                 f'{fig_mini(t, SETTINGS, M_PERF)}</div>')
        parts.append(f'<figure class="mini flip" title="select for task details">'
                     f'{front}{back_html("experiment-1a", t)}</figure>')
    parts.append('</div><div class="tip"><b>Task details:</b> select any task '
                 'card to reveal its objective and scoring method.</div></div>')
    parts.append('</section>')

    # ---------- Experiment 3: archived train/test comparison
    parts.append(section_open("experiment-1b"))
    parts.append('<div class="card"><div class="hd">Two retained original generalization tasks, '
                 'train set versus sealed test</div><div class="sub2">')
    for split, title in (("train", "Training · graded"), ("test", "Sealed test")):
        parts.append(f'<div><div class="ct">{title}</div>'
                     f'{fig_aggregate(GEN_SETTINGS, M_GEN, split)}</div>')
    parts.append('</div>' + legend(set_items) + '</div>')
    parts.append('<div class="card"><div class="hd">Per task, train (graded) and '
                 'sealed test, raw scores</div>')
    parts.append(legend(set_items) + curve_key())
    parts.append('<div class="panels">')
    for t in GEN:
        rbs = [(l, c, M_GEN[l].get(t, [])) for l, c, _ in GEN_SETTINGS]
        cells = fig_task_panel(t, rbs, ["train", "test"],
                               ["Training · graded", "Sealed test"], METRIC[t])
        parts.append(panel("experiment-1b", t, "train to sealed test", cells, ""))
    parts.append('</div></div></section>')

    # ---------- Experiment 4: archived feedback comparison
    parts.append(section_open("experiment-2"))
    e2_items = [("visible train grading", C_VIS), ("hidden validation grading", C_HID)]
    vis = {"visible": {t: M_GEN["gpt-5.5 low"].get(t, []) for t in GEN}}
    hid = {"hidden": {t: M_E2["hidden"].get(f"{t}_e2", []) for t in GEN}}
    parts.append('<div class="card"><div class="hd">Two retained original generalization tasks, '
                 'train, validation, and sealed test</div><div class="sub3">')
    e2_settings_v = [("visible train grading", C_VIS, None)]
    e2_settings_h = [("hidden validation grading", C_HID, None)]
    both = {"visible train grading": vis["visible"], "hidden validation grading": hid["hidden"]}
    for split, title, series in (
            ("train", "Training", ["visible train grading", "hidden validation grading"]),
            ("val", "Hidden validation · graded", ["hidden validation grading"]),
            ("test", "Sealed test", ["visible train grading", "hidden validation grading"])):
        settings = [(l, C_VIS if l.startswith("visible") else C_HID, None)
                    for l in series]
        parts.append(f'<div><div class="ct">{title}</div>'
                     f'{fig_aggregate(settings, both, split, w=TRI_W, h=TRI_H)}</div>')
    parts.append('</div>' + legend(e2_items) + '</div>')
    parts.append('<div class="card"><div class="hd">Per task, train, hidden '
                 'validation (graded for the hidden arm), and sealed test, raw scores</div>')
    parts.append(legend(e2_items) + curve_key("condition mean"))
    parts.append('<div class="panels">')
    for t in GEN:
        rbs = [("visible train grading", C_VIS, vis["visible"].get(t, [])),
               ("hidden validation grading", C_HID, hid["hidden"].get(t, []))]
        cells = fig_task_panel(t, rbs, ["train", "val", "test"],
                               ["Training", "Hidden validation · graded", "Sealed test"],
                               METRIC[t])
        parts.append(panel("experiment-2", t, "visible vs hidden", cells, ""))
    parts.append('</div></div></section>')

    # ---------- Experiment 5: archived size sweep
    parts.append(section_open("experiment-3"))
    e3_items = [("1:4 (largest train)", C_R4), ("1:8", C_R8), ("1:16 (smallest)", C_R16)]
    sizes = {"1:4 (largest train)": {t: M_GEN["gpt-5.5 low"].get(t, []) for t in GEN},
             "1:8": {t: M_R8["1:8"].get(f"{t}_r8", []) for t in GEN},
             "1:16 (smallest)": {t: M_R16["1:16"].get(f"{t}_r16", []) for t in GEN}}
    e3_settings = [("1:4 (largest train)", C_R4, None), ("1:8", C_R8, None),
                   ("1:16 (smallest)", C_R16, None)]
    parts.append('<div class="card"><div class="hd">Two retained original generalization tasks, '
                 'train-set size sweep</div><div class="sub2">')
    for split, title in (("train", "Training · graded"), ("test", "Sealed test")):
        parts.append(f'<div><div class="ct">{title}</div>'
                     f'{fig_aggregate(e3_settings, sizes, split)}</div>')
    parts.append('</div>' + legend(e3_items) + '</div>')
    parts.append('<div class="card"><div class="hd">Per task, train (graded) and '
                 'sealed test, one line per training-set size, raw scores</div>')
    parts.append(legend(e3_items) + curve_key("size mean"))
    parts.append('<div class="panels">')
    for t in GEN:
        rbs = [(l, c, sizes[l].get(t, [])) for l, c, _ in e3_settings]
        cells = fig_task_panel(t, rbs, ["train", "test"],
                               ["Training · graded", "Sealed test"], METRIC[t])
        parts.append(panel("experiment-3", t, "train:test 1:4 / 1:8 / 1:16", cells, ""))
    parts.append('</div></div></section>')

    # ---------- Archived fixed-method study for the revised SLM protocol
    parts.append(section_open("harder-tasks"))
    parts.append('<div class="panels">')
    behavior_baselines = {
        "Online validation": [
            ("BF16 native", 0.0), ("RTN W3 starter", 0.916667),
            ("HQQ W3", 0.816667), ("AQLM 3×8", 0.816667),
            ("optimized QWeight", 0.516667),
        ],
        "Sealed test": [
            ("BF16 native", 0.0), ("HQQ W3", 0.866667),
            ("AQLM 3×8", 0.833333), ("optimized QWeight", 0.466667),
        ],
    }
    behavior_cells = []
    for title, rows in behavior_baselines.items():
        body = "".join(
            f'<div class="base-row"><span>{label}</span><b>{score:.4f}</b></div>'
            for label, score in rows)
        behavior_cells.append(f'<div><div class="ct">{title}</div>{body}</div>')
    parts.append(panel(
        "harder-tasks", "slm_weight_compression_lfm25",
        "aggregate method study · ≤3.5 physical BPW", behavior_cells,
        '<div class="key key-sub"><span>BF16 behavioral regression rate · '
        'lower is better</span><span>fixed methods · not agent runs</span></div>'))
    parts.append('</div>')
    parts.append('<p class="tip"><b>Study semantics:</b> each row is one fixed '
                 'compression method evaluated on the revised protocol. It is '
                 'retained as protocol context and is not mixed into the current '
                 'N=5 agent results.</p>')
    parts.append('</section>')

    footer = CONTENT.FOOTER_HTML
    footer_html = f"<footer>{footer}</footer>" if footer.strip() else ""

    html = f"""<!doctype html>
<!-- GENERATED FILE — do not hand-edit.
     Rebuild with: python3 tools/make_blogpost.py
     Charts/layout: tools/make_blogpost.py · prose: tools/blogpost_content.py
     Archived LFM traces: tools/blogpost_exp4_data.py -->
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>text-opt-bm</title>
<style>{CSS}</style></head><body><div class="wrap">
<header>{wrap_li_bodies(CONTENT.HEADER_HTML)}</header>
<main>{"".join(parts)}</main>
{footer_html}
</div><script>{HOVER_JS}</script></body></html>"""
    return html


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-o", "--out", default=str(ROOT / "docs/blogpost.html"))
    args = ap.parse_args()
    html = build()
    Path(args.out).write_text(html)
    print(f"[blogpost] wrote {args.out} ({len(html)/1024:.0f} KB)")


if __name__ == "__main__":
    main()
