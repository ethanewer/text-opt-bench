"""Archived resource-aware launcher for CPU-only research prototypes.

Active SLM work is intentionally excluded. It must run through the strict-MPS
benchmark/compiler/native paths, which share the repository-wide Metal lease.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = Path(__file__).with_name("manifest.json")


def available_memory_gb():
    try:
        if sys.platform == "darwin":
            value = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True)
            return int(value) / 2 ** 30
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return pages * page_size / 2 ** 30
    except (OSError, ValueError, subprocess.SubprocessError):
        return 16.0


def choose_device(requested):
    if requested != "auto":
        return requested
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def can_launch(spec, running, cpu_capacity, accelerator_capacity, memory_capacity):
    used_cpu = sum(job["spec"].get("cpu_slots", 1) for job in running)
    used_accelerators = sum(job["spec"].get("accelerator_slots", 0) for job in running)
    used_memory = sum(job["spec"].get("memory_gb", 1.0) for job in running)
    return (used_cpu + spec.get("cpu_slots", 1) <= cpu_capacity and
            used_accelerators + spec.get("accelerator_slots", 0) <= accelerator_capacity and
            used_memory + spec.get("memory_gb", 1.0) <= memory_capacity)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--tasks", default="all", help="comma-separated names")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--device", default="cpu", choices=["cpu"],
                        help="archived prototype launcher is CPU-only")
    parser.add_argument("--cpu-slots", type=int, default=2)
    parser.add_argument("--memory-fraction", type=float, default=.70)
    parser.add_argument("--router-data", default="/tmp/routerbench_0shot.pkl")
    parser.add_argument("--text-rows", default="/tmp/text-opt-bm-tinystories-rows.json")
    parser.add_argument("--qwen35-model", default="/tmp/qwen35-08b")
    parser.add_argument("--qwen3-model", default="/tmp/qwen3-06b")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    accelerator_slots = 1

    manifest = json.loads(args.manifest.read_text())["tasks"]
    accelerator_tasks = sorted(
        name for name, spec in manifest.items()
        if spec.get("resource") == "accelerator")
    if accelerator_tasks:
        parser.error(
            "this archived prototype launcher is CPU-only; remove accelerator "
            f"entries: {', '.join(accelerator_tasks)}")
    names = list(manifest) if args.tasks == "all" else [x.strip() for x in args.tasks.split(",") if x.strip()]
    unknown = sorted(set(names) - set(manifest))
    if unknown:
        parser.error(f"unknown tasks: {', '.join(unknown)}")
    # Put an accelerator job first so its model load overlaps useful CPU work.
    names.sort(key=lambda name: manifest[name]["resource"] != "accelerator")
    device = "cpu"
    memory_capacity = available_memory_gb() * args.memory_fraction
    stamp = time.strftime("%Y%m%d-%H%M%S")
    output_dir = args.output_dir or ROOT / "runs" / "research_experiments" / stamp
    context = {
        "python": args.python, "root": str(ROOT), "device": device,
        "router_data": args.router_data, "text_rows": args.text_rows,
        "qwen35_model": args.qwen35_model, "qwen3_model": args.qwen3_model,
    }
    summary = {
        "device": device, "cpu_slots": args.cpu_slots,
        "accelerator_slots": accelerator_slots,
        "memory_capacity_gb": memory_capacity, "tasks": {},
    }
    print(json.dumps({"schedule": names, **{k: summary[k] for k in
          ["device", "cpu_slots", "accelerator_slots", "memory_capacity_gb"]}}, sort_keys=True))
    if args.dry_run:
        for name in names:
            print(json.dumps({"task": name, "resource": manifest[name]["resource"],
                              "command": [part.format(**context) for part in manifest[name]["command"]]}))
        return

    required = {}
    if "llm_routing" in names:
        required["RouterBench pickle (--router-data)"] = Path(args.router_data)
    if {"kv_policy", "slm_compression"} & set(names):
        required["language-model text rows (--text-rows)"] = Path(args.text_rows)
    if "kv_policy" in names:
        required["Qwen3 model directory (--qwen3-model)"] = Path(args.qwen3_model)
    if "slm_compression" in names:
        required["Qwen3.5 model directory (--qwen35-model)"] = Path(args.qwen35_model)
    missing = [f"{label}: {path}" for label, path in required.items() if not path.exists()]
    if missing:
        parser.error("missing experiment assets:\n  " + "\n  ".join(missing))

    output_dir.mkdir(parents=True, exist_ok=True)
    pending = list(names)
    running = []
    while pending or running:
        launched = False
        for name in list(pending):
            spec = manifest[name]
            if not can_launch(spec, running, args.cpu_slots,
                              accelerator_slots, memory_capacity):
                continue
            command = [part.format(**context) for part in spec["command"]]
            log_path = output_dir / f"{name}.log"
            log = log_path.open("w")
            env = os.environ.copy()
            threads = str(max(1, (os.cpu_count() or 2) // max(1, args.cpu_slots)))
            env.update({"OMP_NUM_THREADS": threads, "MKL_NUM_THREADS": threads,
                        "OPENBLAS_NUM_THREADS": threads, "TOKENIZERS_PARALLELISM": "false"})
            process = subprocess.Popen(command, cwd=ROOT, stdout=log,
                                       stderr=subprocess.STDOUT, env=env)
            running.append({"name": name, "spec": spec, "process": process,
                            "log": log, "log_path": str(log_path), "start": time.time()})
            pending.remove(name)
            launched = True
            print(f"[experiments] launch {name} pid={process.pid}", flush=True)
        still = []
        for job in running:
            code = job["process"].poll()
            if code is None:
                still.append(job)
                continue
            job["log"].close()
            elapsed = time.time() - job["start"]
            summary["tasks"][job["name"]] = {
                "returncode": code, "seconds": elapsed, "log": job["log_path"]}
            print(f"[experiments] finish {job['name']} rc={code} {elapsed:.1f}s", flush=True)
        running = still
        if pending or running:
            if not launched and not running:
                blocked = ", ".join(pending)
                raise RuntimeError(f"resource limits cannot schedule: {blocked}")
            time.sleep(.2)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    if any(value["returncode"] for value in summary["tasks"].values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
