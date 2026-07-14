"""Low-overhead process/memory telemetry for a running campaign."""

import argparse
import json
from pathlib import Path
import subprocess
import time


def snapshot(root_pid):
    text = subprocess.check_output(
        ["ps", "-axo", "pid=,ppid=,rss=,%cpu=,command="], text=True)
    rows = []
    for line in text.splitlines():
        fields = line.strip().split(None, 4)
        if len(fields) != 5:
            continue
        try:
            rows.append((int(fields[0]), int(fields[1]), int(fields[2]),
                         float(fields[3]), fields[4]))
        except ValueError:
            pass
    children = {root_pid}
    changed = True
    while changed:
        changed = False
        for pid, ppid, *_ in rows:
            if ppid in children and pid not in children:
                children.add(pid)
                changed = True
    local = [row for row in rows if row[0] in children]
    evaluators = [row for row in local if "/bench/tasks/" in row[4]]
    lfm = [row for row in evaluators if "slm_compression_" in row[4]]
    vm = subprocess.check_output(["sysctl", "-n", "vm.swapusage"], text=True).strip()
    return {
        "t": time.time(), "campaign_alive": root_pid in children and
        any(row[0] == root_pid for row in rows),
        "descendants": max(0, len(local) - 1),
        "campaign_rss_mib": round(sum(row[2] for row in local) / 1024, 2),
        "campaign_cpu_percent": round(sum(row[3] for row in local), 2),
        "evaluators": len(evaluators), "lfm_evaluators": len(lfm),
        "lfm_rss_mib": round(sum(row[2] for row in lfm) / 1024, 2),
        "load_average": list(__import__("os").getloadavg()), "swapusage": vm,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pid", type=int)
    parser.add_argument("output", type=Path)
    parser.add_argument("--interval", type=float, default=5.0)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a", buffering=1) as handle:
        while True:
            row = snapshot(args.pid)
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            if not row["campaign_alive"]:
                break
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
