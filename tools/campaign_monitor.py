"""Emit selective campaign events for the agent to act on.

stdout lines = notifications. Emits:
  - each NEW alert from campaign_watch (STALL/INVALID3/AUDIT/EVALCRASH/...)
  - a STATUS heartbeat every HEARTBEAT_S (with a campaign_watch snapshot,
    which also persists a timeline JSON under runs/_campaign/status/)
  - CAMPAIGN DONE, then exits.

Usage: python3.12 tools/campaign_monitor.py --prefix 5xB- --total 60
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LAUNCH = ROOT / "runs" / "_campaign" / "launcher.jsonl"
HEARTBEAT_S = 900
POLL_S = 90


def emit(msg):
    print(msg, flush=True)


def launcher_state():
    done = running = 0
    campaign_done = False
    if LAUNCH.exists():
        for line in LAUNCH.read_text().splitlines():
            try:
                e = json.loads(line)
            except ValueError:
                continue
            ev = e.get("event")
            if ev in ("finish", "timeout"):
                done += 1
            elif ev == "launch":
                running += 1
            elif ev == "campaign_done":
                campaign_done = True
    return done, max(0, running - done), campaign_done


def watch(prefix, alerts_only):
    cmd = [sys.executable, str(ROOT / "tools" / "campaign_watch.py"), "--prefix", prefix]
    if alerts_only:
        cmd.append("--alerts-only")
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=120, cwd=ROOT)
        return (out.stdout or "").strip()
    except Exception as e:
        return f"(campaign_watch failed: {e})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="5xB-")
    ap.add_argument("--total", type=int, default=60)
    args = ap.parse_args()

    emit(f"MONITOR armed: prefix={args.prefix} total={args.total} jobs")
    seen_alerts = set()
    last_hb = 0.0
    ticks = 0
    while True:
        ticks += 1
        alerts = watch(args.prefix, alerts_only=True)
        for line in alerts.splitlines():
            line = line.strip()
            if line and line not in seen_alerts:
                seen_alerts.add(line)
                emit(f"ALERT: {line}")

        done, running, campaign_done = launcher_state()
        now = time.time()
        if now - last_hb >= HEARTBEAT_S:
            last_hb = now
            watch(args.prefix, alerts_only=False)  # full run -> snapshot on disk
            emit(f"STATUS done={done}/{args.total} running={running} "
                 f"alerts_seen={len(seen_alerts)}")

        if campaign_done:
            watch(args.prefix, alerts_only=False)
            emit(f"CAMPAIGN DONE done={done}/{args.total}")
            return
        time.sleep(POLL_S)


if __name__ == "__main__":
    main()
