"""Campaign monitor: status, anomaly detection, and data snapshots.

Scans runs/<task>/<prefix>*/ (default prefix "5x-"), parses log.jsonl,
prints a status table, and appends a full JSON snapshot to
runs/_campaign/status/ so the campaign's timeline is preserved even if
individual artifacts are later lost.

Usage:
    python3.12 tools/campaign_watch.py            # human status + snapshot
    python3.12 tools/campaign_watch.py --alerts-only   # alert lines only

Alert codes (one line per alert, stable text for diffing):
    STALL       no filesystem activity in the run dir beyond threshold
    INVALID3    >=3 consecutive invalid iterations
    NOCHANGE3   >=3 consecutive "codex made no change" iterations
    EVALCRASH   an iteration error says the evaluator produced no result
    TIMEOUT2    >=2 codex timeouts in one run
    CODEXERR2   >=2 nonzero codex exits in one run
    AUDIT       accepted score improved >500x in one step (verify by hand)
    SATURATED   >=5 consecutive rejections with identical candidate score
    DONE        run reached its expected iteration count
"""

import argparse
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

EXPECTED_ITERS = {"word_problems": 10}
DEFAULT_ITERS = 15
STALL_S = {"word_problems": 2700}
DEFAULT_STALL_S = 1800


def newest_mtime(run_dir):
    newest = 0.0
    for p in [run_dir / "log.jsonl", run_dir / "submissions.jsonl",
              *run_dir.glob("iter_*"), *run_dir.glob("iter_*/PROMPT.md"),
              *run_dir.glob("iter_*/codex_stdout.txt")]:
        try:
            newest = max(newest, p.stat().st_mtime)
        except OSError:
            pass
    return newest


def scan_run(run_dir, now):
    task = run_dir.parent.name
    expected = EXPECTED_ITERS.get(task, DEFAULT_ITERS)
    out = {
        "task": task, "run": run_dir.name, "expected_iters": expected,
        "baseline": None, "iters": [], "alerts": [],
    }
    log = run_dir / "log.jsonl"
    if not log.exists():
        out["alerts"].append("NOLOG no log.jsonl yet")
        return out
    for line in log.read_text().splitlines():
        if not line.strip():
            continue
        e = json.loads(line)
        if e.get("event") == "baseline":
            out["baseline"] = e.get("guide_score")
        elif "accepted" in e:
            out["iters"].append({
                "iter": e["iter"], "ok": e.get("ok"),
                "score": e.get("guide_score"),
                "accepted": e.get("accepted"),
                "best": e.get("best_guide_score"),
                "seconds": e.get("seconds"),
                "error": (" ".join(str(e.get("error")).split())[:160]
                          if e.get("error") else None),
            })

    iters = out["iters"]
    done = len(iters) >= expected
    out["done"] = done
    name = f"{task}/{run_dir.name}"

    # Consecutive-pattern detectors.
    inv = nochange = 0
    same_reject = 0
    last_reject_score = None
    timeouts = codexerr = evalcrash = 0
    prev_best = out["baseline"]
    for it in iters:
        err = it["error"] or ""
        inv = inv + 1 if not it["ok"] else 0
        nochange = nochange + 1 if "made no change" in err else 0
        if "codex timed out" in err:
            timeouts += 1
        if "codex exited" in err:
            codexerr += 1
        if "evaluator produced no result" in err:
            evalcrash += 1
        if it["ok"] and not it["accepted"]:
            if it["score"] == last_reject_score:
                same_reject += 1
            else:
                same_reject = 1
                last_reject_score = it["score"]
        elif it["ok"]:
            same_reject = 0
            last_reject_score = None
        if (it["accepted"] and prev_best and it["score"]
                and it["score"] > 0 and prev_best / it["score"] > 500):
            out["alerts"].append(
                f"AUDIT {name} iter {it['iter']}: jump "
                f"{prev_best:g} -> {it['score']:g} (verify by hand)")
        if it["accepted"]:
            prev_best = it["score"]
    if inv >= 3:
        out["alerts"].append(f"INVALID3 {name}: consecutive invalid iters # n={inv}")
    if nochange >= 3:
        out["alerts"].append(f"NOCHANGE3 {name}: consecutive no-change iters # n={nochange}")
    if evalcrash:
        out["alerts"].append(f"EVALCRASH {name}: evaluator produced no result # n={evalcrash}")
    if timeouts >= 2:
        out["alerts"].append(f"TIMEOUT2 {name}: codex timeouts # n={timeouts}")
    if codexerr >= 2:
        out["alerts"].append(f"CODEXERR2 {name}: codex nonzero exits # n={codexerr}")
    if same_reject >= 5:
        out["alerts"].append(
            f"SATURATED {name}: identical rejections at {last_reject_score:g} "
            f"# n={same_reject}")
    if done:
        out["alerts"].append(f"DONE {name}")
    else:
        age = now - newest_mtime(run_dir)
        limit = STALL_S.get(task, DEFAULT_STALL_S)
        if age > limit:
            out["alerts"].append(f"STALL {name}: no recent activity # age={int(age)}s")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="5x-")
    ap.add_argument("--alerts-only", action="store_true")
    ap.add_argument("--no-snapshot", action="store_true")
    args = ap.parse_args()

    now = time.time()
    runs = []
    for run_dir in sorted(ROOT.glob(f"runs/*/{args.prefix}*")):
        if run_dir.is_dir() and run_dir.parent.name != "_campaign":
            runs.append(scan_run(run_dir, now))

    alerts = [a for r in runs for a in r["alerts"]]
    if not args.no_snapshot:
        snap_dir = ROOT / "runs" / "_campaign" / "status"
        snap_dir.mkdir(parents=True, exist_ok=True)
        (snap_dir / f"{int(now)}.json").write_text(
            json.dumps({"t": now, "runs": runs}, indent=1))

    if args.alerts_only:
        for a in sorted(alerts):
            print(a)
        return

    by_task = {}
    for r in runs:
        by_task.setdefault(r["task"], []).append(r)
    print(f"{'task':<20} {'runs':>4} {'done':>4} {'iters':>7} {'acc':>4} "
          f"{'inv':>4}  best per run")
    for task in sorted(by_task):
        rs = by_task[task]
        n_it = sum(len(r["iters"]) for r in rs)
        exp = sum(r["expected_iters"] for r in rs)
        acc = sum(1 for r in rs for i in r["iters"] if i["accepted"])
        inv = sum(1 for r in rs for i in r["iters"] if not i["ok"])
        bests = []
        for r in rs:
            b = [i["best"] for i in r["iters"] if i.get("best") is not None]
            bests.append(f"{b[-1]:g}" if b else
                         (f"{r['baseline']:g}" if r["baseline"] else "-"))
        print(f"{task:<20} {len(rs):>4} {sum(r['done'] for r in rs):>4} "
              f"{n_it:>3}/{exp:<3} {acc:>4} {inv:>4}  {' '.join(bests)}")
    print()
    for a in sorted(alerts):
        print(a)
    if not alerts:
        print("(no alerts)")


if __name__ == "__main__":
    main()
