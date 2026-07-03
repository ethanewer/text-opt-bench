"""Safety tests for the loop's git-based attempt history.

The invariant under test: NOTHING an agent does inside its workspace
clone can corrupt or rewrite the authoritative history repo.

Run with:  python3.12 tests/test_history_repo.py
"""

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from loop.history import GIT_ENV, HistoryRepo

failures = []


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    if not cond:
        failures.append(name)


def git(cwd, *args):
    env = dict(os.environ, **GIT_ENV)
    return subprocess.run(
        ["git", "-C", str(cwd), *args], capture_output=True, text=True, env=env
    )


def main():
    tmp = Path(tempfile.mkdtemp(prefix="textopt_hist_test_"))
    try:
        run(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print()
    if failures:
        sys.exit(f"{len(failures)} check(s) FAILED: {failures}")
    print("all history-repo safety checks passed")


def run(tmp):
    h = HistoryRepo(tmp)
    h.init("v0\n", "iter 0: baseline, score 100\n")
    check("init", h.enabled and (tmp / "history.git").is_dir())

    # Normal lifecycle: reject, accept, and a no-change (empty-diff) record.
    h.record("v1-bad\n", "iter 1: score 120 rejected (best 100)\n", 1, False)
    h.record("v2\n", "iter 2: score 80 ACCEPTED\n", 2, True)
    h.record("v2\n", "iter 3: INVALID\n\nerror: codex timed out\n", 3, False)

    ws = tmp / "iter_004"
    h.clone_workspace(ws)
    check("clone tracks main", (ws / "program.py").read_text() == "v2\n")
    log = git(ws, "log", "--oneline", "--all").stdout
    check("all attempts visible in clone",
          all(s in log for s in ("baseline", "rejected", "ACCEPTED", "INVALID")))

    # No hardlinks: object files in the clone must not share inodes with
    # the authoritative repo (a hardlinked clone would let tampering
    # corrupt history.git's object store).
    auth_inodes = {
        p.stat().st_ino
        for p in (tmp / "history.git" / "objects").rglob("*") if p.is_file()
    }
    clone_inodes = {
        p.stat().st_ino
        for p in (ws / ".git" / "objects").rglob("*") if p.is_file()
    }
    check("clone shares no object inodes", not (auth_inodes & clone_inodes))

    # Vandalism: corrupt objects, rewrite refs, commit junk, delete .git.
    git(ws, "commit", "--allow-empty", "-am", "evil commit")
    push = git(ws, "push", "--force", "origin", "main")
    check("force-push rejected by pre-receive hook",
          push.returncode != 0 and "read-only" in (push.stderr + push.stdout))
    git(ws, "branch", "-f", "main", "HEAD")
    for p in (ws / ".git" / "objects").rglob("*"):
        if p.is_file():
            p.chmod(0o644)
            p.write_bytes(b"corrupted")
    shutil.rmtree(ws / ".git")
    (ws / "program.py").write_text("vandalized\n")

    # The authoritative repo must be intact and fully usable afterwards.
    h.record("v4\n", "iter 4: score 60 ACCEPTED\n", 4, True)
    check("still enabled after vandalism", h.enabled)
    fsck = git(tmp / "history.git", "fsck", "--strict")
    check("history.git fsck clean", fsck.returncode == 0,
          (fsck.stdout + fsck.stderr).strip()[:120])
    ws2 = tmp / "iter_005"
    h.clone_workspace(ws2)
    check("fresh clone has new best",
          (ws2 / "program.py").read_text() == "v4\n")
    log2 = git(ws2, "log", "--oneline", "--all").stdout
    check("evil commit never reached origin", "evil" not in log2)

    # Graceful degradation: with git "missing", the loop falls back to
    # plain directories and never raises.
    h2 = HistoryRepo(tmp / "nogit")
    h2.enabled = False
    h2.init("x\n", "m\n")
    h2.record("y\n", "m\n", 1, True)
    ws3 = tmp / "nogit_ws"
    h2.clone_workspace(ws3)
    check("degrades to plain dir without git", ws3.is_dir())


if __name__ == "__main__":
    main()
