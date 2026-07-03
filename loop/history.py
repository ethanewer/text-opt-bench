"""Git-based attempt history for the bundled optimization loop.

This is an *optimizer-side* mechanism, not part of the benchmark: the
benchmark's canonical record is the session's `submissions.jsonl` (see
`bench/session.py`). The loop additionally maintains a per-run git repo
because agents already know how to browse git trees — `main` is the
lineage of accepted improvements, `attempts/iter-NNN` hold rejected and
invalid attempts, and commit messages carry scores/errors/agent notes.
Any other optimizer can ignore this module entirely.
"""

import os
import shutil
import subprocess

# Environment for all harness git calls: ignore user/system config (and
# with it any hooks/aliases), never prompt, fixed identity.
GIT_ENV = {
    "GIT_CONFIG_GLOBAL": "/dev/null",
    "GIT_CONFIG_SYSTEM": "/dev/null",
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_TERMINAL_PROMPT": "0",
    "GIT_AUTHOR_NAME": "textopt-harness",
    "GIT_AUTHOR_EMAIL": "harness@textopt.invalid",
    "GIT_COMMITTER_NAME": "textopt-harness",
    "GIT_COMMITTER_EMAIL": "harness@textopt.invalid",
}

WORKSPACE_GITIGNORE = ("PROMPT.md\ncodex_*.txt\n__pycache__/\n"
                       "evals.jsonl\neval_programs/\n")


class HistoryRepo:
    """Authoritative git history of a run, maintained ONLY by the harness.

    Safety model: the agent must never be able to break the recorded
    history, so
      - the repo is bare and lives in the run dir, outside the agent's
        sandbox-writable workspace;
      - all writes go through plumbing (hash-object/mktree/commit-tree/
        update-ref): no working tree, no index, no checkouts, and hooks
        never run; ref updates are atomic;
      - agent workspaces are clones via a file:// URL (which forces a
        full object copy — a plain local-path clone would HARDLINK object
        files, letting workspace tampering corrupt this repo);
      - the harness never reads git state back from a workspace; it reads
        program.py as plain bytes and commits the result here itself.
    A trashed workspace clone therefore costs nothing: the next iteration
    clones fresh from this repo. If git is unavailable, the loop degrades
    to plain directories (enabled=False).
    """

    def __init__(self, run_dir):
        self.path = run_dir / "history.git"
        self.enabled = shutil.which("git") is not None

    def _git(self, *args, input_bytes=None, cwd=None):
        env = os.environ.copy()
        env.update(GIT_ENV)
        proc = subprocess.run(
            ["git", *args], input=input_bytes, capture_output=True,
            env=env, cwd=cwd, timeout=120,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"git {' '.join(map(str, args))} failed: "
                f"{proc.stderr.decode(errors='replace')[:500]}"
            )
        return proc.stdout.decode(errors="replace").strip()

    def _commit_tree(self, program_source, message, parent=None):
        blob = self._git("-C", str(self.path), "hash-object", "-w", "--stdin",
                         input_bytes=program_source.encode())
        ignore = self._git("-C", str(self.path), "hash-object", "-w", "--stdin",
                           input_bytes=WORKSPACE_GITIGNORE.encode())
        tree = self._git(
            "-C", str(self.path), "mktree",
            input_bytes=(
                f"100644 blob {ignore}\t.gitignore\n"
                f"100644 blob {blob}\tprogram.py\n"
            ).encode(),
        )
        args = ["-C", str(self.path), "commit-tree", tree]
        if parent:
            args += ["-p", parent]
        return self._git(*args, input_bytes=message.encode())

    def init(self, program_source, message):
        if not self.enabled:
            return
        try:
            # No --initial-branch (git < 2.28 compat): set HEAD explicitly.
            self._git("init", "--quiet", "--bare", str(self.path))
            self._git("-C", str(self.path), "symbolic-ref", "HEAD",
                      "refs/heads/main")
            # Reject ALL pushes: even if the loop is ever run without the
            # codex sandbox, `git push -f origin` from a workspace clone
            # must not be able to rewrite history. Harness writes use
            # update-ref plumbing, which never goes through the receive
            # path, so this does not affect the harness.
            hooks = self.path / "hooks"
            hooks.mkdir(exist_ok=True)
            hook = hooks / "pre-receive"
            hook.write_text(
                "#!/bin/sh\n"
                "echo 'history.git is read-only: attempts are recorded by "
                "the harness' >&2\n"
                "exit 1\n"
            )
            hook.chmod(0o755)
            sha = self._commit_tree(program_source, message)
            self._git("-C", str(self.path), "update-ref",
                      "refs/heads/main", sha)
        except RuntimeError as e:
            print(f"[loop] git history disabled: {e}")
            self.enabled = False

    def main_tip(self):
        return self._git("-C", str(self.path), "rev-parse", "refs/heads/main")

    def exists(self):
        return (self.path / "HEAD").is_file()

    def attempt_refs(self):
        """Names of existing attempts/* branches (for resumed runs)."""
        if not self.enabled or not self.exists():
            return []
        try:
            out = self._git("-C", str(self.path), "for-each-ref",
                            "--format=%(refname:short)", "refs/heads/attempts")
            return [line for line in out.splitlines() if line.strip()]
        except RuntimeError:
            return []

    def record(self, program_source, message, iter_no, accepted):
        """Commit an attempt: accepted -> advance main; else attempts branch."""
        if not self.enabled:
            return
        try:
            parent = self.main_tip()
            sha = self._commit_tree(program_source, message, parent=parent)
            ref = ("refs/heads/main" if accepted
                   else f"refs/heads/attempts/iter-{iter_no:03d}")
            args = ["-C", str(self.path), "update-ref", ref, sha]
            if accepted:
                args.append(parent)  # compare-and-swap: fail on unexpected tip
            self._git(*args)
        except RuntimeError as e:
            print(f"[loop] git history disabled: {e}")
            self.enabled = False

    def clone_workspace(self, ws):
        """Create the agent workspace as a disposable full clone."""
        if not self.enabled:
            ws.mkdir(exist_ok=True)
            return
        try:
            self._git("clone", "--quiet", f"file://{self.path.resolve()}",
                      str(ws))
        except RuntimeError as e:
            print(f"[loop] git history disabled: {e}")
            self.enabled = False
            ws.mkdir(exist_ok=True)
