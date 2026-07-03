"""Shared helpers for task evaluators.

Evaluators run as standalone scripts in a child process and communicate a
single JSON line on stdout:

    {"ok": bool, "score": float|null, "metrics": {...}, "error": str|null}

Lower score is always better. The program's own stdout is redirected to
stderr so it can't corrupt the protocol.
"""

import ast
import json
import sys
import traceback
import types
from pathlib import Path

# Capture the real stdout for the JSON result, then route everything the
# program (or evaluator) prints to stderr.
_REAL_STDOUT = sys.stdout
sys.stdout = sys.stderr


def emit(ok, score=None, metrics=None, error=None):
    line = json.dumps(
        {"ok": ok, "score": score, "metrics": metrics or {}, "error": error}
    )
    _REAL_STDOUT.write(line + "\n")
    _REAL_STDOUT.flush()


def fail(error, metrics=None):
    emit(False, None, metrics, error)
    sys.exit(0)


def succeed(score, metrics=None):
    emit(True, score, metrics, None)
    sys.exit(0)


def scan_forbidden(source, forbidden):
    """Return the first forbidden module/name referenced, else None.

    This is a cooperative guard, not a security sandbox: it catches the
    honest mistakes (importing zlib in the compression task, using ctypes
    to hide allocations from tracemalloc, etc.).
    """
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        fail(f"program has a syntax error: {e}")
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in forbidden:
                    return root
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if node.level == 0 and root in forbidden:
                return root
        elif isinstance(node, ast.Name) and node.id in forbidden:
            return node.id
    return None


def preimport(path):
    """Import every module the program's source names in import statements.

    Memory-scored evaluators call this BEFORE opening the tracemalloc
    window. Loading a module inside the window — C-extension init in
    particular — makes a handful of small address-dependent allocations,
    which flickers resident/peak scores by tens of bytes between runs of
    the identical program. Pre-warming turns the program's in-window
    imports into plain sys.modules lookups. The program still pays for
    every byte it allocates itself; only the fixed module structures
    (which any program using that module would carry identically) are
    kept out of the measurement.
    """
    import importlib

    try:
        source = Path(path).read_text()
        tree = ast.parse(source)
    except (OSError, SyntaxError, ValueError):
        return  # load_program will report the real problem
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module)
            for alias in node.names:
                names.add(f"{node.module}.{alias.name}")
    for name in sorted(names):
        try:
            importlib.import_module(name)
        except BaseException:
            pass  # forbidden/missing imports fail properly at load_program


def load_program(path, forbidden=frozenset(), required=()):
    """AST-check, load, and validate the program module at `path`.

    The module is exec'd from source under the fixed filename "<program>"
    rather than imported through importlib: the import machinery retains
    filesystem path strings and may write .pyc files, which perturbs
    tracemalloc-based scores by a few bytes depending on where the program
    file happens to live.
    """
    path = Path(path)
    try:
        source = path.read_text()
    except OSError as e:
        fail(f"cannot read program: {e}")
    bad = scan_forbidden(source, forbidden)
    if bad is not None:
        fail(f"forbidden import or name for this task: {bad!r}")
    module = types.ModuleType("program")
    module.__file__ = "<program>"
    sys.modules["program"] = module
    try:
        code = compile(source, "<program>", "exec")
        exec(code, module.__dict__)
    except SystemExit:
        fail("program called sys.exit() at import time")
    except BaseException:
        fail("program failed at import time:\n" + traceback.format_exc(limit=8))
    for name in required:
        if not callable(getattr(module, name, None)):
            fail(f"program must define a function named {name!r}")
    return module


def run_program(fn, *args):
    """Call into the program, converting exceptions into a failure result."""
    try:
        return fn(*args)
    except SystemExit:
        fail("program called sys.exit() during evaluation")
    except BaseException:
        fail("program raised during evaluation:\n" + traceback.format_exc(limit=8))
