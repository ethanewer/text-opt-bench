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

from bench import opcount

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


def scan_forbidden(source, forbidden, forbidden_attrs=frozenset()):
    """Return the first forbidden module/name/attribute referenced, else None.

    This is a cooperative guard, not a security sandbox: it catches the
    honest mistakes (importing zlib in the compression task, using ctypes
    to hide allocations from tracemalloc, walking traceback frames toward
    evaluator globals, etc.).
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
        elif isinstance(node, ast.Attribute) and node.attr in forbidden_attrs:
            return "." + node.attr
    return None


# Curated builtins for tasks that run candidates with safe_builtins=True
# (simulation-scored tasks whose metrics are not self-policing). No
# __import__ (import statements fail), no __build_class__ (class
# definitions fail), no introspection helpers. Tasks using this must list
# the allowed names in their spec.
SAFE_BUILTINS = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "filter": filter,
    "float": float,
    "int": int,
    "isinstance": isinstance,
    "len": len,
    "list": list,
    "map": map,
    "max": max,
    "min": min,
    "print": print,
    "range": range,
    "reversed": reversed,
    "round": round,
    "set": set,
    "slice": slice,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
    "BaseException": BaseException,
    "Exception": Exception,
    "IndexError": IndexError,
    "KeyError": KeyError,
    "LookupError": LookupError,
    "RuntimeError": RuntimeError,
    "StopIteration": StopIteration,
    "TypeError": TypeError,
    "ValueError": ValueError,
    "ZeroDivisionError": ZeroDivisionError,
}


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


def load_program(
    path,
    forbidden=frozenset(),
    required=(),
    forbidden_attrs=frozenset(),
    safe_builtins=False,
    import_budget=None,
    max_source_bytes=None,
    max_literal_items=None,
    max_total_literal_items=None,
    max_string_literal_bytes=None,
):
    """AST-check, load, and validate the program module at `path`.

    The module is exec'd from source under the fixed filename "<program>"
    rather than imported through importlib: the import machinery retains
    filesystem path strings and may write .pyc files, which perturbs
    tracemalloc-based scores by a few bytes depending on where the program
    file happens to live.

    Hardening options for simulation-scored tasks (whose metrics are not
    self-policing the way memory/size metrics are):
      - safe_builtins: exec the module with a curated builtins subset
        (no imports, no class definitions, no introspection);
      - import_budget: deterministic bytecode-instruction budget on the
        module's import-time code (blocks import-time precomputation);
      - max_source_bytes / max_literal_items / max_total_literal_items /
        max_string_literal_bytes: block hardcoded answer tables while
        leaving compact algorithmic solutions room.
    """
    path = Path(path)
    try:
        source = path.read_text()
    except OSError as e:
        fail(f"cannot read program: {e}")
    if max_source_bytes is not None and len(source.encode("utf-8")) > max_source_bytes:
        fail(f"program source is too large "
             f"({len(source.encode('utf-8'))} > {max_source_bytes} bytes)")
    bad = scan_forbidden(source, forbidden, forbidden_attrs)
    if bad is not None:
        fail(f"forbidden import or name for this task: {bad!r}")
    if (max_literal_items is not None
            or max_total_literal_items is not None
            or max_string_literal_bytes is not None):
        tree = ast.parse(source)
        total_items = 0
        for node in ast.walk(tree):
            if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
                n_items = len(node.elts)
                total_items += n_items
                if max_literal_items is not None and n_items > max_literal_items:
                    fail(f"literal container has too many items "
                         f"({n_items} > {max_literal_items})")
            elif isinstance(node, ast.Dict):
                n_items = len(node.keys)
                total_items += n_items
                if max_literal_items is not None and n_items > max_literal_items:
                    fail(f"literal container has too many items "
                         f"({n_items} > {max_literal_items})")
            elif (max_string_literal_bytes is not None
                    and isinstance(node, ast.Constant)
                    and isinstance(node.value, (str, bytes))):
                n_bytes = len(node.value if isinstance(node.value, bytes)
                              else node.value.encode("utf-8"))
                if n_bytes > max_string_literal_bytes:
                    fail(f"string/bytes literal is too large "
                         f"({n_bytes} > {max_string_literal_bytes} bytes)")
        if max_total_literal_items is not None and total_items > max_total_literal_items:
            fail(f"too many literal container items "
                 f"({total_items} > {max_total_literal_items})")
    module = types.ModuleType("program")
    module.__file__ = "<program>"
    if safe_builtins:
        # A fresh copy per load: candidate mutations of the builtins dict
        # cannot persist into other loads within the same evaluation.
        module.__dict__["__builtins__"] = dict(SAFE_BUILTINS)
    sys.modules["program"] = module
    try:
        code = compile(source, "<program>", "exec")
        if import_budget is not None:
            opcount.start(budget=import_budget)
        exec(code, module.__dict__)
        if import_budget is not None:
            used = opcount.stop()
            if used > import_budget:
                fail(f"program import exceeded instruction budget "
                     f"({used} > {import_budget})")
    except opcount.BudgetExceeded:
        opcount.stop()
        fail(f"program import exceeded instruction budget of {import_budget}")
    except SystemExit:
        if import_budget is not None:
            opcount.stop()
        fail("program called sys.exit() at import time")
    except BaseException:
        if import_budget is not None:
            opcount.stop()
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
