"""Shared helpers for task evaluators.

Evaluators run as standalone scripts in a child process and communicate a
single JSON line on stdout:

    {"ok": bool, "score": float|null, "metrics": {...}, "error": str|null}

Lower score is always better. The program's own stdout is redirected to
stderr so it can't corrupt the protocol.
"""

import ast
import json
import os
import resource
import sys
import traceback
import types
from pathlib import Path

from bench import opcount

# Capture the real stdout for the JSON result, then route everything the
# program (or evaluator) prints to stderr.
_REAL_STDOUT = sys.stdout
sys.stdout = sys.stderr

# Per-run nonce from the harness (bench.runner). The result line is
# prefixed with it and the harness accepts only a nonce-prefixed line;
# emit() also hard-exits (os._exit) so nothing can append afterward. This
# defeats CASUAL forgery (stray prints / atexit tricks) but is not
# unforgeable — a candidate that escapes to os can read this nonce from
# the env (the cooperative-model boundary; see the BASELINE_* note below).
_NONCE = os.environ.get("TEXTOPT_RESULT_NONCE", "")


def emit(ok, score=None, metrics=None, error=None):
    # The child's own CPU time (RUSAGE_SELF) is per-process accurate,
    # unlike a parent-side RUSAGE_CHILDREN delta which is process-global;
    # the harness prefers this for the timing trace.
    ru = resource.getrusage(resource.RUSAGE_SELF)
    payload = {"ok": ok, "score": score, "metrics": metrics or {},
               "error": error,
               "eval_self_cpu_seconds": round(ru.ru_utime + ru.ru_stime, 4)}
    line = (_NONCE + " " if _NONCE else "") + json.dumps(payload)
    try:
        sys.stderr.flush()
    except Exception:
        pass
    _REAL_STDOUT.write(line + "\n")
    _REAL_STDOUT.flush()
    # Hard exit: skip atexit handlers / object finalizers so a candidate
    # cannot register code that writes a second, forged result line after
    # the evaluator's real one.
    os._exit(0)


def fail(error, metrics=None):
    emit(False, None, metrics, error)


def succeed(score, metrics=None):
    emit(True, score, metrics, None)


# Escape primitives blocked for EVERY task, on top of each task's own bans.
# IMPORTANT — this is a COOPERATIVE GUARD, NOT A SANDBOX. It rejects the
# honest mistakes and the obvious/lazy cheats (import zlib, __builtins__
# ["__import__"], from bench import opcount, gc.get_referrers, x.__globals__,
# print.__self__, posixpath.os). It does NOT and CANNOT stop a determined
# adversary: an AST source scan can't see attribute access hidden in a
# string, e.g. "{0.__globals__}".format(obj) reaches module globals with no
# import and no forbidden node at all. In-process execution of untrusted
# Python is not securely sandboxable this way. The benchmark's actual
# integrity rests on the cooperative threat model + full auditability of
# every recorded submission + unseen-data validation that catches
# hardcoding — not on this scan being exhaustive.
BASELINE_FORBIDDEN_NAMES = frozenset({
    "__builtins__", "builtins", "__import__", "importlib", "imp",
    "eval", "exec", "compile", "getattr", "setattr", "delattr",
    "globals", "vars", "locals", "input", "open", "bench",
    "os", "sys", "resource", "gc",
})
# Introspection gadgets that reach module globals / builtins from an
# ordinary object (e.g. ().__class__.__base__.__subclasses__(), a
# function's __globals__, or a C builtin's __self__ == the builtins module).
BASELINE_FORBIDDEN_ATTRS = frozenset({
    "__globals__", "__builtins__", "__subclasses__", "__bases__",
    "__base__", "__class__", "__mro__", "__code__", "__closure__",
    "__dict__", "__getattribute__", "__loader__", "__spec__", "__import__",
    "__self__", "f_globals", "f_locals", "f_back", "gi_frame", "cr_frame",
    "tb_frame", "tb_next",
})
# Forbidden modules reached as an attribute of an ALLOWED module launder the
# name ban (e.g. posixpath.os is the real os module). Reject these as
# attribute names too — narrow set to avoid colliding with legit method
# names like re.compile / bytes.eval-free code.
_ATTR_LAUNDER_NAMES = frozenset({
    "os", "sys", "bench", "builtins", "__builtins__",
})


# ---- runtime enforcement of the import + file-read channels ------------
# The AST scan is static and defeatable by obfuscation (split strings,
# inspect.getattr_static, str.format attribute access — no finite blocklist
# closes them). We add RUNTIME enforcement on the two channels every
# demonstrated escape uses to deliver its payload:
#
#   imports — builtins.__import__ is replaced by a guard, so EVERY import
#     (an `import` statement, or an escaped __import__ reached through any
#     obfuscation, cached or fresh) is checked against the forbidden set.
#     A PEP 578 audit hook only sees FRESH imports (cached re-imports raise
#     no event), so it cannot stop re-importing already-loaded os/tracemalloc
#     /bench; replacing __import__ catches those too.
#   file reads — an audit hook blocks opening any benchmark-repo file
#     (held-out .bin data), which imports never touch.
#
# Both are installed at MODULE IMPORT (outside any tracemalloc window, so
# memory scores are unperturbed) and enforce only while candidate code runs.
#
# HONEST LIMIT: this does NOT stop pure in-process frame-walking to
# already-loaded evaluator objects (a generator's gi_frame reached via
# operator.attrgetter needs no import and no forbidden literal). That
# residual is unclosable in-process; see the README threat model.
import builtins as _builtins

# Modules a candidate must never import — escape enablers + interpreter/OS.
# Checked in addition to each task's own FORBIDDEN.
_ESCAPE_IMPORT_BLOCK = frozenset({
    "os", "sys", "builtins", "importlib", "imp", "inspect", "gc", "ctypes",
    "resource", "subprocess", "socket", "mmap", "pdb", "bdb", "cProfile",
    "profile", "trace", "posixpath", "ntpath", "genericpath", "sysconfig",
    "bench", "site", "runpy", "code", "codeop",
})

_candidate_active = False           # True only while candidate code runs
_audit_events = []                  # forbidden ops the candidate attempted
_audit_task_forbidden = frozenset()  # the current task's FORBIDDEN (by ref)
_REPO_ROOT_STR = str(Path(__file__).resolve().parents[1])
_orig_import = _builtins.__import__


def _guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if _candidate_active:
        root = str(name).split(".")[0]
        if root in _audit_task_forbidden or root in _ESCAPE_IMPORT_BLOCK:
            _audit_events.append("import:" + str(name))
            raise ImportError(
                f"import of {name!r} is not allowed during evaluation")
    return _orig_import(name, globals, locals, fromlist, level)


def _audit_hook(event, args):
    if not _candidate_active or not args:
        return
    if event in ("open", "os.open"):
        target = args[0]
        if isinstance(target, (str, bytes)):
            p = os.fsdecode(target)
            # Imports read stdlib files outside the repo; only held-out /
            # benchmark files live under the repo root — never legit reads.
            if os.path.abspath(p).startswith(_REPO_ROOT_STR):
                _audit_events.append("open:" + p)
                raise PermissionError(
                    "reading benchmark files is not allowed during evaluation")


# Installed once, at import, before any evaluator opens a tracemalloc window.
_builtins.__import__ = _guarded_import
sys.addaudithook(_audit_hook)


def _set_candidate_active(active):
    # A bare flag toggle (no object allocation), so enabling enforcement
    # around a candidate call adds nothing to a tracemalloc measurement.
    global _candidate_active
    _candidate_active = active


def scan_forbidden(source, forbidden, forbidden_attrs=frozenset()):
    """Return the first forbidden module/name/attribute referenced, else None.

    Always enforces the benchmark-wide escape blocklist (builtins/import/
    eval/introspection/bench/file-IO) in addition to the task's own bans.
    A cooperative guard, not a sandbox: it stops the obvious escapes but
    cannot catch attribute access hidden in strings (str.format /
    operator.attrgetter). See the BASELINE_* comment above.
    """
    forbidden = forbidden | BASELINE_FORBIDDEN_NAMES
    forbidden_attrs = forbidden_attrs | BASELINE_FORBIDDEN_ATTRS
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
        elif isinstance(node, ast.Attribute) and (
                node.attr in forbidden_attrs
                or node.attr in _ATTR_LAUNDER_NAMES):
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
    expose_budget=False,
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
        leaving compact algorithmic solutions room;
      - expose_budget: inject read-only `remaining()`/`used()` into the
        program namespace for instruction-budget tasks, so programs can
        pace themselves WITHOUT importing `bench` (which is forbidden — an
        imported bench.opcount would expose start/stop/internals). The
        `__globals__` attribute ban prevents escaping these back to the
        opcount module.
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
    if expose_budget:
        # Read-only budget accessors, so budget-aware programs need not
        # import bench (forbidden). Only these two, not start/stop.
        module.__dict__["remaining"] = opcount.remaining
        module.__dict__["used"] = opcount.used
    sys.modules["program"] = module
    # Runtime enforcement of the import/file-read channels (see the guard
    # machinery above): point it at this task's forbidden set (by reference,
    # no allocation inside a tracemalloc window), then run candidate code
    # under the guard.
    global _audit_task_forbidden
    _audit_task_forbidden = forbidden
    try:
        code = compile(source, "<program>", "exec")
        if import_budget is not None:
            opcount.start(budget=import_budget)
        _set_candidate_active(True)
        try:
            exec(code, module.__dict__)
        finally:
            _set_candidate_active(False)
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
    """Call into the program, converting exceptions into a failure result.

    The call runs under the audit-hook guard, so a candidate that reaches
    a forbidden import or file read at CALL time (not just import time) is
    blocked and surfaced as a normal failure.
    """
    _set_candidate_active(True)
    try:
        return fn(*args)
    except SystemExit:
        fail("program called sys.exit() during evaluation")
    except BaseException:
        fail("program raised during evaluation:\n" + traceback.format_exc(limit=8))
    finally:
        _set_candidate_active(False)
