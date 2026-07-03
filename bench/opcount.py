"""Deterministic operation counting via sys.monitoring (Python 3.12+).

Counts bytecode instructions executed while counting is enabled. For a
deterministic program on a fixed Python version, the count is exactly
reproducible and completely unaffected by other load on the machine —
this is the benchmark's substitute for wall-clock timing.

Programs being optimized under an instruction budget may do:

    from bench.opcount import remaining, used

to check how much budget is left and stop gracefully.
"""

import sys

_mon = sys.monitoring
_TOOL = 2  # sys.monitoring.PROFILER_ID slot


class BudgetExceeded(Exception):
    """Raised inside the traced program when the instruction budget runs out."""


_count = 0
_budget = None
_active = False


def _cb(code, offset):
    global _count, _budget
    _count += 1
    if _budget is not None and _count > _budget:
        # Disarm before raising, or the exception handler's own bytecode
        # would re-trigger the budget check and re-raise inside it.
        exceeded = _budget
        _budget = None
        raise BudgetExceeded(f"instruction budget of {exceeded} exceeded")


def start(budget=None):
    """Reset the counter and begin counting executed bytecode instructions."""
    global _count, _budget, _active
    _count = 0
    _budget = budget
    _mon.use_tool_id(_TOOL, "textopt")
    _mon.register_callback(_TOOL, _mon.events.INSTRUCTION, _cb)
    _mon.set_events(_TOOL, _mon.events.INSTRUCTION)
    _active = True


def stop():
    """Stop counting and return the number of instructions executed."""
    global _active
    if _active:
        _mon.set_events(_TOOL, 0)
        _mon.register_callback(_TOOL, _mon.events.INSTRUCTION, None)
        _mon.free_tool_id(_TOOL)
        _active = False
    return _count


def used():
    """Instructions executed since start()."""
    return _count


def remaining():
    """Instructions left in the budget, or None if no budget is set."""
    if _budget is None:
        return None
    return _budget - _count
