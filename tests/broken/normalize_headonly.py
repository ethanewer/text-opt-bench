"""Plausible ITERATION-1 attempt at `normalize`.

Handles only the obvious, common formats visible everywhere in train:
  - compact "1h30m" / "2h" / "45m" (optionally with seconds),
  - "<n> hours/minutes/seconds/days" with a full unit word,
  - "<x.y> hours" decimal.
It ignores the long tail (spelled-out numbers, fractions, ISO-8601, clock
notation, primes, thousands separators, locale commas, idioms, noise
wrappers) and the ambiguous bare-colon form — modelling what a solver
writes in one pass over the head of the distribution.
"""

import re

_UNIT = {"hour": 3600, "hours": 3600, "minute": 60, "minutes": 60,
         "second": 1, "seconds": 1, "day": 86400, "days": 86400}


def predict(raw):
    t = raw.strip().lower()

    # compact "1h30m15s"
    m = re.fullmatch(r"(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?", t)
    if m and any(m.groups()):
        h = int(m.group(1) or 0)
        mi = int(m.group(2) or 0)
        s = int(m.group(3) or 0)
        return str(h * 3600 + mi * 60 + s)

    # decimal + single full unit word: "1.5 hours"
    m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*(hours?|minutes?|seconds?|days?)", t)
    if m:
        return str(int(round(float(m.group(1)) * _UNIT[m.group(2)])))

    # last resort: first integer as seconds
    m = re.search(r"\d+", t)
    return m.group(0) if m else "0"
