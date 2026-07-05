"""Strong reference for `normalize`: canonicalize a messy duration string
to total whole seconds (as a decimal integer string).

Handles the full surface tail: compact unit strings, spaced abbreviations,
spelled-out numbers, word and unicode fractions, ISO-8601, H:MM:SS clock
notation, prime/double-prime marks, thousands separators, locale decimal
commas, idiomatic shorthands, and noisy prefixes/suffixes.

The one genuinely ambiguous surface form (a bare `X:YY` with no third
field) is resolved to its majority interpretation (H:MM); the residual is
the irreducible error floor.
"""

import re

UNIT_SECONDS = {
    "week": 604800, "weeks": 604800, "wk": 604800, "wks": 604800, "w": 604800,
    "day": 86400, "days": 86400, "d": 86400,
    "hour": 3600, "hours": 3600, "hr": 3600, "hrs": 3600, "h": 3600,
    "minute": 60, "minutes": 60, "min": 60, "mins": 60, "m": 60,
    "second": 1, "seconds": 1, "sec": 1, "secs": 1, "s": 1,
}

_ONES = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
         "seventy": 70, "eighty": 80, "ninety": 90}


def _words_to_digits(text):
    """Replace spelled-out number words (0-99) with digit strings."""
    toks = text.split(" ")
    out = []
    i = 0
    while i < len(toks):
        raw = toks[i]
        w = raw.strip(",")
        if w in _TENS:
            val = _TENS[w]
            # optional following ones word: "twenty-one" already handled by
            # hyphen; "thirty" alone too.
            if "-" in raw:
                base, _, rest = raw.partition("-")
                if base in _TENS and rest in _ONES:
                    val = _TENS[base] + _ONES[rest]
            out.append(str(val))
            i += 1
            continue
        if "-" in w:
            base, _, rest = w.partition("-")
            if base in _TENS and rest in _ONES:
                out.append(str(_TENS[base] + _ONES[rest]))
                i += 1
                continue
        if w in _ONES:
            out.append(str(_ONES[w]))
            i += 1
            continue
        out.append(raw)
        i += 1
    return " ".join(out)


def _strip_noise(t):
    for pre in ("approximately ", "approx. ", "approx ", "about ",
                "roughly ", "duration: ", "duration ", "est. ", "~"):
        if t.startswith(pre):
            t = t[len(pre):]
    if t.startswith("lasted "):
        t = t[len("lasted "):]
    for suf in (" long", " total", " in all", " exactly"):
        if t.endswith(suf):
            t = t[: -len(suf)]
    return t.strip()


def _fraction_idioms(t):
    table = {
        "half an hour": 1800,
        "half a day": 43200,
        "quarter of an hour": 900,
        "a quarter of an hour": 900,
        "three quarters of an hour": 2700,
        "an hour and a half": 5400,
        "one and a half hours": 5400,
    }
    if t in table:
        return table[t]
    # "<n> and a half hours" (digit or spelled-out number).
    m = re.fullmatch(r"(\w+) and a half hours", t)
    if m:
        w = m.group(1)
        n = int(w) if w.isdigit() else _ONES.get(w)
        if n is not None:
            return n * 3600 + 1800
    return None


_UNICODE_FRAC = {"½": 0.5, "¼": 0.25, "¾": 0.75}


def predict(raw):
    if not isinstance(raw, str):
        return "0"
    t = raw.strip().lower()

    # ISO-8601 duration (PT#H#M#S) — check before lowering matters little.
    m = re.fullmatch(r"pt(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?", t)
    if m and any(m.groups()):
        h = int(m.group(1) or 0)
        mi = int(m.group(2) or 0)
        s = int(m.group(3) or 0)
        return str(h * 3600 + mi * 60 + s)

    t = _strip_noise(t)

    # H:MM:SS clock notation (three fields -> unambiguous).
    m = re.fullmatch(r"(\d+):(\d{1,2}):(\d{1,2})", t)
    if m:
        h, mi, s = (int(x) for x in m.groups())
        return str(h * 3600 + mi * 60 + s)

    # Bare X:YY with no third field: genuinely ambiguous. Default to the
    # majority interpretation, H:MM.
    m = re.fullmatch(r"(\d+):(\d{2})", t)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        return str(a * 3600 + b * 60)

    # Prime / double-prime marks: ' = minutes, '' = seconds.
    m = re.fullmatch(r"(\d+)'\s*(\d+)''", t)
    if m:
        return str(int(m.group(1)) * 60 + int(m.group(2)))
    m = re.fullmatch(r"(\d+)''", t)
    if m:
        return str(int(m.group(1)))
    m = re.fullmatch(r"(\d+)'", t)
    if m:
        return str(int(m.group(1)) * 60)

    # Fraction idioms.
    fi = _fraction_idioms(t)
    if fi is not None:
        return str(fi)

    # "a couple of <unit>" -> 2 units.
    m = re.fullmatch(r"a couple of (\w+)", t)
    if m and m.group(1) in UNIT_SECONDS:
        return str(2 * UNIT_SECONDS[m.group(1)])

    # Unicode fractions: "<w>½ <unit>" / "½ <unit>".
    m = re.fullmatch(r"(\d*)\s*([½¼¾])\s*([a-z]+)", t)
    if m and m.group(3) in UNIT_SECONDS:
        whole = int(m.group(1)) if m.group(1) else 0
        val = whole + _UNICODE_FRAC[m.group(2)]
        return str(round(val * UNIT_SECONDS[m.group(3)]))

    # Spelled-out numbers -> digits.
    t = _words_to_digits(t)

    # Locale decimal comma: "1,5 h" -> "1.5 h" (comma + exactly 1-2 digits).
    t = re.sub(r"(?<=\d),(\d{1,2})(?=\D|$)", r".\1", t)
    # Thousands separators: comma + exactly 3 digits.
    t = re.sub(r"(?<=\d),(?=\d{3}(\D|$))", "", t)

    # Generic: sum every <number><unit> pair.
    total = 0.0
    found = False
    for num, unit in re.findall(r"(\d+(?:\.\d+)?)\s*([a-z]+)", t):
        if unit in UNIT_SECONDS:
            total += float(num) * UNIT_SECONDS[unit]
            found = True
    if found:
        return str(int(round(total)))

    # Last resort: a bare number is treated as seconds.
    m = re.search(r"\d+", t)
    return m.group(0) if m else "0"
