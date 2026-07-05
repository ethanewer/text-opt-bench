"""Baseline: grab the first integer and return it unchanged as seconds."""

import re


def predict(raw):
    m = re.search(r"\d+", raw)
    return m.group(0) if m else "0"
