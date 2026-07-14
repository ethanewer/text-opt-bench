"""Weak baseline: return the final numeral mentioned in the question."""

import re


def solve(question):
    numbers = re.findall(r"\d+(?:\.\d+)?", question)
    return float(numbers[-1]) if numbers else 0
