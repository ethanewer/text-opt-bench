"""Baseline solver: return the last number mentioned in the question."""

import re


def solve(question):
    numbers = re.findall(r"\d+", question)
    return int(numbers[-1]) if numbers else 0
