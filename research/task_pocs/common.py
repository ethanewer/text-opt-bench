from __future__ import annotations

import json
import math
import time
from pathlib import Path


def choose_device(torch, requested: str):
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def synchronize(torch, device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def timed(torch, device, fn):
    synchronize(torch, device)
    start = time.perf_counter()
    value = fn()
    synchronize(torch, device)
    return value, time.perf_counter() - start


def spearman(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return float("nan")

    def ranks(values):
        order = sorted(range(len(values)), key=values.__getitem__)
        result = [0.0] * len(values)
        start = 0
        while start < len(order):
            end = start + 1
            while end < len(order) and values[order[end]] == values[order[start]]:
                end += 1
            rank = (start + end - 1) / 2.0
            for j in range(start, end):
                result[order[j]] = rank
            start = end
        return result

    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    dy = math.sqrt(sum((b - my) ** 2 for b in ry))
    return num / (dx * dy) if dx and dy else float("nan")


def dump(payload) -> None:
    print(json.dumps(payload, sort_keys=True, allow_nan=False))


def load_rows(path: Path) -> list[str]:
    data = json.loads(path.read_text())
    return [entry["row"]["text"] for entry in data["rows"]]
