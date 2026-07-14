"""Deterministic structured-text corpus generator used by compress_heldout.

This is operator tooling. Benchmark agents may inspect only the visible
``train_*.txt`` files, not this generator or the sealed corpus seeds.
"""

import random


def gen_logs(rng, target):
    methods = ["GET"] * 4 + ["POST", "PUT", "DELETE"]
    paths = [
        "/", "/index.html", "/api/v2/users", "/api/v2/orders",
        "/static/app.js", "/static/style.css", "/login", "/logout",
        "/health", "/api/v2/search", "/img/logo.png",
        "/api/v2/cart/items", "/docs/getting-started",
    ]
    agents = [
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Gecko/20100101 Firefox/126.0",
        "curl/8.4.0", "python-requests/2.31.0",
    ]
    statuses = [200] * 5 + [301, 304, 404, 500]
    lines, size, timestamp = [], 0, 1_700_000_000
    while size < target:
        timestamp += rng.randrange(0, 3)
        ip = f"10.{rng.randrange(4)}.{rng.randrange(256)}.{rng.randrange(256)}"
        line = (
            f'{ip} - - [{timestamp}] "{rng.choice(methods)} {rng.choice(paths)} '
            f'HTTP/1.1" {rng.choice(statuses)} {rng.randrange(200, 40000)} '
            f'"{rng.choice(agents)}"\n'
        )
        lines.append(line)
        size += len(line)
    return "".join(lines).encode()


def gen_jsonl(rng, target):
    plans = ["free", "pro", "team", "enterprise"]
    regions = ["us-east-1", "us-west-2", "eu-central-1", "ap-southeast-2"]
    lines, size, index = [], 0, 0
    while size < target:
        line = (
            f'{{"record_id": {index}, "customer": "cust_{rng.randrange(9999):04d}", '
            f'"plan": "{rng.choice(plans)}", "region": "{rng.choice(regions)}", '
            f'"active": {rng.choice(["true", "false"])}, '
            f'"usage_gb": {rng.randrange(0, 5000) / 10}, '
            f'"renewal_ts": {1_700_000_000 + rng.randrange(31_536_000)}}}\n'
        )
        lines.append(line)
        size += len(line)
        index += 1
    return "".join(lines).encode()


def gen_prose(rng, target):
    subjects = [
        "the engineer", "the committee", "a small team", "the pilot program",
        "our analysis", "the survey", "the northern district", "the archive",
        "the second experiment", "the review board",
    ]
    verbs = [
        "concluded that", "reported that", "suggested that", "confirmed that",
        "noted that", "argued that", "demonstrated that", "assumed that",
    ]
    objects = [
        "the results were consistent with earlier findings",
        "further measurements would be necessary",
        "the equipment performed within expected tolerances",
        "the proposal should be revised before approval",
        "costs had grown faster than projected",
        "the schedule remained achievable despite delays",
        "the data supported a more cautious interpretation",
        "additional staff would be required next quarter",
    ]
    connectives = ["Moreover,", "However,", "In addition,", "By contrast,", "Meanwhile,"]
    output, size = [], 0
    while size < target:
        sentence = (
            f"{rng.choice(subjects).capitalize()} {rng.choice(verbs)} "
            f"{rng.choice(objects)}."
        )
        if rng.random() < 0.4:
            sentence = f"{rng.choice(connectives)} {sentence[0].lower()}{sentence[1:]}"
        sentence += "\n" if rng.random() < 0.15 else " "
        output.append(sentence)
        size += len(sentence)
    return "".join(output).encode()


def gen_csv(rng, target):
    lines = ["sensor_id,timestamp,temperature_c,humidity_pct,pressure_hpa\n"]
    size, timestamp = len(lines[0]), 1_700_000_000
    temperatures = {sensor: 20.0 + rng.random() * 5 for sensor in range(8)}
    humidities = {sensor: 40.0 + rng.random() * 20 for sensor in range(8)}
    while size < target:
        timestamp += 15
        for sensor in range(8):
            temperatures[sensor] += (rng.random() - 0.5) * 0.2
            humidities[sensor] += (rng.random() - 0.5) * 0.6
            line = (
                f"sensor-{sensor:03d},{timestamp},{temperatures[sensor]:.2f},"
                f"{humidities[sensor]:.1f},{1013 + rng.randrange(-40, 40) / 10:.1f}\n"
            )
            lines.append(line)
            size += len(line)
    return "".join(lines).encode()


def gen_corpus(seed, target):
    rng = random.Random(seed)
    return {
        "logs": gen_logs(rng, target),
        "jsonl": gen_jsonl(rng, target),
        "prose": gen_prose(rng, target),
        "csv": gen_csv(rng, target),
    }
