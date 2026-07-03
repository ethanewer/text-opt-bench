"""Reference solution: priority-weighted shortest-job-first admission."""


def order(requests, config):
    return [
        r["id"]
        for r in sorted(
            requests,
            key=lambda r: (
                (r["prompt"] + r["output"]) / (1.0 + 0.8 * r["priority"]),
                r["id"],
            ),
        )
    ]
