"""Evaluator for compress. Score = total compressed bytes (lower better)."""

import random
import sys

sys.path.insert(0, __file__.rsplit("/bench/", 1)[0])
from bench import eval_lib

SEED = 0x5EED
TARGET_BYTES = 150_000  # per document
# Validation corpus: same four genres, different seed, smaller. Round-trip
# must hold and the compression ratio must be in the same ballpark as on
# the scoring corpus — catches programs that embed or regenerate the known
# scoring corpus instead of compressing.
VALIDATION_SEED = 0xD1CE
VALIDATION_RATIO_FACTOR = 0.4
VALIDATION_TARGET_BYTES = 50_000  # per validation document

FORBIDDEN = frozenset(
    {
        "zlib", "gzip", "bz2", "lzma", "zstd", "compression", "zipfile",
        "tarfile", "codecs", "encodings", "os", "io", "open", "mmap",
        "ctypes", "socket", "subprocess", "multiprocessing", "threading",
        "importlib", "__import__",
    }
)


def gen_logs(rng, target=TARGET_BYTES):
    methods = ["GET", "GET", "GET", "GET", "POST", "PUT", "DELETE"]
    paths = [
        "/", "/index.html", "/api/v2/users", "/api/v2/orders", "/static/app.js",
        "/static/style.css", "/login", "/logout", "/health", "/api/v2/search",
        "/img/logo.png", "/api/v2/cart/items", "/docs/getting-started",
    ]
    agents = [
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Gecko/20100101 Firefox/126.0",
        "curl/8.4.0",
        "python-requests/2.31.0",
    ]
    statuses = [200, 200, 200, 200, 200, 301, 304, 404, 500]
    lines = []
    size = 0
    t = 1_700_000_000
    while size < target:
        t += rng.randrange(0, 3)
        ip = f"10.{rng.randrange(4)}.{rng.randrange(256)}.{rng.randrange(256)}"
        line = (
            f'{ip} - - [{t}] "{rng.choice(methods)} {rng.choice(paths)} HTTP/1.1" '
            f'{rng.choice(statuses)} {rng.randrange(200, 40000)} "{rng.choice(agents)}"\n'
        )
        lines.append(line)
        size += len(line)
    return "".join(lines).encode()


def gen_jsonl(rng, target=TARGET_BYTES):
    plans = ["free", "pro", "team", "enterprise"]
    regions = ["us-east-1", "us-west-2", "eu-central-1", "ap-southeast-2"]
    lines = []
    size = 0
    i = 0
    while size < target:
        line = (
            f'{{"record_id": {i}, "customer": "cust_{rng.randrange(9999):04d}", '
            f'"plan": "{rng.choice(plans)}", "region": "{rng.choice(regions)}", '
            f'"active": {rng.choice(["true", "false"])}, '
            f'"usage_gb": {rng.randrange(0, 5000) / 10}, '
            f'"renewal_ts": {1_700_000_000 + rng.randrange(31_536_000)}}}\n'
        )
        lines.append(line)
        size += len(line)
        i += 1
    return "".join(lines).encode()


def gen_prose(rng, target=TARGET_BYTES):
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
    out = []
    size = 0
    while size < target:
        s = f"{rng.choice(subjects).capitalize()} {rng.choice(verbs)} {rng.choice(objects)}."
        if rng.random() < 0.4:
            s = f"{rng.choice(connectives)} {s[0].lower()}{s[1:]}"
        s += "\n" if rng.random() < 0.15 else " "
        out.append(s)
        size += len(s)
    return "".join(out).encode()


def gen_csv(rng, target=TARGET_BYTES):
    lines = ["sensor_id,timestamp,temperature_c,humidity_pct,pressure_hpa\n"]
    size = len(lines[0])
    t = 1_700_000_000
    temps = {s: 20.0 + rng.random() * 5 for s in range(8)}
    hums = {s: 40.0 + rng.random() * 20 for s in range(8)}
    while size < target:
        t += 15
        for s in range(8):
            temps[s] += (rng.random() - 0.5) * 0.2
            hums[s] += (rng.random() - 0.5) * 0.6
            line = (
                f"sensor-{s:03d},{t},{temps[s]:.2f},{hums[s]:.1f},"
                f"{1013 + rng.randrange(-40, 40) / 10:.1f}\n"
            )
            lines.append(line)
            size += len(line)
    return "".join(lines).encode()


def gen_corpus(seed=SEED, target=TARGET_BYTES):
    rng = random.Random(seed)
    return {
        "logs": gen_logs(rng, target),
        "jsonl": gen_jsonl(rng, target),
        "prose": gen_prose(rng, target),
        "csv": gen_csv(rng, target),
    }


def run_corpus(program_path, corpus, label):
    # Two-phase with a FRESH module between phases: compress everything with one
    # module instance, drop it, then decompress with a newly loaded instance.
    # This closes the module-global side channel — compress() cannot stash the
    # payload in a shared global (e.g. `_STORE.append(data)`; return an index)
    # for decompress() to read back, so the blob must ACTUALLY carry the data.
    # A legitimate codec is unaffected: static/source-level codebooks are
    # re-created identically on reload; only dynamic state PASSED from compress
    # to decompress outside the blob is broken (which is the cheat).
    mod = eval_lib.load_program(
        program_path, FORBIDDEN, required=("compress", "decompress")
    )
    blobs = {}
    for name, data in corpus.items():
        blob = eval_lib.run_program(mod.compress, data)
        if not isinstance(blob, (bytes, bytearray)):
            eval_lib.fail(
                f"{label}: compress({name}) returned {type(blob).__name__}, not bytes"
            )
        blobs[name] = bytes(blob)
    del mod

    mod = eval_lib.load_program(
        program_path, FORBIDDEN, required=("compress", "decompress")
    )
    total = 0
    original = 0
    per_doc = {}
    for name, data in corpus.items():
        blob = blobs[name]
        restored = eval_lib.run_program(mod.decompress, blob)
        if not isinstance(restored, (bytes, bytearray)) or bytes(restored) != data:
            eval_lib.fail(
                f"{label}: round-trip failed on {name!r}: original {len(data)} bytes, "
                f"restored {len(restored) if isinstance(restored, (bytes, bytearray)) else type(restored).__name__}"
            )
        per_doc[name] = {"original": len(data), "compressed": len(blob)}
        total += len(blob)
        original += len(data)
    return total, original, per_doc


def main():
    program_path = sys.argv[1]
    # Validate the API early (fresh load); run_corpus reloads per phase.
    eval_lib.load_program(
        program_path, FORBIDDEN, required=("compress", "decompress")
    )

    total, original, per_doc = run_corpus(program_path, gen_corpus(), "scoring corpus")

    v_total, v_original, _ = run_corpus(
        program_path, gen_corpus(VALIDATION_SEED, VALIDATION_TARGET_BYTES), "validation corpus (unseen data)"
    )
    score_cr = total / original
    val_cr = v_total / v_original
    if score_cr < VALIDATION_RATIO_FACTOR * val_cr:
        eval_lib.fail(
            f"compression ratio on the scoring corpus ({score_cr:.4f}) is far "
            f"better than on unseen data of the same distribution ({val_cr:.4f}): "
            "the program appears specialized to the known scoring corpus, "
            "which is cheating"
        )

    eval_lib.succeed(
        float(total),
        metrics={
            "total_compressed_bytes": total,
            "total_original_bytes": sum(d["original"] for d in per_doc.values()),
            "per_doc": per_doc,
        },
    )


if __name__ == "__main__":
    main()
