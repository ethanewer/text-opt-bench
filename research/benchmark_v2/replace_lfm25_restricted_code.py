"""Replace gated code-corpus rows with permissively licensed source releases."""

from collections import Counter
import hashlib
import json

from build_lfm25_hard_eval_data import (
    Builder, OLD_DATA, OUTPUT, SITE_PACKAGES, SOURCES)


DROP_FAMILIES = {
    "code_completion", "code_reasoning", "ml_library_code",
    "network_library_code", "nonpython_code",
}


def main():
    payload = json.loads(OUTPUT.read_text())
    retained = [row for row in payload["records"]
                if row["domain"] not in DROP_FAMILIES]
    builder = Builder()
    replacements = []
    for split in ("validation", "id_test"):
        replacements += builder.collect_code_files(
            split, "ml_library_code", "transformers_code",
            SITE_PACKAGES / "transformers", "*.py")
        replacements += builder.collect_code_files(
            split, "network_library_code", "aiohttp_code",
            SITE_PACKAGES / "aiohttp", "*.py")
    replacements += builder.collect_code_files(
        "ood_test", "nonpython_code", "pybind11_code",
        SITE_PACKAGES / "torch/include/pybind11", "*.h")
    if len(replacements) != 80:
        raise RuntimeError(f"expected 80 replacement rows, found {len(replacements)}")
    calibration = [row for row in json.loads(OLD_DATA.read_text())["records"]
                   if row["split"] == "calibration"]
    retained = [row for row in retained if row["split"] != "calibration"]
    scored = retained + replacements
    scored.sort(key=lambda row: (row["split"], row["domain"], row["prompt_id"]))
    records = calibration + scored
    expected = {"calibration": 128, "validation": 128,
                "id_test": 128, "ood_test": 128}
    counts = Counter(row["split"] for row in records)
    if dict(counts) != expected:
        raise RuntimeError(counts)
    provenance = {
        key: value for key, value in payload["provenance"].items()
        if not any(row["prompt_id"] == key and row["domain"] in DROP_FAMILIES
                   for row in payload["records"])
    }
    provenance.update(builder.provenance)
    payload.update({
        "records": records,
        "provenance": provenance,
        "sources": SOURCES,
        "counts": expected,
        "family_counts": {
            split: dict(sorted(Counter(
                row["domain"] for row in records
                if row["split"] == split).items()))
            for split in expected
        },
    })
    OUTPUT.write_text(json.dumps(
        payload, separators=(",", ":"), ensure_ascii=False) + "\n")
    print(json.dumps({
        "output": str(OUTPUT),
        "sha256": hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
        "counts": expected,
        "family_counts": payload["family_counts"],
        "replacement_sources": {
            key: SOURCES[key] for key in
            ("transformers_code", "aiohttp_code", "pybind11_code")},
    }, indent=2))


if __name__ == "__main__":
    main()
