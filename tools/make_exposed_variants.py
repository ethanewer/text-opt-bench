"""Create the val-EXPOSED task variants for the overfitting coverage experiment.

For each generalization task, an `<task>_exposed` variant is a copy of the task
whose VISIBLE training data is replaced by the DECODED validation split — i.e.
the optimizer can now see (and fit) the exact instances it is graded on. Scoring
(hidden val) and the hidden test split are unchanged, so the exposed run is
directly comparable to the normal (hidden-val) run, and the held-out test reveals
overfitting. Reproducible: rerun this to regenerate the variants; they are NOT
part of the core benchmark (kept out of git; the results are committed instead).

Usage:  python3.12 tools/make_exposed_variants.py        # create all 3
        python3.12 tools/make_exposed_variants.py --rm   # remove them
"""
import base64, json, shutil, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench import heldout

ROOT = Path(__file__).resolve().parents[1]
GEN_TASKS = ["easy_word_problems", "tag_seq", "compress_heldout"]


def make_one(task):
    src = ROOT / "bench" / "tasks" / task
    dst = ROOT / "bench" / "tasks" / f"{task}_exposed"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    if task == "compress_heldout":
        # corpus task: decode the val corpus -> visible train_*.txt (bytes)
        val = {doc: base64.b64decode(b64)
               for doc, b64 in heldout.read(dst / "data" / "heldout_val.bin").items()}
        for old in (dst / "data").glob("train_*.txt"):
            old.unlink()
        for doc, raw in val.items():
            (dst / "data" / f"train_{doc}.txt").write_bytes(raw)
        n = len(val)
    else:
        # labeled-row task: decode the val rows -> visible train.jsonl
        val = heldout.read(dst / "data" / "heldout_val.bin")
        with open(dst / "data" / "train.jsonl", "w") as f:
            for row in val:
                f.write(json.dumps(row) + "\n")
        n = len(val)
    cfg = json.loads((dst / "config.json").read_text())
    cfg["name"] = f"{task}_exposed"
    (dst / "config.json").write_text(json.dumps(cfg, indent=2))
    return n


def main():
    if "--rm" in sys.argv[1:]:
        for t in GEN_TASKS:
            d = ROOT / "bench" / "tasks" / f"{t}_exposed"
            if d.exists():
                shutil.rmtree(d)
        print("removed exposed variants")
        return
    for t in GEN_TASKS:
        n = make_one(t)
        print(f"created {t}_exposed  (train now = {n} decoded val instances)")


if __name__ == "__main__":
    main()
