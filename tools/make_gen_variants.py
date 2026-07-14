"""Create the Exp-2 and Exp-3 generalization task variants from the base
(new-default) generalization tasks.

Like the old make_exposed_variants, these variants are REGENERABLE artifacts,
NOT part of the committed core benchmark (kept out of git; the results are
what get committed). The frozen hidden test is copied byte-identically into
every variant, so all experiments share the same test.

For each base gen task <t> (train = D visible+graded, test = T hidden):

  <t>_r8   Exp-3, train:test = 1:8   -> visible graded train = D/2 (a prefix of
           the base train pool). No val. Score = train error (same as default).
  <t>_r16  Exp-3, train:test = 1:16  -> visible graded train = D/4. No val.
  <t>_e2   Exp-2, restricted info    -> visible train = K smoke examples; a
           HIDDEN VALIDATION set (the rest of the base train pool, D-K) is the
           graded score (agent sees only the aggregate). No test leak.

All variants run in the default `full` feedback: _r8/_r16 have no heldout_val.bin
so the val-optional evaluator scores TRAIN; _e2 has heldout_val.bin so it scores
VAL. The hidden test is sealed every iteration regardless.

Usage:  python3.12 tools/make_gen_variants.py         # (re)create all
        python3.12 tools/make_gen_variants.py --rm    # remove them
"""
import base64
import importlib.util
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bench import heldout

ROW_TASKS = ["easy_word_problems", "tag_seq"]
CMP = "compress_heldout"
ALL = ROW_TASKS + [CMP]
SUFFIXES = ["r8", "r16", "e2"]
K = 5                       # Exp-2 visible smoke-train size (row/seq tasks)

# compress_heldout byte-per-doc targets (ratio applied in bytes; 1 doc/genre)
CMP_TRAIN_SEED = 0xA11CE    # matches tools/gen_compress_heldout.py
CMP_VIS_SEED = 0xA11CF      # distinct -> Exp-2 visible smoke docs disjoint from val
CMP_DEFAULT = 50_000        # base default train doc size
CMP_R8, CMP_R16, CMP_E2_VIS = 25_000, 12_500, 5_000


def _fresh_copy(t, suffix):
    dst = ROOT / "bench" / "tasks" / f"{t}_{suffix}"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(ROOT / "bench" / "tasks" / t, dst)
    # drop any pycache and stale val copied from the base
    for p in dst.rglob("__pycache__"):
        shutil.rmtree(p, ignore_errors=True)
    return dst


def _set_name(dst, name):
    cfg = json.loads((dst / "config.json").read_text())
    cfg["name"] = name
    (dst / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")


def _prepend_spec(dst, note):
    spec = (dst / "spec.md").read_text()
    (dst / "spec.md").write_text(note.rstrip() + "\n\n" + spec)


def _write_rows(dst, rows):
    with open(dst / "data" / "train.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def make_row_variants(t):
    train = [json.loads(l) for l in open(ROOT / "bench" / "tasks" / t / "data" / "train.jsonl")]
    D = len(train)
    for suf, n, ratio in [("r8", D // 2, 8), ("r16", D // 4, 16)]:
        dst = _fresh_copy(t, suf)
        _write_rows(dst, train[:n])                      # prefix subset, no val
        (dst / "data" / "heldout_val.bin").unlink(missing_ok=True)
        _set_name(dst, f"{t}_{suf}")
        _prepend_spec(dst, f"> **Variant `{t}_{suf}` — Experiment 3 (train:test = 1:{ratio}).** "
                      f"Identical to the default, but the visible GRADED train set is only **{n}** "
                      f"examples (a subset of the default train pool). Same frozen hidden test. "
                      f"Score = train error.")
    # Exp-2: K visible smoke examples + hidden validation = the rest of the pool
    dst = _fresh_copy(t, "e2")
    _write_rows(dst, train[:K])
    heldout.write(dst / "data" / "heldout_val.bin", train[K:])
    _set_name(dst, f"{t}_e2")
    _prepend_spec(dst, f"> **Variant `{t}_e2` — Experiment 2 (restricted information).** You see only "
                  f"**{K}** visible TRAIN examples (for smoke-testing your code). Your SCORE is the error "
                  f"on a **HIDDEN VALIDATION set of {D - K}** examples — you see only the aggregate score, "
                  f"never the data. A separate hidden TEST set still measures generalization. This overrides "
                  f"the split sizes described below.")


def _load_gen_corpus():
    spec = importlib.util.spec_from_file_location(
        "compress_corpus", ROOT / "tools" / "compress_corpus.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.stdout = sys.__stdout__          # eval_lib redirects stdout on import; undo
    return mod.gen_corpus


def make_compress_variants():
    gen_corpus = _load_gen_corpus()

    def write_train(dst, seed, target):
        for p in (dst / "data").glob("train_*.txt"):
            p.unlink()
        for name, data in gen_corpus(seed, target).items():
            (dst / "data" / f"train_{name}.txt").write_bytes(data)

    for suf, target, ratio in [("r8", CMP_R8, 8), ("r16", CMP_R16, 16)]:
        dst = _fresh_copy(CMP, suf)
        write_train(dst, CMP_TRAIN_SEED, target)
        (dst / "data" / "heldout_val.bin").unlink(missing_ok=True)
        _set_name(dst, f"{CMP}_{suf}")
        _prepend_spec(dst, f"> **Variant `{CMP}_{suf}` — Experiment 3 (train:test = 1:{ratio} in bytes).** "
                      f"Visible GRADED train = 4 documents (one per genre, ~{target // 1000} KB each). Same "
                      f"frozen hidden test. Score = total compressed train bytes.")
    # Exp-2: tiny visible docs (distinct seed -> disjoint) + hidden val = the pool
    dst = _fresh_copy(CMP, "e2")
    write_train(dst, CMP_VIS_SEED, CMP_E2_VIS)
    val_docs = gen_corpus(CMP_TRAIN_SEED, CMP_DEFAULT)
    heldout.write(dst / "data" / "heldout_val.bin",
                  {name: base64.b64encode(data).decode() for name, data in val_docs.items()})
    _set_name(dst, f"{CMP}_e2")
    _prepend_spec(dst, f"> **Variant `{CMP}_e2` — Experiment 2 (restricted information).** You see only 4 tiny "
                  f"(~{CMP_E2_VIS // 1000} KB) visible train documents (one per genre) for smoke-testing. Your "
                  f"SCORE is the compressed size on a HIDDEN VALIDATION corpus (4 docs, ~{CMP_DEFAULT // 1000} KB "
                  f"each) you never see. A separate hidden TEST corpus still measures generalization.")


def main():
    if "--rm" in sys.argv[1:]:
        n = 0
        for t in ALL:
            for suf in SUFFIXES:
                d = ROOT / "bench" / "tasks" / f"{t}_{suf}"
                if d.exists():
                    shutil.rmtree(d)
                    n += 1
        print(f"removed {n} variant dirs")
        return
    for t in ROW_TASKS:
        make_row_variants(t)
    make_compress_variants()
    made = [f"{t}_{s}" for t in ALL for s in SUFFIXES]
    print(f"created {len(made)} variants: {', '.join(made)}")


if __name__ == "__main__":
    main()
