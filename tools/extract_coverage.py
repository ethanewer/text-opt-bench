"""Aggregate the official coverage into a committed results file.

Every run below uses the identical canonical setup (gpt-5.5, 1-hour box, 40-iter
cap, codex-timeout 1200, 5 runs), so the three configs are directly comparable.
Config -> campaign-prefix mapping (reused where an existing campaign already
ran at this exact setup; filled otherwise):

  low   : 5xE- coverage runs
  none  : CMPN- and cov-none- coverage runs
  lowvv : cov-lowvv- (the 3 generalization *_exposed variants; val made visible)

Writes docs/coverage_results.md and docs/coverage_results.json.
"""
import glob, json, os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORE = ["compress_heldout", "easy_word_problems", "mem_index", "mem_infer",
        "mem_str", "ops_connect", "tag_seq"]
GEN = ["easy_word_problems", "compress_heldout", "tag_seq"]

LOW_PREFIX = {t: "5xE-" for t in CORE}
LOW_PREFIX.update({"tag_seq": "GENF2-"})
NONE_PREFIX = {t: "CMPN-" for t in ["compress_heldout", "easy_word_problems",
                                     "tag_seq", "ops_connect"]}
NONE_PREFIX.update({t: "cov-none-" for t in ["mem_index", "mem_infer", "mem_str"]})


def run_dirs(task, prefix):
    out = []
    for d in sorted(glob.glob(f"{ROOT}/runs/{task}/{prefix}*")):
        log = os.path.join(d, "log.jsonl")
        if not os.path.exists(log):
            continue
        acc = []
        for line in open(log):
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except Exception:
                continue
            if "iter" in o and o.get("accepted") and isinstance(o.get("score"), (int, float)):
                acc.append(o["score"])
        if acc:
            out.append((d, min(acc)))
    return out


def mean(xs):
    return sum(xs) / len(xs) if xs else None


def final_val_test(task, prefix):
    ev = f"{ROOT}/bench/tasks/{task}/evaluate.py"
    out = []
    for d, _ in run_dirs(task, prefix):
        bp = os.path.join(d, "best_program.py")
        if not (os.path.exists(bp) and os.path.exists(ev)):
            continue
        try:
            r = subprocess.run([sys.executable, ev, bp, "--final"],
                               capture_output=True, text=True, timeout=900, cwd=str(ROOT))
            ls = [l for l in r.stdout.splitlines() if l.strip().startswith("{")]
            if ls:
                m = json.loads(ls[-1]).get("metrics", {})
                if "val_score" in m and "test_score" in m:
                    out.append((m["val_score"], m["test_score"]))
        except Exception:
            pass
    return out


def g(x):
    if x is None:
        return "-"
    return f"{x:.4g}"


def main():
    # The val-exposed variants are experiment artifacts, not core tasks (kept out
    # of bench/tasks). Regenerate them if their lowvv runs exist but the dirs are
    # gone, so the overfitting val/test evals below are self-contained.
    if (glob.glob(f"{ROOT}/runs/easy_word_problems_exposed/cov-lowvv-*")
            and not (ROOT / "bench" / "tasks" / "easy_word_problems_exposed").exists()):
        subprocess.run([sys.executable, str(ROOT / "tools" / "make_exposed_variants.py")],
                       cwd=str(ROOT))
    results = {"setup": "gpt-5.5, 1h box (3600s), 40-iter cap, codex-timeout 1200, 5 runs",
               "coverage": {}, "overfitting": {}}
    # coverage table
    cov_rows = []
    for t in CORE:
        nd = run_dirs(t, NONE_PREFIX[t])
        ld = run_dirs(t, LOW_PREFIX[t])
        vd = run_dirs(t + "_exposed", "cov-lowvv-") if t in GEN else []
        n_none, n_low, n_vv = len(nd), len(ld), len(vd)
        b_none = min((s for _, s in nd), default=None)
        b_low = min((s for _, s in ld), default=None)
        b_vv = min((s for _, s in vd), default=None)
        results["coverage"][t] = {
            "N_none": n_none, "N_low": n_low, "N_low_valvisible": n_vv,
            "best_none": b_none, "best_low": b_low, "best_low_valvisible": b_vv,
            "none_prefix": NONE_PREFIX[t], "low_prefix": LOW_PREFIX[t],
        }
        cov_rows.append((t, n_none, n_low, n_vv, b_none, b_low, b_vv))
    # overfitting: hidden-val (low) vs exposed-val (lowvv), val + test
    ovf_rows = []
    for t in GEN:
        hid = final_val_test(t, LOW_PREFIX[t])
        exp = final_val_test(t + "_exposed", "cov-lowvv-")
        hv, ht = mean([v for v, _ in hid]), mean([x for _, x in hid])
        ev, et = mean([v for v, _ in exp]), mean([x for _, x in exp])
        results["overfitting"][t] = {"hidden_val": hv, "hidden_test": ht,
                                     "exposed_val": ev, "exposed_test": et}
        ovf_rows.append((t, hv, ht, ev, et))

    (ROOT / "docs" / "coverage_results.json").write_text(json.dumps(results, indent=2))

    md = ["# text-opt-bm — official coverage results",
          "", f"**Canonical setup (identical for every run, comparable + reproducible):** "
          f"{results['setup']}.", "",
          "Reproduce with `tools/run_official_coverage.sh` (config -> prefix: low <- 5xE-+GENF2-, "
          "none <- CMPN-+cov-none-, lowvv <- cov-lowvv-; `tools/make_exposed_variants.py` builds "
          "the val-exposed variants).", "",
          "## Coverage (N runs) and best score per task", "",
          "| task | N none | N low | N low(val-vis) | best none | best low | best low(val-vis) |",
          "|---|---|---|---|---|---|---|"]
    for t, nn, nl, nv, bn, bl, bv in cov_rows:
        md.append(f"| {t} | {nn} | {nl} | {nv if t in GEN else '—'} | {g(bn)} | {g(bl)} | {g(bv) if t in GEN else '—'} |")
    md += ["", "## Overfitting arm — hidden-val (low) vs exposed-val (low, val visible)", "",
           "Same val instances, both evaluated on the always-hidden test split. Exposing the "
           "eval data lets the optimizer drive val→~0 while test does not follow.", "",
           "| task | hidden val | hidden test | exposed val | exposed test |",
           "|---|---|---|---|---|"]
    for t, hv, ht, ev, et in ovf_rows:
        md.append(f"| {t} | {g(hv)} | {g(ht)} | {g(ev)} | {g(et)} |")
    md.append("")
    (ROOT / "docs" / "coverage_results.md").write_text("\n".join(md))
    print("wrote docs/coverage_results.{md,json}")
    print("\nCoverage summary (N none / N low / N low-valvis):")
    for t, nn, nl, nv, *_ in cov_rows:
        flag = "" if (nn >= 5 and nl >= 5 and (t not in GEN or nv >= 5)) else "  <-- INCOMPLETE"
        print(f"  {t:<17} {nn} / {nl} / {nv if t in GEN else '-'}{flag}")


if __name__ == "__main__":
    main()
