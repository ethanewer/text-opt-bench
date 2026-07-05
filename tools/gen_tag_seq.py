"""One-off generator for the tag_seq task data.

Run:  TAG_SEQ_SEED=<int> python3.12 tools/gen_tag_seq.py
  (or: python3.12 tools/gen_tag_seq.py <int>)

Writes train.jsonl (visible) and obfuscated heldout_val.bin /
heldout_test.bin into bench/tasks/tag_seq/data/.

WHAT THIS BUILDS
----------------
A per-token SEQUENCE LABELING task (a synthetic morphological tagger).
Each example is a sequence of made-up word tokens; the program must emit
one TAG (a letter "A".."F") per token.

Every token is  stem + suffix .  The last two characters (the SUFFIX)
determine the token's CLASS c in 0..C-1.  Stems are random and mostly
UNSEEN across splits, so a token STRING almost never repeats between
train and the hidden splits -- a learner that memorizes token->tag will
not generalize; it must key on the suffix (class) instead.

The GENERAL rule (the head): a token's tag is BASE[c], a fixed
class->tag mapping.  This is broad and easy to read off the data.

The EXCEPTION tail (the idiosyncratic part): a long ordered list of
~N_EXC narrow exception rules, each keyed on a LOCAL CLASS CONTEXT
(the classes of the token and its immediate neighbours / its position),
that OVERRIDE the general tag with an arbitrary sealed tag.  The list is
evaluated LAST-MATCH-WINS.  The override tags are drawn from a fixed
PUBLIC construction seed -- they are idiosyncratic (no formula), so no
single compact rule reproduces them; a solver must recover the whole
list, one exception at a time.

The visible TRAIN split UNDERSAMPLES the exception contexts (they are
resampled away with keep-probability KEEP during train generation),
while the hidden val/test splits contain them at the full natural rate.
So a solver that only captures the broad head, OR that fits a general
context model on train alone, plateaus well above the achievable floor;
recovering the long exception tail takes many iterations of val
feedback.  A small per-token label-noise rate sets a Bayes floor > 0.

NOTE FOR OPTIMIZING AGENTS: reading this file, or using it to
re-generate or infer held-out sequences/labels, is cheating and
disqualifies the run.  Only bench/tasks/tag_seq/data/train.jsonl may be
used.

The data SEED is read from the environment / argv and is NOT stored in
any committed file; it is discarded after this script runs once, so the
hidden splits cannot be regenerated.
"""

import json
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bench import heldout

# ============================================================
# PUBLIC CONSTRUCTION (fixed structure, NOT the secret data seed)
# The suffix table, the head mapping BASE, and the exception list are
# derived from a PUBLIC construction seed.  They are the "ground-truth
# model" the reference solution embodies.  A candidate never sees them
# (tools/ and tests/ are off-limits); it must recover them from data +
# validation feedback.
# ============================================================

C = 12            # number of token classes (one per suffix)
T = 6             # number of tags
TAGS = ["A", "B", "C", "D", "E", "F"]
CONS_SUF = "bdfgklmnprstvz"
VOW_SUF = "aeiou"
BUILD_SEED = 0x7A6C21     # public
N_EXC = int(os.environ.get("TAG_SEQ_NEXC", 72))
NOISE = float(os.environ.get("TAG_SEQ_NOISE", 0.075))   # per-token label noise
KEEP = float(os.environ.get("TAG_SEQ_KEEP", 0.10))      # train exc keep-prob
N_TRAIN = int(os.environ.get("TAG_SEQ_NTRAIN", 700))
N_VAL = int(os.environ.get("TAG_SEQ_NVAL", 450))
N_TEST = int(os.environ.get("TAG_SEQ_NTEST", 900))


def _build_suffixes(r):
    seen = set()
    out = []
    while len(out) < C:
        s = r.choice(CONS_SUF) + r.choice(VOW_SUF)
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _build_base(r):
    # Broad head mapping class -> tag.  Slightly uneven so a couple of tags
    # dominate the head (realistic), but every tag is used.
    base = [r.randrange(T) for _ in range(C)]
    # ensure every tag appears at least once in the head
    missing = [t for t in range(T) if t not in base]
    for t in missing:
        base[r.randrange(C)] = t
    return base


def _build_rules(r):
    """Ordered exception list (last match wins).

    Each rule is (form, params, tag).  Forms (all keyed on CLASSES, so they
    generalise across sequences with unseen stems):
      "pc"  : prev_class == p and cur_class == c            (bigram)
      "cn"  : cur_class == c and next_class == n            (bigram)
      "pcn" : prev==p and cur==c and next==n                (trigram)
      "skip": prev2_class == pp and cur_class == c          (skip-bigram)
      "first": cur_class == c and position == 0
      "last" : cur_class == c and position == last
    """
    rules = []
    for _ in range(N_EXC):
        u = r.random()
        tag = r.randrange(T)
        if u < 0.42:
            rules.append(("pc", (r.randrange(C), r.randrange(C)), tag))
        elif u < 0.72:
            rules.append(("cn", (r.randrange(C), r.randrange(C)), tag))
        elif u < 0.82:
            rules.append(("skip", (r.randrange(C), r.randrange(C)), tag))
        elif u < 0.90:
            rules.append(("pcn", (r.randrange(C), r.randrange(C),
                                  r.randrange(C)), tag))
        elif u < 0.95:
            rules.append(("first", (r.randrange(C),), tag))
        else:
            rules.append(("last", (r.randrange(C),), tag))
    return rules


_R = random.Random(BUILD_SEED)
SUFFIXES = _build_suffixes(_R)
BASE = _build_base(_R)
RULES = _build_rules(_R)


# ============================================================
# RULE ENGINE  (identical copy lives in tests/solutions/tag_seq.py and
# tests/broken/tag_seq_headonly.py -- keep them in sync via the printed
# literals below).
# ============================================================

def _tag_at(classes, i, base, rules):
    """Apply head + last-match-wins exception list at position i."""
    L = len(classes)
    c = classes[i]
    p = classes[i - 1] if i - 1 >= 0 else -1
    n = classes[i + 1] if i + 1 < L else -1
    pp = classes[i - 2] if i - 2 >= 0 else -1
    tag = base[c]
    for form, params, t in rules:
        if form == "pc":
            if p == params[0] and c == params[1]:
                tag = t
        elif form == "cn":
            if c == params[0] and n == params[1]:
                tag = t
        elif form == "skip":
            if pp == params[0] and c == params[1]:
                tag = t
        elif form == "pcn":
            if p == params[0] and c == params[1] and n == params[2]:
                tag = t
        elif form == "first":
            if i == 0 and c == params[0]:
                tag = t
        elif form == "last":
            if i == L - 1 and c == params[0]:
                tag = t
    return tag


def _head_tag_at(classes, i, base):
    return base[classes[i]]


# ============================================================
# DATA GENERATION (uses the SECRET seed)
# ============================================================

def _rand_stem(rng):
    cons = "bcdfghjklmnpqrstvwxyz"
    vow = "aeiou"
    n = rng.randrange(2, 4)   # syllables
    s = []
    for _ in range(n):
        s.append(rng.choice(cons))
        s.append(rng.choice(vow))
    if rng.random() < 0.4:
        s.append(rng.choice(cons))
    return "".join(s)


def _token(rng, c):
    return _rand_stem(rng) + SUFFIXES[c]


def _is_exc(classes, i):
    return _tag_at(classes, i, BASE, RULES) != _head_tag_at(classes, i, BASE)


def _undersample_train(rng, classes):
    """Resample tokens whose CUR-position triggers an exception, keeping
    only a KEEP fraction of them.  Recomputed labels on the final sequence
    stay self-consistent; this thins exception contexts in train without
    dropping tokens (which would corrupt neighbour contexts)."""
    L = len(classes)
    for _ in range(3):
        changed = False
        for i in range(L):
            if _is_exc(classes, i) and rng.random() > KEEP:
                order = list(range(C))
                rng.shuffle(order)
                for cprime in order:
                    if cprime == classes[i]:
                        continue
                    old = classes[i]
                    classes[i] = cprime
                    if not _is_exc(classes, i):
                        changed = True
                        break
                    classes[i] = old
        if not changed:
            break
    return classes


def _make_split(rng, n, undersample):
    out = []
    for _ in range(n):
        L = rng.randrange(9, 17)
        classes = [rng.randrange(C) for _ in range(L)]
        if undersample:
            classes = _undersample_train(rng, classes)
        tokens = [_token(rng, c) for c in classes]
        tags = []
        for i in range(L):
            true = _tag_at(classes, i, BASE, RULES)
            if rng.random() < NOISE:
                obs = rng.choice([t for t in range(T) if t != true])
            else:
                obs = true
            tags.append(TAGS[obs])
        out.append({"tokens": tokens, "tags": tags})
    return out


def _err(split, tagger):
    wrong = tot = 0
    for row in split:
        classes = [SUFFIXES.index(tk[-2:]) for tk in row["tokens"]]
        for i, tk in enumerate(row["tokens"]):
            tot += 1
            if TAGS[tagger(classes, i)] != row["tags"][i]:
                wrong += 1
    return round(wrong / tot, 4)


def _general_model_floor(train, val):
    """Best a SINGLE general learned model reaches: back-off context
    majority-vote fit on TRAIN, evaluated on VAL.  Learns, per context key,
    the majority observed tag, backing off trigram -> bigram(s) -> class.
    Because train undersamples the exception contexts, this cannot recover
    the tail and stays well above the reference floor."""
    tg = defaultdict(Counter)
    pc = defaultdict(Counter)
    cn = defaultdict(Counter)
    cls = defaultdict(Counter)
    for row in train:
        cs = [SUFFIXES.index(tk[-2:]) for tk in row["tokens"]]
        L = len(cs)
        for i in range(L):
            c = cs[i]
            p = cs[i - 1] if i - 1 >= 0 else -1
            n = cs[i + 1] if i + 1 < L else -1
            y = row["tags"][i]
            tg[(p, c, n)][y] += 1
            pc[(p, c)][y] += 1
            cn[(c, n)][y] += 1
            cls[c][y] += 1

    def pred(cs, i):
        c = cs[i]
        p = cs[i - 1] if i - 1 >= 0 else -1
        n = cs[i + 1] if i + 1 < len(cs) else -1
        for key, tbl in (((p, c, n), tg), ((p, c), pc), ((c, n), cn),
                         (c, cls)):
            ctr = tbl[key]
            if sum(ctr.values()) >= 5:
                return ctr.most_common(1)[0][0]
        return cls[c].most_common(1)[0][0] if cls[c] else "A"

    wrong = tot = 0
    for row in val:
        cs = [SUFFIXES.index(tk[-2:]) for tk in row["tokens"]]
        for i in range(len(cs)):
            tot += 1
            if pred(cs, i) != row["tags"][i]:
                wrong += 1
    return round(wrong / tot, 4)


def main():
    seed_str = os.environ.get("TAG_SEQ_SEED")
    if seed_str is None and len(sys.argv) > 1:
        seed_str = sys.argv[1]
    if not seed_str:
        sys.exit("set TAG_SEQ_SEED=<int> (secret; not stored anywhere)")
    seed = int(seed_str, 0)

    data_dir = ROOT / "bench" / "tasks" / "tag_seq" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    train = _make_split(random.Random(seed ^ 0x1111), N_TRAIN, undersample=True)
    val = _make_split(random.Random(seed ^ 0x2222), N_VAL, undersample=False)
    test = _make_split(random.Random(seed ^ 0x3333), N_TEST, undersample=False)

    with open(data_dir / "train.jsonl", "w") as f:
        for row in train:
            f.write(json.dumps(row) + "\n")
    heldout.write(data_dir / "heldout_val.bin", val)
    heldout.write(data_dir / "heldout_test.bin", test)

    ref = lambda cs, i: _tag_at(cs, i, BASE, RULES)
    head = lambda cs, i: _head_tag_at(cs, i, BASE)
    print(f"wrote {len(train)} train, {len(val)} val, {len(test)} test")
    print(f"C={C} T={T} N_EXC={N_EXC} NOISE={NOISE} KEEP={KEEP}")

    def exc_frac(split):
        e = tot = 0
        for row in split:
            cs = [SUFFIXES.index(tk[-2:]) for tk in row["tokens"]]
            for i in range(len(cs)):
                tot += 1
                if _is_exc(cs, i):
                    e += 1
        return round(e / tot, 4)

    print(f"train exc_frac = {exc_frac(train)}   val exc_frac = {exc_frac(val)}")
    print(f"reference_val_err  = {_err(val, ref)}")
    print(f"head_only_val_err  = {_err(val, head)}")
    print(f"general_model_floor(val) = {_general_model_floor(train, val)}")
    print(f"reference_test_err = {_err(test, ref)}")
    tagcnt = Counter()
    for row in val:
        tagcnt.update(row["tags"])
    print("val tag dist:", dict(sorted(tagcnt.items())))

    # ---- print literals for pasting into solution / headonly ----
    if os.environ.get("TAG_SEQ_DUMP"):
        print("\n# ---- LITERALS ----")
        print("SUFFIXES =", SUFFIXES)
        print("BASE =", BASE)
        print("RULES =", RULES)


if __name__ == "__main__":
    main()
