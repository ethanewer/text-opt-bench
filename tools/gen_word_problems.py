"""One-off generator for the word_problems task data (v3).

Run:  python3.12 tools/gen_word_problems.py

Writes train.jsonl (visible) and obfuscated heldout_val.bin /
heldout_test.bin into bench/tasks/word_problems/data/.

History: v1 (10 rigid templates) was solved to 0% error in ONE codex
iteration; v2 (16 templates + distractors + number words) to 3%; v3
(compositional event chains etc.) to 1.6%. Strong models reverse-engineer
any grammar they can see enough samples of. v4 therefore adds (a) an
idiomatic long tail — coin values, "all but N", "doubles their pile",
"loses half of them", ratio sentences, mixed week/day durations, "half
as many again" — and (b) a deliberately SMALL train split (100 problems)
so that many family x template x idiom combinations occur only in the
hidden splits: one-pass solvers cover what they saw, and further
progress requires genuine generalization guided only by the aggregate
validation score.

NOTE FOR OPTIMIZING AGENTS: reading this file, or using it to re-generate
or infer held-out questions/answers, is cheating and disqualifies the run.
Only bench/tasks/word_problems/data/train.jsonl may be used.
"""

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bench import heldout

NAMES = [
    "Liam", "Emma", "Noah", "Olivia", "Ava", "Ethan", "Mia", "Lucas",
    "Sofia", "Mason", "Isla", "Leo", "Zoe", "Owen", "Ruby", "Caleb",
    "Nina", "Felix", "Tara", "Hugo", "Priya", "Diego", "Wren", "Omar",
    "June", "Silas", "Anya", "Marco", "Edie", "Ravi",
]
SMALL_ITEMS = [
    "marbles", "stickers", "pencils", "seashells", "trading cards",
    "buttons", "coins", "beads", "stamps", "bottle caps", "acorns",
    "paper cranes", "erasers", "pins", "tokens", "ribbons", "shells",
]
SHOP_ITEMS = ["notebook", "pen", "mug", "poster", "candle", "keychain",
              "plant", "scarf", "puzzle", "cap", "kite", "lantern"]
VEHICLES = ["train", "bus", "car", "cyclist", "truck", "ferry", "hiker",
            "boat", "van"]
CONTAINERS = ["crate", "box", "carton", "basket", "bag", "tray", "bin"]
UNITS = ["bottles", "apples", "oranges", "cans", "jars", "eggs", "muffins",
         "bricks", "cookies", "peaches"]
ROOMS = ["garden", "patio", "field", "playground", "courtyard", "lawn",
         "terrace"]
COLORS = ["red", "green", "blue", "yellow", "striped", "spotted", "white"]

ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven",
        "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen",
        "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
        "eighty", "ninety"]


def number_word(n):
    if n < 20:
        return ONES[n]
    t, o = divmod(n, 10)
    return TENS[t] + ("-" + ONES[o] if o else "")


def num(rng, n):
    """Render a number, often as words, to defeat naive digit-grabbing."""
    if n == 12 and rng.random() < 0.25:
        return "a dozen"
    if n == 24 and rng.random() < 0.15:
        return "two dozen"
    if n < 100 and rng.random() < 0.35:
        return number_word(n)
    return str(n)


GAIN_VERBS = ["finds", "buys", "wins", "is given", "picks up", "earns",
              "collects"]
LOSE_VERBS = ["loses", "gives away", "sells", "drops", "donates",
              "misplaces"]
PLURAL_VERB = {
    "finds": "find", "buys": "buy", "wins": "win", "is given": "are given",
    "picks up": "pick up", "earns": "earn", "collects": "collect",
    "loses": "lose", "gives away": "give away", "sells": "sell",
    "drops": "drop", "donates": "donate", "misplaces": "misplace",
}

FILLERS = [
    " It was a busy afternoon.",
    " Everyone was excited.",
    " The weather was perfect that day.",
    " It took longer than expected.",
    " Nobody kept track of the time.",
]


def distractor(rng, exclude):
    who = rng.choice([n for n in NAMES if n not in exclude])
    item = rng.choice(SMALL_ITEMS + UNITS)
    n = rng.randrange(2, 80)
    return rng.choice([
        f" {who} also counted {num(rng, n)} {item} yesterday.",
        f" Earlier, {who} had sorted {num(rng, n)} {item} into a drawer.",
        f" ({who} once collected {num(rng, n)} {item}, but that was last year.)",
        f" Meanwhile {who} brought {num(rng, n)} {item} to school.",
        f" A neighbor, {who}, keeps {num(rng, n)} {item} at home.",
    ])


def decorate(rng, body, names):
    if rng.random() < 0.5:
        body = body.rstrip() + distractor(rng, names)
    if rng.random() < 0.25:
        body = body.rstrip() + rng.choice(FILLERS)
    return body


# --------------------------------------------------------------------
# Family 1: event/transfer chains over 1-3 entities
# --------------------------------------------------------------------

def f_chain(rng):
    n_entities = rng.choice([1, 1, 2, 2, 3])
    people = rng.sample(NAMES, n_entities)
    item = rng.choice(SMALL_ITEMS)
    counts = {p: rng.randrange(10, 60) for p in people}
    intro_bits = [f"{p} has {num(rng, c)} {item}" for p, c in counts.items()]
    sentences = [", ".join(intro_bits[:-1]) + (" and " if len(intro_bits) > 1 else "") + intro_bits[-1] + "."]
    sentences[-1] = sentences[-1][0].upper() + sentences[-1][1:]

    last_subject = None
    n_events = rng.randrange(2, 6)
    for _ in range(n_events):
        p = rng.choice(people)
        can_transfer = n_entities >= 2 and counts[p] > 4
        kind = rng.random()
        if kind < 0.18:
            # idiomatic events: multiplicative / "all but" / halves
            idiom = rng.randrange(4)
            subj = "They" if p == last_subject and rng.random() < 0.35 else p
            if idiom == 0:
                counts[p] *= 2
                s = rng.choice([
                    f"{subj} then double{'s' if subj != 'They' else ''} their pile.",
                    f"A lucky trade doubles {p}'s pile.",
                ])
            elif idiom == 1 and counts[p] >= 6:
                keep = rng.randrange(2, min(counts[p] - 1, 12))
                counts[p] = keep
                s = f"{subj} {'loses' if subj != 'They' else 'lose'} all but {num(rng, keep)} of them."
            elif idiom == 2 and counts[p] % 2 == 0 and counts[p] >= 8:
                counts[p] //= 2
                s = f"{subj} {'gives' if subj != 'They' else 'give'} half of them away."
            else:
                n = rng.randrange(2, 20)
                counts[p] += n
                s = f"{subj} {'finds' if subj != 'They' else 'find'} {num(rng, n)} more."
            last_subject = p
            sentences.append(s)
            continue
        if can_transfer and kind < 0.35:
            q = rng.choice([x for x in people if x != p])
            n = rng.randrange(2, counts[p] - 1)
            counts[p] -= n
            counts[q] += n
            s = rng.choice([
                f"{p} gives {num(rng, n)} to {q}.",
                f"{p} hands {q} {num(rng, n)} of them.",
                f"Then {p} passes {num(rng, n)} along to {q}.",
            ])
        elif kind < 0.7 or counts[p] < 6:
            n = rng.randrange(2, 25)
            counts[p] += n
            verb = rng.choice(GAIN_VERBS)
            subj = p
            if p == last_subject and rng.random() < 0.35:
                subj = "They"
                verb = PLURAL_VERB[verb]
            s = rng.choice([
                f"{subj} {verb} {num(rng, n)} more.",
                f"Later, {subj.lower() if subj == 'They' else subj} {verb} another {num(rng, n)}.",
            ])
        else:
            n = rng.randrange(2, counts[p] - 1)
            counts[p] -= n
            verb = rng.choice(LOSE_VERBS)
            subj = p
            if p == last_subject and rng.random() < 0.35:
                subj = "They"
                verb = PLURAL_VERB[verb]
            s = f"{subj} {verb} {num(rng, n)}."
        last_subject = p
        sentences.append(s)

    body = " ".join(sentences)
    target = rng.random()
    if n_entities == 1 or target < 0.5:
        p = rng.choice(people)
        q_str = rng.choice([
            f" How many {item} does {p} have now?",
            f" How many {item} does {p} end up with?",
        ])
        ans = counts[p]
    elif target < 0.8:
        q_str = rng.choice([
            f" How many {item} do they have altogether now?",
            f" In total, how many {item} do they have at the end?",
        ])
        ans = sum(counts.values())
    else:
        a, b = rng.sample(people, 2)
        if counts[a] < counts[b]:
            a, b = b, a
        q_str = f" How many more {item} does {a} have than {b} now?"
        ans = counts[a] - counts[b]
    return decorate(rng, body, people) + q_str, ans


# --------------------------------------------------------------------
# Family 2: shopping (multi-item, change, money short)
# --------------------------------------------------------------------

def f_shopping(rng):
    name = rng.choice(NAMES)
    n_items = rng.choice([1, 2, 2, 3])
    items = rng.sample(SHOP_ITEMS, n_items)
    parts = []
    total = 0
    for it in items:
        k = rng.randrange(1, 7)
        p = rng.randrange(2, 15)
        total += k * p
        if k == 1:
            parts.append(rng.choice([
                f"a {it} for {num(rng, p)} dollars",
                f"one {it} priced at {num(rng, p)} dollars",
            ]))
        else:
            parts.append(rng.choice([
                f"{num(rng, k)} {it}s at {num(rng, p)} dollars each",
                f"{num(rng, k)} {it}s for {num(rng, p)} dollars apiece",
            ]))
    listing = parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]
    body = rng.choice([
        f"{name} buys {listing}.",
        f"At the market, {name} picks up {listing}.",
    ])
    mode = rng.random()
    if mode < 0.4:
        q = " How many dollars does " + name + " spend in total?"
        ans = total
    elif mode < 0.75:
        paid = rng.choice([20, 50, 100])
        while paid < total:
            paid *= 2
        body += f" {name} pays with a {paid} dollar bill."
        q = rng.choice([" How much change is due?",
                        f" How much change does {name} get back?"])
        ans = paid - total
    else:
        short = rng.randrange(1, max(2, total - 1))
        has = total - short
        body += f" {name} only has {num(rng, has)} dollars."
        q = f" How many more dollars does {name} need?"
        ans = short
    return decorate(rng, body, [name]) + q, ans


# --------------------------------------------------------------------
# Family 3: rates (distance legs, savings, pages/day)
# --------------------------------------------------------------------

def f_rates(rng):
    mode = rng.random()
    name = rng.choice(NAMES)
    if mode < 0.4:
        v = rng.choice(VEHICLES)
        legs = rng.choice([2, 2, 3])
        total = 0
        parts = []
        for i in range(legs):
            s = rng.randrange(15, 75)
            t = rng.randrange(1, 6)
            total += s * t
            parts.append(
                f"{num(rng, t)} hour{'s' if t != 1 else ''} at {num(rng, s)} miles per hour"
            )
        seq = ", then ".join(parts)
        if rng.random() < 0.3 and legs == 2:
            body = f"A {v} makes a trip of {seq}, and finally returns exactly the way it came."
            ans = 2 * total
        else:
            body = rng.choice([
                f"A {v} travels {seq}.",
                f"On the journey, a {v} covers {seq}.",
            ])
            ans = total
        q = " How many miles is that in total?"
        return decorate(rng, body, [name]) + q, ans
    if mode < 0.75:
        w = rng.randrange(2, 10)
        daily = rng.random() < 0.5
        if daily:
            d = rng.randrange(2, 9)
            saved = d * 7 * w
            unit = "day"
        else:
            d = rng.randrange(5, 30)
            saved = d * w
            unit = "week"
        e = rng.randrange(3, saved - 1)
        body = rng.choice([
            f"{name} saves {num(rng, d)} dollars every {unit} for {num(rng, w)} weeks, then spends {num(rng, e)} dollars on a gift.",
            f"Putting aside {num(rng, d)} dollars a {unit}, {name} saves for {num(rng, w)} weeks and then spends {num(rng, e)} dollars.",
        ])
        q = f" How many dollars does {name} have left?"
        return decorate(rng, body, [name]) + q, saved - e
    pages = rng.randrange(80, 400)
    per = rng.randrange(8, 35)
    body = rng.choice([
        f"{name} reads {num(rng, per)} pages of a {num(rng, pages)} page book each day.",
        f"A book has {num(rng, pages)} pages, and {name} gets through {num(rng, per)} a day.",
    ])
    q = f" How many days does {name} need to finish it?"
    return decorate(rng, body, [name]) + q, -(-pages // per)


# --------------------------------------------------------------------
# Family 4: grouping (grids, containers, sharing)
# --------------------------------------------------------------------

def f_grouping(rng):
    name = rng.choice(NAMES)
    mode = rng.random()
    if mode < 0.35:
        rows = rng.randrange(3, 10)
        per = rng.randrange(8, 30)
        out = rng.randrange(2, rows * per - 1)
        thing, holder, gone = rng.choice([
            ("books", "shelves", "are checked out"),
            ("jackets", "racks", "get sold"),
            ("muffins", "trays", "are eaten"),
            ("chairs", "rows", "are taken away"),
        ])
        body = rng.choice([
            f"A hall has {num(rng, rows)} {holder} with {num(rng, per)} {thing} on each. {num(rng, out).capitalize()} {thing} {gone}.",
            f"{name} arranges {num(rng, rows)} {holder} of {num(rng, per)} {thing} each, but then {num(rng, out)} {thing} {gone}.",
        ])
        q = f" How many {thing} remain?"
        return decorate(rng, body, [name]) + q, rows * per - out
    if mode < 0.7:
        cap = rng.randrange(6, 30)
        need = rng.randrange(40, 400)
        cont = rng.choice(CONTAINERS)
        unit = rng.choice(UNITS)
        body = rng.choice([
            f"Each {cont} holds {num(rng, cap)} {unit}. A store needs to pack {num(rng, need)} {unit}.",
            f"{name} must pack {num(rng, need)} {unit} into {cont}s of {num(rng, cap)}.",
        ])
        q = f" How many {cont}s does that take?"
        return decorate(rng, body, [name]) + q, -(-need // cap)
    k = rng.randrange(3, 9)
    each = rng.randrange(4, 20)
    extra = rng.randrange(0, k)
    total = k * each + extra
    item = rng.choice(SMALL_ITEMS)
    if rng.random() < 0.5:
        body = f"{name} deals {num(rng, total)} {item} out evenly to {num(rng, k)} classmates."
        q = " How many does each classmate get?"
        ans = each
    else:
        body = f"{num(rng, total).capitalize()} {item} are shared equally by {num(rng, k)} friends."
        q = f" How many {item} are left over?"
        ans = extra
    return decorate(rng, body, [name]) + q, ans


# --------------------------------------------------------------------
# Family 5: percent and fractions (plus composed discount-change)
# --------------------------------------------------------------------

def f_percent(rng):
    name = rng.choice(NAMES)
    mode = rng.random()
    if mode < 0.35:
        pct = rng.choice([10, 20, 25, 50, 75])
        base = rng.randrange(2, 20) * (4 if pct in (25, 75) else 10 if pct == 10 else 5 if pct == 20 else 2)
        item = rng.choice(SHOP_ITEMS)
        up = rng.random() < 0.5
        if up:
            body = rng.choice([
                f"A {item} costs {num(rng, base)} dollars, and the price rises by {pct} percent.",
                f"The {num(rng, base)} dollar {item} gets {pct} percent more expensive.",
            ])
            q = " What is the new price in dollars?"
            ans = base + base * pct // 100
        else:
            body = rng.choice([
                f"A {item} priced at {num(rng, base)} dollars is marked down {pct} percent.",
                f"The store cuts the {num(rng, base)} dollar {item} by {pct} percent.",
            ])
            q = " What does it cost now in dollars?"
            ans = base - base * pct // 100
        return decorate(rng, body, [name]) + q, ans
    if mode < 0.65:
        pct = rng.choice([10, 20, 25, 50])
        base = rng.randrange(2, 15) * (4 if pct == 25 else 10 if pct == 10 else 5 if pct == 20 else 2)
        sale = base - base * pct // 100
        paid = rng.choice([50, 100])
        while paid < sale:
            paid *= 2
        item = rng.choice(SHOP_ITEMS)
        body = rng.choice([
            f"A {item} normally costs {num(rng, base)} dollars but is {pct} percent off today. {name} buys it with a {paid} dollar bill.",
            f"{name} uses a {paid} dollar bill to buy a {item} that was {num(rng, base)} dollars before a {pct} percent discount.",
        ])
        q = " How much change is due?"
        return decorate(rng, body, [name]) + q, paid - sale
    frac, word = rng.choice([(2, "half"), (3, "a third"), (4, "a quarter"),
                             (5, "a fifth")])
    per = rng.randrange(3, 15)
    total = frac * per
    item = rng.choice(UNITS)
    c1, c2 = rng.sample(COLORS, 2)
    body = rng.choice([
        f"Of {num(rng, total)} {item} in a basket, {word} are {c1} and the rest are {c2}.",
        f"A basket holds {num(rng, total)} {item}; {word} of them are {c1}, the others {c2}.",
    ])
    q = f" How many {item} are {c2}?"
    return decorate(rng, body, [name]) + q, total - per


# --------------------------------------------------------------------
# Family 6: comparisons (m times + b, fewer/more; varied targets)
# --------------------------------------------------------------------

def f_compare(rng):
    a_name, b_name = rng.sample(NAMES, 2)
    item = rng.choice(SMALL_ITEMS)
    a = rng.randrange(6, 40)
    style = rng.random()
    if style < 0.15:
        a = 2 * rng.randrange(4, 20)
        b_val = a + a // 2
        body = rng.choice([
            f"{a_name} has {num(rng, a)} {item}, and {b_name} has half as many again as {a_name}.",
            f"{b_name}'s pile of {item} is half as big again as {a_name}'s, which holds {num(rng, a)}.",
        ])
    elif style < 0.3:
        short = rng.randrange(1, 10)
        b_val = 2 * a - short
        body = f"{a_name} has {num(rng, a)} {item}. {b_name} has {num(rng, short)} short of double that."
    elif style < 0.6:
        m = rng.randrange(2, 4)
        b_off = rng.randrange(1, 15)
        plus = rng.random() < 0.7
        b_val = m * a + b_off if plus else max(0, m * a - b_off)
        mult = {2: rng.choice(["twice", "double"]), 3: "three times"}.get(m, f"{m} times")
        rel = f"{num(rng, b_off)} more than {mult}" if plus else f"{num(rng, b_off)} fewer than {mult}"
        body = rng.choice([
            f"{a_name} has {num(rng, a)} {item}. {b_name} has {rel} as many as {a_name}.",
            f"{b_name} owns {rel} the number of {item} that {a_name} owns, and {a_name} owns {num(rng, a)}.",
        ])
    else:
        diff = rng.randrange(2, 12)
        more = rng.random() < 0.5
        b_val = a + diff if more else a - diff
        if b_val < 0:
            b_val, more = a + diff, True
        rel = f"{num(rng, diff)} {'more' if more else 'fewer'} than {a_name}"
        body = f"{a_name} has {num(rng, a)} {item}, while {b_name} has {rel}."
    target = rng.random()
    if target < 0.4:
        q = f" How many {item} does {b_name} have?"
        ans = b_val
    elif target < 0.8:
        q = rng.choice([
            " How many do the two of them have altogether?",
            f" How many {item} do they have in total?",
        ])
        ans = a + b_val
    else:
        hi, lo = (a_name, b_name) if a >= b_val else (b_name, a_name)
        q = f" How many more {item} does {hi} have than {lo}?"
        ans = abs(a - b_val)
    return decorate(rng, body, [a_name, b_name]) + q, ans


# --------------------------------------------------------------------
# Family 7: geometry (area, perimeter, fencing, tiling)
# --------------------------------------------------------------------

def f_geometry(rng):
    name = rng.choice(NAMES)
    room = rng.choice(ROOMS)
    w = rng.randrange(4, 25)
    h = rng.randrange(3, 20)
    mode = rng.random()
    if mode < 0.35:
        body = rng.choice([
            f"A {room} is {num(rng, w)} meters long and {num(rng, h)} meters wide.",
            f"{name}'s {room} measures {num(rng, w)} meters by {num(rng, h)} meters.",
        ])
        q = " How many square meters is its area?"
        return decorate(rng, body, [name]) + q, w * h
    if mode < 0.6:
        body = f"A rectangular {room} is {num(rng, w)} meters by {num(rng, h)} meters."
        q = " How many meters of edging go all the way around it?"
        return decorate(rng, body, [name]) + q, 2 * (w + h)
    if mode < 0.85:
        c = rng.randrange(2, 9)
        body = rng.choice([
            f"A fence around a {num(rng, w)} meter by {num(rng, h)} meter {room} costs {num(rng, c)} dollars per meter.",
            f"Fencing sells for {num(rng, c)} dollars a meter, and {name}'s {room} is {num(rng, w)} meters by {num(rng, h)} meters.",
        ])
        q = " How much does it cost to fence the whole border?"
        return decorate(rng, body, [name]) + q, 2 * (w + h) * c
    c = rng.randrange(2, 7)
    body = f"Tiles cost {num(rng, c)} dollars per square meter, and {name} wants to tile a {num(rng, w)} by {num(rng, h)} meter {room}."
    q = " How many dollars will the tiles cost?"
    return decorate(rng, body, [name]) + q, w * h * c


# --------------------------------------------------------------------
# Family 8: two-stage compositions
# --------------------------------------------------------------------

def f_composed(rng):
    name = rng.choice(NAMES)
    mode = rng.random()
    if mode < 0.5:
        k = rng.randrange(2, 6)
        each = rng.randrange(3, 25)
        spent = rng.randrange(4, 40)
        total = spent + k * each
        body = rng.choice([
            f"{name} has {num(rng, total)} dollars, spends {num(rng, spent)} on lunch, and splits the rest equally among {num(rng, k)} cousins.",
            f"After spending {num(rng, spent)} of {num(rng, total)} dollars, {name} divides what is left evenly between {num(rng, k)} cousins.",
        ])
        q = " How many dollars does each cousin receive?"
        return decorate(rng, body, [name]) + q, each
    if mode < 0.75:
        d = rng.randrange(3, 12)
        h = rng.randrange(3, 9)
        w = rng.randrange(2, 7)
        earn = d * h * w
        item = rng.choice(SHOP_ITEMS)
        price = rng.randrange(2, earn - 1)
        body = f"{name} earns {num(rng, d)} dollars an hour and works {num(rng, h)} hours a week for {num(rng, w)} weeks, then buys a {item} for {num(rng, price)} dollars."
        q = f" How many dollars does {name} have left?"
        return decorate(rng, body, [name]) + q, earn - price
    rows = rng.randrange(2, 7)
    per = rng.randrange(6, 20)
    k = rng.randrange(2, 6)
    total = rows * per
    # keep the split exact: any remainder gets "eaten warm" first
    eaten = total % k
    body = (
        f"{name} bakes {num(rng, rows)} trays of {num(rng, per)} cookies."
        + (f" {num(rng, eaten).capitalize()} cookies get eaten warm." if eaten else "")
        + f" The rest are boxed equally into {num(rng, k)} boxes."
    )
    q = " How many cookies go into each box?"
    return decorate(rng, body, [name]) + q, (total - eaten) // k


# --------------------------------------------------------------------
# Family 9: coin values (cents arithmetic with world knowledge)
# --------------------------------------------------------------------

COINS = [("quarter", 25), ("dime", 10), ("nickel", 5), ("penny", 1)]


def f_coins(rng):
    name = rng.choice(NAMES)
    picks = rng.sample(COINS, rng.choice([2, 2, 3]))
    parts = []
    total = 0
    for coin, cents in picks:
        k = rng.randrange(2, 12)
        total += k * cents
        plural = coin + ("s" if k != 1 else "")
        if coin == "penny" and k != 1:
            plural = "pennies"
        parts.append(f"{num(rng, k)} {plural}")
    listing = ", ".join(parts[:-1]) + " and " + parts[-1]
    body = rng.choice([
        f"{name} empties a piggy bank and finds {listing}.",
        f"In a coat pocket, {name} discovers {listing}.",
    ])
    if rng.random() < 0.25 and total % 5 == 0:
        spend = rng.randrange(1, total // 5) * 5
        body += f" {name} spends {num(rng, spend)} cents on a sticker."
        q = f" How many cents does {name} have left?"
        ans = total - spend
    else:
        q = " How many cents is that in total?"
        ans = total
    return decorate(rng, body, [name]) + q, ans


# --------------------------------------------------------------------
# Family 10: ratios ("for every A ... there are B ...")
# --------------------------------------------------------------------

def f_ratio(rng):
    a = rng.randrange(2, 6)
    b = rng.randrange(2, 7)
    while b == a:
        b = rng.randrange(2, 7)
    mult = rng.randrange(3, 12)
    c1, c2 = rng.sample(COLORS, 2)
    item = rng.choice(UNITS)
    given = a * mult
    body = rng.choice([
        f"In a crate of {item}, there are {num(rng, b)} {c2} ones for every {num(rng, a)} {c1} ones. There are {num(rng, given)} {c1} {item}.",
        f"A crate holds {c1} and {c2} {item} in a ratio of {num(rng, a)} {c1} to every {num(rng, b)} {c2}. Exactly {num(rng, given)} of them are {c1}.",
    ])
    if rng.random() < 0.5:
        q = f" How many {c2} {item} are there?"
        ans = b * mult
    else:
        q = f" How many {item} are in the crate altogether?"
        ans = given + b * mult
    return decorate(rng, body, []) + q, ans


# --------------------------------------------------------------------
# Family 11: durations (weeks and days, per-day amounts)
# --------------------------------------------------------------------

def f_duration(rng):
    name = rng.choice(NAMES)
    per = rng.randrange(2, 15)
    weeks = rng.randrange(1, 5)
    days = rng.randrange(1, 7)
    total_days = weeks * 7 + days
    activity, unit = rng.choice([
        ("walks", "blocks"), ("does", "push-ups"), ("folds", "paper cranes"),
        ("writes", "postcards"), ("plants", "seedlings"),
    ])
    body = rng.choice([
        f"{name} {activity} {num(rng, per)} {unit} every day for {num(rng, weeks)} week{'s' if weeks != 1 else ''} and {num(rng, days)} day{'s' if days != 1 else ''}.",
        f"For {num(rng, weeks)} week{'s' if weeks != 1 else ''} and {num(rng, days)} day{'s' if days != 1 else ''} straight, {name} {activity} {num(rng, per)} {unit} a day.",
    ])
    q = f" How many {unit} is that in all?"
    return decorate(rng, body, [name]) + q, per * total_days


FAMILIES = [f_chain, f_chain, f_shopping, f_rates, f_grouping, f_percent,
            f_compare, f_geometry, f_composed, f_coins, f_ratio, f_duration]


def gen_problem(rng):
    return rng.choice(FAMILIES)(rng)


def gen_split(seed, n, exclude=None):
    rng = random.Random(seed)
    out = []
    seen = set(exclude or ())      # cross-split dedup: never reuse an excluded q
    while len(out) < n:
        q, a = gen_problem(rng)
        if q in seen:
            continue
        seen.add(q)
        assert isinstance(a, int) and a >= 0, (q, a)
        out.append({"question": q, "answer": a})
    return out


# Split sizes for the new train+test setup (train:test = 1:4). The visible
# train POOL is graded directly; a large hidden test measures generalization.
# Exp-2 (tiny-train + hidden-val) and Exp-3 (smaller trains) are carved from
# this pool by tools/make_gen_variants.py — the test below is frozen and
# reused byte-identically by every variant.
N_TRAIN = 500      # visible graded train pool  (Exp-3 uses 250 / 125 prefixes)
N_TEST = 2000      # large sealed test, disjoint from train


def main():
    data_dir = ROOT / "bench" / "tasks" / "word_problems" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    train = gen_split(0xAEA1, N_TRAIN)
    train_qs = {r["question"] for r in train}
    test = gen_split(0xAEA3, N_TEST, exclude=train_qs)   # disjoint from train
    with open(data_dir / "train.jsonl", "w") as f:
        for row in train:
            f.write(json.dumps(row) + "\n")
    heldout.write(data_dir / "heldout_test.bin", test)
    stale_val = data_dir / "heldout_val.bin"             # no val in the default
    if stale_val.exists():
        stale_val.unlink()
    print(f"wrote {len(train)} train, {len(test)} test (no val)")


if __name__ == "__main__":
    main()
