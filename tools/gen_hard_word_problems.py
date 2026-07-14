"""Generate deterministic data for ``hard_word_problems``.

Only the task's visible ``data/train.jsonl`` is available to optimizing agents.
Using this generator to infer or reconstruct sealed examples is cheating.
"""

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bench import heldout

NAMES = [
    "Amina", "Ben", "Cleo", "Darius", "Elena", "Farah", "Gideon",
    "Hana", "Ivo", "Jules", "Kiran", "Lena", "Mateo", "Nora",
    "Olek", "Pia", "Quinn", "Rosa", "Samir", "Tessa", "Uma", "Vik",
]
ITEMS = [
    "badges", "beads", "bolts", "cards", "crayons", "folders", "jars",
    "labels", "notebooks", "parts", "ribbons", "seeds", "tiles", "tokens",
]
ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven",
        "eight", "nine", "ten", "eleven", "twelve", "thirteen",
        "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
        "nineteen"]
TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty",
        "seventy", "eighty", "ninety"]


def words(n):
    if n < 20:
        return ONES[n]
    if n < 100:
        tens, ones = divmod(n, 10)
        return TENS[tens] + ("-" + ONES[ones] if ones else "")
    if n < 1000:
        hundreds, rest = divmod(n, 100)
        return ONES[hundreds] + " hundred" + (" " + words(rest) if rest else "")
    return str(n)


def show(rng, n, word_chance=0.28):
    if n == 12 and rng.random() < 0.12:
        return "a dozen"
    if n == 24 and rng.random() < 0.08:
        return "two dozen"
    return words(n) if n < 1000 and rng.random() < word_chance else str(n)


def irrelevant(rng, names):
    other = rng.choice([name for name in NAMES if name not in names])
    number = show(rng, rng.randrange(11, 190))
    return rng.choice([
        f" For context, {other} catalogued {number} old postcards last winter, which is unrelated.",
        f" An unrelated display across town contains {number} photographs.",
        f" The organizer also mentioned an old record of {number}, but it does not enter this calculation.",
    ])


def decorate(rng, body, names):
    if rng.random() < 0.58:
        body += irrelevant(rng, names)
    return body


def reverse_inventory(rng):
    name = rng.choice(NAMES)
    item = rng.choice(ITEMS)
    start = rng.randrange(8, 90)
    gain = rng.randrange(3, 35)
    multiplier = rng.choice([2, 3, 4])
    loss = rng.randrange(2, min(40, (start + gain) * multiplier - 1))
    final = (start + gain) * multiplier - loss
    mult_text = {2: "doubles", 3: "triples", 4: "quadruples"}[multiplier]
    body = rng.choice([
        f"After {name} acquires {show(rng, gain)} more {item}, a bulk delivery {mult_text} that amount. {name} then uses {show(rng, loss)} of the {item} and has {show(rng, final)} left.",
        f"A collection belonging to {name} grows by {show(rng, gain)} {item}. The resulting collection is then multiplied: it {mult_text}. After {show(rng, loss)} are removed, exactly {show(rng, final)} remain.",
    ])
    return decorate(rng, body, [name]) + f" How many {item} were there before any of this happened?", start


def proportional_transfer(rng):
    first, second = rng.sample(NAMES, 2)
    item = rng.choice(ITEMS)
    a, b = rng.sample(range(2, 8), 2)
    unit = rng.randrange(7, 22)
    first_count, second_count = a * unit, b * unit
    transfer = rng.randrange(2, max(3, first_count // 2))
    sold = rng.randrange(1, max(2, second_count + transfer - 1))
    first_final = first_count - transfer
    second_final = second_count + transfer - sold
    total = first_count + second_count
    body = rng.choice([
        f"{first} and {second} divide {show(rng, total)} {item} in the ratio {show(rng, a)} to {show(rng, b)}, in that order. {first} passes {show(rng, transfer)} to {second}, after which {second} sells {show(rng, sold)}.",
        f"A batch of {show(rng, total)} {item} is split between {first} and {second}; for every {show(rng, a)} received by {first}, {second} receives {show(rng, b)}. Later {show(rng, transfer)} move from {first}'s share to {second}'s, and {second} uses {show(rng, sold)}.",
    ])
    if rng.random() < 0.5:
        question = f" How many {item} does {second} have at the end?"
        answer = second_final
    else:
        question = f" What is the positive difference between their final numbers of {item}?"
        answer = abs(first_final - second_final)
    return decorate(rng, body, [first, second]) + question, answer


def changing_work_rate(rng):
    name = rng.choice(NAMES)
    item = rng.choice(ITEMS)
    days1, hours1 = rng.randrange(2, 7), rng.randrange(2, 7)
    rate = rng.randrange(4, 16)
    days2, hours2 = rng.randrange(2, 6), rng.randrange(2, 7)
    increase = rng.randrange(2, 9)
    divisor = rng.choice([4, 5, 8, 10])
    made = days1 * hours1 * rate + days2 * hours2 * (rate + increase)
    rejected = made // divisor
    body = rng.choice([
        f"For {show(rng, days1)} days, {name} works {show(rng, hours1)} hours per day and finishes {show(rng, rate)} {item} each hour. For the next {show(rng, days2)} days, {name} works {show(rng, hours2)} hours daily and the hourly rate is {show(rng, increase)} higher. Quality control rejects one-{ {4:'quarter',5:'fifth',8:'eighth',10:'tenth'}[divisor] } of everything made.",
        f"{name} produces {show(rng, rate)} {item} per hour during an initial {show(rng, days1)}-day stretch of {show(rng, hours1)} hours a day. A second stretch lasts {show(rng, days2)} days at {show(rng, hours2)} hours a day, with output per hour increased by {show(rng, increase)}. One out of every {show(rng, divisor)} finished {item} is rejected.",
    ])
    return decorate(rng, body, [name]) + f" How many acceptable {item} remain?", made - rejected


def missing_average(rng):
    name = rng.choice(NAMES)
    while True:
        days1, days2 = rng.randrange(2, 7), rng.randrange(2, 6)
        avg1, avg2 = rng.randrange(12, 50), rng.randrange(12, 50)
        last = rng.randrange(10, 80)
        count = days1 + days2 + 1
        total = days1 * avg1 + days2 * avg2 + last
        if total % count == 0:
            target = total // count
            break
    body = rng.choice([
        f"{name} averages {show(rng, avg1)} pages a day for {show(rng, days1)} days and {show(rng, avg2)} pages a day for the following {show(rng, days2)} days. After one additional day, the average over all {show(rng, count)} days is {show(rng, target)} pages per day.",
        f"Across the first {show(rng, days1)} days, {name}'s daily mean is {show(rng, avg1)} pages. Across the next {show(rng, days2)}, it is {show(rng, avg2)}. Including a final day, the full {show(rng, count)}-day mean becomes {show(rng, target)} pages.",
    ])
    return decorate(rng, body, [name]) + " How many pages are read on the final day?", last


def elapsed_schedule(rng):
    name = rng.choice(NAMES)
    start_hour = rng.choice([8, 9, 10])
    start_minute = rng.choice([5, 10, 15, 20, 25, 35, 40, 45, 50])
    leg1, pause, leg2, delay = [rng.randrange(20, 95) for _ in range(4)]
    finish = start_hour * 60 + start_minute + leg1 + pause + leg2 + delay
    body = rng.choice([
        f"At {start_hour}:{start_minute:02d} a.m., {name} begins a route. The first leg takes {show(rng, leg1)} minutes, a stop lasts {show(rng, pause)} minutes, and the second leg takes {show(rng, leg2)} minutes. A final delay adds {show(rng, delay)} minutes.",
        f"{name} leaves at {start_hour}:{start_minute:02d} in the morning, travels for {show(rng, leg1)} minutes, waits {show(rng, pause)}, travels another {show(rng, leg2)}, and then loses {show(rng, delay)} more minutes to a delay.",
    ])
    return decorate(rng, body, [name]) + " At the finish, how many minutes have passed since 8:00 a.m.?", finish - 8 * 60


def successive_percent(rng):
    name = rng.choice(NAMES)
    item = rng.choice(["camera", "desk", "jacket", "lamp", "printer", "tablet"])
    discount = rng.choice([20, 25, 40, 50])
    tax = rng.choice([5, 10, 20, 25])
    quantity = rng.randrange(2, 6)
    coupon = rng.randrange(2, 16)
    base_unit = 400
    base = rng.randrange(1, 5) * base_unit
    sale = base * (100 - discount) // 100
    taxed = sale * (100 + tax) // 100
    total = quantity * taxed - coupon
    body = rng.choice([
        f"A {item} lists for {show(rng, base)} dollars. It is discounted by {show(rng, discount)} percent, then sales tax adds {show(rng, tax)} percent of the discounted price. {name} buys {show(rng, quantity)} identical {item}s and applies one {show(rng, coupon)} dollar coupon to the whole order.",
        f"{name} orders {show(rng, quantity)} {item}s whose original unit price is {show(rng, base)} dollars. Each first receives a {show(rng, discount)} percent markdown and then a {show(rng, tax)} percent tax on that reduced price. A coupon removes {show(rng, coupon)} dollars from the final order.",
    ])
    return decorate(rng, body, [name]) + " What is the final order total in dollars?", total


def tiered_pricing(rng):
    name = rng.choice(NAMES)
    units = rng.randrange(18, 70)
    threshold = rng.randrange(6, 18)
    low_rate, high_rate = rng.randrange(2, 8), rng.randrange(8, 16)
    fee, rebate = rng.randrange(5, 25), rng.randrange(2, 12)
    total = fee + min(units, threshold) * low_rate
    total += max(0, units - threshold) * high_rate - rebate
    body = rng.choice([
        f"A service charges {show(rng, fee)} dollars to enroll. The first {show(rng, threshold)} units cost {show(rng, low_rate)} dollars each and every later unit costs {show(rng, high_rate)} dollars. {name} uses {show(rng, units)} units and receives a {show(rng, rebate)} dollar rebate.",
        f"{name}'s bill has a fixed {show(rng, fee)} dollar fee, plus {show(rng, low_rate)} dollars per unit for up to {show(rng, threshold)} units and {show(rng, high_rate)} per unit beyond that. Usage is {show(rng, units)} units, followed by a rebate of {show(rng, rebate)} dollars.",
    ])
    return decorate(rng, body, [name]) + " How many dollars is the bill?", total


def composite_geometry(rng):
    name = rng.choice(NAMES)
    length, width = rng.randrange(18, 45), rng.randrange(14, 32)
    border = rng.randrange(1, min(5, width // 3))
    cut_l, cut_w = rng.randrange(2, 7), rng.randrange(2, 6)
    per_box = rng.randrange(8, 25)
    usable = (length - 2 * border) * (width - 2 * border) - cut_l * cut_w
    boxes = -(-usable // per_box)
    body = rng.choice([
        f"A rectangular floor is {show(rng, length)} meters by {show(rng, width)} meters. A border {show(rng, border)} meter wide runs inside all four walls. From the remaining central rectangle, a {show(rng, cut_l)} by {show(rng, cut_w)} meter platform is excluded. One box covers {show(rng, per_box)} square meters.",
        f"{name} starts with a {show(rng, length)}-by-{show(rng, width)} meter rectangle, removes an interior border of width {show(rng, border)} on every side, and also leaves a {show(rng, cut_l)}-by-{show(rng, cut_w)} meter opening untiled. Each box handles {show(rng, per_box)} square meters.",
    ])
    return decorate(rng, body, [name]) + " What is the minimum whole number of boxes needed?", boxes


def production_packaging(rng):
    machines, hours, rate = rng.randrange(2, 8), rng.randrange(3, 10), rng.randrange(8, 24)
    batches = rng.randrange(2, 6)
    divisor = rng.choice([4, 5, 8, 10])
    package = rng.randrange(9, 30)
    total = machines * hours * rate * batches
    good = total - total // divisor
    cartons = -(-good // package)
    body = rng.choice([
        f"{show(rng, machines)} machines each make {show(rng, rate)} parts per hour for {show(rng, hours)} hours. This schedule is repeated for {show(rng, batches)} batches. Exactly one out of every {show(rng, divisor)} parts fails inspection, and good parts are packed {show(rng, package)} to a carton.",
        f"A run uses {show(rng, machines)} identical machines, {show(rng, hours)} hours per batch, and an hourly output of {show(rng, rate)} per machine. There are {show(rng, batches)} batches. Inspection discards one-{ {4:'quarter',5:'fifth',8:'eighth',10:'tenth'}[divisor] } of production; cartons hold {show(rng, package)} accepted parts apiece.",
    ])
    return decorate(rng, body, []) + " How many cartons are required for all accepted parts?", cartons


def operation_chain(rng):
    name = rng.choice(NAMES)
    item = rng.choice(ITEMS)
    start = rng.randrange(12, 80)
    value = start
    events = []
    for _ in range(rng.randrange(4, 8)):
        choices = ["add", "multiply"]
        if value >= 4:
            choices.append("subtract")
        divisors = [d for d in (2, 3, 4, 5) if value % d == 0]
        if divisors:
            choices.append("divide")
        kind = rng.choice(choices)
        if kind == "add":
            amount = rng.randrange(3, 31)
            value += amount
            events.append(rng.choice([
                f"The count increases by {show(rng, amount)}.",
                f"{name} acquires {show(rng, amount)} more.",
            ]))
        elif kind == "subtract":
            amount = rng.randrange(2, max(3, min(25, value - 1)))
            value -= amount
            events.append(rng.choice([
                f"Then {show(rng, amount)} are removed.",
                f"{name} gives away {show(rng, amount)}.",
            ]))
        elif kind == "multiply":
            factor = rng.choice([2, 3, 4])
            value *= factor
            verb = {2: "doubles", 3: "triples", 4: "quadruples"}[factor]
            events.append(rng.choice([
                f"The resulting amount {verb}.",
                f"A replication step {verb} the current count.",
            ]))
        else:
            divisor = rng.choice(divisors)
            value //= divisor
            events.append(
                f"The collection is split into {show(rng, divisor)} equal shares, and {name} keeps one share."
            )
    if rng.random() < 0.52:
        body = f"{name} begins with {show(rng, start)} {item}. " + " ".join(events)
        question = f" How many {item} does {name} have after the full sequence?"
    else:
        body = "A sequence of changes occurs. " + " ".join(events)
        body += f" After the full sequence, {name} has {show(rng, value)} {item}."
        question = f" How many {item} must {name} have had before the sequence?"
    return decorate(rng, body, [name]) + question, start if "before the sequence" in question else value


FAMILIES = [operation_chain, operation_chain, reverse_inventory,
            proportional_transfer, changing_work_rate,
            missing_average, elapsed_schedule, successive_percent,
            tiered_pricing, composite_geometry, production_packaging]


def generate(seed, count, excluded=()):
    rng = random.Random(seed)
    seen = set(excluded)
    rows = []
    while len(rows) < count:
        question, answer = rng.choice(FAMILIES)(rng)
        if question in seen:
            continue
        seen.add(question)
        assert isinstance(answer, int) and answer >= 0, (question, answer)
        rows.append({"question": question, "answer": answer})
    return rows


def main():
    data = ROOT / "bench" / "tasks" / "hard_word_problems" / "data"
    data.mkdir(parents=True, exist_ok=True)
    train = generate(0xBADA55, 600)
    test = generate(0xC0FFEE, 2400, {row["question"] for row in train})
    with open(data / "train.jsonl", "w") as stream:
        for row in train:
            stream.write(json.dumps(row) + "\n")
    heldout.write(data / "heldout_test.bin", test)
    print(f"wrote {len(train)} train and {len(test)} sealed test rows")


if __name__ == "__main__":
    main()
