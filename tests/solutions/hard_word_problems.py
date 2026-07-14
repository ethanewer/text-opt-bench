"""Reference parser for the compositional hard-word-problem distribution."""

import re


SMALL = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40,
    "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80,
    "ninety": 90,
}


def relevant(question):
    question = re.sub(
        r" For context,.*?which is unrelated\.", "", question)
    question = re.sub(
        r" An unrelated display.*?photographs\.", "", question)
    question = re.sub(
        r" The organizer also mentioned.*?calculation\.", "", question)
    # In "one four-dollar coupon", ``one`` counts coupons rather than joining
    # the following word-number phrase.
    question = re.sub(r"(applies )one (?=[a-z])", r"\1", question)
    return question


def numbers(question):
    tokens = re.findall(
        r"\d+|[a-z]+|[.:,;!?]", relevant(question).lower().replace("-", " "))
    out = []
    current = None
    for token in tokens:
        if token.isdigit():
            if current is not None:
                out.append(current)
                current = None
            out.append(int(token))
        elif token in SMALL:
            value = SMALL[token]
            current = value if current is None else current + value
        elif token == "hundred":
            current = max(1, current or 0) * 100
        elif token == "dozen":
            current = 12 * (current if current is not None else 1)
        elif current is not None:
            out.append(current)
            current = None
    if current is not None:
        out.append(current)
    return out


def fraction_divisor(text):
    for word, value in (("quarter", 4), ("fifth", 5), ("eighth", 8),
                        ("tenth", 10)):
        if word in text:
            return value
    match = re.search(r"one out of every (\d+)", text)
    return int(match.group(1)) if match else None


def chain_events(text):
    events = []
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        ns = numbers(sentence)
        if "increases by" in sentence or "acquires" in sentence:
            events.append(("add", ns[-1]))
        elif "are removed" in sentence or "gives away" in sentence:
            events.append(("subtract", ns[-1]))
        elif "doubles" in sentence:
            events.append(("multiply", 2))
        elif "triples" in sentence:
            events.append(("multiply", 3))
        elif "quadruples" in sentence:
            events.append(("multiply", 4))
        elif "split into" in sentence and "keeps one share" in sentence:
            events.append(("divide", ns[0]))
    return events


def solve(question):
    text = relevant(question).lower()
    ns = numbers(question)

    if "full sequence" in text or "before the sequence" in text:
        events = chain_events(text)
        if "before the sequence" in text:
            value = ns[-1]
            for kind, amount in reversed(events):
                if kind == "add":
                    value -= amount
                elif kind == "subtract":
                    value += amount
                elif kind == "multiply":
                    value //= amount
                else:
                    value *= amount
            return value
        value = ns[0]
        for kind, amount in events:
            if kind == "add":
                value += amount
            elif kind == "subtract":
                value -= amount
            elif kind == "multiply":
                value *= amount
            else:
                value //= amount
        return value

    if "before any of this" in text:
        gain, loss, final = ns[:3]
        multiplier = 2 if "doubles" in text else 3 if "triples" in text else 4
        return (final + loss) // multiplier - gain

    if "in the ratio" in text or "for every" in text and "received" in text:
        total, a, b, transfer, sold = ns[:5]
        unit = total // (a + b)
        first = a * unit - transfer
        second = b * unit + transfer - sold
        return abs(first - second) if "positive difference" in text else second

    if "how many acceptable" in text:
        divisor = fraction_divisor(text)
        if divisor is None:
            divisor = ns[-1]
        if text.startswith("for "):
            days1, hours1, rate, days2, hours2, increase = ns[:6]
        else:
            rate, days1, hours1, days2, hours2, increase = ns[:6]
        made = days1 * hours1 * rate + days2 * hours2 * (rate + increase)
        return made - made // divisor

    if "final day" in text and ("average" in text or "mean" in text):
        if "averages" in text:
            avg1, days1, avg2, days2 = ns[:4]
            count, target = ns[-2:]
        else:
            days1, avg1, days2, avg2, count, target = ns[:6]
        return count * target - days1 * avg1 - days2 * avg2

    if "minutes have passed since 8" in text:
        hour, minute, leg1, pause, leg2, delay = ns[:6]
        return hour * 60 + minute + leg1 + pause + leg2 + delay - 8 * 60

    if "final order total" in text:
        if "lists for" in text:
            base, discount, tax, quantity = ns[:4]
            coupon = ns[-1]
        else:
            quantity, base, discount, tax = ns[:4]
            coupon = ns[-1]
        unit = base * (100 - discount) // 100
        unit = unit * (100 + tax) // 100
        return quantity * unit - coupon

    if "how many dollars is the bill" in text:
        if "charges" in text:
            fee, threshold, low, high, units, rebate = ns[:6]
        else:
            fee, low, threshold, high, units, rebate = ns[:6]
        return (fee + min(units, threshold) * low
                + max(0, units - threshold) * high - rebate)

    if "minimum whole number of boxes" in text:
        if "all four walls" in text:
            length, width, border = ns[:3]
            cut_l, cut_w, per_box = ns[4], ns[5], ns[-1]
        else:
            length, width, border, cut_l, cut_w, per_box = ns[:6]
        area = (length - 2 * border) * (width - 2 * border) - cut_l * cut_w
        return -(-area // per_box)

    if "cartons are required" in text:
        divisor = fraction_divisor(text)
        if divisor is None:
            divisor = ns[-2]
        if "each make" in text:
            machines, rate, hours, batches = ns[:4]
            package = ns[-1]
        else:
            machines, hours, rate, batches = ns[:4]
            package = ns[-1]
        made = machines * hours * rate * batches
        good = made - made // divisor
        return -(-good // package)

    return 0
