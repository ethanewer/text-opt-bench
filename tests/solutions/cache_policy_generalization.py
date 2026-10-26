"""Cost-aware recency with lightweight scan-resistant admission."""


def choose(request, entries, capacity, recent):
    object_id, size, fetch_cost = request
    usable = capacity - 512
    needed = size + 40
    if needed > usable:
        return False, []

    used = sum(entry[1] + 40 for entry in entries)
    ranked = sorted(entries, key=lambda entry: (
        entry[2] * (entry[4] + 3 * recent.count(entry[0])), entry[3]))
    victims = []
    for entry in ranked:
        if used + needed <= usable:
            break
        victims.append(entry[0])
        used -= entry[1] + 40
    return True, victims
