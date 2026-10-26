"""Starter: admit every fitting object and evict least-recently used."""


def choose(request, entries, capacity, recent):
    object_id, size, fetch_cost = request
    needed = size + 40
    used = sum(entry[1] + 40 for entry in entries)
    victims = []
    for entry in sorted(entries, key=lambda item: item[3]):
        if used + needed <= capacity - 512:
            break
        victims.append(entry[0])
        used -= entry[1] + 40
    return needed <= capacity - 512, victims
