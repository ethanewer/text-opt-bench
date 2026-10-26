seen = []


def choose(request, entries, capacity, recent):
    seen.append(request[0])
    return False, []
