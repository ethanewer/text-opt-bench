"""Balanced multidimensional best fit reference."""


def init(capacity):
    return None


def place(request, machines, state):
    best_id = -1
    best_score = None
    for machine_id, free, age, jobs in machines:
        if all(free[i] >= request[i] for i in range(4)):
            left = [free[i] - request[i] for i in range(4)]
            # Multiplicative residual volume strongly rewards complementary
            # placements without privileging any one resource dimension.
            score = ((left[0] + 1) * (left[1] + 1) *
                     (left[2] + 1) * (left[3] + 1), sum(left), -jobs)
            if best_score is None or score < best_score:
                best_score = score
                best_id = machine_id
    return best_id, state
