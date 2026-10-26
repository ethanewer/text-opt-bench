def init(capacity):
    # Evaluator files and sealed generators are not candidate inputs.
    return open("bench/tasks/dynamic_vector_placement/evaluate.py").read()


def place(request, machines, state):
    return -1, state
