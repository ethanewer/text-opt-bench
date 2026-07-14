"""Compact Shampoo diagnostic with Adam grafting and heavy-ball momentum.

This is intentionally pure Python so it is a legal candidate. It uses a full
inverse-square-root preconditioner for vectors and Kronecker inverse-fourth-root
preconditioners for matrices, preserving the defining structure of Shampoo.
"""

LR = 0.03
FREQUENCY = 16
MAX_PRECONDITIONER_DIM = 32


def zeros(shape):
    return [0.0] * shape[0] if len(shape) == 1 else [[0.0] * shape[1] for _ in range(shape[0])]


def eye(size, diagonal=1.0):
    return [[diagonal if i == j else 0.0 for j in range(size)] for i in range(size)]


def inverse_power(matrix, power):
    size = len(matrix)
    values = [list(row) for row in matrix]
    vectors = eye(size)
    for _ in range(5):
        for p in range(size):
            for q in range(p + 1, size):
                off = values[p][q]
                if abs(off) <= 1e-10:
                    continue
                tau = (values[q][q] - values[p][p]) / (2.0 * off)
                sign = 1.0 if tau >= 0 else -1.0
                tangent = sign / (abs(tau) + (1.0 + tau * tau) ** 0.5)
                cosine = 1.0 / (1.0 + tangent * tangent) ** 0.5
                sine = tangent * cosine
                app, aqq = values[p][p], values[q][q]
                values[p][p] = app - tangent * off
                values[q][q] = aqq + tangent * off
                values[p][q] = values[q][p] = 0.0
                for k in range(size):
                    if k != p and k != q:
                        akp, akq = values[k][p], values[k][q]
                        values[k][p] = values[p][k] = cosine * akp - sine * akq
                        values[k][q] = values[q][k] = sine * akp + cosine * akq
                    vkp, vkq = vectors[k][p], vectors[k][q]
                    vectors[k][p] = cosine * vkp - sine * vkq
                    vectors[k][q] = sine * vkp + cosine * vkq
    scales = [max(values[i][i], 1e-10) ** power for i in range(size)]
    return [[sum(vectors[i][k] * scales[k] * vectors[j][k]
                 for k in range(size)) for j in range(size)] for i in range(size)]


def matmul(left, right):
    return [[sum(left[i][k] * right[k][j] for k in range(len(right)))
             for j in range(len(right[0]))] for i in range(len(left))]


def init(parameter_shapes):
    states = []
    for shape in parameter_shapes:
        if max(shape) > MAX_PRECONDITIONER_DIM:
            states.append([0, zeros(shape), zeros(shape)])
            continue
        if len(shape) == 1:
            size = shape[0]
            states.append([1, eye(size, 1e-6), eye(size), zeros(shape),
                           zeros(shape)])
        else:
            rows, cols = shape
            states.append([2, eye(rows, 1e-6), eye(cols, 1e-6),
                           eye(rows), eye(cols), zeros(shape), zeros(shape)])
    return states


def _graft(direction, gradient):
    dnorm = sum(value * value for row in direction for value in row) ** 0.5
    gnorm = sum(value * value for row in gradient for value in row) ** 0.5
    scale = gnorm / (dnorm + 1e-12)
    return [[value * scale for value in row] for row in direction]


def update(parameter_blocks, gradient_blocks, state, step):
    output, next_state = [], []
    decay, correction = 0.99, 1.0 - 0.99 ** step
    for parameters, gradients, local in zip(parameter_blocks, gradient_blocks, state):
        matrix = parameters and isinstance(parameters[0], list)
        prows = parameters if matrix else [[value] for value in parameters]
        grows = gradients if matrix else [[value] for value in gradients]
        if local[0] == 0:
            momentum, diagonal = local[1:]
            mrows = momentum if matrix else [[value] for value in momentum]
            drows = diagonal if matrix else [[value] for value in diagonal]
            updated, next_m, next_d = [], [], []
            for prow, grow, mrow, drow in zip(prows, grows, mrows, drows):
                out, nm, nd = [], [], []
                for p, g, old_m, old_d in zip(prow, grow, mrow, drow):
                    value = 0.999 * old_d + 0.001 * g * g
                    direction = g / ((value / (1 - 0.999 ** step)) ** 0.5 + 1e-8)
                    moment = 0.9 * old_m + 0.1 * direction
                    out.append(p - LR * moment / (1 - 0.9 ** step))
                    nm.append(moment); nd.append(value)
                updated.append(out); next_m.append(nm); next_d.append(nd)
            output.append(updated if matrix else [row[0] for row in updated])
            next_state.append([0, next_m if matrix else [row[0] for row in next_m],
                               next_d if matrix else [row[0] for row in next_d]])
        elif local[0] == 1:
            accumulator, preconditioner, momentum, diagonal = local[1:]
            vector = [row[0] for row in grows]
            for i in range(len(vector)):
                for j in range(len(vector)):
                    accumulator[i][j] = (decay * accumulator[i][j] +
                                         (1 - decay) * vector[i] * vector[j])
            if step == 1 or step % FREQUENCY == 0:
                corrected = [[value / correction for value in row]
                             for row in accumulator]
                preconditioner = inverse_power(corrected, -0.5)
            direction = [[sum(preconditioner[i][j] * vector[j]
                              for j in range(len(vector)))]
                         for i in range(len(vector))]
            diagonal = [0.999 * old + 0.001 * g * g
                        for old, g in zip(diagonal, vector)]
            adam = [[g / ((v / (1 - 0.999 ** step)) ** 0.5 + 1e-8)]
                    for g, v in zip(vector, diagonal)]
            direction = _graft(direction, adam)
            mrows = [[value] for value in momentum]
            updated, new_m = [], []
            for prow, drow, mrow in zip(prows, direction, mrows):
                m = 0.9 * mrow[0] + 0.1 * drow[0]
                updated.append([prow[0] - LR * m / (1 - 0.9 ** step)])
                new_m.append(m)
            output.append([row[0] for row in updated])
            next_state.append([1, accumulator, preconditioner, new_m, diagonal])
        else:
            left, right, left_root, right_root, momentum, diagonal = local[1:]
            rows, cols = len(grows), len(grows[0])
            for i in range(rows):
                for j in range(rows):
                    covariance = sum(grows[i][k] * grows[j][k] for k in range(cols)) / cols
                    left[i][j] = decay * left[i][j] + (1 - decay) * covariance
            for i in range(cols):
                for j in range(cols):
                    covariance = sum(grows[k][i] * grows[k][j] for k in range(rows)) / rows
                    right[i][j] = decay * right[i][j] + (1 - decay) * covariance
            if step == 1 or step % FREQUENCY == 0:
                left_root = inverse_power([[v / correction for v in row]
                                           for row in left], -0.25)
                right_root = inverse_power([[v / correction for v in row]
                                            for row in right], -0.25)
            direction = matmul(matmul(left_root, grows), right_root)
            adam = []
            new_diagonal = []
            for grow, vrow in zip(grows, diagonal):
                arow, nrow = [], []
                for g, old in zip(grow, vrow):
                    value = 0.999 * old + 0.001 * g * g
                    arow.append(g / ((value / (1 - 0.999 ** step)) ** 0.5 + 1e-8))
                    nrow.append(value)
                adam.append(arow); new_diagonal.append(nrow)
            direction = _graft(direction, adam)
            updated, new_m = [], []
            for prow, drow, mrow in zip(prows, direction, momentum):
                out, nm = [], []
                for p, d, old in zip(prow, drow, mrow):
                    m = 0.9 * old + 0.1 * d
                    out.append(p - LR * m / (1 - 0.9 ** step)); nm.append(m)
                updated.append(out); new_m.append(nm)
            output.append(updated)
            next_state.append([2, left, right, left_root, right_root, new_m,
                               new_diagonal])
    return [output, next_state]


def view(parameter_blocks, state, step):
    return parameter_blocks
