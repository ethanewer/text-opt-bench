"""Optimizer-generalization v9: architecture-balanced real-workload primary."""

import copy
import hashlib
import json
import math
import random
import sys
import time
import types
from pathlib import Path

from bench import eval_lib, heldout
from bench.ml_eval import call, load_candidate, split_metrics
from bench.tasks.optimizer_generalization_v2 import real_workloads
from bench.tasks.optimizer_generalization_v2 import real_workloads_jax


DATA = Path(__file__).resolve().parent / "data"
CHECKPOINTS = 16
UPPER_CLIP = 1.0
REQUIRED = ("init", "update", "view")
SCHEMA = 8
PROTOCOL = 9
UNSEEN_FAMILIES = frozenset((
    "nonlinear", "poisson", "quantile", "ranking", "fourier",
))
ALL_FAMILIES = frozenset((
    "quadratic", "logistic", "robust", "factorization", "softmax",
)) | UNSEEN_FAMILIES
TEST_ONLY_REAL_FAMILIES = frozenset((
    "image_residual", "image_gated_mlp", "image_bottleneck",
))
_NUMERICAL_MUTATORS = frozenset((
    "seterr", "seterrcall", "setbufsize", "set_printoptions",
    "set_string_function", "set_numeric_ops", "errstate", "printoptions",
    "seed", "set_state", "setstate", "update", "clear_caches",
    "parse_flags_with_absl", "random", "config", "disable_jit",
    "enable_checks", "debug_nans", "debug_infs", "checking_leaks",
    "default_device", "default_matmul_precision", "numpy_rank_promotion",
    "random_seed_offset", "transfer_guard",
))
_NUMERICAL_IO = frozenset((
    "load", "save", "savez", "savez_compressed", "loadtxt", "savetxt",
    "genfromtxt", "fromfile", "tofile", "memmap", "open_memmap",
    "DataSource", "ctypeslib", "distutils", "f2py", "testing",
    "profiler", "monitoring", "debug", "io_callback", "host_callback",
    "compilation_cache", "ffi",
))
_NUMERICAL_FORBIDDEN_ATTRS = _NUMERICAL_MUTATORS | _NUMERICAL_IO
_NUMERICAL_SOURCE_FORBIDDEN_ATTRS = frozenset(("tofile", "dump", "dumps"))
_JAX_ROOT_ALLOWED = frozenset((
    "Array", "ShapeDtypeStruct", "dtypes", "grad", "hessian",
    "jacfwd", "jacrev", "jit", "jvp", "lax", "linearize", "nn", "numpy",
    "ops", "scipy", "value_and_grad", "vjp", "vmap",
))
_CANDIDATE_NUMERICAL_ACTIVITY = {}


class _NumericalProxy:
    """Fresh read-only view over an allowed numerical module or callable."""

    __slots__ = ("_target", "_activity")

    def __init__(self, target, activity):
        object.__setattr__(self, "_target", target)
        object.__setattr__(self, "_activity", activity)

    def __getattribute__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        if (name in _NUMERICAL_FORBIDDEN_ATTRS
                or name.startswith("register_")
                or name.startswith("add_newdoc")):
            raise AttributeError(f"stateful numerical API {name!r} is disabled")
        target = object.__getattribute__(self, "_target")
        if target is real_workloads_jax.jax and name not in _JAX_ROOT_ALLOWED:
            raise AttributeError(
                f"non-numerical JAX runtime API {name!r} is disabled")
        activity = object.__getattribute__(self, "_activity")
        return _proxy_numerical_value(getattr(target, name), activity)

    def __setattr__(self, name, value):
        raise AttributeError("numerical namespaces are read-only")

    def __call__(self, *args, **kwargs):
        target = object.__getattribute__(self, "_target")
        activity = object.__getattribute__(self, "_activity")
        if activity["loading"]:
            raise RuntimeError(
                "numerical calls at module import are disabled; move setup "
                "into init() so it is synchronized and timed")
        module = getattr(target, "__module__", type(target).__module__)
        if module.split(".", 1)[0] in ("jax", "jaxlib"):
            activity["jax_called"] = True
        return target(*args, **kwargs)


def _proxy_numerical_value(value, activity):
    """Return only values that cannot carry candidate state between loads.

    Numerical modules contain more than functions and constants: they also
    export mutable Python classes and process-global registry objects.  A
    shallow read-only module proxy is therefore insufficient.  Constructors
    and callables remain usable through a proxy, immutable scalar/dtype values
    pass through, containers are recursively frozen, and opaque shared objects
    are not exposed at all.
    """
    if isinstance(value, types.ModuleType):
        root = value.__name__.split(".", 1)[0]
        if root not in ("numpy", "jax", "jaxlib"):
            raise AttributeError(
                f"non-numerical module {value.__name__!r} is disabled")
        return _NumericalProxy(value, activity)
    if isinstance(value, type):
        # NumPy scalar classes are immutable and are also the public dtype
        # tokens expected by calls such as np.asarray(..., dtype=np.float64).
        # Other classes are callable through a wrapper but never returned raw.
        try:
            if issubclass(value, real_workloads_jax.np.generic):
                return value
        except TypeError:
            pass
        return _NumericalProxy(value, activity)
    # Ordinary functions, built-ins, NumPy ufuncs, and JAX's ufunc wrapper are
    # stateless numerical operations.  Do not use a generic callable check:
    # JAX configuration ``State`` singletons are callable too, and calling one
    # returns a context manager that mutates process-global evaluator state.
    callable_type = type(value)
    callable_module = getattr(
        value, "__module__", callable_type.__module__).split(".", 1)[0]
    type_module = callable_type.__module__
    if callable(value):
        if type_module.startswith("jax._src.config"):
            raise AttributeError("stateful JAX configuration is disabled")
        # NumPy array-function dispatchers and JAX PjitFunction objects are
        # callable instances rather than Python functions. Their value/type
        # modules are nevertheless rooted in the admitted numerical packages.
        if (callable_module in ("numpy", "jax", "jaxlib")
                or type_module.split(".", 1)[0] in
                ("numpy", "jax", "jaxlib")):
            return _NumericalProxy(value, activity)
    if value is None or value is Ellipsis or value is NotImplemented:
        return value
    if type(value) in (bool, int, float, complex, str, bytes, range, slice):
        return value
    if isinstance(value, (real_workloads_jax.np.generic,
                          real_workloads_jax.np.dtype)):
        return value
    if isinstance(value, dict):
        return types.MappingProxyType({
            key: _proxy_numerical_value(item, activity)
            for key, item in value.items()
        })
    if isinstance(value, list):
        return tuple(_proxy_numerical_value(item, activity) for item in value)
    if isinstance(value, tuple):
        return tuple(_proxy_numerical_value(item, activity) for item in value)
    if isinstance(value, set):
        return frozenset(
            _proxy_numerical_value(item, activity) for item in value)
    if isinstance(value, frozenset):
        return frozenset(
            _proxy_numerical_value(item, activity) for item in value)
    raise AttributeError(
        f"shared numerical object {type(value).__name__!r} is disabled")


def _shape_copy(shapes):
    return [list(shape) for shape in shapes]


def _block_copy(blocks):
    return [[list(row) for row in block] if block and isinstance(block[0], list)
            else list(block) for block in blocks]


def load_optimizer_candidate(path):
    """Load source and expose only the intended numerical array namespaces."""
    # Import statements remain forbidden. Injecting the already-loaded CPU
    # modules lets a candidate choose list, NumPy, or JAX update math without
    # gaining evaluator/task objects. Each load receives fresh read-only
    # proxies so candidate writes/random state cannot cross workloads.
    activity = {"jax_called": False, "loading": True}
    module = load_candidate(path, REQUIRED, injected_globals={
        "jax": _NumericalProxy(real_workloads_jax.jax, activity),
        "np": _NumericalProxy(real_workloads_jax.np, activity),
        "jnp": _NumericalProxy(real_workloads_jax.jnp, activity),
    }, forbidden_attrs=_NUMERICAL_SOURCE_FORBIDDEN_ATTRS)
    activity["loading"] = False
    # Evaluator-private mapping: candidate source cannot import or reach this
    # module, and retaining one tiny module per workload avoids identity reuse.
    _CANDIDATE_NUMERICAL_ACTIVITY[module] = activity
    return module


def _plain_container(value):
    if type(value) in (list, tuple):
        return value
    # NumPy/JAX arrays are accepted at the API boundary, then copied back to
    # plain lists so every candidate receives the same representation on the
    # next step and evaluator validation remains framework-independent.
    module = type(value).__module__.split(".", 1)[0]
    if module in ("numpy", "jax", "jaxlib") and hasattr(value, "tolist"):
        return value.tolist()
    return value


def _numeric(value, label):
    if type(value) in (int, float):
        result = float(value)
    elif type(value).__module__.split(".", 1)[0] in ("numpy", "jax", "jaxlib"):
        try:
            result = float(value)
        except (TypeError, ValueError, OverflowError):
            eval_lib.fail(f"{label} must contain scalar numeric values")
    else:
        eval_lib.fail(f"{label} must contain plain or array scalar numbers")
    if not math.isfinite(result):
        eval_lib.fail(f"{label} must contain finite values")
    return result


def _synchronize_candidate_value(value, seen=None):
    """Materialize asynchronous JAX outputs before candidate timing stops."""
    if seen is None:
        seen = set()
    identity = id(value)
    if identity in seen:
        return
    seen.add(identity)
    module = type(value).__module__.split(".", 1)[0]
    if module in ("jax", "jaxlib") and hasattr(value, "block_until_ready"):
        value.block_until_ready()
        return
    if type(value) in (list, tuple):
        for item in value:
            _synchronize_candidate_value(item, seen)
    elif type(value) is dict:
        for item in value.values():
            _synchronize_candidate_value(item, seen)


def _synchronize_candidate_output(module, value):
    activity = _CANDIDATE_NUMERICAL_ACTIVITY.get(module)
    if activity is not None and activity["jax_called"]:
        _synchronize_candidate_value(value)


def validate_blocks(blocks, shapes, label):
    blocks = _plain_container(blocks)
    if type(blocks) not in (list, tuple) or len(blocks) != len(shapes):
        eval_lib.fail(f"{label} returned the wrong number of parameter blocks")
    result = []
    for block, shape in zip(blocks, shapes):
        block = _plain_container(block)
        if len(shape) == 1:
            if type(block) not in (list, tuple) or len(block) != shape[0]:
                eval_lib.fail(f"{label} returned a malformed vector block")
            result.append([_numeric(value, label) for value in block])
        elif len(shape) == 2:
            if type(block) not in (list, tuple) or len(block) != shape[0]:
                eval_lib.fail(f"{label} returned a malformed matrix block")
            matrix = []
            for row in block:
                row = _plain_container(row)
                if type(row) not in (list, tuple) or len(row) != shape[1]:
                    eval_lib.fail(f"{label} returned a malformed matrix row")
                matrix.append([_numeric(value, label) for value in row])
            result.append(matrix)
        else:
            eval_lib.fail(f"{label} encountered an unsupported parameter rank")
    return result


def trusted_blocks_or_none(blocks, shapes):
    """Validate committed baseline output without terminating the driver.

    Candidate evaluation never uses this path. A non-finite literature sweep
    configuration is a dominated hyperparameter trial, not broken benchmark
    infrastructure.
    """
    try:
        if type(blocks) not in (list, tuple) or len(blocks) != len(shapes):
            return None
        result = []
        for block, shape in zip(blocks, shapes):
            if len(shape) == 1:
                if type(block) not in (list, tuple) or len(block) != shape[0]:
                    return None
                values = [float(value) for value in block]
                if not all(math.isfinite(value) for value in values):
                    return None
                result.append(values)
            elif len(shape) == 2:
                if type(block) not in (list, tuple) or len(block) != shape[0]:
                    return None
                matrix = []
                for row in block:
                    if type(row) not in (list, tuple) or len(row) != shape[1]:
                        return None
                    values = [float(value) for value in row]
                    if not all(math.isfinite(value) for value in values):
                        return None
                    matrix.append(values)
                result.append(matrix)
            else:
                return None
        return result
    except (TypeError, ValueError, OverflowError):
        return None


def _batch_indices(task, step, count):
    size = min(task["batch_size"], count)
    value = (task["batch_seed"] ^ (step * 0x9E3779B9)) & 0xFFFFFFFF
    result = []
    for _ in range(size):
        value = (1664525 * value + 1013904223) & 0xFFFFFFFF
        result.append(value % count)
    return result


def _factorization_batch_indices(task, step):
    """Return a randomized edge-cover minibatch without using the full graph.

    Every latent row participates at every update, closing the exact-zero
    family tag created by ordinary edge sampling.  The order and fill are
    deterministic functions of the evaluator-owned seed and step.
    """
    records = task["payload"]["train"]
    count = len(records)
    size = min(int(task["batch_size"]), count)
    value = (task["batch_seed"] ^ (step * 0x9E3779B9)) & 0xFFFFFFFF
    order = list(range(count))
    for position in range(count - 1, 0, -1):
        value = (1664525 * value + 1013904223) & 0xFFFFFFFF
        selected = value % (position + 1)
        order[position], order[selected] = order[selected], order[position]

    left_count = int(task["payload"]["left_count"])
    right_count = len(task["initial"][0]) - left_count
    covered_left, covered_right = set(), set()
    chosen, chosen_set = [], set()
    for index in order:
        left, right = int(records[index][0]), int(records[index][1])
        if left not in covered_left or right not in covered_right:
            chosen.append(index)
            chosen_set.add(index)
            covered_left.add(left)
            covered_right.add(right)
        if (len(covered_left) == left_count and
                len(covered_right) == right_count):
            break
    if (len(covered_left) != left_count or
            len(covered_right) != right_count or len(chosen) > size):
        eval_lib.fail("factorization training graph lacks a minibatch edge cover")
    for index in order:
        if len(chosen) >= size:
            break
        if index not in chosen_set:
            chosen.append(index)
            chosen_set.add(index)
    return chosen


def _softplus(value):
    return max(value, 0.0) + math.log1p(math.exp(-abs(value)))


def _sigmoid(value):
    if value >= 0:
        inverse = math.exp(-min(value, 60.0))
        return 1.0 / (1.0 + inverse)
    exponential = math.exp(max(value, -60.0))
    return exponential / (1.0 + exponential)


def _poisson_clip(value):
    """Keep adversarial iterates numerically finite outside the useful basin."""
    return max(-30.0, min(30.0, value))


def validation_loss(task, blocks):
    if task.get("suite") == "real":
        return real_workloads_jax.validation_loss(task, blocks)
    family, payload = task["family"], task["payload"]
    matrix, bias = blocks
    rows, outputs = len(matrix), len(bias)
    if family == "quadratic":
        total = 0.0
        for record in payload["validation"]:
            features, targets = record[:rows], record[rows:]
            for output, target in enumerate(targets):
                prediction = bias[output] + sum(
                    features[j] * matrix[j][output] for j in range(rows))
                total += 0.5 * (prediction - target) ** 2
        raw = total / (len(payload["validation"]) * outputs)
    if family == "logistic":
        total = 0.0
        for record in payload["validation"]:
            features, labels = record[:rows], record[rows:]
            for output, label in enumerate(labels):
                z = bias[output] + sum(
                    features[j] * matrix[j][output] for j in range(rows))
                total += _softplus(z) - label * z
        raw = total / (len(payload["validation"]) * outputs)
    if family == "robust":
        delta = payload["delta"]
        total = 0.0
        for record in payload["validation"]:
            features, targets = record[:rows], record[rows:]
            for output, target in enumerate(targets):
                error = bias[output] + sum(
                    features[j] * matrix[j][output] for j in range(rows)) - target
                absolute = abs(error)
                total += (0.5 * error * error if absolute <= delta else
                          delta * (absolute - 0.5 * delta))
        raw = total / (len(payload["validation"]) * outputs)
    if family == "factorization":
        left_count = payload["left_count"]
        total = 0.0
        for i, j, target in payload["validation"]:
            right_index = left_count + j
            prediction = sum(
                matrix[i][k] * matrix[right_index][k] * (1.0 + bias[k])
                for k in range(outputs))
            error = prediction - target
            total += 0.5 * error * error
        raw = total / len(payload["validation"])
    if family == "softmax":
        class_scales = payload["class_scales"]
        total = 0.0
        for record in payload["validation"]:
            features, label = record[:-1], int(record[-1])
            logits = [class_scales[c] * (
                bias[c] + sum(features[j] * matrix[j][c]
                              for j in range(len(features))))
                      for c in range(len(bias))]
            maximum = max(logits)
            total += maximum + math.log(sum(math.exp(z - maximum)
                                            for z in logits))
            total -= logits[label]
        raw = total / len(payload["validation"])
    if family == "poisson":
        total = 0.0
        for record in payload["validation"]:
            features, counts = record[:rows], record[rows:]
            for output, count in enumerate(counts):
                z = bias[output] + sum(
                    features[j] * matrix[j][output] for j in range(rows))
                clipped = _poisson_clip(z)
                total += math.exp(clipped) - count * clipped
        raw = total / (len(payload["validation"]) * outputs)
    if family == "quantile":
        quantile = payload["quantile"]
        total = 0.0
        for record in payload["validation"]:
            features, targets = record[:rows], record[rows:]
            for output, target in enumerate(targets):
                prediction = bias[output] + sum(
                    features[j] * matrix[j][output] for j in range(rows))
                error = target - prediction
                total += (quantile * error if error >= 0.0 else
                          (quantile - 1.0) * error)
        raw = total / (len(payload["validation"]) * outputs)
    if family == "ranking":
        temperature = payload["temperature"]
        total = 0.0
        for record in payload["validation"]:
            features, labels = record[:rows], record[rows:]
            for output, label in enumerate(labels):
                z = temperature * (bias[output] + sum(
                    features[j] * matrix[j][output] for j in range(rows)))
                total += _softplus(z) - label * z
        raw = total / (len(payload["validation"]) * outputs)
    if family == "nonlinear":
        output_weights = payload["output"]
        total = 0.0
        for record in payload["validation"]:
            features, target = record[:-1], record[-1]
            hidden = [math.tanh(bias[h] + sum(
                features[j] * matrix[j][h] for j in range(rows)))
                      for h in range(outputs)]
            prediction = sum(a * b for a, b in zip(output_weights, hidden))
            total += 0.5 * (prediction - target) ** 2
        raw = total / len(payload["validation"])
    if family == "fourier":
        output_weights = payload["output"]
        total = 0.0
        for record in payload["validation"]:
            features, target = record[:-1], record[-1]
            phases = [bias[h] + sum(
                features[j] * matrix[j][h] for j in range(rows))
                      for h in range(outputs)]
            if not all(math.isfinite(value) for value in phases):
                return math.inf
            hidden = [math.sin(value) for value in phases]
            prediction = sum(a * b for a, b in zip(output_weights, hidden))
            total += 0.5 * (prediction - target) ** 2
        raw = total / len(payload["validation"])
    if family not in ALL_FAMILIES:
        eval_lib.fail(f"unknown optimizer workload family: {family}")
    return raw * float(task.get("loss_scale", 1.0))


def training_gradient(task, blocks, step):
    if task.get("suite") == "real":
        return real_workloads_jax.training_gradient(
            task, blocks, step, real_workloads._indices(task, step))
    family, payload = task["family"], task["payload"]
    matrix, bias = blocks
    matrix_gradient = [[0.0] * len(bias) for _ in matrix]
    bias_gradient = [0.0] * len(bias)
    rows = payload["train"]
    chosen = (_factorization_batch_indices(task, step)
              if family == "factorization" else
              _batch_indices(task, step, len(rows)))
    parameter_rows, outputs = len(matrix), len(bias)
    if family in ("quadratic", "logistic", "robust"):
        delta = payload.get("delta")
        for index in chosen:
            record = rows[index]
            features, targets = record[:parameter_rows], record[parameter_rows:]
            for output, target in enumerate(targets):
                prediction = bias[output] + sum(
                    features[j] * matrix[j][output]
                    for j in range(parameter_rows))
                if family == "logistic":
                    derivative = _sigmoid(prediction) - target
                elif family == "robust":
                    derivative = max(-delta, min(delta, prediction - target))
                else:
                    derivative = prediction - target
                bias_gradient[output] += derivative
                for j, value in enumerate(features):
                    matrix_gradient[j][output] += derivative * value
        scale = 1.0 / (len(chosen) * outputs)
    if family == "factorization":
        left_count = payload["left_count"]
        for index in chosen:
            i, j, target = rows[index]
            right_index = left_count + j
            prediction = sum(
                matrix[i][k] * matrix[right_index][k] * (1.0 + bias[k])
                for k in range(outputs))
            error = prediction - target
            for k in range(outputs):
                left_value, right_value = matrix[i][k], matrix[right_index][k]
                latent_scale = 1.0 + bias[k]
                matrix_gradient[i][k] += error * right_value * latent_scale
                matrix_gradient[right_index][k] += error * left_value * latent_scale
                bias_gradient[k] += error * left_value * right_value
        scale = 1.0 / len(chosen)
    if family == "softmax":
        class_scales = payload["class_scales"]
        for index in chosen:
            record = rows[index]
            features, label = record[:-1], int(record[-1])
            logits = [class_scales[c] * (
                bias[c] + sum(features[j] * matrix[j][c]
                              for j in range(len(features))))
                      for c in range(len(bias))]
            maximum = max(logits)
            probabilities = [math.exp(z - maximum) for z in logits]
            normalizer = sum(probabilities)
            for c in range(len(bias)):
                error = ((probabilities[c] / normalizer
                          - (1.0 if c == label else 0.0))
                         * class_scales[c])
                bias_gradient[c] += error
                for j, value in enumerate(features):
                    matrix_gradient[j][c] += error * value
        scale = 1.0 / len(chosen)
    if family == "poisson":
        for index in chosen:
            record = rows[index]
            features, counts = record[:parameter_rows], record[parameter_rows:]
            for output, count in enumerate(counts):
                z = bias[output] + sum(
                    features[j] * matrix[j][output]
                    for j in range(parameter_rows))
                if z <= -30.0 or z >= 30.0:
                    derivative = 0.0
                else:
                    derivative = math.exp(z) - count
                bias_gradient[output] += derivative
                for j, value in enumerate(features):
                    matrix_gradient[j][output] += derivative * value
        scale = 1.0 / (len(chosen) * outputs)
    if family == "quantile":
        quantile = payload["quantile"]
        for index in chosen:
            record = rows[index]
            features, targets = record[:parameter_rows], record[parameter_rows:]
            for output, target in enumerate(targets):
                prediction = bias[output] + sum(
                    features[j] * matrix[j][output]
                    for j in range(parameter_rows))
                derivative = (-quantile if target >= prediction else
                              1.0 - quantile)
                bias_gradient[output] += derivative
                for j, value in enumerate(features):
                    matrix_gradient[j][output] += derivative * value
        scale = 1.0 / (len(chosen) * outputs)
    if family == "ranking":
        temperature = payload["temperature"]
        for index in chosen:
            record = rows[index]
            features, labels = record[:parameter_rows], record[parameter_rows:]
            for output, label in enumerate(labels):
                z = temperature * (bias[output] + sum(
                    features[j] * matrix[j][output]
                    for j in range(parameter_rows)))
                derivative = temperature * (_sigmoid(z) - label)
                bias_gradient[output] += derivative
                for j, value in enumerate(features):
                    matrix_gradient[j][output] += derivative * value
        scale = 1.0 / (len(chosen) * outputs)
    if family == "nonlinear":
        output_weights = payload["output"]
        for index in chosen:
            record = rows[index]
            features, target = record[:-1], record[-1]
            hidden = [math.tanh(bias[h] + sum(
                features[j] * matrix[j][h] for j in range(parameter_rows)))
                      for h in range(outputs)]
            prediction = sum(a * b for a, b in zip(output_weights, hidden))
            error = prediction - target
            for h in range(outputs):
                derivative = error * output_weights[h] * (1.0 - hidden[h] ** 2)
                bias_gradient[h] += derivative
                for j, value in enumerate(features):
                    matrix_gradient[j][h] += derivative * value
        scale = 1.0 / len(chosen)
    if family == "fourier":
        output_weights = payload["output"]
        for index in chosen:
            record = rows[index]
            features, target = record[:-1], record[-1]
            phases = [bias[h] + sum(
                features[j] * matrix[j][h]
                for j in range(parameter_rows)) for h in range(outputs)]
            hidden = [math.sin(value) for value in phases]
            prediction = sum(a * b for a, b in zip(output_weights, hidden))
            error = prediction - target
            for h in range(outputs):
                derivative = error * output_weights[h] * math.cos(phases[h])
                bias_gradient[h] += derivative
                for j, value in enumerate(features):
                    matrix_gradient[j][h] += derivative * value
        scale = 1.0 / len(chosen)
    if family not in ALL_FAMILIES:
        eval_lib.fail(f"unknown optimizer workload family: {family}")
    scale *= float(task.get("loss_scale", 1.0))
    return [[[value * scale for value in row] for row in matrix_gradient],
            [value * scale for value in bias_gradient]]


def _checkpoint_steps(horizon):
    return set(max(1, round(horizon * index / CHECKPOINTS))
               for index in range(1, CHECKPOINTS + 1))


def run_task(mod, task, trusted_baseline=False):
    shapes = task["shapes"]
    # The initial denominator is evaluator-owned. Candidate view is never
    # consulted at step zero, closing denominator-inflation exploits.
    params = validate_blocks(task["initial"], shapes, "initial parameters")
    initial = validation_loss(task, params)
    anchor = float(task["reference_anchor"])
    denominator = initial - anchor
    if not math.isfinite(initial) or denominator <= 1e-10:
        eval_lib.fail(f"invalid reference anchor for task {task.get('task_id', '?')}")

    candidate_seconds = 0.0
    init_shapes = _shape_copy(shapes)
    started = time.process_time()
    state = call(mod.init, init_shapes)
    _synchronize_candidate_output(mod, state)
    candidate_seconds += time.process_time() - started
    normalized_curve = [1.0]
    capped = 0
    maximum_raw = 1.0
    checkpoints = _checkpoint_steps(task["horizon"])
    for step in range(1, task["horizon"] + 1):
        gradients = training_gradient(task, params, step)
        update_parameters = _block_copy(params)
        started = time.process_time()
        answer = call(mod.update, update_parameters, gradients, state, step)
        _synchronize_candidate_output(mod, answer)
        candidate_seconds += time.process_time() - started
        if type(answer) not in (list, tuple) or len(answer) != 2:
            eval_lib.fail("update must return [new_parameters, new_state]")
        params = (trusted_blocks_or_none(answer[0], shapes)
                  if trusted_baseline else
                  validate_blocks(answer[0], shapes, "update"))
        if params is None:
            return {
                "task_id": task.get("task_id"),
                "suite": task.get("suite", "analytic"),
                "family": task["family"], "track": task["track"],
                "auc": 1.0, "final": 1.0, "best": 1.0,
                "horizon": task["horizon"], "capped": CHECKPOINTS,
                "negative": 0, "checkpoints": CHECKPOINTS + 1,
                "maximum_raw_normalized_loss": 1e300,
                "candidate_seconds": candidate_seconds,
                "parameter_count": real_workloads.parameter_count(task),
                "invalid_baseline_trial": True,
            }
        state = answer[1]
        # Call view on every step so its invocation pattern cannot reveal the
        # hidden horizon. Isolate state so view remains observational.
        view_parameters = _block_copy(params)
        view_state = copy.deepcopy(state)
        started = time.process_time()
        viewed = call(mod.view, view_parameters, view_state, step)
        _synchronize_candidate_output(mod, viewed)
        candidate_seconds += time.process_time() - started
        viewed = (trusted_blocks_or_none(viewed, shapes) if trusted_baseline
                  else validate_blocks(viewed, shapes, "view"))
        if viewed is None:
            return {
                "task_id": task.get("task_id"),
                "suite": task.get("suite", "analytic"),
                "family": task["family"], "track": task["track"],
                "auc": 1.0, "final": 1.0, "best": 1.0,
                "horizon": task["horizon"], "capped": CHECKPOINTS,
                "negative": 0, "checkpoints": CHECKPOINTS + 1,
                "maximum_raw_normalized_loss": 1e300,
                "candidate_seconds": candidate_seconds,
                "parameter_count": real_workloads.parameter_count(task),
                "invalid_baseline_trial": True,
            }
        if step in checkpoints:
            value = validation_loss(task, viewed)
            normalized = ((value - anchor) / denominator
                          if math.isfinite(value) else UPPER_CLIP)
            raw_normalized = normalized
            if normalized > UPPER_CLIP:
                normalized = UPPER_CLIP
                capped += 1
            maximum_raw = max(maximum_raw, raw_normalized)
            normalized_curve.append(normalized)
    curve_auc = ((0.5 * normalized_curve[0]
                  + sum(normalized_curve[1:-1])
                  + 0.5 * normalized_curve[-1])
                 / (len(normalized_curve) - 1))
    return {
        "task_id": task.get("task_id"),
        "suite": task.get("suite", "analytic"),
        "family": task["family"], "track": task["track"],
        "auc": curve_auc,
        "final": normalized_curve[-1], "best": min(normalized_curve),
        "horizon": task["horizon"], "capped": capped,
        "negative": sum(value < 0 for value in normalized_curve),
        "checkpoints": len(normalized_curve),
        "maximum_raw_normalized_loss": maximum_raw,
        "candidate_seconds": candidate_seconds,
        "parameter_count": real_workloads.parameter_count(task),
    }


def _mean(values):
    return sum(values) / len(values)


def paired_workload_bootstrap(candidate_rows, baseline_rows, replicates=5000):
    """Paired candidate-minus-baseline interval with ranked cell weighting."""
    if len(candidate_rows) != len(baseline_rows):
        raise ValueError("optimizer paired rows have different lengths")
    cells = {}
    for candidate, baseline in zip(candidate_rows, baseline_rows):
        identity = (candidate.get("task_id"), candidate.get("suite", "analytic"),
                    candidate["family"], candidate["track"])
        other = (baseline.get("task_id"), baseline.get("suite", "analytic"),
                 baseline["family"], baseline["track"])
        if identity != other:
            raise ValueError("optimizer paired workload alignment is corrupt")
        if identity[1] == "real":
            cells.setdefault(identity[2:], []).append(
                float(candidate["auc"]) - float(baseline["auc"]))
    if not cells:
        raise ValueError("optimizer paired comparison has no real workloads")

    def reduce(cell_values):
        known = [value for (family, _track), value in cell_values.items()
                 if family not in TEST_ONLY_REAL_FAMILIES]
        unseen = [value for (family, _track), value in cell_values.items()
                  if family in TEST_ONLY_REAL_FAMILIES]
        result = _mean(known)
        return 0.5 * (result + _mean(unseen)) if unseen else result

    observed = reduce({key: _mean(values) for key, values in cells.items()})
    rng = random.Random(0xC01A5E09)
    draws = []
    for _ in range(replicates):
        draws.append(reduce({
            key: _mean([values[rng.randrange(len(values))] for _ in values])
            for key, values in cells.items()
        }))
    draws.sort()
    return {
        "candidate_minus_baseline": round(observed, 8),
        "ci95": [round(draws[int(.025 * replicates)], 8),
                 round(draws[int(.975 * replicates) - 1], 8)],
        "negative_favors_candidate": True,
        "method": ("paired architecture-generalization/track-stratified "
                   "workload bootstrap"),
        "replicates": replicates,
    }


def _reference_rows(split):
    payload = json.loads((DATA / "reference_baselines.json").read_text())
    if (payload.get("schema") != "optimizer-reference-baselines-v1"
            or payload.get("protocol") != PROTOCOL):
        raise ValueError("optimizer reference baseline artifact has wrong schema")
    split_path = DATA / ("heldout_val.bin" if split == "validation"
                         else "heldout_test.bin")
    expected_split_hash = payload.get("split_sha256", {}).get(split)
    if (not isinstance(expected_split_hash, str)
            or hashlib.sha256(split_path.read_bytes()).hexdigest()
            != expected_split_hash):
        raise ValueError("optimizer reference baseline targets stale split bytes")
    name = payload.get("selected_method")
    rows = payload.get("workload_rows", {}).get(split)
    if not isinstance(name, str) or not isinstance(rows, list):
        raise ValueError("optimizer reference baseline artifact is incomplete")
    return name, rows


def score_split(mod_or_factory, tasks, reference_split=None,
                include_rows=False, include_uncertainty=True):
    rows = []
    for task in tasks:
        # Production evaluation passes a factory, reloading candidate source for
        # every workload so mutable module globals cannot communicate task order.
        mod = mod_or_factory() if callable(mod_or_factory) else mod_or_factory
        try:
            rows.append(run_task(mod, task))
        finally:
            # Activity tracking must not turn source isolation into a module
            # retention leak when hundreds of fresh candidates are evaluated.
            _CANDIDATE_NUMERICAL_ACTIVITY.pop(mod, None)
    result = aggregate_rows(rows, include_uncertainty=include_uncertainty)
    if reference_split is not None and include_uncertainty:
        name, baseline_rows = _reference_rows(reference_split)
        result["reference_baseline"] = name
        result["paired_candidate_minus_reference"] = paired_workload_bootstrap(
            rows, baseline_rows)
    if include_rows:
        result["workload_rows"] = rows
    return result


def aggregate_rows(rows, include_uncertainty=True):
    """Aggregate precomputed per-workload rows with the ranked metric math."""
    if not rows:
        raise ValueError("optimizer scored rows are empty")
    cells = {}
    for row in rows:
        cells.setdefault((row.get("suite", "analytic"), row["family"],
                          row["track"]), []).append(row["auc"])
    cell_auc = {key: _mean(values) for key, values in cells.items()}
    real_cells = [value for (suite, _, _), value in cell_auc.items()
                  if suite == "real"]
    unseen_architecture_cells = [
        value for (suite, family, _), value in cell_auc.items()
        if suite == "real" and family in TEST_ONLY_REAL_FAMILIES]
    known_architecture_cells = [
        value for (suite, family, _), value in cell_auc.items()
        if suite == "real" and family not in TEST_ONLY_REAL_FAMILIES]
    analytic_cells = [value for (suite, _, _), value in cell_auc.items()
                      if suite == "analytic"]
    ranked_suite = "real" if real_cells else "analytic"
    ranked_families = sorted({family for (suite, family, _) in cells
                              if suite == ranked_suite})
    ranked_tracks = sorted({track for (suite, _, track) in cells
                            if suite == ranked_suite})
    family_auc = {family: _mean([
        value for (suite, name, _), value in cell_auc.items()
        if suite == ranked_suite and name == family]) for family in ranked_families}
    track_auc = {track: _mean([
        value for (suite, _, name), value in cell_auc.items()
        if suite == ranked_suite and name == track]) for track in ranked_tracks}
    analytic_families = sorted({family for (suite, family, _) in cells
                                if suite == "analytic"})
    analytic_family_auc = {family: _mean([
        value for (suite, name, _), value in cell_auc.items()
        if suite == "analytic" and name == family])
        for family in analytic_families}
    known_cells = [value for (suite, family, _), value in cell_auc.items()
                   if suite == "analytic" and family not in UNSEEN_FAMILIES]
    unseen_cells = [value for (suite, family, _), value in cell_auc.items()
                    if suite == "analytic" and family in UNSEEN_FAMILIES]
    # Research claims are governed only by actual neural training.  The
    # synthetic tier remains visible as a fast debugging/generalization
    # diagnostic and cannot compensate for poor real-workload performance.
    if real_cells:
        known_score = _mean(known_architecture_cells)
        unseen_score = (_mean(unseen_architecture_cells)
                        if unseen_architecture_cells else None)
        score = ((known_score + unseen_score) / 2.0
                 if unseen_score is not None else known_score)
    else:
        known_score, unseen_score = None, None
        score = _mean(analytic_cells)
    # The scalar is a macro-average over family/track cells, so uncertainty
    # must use the same stratification rather than a micro variance over all
    # workloads.
    ranked_cells = {key: value for key, value in cells.items()
                    if key[0] == ranked_suite}
    bootstrap = []
    standard_error = None
    if include_uncertainty:
        rng = random.Random(0xB00757A9)
        for _ in range(2000):
            means = {}
            for key, values in ranked_cells.items():
                means[key] = _mean([values[rng.randrange(len(values))]
                                    for _ in values])
            if ranked_suite == "real":
                known = [value for (_suite, family, _track), value in means.items()
                         if family not in TEST_ONLY_REAL_FAMILIES]
                unseen = [value for (_suite, family, _track), value in means.items()
                          if family in TEST_ONLY_REAL_FAMILIES]
                draw = _mean(known)
                if unseen:
                    draw = 0.5 * (draw + _mean(unseen))
                bootstrap.append(draw)
            else:
                bootstrap.append(_mean(list(means.values())))
        bootstrap.sort()
        standard_error = math.sqrt(sum((value - _mean(bootstrap)) ** 2
                                       for value in bootstrap) /
                                   (len(bootstrap) - 1))
    result = {
        "score": score,
        "reference_normalized_curve_auc": round(score, 8),
        "score_se": (round(standard_error, 8)
                     if standard_error is not None else None),
        "score_ci95": ([round(bootstrap[49], 8), round(bootstrap[1949], 8)]
                       if bootstrap else None),
        "ci_method": (
            "deterministic family/track-stratified workload bootstrap"
            if bootstrap else None),
        "uncertainty_deferred": not include_uncertainty,
        "real_workload_auc": round(_mean(real_cells), 8) if real_cells else None,
        "known_architecture_auc": (round(known_score, 8)
                                   if known_score is not None else None),
        "unseen_architecture_auc": (round(unseen_score, 8)
                                    if unseen_score is not None else None),
        "architecture_generalization_weighting": (
            "50% known + 50% unseen when unseen architectures are present; "
            "known only during development"),
        "analytic_diagnostic_auc": (round(_mean(analytic_cells), 8)
                                    if analytic_cells else None),
        "id_auc": round(track_auc["id"], 8) if "id" in track_auc else None,
        "ood_auc": round(track_auc["ood"], 8) if "ood" in track_auc else None,
        "known_family_auc": (round(_mean(known_cells), 8)
                             if known_cells else None),
        "unseen_family_auc": (round(_mean(unseen_cells), 8)
                              if unseen_cells else None),
        "known_family_count": len({family for family in analytic_families
                                   if family not in UNSEEN_FAMILIES}),
        "unseen_family_count": len({family for family in analytic_families
                                    if family in UNSEEN_FAMILIES}),
        "track_auc": {key: round(value, 8) for key, value in track_auc.items()},
        "family_auc": {key: round(value, 8) for key, value in family_auc.items()},
        "analytic_family_auc": {key: round(value, 8)
                                for key, value in analytic_family_auc.items()},
        "cell_auc": {suite + "/" + family + "/" + track: round(value, 8)
                     for (suite, family, track), value in sorted(cell_auc.items())},
        "worst_family_auc": round(max(family_auc.values()), 8),
        "mean_final_normalized_loss": round(_mean([row["final"] for row in rows]), 8),
        "mean_best_normalized_loss": round(_mean([row["best"] for row in rows]), 8),
        "negative_checkpoint_fraction": round(
            sum(row["negative"] for row in rows) /
            sum(row["checkpoints"] for row in rows), 8),
        "capped_divergence_checkpoints": sum(row["capped"] for row in rows),
        "invalid_baseline_workloads": sum(
            bool(row.get("invalid_baseline_trial")) for row in rows),
        "maximum_raw_normalized_loss": round(max(
            row["maximum_raw_normalized_loss"] for row in rows), 8),
        "candidate_seconds": round(sum(row["candidate_seconds"] for row in rows), 6),
        "candidate_microseconds_per_parameter_step": round(
            1e6 * sum(row["candidate_seconds"] for row in rows) /
            max(1, sum(row["parameter_count"] * row["horizon"] for row in rows)), 8),
        "horizon_min": min(row["horizon"] for row in rows),
        "horizon_max": max(row["horizon"] for row in rows),
        "horizon_mean": round(_mean([row["horizon"] for row in rows]), 4),
        "n_workloads": len(rows),
    }
    return result


def _read(path):
    payload = (json.loads(path.read_text()) if path.suffix == ".json"
               else heldout.read(path))
    if payload.get("schema") != SCHEMA or payload.get("protocol") != PROTOCOL:
        eval_lib.fail(f"optimizer task data at {path.name} has the wrong schema")
    return payload["tasks"]


def _argument_value(name):
    positions = [index for index, value in enumerate(sys.argv[2:])
                 if value == name]
    if len(positions) != 1:
        eval_lib.fail(f"{name} requires exactly one value")
    index = positions[0] + 2
    if index + 1 >= len(sys.argv):
        eval_lib.fail(f"{name} requires a value")
    return sys.argv[index + 1]


def main():
    final = "--final" in sys.argv[2:]
    train_only = "--train-only" in sys.argv[2:]
    test_only = "--test-only" in sys.argv[2:]
    if test_only and (final or train_only):
        eval_lib.fail("--test-only cannot be combined with --final/--train-only")
    test_shard = _argument_value("--test-shard") if test_only else None
    if test_only and test_shard != "full":
        eval_lib.fail(f"unknown deferred optimizer test shard: {test_shard!r}")
    path = sys.argv[1]
    train_all = _read(DATA / "train.json")
    train = [task for task in train_all if task.get("suite") == "real"]

    def fresh(tasks, reference_split=None, include_rows=False,
              include_uncertainty=True):
        return score_split(
            lambda: load_optimizer_candidate(path), tasks,
            reference_split=reference_split, include_rows=include_rows,
            include_uncertainty=include_uncertainty)

    if test_only:
        test_result = fresh(_read(DATA / "heldout_test.bin"), "test", True)
        metrics = {key: value for key, value in test_result.items()
                   if key != "score"}
        metrics.update(schema=SCHEMA, protocol_version=PROTOCOL,
                       deferred_test_shard="full")
        eval_lib.succeed(test_result["score"], metrics)

    train_result = fresh(train, include_uncertainty=final)
    if train_only:
        eval_lib.succeed(train_result["score"], split_metrics(train_result))
    validation_all = _read(DATA / "heldout_val.bin")
    validation = [task for task in validation_all if task.get("suite") == "real"]
    validation_result = fresh(
        validation, "validation", include_uncertainty=final)
    test_result = (fresh(_read(DATA / "heldout_test.bin"), "test", True)
                   if final else None)
    metrics = split_metrics(train_result, validation_result, test_result)
    metrics.update(schema=SCHEMA, protocol_version=PROTOCOL,
                   checkpoints=CHECKPOINTS + 1,
                   validation_ranked_real_families=5,
                   sealed_test_ranked_real_families=8,
                   sealed_test_unseen_real_families=3,
                   sealed_analytic_diagnostic_families=10,
                   ranked_tier="real neural workloads only",
                   diagnostic_tier="analytic family-generalization workloads",
                   metric_provenance=("TaskSet-style empirical-best normalized "
                                      "validation-loss curve area"),
                   upper_clip=UPPER_CLIP,
                   evaluator_backend=real_workloads_jax.backend(),
                   candidate_metadata=("natural parameter blocks, gradients, "
                                       "and step only; list/NumPy/CPU-JAX "
                                       "update math accepted"))
    eval_lib.succeed(validation_result["score"], metrics)


if __name__ == "__main__":
    main()
