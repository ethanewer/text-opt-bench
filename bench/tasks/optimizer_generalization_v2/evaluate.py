"""Optimizer-generalization v8: expanded real-workload primary."""

import copy
import json
import math
import random
import sys
import time
from pathlib import Path

from bench import eval_lib, heldout
from bench.ml_eval import call, finite, load_candidate, split_metrics
from bench.tasks.optimizer_generalization_v2 import real_workloads


DATA = Path(__file__).resolve().parent / "data"
CHECKPOINTS = 16
UPPER_CLIP = 1.0
REQUIRED = ("init", "update", "view")
SCHEMA = 7
PROTOCOL = 8
UNSEEN_FAMILIES = frozenset((
    "nonlinear", "poisson", "quantile", "ranking", "fourier",
))
ALL_FAMILIES = frozenset((
    "quadratic", "logistic", "robust", "factorization", "softmax",
)) | UNSEEN_FAMILIES
TEST_ONLY_REAL_FAMILIES = frozenset(("image_residual",))


def _shape_copy(shapes):
    return [list(shape) for shape in shapes]


def _block_copy(blocks):
    return [[list(row) for row in block] if block and isinstance(block[0], list)
            else list(block) for block in blocks]


def validate_blocks(blocks, shapes, label):
    if type(blocks) not in (list, tuple) or len(blocks) != len(shapes):
        eval_lib.fail(f"{label} returned the wrong number of parameter blocks")
    result = []
    for block, shape in zip(blocks, shapes):
        if len(shape) == 1:
            if type(block) not in (list, tuple) or len(block) != shape[0]:
                eval_lib.fail(f"{label} returned a malformed vector block")
            result.append([finite(value, label) for value in block])
        elif len(shape) == 2:
            if type(block) not in (list, tuple) or len(block) != shape[0]:
                eval_lib.fail(f"{label} returned a malformed matrix block")
            matrix = []
            for row in block:
                if type(row) not in (list, tuple) or len(row) != shape[1]:
                    eval_lib.fail(f"{label} returned a malformed matrix row")
                matrix.append([finite(value, label) for value in row])
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
        return real_workloads.validation_loss(task, blocks)
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
        return real_workloads.training_gradient(task, blocks, step)
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
        candidate_seconds += time.process_time() - started
        if type(answer) not in (list, tuple) or len(answer) != 2:
            eval_lib.fail("update must return [new_parameters, new_state]")
        params = (trusted_blocks_or_none(answer[0], shapes)
                  if trusted_baseline else
                  validate_blocks(answer[0], shapes, "update"))
        if params is None:
            return {
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
        candidate_seconds += time.process_time() - started
        viewed = (trusted_blocks_or_none(viewed, shapes) if trusted_baseline
                  else validate_blocks(viewed, shapes, "view"))
        if viewed is None:
            return {
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


def score_split(mod_or_factory, tasks):
    rows = []
    for task in tasks:
        # Production evaluation passes a factory, reloading candidate source for
        # every workload so mutable module globals cannot communicate task order.
        mod = mod_or_factory() if callable(mod_or_factory) else mod_or_factory
        rows.append(run_task(mod, task))
    return aggregate_rows(rows)


def aggregate_rows(rows):
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
    heldout_architecture_cells = [
        value for (suite, family, _), value in cell_auc.items()
        if suite == "real" and family in TEST_ONLY_REAL_FAMILIES]
    development_architecture_cells = [
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
    score = _mean(real_cells if real_cells else analytic_cells)
    # The scalar is a macro-average over family/track cells, so uncertainty
    # must use the same stratification rather than a micro variance over all
    # workloads.
    ranked_cells = {key: value for key, value in cells.items()
                    if key[0] == ranked_suite}
    rng = random.Random(0xB00757A9)
    bootstrap = []
    for _ in range(2000):
        means = []
        for values in ranked_cells.values():
            means.append(_mean([values[rng.randrange(len(values))]
                                for _ in values]))
        bootstrap.append(_mean(means))
    bootstrap.sort()
    standard_error = math.sqrt(sum((value - _mean(bootstrap)) ** 2
                                   for value in bootstrap) /
                               (len(bootstrap) - 1))
    result = {
        "score": score,
        "reference_normalized_curve_auc": round(score, 8),
        "score_se": round(standard_error, 8),
        "score_ci95": [round(bootstrap[49], 8), round(bootstrap[1949], 8)],
        "ci_method": "deterministic family/track-stratified workload bootstrap",
        "real_workload_auc": round(_mean(real_cells), 8) if real_cells else None,
        "development_architecture_auc": (
            round(_mean(development_architecture_cells), 8)
            if development_architecture_cells else None),
        "heldout_architecture_auc": (
            round(_mean(heldout_architecture_cells), 8)
            if heldout_architecture_cells else None),
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
    train = _read(DATA / "train.json")

    def fresh(tasks):
        return score_split(lambda: load_candidate(path, REQUIRED), tasks)

    if test_only:
        test_result = fresh(_read(DATA / "heldout_test.bin"))
        metrics = {key: value for key, value in test_result.items()
                   if key != "score"}
        metrics.update(schema=SCHEMA, protocol_version=PROTOCOL,
                       deferred_test_shard="full")
        eval_lib.succeed(test_result["score"], metrics)

    train_result = fresh(train)
    if train_only:
        eval_lib.succeed(train_result["score"], split_metrics(train_result))
    validation_result = fresh(_read(DATA / "heldout_val.bin"))
    test_result = (fresh(_read(DATA / "heldout_test.bin")) if final else None)
    metrics = split_metrics(train_result, validation_result, test_result)
    metrics.update(schema=SCHEMA, protocol_version=PROTOCOL,
                   checkpoints=CHECKPOINTS + 1,
                   validation_workload_families=10,
                   sealed_test_workload_families=16,
                   sealed_test_unseen_families=5,
                   ranked_tier="real neural workloads only",
                   diagnostic_tier="analytic family-generalization workloads",
                   metric_provenance=("TaskSet-style empirical-best normalized "
                                      "validation-loss curve area"),
                   upper_clip=UPPER_CLIP,
                   candidate_metadata=("natural parameter blocks, gradients, "
                                       "and step only"))
    eval_lib.succeed(validation_result["score"], metrics)


if __name__ == "__main__":
    main()
