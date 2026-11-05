#!/usr/bin/env python3
"""Dataset-as-course Kinetra: one continuous trajectory, labels only in loss.

Run this file without flags for the registered Wine experiment. Only NumPy,
SciPy/scikit-learn and (for the optional figure) matplotlib are required.
The simulator and exact reverse-mode gradient use NumPy; no PyTorch is needed.
This is deliberately a small, differentiable toy simulator, not rigid-body
physics and not an established neural-network replacement.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
from sklearn import __version__ as sklearn_version
from sklearn.datasets import load_wine
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

HERE = Path(__file__).resolve().parent
CENTERS = np.array([-1.0, 0.0, 1.0])
HIT_RADIUS = 0.32
CONTACT_WIDTH = 0.30
INITIAL_SCALE = np.array([1.2, 0.25, 0.6, 0.6])
FEATURE_COUNT = 13
SPLIT_SEED = 20261104
SEEDS = [701, 702, 703, 704, 705]
PARAMETER_NAMES = [
    "launch_position", "launch_velocity", "launch_angle", "launch_spin",
    "force_gain", "torque_gain", "spin_gain", "velocity_retention",
    "centering_spring", "lateral_bias",
]


def encode_inputs(inputs, mean, scale):
    """Each feature becomes one bounded sideways force; labels are absent.

    Feature order is the original Wine column order. The fixed tanh transform
    is a numerical range bound, not a learned feature extractor. Scaling
    statistics must come from training inputs only.
    """
    inputs = np.asarray(inputs, dtype=np.float64)
    mean = np.asarray(mean, dtype=np.float64)
    scale = np.asarray(scale, dtype=np.float64)
    if inputs.shape[-1] != FEATURE_COUNT:
        raise ValueError("This first encoder expects Wine's 13 features.")
    if mean.shape != (FEATURE_COUNT,) or scale.shape != (FEATURE_COUNT,):
        raise ValueError("Expected 13 training means and scales.")
    if not np.isfinite(inputs).all() or not np.isfinite(mean).all():
        raise ValueError("Inputs and means must be finite.")
    if not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("Scales must be finite and positive.")
    return 0.25 * np.tanh((inputs - mean) / scale / 2.5)


def physical_parameters(raw):
    """Bound shared parameters and return their elementwise derivatives."""
    raw = np.atleast_2d(np.asarray(raw, dtype=np.float64))
    if raw.shape[-1] != 10 or not np.isfinite(raw).all():
        raise ValueError("Expected 10 finite raw parameters per ball.")
    t = np.tanh(raw)
    value = np.empty_like(raw)
    derivative = np.empty_like(raw)
    value[:, :4] = t[:, :4] * INITIAL_SCALE
    derivative[:, :4] = (1 - t[:, :4] ** 2) * INITIAL_SCALE
    scales = np.array([2.0, 1.5, 1.5])
    value[:, 4:7] = t[:, 4:7] * scales
    derivative[:, 4:7] = (1 - t[:, 4:7] ** 2) * scales
    # tanh implements a stable sigmoid without overflow.
    sigmoid = (1 + np.tanh(raw[:, 7:9] / 2)) / 2
    value[:, 7] = 0.55 + 0.40 * sigmoid[:, 0]
    value[:, 8] = 0.005 + 0.12 * sigmoid[:, 1]
    derivative[:, 7] = 0.40 * sigmoid[:, 0] * (1 - sigmoid[:, 0])
    derivative[:, 8] = 0.12 * sigmoid[:, 1] * (1 - sigmoid[:, 1])
    value[:, 9] = 0.10 * t[:, 9]
    derivative[:, 9] = 0.10 * (1 - t[:, 9] ** 2)
    return value, derivative


def initial_parameters(seed):
    raw = np.zeros(10)
    raw[:4] = np.random.default_rng(seed).normal(0, 0.12, 4)
    raw[4:7] = np.arctanh([0.5, 1 / 3, 0.2])
    return raw


def contact(position):
    """Smooth lateral collision surrogate for ALL three fixed target pads.

    The pad layout and impulse are independent of the sample's true class.
    The hard success score uses HIT_RADIUS; this smooth impulse is only the
    differentiable approximation to physical contact, with Gaussian tails.
    """
    delta = position[:, None] - CENTERS[None, :]
    bump = np.exp(-0.5 * (delta / CONTACT_WIDTH) ** 2)
    impulse = 0.04 * np.sum(delta * bump, axis=1)
    derivative = 0.04 * np.sum(bump * (1 - (delta / CONTACT_WIDTH) ** 2), axis=1)
    return impulse, derivative


def rollout(raw, course, initial_state=None):
    """Roll through every station without resets. No label argument exists.

    course: [balls, stations, 13] forces, or [stations, 13].
    raw: one shared parameter vector per ball, held fixed for the whole run.
    Returns endpoint positions and ALL states, including the initial state.
    State order: lateral position, velocity, angle, spin. Each batch member
    is a separate complete course, never an independently reset sample.
    """
    course = np.asarray(course, dtype=np.float64)
    if course.ndim == 2:
        course = course[None, :, :]
    if course.ndim != 3 or course.shape[-1] != FEATURE_COUNT or course.shape[1] == 0:
        raise ValueError("Expected a nonempty course of 13-feature stations.")
    if not np.isfinite(course).all():
        raise ValueError("Course forces must be finite.")
    b, n, f = course.shape
    params, _ = physical_parameters(raw)
    if len(params) == 1 and b != 1:
        params = np.repeat(params, b, axis=0)
    if len(params) != b:
        raise ValueError("Parameter batch and course batch must agree.")
    if initial_state is None:
        state = params[:, :4].copy()
    else:
        state = np.asarray(initial_state, dtype=np.float64)
        if state.shape == (4,):
            state = np.broadcast_to(state, (b, 4)).copy()
        if state.shape != (b, 4) or not np.isfinite(state).all():
            raise ValueError("Expected one finite position/velocity/angle/spin per ball.")
    states = np.empty((n * f + 1, b, 4), dtype=np.float64)
    states[0] = state
    forces = course.reshape(b, n * f).T
    force_gain, torque_gain, spin_gain, drag, spring, bias = params[:, 4:].T
    for step, u in enumerate(forces):
        p, v, a, s = states[step].T
        sn = 0.80 * s + spin_gain * u
        an = 0.75 * a + 0.18 * sn + torque_gain * u
        vn = drag * v - spring * p + 0.18 * np.sin(an) + 0.10 * sn + force_gain * u + bias
        pn = 2.0 * np.tanh((p + 0.35 * vn) / 2.0)
        if (step + 1) % f == 0:
            vn = vn + contact(pn)[0]
        states[step + 1, :, 0] = pn
        states[step + 1, :, 1] = vn
        states[step + 1, :, 2] = an
        states[step + 1, :, 3] = sn
    positions = states[f::f, :, 0].T
    return positions, states


def targets(labels, shape):
    labels = np.asarray(labels)
    if labels.ndim == 1 and shape[0] == 1:
        labels = labels[None, :]
    if labels.shape != shape or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("Expected one integer class per course station.")
    if np.any((labels < 0) | (labels >= len(CENTERS))):
        raise ValueError("Wine classes must be 0, 1, or 2.")
    return CENTERS[labels]


def loss_and_gradient(raw, course, labels):
    """Exact reverse-mode backprop AFTER the entire continuous forward run.

    Loss is mean squared distance from the correct pad center. Every step's
    state dependence (including cross-station contact) is differentiated.
    Returns one loss/gradient per independent complete course.
    """
    course = np.asarray(course, dtype=np.float64)
    if course.ndim == 2:
        course = course[None, :, :]
    positions, states = rollout(raw, course)
    b, n, f = course.shape
    params, derivative = physical_parameters(raw)
    if len(params) == 1 and b != 1:
        params = np.repeat(params, b, axis=0)
        derivative = np.repeat(derivative, b, axis=0)
    error = positions - targets(labels, positions.shape)
    loss = np.mean(error ** 2, axis=1)
    endpoint_gradient = 2 * error / n
    adjoint = np.zeros((b, 4))
    gradient = np.zeros((b, 10))
    forces = course.reshape(b, n * f).T
    for step in range(n * f - 1, -1, -1):
        p, v, _, _ = states[step].T
        pn, _, an, _ = states[step + 1].T
        gp, gv, ga, gs = adjoint.T
        if (step + 1) % f == 0:
            gp = gp + endpoint_gradient[:, step // f] + gv * contact(pn)[1]
        unconstrained_position_grad = gp * (1 - (pn / 2) ** 2)
        gvn = gv + 0.35 * unconstrained_position_grad
        gan = ga + 0.18 * np.cos(an) * gvn
        gsn = gs + 0.10 * gvn + 0.18 * gan
        u = forces[step]
        gradient[:, 4] += gvn * u
        gradient[:, 5] += gan * u
        gradient[:, 6] += gsn * u
        gradient[:, 7] += gvn * v
        gradient[:, 8] -= gvn * p
        gradient[:, 9] += gvn
        adjoint = np.column_stack((
            unconstrained_position_grad - params[:, 8] * gvn,
            params[:, 7] * gvn, 0.75 * gan, 0.80 * gsn,
        ))
    gradient[:, :4] = adjoint
    gradient *= derivative
    return loss, gradient


def score_positions(positions, labels):
    positions = np.atleast_2d(positions)
    correct_centers = targets(labels, positions.shape)
    predicted = np.argmin(abs(positions[:, :, None] - CENTERS), axis=2)
    true = np.argmin(abs(correct_centers[:, :, None] - CENTERS), axis=2)
    hits = abs(positions - correct_centers) <= HIT_RADIUS
    any_contact = np.min(abs(positions[:, :, None] - CENTERS), axis=2) <= HIT_RADIUS
    return {
        "correct_hit_rate": hits.mean(axis=1),
        "class_accuracy": (predicted == true).mean(axis=1),
        "any_contact_rate": any_contact.mean(axis=1),
        "all_obstacles_hit": hits.all(axis=1),
        "mse": ((positions - correct_centers) ** 2).mean(axis=1),
    }


def load_splits():
    """One locked, stratified 60/20/20 split: 106 / 36 / 36 Wine rows."""
    data = load_wine()
    indices = np.arange(len(data.target))
    train_val, test = train_test_split(indices, test_size=0.20, random_state=SPLIT_SEED, stratify=data.target)
    train, val = train_test_split(train_val, test_size=0.25, random_state=SPLIT_SEED + 1, stratify=data.target[train_val])
    mean = data.data[train].mean(axis=0)
    scale = data.data[train].std(axis=0)
    scale[scale == 0] = 1
    splits = {}
    for name, ix in [("train", train), ("validation", val), ("test", test)]:
        splits[name] = {
            "indices": ix, "X": data.data[ix], "labels": data.target[ix],
            "course": encode_inputs(data.data[ix], mean, scale),
        }
    return data, splits, mean, scale


def ordered_courses(course, labels, seed, count):
    rng = np.random.default_rng(seed)
    order = np.array([rng.permutation(len(course)) for _ in range(count)])
    return course[order], labels[order], order


def train_models(train_course, train_labels, val_course, val_labels, epochs=250, orders=2):
    """Fit five independent seeds in each of two modes with vectorized NumPy.

    Independent seeds share no gradients or parameters. Validation chooses
    the epoch by correct-hit rate, with MSE as a tie breaker; test is absent.
    A batch contains complete shuffled courses with continuous state.
    """
    specs = [(mode, seed) for mode in ["launch_only", "shared_motion"] for seed in SEEDS]
    raw = np.array([initial_parameters(seed) for mode, seed in specs])
    mask = np.ones_like(raw)
    mask[:len(SEEDS), 4:] = 0
    initial_raw = raw.copy()
    best_raw = raw.copy()
    best_hit = np.full(len(raw), -np.inf)
    best_mse = np.full(len(raw), np.inf)
    best_epoch = np.zeros(len(raw), dtype=int)
    momentum = np.zeros_like(raw)
    variance = np.zeros_like(raw)
    val_x, val_y, _ = ordered_courses(val_course, val_labels, 1801, 4)
    val_batch = np.tile(val_x, (len(raw), 1, 1))
    val_targets = np.tile(val_y, (len(raw), 1))
    history = []
    start = time.monotonic()
    for epoch in range(epochs + 1):
        if epoch % 10 == 0 or epoch == epochs:
            positions, _ = rollout(np.repeat(raw, 4, axis=0), val_batch)
            val_score = score_positions(positions, val_targets)
            hit = val_score["correct_hit_rate"].reshape(len(raw), 4).mean(axis=1)
            mse = val_score["mse"].reshape(len(raw), 4).mean(axis=1)
            improve = (hit > best_hit) | ((hit == best_hit) & (mse < best_mse))
            best_raw[improve] = raw[improve]
            best_hit[improve] = hit[improve]
            best_mse[improve] = mse[improve]
            best_epoch[improve] = epoch
            for i, (mode, seed) in enumerate(specs):
                history.append({"mode": mode, "seed": seed, "epoch": epoch,
                                "validation_correct_hit_rate": float(hit[i]), "validation_mse": float(mse[i])})
            if epoch % 50 == 0 or epoch == epochs:
                print(f"epoch {epoch:3d}/{epochs} | validation hits "
                      f"launch {hit[:5].mean():.3f}, shared {hit[5:].mean():.3f} | "
                      f"elapsed {time.monotonic() - start:.1f}s", flush=True)
        if epoch == epochs:
            break
        # Identical shuffled courses across model seeds/modes for comparison.
        x, y, _ = ordered_courses(train_course, train_labels, 10000 + epoch, orders)
        batch_x = np.tile(x, (len(raw), 1, 1))
        batch_y = np.tile(y, (len(raw), 1))
        _, grads = loss_and_gradient(np.repeat(raw, orders, axis=0), batch_x, batch_y)
        grad = grads.reshape(len(raw), orders, 10).mean(axis=1) * mask
        norm = np.linalg.norm(grad, axis=1, keepdims=True)
        grad *= np.minimum(1, 2.0 / np.maximum(norm, 1e-12))
        momentum = 0.9 * momentum + 0.1 * grad
        variance = 0.999 * variance + 0.001 * grad ** 2
        t = epoch + 1
        raw -= 0.025 * (momentum / (1 - 0.9 ** t)) / (np.sqrt(variance / (1 - 0.999 ** t)) + 1e-8)
    return specs, initial_raw, best_raw, best_epoch, history


def summarize_orders(raw, course, labels, seed=2901, count=20):
    x, y, order = ordered_courses(course, labels, seed, count)
    positions, _ = rollout(raw, x)
    scores = score_positions(positions, y)
    stats = {}
    for metric, values in scores.items():
        stats[metric] = float(np.mean(values))
        stats[metric + "_min"] = float(np.min(values))
        stats[metric + "_max"] = float(np.max(values))
    # Pair every row back to its identity; disagreement across preceding
    # course histories is order sensitivity, not independent test evidence.
    pred = np.argmin(abs(positions[:, :, None] - CENTERS), axis=2)
    restored = np.empty_like(pred)
    for j in range(count):
        restored[j, order[j]] = pred[j]
    stats["order_sensitive_fraction"] = float(np.any(restored != restored[:1], axis=0).mean())
    stats["fully_won_courses"] = int(scores["all_obstacles_hit"].sum())
    return stats


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def save_launch_sensitivity(output, raw, course):
    """Frozen-model diagnosis: perturb each raw launch setting by +/-1.

    Measure how strongly the starting configuration still affects lateral
    position as a continuous course progresses. No labels or fitting are used.
    """
    effects = []
    for parameter in range(4):
        plus, minus = raw.copy(), raw.copy()
        plus[parameter] += 1
        minus[parameter] -= 1
        effects.append(abs(rollout(plus, course)[0][0] - rollout(minus, course)[0][0]))
    rows = [{"station": i + 1, **{PARAMETER_NAMES[j] + "_position_difference": float(effects[j][i])
                                  for j in range(4)},
             "max_position_difference": float(max(e[i] for e in effects))}
            for i in range(len(course))]
    write_csv(output / "launch_sensitivity.csv", rows)


def save_preview(output, raw, split, selected_mode, selected_seed):
    """Export the first test course's entire trajectory and a scoring view."""
    x, y, order = ordered_courses(split["course"], split["labels"], 2901, 1)
    positions, states = rollout(raw, x)
    trace = [{"step": step, "station": (step - 1) // FEATURE_COUNT if step else -1,
              "position": float(state[0]), "velocity": float(state[1]),
              "angle": float(state[2]), "spin": float(state[3])}
             for step, state in enumerate(states[:, 0])]
    write_csv(output / "trajectory.csv", trace)
    endpoints = []
    for station, p in enumerate(positions[0]):
        label = int(y[0, station])
        prediction = int(np.argmin(abs(p - CENTERS)))
        endpoints.append({"station": station, "wine_row": int(split["indices"][order[0, station]]),
                          "true_class": label, "nearest_class": prediction,
                          "position": float(p), "correct_hit": bool(abs(p - CENTERS[label]) <= HIT_RADIUS)})
    write_csv(output / "obstacle_hits.csv", endpoints)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except ImportError:
        print("matplotlib unavailable: CSV trace saved; figure skipped.", flush=True)
        return
    plt.rcParams["svg.hashsalt"] = "kinetra-dataset-course"
    count = 12
    fig, ax = plt.subplots(figsize=(13, 5))
    for station in range(count):
        time_step = (station + 1) * FEATURE_COUNT
        for cls, center in enumerate(CENTERS):
            ax.add_patch(Rectangle((time_step - 0.8, center - HIT_RADIUS), 1.6, 2 * HIT_RADIUS,
                                   facecolor="#FFD43B" if cls == y[0, station] else "white",
                                   edgecolor="black", linewidth=1.2, zorder=2))
        p = positions[0, station]
        hit = abs(p - CENTERS[y[0, station]]) <= HIT_RADIUS
        ax.plot(time_step, p, "o" if hit else "x", color="black" if hit else "#B51F24", markersize=7, zorder=4)
    ax.plot(np.arange(count * FEATURE_COUNT + 1), states[:count * FEATURE_COUNT + 1, 0, 0],
            color="#1565C0", linewidth=2, label="Continuous ball trajectory", zorder=3)
    ax.set(title=f"Wine course: first 12 unseen obstacles | {selected_mode}, seed {selected_seed}",
           xlabel="Obstacle number (13 input-force steps between obstacles)", ylabel="Lateral position / answer zone",
           ylim=(-2.0, 2.0), xlim=(0, count * FEATURE_COUNT + 3))
    ax.set_yticks(CENTERS, ["Class 0", "Class 1", "Class 2"])
    ax.set_xticks(np.arange(1, count + 1) * FEATURE_COUNT, [str(i + 1) for i in range(count)])
    ax.grid(axis="y", alpha=0.2)
    ax.legend(loc="upper left")
    fig.text(0.5, 0.015, "All three pads are always present. Yellow = correct answer shown only for scoring. "
             "Black dot = correct hit; red x = miss. No state resets between obstacles.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    svg_path = output / "course_preview.svg"
    fig.savefig(svg_path, metadata={"Date": None})
    svg_path.write_text("\n".join(line.rstrip() for line in svg_path.read_text(encoding="utf-8").splitlines()) + "\n", encoding="utf-8")
    fig.savefig(output / "course_preview.png", dpi=130)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--output", type=Path, default=HERE / "results")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    data, splits, mean, scale = load_splits()
    train, val, test = (splits[name] for name in ["train", "validation", "test"])
    print("Wine dataset-as-course | split", {k: len(v["labels"]) for k, v in splits.items()}, flush=True)
    specs, initial, fitted, chosen_epochs, history = train_models(
        train["course"], train["labels"], val["course"], val["labels"], epochs=args.epochs)
    write_csv(output / "learning_curve.csv", history)
    # All models and checkpoints are fixed before any test label is scored.
    models = []
    rows = []
    for i, ((mode, seed), raw) in enumerate(zip(specs, fitted)):
        row = {"mode": mode, "seed": seed, "selected_epoch": int(chosen_epochs[i])}
        for prefix, subset in [("train", train), ("validation", val), ("test", test)]:
            stats = summarize_orders(raw, subset["course"], subset["labels"])
            row.update({prefix + "_" + key: value for key, value in stats.items()})
        untrained = summarize_orders(initial[i], test["course"], test["labels"])
        row["untrained_test_correct_hit_rate"] = untrained["correct_hit_rate"]
        rows.append(row)
        values, _ = physical_parameters(raw)
        models.append({"mode": mode, "seed": seed, "selected_epoch": int(chosen_epochs[i]),
                       "raw_parameters": raw.tolist(),
                       "physical_parameters": dict(zip(PARAMETER_NAMES, values[0].tolist()))})
        print(f"{mode:13s} seed {seed}: test hits {row['test_correct_hit_rate']:.3f} | "
              f"class accuracy {row['test_class_accuracy']:.3f} | "
              f"won {row['test_fully_won_courses']}/20 courses", flush=True)
    write_csv(output / "per_seed_results.csv", rows)
    # Controls use the same split; they do not set the Kinetra force encoding.
    model = LogisticRegression(max_iter=2000, C=1.0)
    model.fit((train["X"] - mean) / scale, train["labels"])
    logistic_accuracy = float(model.score((test["X"] - mean) / scale, test["labels"]))
    majority_class = int(np.argmax(np.bincount(train["labels"], minlength=3)))
    majority_accuracy = float(np.mean(test["labels"] == majority_class))
    aggregates = {}
    for mode in ["launch_only", "shared_motion"]:
        group = [r for r in rows if r["mode"] == mode]
        aggregates[mode] = {key: float(np.mean([r[key] for r in group])) for key in [
            "train_correct_hit_rate", "validation_correct_hit_rate", "test_correct_hit_rate",
            "test_class_accuracy", "untrained_test_correct_hit_rate", "test_order_sensitive_fraction",
        ]}
        aggregates[mode]["test_hit_rate_seed_range"] = [min(r["test_correct_hit_rate"] for r in group),
                                                        max(r["test_correct_hit_rate"] for r in group)]
        aggregates[mode]["test_fully_won_courses"] = int(sum(r["test_fully_won_courses"] for r in group))
    # Choose illustrative model only from validation, never from test results.
    selected = max(range(len(rows)), key=lambda i: (rows[i]["validation_correct_hit_rate"], -rows[i]["validation_mse"]))
    selected_model = models[selected]
    selected_raw = fitted[selected]
    shuffled_labels = np.random.default_rng(4101).permutation(test["labels"])
    shuffled_stats = summarize_orders(selected_raw, test["course"], shuffled_labels)
    zero_stats = summarize_orders(selected_raw, np.zeros_like(test["course"]), test["labels"])
    # Audit the primary forward pass and publish full data provenance.
    repeat_a = rollout(selected_raw, test["course"])[1]
    repeat_b = rollout(selected_raw, test["course"])[1]
    midpoint = len(test["course"]) // 2
    first = rollout(selected_raw, test["course"][:midpoint])[1]
    second = rollout(selected_raw, test["course"][midpoint:], initial_state=first[-1])[1]
    assert np.array_equal(repeat_a, repeat_b)
    assert np.array_equal(repeat_a, np.concatenate((first, second[1:]), axis=0))
    manifest = {
        "experiment": "Wine dataset-as-course: continuous state, label-free dynamics",
        "base_commit": "ca6d99428274fe6bea81ed7019c4d360d6bb4657",
        "split_seed": SPLIT_SEED, "training_seeds": SEEDS, "epochs": args.epochs,
        "split_indices": {k: v["indices"].tolist() for k, v in splits.items()},
        "scaler_mean_train_only": mean.tolist(), "scaler_scale_train_only": scale.tolist(),
        "feature_names": list(data.feature_names), "class_names": list(data.target_names),
        "dataset_sha256": hashlib.sha256(data.data.astype("<f8").tobytes() + data.target.astype("<i8").tobytes()).hexdigest(),
        "input_encoder": "0.25 * tanh(((x - train_mean) / train_std) / 2.5); 13 sequential forces",
        "target_centers": CENTERS.tolist(), "correct_hit_radius": HIT_RADIUS,
        "modes": {"launch_only": "4 trainable launch settings; all later dynamics fixed",
                  "shared_motion": "same 4 launch settings plus 6 shared motion parameters"},
        "optimizer": {"name": "Adam with exact NumPy reverse-mode gradient", "learning_rate": 0.025,
                      "full_courses_per_update": 2, "global_gradient_norm_clip": 2.0},
        "checkpoint_selection": "Every 10 epochs incl. epoch 0; best validation hit rate over 4 fixed shuffled courses, then lowest MSE",
        "test_protocol": "20 fixed shuffled orders of the same 36 unseen rows; no training or checkpoint selection on test",
        "success_goal": "Hit the correct pad at every obstacle in a continuous unseen course",
        "aggregates": aggregates,
        "controls": {"majority_class": majority_class, "majority_test_class_accuracy": majority_accuracy,
                     "logistic_regression_test_class_accuracy": logistic_accuracy,
                     "selected_model_shuffled_label_hit_rate": shuffled_stats["correct_hit_rate"],
                     "selected_model_zero_force_hit_rate": zero_stats["correct_hit_rate"]},
        "audits": {"deterministic_replay": True, "chunked_state_carry_exact": True,
                   "labels_not_accepted_by_encoder_or_rollout": True, "test_labels_used_for_selection": False},
        "preview_model": {"mode": selected_model["mode"], "seed": selected_model["seed"],
                          "selection": "validation hit rate over 20 fixed orders, then validation MSE"},
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "sklearn": sklearn_version},
        "limitations": ["Differentiable toy dynamics and soft-contact surrogate; not real rigid-body physics.",
                        "Fixed class-to-position mapping imposes an arbitrary ordering of classes.",
                        "Only one Wine split; five optimizer seeds are not five independent datasets.",
                        "Repeated test orders reuse the same 36 rows, so they are not independent test samples.",
                        "Input forces are fixed, bounded and unlearned; no separate neural feature extractor.",
                        "No label-conditioned forces or collisions. All answer pads always exist.",
                        "No resets, gradient truncation, per-row parameters, test-time optimization or memorized route.",
                        "Shared-motion mode trains dynamics as well as launch; launch-only mode tests the stricter starting-trajectory claim."],
    }
    (output / "summary.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (output / "models.json").write_text(json.dumps(models, indent=2) + "\n", encoding="utf-8")
    save_launch_sensitivity(output, initial[0], test["course"])
    save_preview(output, selected_raw, test, selected_model["mode"], selected_model["seed"])
    print(json.dumps({"aggregates": aggregates, "controls": manifest["controls"]}, indent=2), flush=True)
    print("Results written to", output, flush=True)


if __name__ == "__main__":
    main()
