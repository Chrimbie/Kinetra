#!/usr/bin/env python3
"""Run the unchanged continuous-course mechanism on all 1,797 8x8 digits.

This is scikit-learn's bundled load_digits dataset, not MNIST. Defaults reproduce
the registered experiment. Wine and its published results remain separate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import sklearn
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import kinetra_dataset_course as core

CONFIG = core.CourseConfig(feature_count=64, class_count=10)
SPLIT_SEED = 20261105


def load_splits():
    data = load_digits()
    indices = np.arange(len(data.target))
    train_val, test = train_test_split(indices, test_size=0.20, random_state=SPLIT_SEED, stratify=data.target)
    train, val = train_test_split(train_val, test_size=0.25, random_state=SPLIT_SEED + 1, stratify=data.target[train_val])
    mean = data.data[train].mean(axis=0)
    scale = data.data[train].std(axis=0)
    scale[scale == 0] = 1.0
    splits = {}
    for name, ix in [("train", train), ("validation", val), ("test", test)]:
        splits[name] = {"indices": ix, "X": data.data[ix], "labels": data.target[ix],
                        "course": core.encode_inputs(data.data[ix], mean, scale, CONFIG)}
    return data, splits, mean, scale


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--output", type=Path, default=HERE / "results")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    data, splits, mean, scale = load_splits()
    train, val, test = (splits[name] for name in ["train", "validation", "test"])
    print("8x8 digits: 64 pixel forces, 10 answer pads, continuous motion", flush=True)
    print("split:", {name: len(s["labels"]) for name, s in splits.items()}, flush=True)
    specs, initial, fitted, epochs, history = core.train_models(
        train["course"], train["labels"], val["course"], val["labels"], epochs=args.epochs, config=CONFIG)
    core.write_csv(args.output / "learning_curve.csv", history)
    models = []
    for i, ((mode, seed), raw) in enumerate(zip(specs, fitted)):
        physical, _ = core.physical_parameters(raw)
        models.append({"mode": mode, "seed": seed, "selected_epoch": int(epochs[i]),
                       "raw_parameters": raw.tolist(),
                       "physical_parameters": dict(zip(core.PARAMETER_NAMES, physical[0].tolist()))})
    (args.output / "models.json").write_text(json.dumps(models, indent=2) + "\n", encoding="utf-8")
    print("Training complete. Models saved and frozen before test scoring.", flush=True)
    rows = []
    # One row per optimizer seed/mode. Orders reuse rows and are not independent
    # statistical replicates. Reordering tests whether prior motion changes a prediction.
    for i, (model, raw) in enumerate(zip(models, fitted)):
        row = {"mode": model["mode"], "seed": model["seed"], "selected_epoch": model["selected_epoch"]}
        for name, subset in splits.items():
            metrics = core.summarize_orders(raw, subset["course"], subset["labels"], config=CONFIG)
            row.update({name + "_" + key: value for key, value in metrics.items()})
        baseline = core.summarize_orders(initial[i], test["course"], test["labels"], config=CONFIG)
        row["untrained_test_correct_hit_rate"] = baseline["correct_hit_rate"]
        row["untrained_test_class_accuracy"] = baseline["class_accuracy"]
        row["untrained_test_fully_won_courses"] = baseline["fully_won_courses"]
        rows.append(row)
        print(f"{model['mode']:13s} seed {model['seed']}: test hits {row['test_correct_hit_rate']:.4f}, "
              f"accuracy {row['test_class_accuracy']:.4f}, wins {row['test_fully_won_courses']}/20", flush=True)
    core.write_csv(args.output / "per_seed_results.csv", rows)
    aggregates = {}
    for mode in ["launch_only", "shared_motion"]:
        group = [r for r in rows if r["mode"] == mode]
        keys = ["train_correct_hit_rate", "validation_correct_hit_rate", "test_correct_hit_rate",
                "test_class_accuracy", "test_order_sensitive_fraction", "untrained_test_correct_hit_rate",
                "untrained_test_class_accuracy"]
        aggregates[mode] = {key: float(np.mean([r[key] for r in group])) for key in keys}
        aggregates[mode]["test_hit_rate_seed_range"] = [float(min(r["test_correct_hit_rate"] for r in group)),
                                                        float(max(r["test_correct_hit_rate"] for r in group))]
        aggregates[mode]["test_fully_won_courses"] = sum(r["test_fully_won_courses"] for r in group)
    # A standard classifier sees the same training-only scaling and split.
    logistic = LogisticRegression(C=1.0, max_iter=2000)
    logistic.fit((train["X"] - mean) / scale, train["labels"])
    logistic_predictions = logistic.predict((test["X"] - mean) / scale)
    majority = int(np.bincount(train["labels"], minlength=10).argmax())
    selected = max(range(len(rows)), key=lambda i: (rows[i]["validation_correct_hit_rate"], -rows[i]["validation_mse"]))
    chosen, raw = models[selected], fitted[selected]
    shuffled_labels = np.random.default_rng(4101).permutation(test["labels"])
    shuffled = core.summarize_orders(raw, test["course"], shuffled_labels, config=CONFIG)
    zero = core.summarize_orders(raw, np.zeros_like(test["course"]), test["labels"], config=CONFIG)
    # Publish predicted-class counts and a confusion matrix over all 20 orders
    # for the validation-selected illustration, not a best-test model.
    courses, labels, _ = core.ordered_courses(test["course"], test["labels"], 2901, 20)
    positions, _ = core.rollout(raw, courses, config=CONFIG)
    predictions = np.argmin(abs(positions[:, :, None] - CONFIG.centers), axis=2)
    confusion = np.zeros((10, 10), dtype=int)
    np.add.at(confusion, (labels.ravel(), predictions.ravel()), 1)
    core.write_csv(args.output / "confusion_matrix.csv", [
        {"true_class": i, **{f"predicted_{j}": int(confusion[i, j]) for j in range(10)}} for i in range(10)])
    # Exact state-carry audit on a full test course, including all four state variables.
    _, complete = core.rollout(raw, test["course"], config=CONFIG)
    halfway = len(test["course"]) // 2
    _, prefix = core.rollout(raw, test["course"][:halfway], config=CONFIG)
    _, suffix = core.rollout(raw, test["course"][halfway:], initial_state=prefix[-1], config=CONFIG)
    np.testing.assert_array_equal(complete, np.concatenate((prefix, suffix[1:]), axis=0))
    manifest = {
        "experiment": "Digits dataset-as-course, unchanged motion mechanism",
        "base_commit": "ae88865bb5332ad04fa7c93f93e0d548d2552fd4",
        "dataset": "scikit-learn load_digits: 1797 8x8 images, 10 classes; not MNIST",
        "dataset_sha256": hashlib.sha256(data.data.astype("<f8").tobytes() + data.target.astype("<i8").tobytes()).hexdigest(),
        "split_seed": SPLIT_SEED, "split_indices": {name: s["indices"].tolist() for name, s in splits.items()},
        "split_class_counts": {name: np.bincount(s["labels"], minlength=10).tolist() for name, s in splits.items()},
        "scaler_mean_train_only": mean.tolist(), "scaler_scale_train_only": scale.tolist(),
        "input_order": "64 pixels, row-major 8x8 image order, no edges, pooling or learned feature extractor",
        "input_encoder": "0.25*tanh(((x-train_mean)/train_std)/2.5), exactly the Wine rule",
        "feature_count": 64, "class_count": 10,
        "target_centers": CONFIG.centers.tolist(), "hit_radius": CONFIG.hit_radius,
        "contact_width": CONFIG.contact_width,
        "geometry_rule": "Centers equally spaced in [-1,1]; hit radius 0.32*spacing, contact width 0.30*spacing",
        "unchanged": ["Four motion-state equations, constants, shared physical parameter transforms and initializations",
                      "4 launch parameters or 4 launch + 6 global motion parameters",
                      "Training-only input scaling followed by fixed tanh force encoding",
                      "Full continuous-course state carry and untruncated reverse-mode gradient",
                      "Mean squared distance to the correct pad center",
                      "250 Adam updates (default), lr 0.025, two shuffled full courses/update, gradient clip 2",
                      "Five launch-initialization seeds; same six physics initial values for every seed",
                      "Validation-only checkpoint selection; frozen held-out test evaluation"],
        "training_seeds": core.SEEDS, "epochs": args.epochs,
        "checkpoint_selection": "Every 10 epochs incl epoch 0, highest validation hit rate over 4 fixed orders, then lowest MSE",
        "test_protocol": "20 fixed orders of the same 360 held-out images; not 7200 independent examples",
        "aggregates": aggregates,
        "controls": {"majority_class": majority,
                     "majority_test_class_accuracy": float(np.mean(test["labels"] == majority)),
                     "logistic_test_class_accuracy": float(np.mean(logistic_predictions == test["labels"])),
                     "logistic_correct": int(np.sum(logistic_predictions == test["labels"])),
                     "selected_model_shuffled_label_correct_hit_rate": shuffled["correct_hit_rate"],
                     "selected_model_zero_force_correct_hit_rate": zero["correct_hit_rate"],
                     "selected_model_prediction_counts_20_orders": np.bincount(predictions.ravel(), minlength=10).tolist()},
        "preview_model": {"mode": chosen["mode"], "seed": chosen["seed"],
                          "selection": "highest validation hit rate over 20 orders, then lowest validation MSE"},
        "audits": {"full_course_matches_chunked_state_carry_exactly": True,
                   "labels_not_accepted_by_encoder_or_rollout": True, "test_used_for_selection": False},
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "sklearn": sklearn.__version__},
        "limitations": ["Only bundled 8x8 digits, not a large image benchmark or MNIST",
                        "One row-stratified split, not a writer-disjoint evaluation",
                        "Five initialization seeds and repeated course orders reuse the same split",
                        "Ten pads in the same lateral range need narrower hit tolerances than Wine",
                        "More steps per obstacle give damping more time to erase earlier inputs",
                        "A harder dataset and longer course do not fix the earlier Wine failure",
                        "Ordinal target positions and squared-distance loss are unchanged",
                        "Toy smooth-contact dynamics; not real rigid-body physics"],
    }
    (args.output / "summary.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    core.save_launch_sensitivity(args.output, initial[0], test["course"], config=CONFIG)
    core.save_preview(args.output, raw, test, chosen["mode"], chosen["seed"], config=CONFIG, dataset_name="8x8 digits")
    print(json.dumps({"aggregates": aggregates, "controls": manifest["controls"]}, indent=2), flush=True)
    print(f"Done in {time.monotonic()-start:.1f}s. Outputs: {args.output}", flush=True)


if __name__ == "__main__":
    main()
