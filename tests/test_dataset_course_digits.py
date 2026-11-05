"""Ten-class mapping and continuous-state checks, plus Wine compatibility."""
import csv
import importlib.util
import json
from pathlib import Path
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "experiments/dataset-course/digits/run_digits.py"
spec = importlib.util.spec_from_file_location("dataset_course_digits", PATH)
d = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d)
k = d.core


class DigitsCourseTests(unittest.TestCase):
    def test_saved_wine_trajectory_is_unchanged(self):
        base = ROOT / "experiments/dataset-course/results"
        summary = json.loads((base / "summary.json").read_text())
        models = json.loads((base / "models.json").read_text())
        chosen = next(m for m in models if all(m[key] == summary["preview_model"][key] for key in ["mode", "seed"]))
        _, splits, _, _ = k.load_splits()
        x, _, _ = k.ordered_courses(splits["test"]["course"], splits["test"]["labels"], 2901, 1)
        _, state = k.rollout(np.array(chosen["raw_parameters"]), x)
        with (base / "trajectory.csv").open() as handle:
            old = np.array([[float(row[key]) for key in ["position", "velocity", "angle", "spin"]]
                            for row in csv.DictReader(handle)])
        np.testing.assert_array_equal(state[:, 0], old)

    def test_ten_nonoverlapping_reachable_pads(self):
        c = d.CONFIG
        self.assertEqual(c.feature_count, 64)
        self.assertEqual(len(c.centers), 10)
        self.assertTrue(np.all(np.diff(c.centers) > 2 * c.hit_radius))
        self.assertTrue(np.all(abs(c.centers) + c.hit_radius < 2))
        score = k.score_positions(c.centers[None, :], np.arange(10), config=c)
        self.assertEqual(score["correct_hit_rate"][0], 1)
        self.assertEqual(score["class_accuracy"][0], 1)
        self.assertTrue(score["all_obstacles_hit"][0])
        with self.assertRaises(ValueError):
            k.score_positions(np.array([[0.0]]), np.array([10]), config=c)

    def test_64_step_state_carry_and_label_independence(self):
        raw = k.initial_parameters(704)
        course = np.random.default_rng(72).normal(0, 0.15, (4, 64))
        _, full = k.rollout(raw, course, config=d.CONFIG)
        _, a = k.rollout(raw, course[:2], config=d.CONFIG)
        _, b = k.rollout(raw, course[2:], initial_state=a[-1], config=d.CONFIG)
        np.testing.assert_array_equal(full, np.concatenate([a, b[1:]]))
        k.loss_and_gradient(raw, course, np.array([0, 1, 8, 9]), config=d.CONFIG)
        k.loss_and_gradient(raw, course, np.array([9, 8, 1, 0]), config=d.CONFIG)
        np.testing.assert_array_equal(full, k.rollout(raw, course, config=d.CONFIG)[1])

    def test_ten_class_gradient_matches_finite_differences(self):
        raw = k.initial_parameters(702)
        course = np.random.default_rng(321).normal(0, 0.15, (3, 64))
        labels = np.array([0, 9, 4])
        _, grad = k.loss_and_gradient(raw, course, labels, config=d.CONFIG)
        numerical = np.zeros(10)
        for j in range(10):
            delta = np.zeros(10); delta[j] = 1e-5
            a = k.loss_and_gradient(raw + delta, course, labels, config=d.CONFIG)[0]
            b = k.loss_and_gradient(raw - delta, course, labels, config=d.CONFIG)[0]
            numerical[j] = ((a - b) / 2e-5).item()
        np.testing.assert_allclose(grad[0], numerical, rtol=2e-5, atol=2e-7)

    def test_digits_split_scaling_and_image_pixel_order(self):
        data, splits, mean, scale = d.load_splits()
        groups = [set(s["indices"]) for s in splits.values()]
        self.assertEqual([len(s) for s in groups], [1077, 360, 360])
        self.assertEqual(len(set.union(*groups)), 1797)
        self.assertFalse(groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
        np.testing.assert_array_equal(data.images.reshape(-1, 64), data.data)
        np.testing.assert_array_equal(mean, data.data[splits["train"]["indices"]].mean(axis=0))
        expected = data.data[splits["train"]["indices"]].std(axis=0)
        expected[expected == 0] = 1
        np.testing.assert_array_equal(scale, expected)
        self.assertTrue(np.isfinite(splits["test"]["course"]).all())


if __name__ == "__main__":
    unittest.main()
