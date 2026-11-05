"""Guard against label leakage, resets, and incorrect trajectory gradients."""
import importlib.util
import inspect
from pathlib import Path
import unittest

import numpy as np

PATH = Path(__file__).resolve().parents[1] / "experiments/dataset-course/kinetra_dataset_course.py"
spec = importlib.util.spec_from_file_location("dataset_course", PATH)
k = importlib.util.module_from_spec(spec)
spec.loader.exec_module(k)


class DatasetCourseTests(unittest.TestCase):
    def setUp(self):
        self.raw = k.initial_parameters(701)
        self.course = np.random.default_rng(44).normal(0, 0.15, (6, 13))
        self.labels = np.array([0, 2, 1, 2, 0, 1])

    def test_no_labels_in_forward_interfaces(self):
        self.assertEqual(list(inspect.signature(k.encode_inputs).parameters), ["inputs", "mean", "scale", "config"])
        self.assertEqual(list(inspect.signature(k.rollout).parameters), ["raw", "course", "initial_state", "config"])
        before = k.rollout(self.raw, self.course)[1]
        k.loss_and_gradient(self.raw, self.course, self.labels)
        k.loss_and_gradient(self.raw, self.course, 2 - self.labels)
        np.testing.assert_array_equal(before, k.rollout(self.raw, self.course)[1])

    def test_full_run_equals_chunked_run_with_all_state_carried(self):
        positions, full = k.rollout(self.raw, self.course)
        p1, s1 = k.rollout(self.raw, self.course[:3])
        p2, s2 = k.rollout(self.raw, self.course[3:], initial_state=s1[-1])
        np.testing.assert_array_equal(full, np.concatenate([s1, s2[1:]]))
        np.testing.assert_array_equal(positions, np.concatenate([p1, p2], axis=1))
        reset_positions, _ = k.rollout(self.raw, self.course[3:])
        self.assertGreater(np.max(abs(p2 - reset_positions)), 1e-6)

    def test_reverse_gradient_matches_finite_difference_across_stations(self):
        _, analytic = k.loss_and_gradient(self.raw, self.course, self.labels)
        numerical = np.zeros(10)
        for j in range(10):
            delta = np.zeros(10)
            delta[j] = 1e-5
            plus = k.loss_and_gradient(self.raw + delta, self.course, self.labels)[0]
            minus = k.loss_and_gradient(self.raw - delta, self.course, self.labels)[0]
            numerical[j] = ((plus - minus) / 2e-5).item()
        np.testing.assert_allclose(analytic[0], numerical, rtol=2e-5, atol=2e-7)

    def test_batched_courses_are_independent_whole_runs(self):
        raw = np.array([self.raw, k.initial_parameters(702)])
        course = np.array([self.course, self.course[::-1]])
        labels = np.array([self.labels, self.labels[::-1]])
        batched_p, batched_s = k.rollout(raw, course)
        batch_loss, batch_grad = k.loss_and_gradient(raw, course, labels)
        for j in range(2):
            p, s = k.rollout(raw[j], course[j])
            loss, grad = k.loss_and_gradient(raw[j], course[j], labels[j])
            np.testing.assert_array_equal(batched_p[j], p[0])
            np.testing.assert_array_equal(batched_s[:, j], s[:, 0])
            np.testing.assert_allclose(batch_loss[j], loss[0])
            np.testing.assert_allclose(batch_grad[j], grad[0])

    def test_replay_determinism_and_input_effect(self):
        _, first = k.rollout(self.raw, self.course)
        _, again = k.rollout(self.raw, self.course)
        np.testing.assert_array_equal(first, again)
        changed = self.course.copy()
        changed[0, 0] += 0.2
        self.assertFalse(np.array_equal(first, k.rollout(self.raw, changed)[1]))

    def test_hit_requires_correct_pad_and_radius(self):
        score = k.score_positions(np.array([[-1.0, 0.0, 1.0, -0.5]]), np.array([0, 0, 2, 1]))
        self.assertEqual(score["correct_hit_rate"][0], 0.5)
        self.assertEqual(score["any_contact_rate"][0], 0.75)
        self.assertFalse(score["all_obstacles_hit"][0])

    def test_split_disjoint_and_scaling_uses_only_train(self):
        data, splits, mean, scale = k.load_splits()
        groups = [set(x["indices"]) for x in splits.values()]
        self.assertEqual([len(x) for x in groups], [106, 36, 36])
        self.assertEqual(len(set.union(*groups)), 178)
        self.assertFalse(groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
        np.testing.assert_array_equal(mean, data.data[splits["train"]["indices"]].mean(axis=0))
        np.testing.assert_array_equal(scale, data.data[splits["train"]["indices"]].std(axis=0))

    def test_encoder_preserves_feature_order_and_rejects_nonfinite_values(self):
        mean, scale = np.zeros(13), np.ones(13)
        x = np.zeros((2, 13))
        x[0, 4] = 1
        encoded = k.encode_inputs(x, mean, scale)
        self.assertEqual(np.count_nonzero(encoded), 1)
        self.assertGreater(encoded[0, 4], 0)
        with self.assertRaises(ValueError):
            k.encode_inputs(np.full((1, 13), np.nan), mean, scale)

    def test_soft_contact_derivative(self):
        p = np.array([-1.1, -0.55, 0.01, 0.7, 1.5])
        _, derivative = k.contact(p)
        numeric = (k.contact(p + 1e-6)[0] - k.contact(p - 1e-6)[0]) / 2e-6
        np.testing.assert_allclose(derivative, numeric, atol=1e-9)


if __name__ == "__main__":
    unittest.main()
