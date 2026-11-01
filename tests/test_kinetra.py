"""Regression checks for Kinetra's reproducible ball-and-track experiment."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ball_track_transfer_test as k


class KinetraTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.track = k.make_track(101)
        cls.learned, cls.fitness = k.train(cls.track)
        cls.baseline = (0.0,) * 5

    def test_deterministic_replay(self):
        self.assertEqual(k.simulate(self.track, self.learned),
                         k.simulate(self.track, self.learned))

    def test_starting_conditions_affect_path(self):
        original = k.simulate(self.track, self.learned)
        changed = k.simulate(self.track, self.learned,
                             initial_angle=0.2, initial_spin=0.1)
        self.assertNotEqual(original[2], changed[2])
        self.assertEqual(changed, k.simulate(self.track, self.learned,
                                             initial_angle=0.2, initial_spin=0.1))

    def test_improved_training_track(self):
        self.assertLess(k.objective(self.track, self.learned),
                        k.objective(self.track, self.baseline))

    def test_transfer_beats_memorized_route(self):
        commands = [(step[8], step[9]) for step in k.simulate(self.track, self.learned)[2]]
        holdout = [k.make_track(1000 + j) for j in range(30)]
        mse_transfer, miss_transfer = k.metrics(holdout, self.learned)
        miss_baseline = k.metrics(holdout, self.baseline)[1]
        miss_replay = sum(k.simulate(t, self.learned, commands=commands)[1]
                          for t in holdout) / len(holdout)
        self.assertLess(miss_transfer, miss_baseline)
        self.assertLess(miss_transfer, miss_replay)
        self.assertLess(mse_transfer, 0.1)


if __name__ == '__main__':
    unittest.main()
