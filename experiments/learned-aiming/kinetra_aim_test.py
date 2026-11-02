#!/usr/bin/env python3
"""Kinetra learned aiming test.

The pass-through contact test (2026-11-01) verified
the MECHANISM with a hand-set controller: the ball can
be steered to a chosen part, and the recorded part is
the answer. Its honest gap: nothing was LEARNED. This
test closes that gap: can a controller LEARN to aim
for a target part, given the target as input?

SETUP (same as the pass test):
- Track width 3.0, lateral range [-1.5, 1.5].
- Problem P1: center 0, width 2, 5 parts (each 0.4
  wide). Fits (0 + 1.0 <= 1.5).
- 10 physics steps per pass (the room to move
  around), Kinetra's constants (angle 0.48, spin
  0.43, velocity 0.48 decay, 0.83 sin(angle), 0.23
  spin, plus push). Fresh ball state per pass.

CONTROLLER (Mia's form, mirroring the original
Kinetra policy with gap_error redefined as target
error):
- angle_target = clip(p*(target - y) - d*velocity
  - wind*push, +-0.9)
- spin_target = clip(s*(target - y) - swind*push,
  +-0.9)
- Five weights (p, d, wind, s, swind). The hand-set
  pass-test controller (0.6, 0.8, 0, 0.3, 0) sits
  inside this family, so a good solution EXISTS in
  the search space; the question is whether random
  search FINDS one.

TRAINING (mirrors the original Kinetra search):
- Derivative-free random search, budget 4000,
  seed 12345. Every 10th proposal global
  uniform(-2.4, 2.4); otherwise local gaussian with
  scale 0.9*(1-j/budget)+0.025, weights clipped
  to +-3. Starts at zero weights.
- Each trial is scored on 20 passes: targets cycle
  balanced [0,1,2,3,4] x 4, push seeds fixed
  (Random(2000 + pass)) so the objective is
  deterministic given weights, push sigma 0.09.
- Objective mirrors the original: mean squared
  error between contact y and target center, plus
  0.12 * miss fraction (miss = recorded part !=
  target part, including contact outside the span).

EVALUATION (fresh noise, never seen in training):
- 100 passes: 20 per target part, balanced, push
  seeds Random(9000 + ...), sigma 0.09. Same again
  at sigma 0.05 as an easier arm.
- References: zero-weight (untrained) controller on
  the same fresh passes; hand-set pass-test rates
  (0.95/1.00/1.00/1.00/1.00 at sigma 0.05) quoted
  for context, not as a bar to beat.

PRE-REGISTERED CRITERION (Mia's, fixed before
running): PASS if overall aimed-part hit rate on
fresh seeds at sigma 0.09 is >= 0.80 AND every
part's hit rate is >= 0.60 (no part unlearnable).
Rates are reported exactly; a failed criterion is
reported as failed, not tuned away.

WHAT THIS TESTS: learning to aim under fresh noise.
It does NOT test generalizing to new targets (all
5 parts appear in training) or to continuous aim
points (targets are part centers only). Those are
follow-ups.

Python standard library only. Run:
python kinetra_aim_test.py
Writes kinetra_aim_results.csv alongside the script.
"""

import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path

TRACK_WIDTH = 3.0
TRACK_HALF = TRACK_WIDTH / 2.0
STEPS_PER_PROBLEM = 10
BUDGET = 4000
SEARCH_SEED = 12345
TRAIN_PUSH_SIGMA = 0.09
OUT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Problem:
    pid: str
    center: float
    width: float
    n_parts: int

    @property
    def lo(self):
        return self.center - self.width / 2.0

    @property
    def hi(self):
        return self.center + self.width / 2.0

    def fits(self):
        return (self.width <= TRACK_WIDTH
                and abs(self.center) + self.width / 2.0 <= TRACK_HALF)

    def part_of(self, y):
        if y < self.lo or y > self.hi:
            return None
        part_w = self.width / self.n_parts
        return min(int((y - self.lo) / part_w), self.n_parts - 1)

    def part_center(self, part):
        part_w = self.width / self.n_parts
        return self.lo + (part + 0.5) * part_w


@dataclass
class Ball:
    y: float = 0.0
    velocity: float = 0.0
    angle: float = 0.0
    spin: float = 0.0


def clip(x, a, b):
    return min(b, max(a, x))


def step(ball, angle_target, spin_target, push):
    ball.angle += 0.48 * (angle_target - ball.angle)
    ball.spin += 0.43 * (spin_target - ball.spin)
    ball.velocity = (0.48 * ball.velocity + 0.83 * math.sin(ball.angle)
                     + 0.23 * ball.spin + push)
    ball.y += ball.velocity


def run_pass(problem, weights, target_part, rng, push_sigma):
    """One pass: 10 maneuver steps, then contact. Returns (y, part)."""
    ball = Ball()
    aim = problem.part_center(target_part)
    p, d, wind, s, swind = weights
    for _ in range(STEPS_PER_PROBLEM):
        push = rng.gauss(0, push_sigma) if push_sigma > 0 else 0.0
        angle_target = clip(p * (aim - ball.y) - d * ball.velocity
                            - wind * push, -0.9, 0.9)
        spin_target = clip(s * (aim - ball.y) - swind * push,
                           -0.9, 0.9)
        step(ball, angle_target, spin_target, push)
    return ball.y, problem.part_of(ball.y)


def objective(problem, weights, targets, seed_base, push_sigma):
    sq_err = 0.0
    misses = 0
    for i, target in enumerate(targets):
        rng = random.Random(seed_base + i)
        y, part = run_pass(problem, weights, target, rng, push_sigma)
        aim = problem.part_center(target)
        sq_err += (y - aim) ** 2
        misses += part != target
    n = len(targets)
    return sq_err / n + 0.12 * misses / n


def evaluate(problem, weights, n_per_part, seed_base, push_sigma):
    """Fresh-seed evaluation. Returns per-part hit rates + overall."""
    per_part = {}
    total_hits = 0
    total = 0
    for part in range(problem.n_parts):
        hits = 0
        for k in range(n_per_part):
            rng = random.Random(seed_base + part * 1000 + k)
            _, got = run_pass(problem, weights, part, rng, push_sigma)
            hits += got == part
        per_part[part] = hits / n_per_part
        total_hits += hits
        total += n_per_part
    return per_part, total_hits / total


def main():
    p1 = Problem('P1', 0.0, 2.0, 5)
    assert p1.fits(), 'P1 must fit the track'

    train_targets = [part for _ in range(4)
                     for part in range(p1.n_parts)]

    r = random.Random(SEARCH_SEED)
    weights = (0.0,) * 5
    best = objective(p1, weights, train_targets, 2000,
                     TRAIN_PUSH_SIGMA)
    checkpoints = [(0, best)]
    for j in range(BUDGET):
        if j % 10 == 0:
            proposal = tuple(r.uniform(-2.4, 2.4) for _ in weights)
        else:
            scale = 0.9 * (1 - j / BUDGET) + 0.025
            proposal = tuple(clip(w + r.gauss(0, scale), -3, 3)
                             for w in weights)
        result = objective(p1, proposal, train_targets, 2000,
                           TRAIN_PUSH_SIGMA)
        if result < best:
            weights, best = proposal, result
        if (j + 1) % 500 == 0:
            checkpoints.append((j + 1, best))
    print(f'LEARNED WEIGHTS (p, d, wind, s, swind): '
          f'{tuple(round(w, 3) for w in weights)}')
    print('TRAINING CHECKPOINTS (trial, best objective):')
    for trial, value in checkpoints:
        print(f'  {trial:5d}  {value:.4f}')

    rows = []
    for sigma, label in [(0.09, 'sigma 0.09'), (0.05, 'sigma 0.05')]:
        per_part, overall = evaluate(p1, weights, 20, 9000, sigma)
        print(f'\nEVAL {label} (fresh seeds, learned): '
              f'overall {overall:.2f}')
        for part, rate in per_part.items():
            print(f'  target part {part}: {rate:.2f}')
            rows.append(('learned', label, part, rate))
        rows.append(('learned', label, 'overall', overall))

    per_part, overall = evaluate(p1, (0.0,) * 5, 20, 9000, 0.09)
    print(f'\nEVAL sigma 0.09 (fresh seeds, untrained zero weights): '
          f'overall {overall:.2f}')
    for part, rate in per_part.items():
        print(f'  target part {part}: {rate:.2f}')
        rows.append(('untrained', 'sigma 0.09', part, rate))
    rows.append(('untrained', 'sigma 0.09', 'overall', overall))

    learned_overall = dict(
        (row[1], row[3]) for row in rows
        if row[0] == 'learned' and row[2] == 'overall')['sigma 0.09']
    learned_parts = [row[3] for row in rows
                     if row[0] == 'learned'
                     and row[1] == 'sigma 0.09'
                     and row[2] != 'overall']
    passed = learned_overall >= 0.80 and min(learned_parts) >= 0.60
    print()
    print(f'CRITERION: overall >= 0.80 and every part >= 0.60 '
          f'at sigma 0.09 -> overall {learned_overall:.2f}, '
          f'worst part {min(learned_parts):.2f} -> '
          f'{"PASS" if passed else "FAIL"}')
    if passed:
        print('VERDICT: PASS. Aiming can be LEARNED, not just '
              'hand-set: the trained controller lands its '
              'contact in the target part on fresh noise.')
    else:
        print('VERDICT: FAIL by the pre-registered criterion. '
              'Aiming was demonstrated hand-set but not '
              'learned to this bar by this controller form '
              'and search.')

    out = OUT / 'kinetra_aim_results.csv'
    with out.open('w', newline='') as f:
        cw = csv.writer(f)
        cw.writerow(['controller', 'condition', 'target_part',
                     'hit_rate'])
        cw.writerows(rows)
    print('Saved:', out.name)


if __name__ == '__main__':
    main()
