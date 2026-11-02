# Kinetra learned aiming test

Closes the honest gap in the pass-through contact
test: aiming there was HAND-SET, not learned. This
test asks whether a controller can LEARN to aim for
a target part, given the target as input.

## Setup (same as the pass test)

- Track width 3.0. Problem P1: center 0, width 2,
  5 parts of 0.4 each.
- 10 physics steps per pass, Kinetra's constants,
  fresh ball state per pass.
- Controller form (mirrors the original Kinetra
  policy, gap_error redefined as target error):
  angle_target = clip(p*(target - y) - d*velocity
  - wind*push, +-0.9); spin_target =
  clip(s*(target - y) - swind*push, +-0.9).
- Training: random search, budget 4000, seed
  12345, 20 balanced passes per trial (targets
  cycle 0-4 x 4, fixed push seeds), push sigma
  0.09, objective = contact MSE + 0.12 * miss
  fraction (same form as the original).
- Evaluation: 100 fresh-seed passes (20 per part)
  at sigma 0.09, repeated at sigma 0.05.

## Results: PASS by the pre-registered criterion

- Learned weights (p, d, wind, s, swind):
  (1.019, 2.631, 2.238, 1.906, -1.474). Note the
  push terms: unlike the hand-set controller, the
  learner USES push feedforward (wind 2.238,
  swind -1.474) to cancel disturbances, plus
  strong velocity damping (d 2.631).
- Training converged fast: objective 0.7342 at
  trial 0 to 0.0011 by trial 500, flat after.
- Fresh-seed evaluation: overall 1.00 at sigma
  0.09 AND at sigma 0.05, every part 1.00.
  Criterion was overall >= 0.80 and every part
  >= 0.60.
- Untrained (zero weights) control on the same
  fresh passes: 0.24 overall (edge parts 0.05,
  middle parts up to 0.45). The task is not
  trivial, and the learned 1.00 is real.

## Honesty boundary

This tests learning to aim under fresh NOISE. It
does not test generalizing to new targets: all 5
parts appear in training, and targets are part
centers only. Continuous aim points (any position,
not just centers) and unseen targets are
follow-ups. Toy scale throughout; no architecture
claim.

## Files

- `kinetra_aim_test.py` — the test (python
  kinetra_aim_test.py; writes
  kinetra_aim_results.csv). Standard library only.
- `kinetra_aim_results.csv` — per-part hit rates
  for learned and untrained controllers.
