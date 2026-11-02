# Kinetra robustness test

Multi-seed transfer + PID baselines. Run under
Christopher's instruction to "test it in any way you
like."

## The two gaps this closes

1. SINGLE SEED. The original Kinetra benchmark and
   the contact-only test each trained on ONE track
   (seed 101). This test trains on FIVE tracks
   (seeds 101-105) with the full treatment each,
   reporting mean +/- std across training seeds on
   the SAME holdout groups (30 similar seeds
   1000-1029, 30 rough seeds 2000-2029).

2. NO CLASSICAL BASELINE. The learned controllers
   had only faced untrained and memorized replay.
   This test adds hand-tuned PD controllers, one per
   sensor condition, so each learned controller faces
   classical control WITH THE SAME INPUTS:
   - full-PID: angle from gap error + velocity, spin
     from gap error (150 grid points);
   - contact-PID: angle from y + velocity + previous
     contact side, spin from y + contact side (576
     grid points).
   PD gains are grid-searched on the training track
   ONLY, then frozen.

## Result

Mean missed gates across 5 training seeds (std in
parentheses):

| Group | Untrained | Full-learned | Replay | Contact-learned | Full-PID | Contact-PID |
|---|---:|---:|---:|---:|---:|---:|
| Similar | 60.8% (0.0) | 1.5% (0.5) | 74.8% (5.0) | 22.6% (3.2) | 1.8% (0.9) | 36.1% (4.9) |
| Rough | 84.5% (0.0) | 22.2% (1.0) | 84.5% (1.6) | 52.1% (3.6) | 30.0% (0.6) | 60.8% (2.9) |

Sanity anchors: seed 101 reproduces the earlier
runs EXACTLY (full-learned 1.9%/23.3%,
contact-learned 20.3%/49.4%).

## Reading

- ROBUSTNESS HOLDS. Contact-learned beats untrained
  and replay on both holdout groups, on every seed
  (similar range 19.6-27.8%, rough 49.0-58.7%).
  The contact PASS was not a seed-101 fluke. The
  original full-sensor result is tight too (similar
  std 0.5, rough std 1.0).
- LEARNED BEATS PID IN ALL FOUR CELLS, but the
  margin varies honestly: on similar tracks with
  full sensors, PID nearly ties learned (1.8% vs
  1.5%). The learned advantage is largest where it
  is hardest: rough tracks (22.2% vs 30.0%) and
  restricted inputs (22.6% vs 36.1% similar, 52.1%
  vs 60.8% rough).
- BUDGET CAVEAT (stated, not hidden): the learned
  search got 4000 trials per seed versus 150/576
  grid evaluations for PID. PID nearly tying on
  similar tracks despite less tuning strengthens
  PID there; the learned wins elsewhere could
  partly reflect tuning budget, though the grids
  are dense enough that a large hidden PID gain is
  unlikely. A finer PID grid would settle it.
- Contact learning is more seed-sensitive than
  full-sensor learning (std 3.2/3.6 vs 0.5/1.0).

## Files

- `kinetra_robust_test.py` — the test (python
  kinetra_robust_test.py; writes
  kinetra_robust_results.csv). Standard library
  only. Track/physics code mirrors Christopher's
  ball_track_transfer_test.py; his file is NOT
  bundled here.

## What this test does NOT test

Noise, delays, or gate-width variation; online
adaptation during evaluation; a finer PID grid;
routing between multiple stored controllers.
