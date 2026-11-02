# Kinetra stress test

Graceful degradation or cliff collapse? Run under
Christopher's standing instruction to test in any way
I like, with his stated faith that the analogy is near
perfect. A near-perfect mechanism should degrade
gracefully, not collapse off a cliff.

## Stress arms (weights frozen, evaluation only)

- NOISE 0.1 / 0.2: Gaussian noise on the policy's
  observations (position, velocity, gap error, push).
  Physics unchanged.
- DELAY +1 / +2: observations arrive late. Full
  policy: gap error and push from 1 / 2 gates ago.
  Contact policy: contact side from 2 / 3 gates ago
  (original: 1 gate ago).
- GATES NARROW 0.25 / WIDE 0.55: evaluation
  clearance changed from the trained +-0.40.
- PUSH x2 AFTER GATE 30: side forces double halfway
  through a run. The world changes mid-run.

Training unchanged: full-sensor and contact-only
controllers on seeds 101, 102, 103 (budget 4000,
search seed 12345). Means across 3 controllers on
the same holdouts (30 similar, 30 rough). Sanity:
seed-101 clean arms reproduce earlier runs exactly
(full 1.9%/23.3%, contact 20.3%/49.4%).

## Result (mean missed gates; clean untrained 60.8% similar / 84.5% rough)

| Arm | Full similar | Full rough | Contact similar | Contact rough |
|---|---:|---:|---:|---:|
| Clean | 1.4% | 22.6% | 22.6% | 52.4% |
| Noise 0.1 | 5.4% | 25.9% | 27.2% | 54.0% |
| Noise 0.2 | 17.2% | 35.2% | 33.0% | 55.5% |
| Delay +1 | 45.0% | 68.1% | 47.4% | 65.0% |
| Delay +2 | 74.6% | 82.0% | 55.8% | 70.0% |
| Gates narrow 0.25 | 5.6% | 36.1% | 47.1% | 71.3% |
| Gates wide 0.55 | 0.4% | 14.0% | 11.6% | 37.7% |
| Push x2 after 30 | 3.1% | 28.4% | 26.9% | 58.1% |

## Reading

- GRACEFUL: noise (both policies stay far below
  untrained even at 0.2; contact rough barely moves,
  52.4% to 55.5%), mid-run push doubling (frozen
  feedback absorbs it: full 1.4% to 3.1% similar,
  contact 22.6% to 26.9%), wider gates (everything
  improves, as expected).
- CLIFF: delay. One extra gate of delay takes full
  similar from 1.4% to 45.0% and contact similar
  from 22.6% to 47.4%; two extra gates take
  full-similar to 74.6%, WORSE than untrained
  (60.8%). A stale observation is nearly as bad as
  no observation, and then worse. The feedback
  loop's value is highly timing-dependent.
- CONTACT'S FRAGILITIES: narrow gates hit contact
  hard (similar 22.6% to 47.1%, rough 52.4% to
  71.3%), and delay makes contact highly
  seed-dependent (std up to 19.4, versus 0.5-1.0
  clean): some controllers survive delay, others
  collapse.
- The honest summary for "near perfect": strong
  and robust in most dimensions, with one sharp
  breaking point, timing, and one fragile regime,
  contact under narrow gates. Any architecture
  built on this needs fresh feedback; delayed
  feedback is a collapse mode, not a minor cost.

## Files

- `kinetra_stress_test.py` — the test (python
  kinetra_stress_test.py; writes
  kinetra_stress_results.csv). Standard library
  only. Track/physics code mirrors Christopher's
  ball_track_transfer_test.py; his file is NOT
  bundled here.

## What this test does NOT test

Routing between multiple stored controllers; online
adaptation during a run (weights frozen throughout
here); noise on the contact signal itself; combined
stresses (noise AND delay together).
