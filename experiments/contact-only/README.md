# Kinetra contact-only test

Does correction transfer without handed sensors?

## The question

Christopher's Kinetra experiment (Chrimbie/Kinetra)
trains a small correction controller on one ball track
and shows it transfers to unseen tracks: 1.9% missed
gates on similar holdouts, 23.3% on rough holdouts,
versus 78.9% / 87.2% for replaying the memorized
route. But the full-sensor policy reads two handed
inputs at every gate: the gap error (gate.center - y)
and the observed side-force (gate.push). So part of
the transfer might come from the sensor, not from
learning. This test removes both.

## The contact-only policy

At each gate the controller may use ONLY:

- the ball's own state: y, velocity, angle, spin
  (proprioception, not gate observation);
- the PREVIOUS gate's contact outcome, delayed by
  one gate: +1 hit the right post, -1 hit the left
  post, 0 passed cleanly. Contact is observable by
  touch, with side but no magnitude.

It may NOT use the current gate's center, the
current gate's push, or any current-gate observation.
The push still moves the ball physically, and the
ball feels that through its own velocity change.

Controller form (same 5-weight shape, new inputs):

- angle_target = clip(py * y - d * velocity
  - wc * contact_prev, -0.9, 0.9)
- spin_target = clip(sy * y - sc * contact_prev,
  -0.9, 0.9)

## Protocol (identical to the original where possible)

- Same tracks: training seed 101; 30 similar seeds
  1000-1029; 30 rough seeds 2000-2029. 60 gates per
  track, miss = |y - center| > 0.40.
- Same training: derivative-free random search,
  budget 4000 trials, seed 12345, training track
  only, objective MSE + 0.12 * miss fraction.
- Arms: untrained, full-sensor trained (re-run as a
  validation arm), route-replay, contact-only trained.
- Weights frozen on evaluation tracks.

## Result: PASS

| Dataset | Untrained | Full-sensor | Route-replay | Contact-only |
|---|---:|---:|---:|---:|
| Training track | 75.0% | 0.0% | 0.0% | 21.7% |
| 30 similar holdouts | 60.8% | 1.9% | 78.9% | **20.3%** |
| 30 rough holdouts | 84.5% | 23.3% | 87.2% | **49.4%** |

Validation: the full-sensor and replay arms
reproduce the README numbers EXACTLY (1.9% / 78.9%
similar, 23.3% / 87.2% rough), so the physics copy
is correct and the contact numbers are trusted.

## Interpretation

The learning story holds without handed sensors:
contact-only beats both untrained and replay on
both holdout groups, by wide margins. Transfer is
not just the sensor. But the handed sensors supply
a large share of the gain: contact-only captures
roughly 60-70% of the untrained-to-full improvement
(40.5 of 58.9 points on similar tracks, 35.1 of
61.2 on rough tracks), with the sensors supplying
the rest. And the contact policy never masters even
its training track (21.7% missed) yet still
transfers, a clean dissociation: training-track
mastery is not required for transfer, because the
thing that transfers is a feedback policy, not a
memorized route.

## Files

- `kinetra_contact_test.py` — the test (python
  kinetra_contact_test.py; writes
  kinetra_contact_results.csv). Standard library
  only. Track/physics code mirrors Christopher's
  ball_track_transfer_test.py; his file is NOT
  bundled here.
- `REFERENCE_ball_track_transfer_test.py` — local
  reading copy of the original, NOT part of the
  deliverable and NOT included in the zip.

## What this test does NOT test

Contact with magnitude (impact depth would smuggle
the gap error back in); current-gate contact;
online adaptation during evaluation; more training
seeds (one training track, as in the original, so
no confidence intervals yet); whether a hand-tuned
PID controller matches the learned one.
