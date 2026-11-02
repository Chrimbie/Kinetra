#!/usr/bin/env python3
"""Kinetra contact-only test: does correction transfer without
handed sensors?

Christopher's Kinetra experiment (Chrimbie/Kinetra) trains a
small correction controller on one ball track and shows it
transfers to unseen tracks: 1.9% missed gates on similar
holdouts, 23.3% on rough holdouts, versus 78.9% / 87.2% for
replaying the memorized route. But the full-sensor policy
reads two handed inputs at every gate: the gap error
(gate.center - y, which requires knowing where the gate
is) and the observed side-force (gate.push). So part of
the transfer might come from the sensor, not from
learning. This test removes both.

CONTACT-ONLY POLICY (this test's design, Mia's):
At each gate the controller may use ONLY:
- the ball's own state: y (position), velocity, angle,
  spin (proprioception — the ball knows how IT moves,
  not where the gate is);
- the PREVIOUS gate's contact outcome, delayed by one
  gate: +1 if the ball hit the right post (it was right
  of the gap), -1 if it hit the left post, 0 if it
  passed cleanly. Contact is observable by touch: the
  ball feels which post it hit, without ever being told
  the gate's center. No magnitude is given, only side.
It may NOT use the current gate's center, the current
gate's push, or any current-gate observation. The push
still affects the ball physically (velocity += push),
and the ball feels that through its own velocity change
— that is proprioception, not a handed sensor.

Controller form (same 5-weight shape, different inputs):
  angle_target = clip(py * y - d * velocity
                      - wc * contact_prev, -0.9, 0.9)
  spin_target  = clip(sy * y - sc * contact_prev,
                      -0.9, 0.9)
Sign conventions do not matter: the derivative-free
random search finds the signs, exactly as in the
original.

PROTOCOL (identical to the original wherever the test
allows, so the numbers are comparable):
- Same tracks: training track seed 101; 30 similar
  tracks seeds 1000-1029; 30 rough tracks 2000-2029.
  60 gates per track, miss = |y - center| > 0.40.
- Same training: derivative-free random search, budget
  4000 trials, seed 12345, on the training track ONLY,
  same objective (MSE + 0.12 * miss fraction).
- Arms on every group: untrained (all-zero weights),
  full-sensor trained (the original policy, re-run here
  as a validation arm), route-replay (training-track
  commands replayed), contact-only trained.
- The full-sensor arm MUST reproduce the README
  numbers (training 0.0%, similar 1.9%, rough 23.3%;
  replay 78.9% / 87.2%) — if it does not, this script's
  physics copy is wrong and nothing here is trusted.

PASS CRITERION (Mia's, fixed before running):
contact-only missed-gate fraction beats BOTH untrained
AND route-replay on BOTH holdout groups (similar and
rough). That is "transfer without handed sensors."
A secondary, reported-not-gated comparison: how close
contact-only gets to full-sensor (1.9% / 23.3%).
All numbers are reported regardless of pass/fail.

WHAT THIS TEST DOES NOT TEST: contact with magnitude
(impact depth would smuggle the gap error back in);
current-gate contact (contact during a gate arrives
too late to steer through that gate); online adaptation
during evaluation (weights are frozen, as in the
original); more training seeds (one training track,
as in the original).

Track/physics code mirrors Christopher's
ball_track_transfer_test.py (Chrimbie/Kinetra, public
repo, read via the GitHub connector 2026-11-01). His
file is NOT bundled here; the mirror exists so this
script runs standalone. Python standard library only.

Run: python kinetra_contact_test.py
Writes kinetra_contact_results.csv alongside the script.
"""

import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path

STEPS = 60
GAP_HALF_WIDTH = 0.40
OUT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Gate:
    center: float
    push: float


def make_track(seed, kind='similar', steps=STEPS):
    r = random.Random(seed)
    gates = []
    center = 0.0
    for i in range(steps):
        if kind == 'similar':
            center = max(-1.2, min(1.2, 0.82 * center + r.gauss(0, 0.24)))
            push = r.gauss(0, 0.09)
        elif kind == 'rough':
            center = max(-1.4, min(1.4, 0.60 * center + r.gauss(0, 0.48)))
            push = r.gauss(0, 0.19)
        else:
            raise ValueError(kind)
        gates.append(Gate(center, push))
    return tuple(gates)


def clip(x, a, b):
    return min(b, max(a, x))


def simulate_full(track, weights, commands=None,
                  initial_angle=0.0, initial_spin=0.0):
    """The original Kinetra policy (handed sensors)."""
    y = velocity = 0.0
    angle, spin = initial_angle, initial_spin
    misses = 0
    sum_sq_err = 0.0
    p, d, wind, s, swind = weights
    for index, gate in enumerate(track):
        gap_error = gate.center - y
        if commands is None:
            angle_target = clip(p * gap_error - d * velocity
                                - wind * gate.push, -0.9, 0.9)
            spin_target = clip(s * gap_error - swind * gate.push,
                               -0.9, 0.9)
        else:
            angle_target, spin_target = commands[index]
        angle += 0.48 * (angle_target - angle)
        spin += 0.43 * (spin_target - spin)
        velocity = (0.48 * velocity + 0.83 * math.sin(angle)
                    + 0.23 * spin + gate.push)
        y += velocity
        gap_error_after = gate.center - y
        misses += abs(gap_error_after) > GAP_HALF_WIDTH
        sum_sq_err += gap_error_after ** 2
    return sum_sq_err / len(track), misses / len(track)


def contact_of(y, gate):
    """What the ball feels at a gate, without knowing center:
    +1 hit the right post, -1 hit the left post, 0 passed."""
    if y > gate.center + GAP_HALF_WIDTH:
        return 1
    if y < gate.center - GAP_HALF_WIDTH:
        return -1
    return 0


def simulate_contact(track, weights):
    """Contact-only policy: own state + previous gate's
    contact side. No current-gate center, no push sensor."""
    y = velocity = 0.0
    angle, spin = 0.0, 0.0
    misses = 0
    sum_sq_err = 0.0
    contact_prev = 0
    py, d, wc, sy, sc = weights
    for gate in track:
        angle_target = clip(py * y - d * velocity - wc * contact_prev,
                            -0.9, 0.9)
        spin_target = clip(sy * y - sc * contact_prev, -0.9, 0.9)
        angle += 0.48 * (angle_target - angle)
        spin += 0.43 * (spin_target - spin)
        velocity = (0.48 * velocity + 0.83 * math.sin(angle)
                    + 0.23 * spin + gate.push)
        y += velocity
        gap_error_after = gate.center - y
        misses += abs(gap_error_after) > GAP_HALF_WIDTH
        sum_sq_err += gap_error_after ** 2
        contact_prev = contact_of(y, gate)
    return sum_sq_err / len(track), misses / len(track)


def train_contact(track, budget=4000, seed=12345):
    """Same search as the original, on the training track
    only, but scoring the contact-only policy."""
    r = random.Random(seed)
    weights = (0.0, 0.0, 0.0, 0.0, 0.0)

    def objective(w):
        mse, miss = simulate_contact(track, w)
        return mse + 0.12 * miss

    best = objective(weights)
    for j in range(budget):
        if j % 10 == 0:
            proposal = tuple(r.uniform(-2.4, 2.4) for _ in weights)
        else:
            scale = 0.9 * (1 - j / budget) + 0.025
            proposal = tuple(clip(w + r.gauss(0, scale), -3, 3)
                             for w in weights)
        result = objective(proposal)
        if result < best:
            weights, best = proposal, result
    return weights, best


def train_full(track, budget=4000, seed=12345):
    """The original's training, re-run for validation."""
    r = random.Random(seed)
    weights = (0.0, 0.0, 0.0, 0.0, 0.0)

    def objective(w):
        mse, miss = simulate_full(track, w)
        return mse + 0.12 * miss

    best = objective(weights)
    for j in range(budget):
        if j % 10 == 0:
            proposal = tuple(r.uniform(-2.4, 2.4) for _ in weights)
        else:
            scale = 0.9 * (1 - j / budget) + 0.025
            proposal = tuple(clip(w + r.gauss(0, scale), -3, 3)
                             for w in weights)
        result = objective(proposal)
        if result < best:
            weights, best = proposal, result
    return weights, best


def mean_metrics(tracks, fn):
    vals = [fn(t) for t in tracks]
    return (sum(v[0] for v in vals) / len(vals),
            sum(v[1] for v in vals) / len(vals))


def main():
    train_track = make_track(101)
    baseline = (0.0,) * 5
    learned, _ = train_full(train_track)
    contact, _ = train_contact(train_track)
    # Determinism: same input, same trajectory.
    assert simulate_contact(train_track, contact) == \
        simulate_contact(train_track, contact), \
        'Contact replay not deterministic'
    assert simulate_full(train_track, learned) == \
        simulate_full(train_track, learned), \
        'Full replay not deterministic'

    # Training-track commands for the replay arm (from the
    # full-sensor policy, exactly as in the original).
    # Recompute angle/spin targets per gate by re-running
    # the full policy and recording its targets.
    def full_targets(track, weights):
        y = velocity = 0.0
        angle, spin = 0.0, 0.0
        p, d, wind, s, swind = weights
        cmds = []
        for gate in track:
            gap_error = gate.center - y
            at = clip(p * gap_error - d * velocity - wind * gate.push,
                      -0.9, 0.9)
            st = clip(s * gap_error - swind * gate.push, -0.9, 0.9)
            cmds.append((at, st))
            angle += 0.48 * (at - angle)
            spin += 0.43 * (st - spin)
            velocity = (0.48 * velocity + 0.83 * math.sin(angle)
                        + 0.23 * spin + gate.push)
            y += velocity
        return cmds

    training_commands = full_targets(train_track, learned)

    print('Learned full-sensor weights: '
          + ', '.join(f'{w:+.3f}' for w in learned))
    print('Learned contact-only weights: '
          + ', '.join(f'{w:+.3f}' for w in contact))
    print()
    groups = {
        'Training track': [train_track],
        'New similar tracks (30)': [make_track(1000 + s)
                                     for s in range(30)],
        'New rough tracks (30)': [make_track(2000 + s, kind='rough')
                                   for s in range(30)],
    }
    print(f'{"Dataset":27s} | {"Mode":14s} | {"MSE":>9s} '
          f'| {"Missed gates":>12s}')
    print('-' * 72)
    rows = []
    results = {}
    for group, tracks in groups.items():
        arms = {
            'untrained': lambda t: simulate_full(t, baseline),
            'full-sensor': lambda t: simulate_full(t, learned),
            'route-replay': lambda t: simulate_full(
                t, learned, commands=training_commands),
            'contact-only': lambda t: simulate_contact(t, contact),
        }
        results[group] = {}
        for label, fn in arms.items():
            mse, miss = mean_metrics(tracks, fn)
            results[group][label] = (mse, miss)
            print(f'{group:27s} | {label:14s} | {mse:9.4f} '
                  f'| {miss*100:11.1f}%')
            rows.append((group, label, mse, miss))

    sim = results['New similar tracks (30)']
    rough = results['New rough tracks (30)']
    passed = (
        sim['contact-only'][1] < sim['untrained'][1]
        and sim['contact-only'][1] < sim['route-replay'][1]
        and rough['contact-only'][1] < rough['untrained'][1]
        and rough['contact-only'][1] < rough['route-replay'][1]
    )
    print()
    print('Validation vs README (full-sensor / replay): '
          f"similar {sim['full-sensor'][1]*100:.1f}% / "
          f"{sim['route-replay'][1]*100:.1f}% "
          f"(README 1.9% / 78.9%), rough "
          f"{rough['full-sensor'][1]*100:.1f}% / "
          f"{rough['route-replay'][1]*100:.1f}% "
          f'(README 23.3% / 87.2%)')
    print(f"PASS (contact-only beats untrained AND replay on "
          f"both holdout groups): {passed}")
    print('Missed gates: abs(ball y - gate center) > 0.40. '
          'No adaptation on evaluation tracks.')

    out = OUT / 'kinetra_contact_results.csv'
    with out.open('w', newline='') as f:
        cw = csv.writer(f)
        cw.writerow(['track_group', 'mode', 'mean_squared_error',
                     'missed_gate_fraction'])
        cw.writerows(rows)
    print('Saved:', out.name)


if __name__ == '__main__':
    main()
