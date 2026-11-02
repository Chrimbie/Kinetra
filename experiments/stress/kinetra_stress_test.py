#!/usr/bin/env python3
"""Kinetra stress test: graceful degradation or cliff collapse?

Christopher's standing instruction for this run is "test
it in any way you like," with his stated faith that the
analogy is near perfect. A near-perfect mechanism should
degrade GRACEFULLY under realistic damage, not collapse
off a cliff. This test damages the ball's world in four
ways and watches the shape of the decline:

1. OBSERVATION NOISE. The policy's inputs get Gaussian
   noise (sigma 0.1, then 0.2): the ball mis-sees its own
   position/velocity, and the full-sensor policy mis-sees
   the gap error and push. Physics itself is unchanged;
   only observation is noisy. Noise draws come from a
   fresh Random(999) per simulation so arms are
   comparable, and no noise is drawn at all when sigma
   is 0, keeping clean runs bit-identical to earlier
   scripts.
2. DELAY. Observations arrive late. Full-sensor policy:
   gap error and push from 1 gate ago (stale). Contact
   policy: contact side from 2 gates ago, then 3 gates
   ago (the original is 1 gate ago). Delayed feedback is
   the realistic version of the contact loop.
3. GATE WIDTH. The evaluation clearance changes from
   the trained +-0.40 to narrower +-0.25 and wider
   +-0.55. Training never sees these widths.
4. MID-RUN DYNAMICS CHANGE. After gate 30, every side
   push doubles (x2.0). The world changes halfway
   through a run; weights stay frozen, exactly as in
   the original evaluation.

TRAINING (unchanged from earlier scripts): full-sensor
and contact-only controllers trained on seeds 101, 102,
103 (budget 4000, search seed 12345, training tracks
only). Results are means across the 3 controllers, on
the SAME holdout groups (30 similar seeds 1000-1029,
30 rough seeds 2000-2029).

READING RULE (Mia's, fixed before running): graceful
means misses rise monotonically with stress but stay
below untrained-clean levels under MODERATE stress
(noise 0.1, +1 extra delay gate, wider gates). A CLIFF
means any moderate stress pushes a policy to or above
untrained-clean levels, or a small stress step causes
a disproportionate jump. All numbers reported
regardless; there is no pass/fail gate on this
characterization, only the shape of decline.

SANITY ANCHORS: seed-101 clean arms must reproduce
earlier runs exactly (full-learned 1.9%/23.3%,
contact-learned 20.3%/49.4%). If they do not, this
script's physics copy is wrong.

Track/physics code mirrors Christopher's
ball_track_transfer_test.py (Chrimbie/Kinetra).
Python standard library only.

Run: python kinetra_stress_test.py
Writes kinetra_stress_results.csv alongside the script.
"""

import csv
import math
import random
import statistics
from dataclasses import dataclass
from pathlib import Path

STEPS = 60
GAP_HALF_WIDTH = 0.40
OUT = Path(__file__).resolve().parent
TRAIN_SEEDS = [101, 102, 103]
SEARCH_SEED = 12345
BUDGET = 4000


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


def step_physics(y, velocity, angle, spin, angle_target, spin_target,
                 push):
    angle += 0.48 * (angle_target - angle)
    spin += 0.43 * (spin_target - spin)
    velocity = (0.48 * velocity + 0.83 * math.sin(angle)
                + 0.23 * spin + push)
    y += velocity
    return y, velocity, angle, spin


def contact_of(y, gate_center, gap_hw):
    if y > gate_center + gap_hw:
        return 1
    if y < gate_center - gap_hw:
        return -1
    return 0


def effective_push(gate, index, push_change):
    if push_change is not None:
        at, mult = push_change
        if index >= at:
            return gate.push * mult
    return gate.push


def simulate_full(track, weights, y_noise=0.0, vel_noise=0.0,
                  obs_delay=0, gap_hw=GAP_HALF_WIDTH,
                  push_change=None):
    rng = random.Random(999)
    y = velocity = 0.0
    angle, spin = 0.0, 0.0
    misses = 0
    sum_sq_err = 0.0
    p, d, wind, s, swind = weights
    obs_hist = []
    for index, gate in enumerate(track):
        push = effective_push(gate, index, push_change)
        gap_error = gate.center - y
        obs_hist.append((gap_error, push))
        if obs_delay == 0:
            gap_obs, push_obs = gap_error, push
        elif index - obs_delay >= 0:
            gap_obs, push_obs = obs_hist[index - obs_delay]
        else:
            gap_obs, push_obs = 0.0, 0.0
        if y_noise > 0:
            gap_obs += rng.gauss(0, y_noise)
        if vel_noise > 0:
            vel_obs = velocity + rng.gauss(0, vel_noise)
            push_obs = push_obs + rng.gauss(0, vel_noise)
        else:
            vel_obs = velocity
        angle_target = clip(p * gap_obs - d * vel_obs - wind * push_obs,
                            -0.9, 0.9)
        spin_target = clip(s * gap_obs - swind * push_obs, -0.9, 0.9)
        y, velocity, angle, spin = step_physics(
            y, velocity, angle, spin, angle_target, spin_target, push)
        gap_error_after = gate.center - y
        misses += abs(gap_error_after) > gap_hw
        sum_sq_err += gap_error_after ** 2
    return sum_sq_err / len(track), misses / len(track)


def simulate_contact(track, weights, y_noise=0.0, vel_noise=0.0,
                     contact_delay=1, gap_hw=GAP_HALF_WIDTH,
                     push_change=None):
    rng = random.Random(999)
    y = velocity = 0.0
    angle, spin = 0.0, 0.0
    misses = 0
    sum_sq_err = 0.0
    contact_hist = []
    py, d, wc, sy, sc = weights
    for index, gate in enumerate(track):
        push = effective_push(gate, index, push_change)
        if index - contact_delay >= 0:
            contact_prev = contact_hist[index - contact_delay]
        else:
            contact_prev = 0
        if y_noise > 0:
            y_obs = y + rng.gauss(0, y_noise)
        else:
            y_obs = y
        if vel_noise > 0:
            vel_obs = velocity + rng.gauss(0, vel_noise)
        else:
            vel_obs = velocity
        angle_target = clip(py * y_obs - d * vel_obs
                            - wc * contact_prev, -0.9, 0.9)
        spin_target = clip(sy * y_obs - sc * contact_prev, -0.9, 0.9)
        y, velocity, angle, spin = step_physics(
            y, velocity, angle, spin, angle_target, spin_target, push)
        gap_error_after = gate.center - y
        misses += abs(gap_error_after) > gap_hw
        sum_sq_err += gap_error_after ** 2
        contact_hist.append(contact_of(y, gate.center, gap_hw))
    return sum_sq_err / len(track), misses / len(track)


def random_search(track, sim_fn, budget=BUDGET, seed=SEARCH_SEED):
    r = random.Random(seed)
    weights = (0.0,) * 5

    def objective(w):
        mse, miss = sim_fn(track, w)
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
    return weights


def mean_miss(tracks, fn):
    return sum(fn(t)[1] for t in tracks) / len(tracks)


def main():
    similar = [make_track(1000 + s) for s in range(30)]
    rough = [make_track(2000 + s, kind='rough') for s in range(30)]
    baseline = (0.0,) * 5
    controllers = {}
    for tseed in TRAIN_SEEDS:
        train_track = make_track(tseed)
        controllers[tseed] = (
            random_search(train_track, simulate_full),
            random_search(train_track, simulate_contact))
    print('trained controllers for seeds', TRAIN_SEEDS, flush=True)

    # Stress arms: (label, kwargs for full, kwargs for contact,
    #               include untrained?)
    arms = [
        ('clean', {}, {}, True),
        ('noise 0.1', {'y_noise': 0.1, 'vel_noise': 0.1},
         {'y_noise': 0.1, 'vel_noise': 0.1}, False),
        ('noise 0.2', {'y_noise': 0.2, 'vel_noise': 0.2},
         {'y_noise': 0.2, 'vel_noise': 0.2}, False),
        ('delay +1', {'obs_delay': 1}, {'contact_delay': 2}, False),
        ('delay +2', {'obs_delay': 2}, {'contact_delay': 3}, False),
        ('gates narrow 0.25', {'gap_hw': 0.25}, {'gap_hw': 0.25}, True),
        ('gates wide 0.55', {'gap_hw': 0.55}, {'gap_hw': 0.55}, True),
        ('push x2 after gate 30', {'push_change': (30, 2.0)},
         {'push_change': (30, 2.0)}, True),
    ]
    rows = []
    table = {}
    for label, full_kw, contact_kw, with_untrained in arms:
        for group_name, tracks in [('similar', similar),
                                   ('rough', rough)]:
            full_vals, contact_vals, untrained_vals = [], [], []
            for tseed in TRAIN_SEEDS:
                learned, contact = controllers[tseed]
                full_vals.append(mean_miss(
                    tracks,
                    lambda t: simulate_full(t, learned, **full_kw)))
                contact_vals.append(mean_miss(
                    tracks,
                    lambda t: simulate_contact(t, contact,
                                               **contact_kw)))
                if with_untrained:
                    untrained_vals.append(mean_miss(
                        tracks,
                        lambda t: simulate_full(t, baseline, **full_kw)))
            table[(label, group_name, 'full-learned')] = (
                sum(full_vals) / len(full_vals),
                statistics.pstdev(full_vals))
            table[(label, group_name, 'contact-learned')] = (
                sum(contact_vals) / len(contact_vals),
                statistics.pstdev(contact_vals))
            if with_untrained:
                table[(label, group_name, 'untrained')] = (
                    sum(untrained_vals) / len(untrained_vals),
                    statistics.pstdev(untrained_vals))
            rows.append((label, group_name, 'full-learned',
                         table[(label, group_name, 'full-learned')][0]))
            rows.append((label, group_name, 'contact-learned',
                         table[(label, group_name,
                                'contact-learned')][0]))
            if with_untrained:
                rows.append((label, group_name, 'untrained',
                             table[(label, group_name, 'untrained')][0]))
        print(f'arm {label} done', flush=True)

    print()
    print(f'{"Arm":24s} | {"Group":8s} | {"Policy":15s} | '
          f'{"Missed":>7s} | {"Std":>4s}')
    print('-' * 72)
    for label, _, _, _ in arms:
        for group_name in ('similar', 'rough'):
            for policy in ('full-learned', 'contact-learned',
                           'untrained'):
                key = (label, group_name, policy)
                if key not in table:
                    continue
                mean, std = table[key]
                print(f'{label:24s} | {group_name:8s} | {policy:15s} | '
                      f'{mean*100:6.1f}% | {std*100:3.1f}')

    # Sanity anchors: seed-101 clean arms.
    learned101, contact101 = controllers[101]
    print()
    print(f"Sanity seed 101 clean: full-learned "
          f"{mean_miss(similar, lambda t: simulate_full(t, learned101))*100:.1f}%/"
          f"{mean_miss(rough, lambda t: simulate_full(t, learned101))*100:.1f}% "
          f"(expect 1.9%/23.3%), contact-learned "
          f"{mean_miss(similar, lambda t: simulate_contact(t, contact101))*100:.1f}%/"
          f"{mean_miss(rough, lambda t: simulate_contact(t, contact101))*100:.1f}% "
          f"(expect 20.3%/49.4%)")

    out = OUT / 'kinetra_stress_results.csv'
    with out.open('w', newline='') as f:
        cw = csv.writer(f)
        cw.writerow(['stress_arm', 'holdout_group', 'policy',
                     'missed_gate_fraction_mean'])
        cw.writerows(rows)
    print('Saved:', out.name)


if __name__ == '__main__':
    main()
