#!/usr/bin/env python3
"""Kinetra robustness test: multi-seed transfer + PID baselines.

Christopher's instruction for this run was "test it in any
way you like." This script takes the two biggest honesty
gaps in the story so far and closes them as far as a toy
allows:

1. SINGLE SEED. The original Kinetra benchmark and the
   contact-only test each trained on ONE track (seed
   101). A single seed cannot tell luck from mechanism.
   Here, five training tracks (seeds 101-105) each get
   the full treatment, and results are reported as
   mean +/- std across training seeds, on the SAME
   holdout groups (30 similar seeds 1000-1029, 30 rough
   seeds 2000-2029) for comparability.

2. NO CLASSICAL BASELINE. The learned controllers were
   compared against untrained and memorized replay, but
   never against classical control. A hand-tuned PD
   controller is the obvious classical baseline: if PD
   matches the learned controller, the "learning" story
   weakens considerably. Two PD baselines are tuned,
   one per sensor condition, so each learned controller
   faces classical control WITH THE SAME INPUTS:
   - full-PID: angle from gap error + velocity, spin
     from gap error (the handed sensors);
   - contact-PID: angle from y + velocity + previous
     contact side, spin from y + contact side (the
     restricted inputs).
   PD gains are grid-searched on the training track
   ONLY, then frozen — the same access the learned
   controllers get. Honest budget caveat: the learned
   search gets 4000 trials while the PD grids get
   150 / 576 evaluations. If PD wins, it wins despite
   less tuning; if learned wins, part of the margin
   may be tuning budget, and that is stated, not
   hidden.

ARMS (per training seed): untrained, full-learned,
route-replay, contact-learned, full-PID, contact-PID.

SANITY ANCHOR: the seed-101 full-learned arm must
reproduce the README numbers (similar 1.9%, rough
23.3%) and the contact-only arm must reproduce the
contact test (similar 20.3%, rough 49.4%). If either
fails, this script's physics copy is wrong.

READING THE RESULT (Mia's, fixed before running):
- Robustness: contact-learned mean beats untrained
  AND replay on both holdout groups across seeds.
- Classical check: within each sensor condition, does
  learned beat PD on holdouts? Four cells, reported
  regardless: full-learned vs full-PID, contact-learned
  vs contact-PID, on similar and rough.
No single number here "proves" anything beyond the
toy; the point is whether the pattern survives seeds
and classical control.

Track/physics code mirrors Christopher's
ball_track_transfer_test.py (Chrimbie/Kinetra) and the
contact-only policy from kinetra_contact_test.py.
Python standard library only.

Run: python kinetra_robust_test.py
Writes kinetra_robust_results.csv alongside the script.
"""

import csv
import itertools
import math
import random
import statistics
from dataclasses import dataclass
from pathlib import Path

STEPS = 60
GAP_HALF_WIDTH = 0.40
OUT = Path(__file__).resolve().parent
TRAIN_SEEDS = [101, 102, 103, 104, 105]
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
                 gate):
    angle += 0.48 * (angle_target - angle)
    spin += 0.43 * (spin_target - spin)
    velocity = (0.48 * velocity + 0.83 * math.sin(angle)
                + 0.23 * spin + gate.push)
    y += velocity
    return y, velocity, angle, spin


def simulate_full(track, weights, commands=None):
    y = velocity = 0.0
    angle, spin = 0.0, 0.0
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
        y, velocity, angle, spin = step_physics(
            y, velocity, angle, spin, angle_target, spin_target, gate)
        gap_error_after = gate.center - y
        misses += abs(gap_error_after) > GAP_HALF_WIDTH
        sum_sq_err += gap_error_after ** 2
    return sum_sq_err / len(track), misses / len(track)


def contact_of(y, gate):
    if y > gate.center + GAP_HALF_WIDTH:
        return 1
    if y < gate.center - GAP_HALF_WIDTH:
        return -1
    return 0


def simulate_contact(track, weights):
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
        y, velocity, angle, spin = step_physics(
            y, velocity, angle, spin, angle_target, spin_target, gate)
        gap_error_after = gate.center - y
        misses += abs(gap_error_after) > GAP_HALF_WIDTH
        sum_sq_err += gap_error_after ** 2
        contact_prev = contact_of(y, gate)
    return sum_sq_err / len(track), misses / len(track)


def simulate_full_pid(track, gains):
    kp, kd, ks = gains
    y = velocity = 0.0
    angle, spin = 0.0, 0.0
    misses = 0
    sum_sq_err = 0.0
    for gate in track:
        gap_error = gate.center - y
        angle_target = clip(kp * gap_error - kd * velocity, -0.9, 0.9)
        spin_target = clip(ks * gap_error, -0.9, 0.9)
        y, velocity, angle, spin = step_physics(
            y, velocity, angle, spin, angle_target, spin_target, gate)
        gap_error_after = gate.center - y
        misses += abs(gap_error_after) > GAP_HALF_WIDTH
        sum_sq_err += gap_error_after ** 2
    return sum_sq_err / len(track), misses / len(track)


def simulate_contact_pid(track, gains):
    kp, kd, kc, ks, kcs = gains
    y = velocity = 0.0
    angle, spin = 0.0, 0.0
    misses = 0
    sum_sq_err = 0.0
    contact_prev = 0
    for gate in track:
        angle_target = clip(kp * y - kd * velocity - kc * contact_prev,
                            -0.9, 0.9)
        spin_target = clip(ks * y - kcs * contact_prev, -0.9, 0.9)
        y, velocity, angle, spin = step_physics(
            y, velocity, angle, spin, angle_target, spin_target, gate)
        gap_error_after = gate.center - y
        misses += abs(gap_error_after) > GAP_HALF_WIDTH
        sum_sq_err += gap_error_after ** 2
        contact_prev = contact_of(y, gate)
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


def grid_search(track, sim_fn, grid):
    best_gains, best = None, float('inf')
    for gains in grid:
        mse, miss = sim_fn(track, gains)
        result = mse + 0.12 * miss
        if result < best:
            best_gains, best = gains, result
    return best_gains


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
        y, velocity, angle, spin = step_physics(
            y, velocity, angle, spin, at, st, gate)
    return cmds


def mean_miss(tracks, fn):
    return sum(fn(t)[1] for t in tracks) / len(tracks)


def main():
    similar = [make_track(1000 + s) for s in range(30)]
    rough = [make_track(2000 + s, kind='rough') for s in range(30)]
    baseline = (0.0,) * 5
    full_pid_grid = list(itertools.product(
        [0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
        [0.0, 0.5, 1.0, 1.5, 2.0],
        [0.0, 0.5, 1.0, 1.5, 2.0]))
    contact_pid_grid = list(itertools.product(
        [0.5, 1.0, 2.0, 3.0],
        [0.0, 0.5, 1.0, 2.0],
        [0.0, 0.5, 1.0, 2.0],
        [0.0, 1.0, 2.0],
        [0.0, 1.0, 2.0]))

    rows = []
    per_seed = {}
    for tseed in TRAIN_SEEDS:
        train_track = make_track(tseed)
        learned = random_search(train_track, simulate_full)
        contact = random_search(train_track, simulate_contact)
        full_pid = grid_search(train_track, simulate_full_pid,
                               full_pid_grid)
        contact_pid = grid_search(train_track, simulate_contact_pid,
                                  contact_pid_grid)
        cmds = full_targets(train_track, learned)
        arms = {
            'untrained': lambda t: simulate_full(t, baseline),
            'full-learned': lambda t: simulate_full(t, learned),
            'route-replay': lambda t: simulate_full(t, learned,
                                                    commands=cmds),
            'contact-learned': lambda t: simulate_contact(t, contact),
            'full-PID': lambda t: simulate_full_pid(t, full_pid),
            'contact-PID': lambda t: simulate_contact_pid(t, contact_pid),
        }
        per_seed[tseed] = {}
        for group_name, tracks in [('similar', similar),
                                   ('rough', rough)]:
            for label, fn in arms.items():
                miss = mean_miss(tracks, fn)
                per_seed[tseed][(group_name, label)] = miss
                rows.append((tseed, group_name, label, miss))
        print(f'train seed {tseed}: full-learned '
              f"{per_seed[tseed][('similar', 'full-learned')]*100:.1f}%/"
              f"{per_seed[tseed][('rough', 'full-learned')]*100:.1f}% "
              f'similar/rough, contact-learned '
              f"{per_seed[tseed][('similar', 'contact-learned')]*100:.1f}%/"
              f"{per_seed[tseed][('rough', 'contact-learned')]*100:.1f}%, "
              f'full-PID '
              f"{per_seed[tseed][('similar', 'full-PID')]*100:.1f}%/"
              f"{per_seed[tseed][('rough', 'full-PID')]*100:.1f}%, "
              f'contact-PID '
              f"{per_seed[tseed][('similar', 'contact-PID')]*100:.1f}%/"
              f"{per_seed[tseed][('rough', 'contact-PID')]*100:.1f}%",
              flush=True)

    print()
    print(f'{"Group":8s} | {"Arm":16s} | {"Mean missed":>11s} '
          f'| {"Std":>5s}')
    print('-' * 52)
    summary = {}
    for group_name in ('similar', 'rough'):
        for label in ('untrained', 'full-learned', 'route-replay',
                      'contact-learned', 'full-PID', 'contact-PID'):
            vals = [per_seed[s][(group_name, label)]
                    for s in TRAIN_SEEDS]
            mean = sum(vals) / len(vals)
            std = statistics.pstdev(vals)
            summary[(group_name, label)] = (mean, std)
            print(f'{group_name:8s} | {label:16s} | {mean*100:10.1f}% '
                  f'| {std*100:4.1f}')

    # Sanity anchors (seed 101 must reproduce earlier runs).
    s101_sim_full = per_seed[101][('similar', 'full-learned')]
    s101_rough_full = per_seed[101][('rough', 'full-learned')]
    s101_sim_contact = per_seed[101][('similar', 'contact-learned')]
    s101_rough_contact = per_seed[101][('rough', 'contact-learned')]
    print()
    print(f'Sanity seed 101: full-learned '
          f'{s101_sim_full*100:.1f}%/{s101_rough_full*100:.1f}% '
          f'(expect 1.9%/23.3%), contact-learned '
          f'{s101_sim_contact*100:.1f}%/{s101_rough_contact*100:.1f}% '
          f'(expect 20.3%/49.4%)')
    robust = all(
        summary[(g, 'contact-learned')][0]
        < summary[(g, 'untrained')][0]
        and summary[(g, 'contact-learned')][0]
        < summary[(g, 'route-replay')][0]
        for g in ('similar', 'rough'))
    print(f'Robustness (contact-learned mean beats untrained AND '
          f'replay on both groups): {robust}')
    for g in ('similar', 'rough'):
        print(f'{g}: full-learned '
              f"{summary[(g, 'full-learned')][0]*100:.1f}% vs full-PID "
              f"{summary[(g, 'full-PID')][0]*100:.1f}%; contact-learned "
              f"{summary[(g, 'contact-learned')][0]*100:.1f}% vs "
              f"contact-PID "
              f"{summary[(g, 'contact-PID')][0]*100:.1f}%")

    out = OUT / 'kinetra_robust_results.csv'
    with out.open('w', newline='') as f:
        cw = csv.writer(f)
        cw.writerow(['train_seed', 'holdout_group', 'arm',
                     'missed_gate_fraction'])
        cw.writerows(rows)
    print('Saved:', out.name)


if __name__ == '__main__':
    main()
