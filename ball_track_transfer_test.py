#!/usr/bin/env python3
"""Fixed-track ball experiment: training corrections vs generalization.

No third-party packages required. Run: python ball_track_transfer_test.py

Analogy:
- track = dataset, gates/side-forces = observations (data)
- ball = AI, angle + spin = dynamically controlled parameters
- controller weights = *learned correction pattern*, fixed on transfer
- resetting the ball/track tests deterministic repeatability.

This is a computational toy, NOT a test of consciousness.
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
        # Different obstacle locations from the training track.
        if kind == 'similar':
            center = max(-1.2, min(1.2, 0.82 * center + r.gauss(0, 0.24)))
            push = r.gauss(0, 0.09)
        elif kind == 'rough':
            center = max(-1.4, min(1.4, 0.60 * center + r.gauss(0, 0.48)))
            push = r.gauss(0, 0.19)
        elif kind == 'reversed':
            # Out-of-distribution: observations of the push are now misleading.
            center = max(-1.2, min(1.2, 0.82 * center + r.gauss(0, 0.24)))
            push = r.gauss(0, 0.09)
        else:
            raise ValueError(kind)
        gates.append(Gate(center, push))
    return tuple(gates)


def clip(x, a, b):
    return min(b, max(a, x))


def simulate(track, weights, reverse_sensor=False, commands=None, initial_angle=0.0, initial_spin=0.0):
    # Initial state is restored identically on every run.
    y = velocity = 0.0
    angle, spin = initial_angle, initial_spin
    misses = 0
    sum_sq_err = 0.0
    history = []
    p, d, wind, s, swind = weights
    for index, gate in enumerate(track):
        gap_error = gate.center - y
        observed_push = -gate.push if reverse_sensor else gate.push
        # The learned correction pattern acts on the ball's angle and rotation.
        if commands is None:
            angle_target = clip(p * gap_error - d * velocity - wind * observed_push, -0.9, 0.9)
            spin_target = clip(s * gap_error - swind * observed_push, -0.9, 0.9)
        else:
            angle_target, spin_target = commands[index]
        angle += 0.48 * (angle_target - angle)
        spin += 0.43 * (spin_target - spin)
        velocity = 0.48 * velocity + 0.83 * math.sin(angle) + 0.23 * spin + gate.push
        y += velocity
        gap_error_after = gate.center - y
        misses += abs(gap_error_after) > GAP_HALF_WIDTH
        sum_sq_err += gap_error_after ** 2
        history.append((index, gate.center, gate.push, y, velocity, angle, spin, gap_error_after, angle_target, spin_target))
    return sum_sq_err / len(track), misses / len(track), history


def objective(track, weights):
    mse, miss_fraction, _ = simulate(track, weights)
    return mse + 0.12 * miss_fraction


def train(track, budget=4000, seed=12345):
    r = random.Random(seed)
    weights = (0.0, 0.0, 0.0, 0.0, 0.0)
    best = objective(track, weights)
    # Derivative-free perturbation and selection, on the training track ONLY.
    # Two kinds of proposals: explore globally, then refine locally.
    for j in range(budget):
        if j % 10 == 0:
            proposal = tuple(r.uniform(-2.4, 2.4) for _ in weights)
        else:
            scale = 0.9 * (1 - j / budget) + 0.025
            proposal = tuple(clip(w + r.gauss(0, scale), -3, 3) for w in weights)
        result = objective(track, proposal)
        if result < best:
            weights, best = proposal, result
    return weights, best


def metrics(tracks, weights, reverse_sensor=False):
    vals = [simulate(t, weights, reverse_sensor) for t in tracks]
    return (sum(x[0] for x in vals) / len(vals),
            sum(x[1] for x in vals) / len(vals))


def main():
    train_track = make_track(101)
    baseline = (0.0,) * 5
    learned, fitness = train(train_track)
    assert simulate(train_track, learned) == simulate(train_track, learned), 'Replay not deterministic'
    alternative_start = simulate(train_track, learned, initial_angle=0.2, initial_spin=0.1)
    assert alternative_start == simulate(train_track, learned, initial_angle=0.2, initial_spin=0.1)
    assert alternative_start != simulate(train_track, learned), 'Changing ball starting parameters had no effect'
    training_commands = [(h[8], h[9]) for h in simulate(train_track, learned)[2]]
    replay_train = simulate(train_track, learned, commands=training_commands)
    assert replay_train[0:2] == simulate(train_track, learned)[0:2], 'Training route replay mismatch'
    print('Different initial angles and spin: PASS (different trajectory, same result on replay)')
    print('Identical replay: PASS (exactly matching position, velocity, angle and spin at all 60 gates)')
    print('Learned correction weights: ' + ', '.join(f'{w:+.3f}' for w in learned))
    print('Note: only the training track is used to choose these weights.\n')
    groups = {
        'Training track': [train_track],
        'New similar tracks (30)': [make_track(1000 + s) for s in range(30)],
        'New rough tracks (30)': [make_track(2000 + s, kind='rough') for s in range(30)],
    }
    print(f'{"Dataset":27s} | {"Mode":12s} | {"MSE":>9s} | {"Missed gates":>12s}')
    print('-' * 70)
    rows=[]
    for group, tracks in groups.items():
        for label, w in [('untrained', baseline), ('trained', learned), ('route-replay', learned)]:
            if label == 'route-replay':
                vals = [simulate(t, w, commands=training_commands) for t in tracks]
                mse, miss = (sum(v[0] for v in vals)/len(vals), sum(v[1] for v in vals)/len(vals))
            else:
                mse, miss = metrics(tracks, w)
            print(f'{group:27s} | {label:12s} | {mse:9.4f} | {miss*100:11.1f}%')
            rows.append((group, label, mse, miss))
    # A causal control: keep the same learned weights but disable their output by setting them to zero.
    # A distribution shift control: the learned sensor-to-force mapping is reversed at evaluation.
    for group, tracks in [('Similar / reversed push sensor (30)', groups['New similar tracks (30)'])]:
        for label, w in [('untrained', baseline), ('trained', learned)]:
            mse, miss = metrics(tracks, w, reverse_sensor=True)
            print(f'{group:27s} | {label:12s} | {mse:9.4f} | {miss*100:11.1f}%')
            rows.append((group, label, mse, miss))
    print('\nNo adaptation on evaluation tracks. Missed gates: abs(ball y - gate center) > 0.40.')
    print('MSE: mean square distance to the gate opening center (lower is better).')

    summary = OUT / 'ball_track_results.csv'
    with summary.open('w', newline='') as f:
        cw = csv.writer(f)
        cw.writerow(['track_group', 'ball', 'mean_squared_error', 'missed_gate_fraction'])
        cw.writerows(rows)
    example = OUT / 'ball_track_trajectories.csv'
    with example.open('w', newline='') as f:
        cw = csv.writer(f)
        cw.writerow(['ball', 'gate_index', 'opening_center', 'obstacle_push', 'ball_y', 'ball_lateral_velocity', 'ball_angle', 'ball_spin', 'signed_distance_to_opening', 'commanded_angle', 'commanded_spin'])
        new_track = groups['New similar tracks (30)'][0]
        for label, w in [('untrained', baseline), ('trained', learned)]:
            for entry in simulate(new_track, w)[2]:
                cw.writerow([label] + list(entry))
    print('Saved:', summary.name, 'and', example.name)


if __name__ == '__main__':
    main()
