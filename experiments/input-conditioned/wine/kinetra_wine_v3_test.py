#!/usr/bin/env python3
"""Kinetra v3: the ball as Christopher frames it.

His framing (2026-11-01): every model IS the ball. It sits at the start
position with default settings. Its parameters control how it interacts
with obstacles; change the parameters, change the trajectory. On the
forward pass the user input changes the parameters of the trajectory.
On the backward pass the loss flows back and the trajectory parameters
change. Ball and track can both be trained (track training is deferred
to a follow-up; here the track is fixed).

This inverts v1/v2. There, the input WAS the track (features as gates)
and the ball was frozen. Here the track is fixed and the input changes
the ball's parameters.

Setup (confirmed shape, 2026-11-01):
  - Data: UCI Wine, same stratified 70/30 split, seed 42, as v1/v2
    (124 train / 54 test; test counts 18/21/15; majority floor 0.389).
    Standardization uses TRAIN statistics only.
  - Fixed track: 10 gates, centers alternating +0.8/-0.8, pushes a
    fixed seeded pattern ~ N(0, 0.09) (seed 7). Same for every sample.
    Nonzero centers AND pushes matter: with all-zero centers, pushes,
    and initial state the ball would never move and no weight could
    have any effect.
  - Ball: base weights (5, learned, shared) plus an input-conditioned
    delta: delta = W_map @ x + b_map (W_map 5x13, learned). Effective
    weights = base + delta. The ball starts at default state
    (y=vel=angle=spin=0) and rolls the fixed track with its effective
    weights. Final position bins into 3 classes via 2 learned
    thresholds (projected to keep t2 - t1 >= 0.05).
  - Learning: REAL backprop (torch autograd) through the unrolled
    physics. Full-batch Adam, lr 0.01, 1000 epochs, 5 seeds. Loss is the
    same margin loss as v1/v2 (0 if final y lands in the correct bin,
    else distance to the nearest edge).
  - Control: the linear map alone (13 -> 3 logits, cross-entropy,
    same optimizer/epochs/seeds, no ball). It isolates the ball's
    contribution: ball vs map-alone.

Pre-registered bar (same as v1/v2):
  - PASS: mean test accuracy >= 0.70. STRONG: >= 0.90.
  - Reported WITHOUT test peeking: final-epoch test accuracy, plus
    test accuracy at the best-TRAIN-accuracy epoch (train-selected
    checkpoint, legitimate).

Mia's choices (EMPIRICAL, not original rules): 10-gate alternating
track, fixed push pattern seed 7, init scales (base 0, map N(0, 0.1),
thresholds -0.5/0.5), Adam lr 0.01, 1000 epochs, threshold-gap
projection 0.05.
"""

import copy
import csv
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

HERE = Path(__file__).resolve().parent

TRACK_CENTERS = [0.8 if i % 2 == 0 else -0.8 for i in range(10)]
TRACK_PUSHES = None  # set in main from seed 7
SEEDS = [12345, 12346, 12347, 12348, 12349]
EPOCHS = 1000
LR = 0.01
MIN_GAP = 0.05


def clip(x):
    return torch.clamp(x, -0.9, 0.9)


def roll(w_eff, centers, pushes):
    """Unrolled ball physics, batched over samples. Returns final y."""
    y = torch.zeros(w_eff.shape[0], dtype=torch.float32)
    vel = torch.zeros_like(y)
    angle = torch.zeros_like(y)
    spin = torch.zeros_like(y)
    p, d, wind, s, swind = w_eff[:, 0], w_eff[:, 1], w_eff[:, 2], w_eff[:, 3], w_eff[:, 4]
    for c, u in zip(centers, pushes):
        gap = c - y
        a_t = clip(p * gap - d * vel - wind * u)
        s_t = clip(s * gap - swind * u)
        angle = 0.52 * angle + 0.48 * a_t
        spin = 0.57 * spin + 0.43 * s_t
        vel = 0.48 * vel + 0.83 * torch.sin(angle) + 0.23 * spin + u
        y = y + vel
    return y


def predict_bins(y, t1, t2):
    return torch.where(y < t1, torch.zeros_like(y, dtype=torch.long),
                       torch.where(y < t2, torch.ones_like(y, dtype=torch.long),
                                   2 * torch.ones_like(y, dtype=torch.long)))


def margin_loss(y, true, t1, t2):
    pred = predict_bins(y, t1, t2)
    loss = torch.zeros_like(y)
    wrong = pred != true
    # class 0 wrong: y >= t1, loss = y - t1
    loss = torch.where(wrong & (true == 0), y - t1, loss)
    # class 1 wrong: y < t1 -> t1 - y; y >= t2 -> y - t2
    loss = torch.where(wrong & (true == 1) & (y < t1), t1 - y, loss)
    loss = torch.where(wrong & (true == 1) & (y >= t2), y - t2, loss)
    # class 2 wrong: y < t2, loss = t2 - y
    loss = torch.where(wrong & (true == 2), t2 - y, loss)
    return loss.mean()


def accuracy(y, true, t1, t2):
    return (predict_bins(y, t1, t2) == true).float().mean().item()


def train_v3(X_train, y_train, X_test, y_test, centers, pushes, seed):
    torch.manual_seed(seed)
    base = torch.zeros(5, requires_grad=True)
    W_map = torch.randn(5, 13) * 0.1
    W_map.requires_grad_(True)
    b_map = torch.zeros(5, requires_grad=True)
    t1 = torch.tensor(-0.5, requires_grad=True)
    t2 = torch.tensor(0.5, requires_grad=True)
    params = [base, W_map, b_map, t1, t2]
    opt = torch.optim.Adam(params, lr=LR)

    best_train_acc = -1.0
    best_state = None
    for _ in range(EPOCHS):
        opt.zero_grad()
        w_eff = base + (X_train @ W_map.T + b_map)
        y = roll(w_eff, centers, pushes)
        loss = margin_loss(y, y_train, t1, t2)
        loss.backward()
        opt.step()
        with torch.no_grad():
            if (t2 - t1).item() < MIN_GAP:
                t2.data = t1.data + MIN_GAP
        with torch.no_grad():
            acc = accuracy(y, y_train, t1, t2)
            if acc > best_train_acc:
                best_train_acc = acc
                best_state = (base.detach().clone(), W_map.detach().clone(),
                              b_map.detach().clone(), t1.detach().clone(),
                              t2.detach().clone())
    with torch.no_grad():
        w_eff = base + (X_train @ W_map.T + b_map)
        y_tr = roll(w_eff, centers, pushes)
        final_train = accuracy(y_tr, y_train, t1, t2)
        w_eff_te = base + (X_test @ W_map.T + b_map)
        y_te = roll(w_eff_te, centers, pushes)
        final_test = accuracy(y_te, y_test, t1, t2)
        b0, bW, bb, bt1, bt2 = best_state
        w_eff_best = b0 + (X_test @ bW.T + bb)
        y_best = roll(w_eff_best, centers, pushes)
        best_test = accuracy(y_best, y_test, bt1, bt2)
    return {
        "final_train_accuracy": round(final_train, 4),
        "final_test_accuracy": round(final_test, 4),
        "best_train_accuracy": round(best_train_acc, 4),
        "best_train_test_accuracy": round(best_test, 4),
        "threshold_t1": round(t1.item(), 6),
        "threshold_t2": round(t2.item(), 6),
        "base_weights": ",".join(f"{v:.6f}" for v in base.detach().tolist()),
    }


def train_control(X_train, y_train, X_test, y_test, seed):
    torch.manual_seed(seed)
    Wc = torch.randn(3, 13) * 0.1
    Wc.requires_grad_(True)
    bc = torch.zeros(3, requires_grad=True)
    opt = torch.optim.Adam([Wc, bc], lr=LR)
    best_train_acc = -1.0
    best_state = None
    for _ in range(EPOCHS):
        opt.zero_grad()
        logits = X_train @ Wc.T + bc
        loss = F.cross_entropy(logits, y_train)
        loss.backward()
        opt.step()
        with torch.no_grad():
            acc = (logits.argmax(1) == y_train).float().mean().item()
            if acc > best_train_acc:
                best_train_acc = acc
                best_state = (Wc.detach().clone(), bc.detach().clone())
    with torch.no_grad():
        final_test = (X_test @ Wc.T + bc).argmax(1).eq(y_test).float().mean().item()
        bW, bb = best_state
        best_test = (X_test @ bW.T + bb).argmax(1).eq(y_test).float().mean().item()
    return round(final_test, 4), round(best_test, 4)


def main():
    import random
    rng = random.Random(7)
    pushes = [rng.gauss(0.0, 0.09) for _ in range(10)]
    centers = TRACK_CENTERS

    data = load_wine()
    X_all, y_all = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.3, random_state=42, stratify=y_all)
    means = X_train.mean(axis=0)
    stds = X_train.std(axis=0)
    stds[stds == 0] = 1.0
    X_train_z = torch.tensor((X_train - means) / stds, dtype=torch.float32)
    X_test_z = torch.tensor((X_test - means) / stds, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    y_test_t = torch.tensor(y_test, dtype=torch.long)
    n_test = len(y_test)
    majority_floor = max(int((y_test == c).sum()) for c in (0, 1, 2)) / n_test
    print(f"train {len(y_train)} / test {n_test} | "
          f"majority floor on test: {majority_floor:.3f} | "
          f"fixed track: 10 gates, centers alternate +-0.8")

    rows = []
    for seed in SEEDS:
        v3 = train_v3(X_train_z, y_train_t, X_test_z, y_test_t, centers, pushes, seed)
        ctrl_final, ctrl_best = train_control(X_train_z, y_train_t, X_test_z, y_test_t, seed)
        print(f"seed {seed}: v3 final train {v3['final_train_accuracy']:.3f} / "
              f"test {v3['final_test_accuracy']:.3f} | best-train test "
              f"{v3['best_train_test_accuracy']:.3f} (train "
              f"{v3['best_train_accuracy']:.3f}) | control final test "
              f"{ctrl_final:.3f} / best-train test {ctrl_best:.3f} | "
              f"thresholds ({v3['threshold_t1']:.3f}, {v3['threshold_t2']:.3f})")
        rows.append({"seed": seed, **v3,
                     "control_final_test_accuracy": ctrl_final,
                     "control_best_train_test_accuracy": ctrl_best})

    mean_final = sum(r["final_test_accuracy"] for r in rows) / len(rows)
    mean_best = sum(r["best_train_test_accuracy"] for r in rows) / len(rows)
    mean_ctrl = sum(r["control_final_test_accuracy"] for r in rows) / len(rows)
    best_final = max(r["final_test_accuracy"] for r in rows)
    print(f"\nVERDICT: v3 mean final test {mean_final:.3f}, best {best_final:.3f}, "
          f"mean best-train test {mean_best:.3f} | control mean final test "
          f"{mean_ctrl:.3f} | majority floor {majority_floor:.3f} "
          f"(v1 0.422, v2 0.526)")
    if mean_final >= 0.90:
        print("STRONG: mean final test accuracy clears 0.90.")
    elif mean_final >= 0.70:
        print("PASS: mean final test accuracy clears 0.70.")
    else:
        print("FAIL: mean final test accuracy below 0.70.")

    with open(HERE / "kinetra_wine_v3_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "train_size": len(y_train), "test_size": n_test,
        "majority_floor_test": majority_floor,
        "track_centers": centers, "track_pushes": pushes,
        "searches": rows,
        "mean_final_test_accuracy": mean_final,
        "mean_best_train_test_accuracy": mean_best,
        "mean_control_final_test_accuracy": mean_ctrl,
        "best_final_test_accuracy": best_final,
    }
    with open(HERE / "kinetra_wine_v3_summary.json", "w") as f:
        json.dump(summary, f, indent=1)
    print("wrote kinetra_wine_v3_results.csv and kinetra_wine_v3_summary.json")


if __name__ == "__main__":
    main()
