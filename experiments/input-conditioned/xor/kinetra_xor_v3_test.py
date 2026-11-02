#!/usr/bin/env python3
"""Kinetra v3-XOR: XOR under the input-conditioned shape.

Wine v3 passed its bar (0.885) but the linear map alone scored 0.982,
so on Wine the ball's dynamics added nothing: the map did the work.
This test asks whether the ball adds anything at all. XOR is the
decisive case because a linear map CANNOT solve it: the best linear
classifier on XOR is 3/4 (a line can isolate one corner, never
separate the diagonals). If ball-plus-map reaches 4/4, the ball's
dynamics supply the nonlinearity. If it cannot, the ball adds nothing
beyond the map even where the map provably fails.

Shape (same machine as Wine v3, different input map):
  - Inputs: the 4 XOR combos, bits encoded as +/-1 (EMPIRICAL choice;
    symmetric encoding, cleaner for the linear map than 0/1).
  - Fixed track: IDENTICAL to Wine v3 (10 gates, centers alternating
    +0.8/-0.8, pushes fixed seeded pattern ~ N(0, 0.09), seed 7), so
    v3-Wine and v3-XOR are the same machine with different maps.
  - Ball: base weights (5, learned, shared) plus an input-conditioned
    delta: delta = W_map @ x + b_map (W_map 5x2, learned). Effective
    weights = base + delta. Ball starts at default state (all zeros),
    rolls the fixed track, final position thresholds into 2 classes
    (1 threshold, learned, init 0.0; pred 1 iff y > t).
  - Learning: real backprop (torch autograd) through the unrolled
    physics. Full-batch Adam, lr 0.01, 1000 epochs, 5 seeds. Binary
    margin loss: true 1 -> relu(t - y), true 0 -> relu(y - t).
  - Control: linear map alone (2 -> 2 logits, cross-entropy, same
    optimizer/epochs/seeds). Linear ceiling 3/4 by the separability
    argument above; a control at 4/4 would invalidate the test.

Pre-registered bar: PASS = ball 4/4 on all 5 seeds. PARTIAL =
majority of seeds 4/4. FAIL = otherwise. Control must stay below 4/4.

Caveat (same as the old frozen-ball XOR test): only 4 combos exist,
so train = test. This is a CAPACITY test (can the machine express
XOR through weight-deltas at all?), not a generalization test. The
question differs from the old test: there, the input entered as the
track and the ball was frozen; here the input enters ONLY through
the weight-deltas, so the nonlinearity must come from the dynamics
as a function of the weights, which is a different and harder
requirement.

Mia's choices (EMPIRICAL, not original rules): +/-1 encoding, Adam
lr 0.01, 1000 epochs, threshold init 0.0, map init N(0, 0.1), base
init 0.
"""

import csv
import json
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent

TRACK_CENTERS = [0.8 if i % 2 == 0 else -0.8 for i in range(10)]
SEEDS = [12345, 12346, 12347, 12348, 12349]
EPOCHS = 1000
LR = 0.01

XOR_X = torch.tensor([[ -1., -1.], [-1., 1.], [1., -1.], [1., 1.]])
XOR_Y = torch.tensor([0, 1, 1, 0], dtype=torch.long)


def clip(x):
    return torch.clamp(x, -0.9, 0.9)


def roll(w_eff, centers, pushes):
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


def train_ball(centers, pushes, seed):
    torch.manual_seed(seed)
    base = torch.zeros(5, requires_grad=True)
    W_map = torch.randn(5, 2) * 0.1
    W_map.requires_grad_(True)
    b_map = torch.zeros(5, requires_grad=True)
    t = torch.tensor(0.0, requires_grad=True)
    opt = torch.optim.Adam([base, W_map, b_map, t], lr=LR)
    for _ in range(EPOCHS):
        opt.zero_grad()
        w_eff = base + (XOR_X @ W_map.T + b_map)
        y = roll(w_eff, centers, pushes)
        loss_true1 = F.relu(t - y)
        loss_true0 = F.relu(y - t)
        loss = torch.where(XOR_Y == 1, loss_true1, loss_true0).mean()
        loss.backward()
        opt.step()
    with torch.no_grad():
        w_eff = base + (XOR_X @ W_map.T + b_map)
        y = roll(w_eff, centers, pushes)
        pred = (y > t).long()
        acc = (pred == XOR_Y).float().mean().item()
    return {
        "accuracy": round(acc, 4),
        "correct": int((pred == XOR_Y).sum().item()),
        "final_positions": ",".join(f"{v:.6f}" for v in y.tolist()),
        "threshold": round(t.item(), 6),
        "base_weights": ",".join(f"{v:.6f}" for v in base.detach().tolist()),
    }


def train_control(seed):
    torch.manual_seed(seed)
    Wc = torch.randn(2, 2) * 0.1
    Wc.requires_grad_(True)
    bc = torch.zeros(2, requires_grad=True)
    opt = torch.optim.Adam([Wc, bc], lr=LR)
    for _ in range(EPOCHS):
        opt.zero_grad()
        logits = XOR_X @ Wc.T + bc
        loss = F.cross_entropy(logits, XOR_Y)
        loss.backward()
        opt.step()
    with torch.no_grad():
        acc = (logits.argmax(1) == XOR_Y).float().mean().item()
    return round(acc, 4)


def main():
    import random
    rng = random.Random(7)
    pushes = [rng.gauss(0.0, 0.09) for _ in range(10)]
    centers = TRACK_CENTERS

    rows = []
    for seed in SEEDS:
        ball = train_ball(centers, pushes, seed)
        ctrl = train_control(seed)
        print(f"seed {seed}: ball {ball['correct']}/4 "
              f"(acc {ball['accuracy']:.3f}) | control {ctrl:.3f} | "
              f"threshold {ball['threshold']:.3f} | "
              f"finals [{ball['final_positions']}]")
        rows.append({"seed": seed, **ball, "control_accuracy": ctrl})

    n_solved = sum(1 for r in rows if r["correct"] == 4)
    mean_ball = sum(r["accuracy"] for r in rows) / len(rows)
    mean_ctrl = sum(r["control_accuracy"] for r in rows) / len(rows)
    print(f"\nVERDICT: ball solved {n_solved}/5 seeds, mean acc "
          f"{mean_ball:.3f} | control mean {mean_ctrl:.3f} "
          f"(linear ceiling 0.75)")
    if n_solved == 5:
        print("PASS: ball 4/4 on all 5 seeds; the ball supplies the "
              "nonlinearity the linear map cannot.")
    elif n_solved >= 3:
        print("PARTIAL: majority of seeds solved; nonlinearity present "
              "but unreliable across inits.")
    else:
        print("FAIL: ball does not solve XOR through weight-deltas; "
              "the dynamics add nothing beyond the map even where the "
              "map provably fails.")
    if mean_ctrl >= 1.0:
        print("WARNING: control reached 4/4; test invalid (linear map "
              "should not be able to solve XOR).")

    with open(HERE / "kinetra_xor_v3_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "inputs": "XOR combos, bits encoded +/-1",
        "track_centers": centers, "track_pushes": pushes,
        "searches": rows,
        "seeds_solved": n_solved, "mean_ball_accuracy": mean_ball,
        "mean_control_accuracy": mean_ctrl,
        "linear_ceiling": 0.75,
    }
    with open(HERE / "kinetra_xor_v3_summary.json", "w") as f:
        json.dump(summary, f, indent=1)
    print("wrote kinetra_xor_v3_results.csv and kinetra_xor_v3_summary.json")


if __name__ == "__main__":
    main()
