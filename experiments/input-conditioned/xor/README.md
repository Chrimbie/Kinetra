# Kinetra v3-XOR: does the ball add anything the map cannot?

## The question

Wine v3 passed its bar (0.885) but the linear map alone scored 0.982,
so on Wine the ball's dynamics added nothing: the map did the work.
This test asks whether the ball adds anything at all. XOR is the
decisive case because a linear map CANNOT solve it: the best linear
classifier on XOR is 3/4 (a line can isolate one corner, never
separate the diagonals). If ball-plus-map reaches 4/4, the ball's
dynamics supply the nonlinearity. If it cannot, the ball adds nothing
beyond the map even where the map provably fails.

## Shape

The same machine as Wine v3, with a different input map:

- Inputs: the 4 XOR combos, bits encoded as +/-1.
- Fixed track: IDENTICAL to Wine v3 (10 gates, centers alternating
  +0.8/-0.8, pushes fixed seeded pattern ~ N(0, 0.09), seed 7).
- Ball: base weights (5, learned, shared) plus an input-conditioned
  delta: delta = W_map @ x + b_map (5x2, learned). Effective weights
  = base + delta. Ball starts at default state (all zeros), rolls the
  fixed track, final position thresholds into 2 classes (1 threshold,
  learned, init 0.0; pred 1 iff y > t).
- Learning: real backprop (torch autograd) through the unrolled
  physics. Full-batch Adam, lr 0.01, 1000 epochs, 5 seeds. Binary
  margin loss: true 1 -> relu(t - y), true 0 -> relu(y - t).
- Control: linear map alone (2 -> 2 logits, cross-entropy, same
  optimizer/epochs/seeds). Linear ceiling 3/4.

Pre-registered bar: PASS = ball 4/4 on all 5 seeds. Control must stay
below 4/4 or the test is invalid.

## Results

| seed | ball | control | threshold | final positions (combos 00, 01, 10, 11) |
|------|------|---------|-----------|------------------------------------------|
| 12345 | 4/4 | 0.500 | 0.165 | 0.144, 0.167, 0.168, 0.142 |
| 12346 | 4/4 | 0.500 | 0.133 | 0.055, 0.163, 0.193, -0.181 |
| 12347 | 4/4 | 0.750 | 0.100 | 0.044, 0.145, 0.138, 0.080 |
| 12348 | 4/4 | 0.500 | 0.079 | -1.011, 0.158, 0.185, -0.014 |
| 12349 | 4/4 | 0.500 | 0.128 | 0.016, 0.144, 0.163, 0.086 |
| **mean** | **4/4** | **0.550** | | |

## Verdict: PASS

**Ball 4/4 on all 5 seeds; control mean 0.550, max 0.750 (exactly the
linear ceiling on one seed, as theory predicts).** The ball supplies
the nonlinearity the linear map cannot. Combined with Wine v3, the
picture is now complete in both directions: on Wine the map beat the
ball because Wine is linearly easy; on XOR the ball beats the map
because XOR is linearly impossible. The ball is not a redundant
readout. It contributes exactly where linearity fails.

## Caveats

- **Thin margins on three seeds.** Seeds 12345, 12347, and 12349 solve
  with margins as small as 0.002-0.045 past the threshold. This is
  expected from the loss, not a surprise: margin loss stops pushing
  once a sample is correct, so there is no pressure to widen margins.
  The solutions are real (4/4 on every seed) but fragile to small
  perturbations. A margin-maximizing loss would test robustness; it
  is named, not run.
- **Capacity, not generalization.** Only 4 combos exist, so train =
  test, same caveat as the old frozen-ball XOR test. What is new here
  is the requirement: the input enters ONLY through the
  weight-deltas, so the nonlinearity must come from the dynamics as
  a function of the weights, which is a different and harder
  requirement than the old input-as-track version.
- The fixed track is the same arbitrary slalom as Wine v3. A
  different fixed track could change the margins.

## Files

- `kinetra_xor_v3_test.py` — the test (torch autograd, CPU).
- `kinetra_xor_v3_results.csv` — per-seed results.
- `kinetra_xor_v3_summary.json` — summary including the fixed track.

Mia's choices (EMPIRICAL, not original rules): +/-1 encoding, Adam
lr 0.01, 1000 epochs, threshold init 0.0, map init N(0, 0.1), base
init 0.
