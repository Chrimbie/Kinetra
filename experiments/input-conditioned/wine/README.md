# Kinetra v3: input-conditioned trajectory parameters, real backprop

## The framing (Christopher's, 2026-11-01)

Every model IS the ball. It sits at the start position with default
settings. Its parameters control how it interacts with obstacles; change
the parameters, change the trajectory. On the forward pass, the user
input changes the parameters of the trajectory. On the backward pass,
the loss flows back and the trajectory parameters change. Ball and
track can both be trained.

## What this test is

v1 and v2 had the shape inverted relative to that framing: the input
WAS the track (features as gates) and the ball was frozen (one
parameter set for all samples, tuned by random search between runs).
v3 flips it: the track is FIXED, and the input changes the ball's
parameters, with real gradients updating them.

- Data: UCI Wine, same stratified 70/30 split, seed 42, as v1/v2
  (124 train / 54 test; test counts 18/21/15; majority floor 0.389).
  Standardization uses train statistics only.
- Fixed track: 10 gates, centers alternating +0.8/-0.8, pushes a fixed
  seeded pattern ~ N(0, 0.09) (seed 7). Same for every sample.
- Ball: base weights (5, learned, shared) plus an input-conditioned
  delta: delta = W_map @ x + b_map (5x13, learned). Effective weights
  = base + delta. The ball starts at default state (all zeros) and
  rolls the fixed track with its effective weights. Final position
  bins into 3 classes via 2 learned thresholds (projected to keep
  t2 - t1 >= 0.05).
- Learning: real backprop (torch autograd) through the unrolled
  physics. Full-batch Adam, lr 0.01, 1000 epochs, 5 seeds. Same margin
  loss as v1/v2.
- Control: the linear map alone (13 -> 3 logits, cross-entropy, same
  optimizer/epochs/seeds, no ball). It isolates the ball's
  contribution.

Pre-registered bar (same as v1/v2): PASS = mean test >= 0.70,
STRONG = >= 0.90. Reported without test peeking: final-epoch test
accuracy, plus test accuracy at the best-train-accuracy epoch.

## Results

| seed | v3 train | v3 test | v3 best-train test | control test |
|------|----------|---------|--------------------|--------------|
| 12345 | 1.000 | 1.000 | 1.000 | 0.982 |
| 12346 | 1.000 | 0.815 | 0.815 | 0.982 |
| 12347 | 1.000 | 0.963 | 0.963 | 0.982 |
| 12348 | 1.000 | 0.833 | 0.870 | 0.982 |
| 12349 | 1.000 | 0.815 | 0.815 | 0.982 |
| **mean** | **1.000** | **0.885** | **0.893** | **0.982** |

Thresholds learned: t1 roughly -0.23 to -0.02, t2 roughly 0.24 to 0.36.

## Verdict: PASS, with one honest cut

**v3 passes its bar: mean final test 0.885, best seed 1.000.**
The progression tells the story: v1 (frozen ball, input-as-track)
0.422, v2 (track per feature, still frozen) 0.526, v3 (input sets the
trajectory parameters, real backprop) 0.885. His framing unblocks
Wine decisively. The frozen ball was a large part of the bottleneck,
just as the framing implied.

**But the ball is not the contributor on Wine.** The linear map alone
scores 0.982, better than the ball's 0.885. On this dataset the map
does the work and the ball's dynamics are a lossy readout, costing
about 0.10. The pass proves the FRAMING (input-conditioned parameters
plus gradients), not that the ball's dynamics carry the
classification. Wine is easy for linear methods, so this is the wrong
dataset to prove the ball adds anything.

What would prove the ball adds value: a task where the linear map
fails but ball-plus-map succeeds. XOR is the natural candidate (a
linear map cannot do XOR; the earlier frozen-ball XOR test learned
4/4, but under the old shape). That test is named, not ordered.

## Caveats

- Train accuracy is 1.000 on all seeds; test spread across seeds is
  wide (0.815-1.000), so single-seed numbers should not be trusted.
- The fixed track is arbitrary (an alternating slalom). It is a
  runway for the dynamics, not a meaningful obstacle course. A
  different fixed track could change the numbers.
- Thresholds came out narrow (t2 - t1 roughly 0.3-0.5). The bins are
  thin slices of the final-position range.
- Track training (the second half of his framing, where the gate
  layout itself learns) is NOT tested here. It is the named
  follow-up, along with XOR under the v3 shape.

## Files

- `kinetra_wine_v3_test.py` — the test (torch autograd, CPU).
- `kinetra_wine_v3_results.csv` — per-seed results.
- `kinetra_wine_v3_summary.json` — summary including the fixed track.

Mia's choices (EMPIRICAL, not original rules): 10-gate alternating
track, fixed push pattern seed 7, init scales (base 0, map N(0, 0.1),
thresholds -0.5/0.5), Adam lr 0.01, 1000 epochs, threshold-gap
projection 0.05.
