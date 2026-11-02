# Input-conditioned parameters: the ball as Christopher frames it

## The framing (Christopher's, 2026-11-01)

Every model IS the ball. It sits at the start position with default
settings. Its parameters control how it interacts with obstacles;
change the parameters, change the trajectory. On the forward pass, the
user input changes the parameters of the trajectory. On the backward
pass, the loss flows back and the trajectory parameters change. Ball
and track can both be trained.

## What changed from earlier experiments

Earlier Kinetra tests had the shape inverted relative to that framing:
the input WAS the track (features as gates) and the ball was frozen
(one parameter set for all samples, tuned by random search between
runs). Here the track is FIXED and the input changes the ball's
parameters, with real gradients updating them. The fixed track is the
same in both experiments below (10 gates, centers alternating
+0.8/-0.8, pushes a fixed seeded pattern ~ N(0, 0.09), seed 7), so they
are the same machine with different input maps.

## Wine: the framing passes, the ball adds nothing (linearly easy data)

`wine/` — UCI Wine, same split and bar as the earlier frozen-ball wine
tests. Mean test 0.885, best seed 1.000, train 1.000 on all seeds: PASS,
against the bar v1 (0.422) and v2 (0.526) failed. But the linear map
alone scored 0.982, better than the ball. On Wine the map does the
work and the ball's dynamics are a lossy readout. The pass proves the
framing, not that the ball's dynamics carry the classification.

## XOR: the ball supplies what the map cannot (linearly impossible data)

`xor/` — the 4 XOR combos, bits encoded as +/-1. Ball 4/4 on all 5
seeds: PASS. Control (linear map alone) mean 0.550, max 0.750, exactly
the linear ceiling: a line can isolate one corner of XOR, never
separate the diagonals. Where the map provably fails, the ball
supplies the nonlinearity. Caveat: three seeds solve with thin margins
(as small as 0.002 past the threshold), because margin loss stops
pushing once a sample is correct. The solutions are real but fragile;
a margin-maximizing loss would test robustness.

## Combined reading

The ball is not a redundant readout. It contributes exactly where
linearity fails. Wine and XOR together show both directions: the map
beats the ball where the data is linearly easy, the ball beats the map
where the data is linearly impossible.

## Still open

- Track training: the second half of the framing, where the gate
  layout itself learns. Not tested here.
- XOR robustness under a margin-maximizing loss.
- Online adaptation (follow-up #6) remains open.

## Files

- `wine/` — script, README, per-seed results, summary.
- `xor/` — script, README, per-seed results, summary.
