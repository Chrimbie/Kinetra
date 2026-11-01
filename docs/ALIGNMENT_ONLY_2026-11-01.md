# Kinetra — Alignment without learning: controlled simulation

## Research question
Can a physical ball-and-track simulator exhibit transferable, apparently
habitual correction behavior through moment-to-moment alignment alone,
without modifying parameters or retaining memory?

## Conclusion
**Yes, in the narrow sense that a human-designed, fixed trajectory-aware
alignment rule traversed many unseen tracks with fewer misses, without any
parameter updates or retained state.** This is *not* evidence of the ball
acquiring habits or learning spontaneously. The alignment procedure was
explicitly designed using the simulator's known equations.

## Mechanism and transparency
The experiment reuses the existing Kinetra toy simulator (60 gates,
gate half-width 0.4, instantaneous lateral position and velocity,
angle and spin with inertia, and a push disturbance at each gate).

For each gate, an alignment rule attempts to make the ball's lateral position
at the next gate equal the opening center. It uses the current position,
velocity, angle and spin to calculate an appropriate target angle from the
simulator's one-step dynamics. Its spin target is always zero.

The main rule does NOT know the upcoming push, and has **zero learnable
weights, no training, no memory, no persistent adaptation and no trial-based
tuning**. It knows the next opening center, and it has exact access to
current ball state and physics. Thus results demonstrate fixed, engineered
feedback, not emergent intelligence.

Comparison conditions:
- Passive: target angle and spin zero.
- Point to opening: point angle in proportion to instantaneous positional
  difference, ignoring velocity and other ball state.
- Align now: exact one-step inverse dynamics aiming to align with center,
  assuming the next disturbance is zero.
- Ignore motion: identical inverse calculation, but incorrectly sets the
  current velocity, angle, spin to zero at every step.
- Oracle align: as above, but knows next disturbance (unrealistically).
- Prior trained comparator: 5 optimized weights trained only on original
  track 101 using existing experiment code. No re-training on evaluation tracks.

## Evaluation
- 60 novel similar tracks (seeds 1000–1059).
- 60 novel rough tracks (seeds 2000–2059).
- Each track is tested at four start-state categories: canonical (1),
  changed angle/spin (40), moderately changed full state (40),
  and severely changed full state (40). Same start profiles reused for each
  track and for every controller to make comparisons paired.
- 6 controller conditions, totaling 87,120 simulated runs and 5,227,200 gate
  interactions. Each run resets the state; no training or parameters persist.
- Deterministic exact replay checked before and after intervening runs.

## Missed gates (lower is better; canonical starts)

| Controller | Similar (60 tracks) | Rough (60 tracks) |
|---|---:|---:|
| Passive | 62.25% | 83.06% |
| Point to opening | 31.11% | 60.75% |
| Alignment, no push preview | 4.19% | 33.94% |
| Alignment but ignore motion | 46.19% | 64.22% |
| Alignment, with oracle push preview | 2.92% | 29.78% |
| Earlier trained controller | 2.11% | 24.67% |

## Severe variation in the starting trajectory

| Controller | Similar (2400 runs) | Rough (2400 runs) |
|---|---:|---:|
| Passive | 82.36% | 86.47% |
| Alignment, no push preview | 5.22% | 34.23% |
| Alignment but ignore motion | 50.07% | 65.55% |
| Earlier trained controller | 3.37% | 25.51% |

When start state is severely perturbed, the alignment controller misses
40.04% of *first* gates on similar tracks but only 3.67% of the *last
five* gates. On rough tracks, its first gate miss rate is 44.21% and the last
five miss rate 34.84%. Some of this decline can be attributable to changing
gate difficulty, but the positions additionally converge: on similar
tracks, the mean position difference from the canonical-start run
falls from 0.3587 at gate one to near zero at gate 60. On rough tracks,
it falls from 0.4108 to 0.0180.

Correctly taking account of **current trajectory** is essential:
ignoring the state of motion increased similar-track misses from
4.19% to 46.19% under canonical starts.

## Repeated attempts, habit and generalization
Identical track + identical starting trajectory + identical fixed rule
yield an identical entire trajectory, including after other trials.
Therefore **no improvement from run to run is possible** without some
persistent change in either the ball/controller or the track/environment.

The fixed alignment rule responds differently at different obstacles
and transfers its *pre-designed* action selection to unfamiliar tracks.
This is generalization-like *behavioral transfer*, not acquired
generalization or a newly learned habit.

A more ambitious next experiment could add an explicit, untrained physical
mechanism (e.g., persistent mechanical settling, hysteresis, or frictional
changes), to see whether physical alignment actually *creates* a durable
habit rather than simply executing a fixed correction law. The mechanism
must be specified and tested without assuming the result.

## Limits
This is a deterministic, simplified one-dimensional lateral ball model
with angle and spin, not a realistic rolling 3D ball, collision simulation,
human speech model, or newly discovered learning algorithm.
There is an explicit gate-center target and a programmer-provided rule.
Results are sample means for the indicated seeded test tracks, not a
general performance guarantee. Runs sharing the same track are paired and
not statistically independent.

## Reproduction
Requirements: Python 3 standard library only (matplotlib is not needed
for the experiments; the plot is bundled).

Run in this directory:
```
python kinetra_alignment_test.py
python kinetra_alignment_diagnostics.py
```
Source: ball_track_transfer_test.py (the original Kinetra simulator),
kinetra_alignment_test.py (main experiment),
kinetra_alignment_diagnostics.py (start-state sensitivity).
Outputs: kinetra_alignment_summary.csv,
kinetra_alignment_all_trials.csv,
kinetra_alignment_diagnostics.csv,
kinetra_alignment_example_path.csv,
kinetra_start_recovery.png,
kinetra_alignment_metadata.json.