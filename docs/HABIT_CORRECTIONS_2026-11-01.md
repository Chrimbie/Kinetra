# Kinetra — correction habits and transfer experiment

**Experiment:** November 1, 2026. **Status:** executed, replicated and saved.  
**Question:** Can corrections acquired from earlier ball-and-track trials become habitual, in the narrow sense of being **retained and deployed in appropriate situations** on previously unseen tracks, when starting trajectory is varied?

## What we did

We kept Kinetra's original deterministic 60-obstacle ball simulator: the ball has position, lateral velocity, angle and spin; each obstacle has an opening and lateral push. A correction consists of a target angle and target spin. The actual next position follows the existing inertial physics. The ball is a simulated mechanical system, not a conventional neural network.

* No correction parameters at the beginning: all 10 values were zero.
* The candidate habit takes the current opening error, velocity, angle and spin, and generates two bounded correction commands. It **does not preview the obstacle push**.
* An external trial-and-error selection process compared sets of corrections, scoring mean position error and missed obstacles. It **retained** corrections that performed better across **18 training tracks × 6 starting trajectories**, with **55 generations of 80 candidates**. This is an optimization process implemented by the researchers; no intrinsic physical plasticity or autonomous discovery mechanism was demonstrated.
* Corrections were then **frozen**. Evaluation used 60 unseen similar tracks (seeds 3000–3059) and 60 unseen rough tracks (4000–4059). There were separate tests with unchanged starting state, with varied initial angle and spin, and with varied starting position, velocity, angle and spin.
* The main difficult-start tests used 16 starting trajectories per track: **960 evaluation runs in each track category**, with 60 obstacles per run.

## Outcomes on varied starting trajectories

**Missed obstacles (%), lower is better.** Same held-out tracks and initial states per condition.

| Condition | 60 unseen similar tracks | 60 unseen rough tracks |
|---|---:|---:|
| Passive, no correction | 76.83% | 85.87% |
| Memorized training-route command sequence | 79.93% | 85.66% |
| Learned corrections from opening error only (same training budget) | 29.17% | 58.38% |
| Learned state-aware correction, but motion-state input removed at test time | 53.70% | 66.37% |
| **Retained correction habit applied at the appropriate moment** | **1.83%** | **28.14%** |
| Same retained feedback delayed by one obstacle | 78.55% | 77.97% |

**Timing control:** the delayed condition computes each command from the current state, but executes the *preceding* gate's command (zero on gate one). Thus this is a one-gate sensor-to-actuation delay, not merely replaying the exact same command list on every track. The distinction matters: it tests timely feedback, not a precomputed sequence.

**Start-state control:** the identical saved correction habit obtained 1.42% misses on similar tracks when starting position, velocity, angle and spin were zero; 1.45% when initial angle and spin were varied; and 1.83% when all four start parameters were varied. The current state informs each correction, rather than trying to follow a universal fixed path.

## Robustness and replication

Repeated **the entire training stage** for five independent training sets and optimizer seeds (18 distinct tracks per replication, six starts each), always evaluating on the same previously unseen 60+60 tracks and 16 held-out starting states:

| Replication | Similar missed | Rough missed |
|---|---:|---:|
| 1 | 1.83% | 28.14% |
| 2 | 1.90% | 27.99% |
| 3 | 1.52% | 27.75% |
| 4 | 1.54% | 27.66% |
| 5 | 1.78% | 28.15% |

Because these are repeated fits evaluated on the **same test tracks**, they demonstrate training robustness, not 5× more independent test evidence. Paired track-level bootstrap (resampling 60 track means, each mean including its 16 starting states) gives an approximate 95% interval for the reduction in miss rate over a passive ball: **75.00 percentage points [73.75, 76.27] on similar tracks** and **57.73 points [55.49, 60.04] on rough tracks**. These reflect variation across our synthetic test tracks, not an inference to real physical systems.

## Interpretation

**Observed:** a trial-and-error-optimized, retained *correction policy* behaves like a reusable habit: it is acquired during earlier runs, retained when practice stops, and influences action placement based on *current* ball-and-track context. The same mechanism transfers to novel arrangements. A memorized movement sequence does not. Motion state and command timing both matter greatly.

**Not established:** spontaneous habit formation by a passive physical ball, learning without an external optimizer, emergence of intelligent behavior, a novel general-purpose learning algorithm, or performance in real mechanical tracks. The system has a deliberately chosen feedback representation, a predefined 10-parameter family of possible correction tendencies, and an explicit score defining successful alignment. The learned object is a *retained mathematical feedback rule*, not a persistent material change in the ball itself.

**Important limit:** the harder rough tracks are *not* solved: approximately 28% of obstacles are still missed. The starting trajectory has a strong causal effect on actual movement, but continual sensory feedback can counter much of its effect; stating that starting conditions alone dictate everything would overstate these results.

This experiment supports the narrower claim that **corrections can become reusable, situation-dependent habitual patterns in the ball-and-track simulation**. It does *not* show that habit formation arises from alignment alone without a mechanism to retain corrections.

## How to reproduce

Requires `numpy`, otherwise only Python 3 standard library. In a folder containing the included files:

```bash
python Kinetra_habit_corrections.py
python Kinetra_habit_replications.py
```

Main script recreates learning curves, test results, per-track evaluation, trajectory traces, and training configuration. Replication script retrains on four additional independent sets and calculates track-level bootstrap intervals. Use `OPENBLAS_NUM_THREADS=1` optionally to restrict numeric library threads.

Files in this archive: original ball-and-track physics, training and replication scripts, raw results CSVs, reproducibility metadata and visual plots.