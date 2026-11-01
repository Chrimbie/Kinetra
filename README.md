# Kinetra

**Kinetra** is an exploratory computational experiment for learning and transferring movement corrections across obstacle tracks.

## Concept

The original analogy assigns:

- **Track** = dataset
- **Obstacles / gates** = observations (data)
- **Moving ball** = AI system
- **Ball angle and spin** = parameters that affect its trajectory
- **Starting configuration** = resettable state, enabling repeatable runs
- **Correction pattern** = learned behavior that adjusts movement based on the current obstacles

The track geometry is fixed during each run. Training changes a small set of correction-controller weights on **one training track**. Those learned weights are frozen for evaluation on other tracks.

**This prototype uses a predefined corrective-feedback mechanism.** It does not discover the mechanism from scratch, establish broad machine intelligence, or test consciousness.

## Quick start

Python 3.9+; the original and alignment-only experiments use only the standard library. The correction-habits experiment additionally requires `numpy`.

```bash
python ball_track_transfer_test.py
python -m unittest discover -s tests -v
```

The experiment writes `ball_track_results.csv` and `ball_track_trajectories.csv` alongside the Python script. Reference outputs from the initial run live in `results/`, including an SVG benchmark chart.

## Initial benchmark (November 1, 2026)

Each track contains 60 gates. The training optimization evaluates 4,000 trial weight vectors on exactly one track (seed 101). The holdouts are 30 similar tracks (seeds 1000–1029) and 30 rough tracks (seeds 2000–2029). Gate misses occur when the ball exits the gate's ±0.40-wide clearance.

| Dataset | Untrained missed gates | Learned correction missed gates | Memorized route missed gates |
|---|---:|---:|---:|
| Training track | 75.0% | 0.0% | 0.0% |
| 30 unseen similar tracks | 60.8% | **1.9%** | 78.9% |
| 30 unseen rough tracks | 84.5% | **23.3%** | 87.2% |

Lower is better. Percentages are fractions of missed gates pooled within each group; see `results/ball_track_results.csv` for unrounded figures and MSE.

### Interpretation

The tested correction policy transfers substantially better than replaying the exact movement commands from the training track. The feedback policy reads the current obstacle position and side-force; this is a transfer test of a small controller, not proof that intelligent rules emerged spontaneously.

## Project files

- `ball_track_transfer_test.py`: standalone reproducible experiment (Python standard library only).
- `tests/test_kinetra.py`: repeatability, training, and held-out-track regression tests.
- `results/`: snapshot of benchmark CSVs and visual SVG plot.
- `docs/NEXT_EXPERIMENTS.md`: proposed controls and follow-up experiments.

## Status

Research prototype, not production software. No external dataset or physical ball is used. Avoid interpreting toy-model improvements as general-purpose learning or consciousness.

## Follow-up experiments (November 1, 2026)

**Kinetra is a literal simulated ball-and-track system.** Model-like behavior is what the experiments observe; the ball does not possess cognition. In this project, a **habit** means a correction tendency retained from earlier trials. **Generalization** means deploying that tendency at context-appropriate moments on new tracks, not replaying memorized movement. Initial position, velocity, angle and spin are explicitly varied and affect the trajectory.

### Alignment-only control (no training or retained state)

A fixed, hand-designed rule continuously realigns the simulated ball using its current motion state and the next gate's opening. With canonical starts, missed-gate rates were **4.19% vs 62.25% passive** on 60 unseen similar tracks and **33.94% vs 83.06% passive** on 60 rough tracks. Ignoring motion state increased similar-track misses to **46.19%**. This demonstrates engineered feedback, **not** the acquisition of habitual corrections. [Full method and limits](docs/ALIGNMENT_ONLY_2026-11-01.md).

### Retained corrections as habits (trial-and-error training)

An external, derivative-free optimizer started with **10 zero correction weights**, then evaluated candidates across **18 training tracks × 6 different starting trajectories**, using **55 generations × 80 candidates**. The final feedback rule was frozen and tested on 60 unseen similar and 60 unseen rough tracks, with 16 varied initial states per track. Missed-gate rates (lower is better):

| Frozen condition | Similar tracks | Rough tracks |
|---|---:|---:|
| Passive / no corrections | 76.83% | 85.87% |
| Memorized command sequence | 79.93% | 85.66% |
| Learned correction from opening error alone | 29.17% | 58.38% |
| **Retained state-aware correction habit** | **1.83%** | **28.14%** |
| Same feedback applied one gate late | 78.55% | 77.97% |

The main result reproduced across **five independently optimized training runs** (similar-track misses **1.52–1.90%**, rough-track misses **27.66–28.15%**, measured on the same held-out test sets). Starting motion state and correction timing were important. Rough tracks remained challenging.

**Scope of the finding:** A purposefully defined ball-and-track simulator plus an **external optimizer** can acquire reusable, context-sensitive correction rules. This does **not** establish spontaneous physical habit formation, learning through passive alignment alone, general-purpose intelligence, or results from a real rolling ball. [Full report and methodology](docs/HABIT_CORRECTIONS_2026-11-01.md).

To reproduce (run from repository root):

```bash
python kinetra_alignment_test.py
python kinetra_alignment_diagnostics.py
python -m pip install numpy
python Kinetra_habit_corrections.py
python Kinetra_habit_replications.py
```

Selected published data: [alignment summary](results/kinetra_alignment_summary.csv), [initial-state diagnostics](results/kinetra_alignment_diagnostics.csv), [habit-condition comparisons](results/Kinetra_habit_summary.csv), [learning curve](results/Kinetra_habit_learning_curve.csv), [replication results](results/Kinetra_habit_replications.csv), [paired-bootstrap intervals](results/Kinetra_habit_robustness.csv). The scripts can regenerate larger per-trial CSVs locally; those are not checked in.
