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

Python 3.9+; no external dependencies.

```bash
python ball_track_transfer_test.py
python -m unittest discover -s tests -v
```

The experiment writes `ball_track_results.csv` and `ball_track_trajectories.csv` alongside the Python script. Reference outputs from the initial run live in `results/`.

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
- `results/`: snapshot of benchmark CSVs and visual plot.
- `docs/NEXT_EXPERIMENTS.md`: proposed controls and follow-up experiments.

## Status

Research prototype, not production software. No external dataset or physical ball is used. Avoid interpreting toy-model improvements as general-purpose learning or consciousness.
