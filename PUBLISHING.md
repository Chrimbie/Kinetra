# Reproducing and contributing to Kinetra

Repository: https://github.com/Chrimbie/Kinetra

Run with Python 3.9+ (standard library only):

```bash
python ball_track_transfer_test.py
python -m unittest discover -s tests -v
```

The experiment writes `ball_track_results.csv` and `ball_track_trajectories.csv`. The frozen initial results are stored in `results/`. The SVG figure visualizes the archived missed-gate percentages.

Do not use unseen evaluation tracks to tune the controller. Check `docs/NEXT_EXPERIMENTS.md` before making claims about generalization.
