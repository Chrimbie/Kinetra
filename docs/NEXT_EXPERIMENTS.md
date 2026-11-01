# Kinetra: next experiments

## Hypothesis

A repeatable moving-agent system can improve performance on novel obstacle sequences by learning correction patterns on an initial sequence, without memorizing the original route.

## Implemented controls

1. **Deterministic restart:** identical initial state and parameters lead to identical trajectories.
2. **Changed starting angle / spin:** changing the ball's starting conditions changes its trajectory.
3. **Open-loop replay:** same commands from training replayed on unfamiliar tracks fail, distinguishing feedback from memorization.
4. **No-adaptation holdout:** learned controller parameters are frozen before test tracks.
5. **Reversed side-force sensor:** a simple out-of-distribution sensor manipulation tests sensitivity to input interpretation.

## Recommended follow-ups

1. Test substantially more training seeds and holdout seeds, with confidence intervals.
2. Compare against a conventional hand-tuned PID controller, random search, and a no-feedback policy under equal evaluation budgets.
3. Avoid tuning on test tracks; preserve a truly untouched final test set.
4. Remove or restrict the supplied obstacle center and force inputs to assess what the ball can discover through contact, trial and error, or delayed feedback.
5. Introduce randomized measurement noise, sensor delays, wider and narrower gates, and changing dynamics.
6. Measure performance when the obstacle structure changes mid-run (online adaptation), separately from frozen-policy transfer.
7. Distinguish **learning** (improvement on training experience) from **generalization** (improvement on new observations) using explicit thresholds.

The goal is to evaluate mechanisms for adaptive behavior. None of these experiments would, by itself, establish consciousness.
