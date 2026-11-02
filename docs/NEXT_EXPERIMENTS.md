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

## Status update (2026-11-01)

Results from the first round of follow-up experiments now live under `experiments/`. The original text above is unchanged; this section only records what each follow-up now stands as.

- #1 (more seeds, with confidence intervals): ADDRESSED — `experiments/robustness/` repeats the contact-only test across 5 training seeds and reports means and ranges.
- #2 (PID / random / no-feedback baselines): ADDRESSED, with a caveat — `experiments/robustness/` compares learned controllers against a hand-tuned PID controller, untrained, and replay baselines, but the budgets were not equal (4,000 learned trials vs 150/576-eval PID grids), so the comparison favors the learned side.
- #4 (remove or restrict supplied center and force inputs): ADDRESSED — `experiments/contact-only/` uses only the ball's own state plus the previous gate's contact side, with no handed center or force.
- #5 (noise, delays, gate widths, changing dynamics): ADDRESSED — `experiments/stress/` covers measurement noise, feedback delay, narrow gates, and a mid-run force change. Delay is the sharp edge: +1 gate of delay collapses performance, +2 gates falls below untrained.
- #3 (untouched final test set): PARTIAL — holdout tracks were never trained on, but the same holdout sets were reused across experiments; no pristine final set has been held back.
- #7 (learning vs generalization thresholds): PARTIAL — each test used pre-registered pass criteria, but no formal learning-vs-generalization distinction has been drawn.
- #6 (online adaptation vs frozen transfer): OPEN — the stress test's mid-run change used frozen weights; true online adaptation is untested.

New direction, not an answer to any follow-up above: `experiments/pass-through/` verifies that a problem can be passed through with contact recorded (parts of a problem as answers, fit rules enforced), and `experiments/learned-aiming/` shows aiming at a part learned rather than hand-set.
