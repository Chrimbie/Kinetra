# Dataset-as-course experiment: protocol fixed before the benchmark

Date: November 4, 2026 (America/Chicago).

## Question

Can one continuous simulated ball trajectory hit the correct answer zone at
every obstacle when each Wine example is a section of the course? Can training
the starting trajectory alone do this, and what changes when six global motion
parameters are also trained?

## Fixed encoding and dynamics

- Wine's 13 feature values become 13 sequential sideways forces. Each feature
  is standardized with training statistics and mapped to `0.25*tanh(z/2.5)`.
- One station (one problem) ends after those 13 forces. Every obstacle has the
  same three pads centered at -1, 0 and +1, representing classes 0, 1 and 2.
- A **correct hit** is lateral position within 0.32 of the true class's pad.
  Nearest-class accuracy and contact with any pad are separate metrics.
- All pads exist for every example. The label never enters geometry, forces,
  the state transition, collision response, initial state, or inference.
- Position, velocity, angle and spin carry across all station boundaries. A
  fresh course starts at its trained launch state; examples within it do not.
- The ball advances longitudinally one fixed step per force. Its four lateral
  state equations and label-independent, smooth contact impulse are in the
  script. These are toy dynamics, not a calibrated rigid-body simulation.
- There is no learned input map, feature weighting, per-row parameter, or
  learned readout. The numeric encoding and answer-pad layout stay fixed.

## Training and selection

- Stratified 106/36/36 train/validation/test split, split seeds 20261104 and
  20261105. Publish exact row indices and training-only scaling statistics.
- Five optimizer seeds: 701 through 705. The same split is used by every seed.
  Seeds vary the initial four launch settings; the six motion settings have
  the same initial values in all runs. These are not broad physics restarts.
- `launch_only`: optimize four launch settings, freeze the six motion settings.
- `shared_motion`: optimize four launch settings plus six global parameters
  (force/torque/spin gains, velocity retention, spring strength, lateral bias).
  These six parameters are constant during each complete run.
- 250 Adam updates, learning rate 0.025, global gradient norm limit 2.0.
  Each update uses two shuffled complete training courses, with no resets or
  truncated gradients. Shuffle seeds 10000 through 10249 are shared across modes.
- After each complete forward run, reverse-mode backpropagation minimizes mean
  squared distance to the correct pad center across all obstacles.
- Evaluate validation every 10 updates, including the initial model. Select
  the highest correct-hit rate over four fixed shuffled validation courses
  (order seed 1801), then the lowest MSE to break ties.
- Test rows and labels never select parameters, epochs, or hyperparameters.

## Evaluation and controls

- Evaluate each frozen model on 20 complete test courses: the same 36 unseen
  rows in different orders (order seed 2901). This measures history sensitivity;
  20 orders are NOT 20 independent datasets or 720 independent examples.
- Report correct-hit rate, nearest-class accuracy, any-pad contact, full-course
  wins, MSE, and fraction of rows whose predicted class changes with course order.
- Report all five seeds, their mean and range; no best-test-seed selection.
- Compare against untrained identical dynamics, a training-majority classifier,
  and standard logistic regression (C=1, max_iter=2000) on the same scaling/split.
  Logistic classification accuracy is contextual, not a physical-contact metric.
- On a validation-selected illustrative model, score shuffled test labels and
  zero input forces as diagnostic controls. No control result changes training.
- Publish the full trajectory, obstacle-level results, and a figure of the
  first 12 obstacles of the first test order. Select the illustrative model by
  validation hit rate over 20 orders, then validation MSE.

## Interpretation

The original objective is a correct hit at **every** obstacle of an unseen
continuous course. Report the number of fully won courses even if it is zero.
Partial improvements are not a successful all-obstacle solution. Exceeding the
untrained model indicates improvement under this encoding; it does not prove
that all datasets can be represented or solved by a few trajectory parameters.

The Wine dataset has been used by older experiments, so this split is held out
from this run, not a new pristine external dataset. Results are a first test of
this particular encoding and simulator. No test-driven redesign in this run.
