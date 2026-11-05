# Harder dataset test: 8x8 handwritten digits

Protocol fixed before training or reading test performance, November 4, 2026
(America/Chicago). This is the next test after the negative Wine result.

## Dataset and mapping

- Use all **1,797 scikit-learn `load_digits` images**, 8x8 pixels and 10 digit
  classes. This is the bundled UCI-derived digits dataset, **not MNIST**.
- Stratified 60/20/20 row split: 1,077 training, 360 validation, 360 test images.
  Split seeds: 20261105 and 20261106. Publish exact indices and scaling values.
- Each image is 64 input-force steps in row-major pixel order, followed by one
  obstacle. Use the same training-only standardization and `0.25*tanh(z/2.5)`
  force transform as Wine. Constant training pixels use scale 1.
- Every obstacle has 10 pads, equally spaced from -1 to +1. To keep pads
  distinct, their hit radius is 0.32 times center spacing: **0.071111...**.
  Contact width is 0.30 times spacing: **0.066666...**. This general rule gives
  exactly the original Wine geometry for three classes. State bounds stay +/-2.
- The class label only scores the completed run and supplies the loss. The
  forward encoder/dynamics never receive it. All 10 pads always exist.

## What stays fixed

The same four motion-state equations, fixed constants, nonlinearities, contact
impulse rule, trainable parameter transforms, initialization scheme and loss.
Only the dataset, input count, class count and corresponding pad layout change.
No new controller, learned input transform, feature selection, per-image weights,
reset between images, gradient truncation, or test-time adaptation is introduced.

Train both modes: four launch settings alone, and those settings plus six global
motion parameters. Five seeds (701-705) vary launch initialization; motion
parameters start identically across seeds. Every complete run keeps all its
parameters fixed and carries position/velocity/angle/spin between obstacles.

Use the same **250 Adam updates**, learning rate 0.025, norm clip 2, and two
shuffled complete training courses per update. Backpropagate after the whole
course. Check validation every 10 updates including epoch 0; choose highest
correct-hit rate over four fixed orders, then lowest squared-distance loss.
The number of updates is the same as Wine; total simulation work is much larger.

## Locked evaluation

- Freeze all ten selected models before test scoring. Evaluate 20 fixed orders
  of the same 360 test images per model, using order seed 2901. The 20 orders are
  history-sensitivity probes, not independent datasets.
- Report means/ranges across all seeds, correct-pad hits, nearest-class accuracy,
  full-course wins, order sensitivity, selected checkpoints and model parameters.
- Compare identical untrained balls, training-majority classification, and
  logistic regression (C=1, max_iter=2000) on the same training split/scaling.
  A classifier's accuracy and the stricter pad-hit rate are different metrics.
- For a validation-selected illustrative model, report shuffled-label and
  zero-force controls, predicted-class counts and a confusion matrix across the
  20 test orders. Also save full motion and the first-order obstacle hit record.
- Preserve the Wine outputs and verify that parameterizing input/class counts
  reproduces its saved trajectory. Check ten-class gradients and state carry.
- No test-driven changes or retries to improve the result in this run.

## Interpretation

Success still means a correct hit at **every** obstacle of an unseen continuous
course. Partial accuracy is not a full-course solution. A 10-class majority
baseline is around 10%; report its actual measured value. More classes, input
dimensions, examples and longer continuous courses make this a larger stress
test, but do not guarantee it is harder for every conventional classifier.

Because pad spacing shrinks for 10 classes, report class accuracy alongside
strict hits; Wine and digits hit percentages do not have identical tolerances.
The split is by image row, not by writer. This remains a small dataset and one
split, with neither large-scale performance nor broad generalization established.
