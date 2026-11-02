# Kinetra pass-through contact test

Tests Christopher's refined model BEFORE any claims
are made about it:

1. A track has a width. A problem has a width. The
   problem must FIT on the track, or it overflows.
   (His example: 3-inch track, 4-inch problem =
   overflow.)
2. The ball PASSES THROUGH the problem. It does not
   stop at it or bounce off it.
3. The CONTACT is RECORDED: where on the problem the
   ball passed through.
4. Every part touched is ANOTHER ANSWER: the recorded
   contact position selects the answer.

## Setup (his units, full widths)

- Track width 3.0, lateral range [-1.5, 1.5].
- A problem fits iff width <= track width AND
  |center| + width/2 <= track half-width.
- Problems: P1 (center 0, width 2, 5 parts), P2
  (center 0.5, width 2, 5 parts). Each part is 0.4
  wide; contact outside the span records NO answer.
- The ball spends 10 physics steps "at" each
  problem (the room to move around), using Kinetra's
  constants, then passes through and the contact is
  appended to the record.
- Wander passes: zero targets, pushes only
  (sigma 0.09), state carries P1 -> P2.
- Aimed passes: fresh state per aim, hand-set
  proportional controller (angle 0.6*(aim - y)
  - 0.8*velocity, spin 0.3*(aim - y), pushes
  sigma 0.05). The controller is NOT learned; it
  only shows the ball can be steered to a part.

## Results: all pre-registered checks passed

- FIT: P1 and P2 accepted; width-4 problem
  rejected; misplaced problem (center 1.0, width 2)
  rejected.
- FORCED MISS: a pass starting at y=1.4 records
  no answer on P1 (outside the span).
- AIMED (20 passes per part): part 0 at 0.95
  (19 hits, 1 miss), parts 1-4 at 1.00. The modal
  recorded answer equals the aimed part everywhere.
- WANDER (200 passes): P1 contacts cover all 5
  parts, with 9 misses. Contact genuinely varies;
  it is not a fixed point.
- RECORD: 501 records total, append-only verified
  against a mid-run snapshot, P1 301 / P2 200 on
  query.

## Honesty boundary

This tests the MECHANISM in the toy, not a learned
behavior and not an architecture claim. The aimed
controller is hand-set, not trained. What it
establishes: the pass-through + recorded-contact +
part-as-answer model works as specified, with fit
enforced, misses possible, and records persistent.
What it does not establish: anything about real
models, learning, or whether this scales beyond
the toy.

## Files

- `kinetra_pass_test.py` — the test (python
  kinetra_pass_test.py; writes
  kinetra_pass_records.csv). Standard library only.
- `kinetra_pass_records.csv` — all 501 records.
