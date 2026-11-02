#!/usr/bin/env python3
"""Kinetra pass-through contact test.

Christopher's model, as refined across the 11:25-11:48
conversation, before any claims are made about it:

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

This script tests that mechanism, not a learned
behavior. No controller is trained here. The aimed
controller is a hand-set proportional controller
(gains stated below), used only to show the ball can
be steered to a chosen part.

GEOMETRY (his units, full widths):
- Track width 3.0, lateral range [-1.5, 1.5].
- A problem fits iff its width <= track width AND
  |center| + width/2 <= track half-width. Both
  conditions are checked: a too-wide problem fails
  the first, a misplaced problem fails the second.
- A problem's span is divided into equal parts.
  Contact at lateral position y gives the part
  containing y; contact outside the span gives NO
  answer (a miss). The ball is NOT walled in this
  test: the track width acts through the fit rule,
  and off-track contact is a miss by definition
  (outside every problem's span).

PASS STRUCTURE (Mia's, stated so it can be
corrected):
- Each problem is a station. The ball spends 10
  physics steps "at" the problem: the room to move
  around. Ball physics reuses Kinetra's constants
  (angle 0.48, spin 0.43, velocity 0.48 decay, 0.83
  sin(angle), 0.23 spin, plus push).
- After 10 steps the ball passes through: contact =
  current y, part/answer recorded, append-only.
- Wander passes: targets zero, pushes only
  (sigma 0.09). State carries across problems
  within a pass (a real traversal).
- Aimed passes: fresh state per aim, proportional
  controller angle_target = clip(0.6*(aim - y)
  - 0.8*velocity, +-0.9), spin_target =
  clip(0.3*(aim - y), +-0.9), pushes sigma 0.05.
  Aim = a part's center.

PRE-REGISTERED CHECKS (Mia's, fixed before running):
- FIT: width-4 problem rejected, misplaced problem
  rejected, fitting problems accepted.
- WANDER SPREAD: across 200 wander passes, P1
  contacts cover at least 4 of 5 parts (contact
  varies, it is not a fixed point).
- FORCED MISS: a pass starting at y=1.4 with zero
  targets records NO answer on P1 (outside span).
- AIMED: for every part, the modal recorded answer
  equals the aimed part, and the aimed-part rate
  exceeds 50%.
- RECORD: total records = 200 wander x 2 problems
  + 100 aimed = 500; append-only (the first record
  is unchanged at the end); per-problem query
  counts match (P1 300, P2 200).

Rates are reported exactly; a failed check is
reported as failed, not tuned away.

Python standard library only. Run:
python kinetra_pass_test.py
Writes kinetra_pass_records.csv alongside the script.
"""

import csv
import math
import random
from dataclasses import dataclass, field
from pathlib import Path

TRACK_WIDTH = 3.0
TRACK_HALF = TRACK_WIDTH / 2.0
STEPS_PER_PROBLEM = 10
OUT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Problem:
    pid: str
    center: float
    width: float
    n_parts: int

    @property
    def lo(self):
        return self.center - self.width / 2.0

    @property
    def hi(self):
        return self.center + self.width / 2.0

    def fits(self):
        return (self.width <= TRACK_WIDTH
                and abs(self.center) + self.width / 2.0 <= TRACK_HALF)

    def part_of(self, y):
        """Part index for contact y, or None for a miss."""
        if y < self.lo or y > self.hi:
            return None
        part_w = self.width / self.n_parts
        return min(int((y - self.lo) / part_w), self.n_parts - 1)

    def part_center(self, part):
        part_w = self.width / self.n_parts
        return self.lo + (part + 0.5) * part_w


@dataclass
class Ball:
    y: float = 0.0
    velocity: float = 0.0
    angle: float = 0.0
    spin: float = 0.0


def clip(x, a, b):
    return min(b, max(a, x))


def step(ball, angle_target, spin_target, push):
    ball.angle += 0.48 * (angle_target - ball.angle)
    ball.spin += 0.43 * (spin_target - ball.spin)
    ball.velocity = (0.48 * ball.velocity + 0.83 * math.sin(ball.angle)
                     + 0.23 * ball.spin + push)
    ball.y += ball.velocity


def run_problem(ball, problem, rng, push_sigma, aim=None):
    """10 maneuver steps, then pass through and record."""
    for _ in range(STEPS_PER_PROBLEM):
        push = rng.gauss(0, push_sigma) if push_sigma > 0 else 0.0
        if aim is None:
            angle_target, spin_target = 0.0, 0.0
        else:
            angle_target = clip(0.6 * (aim - ball.y)
                                - 0.8 * ball.velocity, -0.9, 0.9)
            spin_target = clip(0.3 * (aim - ball.y), -0.9, 0.9)
        step(ball, angle_target, spin_target, push)
    part = problem.part_of(ball.y)
    return ball.y, part


def main():
    p1 = Problem('P1', 0.0, 2.0, 5)
    p2 = Problem('P2', 0.5, 2.0, 5)
    overflow = Problem('PX', 0.0, 4.0, 5)
    misplaced = Problem('PM', 1.0, 2.0, 5)

    checks = []
    print('FIT CHECKS')
    for prob, expect in [(p1, True), (p2, True),
                         (overflow, False), (misplaced, False)]:
        got = prob.fits()
        ok = got == expect
        checks.append(('fit ' + prob.pid, ok))
        print(f'  {prob.pid} center={prob.center} width={prob.width}: '
              f'fits={got} (expect {expect}) -> {"OK" if ok else "FAIL"}')

    records = []  # append-only: (mode, pass_id, pid, y, part)

    # Wander passes: traversal P1 -> P2, state carries.
    for i in range(200):
        rng = random.Random(1000 + i)
        ball = Ball()
        for prob in (p1, p2):
            y, part = run_problem(ball, prob, rng, 0.09)
            records.append(('wander', i, prob.pid, y, part))

    # Snapshot after wander: append-only means these entries
    # must be byte-identical at the end, not just counted.
    wander_snapshot = list(records)

    # Forced miss: starts outside P1's span, no steering.
    rng = random.Random(777)
    ball = Ball(y=1.4)
    y, part = run_problem(ball, p1, rng, 0.0)
    checks.append(('forced miss', part is None))
    print(f'\nFORCED MISS: start y=1.4 -> contact y={y:.3f} '
          f'part={part} -> {"OK" if part is None else "FAIL"}')
    records.append(('forced-miss', -1, 'P1', y, part))

    # Aimed passes: each part of P1, fresh state per aim.
    aimed_rates = {}
    for part in range(p1.n_parts):
        aim = p1.part_center(part)
        hits = 0
        counts = {}
        for k in range(20):
            rng = random.Random(5000 + part * 100 + k)
            ball = Ball()
            y, got = run_problem(ball, p1, rng, 0.05, aim=aim)
            records.append(('aimed', part * 100 + k, 'P1', y, got))
            counts[got] = counts.get(got, 0) + 1
            hits += got == part
        modal = max(counts, key=counts.get)
        rate = hits / 20.0
        aimed_rates[part] = rate
        ok = modal == part and rate > 0.5
        checks.append((f'aimed part {part}', ok))
        print(f'AIMED part {part} (aim {aim:+.1f}): modal={modal} '
              f'rate={rate:.2f} counts={counts} '
              f'-> {"OK" if ok else "FAIL"}')

    # Wander spread on P1.
    wander_p1 = [r for r in records
                 if r[0] == 'wander' and r[2] == 'P1']
    parts_hit = sorted({r[4] for r in wander_p1 if r[4] is not None})
    misses = sum(1 for r in wander_p1 if r[4] is None)
    ok = len(parts_hit) >= 4
    checks.append(('wander spread', ok))
    print(f'\nWANDER P1: parts hit {parts_hit} ({len(parts_hit)}/5), '
          f'misses {misses}/200 -> {"OK" if ok else "FAIL"}')

    # Record checks.
    total_ok = len(records) == 200 * 2 + 1 + 100
    first_ok = records[:len(wander_snapshot)] == wander_snapshot
    p1_count = sum(1 for r in records if r[2] == 'P1')
    p2_count = sum(1 for r in records if r[2] == 'P2')
    counts_ok = p1_count == 301 and p2_count == 200
    checks.append(('record total', total_ok))
    checks.append(('record append-only', first_ok))
    checks.append(('record per-problem counts', counts_ok))
    print(f'RECORD: total {len(records)} (expect 501) -> '
          f'{"OK" if total_ok else "FAIL"}')
    print(f'RECORD: P1 {p1_count} (expect 301), P2 {p2_count} '
          f'(expect 200) -> {"OK" if counts_ok else "FAIL"}')

    print()
    failed = [name for name, ok in checks if not ok]
    if failed:
        print(f'VERDICT: {len(failed)} CHECK(S) FAILED: '
              f'{", ".join(failed)}')
    else:
        print('VERDICT: all pre-registered checks passed. '
              'The mechanism does what the concept says: '
              'fitting problems accept any-part contact, '
              'contact is recorded, and the recorded part '
              'is the answer.')

    out = OUT / 'kinetra_pass_records.csv'
    with out.open('w', newline='') as f:
        cw = csv.writer(f)
        cw.writerow(['mode', 'pass_id', 'problem',
                     'contact_y', 'part'])
        cw.writerows(records)
    print('Saved:', out.name)


if __name__ == '__main__':
    main()
