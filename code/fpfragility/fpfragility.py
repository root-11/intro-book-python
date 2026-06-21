# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy"]
# ///
"""
§55 exhibit - the same numbers, a different total, in Python+numpy.

Counterpart to the Rust edition's code/fpfragility crate. A pivot is a pile of
additions, and floating-point addition is neither associative nor exact. The
totals are order-dependent and, at scale, wrong - and no layout fixes it. This
is the arc's honest counterweight: correctness is orthogonal to layout.

Three measurements, with the Python-specific shading:

  C1 summation        the same column summed different ways gives different
                      totals. A naive left-to-right sum (a Python loop, or the
                      builtin sum()) can lose the whole answer. Python hands you
                      the fix for free in two places: numpy.sum already uses
                      PAIRWISE summation (the "add in pairs" method), and
                      math.fsum is exact. numpy.sum is both more accurate than a
                      Python loop AND faster - but it is still not exact, and its
                      tree shape still depends on length, so it is not a license
                      to stop thinking (see §48).

  C2 incremental drift  maintaining a running total by deltas (the cheap
                      aggregate §54 wanted) never equals a fresh recompute; the
                      relative error explodes when the true total nearly cancels.

  C3 orientation      the f64 left-or-right-of-line predicate gives the wrong
                      sign on near-collinear points. Python's arbitrary-precision
                      int makes the EXACT predicate trivial - no wide-int juggling,
                      no overflow - so here Python is the easy place to be correct.

Run:  uv run code/fpfragility/fpfragility.py
"""

import math
import random
import time

import numpy as np


def median(fn, k=3):
    return sorted(fn() for _ in range(k))[k // 2]


def time_call(fn):
    t0 = time.perf_counter()
    fn()
    return (time.perf_counter() - t0) * 1e3  # ms


def kahan(xs):
    """Neumaier compensated sum: carry the lost low-order bits and fold them back."""
    s = 0.0
    c = 0.0
    for x in xs:
        t = s + x
        if abs(s) >= abs(x):
            c += (s - t) + x
        else:
            c += (x - t) + s
        s = t
    return s + c


def c1_summation():
    print("== C1: an ill-conditioned column, summed several ways ==")
    rng = random.Random(0xF9)
    n = 2_000_000
    smalls = [rng.random() for _ in range(n)]          # ~n/2 in true total
    true_total = math.fsum(smalls)                     # ground truth (exact, correctly rounded)
    # the giants: a large credit and the debit that cancels it, around the column
    col = [1e16] + smalls + [-1e16]
    arr = np.array(col, dtype=np.float64)

    naive = 0.0
    for x in col:                                      # left-to-right, what a pivot does
        naive += x
    builtin = sum(col)
    np_sum = float(arr.sum())                          # numpy: pairwise summation
    fs = math.fsum(col)                                # exact
    kah = kahan(col)

    print(f"  true total (sum of the small entries) : {true_total:.6f}")
    print(f"  naive hand-loop  acc += x              : {naive:.6f}   err {abs(naive - true_total):.2e}  (lost it)")
    print(f"  builtin sum()  (Neumaier since 3.12)   : {builtin:.6f}   err {abs(builtin - true_total):.2e}")
    print(f"  numpy .sum()  (pairwise)               : {np_sum:.6f}   err {abs(np_sum - true_total):.2e}")
    print(f"  math.fsum     (exact)                  : {fs:.6f}   err {abs(fs - true_total):.2e}")
    print(f"  Neumaier compensated                   : {kah:.6f}   err {abs(kah - true_total):.2e}")
    # reverse the column: a different wrong naive answer
    rev = 0.0
    for x in reversed(col):
        rev += x
    print(f"  naive, column reversed                 : {rev:.6f}   (different wrong answer)")

    # timing: numpy pairwise vs a Python-loop naive sum - accuracy AND speed
    big = np.random.default_rng(1).random(5_000_000)
    big_list = big.tolist()
    t_np = median(lambda: time_call(lambda: big.sum()))
    t_py = median(lambda: time_call(lambda: sum(big_list)), k=1)
    print(f"\n  sum of 5e6 values: numpy .sum() {t_np:.2f} ms  vs  python sum() {t_py:.1f} ms"
          f"  ({t_py / t_np:.0f}x)")
    print("  the accurate method (pairwise) is also the fast one - independent adds, not a chain.")


def c2_incremental_drift():
    print("\n== C2: a maintained running total drifts from a fresh recompute ==")
    rng = random.Random(0xD41F7)
    n = 100_000
    col = [rng.random() for _ in range(n)]
    running = math.fsum(col)         # start from the exact total
    edits = 2_000_000
    for _ in range(edits):
        i = rng.randrange(n)
        new = rng.random()
        running += new - col[i]      # the cheap incremental patch
        col[i] = new
    fresh = math.fsum(col)
    abs_gap = abs(running - fresh)
    print(f"  after {edits:,} delta-edits:")
    print(f"  maintained running total : {running:.10f}")
    print(f"  fresh recompute (fsum)   : {fresh:.10f}")
    print(f"  absolute gap             : {abs_gap:.2e}")
    # relative error explodes near a cancelling total
    near_zero = [1e9, -1e9] + [rng.random() * 1e-6 for _ in range(1000)]
    run2 = math.fsum(near_zero)
    for _ in range(100_000):
        i = rng.randrange(len(near_zero))
        new = (rng.random() * 1e-6) if i >= 2 else near_zero[i]
        run2 += new - near_zero[i]
        near_zero[i] = new
    fresh2 = math.fsum(near_zero)
    print(f"  near-cancelling total: maintained {run2:.3e} vs fresh {fresh2:.3e}"
          f"  -> relative gap {abs(run2 - fresh2) / abs(fresh2):.2e}")
    print("  the gap never closes; as a fraction it is worst when the true total cancels.")


def c3_orientation():
    print("\n== C3: left-or-right-of-line for near-collinear points ==")
    rng = random.Random(0x0817)
    trials = 100_000
    disagree = 0
    for _ in range(trials):
        # a = origin; b = (p, q) with q = p + r (r tiny); c = (p+1, q+1).
        # exact orientation det = p - q = -r (tiny: +-1 or +-2), but the products
        # are ~2^60, far past f64's 53-bit mantissa, so the rounded f64 difference
        # is noise and the sign is unreliable.
        p = (rng.getrandbits(30)) + (1 << 29)
        r = rng.choice((-2, -1, 1, 2))
        q = p + r
        ax, ay, bx, by, cx, cy = 0, 0, p, q, p + 1, q + 1
        # exact, in Python's arbitrary-precision int - no overflow, ever
        exact = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
        # f64, the way a columnar SoA pipeline would compute it
        f = (float(bx) - ax) * (float(cy) - ay) - (float(by) - ay) * (float(cx) - ax)
        sign_exact = (exact > 0) - (exact < 0)
        sign_f = (f > 0) - (f < 0)
        if sign_exact != sign_f:
            disagree += 1
    print(f"  {trials:,} near-collinear triples, ~2^30 integer coordinates:")
    print(f"  f64 sign disagrees with the exact int sign: {disagree:,} ({100 * disagree / trials:.1f}%)")
    print("  the exact predicate is trivial in Python (bignum int); no layout fixes the f64 one.")


def main():
    c1_summation()
    c2_incremental_drift()
    c3_orientation()


if __name__ == "__main__":
    main()
