# 48 - Reductions don't parallelize freely - reference notes

## Exercises 1-2: make it diverge, then compound it

Fold a million `float`s in `w` contiguous chunks (sum each chunk, then the partials) at `w` = 1, 2, 4, 8 and hash each result. The hashes differ, and each is *stable* on repeated runs at a fixed `w` - the bug is deterministic per worker count, which is exactly why it passes your single-config tests. `reduction_divergence.py` is the specimen: the low bits move with `w` (harmonic values make the rounding visible). Feed that reduction into a per-tick global (an energy normalisation), run 1,000 ticks at two worker counts from one seed, and the last-bit difference compounds through the feedback loop into a different population - the [§16](16_determinism_by_order.md) world-hash now depends on `nproc`.

## Exercises 3-4: the two fixes

Fix the *order*: reduce into a **fixed** number of partitions - say 64, not one per worker - and fold the partials serially in id order. Re-run exercise 1 across worker counts and the hashes are identical, because the grouping is now the fixed 64, independent of how many workers ran. The trap to demonstrate: make the partition count *equal* the worker count and it diverges again - the partition count must be fixed, not the worker count. (The fixed-order value differs from the naive index-order sum in the low bits; that is reproducible, not more accurate.)

Or accumulate in **integers**: scale each value to a fixed-point `int`, sum exactly (integer addition is associative, so any grouping agrees), scale back. Identical across worker counts and any order. Python's `int` does not overflow, so the only judgement left is the scale - precision - not the range, which removes the one footgun the compiled languages carry here. Integer is deterministic by construction; fixed-order float is deterministic by discipline.

## Exercises 5-6: replay across worker counts; numpy.sum is a reduction too

Replaying one committed log at two worker counts is bit-identical only with a fixed-order or integer reduction - that is the [§37](37_log_is_world.md) distribution claim made true on heterogeneous hardware. And the bug is not only multiprocessing: compare `arr.sum()`, a Python `for`-loop fold, and `math.fsum` on the same array. They differ in the low bits. `numpy.sum` uses pairwise summation (a tree whose shape depends on size and SIMD width); the Python fold is strict left-to-right; `math.fsum` is exact. For a world hash, pin one grouping and never mix - `arr.sum()` is fine *if* the array shape is fixed, because then its tree is fixed.

## Exercise 7: the canary test

A CI check that runs the simulator at two worker counts and asserts equal world hashes fails the moment a racy reduction exists, and only then. It is the regression guard that catches the next one before a server with a different core count does - the canary is precise: hashes stable at fixed worker count, unstable when you change it. This is the cost-side analogue of [§43](43_tests_are_systems.md)'s `pytest -n 8` determinism check, aimed at the reduction rather than the iteration order.
