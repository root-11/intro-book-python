# Solutions: 55 - The same numbers, a different total

Error figures are IEEE-754 and portable; timings are the Ryzen 9 270 / CPython 3.14.5 / numpy 2.4.4 figures from [`code/fpfragility/fpfragility.py`](https://github.com/root-11/intro-book-python/blob/main/code/fpfragility/fpfragility.py).

## Exercise 1 - Two orders, two answers

`(1e16 + -1e16) + 1 = 1`; `(1e16 + 1) + -1e16 = 0`. The losing step is `1e16 + 1`: near ten quadrillion the gap between representable `float`s is larger than 1, so the `1` rounds away and is gone before the `-1e16` can cancel the giant back down. A triple of your own: `1e8 + 0.001 + (-1e8)` gives `0.0` left to right but `0.001` if the giants cancel first. The addition that loses information is always the one that adds a small number to a much larger running total, where the small one falls below the large one's resolution.

## Exercise 2 - Lose a column

Two million values in `[0, 1)` (true total ~1,000,266) with a `+1e16`/`-1e16` pair around them:

| method | result | error |
|---|---|---|
| hand loop `acc += x` | 0.0 | ~1e6 (lost it) |
| hand loop, reversed | 0.0 | ~1e6 (wrong again) |
| builtin `sum()` | 1,000,265.923623 | ~4e-8 |
| `numpy.sum` | 1,000,264.0 | ~1.9 |
| `math.fsum` | 1,000,265.923623 | 0 |

The hand-written accumulator loses everything: the running total parks at 1e16 and every small entry falls below its resolution. Python's builtin `sum()` does not, because since CPython 3.12 it sums floats with Neumaier compensation - it carries the lost low bits and folds them back. `numpy.sum` adds in a pairwise tree, so small entries meet each other before the giant and almost all of the answer survives - but "almost": it is still off by about two, because pairwise is better-conditioned, not exact. `math.fsum` is correctly rounded and exact. The lesson: the *only* way to lose the whole answer in Python is to write the naive loop yourself; reach for `numpy.sum` for speed, `math.fsum` when you need the last bit.

## Exercise 3 - Pairwise is fast and accurate

Summing five million values: `numpy.sum` ~0.86 ms, a per-element Python loop ~22 ms - about **26x**. The Python loop pays the interpreter once per element and chains each add onto the previous (a dependency chain); `numpy.sum` runs in C over a contiguous array and adds independent subtrees, which both vectorises and breaks the chain. It is the accurate-and-fast option at once. It is still not a license to stop thinking: the pairwise tree's shape depends on the array length and the SIMD width, so two differently-shaped arrays of the same numbers can sum to different low bits - which is why [§48](48_reductions_dont_parallelize_freely.md) insists on a fixed reduction shape when a world hash depends on the total.

## Exercise 4 - Watch it drift

Start `running = math.fsum(col)`, then for two million edits do `running += new - col[i]; col[i] = new`. A fresh `math.fsum` afterwards differs in the last digits - an absolute gap around 3e-9 that never closes, because each delta step rounds and the rounding errors accumulate without bound in count. The absolute gap stays small, but divide it by the answer and it is worst when the true total nearly cancels (a near-zero denominator). This is why aggregates are periodically re-anchored with a fresh recompute rather than trusted forever: a maintained total trades correctness for speed, a sliver at a time, and the only way back is to recompute.

## Exercise 5 - The wrong side of the line

```python
def orient_exact(a, b, c):                 # Python int: arbitrary precision, no overflow
    return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])

def orient_float(a, b, c):                 # the columnar f64 way
    return (float(b[0])-a[0])*(float(c[1])-a[1]) - (float(b[1])-a[1])*(float(c[0])-a[0])
```

With `a=(0,0)`, `b=(p, p+r)`, `c=(p+1, p+r+1)` for `p~2^30` and `r in {-2,-1,1,2}`, the exact determinant is `-r` (tiny), but the products are ~2^60, past `float`'s 53-bit mantissa, so each rounds and their difference is noise. Over a hundred thousand such triples the `float` sign disagrees with the exact `int` sign **98.6%** of the time. The exact version is right every time and needs no special type: Python's `int` is already arbitrary-precision, so where the Rust edition reaches for `i128`, Python just multiplies. Same cost, always correct.

## Exercise 6 - A layout cannot save you (stretch)

Store the coordinates in perfect numpy columns and compute the `float` orientation vectorised over all triples at once: the answer is exactly as wrong (98.6%), just computed faster. The arc's usual move - lay it out flat and stream it - changes the *speed* of the computation and nothing about its *correctness*, because the error lives in the `float` arithmetic, not the memory layout. What fixes it is arithmetic: a defined summation order, a compensated sum, a wider accumulator, or - here - the exact `int` predicate. Correctness is orthogonal to layout; this whole chapter is the counterweight to the rest of the book.
