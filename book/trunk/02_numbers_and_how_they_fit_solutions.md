# Solutions: 2 — Numbers and how they fit

## Exercise 1 — Per-value cost

```python
import sys, numpy as np
print(sys.getsizeof(0))         # 28
print(sys.getsizeof(2**31))     # 32
print(sys.getsizeof(2**127))    # 44
print(sys.getsizeof(0.0))       # 24
print(sys.getsizeof(True))      # 28  — bool is a subclass of int
print(np.array([0, 2**31, 0], dtype=np.int64).nbytes)  # 24
```

A single `bool` costs 28 bytes — same as a small `int`. Three int64s in a numpy array cost 24 bytes total: no per-element header, no per-element pointer. That ratio (28 bytes per Python int *each*, vs 8 bytes per int64 in a column) is the size budget the rest of the book leans on.

## Exercise 2 — Cache-line packing

A cache line is 64 bytes:

| dtype     | bytes | per 64-byte line |
|-----------|------:|-----------------:|
| `int8`    |   1   |        64        |
| `int16`   |   2   |        32        |
| `int32`   |   4   |        16        |
| `int64`   |   8   |         8        |
| `float32` |   4   |        16        |
| `float64` |   8   |         8        |

A `np.array(..., dtype=np.int32)` of 16 elements is exactly one cache line. A `np.array(..., dtype=np.float64)` of 8 elements is exactly one. A Python `list` of *anything* is one pointer (8 bytes) per element plus the elements as separate objects elsewhere — at most 8 list pointers per line, with the actual values at unpredictable addresses.

## Exercise 3 — Width and speed

```python
import numpy as np, time
n = 100_000_000
a8  = np.ones(n, dtype=np.int8)
a64 = np.ones(n, dtype=np.int64)

t0 = time.perf_counter(); int(a8.sum());  t1 = time.perf_counter()
t2 = time.perf_counter(); int(a64.sum()); t3 = time.perf_counter()
print(f"int8  sum: {(t1-t0)*1000:.1f} ms")
print(f"int64 sum: {(t3-t2)*1000:.1f} ms")
```

```
int8  sum: 20.8 ms
int64 sum: 14.8 ms
```

The result is *counterintuitive*: int64 is faster than int8 despite reading 8× more bytes. The reason is how numpy reductions work. `arr.sum()` does not accumulate in the array's dtype; it widens to a 64-bit accumulator by default to avoid silent overflow. That widening means each int8 is read as one byte then promoted to eight bytes inside the loop, so the int8 case pays bandwidth-savings *plus* per-element widening — and on this machine the widening cost dominates.

To force overflow (the chapter's "hint about why the book picks widths with the maximum value in mind"), pin the accumulator:

```python
a8.sum(dtype=np.int8)  # now wraps; result is some int8 value, not 100_000_000
```

The book's discipline therefore has two parts. *Pick the narrowest dtype that holds your values* (storage) and *be explicit about the accumulator* (arithmetic). The two are different choices.

## Exercise 4 — Float weirdness

In pure Python, three of the five prompts *raise* — Python's defaults protect you from the IEEE 754 edges:

```python
>>> 0.0 / 0.0
ZeroDivisionError: float division by zero
>>> 1.0 / 0.0
ZeroDivisionError: float division by zero
>>> math.sqrt(-1.0)
ValueError: math domain error
>>> (-1.0) ** 0.5
(6.123233995736766e-17+1j)        # promoted to complex, not nan
>>> nan = float("nan"); nan != nan
True
```

The IEEE 754 behaviour the chapter prose describes — `nan` and `inf` from division — surfaces through numpy:

```python
>>> import numpy as np, warnings
>>> with warnings.catch_warnings():
...     warnings.simplefilter("ignore")
...     print(np.float64(0.0) / np.float64(0.0))   # nan
...     print(np.float64(1.0) / np.float64(0.0))   # inf
...     print(np.sqrt(np.float64(-1.0)))           # nan
nan
inf
nan
```

`nan != nan` works in pure Python because `float("nan")` constructs the IEEE bit pattern directly; the *generation* of nan from arithmetic is what numpy provides and pure Python guards against. Both views matter: when you leave the interpreter for numpy columns you trade exception protection for IEEE behaviour, and you need to know the rules of the side you're on.

## Exercise 5 — `==` is the wrong tool

```python
>>> 0.1 + 0.2 == 0.3
False
>>> 0.1 + 0.2
0.30000000000000004
>>> import math
>>> math.isclose(0.1 + 0.2, 0.3)
True
```

`0.1` and `0.2` are not exactly representable in binary; their sum lands one ulp past `0.3`. `math.isclose` exists because the standard library acknowledges `==` is the wrong tool for floats. The default `rel_tol=1e-9` is a *choice* — make it deliberate when the problem demands a tighter or looser tolerance. The pattern you'll learn to reach for:

```python
math.isclose(a, b, rel_tol=1e-9, abs_tol=0.0)   # near-zero values need abs_tol too
```

## Exercise 6 — Catastrophic cancellation

```python
import numpy as np
a32 = np.float32(1e10); b32 = a32 - np.float32(1.0)
print(a32 - b32)                          # 0.0  — should be 1.0

a64 = np.float64(1e10); b64 = a64 - np.float64(1.0)
print(a64 - b64)                          # 1.0
```

`float32` has ~7 decimal digits of precision; `1e10` already exhausts them, so `1e10 - 1.0` cannot be distinguished from `1e10` and the subtraction returns `0.0`. `float64` has ~15 digits, room to spare for this size. The lesson is *not* "use float64" — it is that the right precision depends on the dynamic range of the values you'll subtract. A simulation that subtracts two large nearly-equal positions to compute a small velocity *needs* the wider type even if the final answer fits in a narrower one.

## Exercise 7 — Run the summation exhibit

```sh
uv run code/measurement/sums.py
```

Source: [`code/measurement/sums.py`](https://github.com/root-11/intro-book-python/blob/main/code/measurement/sums.py). Five datasets × three orders × five algorithms. The dataset where the spread is largest is `large_plus_small` (a few values of size 10⁶ added to many values of size 1):

```
=== DATASET: large_plus_small (N=2000002) ===
-- Order: original --
Reference: 2000000
builtin_sum        | time_s: 0.0085 | result: 2000000     | abs_err: 0
math_fsum          | time_s: 0.0077 | result: 2000000     | abs_err: 0
kahan_sum          | time_s: 0.0918 | result: 2000000     | abs_err: 0
neumaier_sum       | time_s: 0.1427 | result: 2000000     | abs_err: 0
pairwise_sum       | time_s: 0.3901 | result: 1999998     | abs_err: 2
```

`pairwise_sum` — usually the recommended general-purpose stable summation — is *off by 2 absolute* on this dataset. Two of the three large values get absorbed during a partial-sum step where they are paired with a million-and-something accumulated 1s. `builtin sum`, `math.fsum`, `kahan_sum`, and `neumaier_sum` all return the exact integer answer. The lesson: stability across reorderings is not the same as stability across magnitude mixtures. `math.fsum` is the safest single-pass default when you cannot bound the data; pairwise wins only when magnitudes are uniform.

## Exercise 8 — Choose a width

| column                                  | dtype      | reasoning                                                                                                  |
|-----------------------------------------|-----------:|------------------------------------------------------------------------------------------------------------|
| age in ticks at 30 Hz × 1 yr            | `uint32`   | 30 × 60 × 60 × 24 × 365 ≈ 9.5×10⁸; `uint32` holds 4.3×10⁹                                                  |
| card suit                               | `uint8`    | 4 values; 252 spare slots                                                                                  |
| 4K pixel count                          | `uint32`   | 8.3 million pixels per frame                                                                               |
| user id, 100M users                     | `uint32`   | 4×10⁹ headroom; `uint64` only if you anticipate gen-2 ids or sparse handles                                |
| 16-bit PCM sample                       | `int16`    | the format defines it; signed because PCM is signed                                                        |

The discipline is to *write down why*. Two years later, when someone changes the budget (10M users → 1B users), the column type's reasoning is the diff that matters.

## Exercise 9 — The `eps` of a float

```python
import numpy as np
eps = np.finfo(np.float32).eps            # 1.1920928955078125e-07
print(np.float32(1.0) + np.float32(0.5) * eps)   # 1.0
print(np.float32(1.0) + eps)                     # 1.0000001
```

Half an `eps` added to `1.0` is *absorbed* — the result is still exactly `1.0`. One full `eps` added to `1.0` produces the next representable float above `1.0`. This is the *unit in the last place* rule: floats near `1.0` have a spacing of `eps`; smaller additions cannot be represented and are silently dropped.

The implication for summation: adding 10⁹ values each of size `0.5 * eps` to a running total of `1.0` produces a final total of `1.0`, not `1.0 + 5×10²`. Every step rounds away the contribution. This is the failure mode `kahan_sum` and `neumaier_sum` correct: they keep a *compensation term* that accumulates the dropped bits across iterations and folds them back in. The book uses `math.fsum` (which keeps full precision via a list of exact partials) when input magnitudes are unbounded.
