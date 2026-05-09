# 2 — Numbers and how they fit

> *Concept node: see the [DAG](../../concepts/dag.md) and [glossary entry 2](../../concepts/glossary.md#2--numbers-and-how-they-fit).*

A cache line is 64 bytes. That is the unit of memory the CPU loads at a time. Everything you do with data is, in part, a question of how many things fit in 64 bytes.

## What an `int` actually costs

You wrote `x = 1` last week and that was the end of the question. What sat in memory was a `PyLong` object: a header, a refcount, a length, and one or more 32-bit "digit" limbs holding the value. The minimum size, even for `0`, is **28 bytes**. As the value grows past one digit, the object grows by four bytes per additional digit. From [`code/measurement/number_footprint.py`](../../code/measurement/number_footprint.py) on this machine:

```
int 0                          28 bytes
int 1                          28 bytes
int 256 (last interned)        28 bytes
int 257                        28 bytes
int 1_000                      28 bytes
int 2**31                      32 bytes
int 2**63                      36 bytes
int 2**127                     44 bytes
float 0.0                      24 bytes
float 3.14                     24 bytes
float 1e300                    24 bytes
```

A `PyFloat` is 24 bytes, fixed. A `PyLong` is at least 28 bytes and grows with magnitude. A `bool` is also a `PyLong`. A `complex` is 32 bytes. The header alone is bigger than the value in every case.

This is the first part of the chapter's question. *Picking the narrowest type that holds your range* — the discipline that defines whether a cache line packs 8 things or 64 things — does not exist in pure Python. There is no `uint8`. There is no `int32`. Every Python `int` is the same costly object regardless of whether it holds the value `0` or `2**63`. You cannot trade range for cache lines, because you cannot pick the range.

> [!NOTE]
> CPython caches small integers in `[-5, 256]` as singletons (the *small-int cache*). A list of zeros does not allocate a million `PyLong(0)` objects — it allocates a million pointers, all to the same one. Once the values escape that range, every value is a fresh allocation. Confirm this with `id(0) == id(0)` (true) versus `id(257) == id(257)` (sometimes true, sometimes not, depending on the parser's caching of literal constants in the same compilation unit — but never reliable). Treat the small-int cache as a CPython implementation detail you cannot lean on.

## What numpy gives you back

`numpy` makes the width budget exist again. `np.int8` is one byte, range -128 to 127. `np.int16` is two bytes, `np.int32` is four, `np.int64` is eight. `np.float32` is four bytes (~7 decimal digits of precision); `np.float64` is eight (~15 digits). The signed/unsigned and integer/float variants compose freely.

A `np.zeros(N, dtype=np.uint8)` is N bytes — flat, contiguous, no per-element header. A cache line packs **64** of them. A `np.zeros(N, dtype=np.int64)` is 8N bytes; one cache line packs **8**. If your loop touches one element per cache line, the int64 version makes 8× as many memory loads as the uint8 version. The width budget is back.

Same exhibit, the data column tells the story at N=1,000,000:

| layout                             | data size | sum (ms) |
|------------------------------------|----------:|---------:|
| Python list of large ints          |  38.25 MB |   2.56   |
| Python list of floats              |  38.38 MB |   4.27   |
| numpy int8                         |   0.95 MB |   0.28   |
| numpy int16                        |   1.91 MB |   0.34   |
| numpy int32                        |   3.81 MB |   0.45   |
| numpy int64                        |   7.63 MB |   0.42   |
| numpy float32                      |   3.81 MB |   0.22   |
| numpy float64                      |   7.63 MB |   0.36   |

The Python-list-vs-numpy ratio at this scale: **40× more bytes** in the list compared to numpy int8, **20×** vs int16, **10×** vs int32, **5×** vs int64. Choosing the narrowest numpy width that holds your range gives you up to 8× *additional* shrink on top of the list-to-numpy step. Sum times collapse from milliseconds to fractions of a millisecond — two orders of magnitude.

Pick the narrowest type that holds your range, and write down why. A 52-card deck's `suits` need 4 values, `ranks` need 13, `locations` need maybe 8 — all fit in `np.uint8`. A creature's `pos` needs about ten kilometres of grid resolved to centimetre precision; that fits in `np.float32`. A timestamp in microseconds for a year-long simulation needs something like 3×10¹³, which does not fit in `np.uint32` (4×10⁹) but fits comfortably in `np.uint64`. Choose, and write the choice down.

## Floats are not real numbers

They look like real numbers but are not. There are only about 4 billion `float32` values; there are only about 18 quintillion `float64` values; that is finite. Operations have edges: `1.0 / 0.0 = inf`, `0.0 / 0.0 = nan`, and `nan != nan` — yes, equality is broken on purpose for `nan`, because there is no reasonable answer. But `==` is also unreliable for *ordinary* floats: `0.1 + 0.2 == 0.3` is `False`, because `0.1` and `0.2` cannot be represented exactly in binary and the rounding error happens to land just past `0.3`. This is why `math.isclose(a, b, rel_tol=1e-9, abs_tol=0.0)` exists — it is the standard library's acknowledgement that `==` is the wrong tool for floats and that comparing them needs a tolerance you choose deliberately. Subtracting two nearly equal floats loses most of their precision (this is *catastrophic cancellation*). Adding a tiny float to a large one quietly drops the tiny one (this is *absorption*). None of this is a problem if you know it is there; all of it is a problem if you assume floats are mathematics.

[`code/measurement/sums.py`](../../code/measurement/sums.py) demonstrates the consequences across five pathological datasets — random balanced, large-plus-many-small, alternating signs, tiny increments, and arrays containing NaNs — using six summation algorithms (`sum`, `math.fsum`, Kahan, Neumaier, pairwise, decimal reference). Run it; read the discrepancies. The same input data summed in different orders gives different answers, and the "naive" answer is sometimes off by orders of magnitude. The fix is not "use float64 instead of float32" — it is *picking a summation algorithm aware of the data shape*. `math.fsum` and Neumaier are usually the right defaults for a single-pass sum where you cannot bound the input.

Most of this book uses `np.uint8`, `np.uint16`, `np.uint32`, `np.float32`, and `np.uint64` for time. `int*` and `float64` appear when the range or precision genuinely demands it. The choice is documented at every column declaration.

## Exercises

1. **Per-value cost.** Print `sys.getsizeof(0)`, `sys.getsizeof(2**31)`, `sys.getsizeof(2**127)`, `sys.getsizeof(0.0)`, `sys.getsizeof(True)`. Confirm that even a `bool` costs 28 bytes (`bool` is a subclass of `int`). Now print `np.array([0, 2**31, 0], dtype=np.int64).nbytes`. Three int64s = 24 bytes total, no headers, no per-value pointers.
2. **Cache-line packing.** For each numpy dtype — `int8`, `int16`, `int32`, `int64`, `float32`, `float64` — compute how many fit in a 64-byte cache line. A `np.array(_, dtype=np.int32)` of 16 elements is exactly one line; a `np.array(_, dtype=np.float64)` of 8 elements is exactly one line.
3. **Width and speed.** Sum a `np.ones(100_000_000, dtype=np.int8)`, then a `np.ones(100_000_000, dtype=np.int64)`. The ratio in time should be smaller than the ratio in bytes (8×) because compute is not the bottleneck — memory bandwidth is. Note also that the int8 sum overflows; this is a hint about why the book picks widths *with the maximum value in mind*.
4. **Float weirdness.** Compute `0.0 / 0.0`, `1.0 / 0.0`, `(-1.0) ** 0.5`, `math.sqrt(-1.0)`. Print them. Then `nan = float("nan"); assert nan != nan` — confirm it does not raise.
5. **`==` is the wrong tool.** Print `0.1 + 0.2 == 0.3`. Observe `False`. Print `0.1 + 0.2` to see the rounding error: `0.30000000000000004`. Now use `math.isclose(0.1 + 0.2, 0.3)` and observe `True`. Read [the `math.isclose` docs](https://docs.python.org/3/library/math.html#math.isclose) — note that the default `rel_tol=1e-9` is a *choice* you should be making explicitly when the problem demands a tighter or looser tolerance. The standard library has `isclose` because the language designers know `==` is unreliable here; lean on it.
6. **Catastrophic cancellation.** Compute `np.float32(1e10) - (np.float32(1e10) - np.float32(1.0))`. The result should be `1.0`; on `float32` it usually is not. Repeat with `np.float64` and observe it gets closer (but not always exactly `1.0`).
7. **Run the summation exhibit.** `uv run code/measurement/sums.py`. Read the discrepancies between the algorithms across the five datasets. Note the dataset where the spread is largest. That dataset is the one that decides which summation routine you should reach for in production.
8. **Choose a width.** For each of these columns, write down the dtype you would pick and why: a creature's age in ticks at 30 Hz over a year-long simulation; a card's suit; the pixel count of a 4K screen; the user id in a system with up to 100 million users; an audio sample value in 16-bit PCM.
9. *(stretch)* **The `eps` of a float.** `np.finfo(np.float32).eps` is the smallest `x` such that `1.0 + x != 1.0` in float32. Compute the value, then compute `np.float32(1.0) + np.float32(0.5) * np.finfo(np.float32).eps` — is the result `1.0` or `1.0 + eps/2`? What does this say about a sum of small numbers added one at a time to a large running total?

Reference notes in [02_numbers_and_how_they_fit_solutions.md](02_numbers_and_how_they_fit_solutions.md).

## What's next

[§3 — The `Vec` is a table](03_the_vec_is_a_table.md) takes the next step: now that you know how big the elements are, what does an `np.array` *do* with them, and what shape does the rest of the book expect them to be in?
