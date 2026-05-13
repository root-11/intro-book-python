# Solutions: 3 — The `Vec` is a table

## Exercise 1 — Pointer-chase or value-read

```python
>>> import sys, numpy as np
>>> sys.getsizeof(0)            # 28
>>> sys.getsizeof(1000)         # 28
>>> sys.getsizeof(10**100)      # 72  — large ints grow per limb
>>> np.array([0, 1000, 10**18], dtype=np.int64).nbytes
24
```

Three int64s in a numpy array: 24 bytes, no per-element headers. Three `PyLong`s in a list: 28 + 28 + 72 = 128 bytes for the values *plus* 8 × 3 = 24 bytes of pointers in the list's backing array *plus* the list header. The numpy column is the values; everything else in the list version is bookkeeping.

## Exercise 2 — The interning trap

```python
>>> a = [0] * 1_000_000
>>> b = [1000 + i for i in range(1_000_000)]
>>> len(set(id(x) for x in a[:100]))         # 1   — all the same object
>>> len(set(id(x) for x in b[:100]))         # 100 — every value is its own PyLong
```

`[0] * 1_000_000` does not allocate a million `PyLong(0)`s; it allocates a million pointers, all to one shared `0` object. The list weighs 8 MB of pointers + one 28-byte int. The intuition "a list of small ints is cheap" is true *inside CPython's small-int cache* (`[-5, 256]`) and false everywhere else.

`id(257) == id(257)` and `id(1000) == id(1000)` may both return `True` *within a single statement* because the parser caches literal constants in a compilation unit. Across statements, identity is not guaranteed for values outside `[-5, 256]`. Don't lean on that — it's an implementation detail of how the bytecode compiler stores literals, not a runtime property of integers.

## Exercise 3 — Capacity vs length

```python
import sys
lst = []
prev = sys.getsizeof(lst)
sizes = []
for i in range(1001):
    lst.append(i)
    s = sys.getsizeof(lst)
    if s != prev:
        sizes.append((len(lst), s))
        prev = s
print(sizes[:8])
print(f"growth points up to N=1000: {len(sizes)}")
```

```
[(1, 88), (5, 120), (9, 184), (17, 248), (25, 312), (33, 376), (41, 472), (53, 568)]
growth points up to N=1000: 28
```

`list` over-allocates and re-allocates in chunks, like Rust's `Vec::push`. The growth pattern (currently `~1.125 ×` capacity) is a CPython implementation detail — different versions and `pypy`/`micropython` will pick different multipliers. The principle is identical to `Vec`: amortised O(1) push, occasional copy. The takeaway is the same as in any growable container: if you know the final size, pre-allocate (`np.zeros(N, ...)` for numpy; `[None] * N` then assign for lists) instead of pushing.

## Exercise 4 — Run the §3 exhibit

```sh
uv run code/measurement/aos_vs_soa_footprint.py
```

Source: [`code/measurement/aos_vs_soa_footprint.py`](https://github.com/root-11/intro-book-python/blob/main/code/measurement/aos_vs_soa_footprint.py). N=1,000,000 rows, K=10 ints per row, values past the small-int cache, each layout in a fresh subprocess:

```
layout                                          build (s)   RSS (MB)  sum c0 (s)
--------------------------------------------------------------------------------
1. list of tuples              (AoS)                0.744      437.0      0.0249
2. list of lists               (AoS)                0.615      498.2      0.0269
3. tuple of lists              (SoA stdlib)         0.463      382.9      0.0025
4. tuple of array.array        (SoA typed)          0.660       76.7      0.0116
5. tuple of numpy int64 arrays (SoA numpy)          0.092       93.8      0.0004

Ratios vs layout 5 (numpy SoA):
  1. list of tuples              (AoS)             4.7× memory     8.1× build    69.2× sum-c0
  2. list of lists               (AoS)             5.3× memory     6.7× build    74.7× sum-c0
  3. tuple of lists              (SoA stdlib)      4.1× memory     5.1× build     6.9× sum-c0
  4. tuple of array.array        (SoA typed)       0.8× memory     7.2× build    32.2× sum-c0
```

The five rows separate three independent wins:

- **AoS → SoA (1/2 → 3): the speed flip.** ~12% storage win, **10× speedup** on column-sum. Walking one contiguous list of pointers beats walking N tuples and dereferencing through each one to reach `row[0]`. No numpy required.
- **SoA-list → SoA-typed (3 → 4): the memory flip.** **5× storage win** (~383 MB → ~77 MB) from dropping the `PyLong` boxes. *But the sum slows down* (2.5 ms → 11.6 ms) because Python unboxes each int64 to a `PyLong` before adding it. Typed storage saves bytes; it does not save the inner loop.
- **SoA-typed → SoA-numpy (4 → 5): the C-vectorisation flip.** Same bytes, **30× speedup** on the same sum. The loop moves into C; the interpreter is stepped out.

The four-row form of this exhibit collapsed steps 2 and 3 into "use numpy." The five-row form shows they are separate. Numpy happens to bundle them; `array.array` lets you take the memory win without the C-loop win, which is sometimes the right trade for a project that wants stdlib-only deps.

## Exercise 5 — The dict trap

```python
import time, random, numpy as np

d   = {i: i*i for i in range(1_000_000)}
arr = np.arange(1_000_000) ** 2
idx = np.array([random.randrange(1_000_000) for _ in range(100_000)])

t0 = time.perf_counter()
for k in idx: d[int(k)]
t1 = time.perf_counter()
arr[idx]
t2 = time.perf_counter()

print(f"dict 100K lookups:  {(t1-t0)*1000:.1f} ms")
print(f"numpy 100K gather:  {(t2-t1)*1000:.2f} ms")
print(f"ratio: {(t1-t0)/(t2-t1):.0f}×")
```

```
dict 100K lookups:  34.6 ms
numpy 100K gather:   0.75 ms
ratio: 46×
```

Both look up "by integer." The dict pays a hash, a probe, and a `PyObject*` dereference per access — all in pure Python. The numpy gather is one indirection through a typed buffer in C. Same operation, 46× cost gap. When the keys are dense integers, dicts are not the right tool — the only thing they buy you is sparse indexing, and a dense column gets you indexing for free.

## Exercise 6 — swap-remove vs remove

```python
import time
lst1 = list(range(1_000_000))
t0 = time.perf_counter()
for _ in range(100):
    lst1.pop(500_000)
t1 = time.perf_counter()
print(f"100 pop(middle):    {(t1-t0)*1000:.2f} ms")

lst2 = list(range(1_000_000))
t0 = time.perf_counter()
for _ in range(100):
    i = 500_000
    lst2[i] = lst2[-1]; lst2.pop()
t1 = time.perf_counter()
print(f"100 swap_remove:    {(t1-t0)*1000:.3f} ms")
```

```
100 pop(middle):    3.9 ms
100 swap_remove:    0.019 ms
```

~200× difference. `lst.pop(i)` for `i` in the middle costs O(N) because every element after `i` shifts down one slot; 100 mid-list pops at N=1M is ~50M element moves. The swap-remove pattern is O(1): overwrite the gap with the last element, then truncate. It changes the *order* of remaining elements, which is fine wherever order doesn't carry meaning. [§21](21_swap_remove.md) builds the rest of the discipline around it.

## Exercise 7 — Read your own array

```python
>>> import numpy as np
>>> a = np.arange(10, dtype=np.int64)
>>> raw = a.tobytes()
>>> len(raw)                                  # 80 — exactly 10 × 8 bytes
>>> b = np.frombuffer(raw, dtype=np.int64)
>>> (a == b).all()                            # True
```

The bytes you would write to disk *are* the bytes already in memory. There is no serialization step. A typed numpy column is its own on-disk format up to byte-order and dtype. [§36](36_persistence_is_serialization.md) builds on this directly: the persistence layer stores `(N, dtype, raw_bytes)` and round-trips losslessly, with no encoder/decoder pair to maintain.
