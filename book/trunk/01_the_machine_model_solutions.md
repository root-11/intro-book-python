# Solutions: 1 — The machine model

These exercises are about *measuring your machine*. Numbers vary; ratios are stable. Run them and write down what you see.

## Exercise 1 — Cache sizes

Linux: `lscpu | grep -i cache`. macOS: `sysctl -a | grep cache`.

Typical desktop x86-64 in 2026: L1d 32-48 KB per core, L2 1-2 MB per core, L3 16-128 MB shared. Apple Silicon: larger L1, very large shared L2. Older or smaller chips (Pi 4, 2012-era Intel) show a graded L1 → L2 → L3 → RAM staircase; modern desktops often show one big cliff at L3 → RAM.

Write the numbers down. [§27](27_working_set_vs_cache.md) refers back.

## Exercise 2 — Run the cache-cliffs exhibit

```sh
uv run code/measurement/cache_cliffs.py
```

Source: [`code/measurement/cache_cliffs.py`](https://codeberg.org/root-11/intro-book-python/src/branch/main/code/measurement/cache_cliffs.py).

```
           N     Python list     numpy seq    numpy gather   gather/seq
-----------------------------------------------------------------------
      10,000         5.72 ns      0.420 ns         1.62 ns         3.9×
     100,000         6.01 ns      0.234 ns         2.24 ns         9.6×
   1,000,000         4.78 ns      0.203 ns         3.69 ns        18.2×
  10,000,000         4.46 ns      0.196 ns         7.60 ns        38.7×
 100,000,000         4.59 ns      0.152 ns         7.78 ns        51.3×

Read the columns:
  Python list — roughly flat across sizes; interpreter dispatch dominates.
  numpy seq   — staircase; cliffs reveal L1/L2/L3/RAM transitions.
  numpy gather — random access; gap to seq widens as working set spills caches.
```

The L1 → L2 step in the gather column is shallow (2-3×). The L3 → RAM step is the dramatic one. The Python list column is the chapter's whole point: from inside the interpreter the cache hierarchy is invisible.

## Exercise 3 — Confirm the interpreter mask

Add to the per-N loop in `cache_cliffs.py`:

```python
lst = arr.tolist()
t0 = time.perf_counter()
sum(lst)
ns_lst = (time.perf_counter() - t0) * 1e9 / n
print(f"  list cost: {ns_lst:.2f} ns/elem")
```

Same data, same arithmetic. The number stays in the 4-6 ns/elem band at every N. The cliff is not in the data; it's in what is *touching* the data.

## Exercise 4 — Run the try/except exhibit

```sh
uv run code/measurement/try_except.py
```

Source: [`code/measurement/try_except.py`](https://codeberg.org/root-11/intro-book-python/src/branch/main/code/measurement/try_except.py). Four points along the rate axis from one author's run:

| hits / misses          | try/except (s) | if (s) | if / try-except |
|------------------------|---------------:|-------:|----------------:|
| 1 / 999,999            |          0.509 |  0.047 |         10.75×  |
| 500,000 / 500,000      |          0.297 |  0.072 |          4.12×  |
| 960,000 /  40,000      |          0.100 |  0.097 |          1.03×  |
| 999,999 /       1      |          0.083 |  0.102 |          0.82×  |

At 0% hits (every call raises), `try/except` costs ~11× more. At 50/50, ~4×. Around 96% hits the two cross over. At ~100% hits, `try/except` is the cheaper form because no exception is raised and the path is straight-line; the `if` form pays the comparison every time. The branch predictor does the rest: a branch with a stable outcome predicts ~100% and costs ~0 cycles; a flipping one costs 10-20.

The lesson is not "use one or the other" — it is that constant factors are rate-dependent.

## Exercise 5 — Run the string-format exhibit

```sh
uv run code/measurement/string_methods.py
```

Source: [`code/measurement/string_methods.py`](https://codeberg.org/root-11/intro-book-python/src/branch/main/code/measurement/string_methods.py). Median over seven runs, one author's machine:

|  format       | median (s) |
|---------------|-----------:|
| `%`-format    |      0.477 |
| `.format`     |      0.541 |
| f-string      |      0.547 |

`%`-format wins by ~14%. f-string and `.format` are within 1% of each other on this run; their order flips between CPython versions and between integer-only vs string-heavy payloads. Measure on yours; do not memorise.

## Exercise 6 — A linked list of pointers

```python
import time
import numpy as np

class Node:
    __slots__ = ("value", "next")
    def __init__(self, value, nxt=None):
        self.value = value
        self.next  = nxt

def build(n):
    head = Node(1)
    for _ in range(n - 1):
        head = Node(1, head)
    return head

def walk_sum(head):
    s = 0
    while head is not None:
        s += head.value
        head = head.next
    return s

n = 1_000_000
head = build(n)
arr  = np.ones(n, dtype=np.int64)

t0 = time.perf_counter(); walk_sum(head); t1 = time.perf_counter()
arr.sum();                                 t2 = time.perf_counter()

print(f"linked list: {(t1 - t0) * 1e9 / n:.1f} ns/elem")
print(f"numpy array: {(t2 - t1) * 1e9 / n:.2f} ns/elem")
```

```
linked list: 18.4 ns/elem
numpy array: 0.36 ns/elem
```

Ratio ~50×. That is *not* the full L1-to-RAM ratio, and the reason matters: nodes built in a tight loop land contiguously in memory because the allocator reuses freshly-freed slots. Walking the chain accidentally inherits some of the array's locality; the prefetcher catches part of it.

To see the cost without that accident, link nodes in shuffled order:

```python
import random, gc
nodes = [Node(1) for _ in range(n)]
order = list(range(n))
random.shuffle(order)
for i in range(n - 1):
    nodes[order[i]].next = nodes[order[i + 1]]
head = nodes[order[0]]
del nodes; gc.collect()
```

```
linked list: 107.7 ns/elem
numpy array: 0.36 ns/elem
```

Now each `head.next` is an unpredictable jump — close to a full RAM round-trip per node, ~300× slower than the numpy sum.

The structural label "linked list" doesn't tell you the cost. The *layout in memory* does. `__slots__` is the floor here, not the ceiling — without it, every `Node` carries a `__dict__` and the numbers worsen further.

## Exercise 7 — Reading lscpu against your benchmarks

The transitions are noisy because:

- Cache levels overlap (a hot line stays in L1 after spilling to L2).
- Hardware prefetchers help even shuffled accesses up to a point.
- The OS may evict pages between runs.
- The shuffle is fixed across runs; some indices land near recently-touched lines and amortise.

If your noise is worse than your signal: median of five runs. If transitions still don't line up with `lscpu` (e.g. L2 is 1 MB but the cliff appears at 200 KB), convert byte budgets to elements — the gather array is 8 bytes per `int64`, so 1 MB of L2 holds 128K elements, not 1M.
