# Solutions: 4 — Cost is layout, and you have a budget

## Exercise 1 — Pick your rates

| system                   | target rate            | budget per tick |
|--------------------------|------------------------|-----------------|
| card game                | 30 Hz (or event-driven) | 33 ms          |
| real-time strategy game  | 30-60 Hz                | 17-33 ms       |
| market data feed         | depends — 100 Hz – 1 MHz | 10 ms – 1 µs |
| embedded sensor controller | 1-10 kHz              | 100 µs – 1 ms  |
| web API endpoint         | per-request, ~10-200 ms | 10-200 ms      |
| offline batch (1B rows)  | throughput, not Hz      | minutes-to-hours total |

The point of writing these down is that "this should be fast" is not a budget. "33 ms" is. The instant you have a number, every line of code in the inner loop is either spending bytes of that budget or it isn't.

## Exercise 2 — Count an operation

```python
import timeit
d = {i: i*i for i in range(1_000_000)}
t = timeit.timeit("d[42]", globals={"d": d}, number=10_000_000)
print(f"dict[k] lookup: {t/10_000_000*1e9:.1f} ns")
```

```
dict[k] lookup: 15.1 ns
```

At 30 Hz (33 ms): **~2.2 million** lookups per tick.
At 1 kHz (1 ms): **~66,000** lookups per tick.

A 1-million-entity update that does *one* dict lookup per entity would cost 15 ms — half a 30 Hz budget on a single bookkeeping op. Two dict lookups per entity blows the budget on bookkeeping alone, with no actual simulation work done yet.

## Exercise 3 — The layout difference

```python
import time, numpy as np
n = 1_000_000
arr = np.arange(n, dtype=np.int64)
d   = {i: i for i in range(n)}
arr.sum(); sum(d.values())                     # warmup

t0 = time.perf_counter(); int(arr.sum());    t1 = time.perf_counter()
t2 = time.perf_counter(); sum(d.values());   t3 = time.perf_counter()

print(f"numpy sum:        {(t1-t0)*1e9/n:.2f} ns/elem")
print(f"sum(d.values()):  {(t3-t2)*1e9/n:.1f} ns/elem")
```

```
numpy sum:        0.20 ns/elem
sum(d.values()):  3.6 ns/elem
ratio: 18×
```

The dict version is **interpreter-bound**: the inner loop is a pure-Python `for v in values: total += v`, which pays bytecode dispatch + `PyLong` arithmetic + refcount work per element — about 3-6 ns. The numpy version is **bandwidth-bound**: a tight C loop reading int64s sequentially, the prefetcher loaded ahead, the L1 line warm. Same 1M `int64` payload, two regimes apart, 18× cost gap.

## Exercise 4 — The cliff

```python
import time, numpy as np
for size in [100_000, 200_000, 1_000_000, 10_000_000]:
    a = np.ones(size, dtype=np.int64)
    a.sum()                                       # warmup
    best = float("inf")
    for _ in range(3):
        t0 = time.perf_counter(); a.sum(); t1 = time.perf_counter()
        if t1 - t0 < best: best = t1 - t0
    print(f"  N={size:>10,} ({size*8/1024:>7.0f} KB): {best*1e9/size:.2f} ns/elem")
```

```
  N=   100,000 (    781 KB):  0.11 ns/elem
  N=   200,000 (   1562 KB):  0.12 ns/elem
  N= 1,000,000 (   7812 KB):  0.11 ns/elem
  N=10,000,000 (  78125 KB):  0.20 ns/elem
```

On this machine the cliff between L2-fitting (200 KB - 1 MB) and L3-spilling (10 MB+) is shallow on the *sequential* sum (0.11 → 0.20 ns/elem, less than 2× slowdown). The prefetcher is doing its job: even with the working set in RAM, sequential-access numpy hovers near memory bandwidth limits. The dramatic cliff is on the *gather* column from §1; sequential numpy is forgiving.

This is why the chapter distinguishes bandwidth-bound from latency-bound: same N, same array, very different cliff depending on access pattern. The cliff exists; sequential numpy hides most of it.

## Exercise 5 — Working backwards from the budget

Target 60 Hz (16.67 ms = 16,666 µs); 100,000 entities; one cache line touched per entity.

| regime              | per-element | for 100K entities | % of 60 Hz budget |
|---------------------|------------:|------------------:|------------------:|
| compute-bound       |    ~1 ns    |       100 µs      |        0.6%       |
| bandwidth-bound     |  ~0.2 ns    |        20 µs      |        0.1%       |
| latency-bound       |   ~12 ns    |     1,200 µs      |        7.2%       |
| interpreter-bound   |    ~5 ns    |       500 µs      |        3.0%       |

100K is small enough that even a Python-loop version fits comfortably. Scale to 10M:

| regime              | per-element | for 10M entities  | % of 60 Hz budget |
|---------------------|------------:|------------------:|------------------:|
| compute-bound       |    ~1 ns    |    10,000 µs      |       60%         |
| bandwidth-bound     |  ~0.2 ns    |     2,000 µs      |       12%         |
| latency-bound       |   ~12 ns    |   120,000 µs      |      720% (over)  |
| interpreter-bound   |    ~5 ns    |    50,000 µs      |      300% (over)  |

At 10M entities, latency-bound and interpreter-bound layouts blow the budget by 3-7×. Bandwidth-bound finishes with 88% headroom. Same algorithm, same data, same machine.

## Exercise 6 — A bad design

```python
from dataclasses import dataclass

@dataclass
class Entity:
    x: float
    y: float
    vx: float
    vy: float

entities = [Entity(0.0, 0.0, 0.1, 0.1) for _ in range(1_000_000)]

# per tick:
for e in entities:
    e.x += e.vx
    e.y += e.vy
```

This is the canonical "obviously fast" Python design. Big-O is O(N); the inner work is two floating-point adds. Estimating from the regime table: interpreter-bound at ~5 ns × 4 attribute touches ≈ 20 ns/entity × 1M = **20 ms per tick**, ~60% of a 30 Hz budget on simulation work alone. The exhibit `tick_budget.py` confirms this empirically:

```
  1,000,000  Python dataclass list      27.525 ms     30 Hz: 82.6%     60 Hz: 165% OVER
  1,000,000  numpy SoA                   0.278 ms     30 Hz:  0.8%     60 Hz:  1.7%
```

100× cost gap. The big-O is the same. The constant factor — the per-element interpreter dispatch through four attribute accesses on a heap-allocated `dataclass` — is what blows the budget.

## Exercise 7 — Find your CPU's TDP

Linux:
```sh
sudo dmidecode -t processor | grep -i 'power\|TDP'
```

Or look up the CPU model on the manufacturer's spec sheet (Intel ARK, AMD product page, Apple silicon spec). Typical 2026 figures:

| segment            | sustained TDP |
|--------------------|--------------:|
| Raspberry Pi 5     |    ~5 W       |
| ultrabook (mobile) |   15-28 W     |
| desktop            |   65-125 W    |
| workstation        |   125-280 W   |

Burst can run 1.5-2× higher for tens of seconds; sustained settles back to TDP. The number matters because it's the ceiling for *energy per tick* on your machine — useful when budgeting battery life or cooling.

## Exercise 8 — Battery budget

50 Wh laptop battery, simulator at 30 Hz:

- 8 W draw: **6.25 hours** runtime.
- 14 W draw (after a layout change): **3.57 hours** runtime.

The layout change cost 2.68 hours, or **43% of battery life**. A change that adds memory loads to the inner loop is a change that shortens battery life by nearly half. For mobile, embedded, or any battery-powered work, this matters more than the wall-clock tick time.

## Exercise 9 — Measure delta power

```sh
# Terminal 1: sustained sequential numpy sum
python3 -c "
import numpy as np
arr = np.arange(10_000_000, dtype=np.int64)
while True: _ = int(arr.sum())
"

# Terminal 2: read package energy over 30 seconds
sudo perf stat -a -e power/energy-pkg/ -- sleep 30
```

Repeat for the random-gather version (`arr[idx].sum()` with shuffled `idx`) and for an idle baseline. Convert each to average watts (J/30s = W).

Expected ordering: **idle < sequential < gather**. The gap between sequential and gather is the energy cost of breaking the prefetcher — same arithmetic, same data volume, but more memory-controller and DRAM activity per useful operation.

This exercise needs root for `perf` access to RAPL counters, and works on x86 Linux. On macOS, `powermetrics` is the analog. On bare-metal embedded, an external power meter is the honest answer.

## Exercise 10 — Joules per access

Approximate energies per memory read:

| level | energy per access |
|-------|------------------:|
| L1    |          ~0.1 nJ  |
| L2    |          ~1   nJ  |
| RAM   |          ~30  nJ  |

For 10M `int64` reads:

- **Sequential (mostly prefetched)**: assume mostly L1-equivalent cost. 10⁷ × 0.1 nJ = 1 mJ.
- **Random gather (mostly RAM misses)**: 10⁷ × 30 nJ = 300 mJ.

300× more energy. Convert: 1 mJ = 0.28 µWh; 300 mJ = 83 µWh. As a fraction of a 50 Wh battery: 5.6 × 10⁻⁹ vs 1.7 × 10⁻⁶ — both tiny in absolute terms. The *ratio* is what compounds across millions of ticks per day across millions of laptops, or across the lifetime power bill of a data centre. The disciplined layout is also the cheap one, twice over: faster *and* cooler per useful operation.
