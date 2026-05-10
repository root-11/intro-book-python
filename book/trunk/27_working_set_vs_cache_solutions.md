# Solutions: 27 — Working set vs cache

## Exercise 1 — Compute your working sets

For the motion system (`pos_x, pos_y, vel_x, vel_y, energy` at float32):

| N         | bytes      | cache regime (typical 2026 desktop) |
|----------:|-----------:|--------------------------------------|
|     1,000 |       20 KB | fits L1 (32-48 KB)                  |
|    10,000 |      200 KB | spills to L2 (1-2 MB)               |
|   100,000 |        2 MB | borderline L2/L3                    |
| 1,000,000 |       20 MB | fits L3 (16-32 MB)                  |
|10,000,000 |      200 MB | spills L3 to RAM                    |

The cliff is at the L3 → RAM transition. The exact size depends on your CPU's L3 (run `lscpu` from §1 exercise 1 to confirm).

For the starvation system (reads `energy` only — 4 bytes per creature):

| N         | bytes     |  regime              |
|----------:|----------:|----------------------|
|   100,000 |    400 KB |  L1 cap on this CPU  |
| 1,000,000 |      4 MB |  L2/L3 boundary      |
|10,000,000 |     40 MB |  spills L3           |

The starvation system fits more creatures per cache level than motion, because it touches fewer bytes per row. *Smaller working set, larger headroom.*

## Exercise 2 — Find your cliff

```sh
uv run code/measurement/cache_cliffs.py
```

From §1 — gather column (random access):

```
N           gather (ns/elem)
10,000          1.62
100,000         2.24
1,000,000       3.69
10,000,000      7.60
100,000,000     7.78
```

Transitions visible: 10K → 100K (L1 → L2, ~1.4×), 100K → 1M (L2 → L3, ~1.6×), 1M → 10M (L3 → RAM, ~2.1×). The cliff is shallowest at the L1/L2 boundary and steepest at L3/RAM on this machine.

## Exercise 3 — Reduce the working set

Splitting the motion's row from `(pos_x, pos_y, vel_x, vel_y, energy, birth_t, id, gen)` = 36 bytes to `(pos_x, pos_y, vel_x, vel_y, energy)` = 20 bytes:

- Motion's working set at 1M creatures: 36 MB → 20 MB. Still fits L3.
- At 2M: 72 MB → 40 MB. The 72-MB version spilled to RAM; the 40-MB version fits L3.

So the cliff moves outward by ~1.8× — exactly the bytes-ratio. **But in pure SoA-in-numpy, motion *already* reads only the hot columns** because each column is its own buffer; reading `pos_x` does not touch `birth_t`'s memory. The split is organisational, not a working-set reduction. The chapter's framing applies: timing does not change.

The split *does* reduce working set in *structured array* layout, where reading `arr['pos_x']` strides past `birth_t`'s bytes. Confirmed by exercise 4 of §26: structured array is 8× slower than SoA columns at the same N.

## Exercise 4 — A wider dtype

```python
energy = np.zeros(n, dtype=np.float64)         # was float32 — doubles the bytes
```

Working set per creature: 20 → 24 bytes (one column doubled). Cliff moves *inward* by ~20%. At N=1M, working set 24 MB → still fits typical L3. At N=1.5M, 36 MB → starts to spill. The motion timing rises proportionally to the bytes read (sequential access is bandwidth-bound; bytes moved is the cost).

This is §2's *narrowest-dtype* discipline re-applied at scale. Choosing `float32` over `float64` doubles your population headroom in cache. The choice is not aesthetic — it is "how many creatures can my simulator host at L3-resident speed?"

## Exercise 5 — Random vs sequential, your machine

From your `cache_cliffs.py` output:

| size       | gather/seq |
|-----------:|-----------:|
| 10K        |   2-4×     |
| 100K       |   ~10×     |
| 1M         |   ~20×     |
| 10M        |   ~40-50×  |
| 100M       |   ~50-80×  |

The 100M figure is **your machine's L1-to-RAM cost gap on this run**. On modern desktops 50-80×; on Pi 4 / 2012 Intel, closer to 30-40×; on Apple Silicon, somewhere in between.

Memorise the number. When a colleague says "the data structure I wrote does random lookups; I think it's fast," ask them for N. If N puts the working set past L3, multiply their best-case estimate by your machine's gather/seq ratio. That's the real cost.

## Exercise 6 — The L1 sweet spot (stretch)

L1 is ~48 KB on this CPU; 75% = 36 KB. At 20 bytes per row, that's ~1,800 creatures. Closest power-of-10-ish: 1,500-2,000.

```python
import time, numpy as np

for n in (1_500, 1_800, 2_000, 10_000):
    pos_x  = np.zeros(n, dtype=np.float32)
    pos_y  = np.zeros(n, dtype=np.float32)
    vel_x  = np.ones(n,  dtype=np.float32)
    vel_y  = np.ones(n,  dtype=np.float32)
    energy = np.zeros(n, dtype=np.float32)
    dt = 1/30.0
    # warm up
    for _ in range(50):
        pos_x += vel_x * dt; pos_y += vel_y * dt
    t = time.perf_counter()
    for _ in range(1_000):
        pos_x += vel_x * dt; pos_y += vel_y * dt
    elapsed = (time.perf_counter() - t) / 1_000
    print(f"N={n:>5}: motion {elapsed*1e6:.2f} µs ({elapsed*1e9/n:.2f} ns/elem)")
```

Expected pattern: N=1500 and N=1800 stay around 0.2 ns/elem (L1-resident). N=10,000 jumps to 0.5-0.8 ns/elem (L2-resident — 3-5× slower).

The L1-resident regime is where you want hot inner loops to live. Any code path that runs every tick over a small data set should be sized so the data fits L1 — that's the difference between "fast" and "very fast." For the simulator, this matters most for per-creature derived columns (an `urgency_score` of length N_hot) that are computed and consumed within a single system.
