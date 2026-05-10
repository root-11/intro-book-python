# Solutions: 28 — Sort for locality

## Exercise 1 — Compute spatial cells

```python
import numpy as np

def spatial_cell(pos_x: np.ndarray, pos_y: np.ndarray, cell_size: float) -> np.ndarray:
    cx = (pos_x / cell_size).astype(np.int32)
    cy = (pos_y / cell_size).astype(np.int32)
    return ((cx & 0xFFFF) << 16) | (cy & 0xFFFF)

rng = np.random.default_rng(0)
n = 1_000
pos_x = rng.uniform(0, 100, n).astype(np.float32)
pos_y = rng.uniform(0, 100, n).astype(np.float32)

cells = spatial_cell(pos_x, pos_y, cell_size=10.0)
unique, counts = np.unique(cells, return_counts=True)
print(f"{len(unique)} cells occupied, max {counts.max()} creatures per cell")
print(f"first 5 cells: {unique[:5].tolist()}")
print(f"histogram of cell counts: {np.bincount(counts)[:20]}")
```

For uniformly distributed creatures in a 100×100 world with 10-unit cells, expect ~100 cells (10×10 grid), ~10 creatures per cell on average. The histogram is Poisson-shaped — most cells have 5-15 creatures, a few have 0 or 25+.

## Exercise 2 — Sort by cell

```python
def sort_for_locality(world, cell_size: float):
    cells = spatial_cell(world.pos_x, world.pos_y, cell_size)
    order = np.argsort(cells, kind="stable")
    for col in ("pos_x", "pos_y", "vel_x", "vel_y", "energy", "id"):
        arr = getattr(world, col)
        arr[:] = arr[order]
    # rebuild id_to_slot
    world.id_to_slot[world.id[:world.n_active]] = np.arange(world.n_active, dtype=np.uint32)

sort_for_locality(world, cell_size=10.0)
print(f"first 10 positions after sort:")
for i in range(10):
    print(f"  ({world.pos_x[i]:.2f}, {world.pos_y[i]:.2f}) cell={spatial_cell(world.pos_x[i:i+1], world.pos_y[i:i+1], 10.0)[0]}")
```

After the sort, the first 10 positions belong to creatures in the same (or adjacent) cells — their `(pos_x, pos_y)` values cluster instead of scattering randomly.

## Exercise 3 — Maintain `id_to_slot`

```python
# Before sort: held_id is at some slot
held_id = int(world.id[42])
before_slot = 42
before_pos  = (float(world.pos_x[42]), float(world.pos_y[42]))

sort_for_locality(world, cell_size=10.0)

# After sort: look up by id
after_slot = int(world.id_to_slot[held_id])
after_pos  = (float(world.pos_x[after_slot]), float(world.pos_y[after_slot]))

print(f"before: slot={before_slot}, pos={before_pos}")
print(f"after:  slot={after_slot}, pos={after_pos}")
assert before_pos == after_pos, "data moved but is the same value"
```

The held id resolves to a new slot. The position at the new slot equals the position at the old slot. The id_to_slot map is the bridge; without it, the held reference would dereference garbage.

## Exercise 4 — Time `next_event` before and after

```python
import time, numpy as np

def next_event_scan(pos_x, pos_y, radius=1.0):
    """For each creature, count neighbours within radius among the next 100 entries."""
    n = len(pos_x)
    count = np.zeros(n, dtype=np.uint32)
    for i in range(n):
        end = min(i + 100, n)
        dx = pos_x[i+1:end] - pos_x[i]
        dy = pos_y[i+1:end] - pos_y[i]
        count[i] = int(np.sum(dx*dx + dy*dy < radius*radius))
    return count

# Pre-sort timing
t = time.perf_counter()
next_event_scan(world.pos_x[:10_000], world.pos_y[:10_000])
t_pre = time.perf_counter() - t

sort_for_locality(world, cell_size=10.0)

# Post-sort timing
t = time.perf_counter()
next_event_scan(world.pos_x[:10_000], world.pos_y[:10_000])
t_post = time.perf_counter() - t

print(f"pre-sort:  {t_pre*1000:.2f} ms")
print(f"post-sort: {t_post*1000:.2f} ms")
print(f"ratio:     {t_pre/t_post:.2f}×")
```

Expect a ~1.5-3× speedup on the post-sort version. The reason: post-sort, the `pos_x[i+1:end]` slice is more likely to contain creatures in the same spatial cell — so the boolean mask has more `True` values clustered together, and the subsequent indexing operations are more cache-friendly.

The exact ratio depends on the spatial distribution and the scan-window size. A scan window of 100 might capture exactly one cell (if cells average 10 creatures) or several adjacent cells; the locality benefit is biggest when the scan window matches the typical cell occupancy.

## Exercise 5 — Sort cadence

```python
results = {}
for cadence in (1, 10, 100, 1_000_000):       # last one = "never"
    world = build_world(n=10_000)
    t0 = time.perf_counter()
    for tick in range(100):
        motion(world, dt=1/30)
        if tick % cadence == 0:
            sort_for_locality(world, cell_size=10.0)
        next_event_scan(world.pos_x, world.pos_y)
    results[cadence] = time.perf_counter() - t0
for c, t in results.items():
    print(f"sort every {c:>8} ticks: {t:.2f} s total")
```

Typical shape:

```
sort every       1 ticks: 0.85 s   (sort cost dominates)
sort every      10 ticks: 0.62 s   (sweet spot, often)
sort every     100 ticks: 0.75 s   (scan cost grows as positions drift)
sort every 1000000 ticks: 1.20 s   (no resort; scan cost stays high)
```

The optimum is wherever the sort's amortised cost balances the scan's per-tick savings. For most simulators that's "resort every 10-100 ticks," depending on motion speed. A re-sort triggered by *accumulated drift* (resort once total motion since last sort exceeds half a cell width) generalises this to scenarios with variable motion rates.

## Exercise 6 — Z-order curve (stretch)

```python
def _spread2(v: int) -> int:
    """Interleave 16 bits of v with zeros (Morton helper)."""
    v &= 0xFFFF
    v = (v | (v << 8)) & 0x00FF00FF
    v = (v | (v << 4)) & 0x0F0F0F0F
    v = (v | (v << 2)) & 0x33333333
    v = (v | (v << 1)) & 0x55555555
    return v

def morton_cell(pos_x: np.ndarray, pos_y: np.ndarray, cell_size: float) -> np.ndarray:
    cx = np.clip((pos_x / cell_size).astype(np.int32), 0, 0xFFFF)
    cy = np.clip((pos_y / cell_size).astype(np.int32), 0, 0xFFFF)
    return np.array([(_spread2(int(x)) | (_spread2(int(y)) << 1)) for x, y in zip(cx, cy)],
                    dtype=np.uint32)
```

For pure-numpy efficiency, vectorise `_spread2`:

```python
def spread_vec(v: np.ndarray) -> np.ndarray:
    v = v & 0xFFFF
    v = (v | (v << 8)) & 0x00FF00FF
    v = (v | (v << 4)) & 0x0F0F0F0F
    v = (v | (v << 2)) & 0x33333333
    v = (v | (v << 1)) & 0x55555555
    return v

def morton_cell(pos_x, pos_y, cell_size):
    cx = np.clip((pos_x / cell_size).astype(np.int32), 0, 0xFFFF)
    cy = np.clip((pos_y / cell_size).astype(np.int32), 0, 0xFFFF)
    return spread_vec(cx) | (spread_vec(cy) << 1)
```

Compared to the simple `(cx << 16) | cy` packing, Z-order keeps cells (1,0), (0,1), (1,1) close to (0,0) in the linear order — instead of (1,0) being adjacent to (0,0) but (0,1) being far away. The result is that 2D adjacency is *approximately* preserved in 1D adjacency.

For typical simulator workloads where `next_event_scan` looks at horizontal-and-vertical neighbours, Z-order outperforms simple packing by 10-30%. The difference is biggest for densely-packed simulations where vertical neighbours within the scan window matter.

The full Hilbert curve preserves 2D locality even better but is more expensive to compute. For most simulators, Z-order is the sweet spot — close to optimal, vectorisable in numpy.
