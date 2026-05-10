# 28 — Sort for locality

> *Concept node: see the [DAG](../../concepts/dag.md) and [glossary entry 28](../../concepts/glossary.md#28--sort-for-locality).*

<p align="center"><img src="../illustrations/optimization.jpg" alt="Optimization: minimize f(x) — sorting for locality is reordering for cost" style="max-height: 300px; max-width: 100%;"></p>

In [§9](09_sort_breaks_indices.md) you learned the sort-breaks-indices bug. In [§10](10_stable_ids_and_generations.md) you fixed it with stable ids. In [§23](23_index_maps.md) you made id-to-slot lookup O(1). With those three pieces in place, the simulator can now do something it could not before: rearrange its rows for locality.

The principle is simple. Rows accessed near each other in time should sit near each other in memory. Two creatures that interact (collide, query a neighbour, broadphase against each other) should land on adjacent cache lines.

The classic technique is a *spatial sort*. Each creature's position is hashed to a spatial cell; the creatures table is sorted by cell. Reading "all creatures in cell C" becomes a contiguous range read.

```python
def spatial_cell(pos_x: np.ndarray, pos_y: np.ndarray,
                 cell_size: float) -> np.ndarray:
    """Returns a uint32 cell id for each creature. Pack (x, y) cells
       into one integer. Z-order or Hilbert curves work too."""
    cx = (pos_x / cell_size).astype(np.int32)
    cy = (pos_y / cell_size).astype(np.int32)
    return ((cx & 0xFFFF) << 16) | (cy & 0xFFFF)


def sort_creatures_for_locality(world, cell_size: float) -> None:
    cells = spatial_cell(world.pos_x, world.pos_y, cell_size)
    order = np.argsort(cells, kind="stable")
    # Apply the same permutation to every column, in lockstep — §6's rule.
    world.pos_x[:] = world.pos_x[order]
    world.pos_y[:] = world.pos_y[order]
    world.vel_x[:] = world.vel_x[order]
    world.vel_y[:] = world.vel_y[order]
    world.energy[:] = world.energy[order]
    world.id[:]     = world.id[order]
    # Rebuild id_to_slot in lockstep — §23.
    world.id_to_slot[world.id] = np.arange(world.id.size, dtype=np.uint32)
```

Two creatures in the same spatial cell are now adjacent in `pos_x` and `pos_y`. The next-event system, which checks every creature against its spatial neighbours, strides through `pos_x` and reads neighbours from the same cache line.

## Why this matters in numpy

The locality gap is not theoretical. From [§1's `cache_cliffs.py`](../../code/measurement/cache_cliffs.py), at 100M elements the gather (random-index) read is **72× slower than sequential** on this machine. That ratio is the cost of every cache-unfriendly access pattern — every iteration that visits creatures in a non-spatial order pays it. Spatial sort converts gather-shaped reads into sequential ones, **which is exactly the operation that ratio measures.**

The cost is the sort itself. At 1M `uint32` keys, `np.argsort` takes ~10-30 ms depending on input distribution. Done every tick this would be too expensive — but typically the sort is done every ~100 ticks (or when accumulated motion exceeds a threshold), amortising to ~0.1-0.3 ms per tick. The savings on the inner loop dwarf the cost.

## Other sort orders, when they pay off

- **Sort by id.** Stable across runs; nice for debugging; but no locality benefit unless ids correlate with access patterns.
- **Sort by access frequency.** Hot creatures first; cold last. Useful only when the inner loop respects the order — and most numpy bulk ops do not, they walk the whole column.
- **Sort by behaviour.** All hungry creatures together; all sleepy together. Mostly redundant in a presence-based system ([§19](19_ebp_dispatch.md)) where the hungry-driver iterates `hungry` directly.

Sort cadence is its own decision. Sorting every tick is wasted work if the world is mostly stationary. Sorting once at startup is wrong if the world drifts. Most simulators trigger a re-sort when accumulated motion since the last sort exceeds a fraction of the cell size.

## The pieces this lesson assumes

The sort interacts with three earlier lessons:

- **Lockstep reordering ([§6](06_a_row_is_a_tuple.md), [§9](09_sort_breaks_indices.md)).** Every column gets the same permutation applied. The `world.pos_x[:] = world.pos_x[order]` form is in-place rebinding to the same backing array — it does not allocate, and it does not break aliases held elsewhere. Doing this column-by-column for every column is the disciplined form.
- **Stable ids ([§10](10_stable_ids_and_generations.md)).** Code outside the sort holds *ids*, not slots; the sort moves slots, and the `id_to_slot` map (the last line of `sort_creatures_for_locality`) keeps the ids correct.
- **Index maps ([§23](23_index_maps.md)).** The `id_to_slot` rebuild is one bulk numpy assignment that runs in O(N) once per sort, not O(N) per id. The pieces compose.

This is the pattern Bevy, Unity DOTS, Unreal's Mass Entities, and most production ECS engines use under the hood. Locality is paid up front (one sort) and amortised over many cache-friendly inner loops.

## Exercises

1. **Compute spatial cells.** Write `spatial_cell(pos_x, pos_y, cell_size)` as in the prose. Apply it to a 1,000-creature world with random positions. Print `np.bincount` of the cell ids; this is the histogram of how many creatures land in each cell.
2. **Sort by cell.** Implement `sort_creatures_for_locality` with the lockstep column reorder. Run it. Verify: print the first ten `(pos_x, pos_y)` after the sort — these should be near-neighbour positions, not random ones.
3. **Maintain `id_to_slot`.** Confirm the `id_to_slot[world.id] = np.arange(N)` rewrite resolves correctly. Take a held id from before the sort; look up its slot after; confirm `pos_x[slot]` is the same value as before (it has just moved).
4. **Time `next_event` before and after.** Write a `next_event` system that, for each creature, scans the next 100 entries of `pos_x, pos_y` for collisions. Time it pre-sort vs post-sort at 100,000 creatures. The post-sort version should be measurably faster — by how much depends on how much the scan happens to land in the same cache line.
5. **Sort cadence.** Run a 100-tick simulation, sorting every tick. Run the same simulation, sorting every 10 ticks, and every 100 ticks. Compare total cost (sort cost + neighbour-scan cost). Find the cadence where sort cost equals neighbour-scan savings — that is your sweet spot.
6. *(stretch)* **Z-order curve.** Replace the simple `(x, y)` packing with a Z-order (Morton) hash — interleave the bits of `cx` and `cy`. Compare `next_event` timings. Z-order keeps spatially close cells close in the linear order; it usually outperforms simple stripe packing because two-cell horizontal neighbours stay adjacent.

Reference notes in [28_sort_for_locality_solutions.md](28_sort_for_locality_solutions.md).

## What's next

[§29 — The wall at 10K → 1M](29_wall_10k_to_1m.md) is where these techniques start to bind. Code that ran fine at 10K stops running fine at 1M; the chapter is about finding out where and why.
