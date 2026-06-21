# Solutions: 28 - Proximity is a property of position

All measured on the Ryzen 9 270; the full runnable versions are in [`proximity.py`](https://github.com/root-11/intro-book-python/blob/main/code/measurement/proximity.py).

## Exercise 1 - The all-pairs wall

```python
def all_pairs_count(px, py, r, chunk=1000):
    n = px.size; r2 = r * r; out = np.empty(n, dtype=np.int64)
    for s in range(0, n, chunk):                       # chunk so the N x N distances fit memory
        e = min(s + chunk, n)
        dx = px[s:e, None] - px[None, :]; dy = py[s:e, None] - py[None, :]
        out[s:e] = ((dx * dx + dy * dy) <= r2).sum(axis=1)
    return out
```

```
N=  2000:    24.7 ms
N=  5000:   125.0 ms
N= 10000:   712.2 ms
N= 20000:  2874.3 ms
```

Doubling N roughly quadruples the time - O(N²). At 20K the single neighbour pass is ~2.9 s, ~86 frames of a 30 Hz budget for one query over a fraction of the world. The vectorised broadcast is C-speed per element, but there are N² elements; vectorisation does not save you from the wrong complexity.

## Exercise 2 - Cell as a derived column

```python
def cell_of(px, py, cell_size, ncols):
    return (px / cell_size).astype(np.int64) * ncols + (py / cell_size).astype(np.int64)
```

One vectorised expression, one cheap arithmetic op per creature. It rides along in the pass that already touches `px`, `py` (motion), so its marginal cost is a few milliseconds at 1M - free relative to anything that queries it.

## Exercise 3 - The CSR bin

```python
def build_csr(px, py, r):
    ncols = int(WORLD / r) + 2
    cell = cell_of(px, py, r, ncols)
    order   = np.argsort(cell, kind="stable")          # point indices grouped by cell
    offsets = np.zeros(ncols * ncols + 1, dtype=np.int64)
    np.cumsum(np.bincount(cell, minlength=ncols * ncols), out=offsets[1:])
    return order, offsets, ncols
```

Cell `c`'s members are `order[offsets[c]:offsets[c+1]]`. Three vectorised passes (histogram, prefix-sum, scatter-by-sort), all over contiguous memory. No dict, no per-cell allocation.

## Exercise 4 - The trap, then the fix

The Python loop (read each creature's 3x3 block) is O(1) per query but interpreter-bound:

```
naive Python-loop grid @100k :  447.2 ms   vs cKDTree  90.0 ms   ->  5.0x SLOWER
```

This is the result that sends people back to the library. It is wrong about the grid. Run the million reads as one batch - generate every candidate pair from the CSR with a cumsum range-expansion, then one vectorised distance pass:

```python
def expand_ranges(starts, ends):
    """Flat (src, pos): for each i, the range [starts[i], ends[i])."""
    lengths = ends - starts; mask = lengths > 0
    starts = starts[mask]; lengths = lengths[mask]; src_ids = np.nonzero(mask)[0]
    if starts.size == 0:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    total = int(lengths.sum()); out = np.ones(total, dtype=np.int64); out[0] = starts[0]
    out[np.cumsum(lengths)[:-1]] = starts[1:] - (starts[:-1] + lengths[:-1]) + 1
    return np.repeat(src_ids, lengths), np.cumsum(out)

def grid_query(px, py, r):
    n = px.size; r2 = r * r
    cx = (px / r).astype(np.int64); cy = (py / r).astype(np.int64)
    order, offsets, ncols = build_csr(px, py, r)
    counts = np.zeros(n, dtype=np.int64)
    for dx in (-1, 0, 1):                              # 9 iterations, not n
        for dy in (-1, 0, 1):
            ncx = cx + dx; ncy = cy + dy
            valid = (ncx >= 0) & (ncx < ncols) & (ncy >= 0) & (ncy < ncols)
            nc = np.where(valid, ncx * ncols + ncy, 0)
            starts = offsets[nc]; ends = np.where(valid, offsets[nc + 1], starts)
            src, pos = expand_ranges(starts, ends)
            j = order[pos]
            hit = (px[src] - px[j])**2 + (py[src] - py[j])**2 <= r2
            counts += np.bincount(src[hit], minlength=n)
    return counts
```

```
vectorised grid : 1026.5 ms      cKDTree : 2341.3 ms     ->  grid 2.28x FASTER
```

Validate it against exercise 1's brute force at small N - the counts are identical. The loop became nine fixed iterations; everything inside is a whole-array op. The grid is O(N), the tree O(N log N), so once the interpreter is out of the inner loop the grid's better complexity shows.

## Exercise 5 - Recompute beats maintain

```
CSR rebuild (argsort 1M) :   63.1 ms
full query               : 1026.5 ms      rebuild / query = 6.2%
```

Rebuilding the whole structure from scratch is ~6% of the query it serves. Maintaining incrementally - patching only the ~1k creatures that crossed a cell this tick - is cheaper still (a dict patch is ~0.3 ms). But it does not help, because the *vectorised* query needs the sorted CSR, and a CSR is not incrementally patchable (inserting into a packed bucket shifts everything after it). The structure you can patch cheaply is a dict of lists - and a dict cannot feed the vectorised candidate-generation; its per-beast read is the slow loop from exercise 4. You cannot have cheap-maintain and fast-query at once, so you rebuild the CSR each tick and pay the 6%. The "how often to re-sort" knob disappears.

## Exercise 6 - The pack-leader

```python
def cohesion_all_pairs(px, py):                        # each agent -> mean of all others, O(N^2)
    out = np.empty(px.size, dtype=np.float32); chunk = 1000
    for s in range(0, px.size, chunk):
        e = min(s + chunk, px.size)
        out[s:e] = (px[s:e, None] - px[None, :]).mean(axis=1)
    return out

def cohesion_leader(px, py):                            # one centroid, every agent reads it, O(N)
    return px - px.mean(), py - py.mean()
```

```
N=20000:  all-pairs cohesion  565.4 ms      centroid  13.5 us   ->  42024x
```

The leader does the one expensive thing (decide the centre); every member reads one value and steers relative to it. No agent knows about any other - the swarm-like behaviour falls out of each member tracking the shared leader. The all-pairs version computes the same global average N times over; the leader computes it once.

## Exercise 7 - Z-order and the compaction (stretch)

A stripe pack (`cx * ncols + cy`) keeps a row of cells adjacent in memory but jumps a full stripe between vertically-adjacent cells. A Z-order (Morton) hash interleaves the bits of `cx` and `cy`, so 2D neighbours stay close in 1D:

```python
def morton(cx, cy):
    def part(v):
        v = (v | (v << 8)) & 0x00FF00FF
        v = (v | (v << 4)) & 0x0F0F0F0F
        v = (v | (v << 2)) & 0x33333333
        v = (v | (v << 1)) & 0x55555555
        return v
    return part(cx) | (part(cy) << 1)
```

Order the [§24](24_append_only_and_recycling.md) compaction by Morton cell and the neighbour query's gather reads adjacent cells from adjacent memory. The remaining query cost after the scattered gather is removed is the candidate-distance arithmetic itself - which is the irreducible work. This is the [§26](26_subscription_tables.md) compaction doing double duty: it reclaims dead slots *and* makes the spatial gather stream.

## Exercise 8 - The density wall and the representative

In a fixed world, growing the population grows the *density*: a cell of fixed size holds more creatures, so the 3x3 block a query reads holds O(N) of them, and the per-query cost is O(N) - over all targets, O(N²). The vectorised grid did not buy O(N); it bought O(N) *at constant density*. Sweeping 100K -> 300K -> 1M in a fixed 100x100 world, the per-query time grows faster than 3x per step (the reference forage grew ~9-14x) - the quadratic, with the grid still in place.

The representative breaks it. Keep one occupant per cell (the first scattered in by `argsort` is deterministic and fine); a query reads at most nine representatives regardless of how full the cells are, so the work is O(targets) - linear even in a fixed world (the reference held ~3.3x per 3x where the exhaustive 3x3 scan went ~12x). The catch is that you have answered a slightly different question: "the nearest *representative*," not "the nearest creature." But both the kept and the dropped occupants of a cell sit inside the same cell, so the representative is within one cell-width of the true nearest - and the cell is already the resolution at which the grid knows position. Counting the mismatches, a large fraction differ under crowding yet none by more than a cell, and at the simulator's working density the representative herd and the exact herd stayed within 0.5% of each other. The approximation is bounded by the grid you already chose; for "eat what you can reach," that is free.

Two routes back to O(N), then, and they differ in kind: hold the density (grow the world with the population - a constraint on the simulation), or collapse the cell to a representative (a change to the query). The first keeps the exact nearest; the second trades a within-cell approximation for O(N) in a world that does not grow. The reference simulator's `forage` is this second choice made at scale - it is why the herd stays O(N) in a fixed world.
