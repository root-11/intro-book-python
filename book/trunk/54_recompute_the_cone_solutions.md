# Solutions: 54 - A spreadsheet is a dependency graph

Numbers below are the Ryzen 9 270 / CPython 3.14.5 / numpy 2.4.4 figures from [`code/spreadsheet/spreadsheet.py`](https://github.com/root-11/intro-book-python/blob/main/code/spreadsheet/spreadsheet.py); cross-machine and the >RAM disk pivot are pending.

## Exercise 1 - The cone by hand

Editing `A1`: `B1` reads `A1` (stale), `B2` reads `B1` and `A1` (stale), `T` reads `B1` and `B2` (stale). `A2` reads nothing that changed, so it is untouched. Order: `B1`, then `B2`, then `T` - each after the cells it reads. Editing `A2` instead: `A2` feeds only `B1` (in this sheet), so the cone is `B1`, `B2` (via `B1`), `T` - a different set reached by different edges. The two cones differ because the cone is whatever the change *reaches* along feeds-into edges, and the two inputs feed different things. There is no "below" in a graph; there is only reachable.

## Exercise 2 - Recompute in order

Store cells in a topological order (every cell after the ones it reads) and a full recompute is one forward pass. For a cone: from the edited cell, walk dependents transitively to collect the reachable set, then recompute those *in topological order* (so each is recomputed after its stale inputs). The result is bit-identical to a full recompute, because the topological order guarantees every cell sees fresh inputs - that is the [§14](14_systems_compose_into_a_dag.md) guarantee made executable.

## Exercise 3 - The fill-down crossover

A 200,000 by 50 grid; column `c` reads column `c-1` row by row, so a fill-down of `k` input rows dirties `k` rows across all 50 columns. Cone-recompute (the dirty rows, vectorised per column) against full:

| fill-down rows | cone | vs full (170 ms) |
|---:|---:|---:|
| 10 | 0.10 ms | 1680x |
| 1,000 | 0.25 ms | 674x |
| 20,000 | 3.4 ms | 50x |
| 100,000 | 29 ms | 5.9x |
| 200,000 | 158 ms | 1.08x |

Cone wins enormously when little was filled and converges to the full recompute once the fill-down covers the sheet. You cannot make a *random* dirty set with real edits: a person types one cell or fills down a contiguous run, so the dirty set is always the cone of such an edit, its shape fixed by the formula topology - not sampled at random as the scenegraph's could be.

## Exercise 4 - The sum that is not incremental

A `SUM` over a million-row column. Recompute it after one edit: 0.16 ms. After a hundred thousand edits: 0.16 ms. Identical, because the formula `col.sum()` re-reads the whole column either way - a sum keeps no memory of its old value, so one changed cell forces all million additions. The cone is one cell; the work is the column. To make it incremental you must keep a running total and patch it (add the new value, subtract the old), which trades the re-scan for a maintained aggregate - and [§55](55_floating_point_fragility.md) is about why that subtraction is dangerous in floating point.

## Exercise 5 - Early cutoff

A `MAX` over a 1000-cell column feeding a 100,000-cell dashboard, each dashboard cell its own formula recomputed in a loop:

```python
m = col.max()
if m == old_max:        # validation: did the MAX actually change?
    return              # absorbed - touch none of the dashboard
for i in range(D):      # only reached if MAX changed
    dashboard[i] = m * coef[i]
```

| | time |
|---|---:|
| no cutoff (recompute MAX + 100k dashboard) | 11 ms |
| with cutoff (recompute MAX, unchanged, stop) | 3 µs |

About **4000x**. The principle: *validation is cheaper than recomputation* - checking whether a value changed is O(1), recomputing everything downstream on the assumption that it did is O(dashboard). The gap dwarfs the Rust edition's 54x because the saved work is a hundred thousand *per-cell* recomputes - the interpreter-bound loop the trunk warns about - so skipping it skips the most expensive thing in the language. (If the dashboard were a single uniform numpy column, the recompute would be cheap and the cutoff would save less; it is the per-cell, distinct-formula case where the cutoff is decisive.)

## Exercise 6 - The program goes flat

`sys.getsizeof(("mul", 1234567, 2345678))` plus its members is about 172 bytes. One per cell at a billion cells is ~170 GB - it cannot be allocated, and it is *heavier* than the Rust edition's 160 GB because Python objects carry per-object overhead Rust's `enum` does not. Represented as one template per column - about 300 formulas for a real sheet - the program is ~50 KB. What collapsed: the per-cell formula objects, into per-column templates; and the stored dependency edges, into an implicit "this column reads that one, row by row." That collapse is also what makes the recompute fast: a template over a column is one numpy expression, where a million distinct cell objects would be a million interpreted evaluations.

## Exercise 7 - Peg the memory (stretch)

```python
def tiled_sum(arr, tile=4_000_000):        # arr is a numpy.memmap
    total = np.float64(0.0)
    for s in range(0, arr.size, tile):
        total += arr[s:s + tile].sum(dtype=np.float64)
    return total
```

`tracemalloc`'s peak heap over a 100-million-element column is a fraction of a megabyte and does not move when you feed it ten times the data: `arr[s:s+tile]` is a zero-copy view into the memmap and `.sum` streams the reduction in C, so the loop never materialises more than a tile. Peak memory is bounded by the tile you chose, not by the sheet - so OOM is not unlikely here, it is *impossible by construction*. Size a sheet with RAM < problem < disk (each GB of RAM is 250 million `float32`), lay the columns out one after another on disk, and a patch reads only the dirty columns' bytes - 25x less on a fifty-column sheet than reading the whole file. The wall-clock advantage at a genuinely-larger-than-RAM sheet is the same shape, and is what the Rust edition measures at 36 GB.
