# Solutions: 21 — `swap_remove`

## Exercise 1 — Compare timings, simple case

```python
import time
N = 1_000_000

# pop(0) — worst case
lst = list(range(N))
t0 = time.perf_counter()
for _ in range(1_000):
    lst.pop(0)
t1 = time.perf_counter()
print(f"pop(0) × 1000:       {(t1-t0)*1000:.1f} ms")

# swap_remove at front
lst = list(range(N))
t0 = time.perf_counter()
for _ in range(1_000):
    lst[0] = lst[-1]; lst.pop()
t1 = time.perf_counter()
print(f"swap_remove × 1000:  {(t1-t0)*1000:.3f} ms")
```

```
pop(0) × 1000:       380 ms
swap_remove × 1000:    0.2 ms
```

~2000× ratio. `pop(0)` shifts every element down one slot — N pointer copies per call, K × N total. `swap_remove` writes one slot and pops — 2 ops per call, K total. The ratio scales with N.

## Exercise 2 — Mid-table delete

```python
import time, numpy as np
N = 1_000_000

# np.delete returns a fresh array each call
arr = np.arange(N, dtype=np.int64)
t0 = time.perf_counter()
for _ in range(1_000):
    arr = np.delete(arr, 500_000 if len(arr) > 500_000 else 0)
t1 = time.perf_counter()
print(f"np.delete × 1000:    {(t1-t0)*1000:.0f} ms")

# swap_remove
arr = np.arange(N, dtype=np.int64)
n_active = N
t0 = time.perf_counter()
for _ in range(1_000):
    i = 500_000 if i < n_active - 1 else 0
    arr[i] = arr[n_active - 1]
    n_active -= 1
t1 = time.perf_counter()
print(f"swap_remove × 1000:  {(t1-t0)*1000:.3f} ms")
```

`np.delete` re-allocates the entire array each call — N int64s = 8 MB copied per delete. After 1000 calls, 8 GB has been written. swap_remove writes 8 bytes per call. ~10⁵-10⁶× ratio.

## Exercise 3 — Run the §21 exhibit

```sh
uv run code/measurement/swap_remove.py
```

Source: [`code/measurement/swap_remove.py`](https://codeberg.org/root-11/intro-book-python/src/branch/main/code/measurement/swap_remove.py). Removing 100,000 mid-table rows from a 1M-row table:

```
layout                                              time      remove rate
-------------------------------------------------------------------------
Python list, list.pop(i)                            3.59 s          28K ops/s
numpy, np.delete(arr, i)                           23.09 s           4K ops/s
numpy active counter, sequential swap_remove       0.011 s        8.0M ops/s
numpy bulk filter, arr[keep_mask]                  0.004 s       25.4M ops/s
```

Surprises that calibrate intuition:

- **`np.delete` is the slowest** — 6500× slower than the bulk filter. The "numpy way" sounds right but is algorithmically wrong: it reallocates on every call.
- **Python list pop(i) beats `np.delete`** at this scale, because pointer-shifts in a Python list are ~8 bytes each whereas reallocation copies the whole int64 array.
- **Bulk filter is 3× faster than sequential swap_remove**, even though both are O(K). The Python loop crossing the C boundary 100K times has measurable overhead; the bulk version pays the boundary cost once.

For a simulator's cleanup pass: collect `to_remove` indices during the tick (cheap, append-only), then apply with one bulk-filter call at the boundary. This is the §22 pattern.

## Exercise 4 — The iteration hazard

```python
import numpy as np
arr = np.arange(100, dtype=np.int64)
n_active = 100

i = 0
while i < n_active:
    if arr[i] % 2 == 0:
        arr[i] = arr[n_active - 1]
        n_active -= 1
    else:
        i += 1

print(arr[:n_active].tolist())     # should be [1, 3, 5, ..., 99]
```

Without the `else: i += 1` (i.e. plain `for i in range(...)`), the bug is: after a swap, the slot at `i` now holds a *different* value (the one that was at the end). The forward `for` loop has already moved past `i` and won't re-check it. Half the evens get skipped.

The version above with the explicit `while` and conditional increment is correct: when a swap happens, *don't advance* `i` — the slot has new contents that need re-checking. This is the canonical fix when iterating-while-mutating cannot be avoided.

## Exercise 5 — The fix in one shape: iterate backwards

```python
arr = np.arange(100, dtype=np.int64)
n_active = 100
for i in range(n_active - 1, -1, -1):
    if arr[i] % 2 == 0:
        arr[i] = arr[n_active - 1]
        n_active -= 1
print(arr[:n_active].tolist())     # all odds
```

Why it works: when you swap `arr[i] = arr[n_active - 1]`, the slot at `i` now holds the *old last* element, but `i` is decreasing, so we move to `i - 1` next — a slot we have not yet visited. We never re-encounter a swapped slot. The "old last" element gets to be checked for evenness at its new position because that position was never visited by the iteration.

## Exercise 6 — The fix in another shape: deferred cleanup

```python
arr = np.arange(100, dtype=np.int64)
n_active = 100

to_remove = []
for i in range(n_active):
    if arr[i] % 2 == 0:
        to_remove.append(i)

# Apply at end — reverse order so swap_remove indices stay valid
for i in sorted(to_remove, reverse=True):
    arr[i] = arr[n_active - 1]
    n_active -= 1

print(arr[:n_active].tolist())
```

The `for` loop is now read-only — no swap during iteration. Mutations are buffered and applied at the boundary (`for i in sorted(to_remove, reverse=True)`). This is the [§22](22_mutations_buffer.md) pattern: filter (read-only) and apply (single batch) are *separate phases*.

For very large `to_remove` buffers, the bulk-filter form (`arr = arr[~np.isin(np.arange(n_active), to_remove)]`) is faster than per-index swap_remove. Both forms are correct; the bulk one wins on speed.

## Exercise 7 — Aligned per-element swap_remove

```python
class World:
    def __init__(self, n):
        self.pos_x   = np.zeros(n, dtype=np.float32)
        self.pos_y   = np.zeros(n, dtype=np.float32)
        self.vel_x   = np.zeros(n, dtype=np.float32)
        self.vel_y   = np.zeros(n, dtype=np.float32)
        self.energy  = np.zeros(n, dtype=np.float32)
        self.id      = np.arange(n, dtype=np.uint32)
        self.n_active = n

def delete_creature(world: World, slot: int) -> None:
    last = world.n_active - 1
    if slot != last:
        for arr in (world.pos_x, world.pos_y, world.vel_x,
                    world.vel_y, world.energy, world.id):
            arr[slot] = arr[last]
    world.n_active -= 1
```

Each column gets the same `slot` and `last`. Forgetting to apply this to one column produces the [§9](09_sort_breaks_indices.md) misalignment bug. The discipline is: the function above is the *only* place that does swap_remove on a creature; no caller writes to one column without going through it.

## Exercise 8 — Aligned bulk filter

```python
def delete_batch(world: World, indices_to_remove: np.ndarray) -> None:
    keep = np.ones(world.n_active, dtype=bool)
    keep[indices_to_remove] = False
    for name in ("pos_x", "pos_y", "vel_x", "vel_y", "energy", "id"):
        col = getattr(world, name)
        # in-place compress: copy survivors to the front
        n_keep = int(keep.sum())
        col[:n_keep] = col[:world.n_active][keep]
    world.n_active -= len(indices_to_remove)

# spot-check alignment
row17_before = (world.pos_x[17], world.pos_y[17], int(world.id[17]))
delete_batch(world, np.array([5, 13, 87]))
# whichever creature is now at slot 17 — its row tuple should still be coherent
row17_after  = (world.pos_x[17], world.pos_y[17], int(world.id[17]))
# verify that row17_after matches the row in the original world whose id == world.id[17] now
```

The same `keep` mask is applied to every column. *One* boolean indexing pass per column, *one* mask shared across all columns. Forgetting one column lands the broken version: rows misaligned exactly as §9 predicted.

The broken version (apply mask to half the columns):

```python
# anti-pattern: bad! demonstrates the bug
def delete_batch_broken(world, indices):
    keep = np.ones(world.n_active, dtype=bool)
    keep[indices] = False
    world.pos_x = world.pos_x[keep]
    world.pos_y = world.pos_y[keep]
    # forgot vel_x, vel_y, energy, id — they keep their old length and contents
```

Now `pos_x[i]` and `vel_x[i]` are from different rows. Reading "the velocity of the creature at slot i" returns garbage. The fix is structural: one function, all columns, one mask.

## Exercise 9 — The bandwidth cost (stretch)

```python
import numpy as np

# np.delete on a 1 GB int64 array — copies (1 GB - 8 bytes)
arr = np.zeros(1_000_000_000 // 8, dtype=np.int64)   # 125 M elements = 1 GB
arr2 = np.delete(arr, 0)                              # bytes moved ≈ 1 GB

# swap_remove — copies 8 bytes
n_active = len(arr)
arr[0] = arr[n_active - 1]
n_active -= 1                                          # bytes moved = 8
```

Bytes moved per delete:

| operation               | bytes moved | as fraction of array |
|-------------------------|------------:|---------------------:|
| `np.delete(arr, 0)`     | ~1 GB       | ~100%                |
| swap_remove             | 8 bytes     | 8e-9 = 8 nano-percent |

Ratio: ~125,000,000×. At a 30 Hz tick rate, doing one `np.delete` per tick on a 1 GB array would mean moving 30 GB/s — past the bandwidth ceiling of most desktop systems. Doing one swap_remove takes 30 × 8 = 240 bytes per second.

The structural cost dominates the constant factors. This is why the chapter's table shows `np.delete` losing to `list.pop` at this scale: the algorithmic shape is wrong regardless of how typed and contiguous the data is.
