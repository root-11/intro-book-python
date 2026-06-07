# Solutions: 23 - Index maps

## Exercise 1 - Build the map

```python
import numpy as np
INVALID = np.iinfo(np.uint32).max

class World:
    def __init__(self, capacity: int, n_ids: int):
        self.capacity = capacity
        self.n_active = 0
        self.id           = np.zeros(capacity, dtype=np.uint32)
        # ... other columns ...
        self.id_to_slot   = np.full(n_ids, INVALID, dtype=np.uint32)

    def append(self, new_id: int, **fields):
        slot = self.n_active
        self.id[slot] = new_id
        for k, v in fields.items():
            getattr(self, k)[slot] = v
        self.id_to_slot[new_id] = slot
        self.n_active += 1
```

Adding the map is one extra column and one extra line in `append`. Removal updates happen in cleanup (next exercise).

## Exercise 2 - Build the sparse set

```python
INVALID = np.iinfo(np.uint32).max

class SparseSet:
    """dense list of present slots + a slot-indexed map to their position in dense."""
    def __init__(self, n_max: int):
        self.dense  = np.empty(n_max, dtype=np.uint32)         # present slots; walk dense[:n]
        self.sparse = np.full(n_max, INVALID, dtype=np.uint32)  # slot -> position in dense
        self.n = 0

    def is_member(self, i: int) -> bool:
        return bool(self.sparse[i] != INVALID)

    def subscribe(self, i: int) -> None:
        if self.sparse[i] != INVALID:
            return
        self.sparse[i] = self.n
        self.dense[self.n] = i
        self.n += 1

    def unsubscribe(self, i: int) -> None:
        p = self.sparse[i]
        if p == INVALID:
            return
        last = self.dense[self.n - 1]
        self.dense[p] = last          # backfill the hole
        self.sparse[last] = p
        self.sparse[i] = INVALID
        self.n -= 1
```

All three are O(1), no scan, no boolean. `dense[:n]` is the iteration list the hot loop walks; `sparse` answers "present?" *and* hands back the position needed to swap_remove in O(1). A boolean column could do `is_member` but not the removal - and it is the flag §17 abolished.

The cost is one `uint32` per slot for `sparse` plus the `dense` backing - more bytes than a boolean column, but the boolean cannot give you O(1) unsubscribe. Pay it when the membership is stable and only a few entries change per tick (maintain incrementally). When the membership churns almost completely each tick, skip `sparse` entirely and rebuild `dense` from a mask (`np.flatnonzero(predicate)`) - no index needed.

## Exercise 3 - Maintain on bulk-filter cleanup

```python
def cleanup(world, buffer):
    if buffer.to_remove:
        ids = np.unique(np.array(buffer.to_remove, dtype=np.uint32))
        slots = world.id_to_slot[ids]
        keep_mask = np.ones(world.n_active, dtype=bool)
        keep_mask[slots] = False

        # mark the removed ids as no longer in the table
        world.id_to_slot[ids] = INVALID

        # compress every column
        n_keep = int(keep_mask.sum())
        for col_name in world.column_names:
            col = getattr(world, col_name)
            col[:n_keep] = col[: world.n_active][keep_mask]
        world.n_active = n_keep
        # rewrite id_to_slot for survivors - one bulk numpy assignment
        world.id_to_slot[world.id[:n_keep]] = np.arange(n_keep, dtype=np.uint32)
        buffer.to_remove.clear()

    # ... insertions: append new ids and write id_to_slot[new_id] = slot ...
```

The `id_to_slot[ids[:n_keep]] = np.arange(n_keep)` line is the keystone. It rewrites every surviving id's slot in one bulk numpy assignment - exactly the same shape as the column compress, applied to the index map.

## Exercise 4 - Time the difference

```python
import time, numpy as np

world = build_world(n=1_000_000, hungry_count=100_000)
slots = np.random.default_rng(0).choice(1_000_000, size=100_000)

# Linear scan version (§17 ex 6): scan the dense list for the slot
def is_member_scan(hungry, slot):
    return bool(np.any(hungry == slot))

t = time.perf_counter()
for s in slots:
    is_member_scan(world.hungry, int(s))
print(f"linear scan × 100K: {time.perf_counter()-t:.2f} s")

# Sparse-set version: O(1) membership
t = time.perf_counter()
for s in slots:
    world.hungry_set.is_member(int(s))
print(f"sparse set × 100K: {time.perf_counter()-t:.3f} s")
```

Typical: linear scan ~5-10 minutes (10⁵ × 10⁵ = 10¹⁰ ops). Sparse set: ~30 ms (one C-level read per call, plus Python loop overhead). Ratio: ~10⁵-10⁶×.

For a real simulator that does many membership queries per tick, the sparse set is the difference between *workable* and *unsalvageable*. Without it, presence-replaces-flags would only be defensible for whole-table operations, not individual queries.

## Exercise 5 - Run the exhibit (honestly)

```sh
uv run "code/measurement/csr_matrix or python dict.py"
```

```
Benchmarking with a 1000x1000 matrix, 1.0% density (9954 non-zero elements).
Performing 10000 random lookups.

CSR Matrix lookup time:        0.0616 s
Python Dictionary lookup time: 0.00072 s

Python Dictionary is faster for lookups by approximately 85.62 times.
```

The headline ("Dict is 86× faster") is true *for the access pattern in the file* (random scalar lookups). The *right reading* is that scipy gave you a sparse *matrix*, not a sparse *map*. CSR is excellent at:

```python
import numpy as np
from scipy.sparse import csr_matrix

mat = csr_matrix((1000, 1000))
# ... populate ...
v = np.zeros(1000)
result = mat @ v               # SpMV - what CSR is actually for
```

For SpMV at 1000×1000 with 1% density, CSR is dramatically faster than naive dense or dict-based approaches - nine thousand multiplications instead of a million. That's the operation it's optimised for.

The lesson: **pick the structure that matches your access pattern.** A dict is a sparse *point-lookup* map. CSR is a sparse *matrix*. They share the word "sparse" and almost nothing else.

## Exercise 6 - The bandwidth cost

```
1M id_to_slot entries × 4 bytes = 4 MB total
1500 cleanup writes per tick × 4 bytes = 6 KB written
At ~10 GB/s memory bandwidth: ~0.6 µs to write 6 KB
30 Hz tick budget: 33 ms
```

The cleanup map-update cost is **0.002% of the tick budget** at typical mutation rates. The id_to_slot maintenance is invisible against the rest of the work. The 4 MB total memory cost is the dominant concern at scale, not the bandwidth - which mitigates to 400 KB once recycling caps the high-water id count.

## Exercise 7 - Compaction compatibility

```python
def sort_for_locality(world, key_col_name: str):
    """Sort the table in-place by some key (e.g., spatial bucket).
       Updates id_to_slot to reflect the new positions."""
    key = getattr(world, key_col_name)[: world.n_active]
    order = np.argsort(key, kind="stable")

    for col_name in world.column_names:
        col = getattr(world, col_name)
        col[: world.n_active] = col[: world.n_active][order]

    # the keystone again - one bulk update
    world.id_to_slot[world.id[: world.n_active]] = np.arange(world.n_active,
                                                              dtype=np.uint32)
```

After the sort, `world.id[k]` is some new id, and `id_to_slot[world.id[k]] == k`. External code holding a reference to id `42` looks up `id_to_slot[42]`, gets the new slot, reads the (now-relocated) row.

The sort changed every slot. The map update changed every entry of `id_to_slot`. Both are O(N) bulk numpy operations - fast enough to do every tick if needed.

## Exercise 8 - A from-scratch generational arena (stretch)

```python
import numpy as np
from typing import NamedTuple

class CreatureRef(NamedTuple):
    id:  int
    gen: int

INVALID = np.iinfo(np.uint32).max

class SlotMap:
    """Generational arena: stable handles, O(1) lookup, slot recycling, generation checks."""

    def __init__(self, capacity: int = 65536, n_ids: int = 1_000_000):
        self.capacity = capacity
        self.n_active = 0
        self.id    = np.zeros(capacity, dtype=np.uint32)
        self.gens  = np.zeros(capacity, dtype=np.uint32)
        self.value = np.zeros(capacity, dtype=np.float32)
        self.id_to_slot = np.full(n_ids, INVALID, dtype=np.uint32)
        self.next_id = 0

    def insert(self, value: float) -> CreatureRef:
        if self.n_active >= self.capacity:
            raise MemoryError("SlotMap full")
        slot = self.n_active
        new_id = self.next_id
        self.next_id += 1
        self.id[slot]    = new_id
        self.gens[slot]  = 0
        self.value[slot] = value
        self.id_to_slot[new_id] = slot
        self.n_active += 1
        return CreatureRef(id=new_id, gen=0)

    def remove(self, ref: CreatureRef) -> bool:
        slot = self._slot_of(ref)
        if slot is None: return False
        last = self.n_active - 1
        moved_id = int(self.id[last])
        if slot != last:
            self.id[slot]    = self.id[last]
            self.gens[slot]  = self.gens[last]
            self.value[slot] = self.value[last]
            self.id_to_slot[moved_id] = slot
        self.id_to_slot[ref.id] = INVALID
        self.gens[last] += 1                      # bump generation for next reuse
        self.n_active -= 1
        return True

    def get(self, ref: CreatureRef) -> float | None:
        slot = self._slot_of(ref)
        return None if slot is None else float(self.value[slot])

    def _slot_of(self, ref: CreatureRef) -> int | None:
        slot = int(self.id_to_slot[ref.id])
        if slot == INVALID: return None
        if int(self.gens[slot]) != ref.gen: return None
        return slot
```

Compare with [`slotmap::SlotMap`](https://docs.rs/slotmap/) (Rust): same machinery, different organisation. Rust packs `(index, generation)` into one `Key` (a `u64`); we use a `NamedTuple`. Rust uses `Vec<Slot>` with an internal free list; we use an active counter and bump generations on remove. The structural pieces - id allocator, generation array, id_to_slot map, swap_remove on delete - are identical.

Combined with [§22](22_mutations_buffer.md)'s deferred cleanup, this `SlotMap` is the simulator's table primitive. Once you have it, every variable-quantity table in the book reuses the shape - creatures, food, pending events, transition log entries - each one a `SlotMap` with different columns.
