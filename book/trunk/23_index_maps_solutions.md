# Solutions: 23 — Index maps

## Exercise 1 — Build the map

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

## Exercise 2 — O(1) presence query

```python
class World:
    hungry: np.ndarray = np.empty(0, dtype=np.uint32)
    hungry_member: np.ndarray = np.zeros(N_max, dtype=bool)

def become_hungry(world, creature_id: int):
    world.hungry = np.concatenate([world.hungry, [creature_id]])
    world.hungry_member[creature_id] = True

def stop_being_hungry(world, creature_id: int):
    world.hungry = world.hungry[world.hungry != creature_id]
    world.hungry_member[creature_id] = False

def is_hungry(world, creature_id: int) -> bool:
    return bool(world.hungry_member[creature_id])
```

Two parallel structures: `hungry` (the iteration list — O(K) walk) and `hungry_member` (the membership map — O(1) check). The list is for iterating; the bool array is for asking "is this id in the table?". Both updated together; one read for each access pattern.

The cost is one byte per id ever issued (~1 MB at 1M ids), which is the memory price of constant-time membership.

## Exercise 3 — Maintain on bulk-filter cleanup

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
        # rewrite id_to_slot for survivors — one bulk numpy assignment
        world.id_to_slot[world.id[:n_keep]] = np.arange(n_keep, dtype=np.uint32)
        buffer.to_remove.clear()

    # ... insertions: append new ids and write id_to_slot[new_id] = slot ...
```

The `id_to_slot[ids[:n_keep]] = np.arange(n_keep)` line is the keystone. It rewrites every surviving id's slot in one bulk numpy assignment — exactly the same shape as the column compress, applied to the index map.

## Exercise 4 — Time the difference

```python
import time, numpy as np

world = build_world(n=1_000_000, hungry_count=100_000)
ids = np.random.default_rng(0).choice(1_000_000, size=100_000)

# Linear scan version (§17 ex 6)
def is_hungry_scan(hungry, target):
    return bool(np.any(hungry == target))

t = time.perf_counter()
for cid in ids:
    is_hungry_scan(world.hungry, int(cid))
print(f"linear scan × 100K: {time.perf_counter()-t:.2f} s")

# Indexed version
t = time.perf_counter()
for cid in ids:
    bool(world.hungry_member[int(cid)])
print(f"indexed × 100K: {time.perf_counter()-t:.3f} s")
```

Typical: linear scan ~5-10 minutes (10⁵ × 10⁵ = 10¹⁰ ops). Indexed: ~30 ms (one C-level read per call, plus Python loop overhead). Ratio: ~10⁵-10⁶×.

For a real simulator that does many membership queries per tick, the index map is the difference between *workable* and *unsalvageable*. Without it, presence-replaces-flags would only be defensible for whole-table operations, not individual queries.

## Exercise 5 — Run the exhibit (honestly)

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
result = mat @ v               # SpMV — what CSR is actually for
```

For SpMV at 1000×1000 with 1% density, CSR is dramatically faster than naive dense or dict-based approaches — nine thousand multiplications instead of a million. That's the operation it's optimised for.

The lesson: **pick the structure that matches your access pattern.** A dict is a sparse *point-lookup* map. CSR is a sparse *matrix*. They share the word "sparse" and almost nothing else.

## Exercise 6 — The bandwidth cost

```
1M id_to_slot entries × 4 bytes = 4 MB total
1500 cleanup writes per tick × 4 bytes = 6 KB written
At ~10 GB/s memory bandwidth: ~0.6 µs to write 6 KB
30 Hz tick budget: 33 ms
```

The cleanup map-update cost is **0.002% of the tick budget** at typical mutation rates. The id_to_slot maintenance is invisible against the rest of the work. The 4 MB total memory cost is the dominant concern at scale, not the bandwidth — which mitigates to 400 KB once recycling caps the high-water id count.

## Exercise 7 — Sort-for-locality compatibility

```python
def sort_for_locality(world, key_col_name: str):
    """Sort the table in-place by some key (e.g., spatial bucket).
       Updates id_to_slot to reflect the new positions."""
    key = getattr(world, key_col_name)[: world.n_active]
    order = np.argsort(key, kind="stable")

    for col_name in world.column_names:
        col = getattr(world, col_name)
        col[: world.n_active] = col[: world.n_active][order]

    # the keystone again — one bulk update
    world.id_to_slot[world.id[: world.n_active]] = np.arange(world.n_active,
                                                              dtype=np.uint32)
```

After the sort, `world.id[k]` is some new id, and `id_to_slot[world.id[k]] == k`. External code holding a reference to id `42` looks up `id_to_slot[42]`, gets the new slot, reads the (now-relocated) row.

The sort changed every slot. The map update changed every entry of `id_to_slot`. Both are O(N) bulk numpy operations — fast enough to do every tick if needed.

## Exercise 8 — A from-scratch generational arena (stretch)

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

Compare with [`slotmap::SlotMap`](https://docs.rs/slotmap/) (Rust): same machinery, different organisation. Rust packs `(index, generation)` into one `Key` (a `u64`); we use a `NamedTuple`. Rust uses `Vec<Slot>` with an internal free list; we use an active counter and bump generations on remove. The structural pieces — id allocator, generation array, id_to_slot map, swap_remove on delete — are identical.

Combined with [§22](22_mutations_buffer.md)'s deferred cleanup, this `SlotMap` is the simulator's table primitive. Once you have it, every variable-quantity table in the book reuses the shape — creatures, food, pending events, transition log entries — each one a `SlotMap` with different columns.
