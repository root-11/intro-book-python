# Solutions: 24 — Append-only and recycling

## Exercise 1 — Two append-only logs

```python
import numpy as np

class AppendLog:
    def __init__(self, capacity: int, dtype):
        self.tick      = np.zeros(capacity, dtype=np.uint32)
        self.creature  = np.zeros(capacity, dtype=np.uint32)
        self.value     = np.zeros(capacity, dtype=dtype)
        self.n_active  = 0
        self.capacity  = capacity

    def append(self, tick: int, creature_id: int, value):
        if self.n_active >= self.capacity:
            raise MemoryError("log full — snapshot and truncate")
        self.tick[self.n_active]     = tick
        self.creature[self.n_active] = creature_id
        self.value[self.n_active]    = value
        self.n_active += 1

eaten = AppendLog(capacity=1_000_000, dtype=np.float32)
born  = AppendLog(capacity=1_000_000, dtype=np.uint32)

# After 1000 ticks of the simulator
print(f"eaten: {eaten.n_active} entries (monotonic — never shrinks)")
print(f"born:  {born.n_active} entries (monotonic)")
```

Both `n_active` counters only ever increment. Once entries are written, they stay. Capacity is the high-water-mark of *total events ever recorded*, not of *current population*.

## Exercise 2 — A recycling pool

```python
class SlotPool:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.free_slots: list[int] = []
        self.next_slot: int = 0
        self.gens = np.zeros(capacity, dtype=np.uint32)

    def allocate(self) -> tuple[int, int]:
        if self.free_slots:
            slot = self.free_slots.pop()
        else:
            slot = self.next_slot
            self.next_slot += 1
        return slot, int(self.gens[slot])

    def free(self, slot: int):
        self.gens[slot] += 1
        self.free_slots.append(slot)


pool = SlotPool(capacity=10_000)
first_batch = [pool.allocate()[0] for _ in range(1_000)]
print(f"first 1000 slots: {first_batch[:5]}...{first_batch[-3:]}")

# Free 500
for slot in first_batch[:500]:
    pool.free(slot)

# Allocate 500 more
second_batch = [pool.allocate()[0] for _ in range(500)]
print(f"second 500 slots: {second_batch[:5]}...{second_batch[-3:]}")
print(f"all reused?      {set(second_batch).issubset(set(first_batch[:500]))}")
print(f"next_slot now:   {pool.next_slot}")  # still 1000 — no growth
```

```
first 1000 slots: [0, 1, 2, 3, 4]...[997, 998, 999]
second 500 slots: [499, 498, 497]...[2, 1, 0]
all reused?      True
next_slot now:   1000
```

The second batch reuses the freed slots in LIFO order (the most recently freed slot is allocated next). Total `next_slot` stays at 1000 — the pool did not grow.

## Exercise 3 — Stale reference detection

```python
pool = SlotPool(capacity=100)
slot, gen = pool.allocate()                           # slot=0, gen=0
old_ref = (slot, gen)                                  # save
pool.free(slot)                                        # gens[0] = 1
new_slot, new_gen = pool.allocate()                    # reuses slot 0, gen=1
new_ref = (new_slot, new_gen)

def deref(pool, ref):
    slot, gen = ref
    return None if int(pool.gens[slot]) != gen else slot

print(f"new ref deref:  {deref(pool, new_ref)}")      # 0
print(f"old ref deref:  {deref(pool, old_ref)}")      # None — stale!
```

The old reference's generation is stale. Even though the *slot* is alive again, the generation check correctly identifies that the holder of `old_ref` is looking at a *different* row than they expect. The reference is rejected; the holder must re-fetch.

## Exercise 4 — Switch creatures to append-only

```python
# After 10,000 ticks, steady-state birth/death:
# n_active ≈ 100,000 (live population, oscillates around equilibrium)
# next_slot = total ever issued = (births_per_tick × 10,000)
#           ≈ 100 × 10,000 = 1,000,000 (10× the live population)

# Memory cost: next_slot × row_size = 1M × ~32 bytes = 32 MB
# Live data:    n_active × row_size = 100K × 32 bytes = 3.2 MB
# Wasted:       28.8 MB sitting in dead slots
```

The append-only `creatures` table has 90% of its memory occupied by *tombstones* — slots whose previous occupants are dead. Reading `n_active` is correct, but the table's allocated bytes grow with elapsed time. For a 1-hour simulation, the wasted memory might be 100× the live data.

For history tables this is fine (the tombstones are the history). For the live population it's a memory leak waiting to be named. Recycling is the structural fix.

## Exercise 5 — Switch `eaten` to recycling

```python
# eaten is now a SlotPool-managed table with capacity 10_000
# After 100 ticks at 50 eats/tick, 5000 events recorded into 10_000 slots
# After 200 ticks: free list starts being used; old eat events are overwritten
# After 300 ticks: ~10,000 events have been recycled into existing slots

# Query: "what did creature 42 eat at tick 50?"
# Search eaten.tick[:n] == 50 → finds it (tick 100)
# Search after tick 250: finds nothing (the row was recycled at ~tick 250)
```

The history is *gone* once a slot is recycled. There is no record that creature 42 ate at tick 50 — the slot now holds tick 273's eat event for creature 91. Recycling for a history table is *category error*.

This is exactly the failure mode that makes append-only correct for logs. Logs grow forever; you handle that with snapshot-and-truncate or tiered storage, not by recycling slots.

## Exercise 6 — A capacity-aware allocator (stretch)

```python
class CapacityAwarePool:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.free_slots: list[int] = []
        self.next_slot: int = 0
        self.gens = np.zeros(capacity, dtype=np.uint32)

    def allocate(self) -> tuple[int, int] | None:
        if self.free_slots:
            slot = self.free_slots.pop()
        elif self.next_slot < self.capacity:
            slot = self.next_slot
            self.next_slot += 1
        else:
            return None                              # full!
        return slot, int(self.gens[slot])
```

Returning `None` from `allocate` is the simulator's signal that the world has hit its population cap. Three reasonable policies:

1. **Drop the new entity.** "Sorry, no room." A reproduction event silently fails. Simplest, hides the resource limit, may distort the simulation's behaviour.
2. **Delete the oldest one.** A LRU-style eviction. The pool needs an oldest-tracking scheme (a tick column, a queue). The simulation continues at capacity but loses identifiers; not appropriate for *logs* but reasonable for *active scenes*.
3. **Resize the pool.** Allocate a larger backing buffer, copy, retry. Most flexible, but introduces a slow-path that may blow a tick budget; consider doubling the pool every time it fills, like `Vec::push`'s amortised growth.

The book's simulator picks whichever fits the table:

- `creatures`: option 3 with periodic doubling. The scenario should never hit the cap in practice, but if it does, the simulation continues.
- `pending_event`: option 1. Events that don't fit the cap are dropped; the simulation makes do with what it has.
- `eaten`/`born` (append-only): option 1, but with snapshot+truncate as the recovery, not silent drop.

The choice is *per-table*. Document it next to the table's allocation.
