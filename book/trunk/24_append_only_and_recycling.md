# 24 — Append-only and recycling

> *Concept node: see the [DAG](../../concepts/dag.md) and [glossary entry 24](../../concepts/glossary.md#24--append-only-and-recycling).*

<p align="center"><img src="../illustrations/hard_hat_repeat.jpg" alt="MEASURE / CALCULATE / DESIGN / BUILD / REPEAT — recycling is the construction cycle" style="max-height: 300px; max-width: 100%;"></p>

When a row is removed from a table, its slot is freed. There are two strategies for what happens to that slot.

**Append-only.** Old slots stay valid forever. The table grows monotonically. New rows always go to the end.

**Recycling.** Freed slots are reused. The table's length stays bounded. New rows go into freed slots before the table grows.

Each is correct; they have very different access patterns and costs.

## When you have to think about slot reuse

A short Python aside before the strategies. Most Python code never thinks about slot reuse because the language hides it: `del obj` lets the garbage collector reclaim the memory, and the next `obj = something()` may or may not land in the same address — you do not know and do not care. The runtime decides.

Numpy columns are the opposite. You allocated `np.empty(N_max, dtype=...)` once, at startup. The slots are *positional*: slot 17 is the bytes at offset `17 * 4`. There is no GC to reclaim them; there is just `n_active` and a discipline about whether slot 17, once freed, gets reused or sits empty until the table is rebuilt. **The Python edition's lifecycle phase is exactly the work the runtime usually does for you, made explicit because numpy will not.**

## Append-only

Use append-only when:

- *History matters.* The simulator's `eaten`, `born`, `dead` logs from `code/sim/SPEC.md` are all append-only — they record what happened. Removed entries would be lost history.
- *Old references must remain valid forever.* Some slot-as-pointer designs assume the table never shrinks.
- *Total volume is bounded by elapsed time, not by population.* A 30-second 30 Hz simulation produces at most 900 frames; an append-only frame log is at most 900 rows. No need to recycle.

The cost is monotonic memory growth. A long-running simulator with append-only `eaten` accumulates millions of rows over hours. Mitigations:

1. Periodic snapshot + truncate (the log is replaced by a recent slice).
2. Tiered storage — recent in memory, older streamed to disk ([§30](30_streaming_wall.md)).
3. Just accept the memory, if the run is short.

## Recycling

Use recycling when:

- *Steady-state size is small even though total inserted is large.* The simulator's `creatures` table at 100,000 alive with 100,000 deaths and 100,000 births per second — net flow zero, but total ever issued grows linearly. Recycling keeps memory bounded.
- *Memory matters.* Recycling caps the table at the high-water mark of live rows.

The cost is reference-stability complications. A new row in a recycled slot has the same slot as a previous, removed row. Code holding an old slot reference would silently dereference the new row. The fix is generational ids ([§10](10_stable_ids_and_generations.md)): each slot has a generation counter that increments on every recycle. References hold `(id, gen)`; dereference checks the generation. A stale reference fails its check.

A slot allocator looks like:

```python
class SlotPool:
    """Allocates fixed-capacity slot indices, recycling freed ones.
       Generation increments on every free, so old (slot, gen) refs
       can detect they are stale."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.free_slots: list[int] = []          # stack of freed slots
        self.next_slot: int = 0                  # high-water mark
        self.gens = np.zeros(capacity, dtype=np.uint32)

    def allocate(self) -> tuple[int, int]:
        if self.free_slots:
            slot = self.free_slots.pop()         # reuse a freed one
        else:
            slot = self.next_slot                # grow
            self.next_slot += 1
            assert self.next_slot <= self.capacity, "pool exhausted"
        return slot, int(self.gens[slot])

    def free(self, slot: int) -> None:
        self.gens[slot] += 1                     # invalidate old refs
        self.free_slots.append(slot)
```

`allocate` pops a freed slot if any are available, otherwise grows. `free` bumps the generation and adds the slot back to the free list. Stale references (with the *old* generation) cannot dereference the recycled row.

The free list is a Python `list` used as a LIFO stack — `append` and `pop` are both O(1). The generation column is numpy because it is touched in lockstep with cleanup ([§22](22_mutations_buffer.md)) and benefits from bulk numpy ops when many slots are freed together.

## Choosing between them

Match the strategy to the table's role:

| table              | strategy    | reason                            |
|--------------------|-------------|-----------------------------------|
| `creatures`        | recycling   | bounded population                |
| `eaten`            | append-only | history record                    |
| `born`             | append-only | history record                    |
| `dead`             | append-only | history record                    |
| `pending_event`    | recycling   | rebuilt every tick                |
| `food`             | recycling   | bounded                           |
| `food_spawner`     | constant    | no removals                       |

Mixing strategies in one simulator is normal. The discipline is to be explicit about which table is which, and apply the right machinery to each.

## Exercises

1. **Two append-only logs.** Implement `eaten` and `born` as append-only numpy columns with their own `n_active` counters. After 1,000 ticks, examine the lengths and verify they grow monotonically.
2. **A recycling pool.** Implement the `SlotPool` above. Allocate 1,000 slots, free 500, allocate 500 more. Print the slot indices the second `allocate` batch returns. Did the pool reuse the freed slots, or grow?
3. **Stale reference detection.** Allocate a slot with `(slot, gen=0)`. Free it. Allocate a new row in the same slot — its gen is 1. Try to dereference the old `(slot, 0)` against the live `gens` column; confirm the check fails.
4. **Switch creatures to append-only.** Run the simulator with `creatures` as append-only (no recycling, every birth grows the table). Run for 10,000 ticks with steady birth and death. Plot `n_active` and `next_slot` over time — `n_active` is roughly flat (deaths balance births), `next_slot` grows monotonically. Memory cost: `next_slot * row_size`.
5. **Switch eaten to recycling.** Run with `eaten` recycled. After 100 ticks, all "what did this creature eat at tick 50" queries fail because the rows were reused. The history is gone. This is the failure mode that makes append-only the right pick for logs.
6. *(stretch)* **A capacity-aware allocator.** Modify `SlotPool.allocate` to return `None` when the pool is full instead of asserting. The simulator now has to handle "no slot available" as a real condition — what does it mean? (Hint: the world has hit its population cap; either rebuild bigger, drop the new entity, or delete the oldest one to make room.)

Reference notes in [24_append_only_and_recycling_solutions.md](24_append_only_and_recycling_solutions.md).

## What's next

[§25 — Ownership of tables](25_ownership_of_tables.md) is the rule that makes every other discipline in the phase work: each table has exactly one writer.
