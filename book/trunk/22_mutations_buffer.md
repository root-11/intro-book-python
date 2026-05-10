# 22 — Mutations buffer; cleanup is batched

> *Concept node: see the [DAG](../../concepts/dag.md) and [glossary entry 22](../../concepts/glossary.md#22--mutations-buffer-cleanup-is-batched).*

<p align="center"><img src="../illustrations/engineer_fuel.jpg" alt="Engineer-fuel coffee, mouse soldering — work buffered on the bench, applied in a batch" style="max-height: 300px; max-width: 100%;"></p>

This rule has been forward-referenced through ten chapters. Time to make it concrete.

Mutations during a tick do not apply immediately; they queue, and a single cleanup pass applies them all at the tick boundary. The shape:

```python
@dataclass
class CleanupBuffer:
    to_remove: list[int]               # creature ids to delete this tick
    to_insert_pos_x: list[float]       # parallel arrays of inserted-row data
    to_insert_pos_y: list[float]
    to_insert_vel_x: list[float]
    to_insert_vel_y: list[float]
    to_insert_energy: list[float]
    to_insert_id: list[int]
```

(The insert side has one list per column. Per [§6](06_a_row_is_a_tuple.md), a row is a tuple-at-index, and that's true of the insert buffer too — it is an SoA buffer, not a list of `Creature` objects. The reason is the same reason the rest of the simulator is SoA: numpy gets to work on the bytes when cleanup runs.)

During the tick, every system that wants to delete appends an id to `to_remove`. Every system that wants to add appends one row's worth of data to the parallel insert columns. **No system mutates the live tables.**

## The cleanup pass

At the end of the tick, one system runs:

```python
def cleanup(world: World, buffer: CleanupBuffer) -> None:
    # 1. Removals: build one keep_mask, apply to every column at once.
    if buffer.to_remove:
        ids_to_remove = np.unique(np.array(buffer.to_remove, dtype=np.uint32))
        slots = world.id_to_slot[ids_to_remove]              # see §23
        keep_mask = np.ones(world.n_active, dtype=bool)
        keep_mask[slots] = False
        for col_name in world.column_names:
            col = getattr(world, col_name)
            col[: keep_mask.sum()] = col[: world.n_active][keep_mask]
        world.n_active = int(keep_mask.sum())
        # (Update id_to_slot — covered in §23.)
        buffer.to_remove.clear()

    # 2. Insertions: bulk concatenate parallel insert columns into the table.
    n_inserts = len(buffer.to_insert_id)
    if n_inserts:
        new_n = world.n_active + n_inserts
        # The columns were sized at maximum capacity at startup; we are
        # writing into the previously unused tail [n_active : new_n).
        world.pos_x[world.n_active : new_n] = buffer.to_insert_pos_x
        world.pos_y[world.n_active : new_n] = buffer.to_insert_pos_y
        world.vel_x[world.n_active : new_n] = buffer.to_insert_vel_x
        world.vel_y[world.n_active : new_n] = buffer.to_insert_vel_y
        world.energy[world.n_active : new_n] = buffer.to_insert_energy
        world.id[world.n_active : new_n] = buffer.to_insert_id
        world.n_active = new_n
        # (Append the new ids to id_to_slot — §23.)
        for lst in (buffer.to_insert_pos_x, buffer.to_insert_pos_y,
                    buffer.to_insert_vel_x, buffer.to_insert_vel_y,
                    buffer.to_insert_energy, buffer.to_insert_id):
            lst.clear()
```

Two passes, both *bulk* operations. The world is in a fully consistent state at the end. The keep_mask is built once and applied to every column; the insert tail is filled with one slice assignment per column. Per [§21](21_swap_remove.md), the bulk-filter form is **5× faster than per-element swap_remove** at K=100,000 mutations per tick — and per the [editions-diverge framing in the prose of §10](10_stable_ids_and_generations.md) and elsewhere, this is where the Python edition's cleanup actually diverges from the Rust edition's: Rust §22 uses a per-element swap_remove loop because compiled code pays no interpreter-boundary tax; Python §22 uses the bulk-mask form because we measured the boundary cost and it dominates at scale.

## What this fixes

The iteration-corruption problem from [§21](21_swap_remove.md) goes away because the table is never mutated while any system is iterating. By the time cleanup runs, every system has finished. There is no concurrent iteration to confuse. The list-during-iteration and dict-during-iteration footguns from [§15](15_state_changes_between_ticks.md) cannot happen — there is no `creatures.remove(c)` inside a `for c in creatures` loop, because nothing inside the tick mutates the live tables.

The race-condition problem from concurrent mutation goes away. Two systems may both want to remove a creature; both append to `to_remove`; cleanup deduplicates with `np.unique`. Neither system needs to coordinate.

The composition problem from [§14](14_systems_compose_into_a_dag.md) goes away. Systems read consistent snapshots; they read the world *as it was at tick start*, not the world *as some other system half-rewrote it*.

## What it costs

Every mutation is one extra entry pushed to a side list. For a simulator with 1,000 deaths and 500 reproductions per tick, that is 1,500 entries of bookkeeping per tick — a few thousand bytes, completely negligible against the cost of running the systems themselves.

The cleanup pass is one additional system in the DAG. It is empty (no work) when no mutations are queued ([§20](20_empty_tables_are_free.md)); it runs the bulk filter and bulk concatenate when there are. The system is wired in once and never removed.

## What it does not fix

**Dedup is the system's job.** Two systems may both push the *same* id to `to_remove` if they independently detect the same death condition. The cleanup uses `np.unique(to_remove)` to reduce to distinct ids before computing slots. The cost is one O(K log K) sort on a small array — irrelevant against the bulk filter.

**Order matters.** Inside cleanup, deletions run first, then insertions. If you insert first, an inserted row might land in a slot you are about to delete. Deleting first frees up tail capacity that subsequent inserts can reuse — though slot recycling is its own decision ([§24](24_append_only_and_recycling.md)).

The pattern itself is universal. Database transactions buffer writes and commit at the boundary. Graphics pipelines render to a back buffer and swap. Version-controlled file systems collect changes and commit. They all solve the same problem: how do you let many independent operations modify shared state without stepping on each other? The answer is always the same — accumulate, then apply atomically.

## Exercises

1. **Implement the side buffers.** Add `to_remove: list[int]` and the parallel insert lists (one per column) to your simulator's world. They are empty at the start of every tick.
2. **Push from `apply_starve`.** Modify your starvation system to append to `to_remove` instead of any direct table mutation. Verify the system no longer touches the live `creatures` columns.
3. **Push from `apply_reproduce`.** Modify reproduction to append the parent's offspring rows to the parallel insert lists. Verify reproduction no longer mutates `creatures` directly.
4. **Implement bulk cleanup.** Write the cleanup system as in the prose. Apply removals first (one keep_mask, applied to every column), then insertions (one slice-write per column). Run a tick with both kinds of mutations; verify the world is consistent after.
5. **Compare cleanup forms.** Implement a *second* cleanup that uses per-element swap_remove in a Python loop instead of the bulk mask. Time both at 1,000,000 creatures with 1,000 mutations per tick. The bulk form should win by ~5× per the §21 numbers — confirm on your machine.
6. **The dedup question.** Push id 42 to `to_remove` from two different systems in the same tick. Run cleanup *without* the `np.unique` step. What happens? (Hint: `id_to_slot[42]` is looked up twice; the second lookup may produce garbage if the first removal moved another row to that slot.) Now add the `np.unique` and re-run. The result is correct.
7. **Tick-delayed visibility.** A creature inserted in tick 5 (via the `to_insert_*` lists) does not appear in the live columns during tick 5's systems — only at the end, in cleanup. Verify by adding an `age_in_ticks` column that increments at the end of each tick; the new creature's value starts at 0 in tick 6, not tick 5.
8. *(stretch)* **A graphics pipeline analogy.** A rendering pipeline draws to a "back buffer" while the "front buffer" is being displayed. At the boundary of one frame to the next, the buffers swap. Argue why this is the same pattern as `to_remove` / `to_insert` plus `cleanup`. (Hint: it is the same atomic-commit shape; the back buffer is exactly the side table.)

Reference notes in [22_mutations_buffer_solutions.md](22_mutations_buffer_solutions.md).

## What's next

[§23 — Index maps](23_index_maps.md) is the missing piece for swap_remove and bulk-filter cleanup to be useful: a parallel data structure that tracks where every id currently lives, updated whenever the columns move.
