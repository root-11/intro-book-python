# 21 — `swap_remove`

<p align="center"><img src="../covers/phase_memory_lifecycle.jpg" alt="Memory & lifecycle phase" style="max-height: 380px; max-width: 100%;"></p>

> *Concept node: see the [DAG](../../concepts/dag.md) and [glossary entry 21](../../concepts/glossary.md#21--swap_remove).*

The presence-replaces-flags substitution from [§17](17_presence_replaces_flags.md) raised a problem we deferred. When a creature stops being hungry, you remove its id from `hungry`. When a creature dies, you remove its row from every table. *Removing rows from the middle of an array is expensive* — every later row has to shift left by one, costing O(N).

For a 1,000,000-creature simulator with 1,000 deaths per tick, naive remove costs roughly 10⁹ moves per tick — far past the budget of any real-time loop.

Python gives you four options. Two are wrong, two are right — and the right two are right in different situations.

## Four options, ranked

```python
# anti-pattern: bad!
lst.pop(i)            # O(N) — shifts every subsequent element left
np.delete(arr, i)     # O(N) plus a fresh allocation — usually the slowest
```

```python
# disciplined: per-element swap_remove with an active counter
arr[i] = arr[n_active - 1]   # move last live element into the freed slot
n_active -= 1                # the "table" is now arr[:n_active]
```

```python
# disciplined and faster: bulk filter when you have K indices in hand
keep_mask = np.ones(n_active, dtype=bool)
keep_mask[indices_to_remove] = False
arr = arr[keep_mask]         # one C-level pass; survivors keep original order
```

The mechanism for the per-element version is small: take the last live element, move it into the deleted slot, decrement the active count by one. Two memory writes and a counter decrement. **O(1) regardless of N.** The "active counter" pattern means you allocate a numpy column once at the maximum size you need, and `n_active` tells you how many rows are currently in use. The table is the prefix `arr[:n_active]`. Removing a row never resizes the backing storage; only the counter changes. Inserting a row writes to `arr[n_active]` and increments. (Insertion details in [§24 — Append-only and recycling](24_append_only_and_recycling.md).)

The bulk-filter version takes a *batch* of indices and processes them in a single numpy call. It allocates a fresh column of size `n_active - K`, but pays the allocation only once for the whole batch instead of once per row. It is the natural pair to [§22 — Mutations buffer](22_mutations_buffer.md), which is exactly the pattern of "collect K removes during the tick; apply them all at once at the boundary." The batch is the unit of work; the single numpy call is the application.

**The SoA reminder from [§6](06_a_row_is_a_tuple.md) still applies.** Both the per-element swap_remove and the bulk filter are *single-column* operations as shown above — and a creature table is six or eight columns, not one. Removing creature `i` is `pos_x[i] = pos_x[-1]; pos_y[i] = pos_y[-1]; ...; n_active -= 1` across every column with the same `i`. The bulk-filter form is the same shape — *one* `keep_mask` computed once, applied to every column with the same indices, in lockstep. Apply it to half the columns and rows go out of alignment, exactly the bug from [§9](09_sort_breaks_indices.md). The discipline is the same as it was for sort: every operation that reorders any column reorders all columns of that table together.

## Cost, measured

From [`code/measurement/swap_remove.py`](../../code/measurement/swap_remove.py), removing 100,000 mid-table rows from a 1,000,000-row table on this machine:

| layout                                              | time      | remove rate         |
|-----------------------------------------------------|----------:|--------------------:|
| Python list, `list.pop(i)`                          |  3.456 s  |       28,938 ops/s  |
| numpy, `np.delete(arr, i)`                          | 21.880 s  |        4,570 ops/s  |
| numpy active counter, sequential swap_remove        |  0.016 s  |    5,511,389 ops/s  |
| **numpy bulk filter, `arr[keep_mask]`**             | **0.003 s** | **29,571,444 ops/s** |

Four readings.

**`np.delete` is the worst.** This will surprise readers who reach for it because it sounds like the "numpy way" to remove a row. It is not — `np.delete` returns a *new* array with the element removed, allocating fresh memory and copying the surviving elements every call. At 100,000 sequential deletes from a 1M-row array, you allocate 100,000 progressively-shrinking arrays. The bytes are typed, the operation is C-level, and it is still **7,151× slower than the bulk filter** because the algorithmic shape is wrong.

**`list.pop(i)` is the AoS middle ground**, but only because Python lists are pointer arrays — shifting an N-element list is N pointer copies, which is faster than shifting and reallocating an N-element typed numpy array. Either way: O(N) per remove, **1,129× slower than the bulk filter**.

**Sequential swap_remove processes 5.5 million removes per second.** Each remove is O(1), but the loop that drives it crosses the Python-numpy boundary 100,000 times — one bounds check, one assignment, one `n_active -= 1` per iteration. That overhead is the only thing keeping it from being the fastest line in the table.

**Bulk filter processes 29.5 million removes per second** — **5× faster than sequential swap_remove**. The boolean-mask pass and the compress are both single C-level operations over the whole array. The Python interpreter is touched once, not 100,000 times. This is the version the simulator's cleanup pass should use whenever it has a buffer of indices to remove.

Reading the table together: **per-element swap_remove is the right tool when you genuinely have one row to remove (rare). Bulk filter is the right tool when you have a buffer of K indices (the typical case once buffering is in place — §22).** Both forms beat the AoS reflexes by orders of magnitude. The choice between them is set by whether the buffering pattern from §22 has happened upstream.

## Cost paid

Order is sacrificed. If your code depended on rows being in any particular order, swap_remove reorders them. Two specific consequences:

- **Iteration corruption.** If you iterate the table and call swap_remove during iteration, the slot you just visited now holds a different row, but your loop counter has moved past it. Half the rows after a swap_remove get skipped or revisited inconsistently. (The same iterate-and-mutate footgun from §15.)
- **External references break.** Any code holding a slot index into the table now refers to a different row. This is the same bug as [§9](09_sort_breaks_indices.md): rearrangement breaks slot-based references.

Both problems have fixes already named in the book. The iteration corruption is fixed by [§22 — Mutations buffer](22_mutations_buffer.md): swap_remove never runs during iteration; it runs during cleanup at the tick boundary, when no system is iterating. The external-reference problem is fixed by [§23 — Index maps](23_index_maps.md): an `id_to_slot` map is updated whenever a row moves, so id-based references survive.

## When the lifecycle phase matters

This whole phase — Memory & lifecycle — only matters for *variable-quantity* tables. Constant-quantity tables like the 52-card deck never grow or shrink, never need swap_remove, never need any of the machinery in this phase. The card game ran for ten chapters without it. The simulator from §11 onward needs all of it, because creatures are born and die every tick.

The *constant vs variable* distinction is what determines whether a programmer reaches into the lifecycle toolbox at all. Once you have a table whose row count varies at runtime, every tool in this phase becomes load-bearing.

## Exercises

1. **Compare timings, simple case.** Build a `list` of length 1,000,000. Time 1,000 calls to `lst.pop(0)` (front delete, the worst case). Time the same with the swap_remove pattern (`lst[0] = lst[-1]; lst.pop()`). The ratio is roughly N.
2. **Mid-table delete.** Build a numpy `int64` array of length 1,000,000. Time 1,000 calls to `np.delete(arr, 500_000)` (rebinding `arr` each time). Time 1,000 calls to the swap_remove pattern (`arr[500_000] = arr[n_active - 1]; n_active -= 1`). The ratio is enormous — `np.delete` allocates a fresh array each call.
3. **Run the §21 exhibit.** `uv run code/measurement/swap_remove.py`. Note the order of the four rows. Confirm `np.delete` is the slowest, not the fastest, despite being the "numpy way." Note the gap between sequential swap_remove and bulk filter — both are O(K) algorithmically, but the bulk version pays the Python-loop overhead once instead of K times.
4. **The iteration hazard.** Build a numpy `int64` array of length 100 with values `0..100` and an `n_active = 100`. In a forward loop, iterate `i in range(n_active)` and apply swap_remove whenever `arr[i] % 2 == 0`. Compare with the expected output (only odd values remaining). What did you actually get? (Spoiler: you missed half the evens.)
5. **The fix in one shape: iterate backwards.** Repeat exercise 4, but iterate `range(n_active - 1, -1, -1)`. Does it work now? Why does it work?
6. **The fix in another shape: deferred cleanup.** Repeat exercise 4, but instead of calling swap_remove inside the loop, append the index to `to_remove`. After the loop, sort `to_remove` in reverse order and apply swap_remove. This is the [§22](22_mutations_buffer.md) pattern in miniature.
7. **Aligned per-element swap_remove.** Build the simulator's six creature columns (`pos_x, pos_y, vel_x, vel_y, energy, id`). Write `def delete_creature(world, slot)` that calls swap_remove on every column in lockstep. Verify all columns remain aligned after a sequence of deletes.
8. **Aligned bulk filter.** Take the same six creature columns. Write `def delete_batch(world, indices_to_remove)` that builds *one* `keep_mask` and applies it to every column. Verify alignment by spot-checking row 17 (the `(pos_x[17], pos_y[17], ..., id[17])` tuple) before and after the batch — its values should match the original row whose id is now at slot 17. Now write the *broken* version that applies the mask to only some columns; verify that row alignment is destroyed exactly as [§9](09_sort_breaks_indices.md) predicted. The single-column bulk filter shown in the prose is for clarity; the table version always reads the mask once and uses it everywhere.
9. *(stretch)* **The bandwidth cost.** Compute the bytes moved by `np.delete(arr, 0)` on a 1 GB int64 array: roughly the whole 1 GB (the source array, copied minus the deleted element). Compute the same for the swap_remove pattern: roughly 8 bytes (one `int64` move). The ratio is `N / 1`. Verify with `tracemalloc` or `psutil`.

Reference notes in [21_swap_remove_solutions.md](21_swap_remove_solutions.md).

## What's next

[§22 — Mutations buffer; cleanup is batched](22_mutations_buffer.md) is the rule that makes swap_remove safe to use: it never runs while any system is iterating.
