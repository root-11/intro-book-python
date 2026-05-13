# 23 — Index maps

> *Concept node: see the [DAG](../../concepts/dag.md) and [glossary entry 23](../../concepts/glossary.md#23--index-maps).*

<p align="center"><img src="../illustrations/linear_algebra.jpg" alt="Linear algebra: Ax = b — a lookup is a matrix-vector product" style="max-height: 300px; max-width: 100%;"></p>

The presence-replaces-flags substitution from [§17](17_presence_replaces_flags.md) had a sting in its tail. A presence query — "is creature 42 hungry?" — costs O(K) when implemented naively as `np.any(hungry == 42)`. At a 1,000,000-creature simulator with thousands of such queries per tick, that is too slow.

The fix is a parallel data structure: an *index map* that maps every id to its current slot in the table. Lookup is now O(1).

Python gives you two reasonable shapes for the map, and one trap.

## Two shapes that work

**A numpy array, when ids are dense.** If your ids are integers in `[0, N_max)` and most are in use, a single typed column does the job:

```python
INVALID = np.iinfo(np.uint32).max  # 4_294_967_295
id_to_slot = np.full(N_max, INVALID, dtype=np.uint32)

def slot_of(id_to_slot: np.ndarray, creature_id: int) -> int | None:
    slot = int(id_to_slot[creature_id])
    return None if slot == INVALID else slot
```

The sentinel value (`np.iinfo(np.uint32).max`) marks "no slot — this id has no current row". 4 MB at 1,000,000 ids; a single C-level memory read per lookup; bulk lookups via fancy indexing (`id_to_slot[ids_to_remove]`) run at numpy speed and are exactly what cleanup uses ([§22](22_mutations_buffer.md)). One cache line per 16 ids; cleanup streams through it sequentially.

**A `dict[int, int]`, when ids are sparse.** If the id space is large but few are in use — id is a hash of a string, an external system's UUID-as-int, a timestamp truncated to a slot — a Python dict is the right pick:

```python
id_to_slot: dict[int, int] = {}

def slot_of(id_to_slot: dict[int, int], creature_id: int) -> int | None:
    return id_to_slot.get(creature_id)
```

Dict lookup is O(1) amortised, ~30-40 million ops/sec for integer keys (per [`code/measurement/float_or_int_tuple.py`](https://github.com/root-11/intro-book-python/blob/main/code/measurement/float_or_int_tuple.py) — note that *which* integer matters; int keys are 2.4× faster than float-tuple keys at the same map size). Dict pays for hash machinery on every lookup and one pointer chase per access; numpy pays neither. But dict pays *only* for ids that actually exist, which is the right shape for a sparse id space.

The choice is set by id density, not by taste. The simulator's surrogate ids from [§10](10_stable_ids_and_generations.md) are dense — a fresh integer per creature, recycled when slots are reused. The numpy array is the right pick. An audit log indexed by 64-bit hash would be sparse — the dict is the right pick.

## One shape that is wrong

```python
# anti-pattern: bad!
from scipy.sparse import csr_matrix
m = csr_matrix(...)            # built for sparse 2D matrix arithmetic
slot = m[creature_id, 0]        # used here as a 1D point-lookup map
```

The `scipy.sparse` family — CSR, CSC, COO — are not index maps. They are sparse-matrix data structures, optimised for matrix-vector products and slicing entire rows or columns. Used for individual point lookups, they are very slow. From [`code/measurement/csr_matrix or python dict.py`](https://github.com/root-11/intro-book-python/blob/main/code/measurement/csr_matrix%20or%20python%20dict.py) at 1,000 × 1,000 with 1% density, **a Python dict is roughly 108× faster** than CSR at random scalar lookups.

The exhibit's headline reads "CSR matrix is 108× slower than Python dict." That is true *for the access pattern in the file* — and it is the wrong reading. The right reading is: **scipy gave you a sparse *matrix*, not a sparse *map*. Pick the structure that matches your access pattern.** CSR is excellent at SpMV (sparse-matrix-vector-product, the common dense-vector-multiplied-by-sparse-matrix operation in scientific computing). It is poor at point-and-shoot lookups because its internal layout — three `indices`, `indptr`, `data` arrays — is optimised for stride-skipping, not for O(1) random access. The lesson is not "CSR is slow"; it is "wrong tool for this job, every time, by design."

## Maintenance

The map must be updated whenever a row moves. The events that move rows in this book are exactly three:

- **Bulk filter cleanup** ([§22](22_mutations_buffer.md)). Every removed slot's id is set to `INVALID`. Every surviving id whose slot changed has its entry rewritten — exactly the rows that moved during the keep-mask compress.
- **Append.** When a new row lands at slot `n`, set `id_to_slot[new_row.id] = n`. The cleanup pass writes this in lockstep with the insert tail.
- **Sort or reshuffle** (for locality, [§28](28_sort_for_locality.md)). When the table is reordered, every slot moves. The full map is rewritten in lockstep with the sort. In numpy this is one assignment: `id_to_slot[ids[order]] = np.arange(n_active)`.

The cleanup system from [§22](22_mutations_buffer.md) is the natural home for these updates. Every removal and every insertion goes through cleanup; cleanup keeps the map in step.

## Cost

The numpy array adds one `uint32` per id ever issued, including ids that are currently dead but whose slots have not been recycled. For a simulator that issues a million ids over its lifetime but has 100,000 alive at any moment, the map is 4 MB. That is a real cost — bigger than the alive table itself if the table has narrow columns. Mitigations:

- **Generational ids** ([§10](10_stable_ids_and_generations.md)) plus a **separate id allocator** that recycles dead ids bound the map's size to the *high-water mark* of live ids, not the total ever issued. With recycling, the map stays at 100,000 × 4 = 400 KB.
- **A dict-of-int-to-int** trades a constant-factor lookup overhead for tighter memory; useful when ids are sparse, as named above.

For most simulators with recycling, the dense `np.ndarray` is the right shape. One cache line per 16 ids; the bulk lookup `id_to_slot[ids]` is bandwidth-bound at numpy speed.

## The pattern in the wild

Every ECS engine ships an index map. Bevy's `Entity` (Rust) is a 64-bit handle whose unpacking is essentially a slot lookup with a generation check. `slotmap`'s `SlotMap` keeps an internal map. Database engines maintain index maps as B-trees over primary keys. The shape — id-to-slot lookup, maintained on every move — is universal.

Combined with [§10](10_stable_ids_and_generations.md)'s stable ids and [§24](24_append_only_and_recycling.md)'s slot recycling, the index map is the third piece of the *generational arena* — the canonical handle-based data structure in modern systems software.

## Exercises

1. **Build the map.** Add `id_to_slot = np.full(N_max, INVALID, dtype=np.uint32)` to your simulator. When a creature is appended at slot N, set `id_to_slot[id] = N`. When a creature's slot changes during cleanup, update accordingly.
2. **O(1) presence query.** Add a parallel `hungry_membership = np.zeros(N_max, dtype=bool)` set to `True` when an id is in `hungry`. Now `is_hungry(id)` is two array lookups, both O(1).
3. **Maintain on bulk-filter cleanup.** Modify your [§22](22_mutations_buffer.md) cleanup to update `id_to_slot` after the keep_mask compress. The fastest form: after `id[: new_n] = id[: n_active][keep_mask]`, run `id_to_slot[id[:new_n]] = np.arange(new_n, dtype=np.uint32)` — one bulk write, every surviving id's slot rewritten in one pass.
4. **Time the difference.** Rerun the simulator at 1M creatures, calling `is_hungry(random_id)` 100,000 times per tick. Compare the linear-scan version (§17 exercise 6) and the indexed version. The ratio is roughly N — about a million.
5. **Run the exhibit (honestly).** `uv run "code/measurement/csr_matrix or python dict.py"`. Read the file's headline ("CSR matrix is 108× slower"). Then read the chapter's reframing. Confirm with one small experiment of your own that scipy's CSR is fast at *its* job — `csr.dot(some_dense_vector)` for a 1000×1000 matrix — and slow at the job the file gave it.
6. **The bandwidth cost.** At 1M ids, `id_to_slot` is 4 MB. Cleanup's bulk update on a tick with 1,000 swap_removes and 500 inserts writes ~1,500 entries — 6 KB. Compute the cleanup cost in microseconds for those writes against a 30 Hz budget.
7. **Sort-for-locality compatibility.** When `creatures` is sorted (a preview of [§28](28_sort_for_locality.md)), every slot moves. Rewrite `id_to_slot` in lockstep with one bulk numpy assignment: `id_to_slot[ids[order]] = np.arange(n_active)`. Verify external references (held as ids) are still correct after the sort.
8. *(stretch)* **A from-scratch generational arena.** Combine [§10](10_stable_ids_and_generations.md)'s `gens: np.ndarray`, [§22](22_mutations_buffer.md)'s deferred cleanup, and §23's `id_to_slot` map into a `SlotMap` class. Provide `insert(row) -> CreatureRef`, `remove(ref)`, `get(ref) -> int | None`. Compare the shape with [`slotmap::SlotMap`](https://docs.rs/slotmap/) (Rust) — same machinery, organised differently.

Reference notes in [23_index_maps_solutions.md](23_index_maps_solutions.md).

## What's next

[§24 — Append-only and recycling](24_append_only_and_recycling.md) names two strategies for what happens to a slot after it has been freed. The choice is decided by access pattern, not by taste.
