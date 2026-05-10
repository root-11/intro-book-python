# 26 — Hot/cold splits

<p align="center"><img src="../covers/phase_scale.jpg" alt="Scale phase" style="max-height: 380px; max-width: 100%;"></p>

> *Concept node: see the [DAG](../../concepts/dag.md) and [glossary entry 26](../../concepts/glossary.md#26--hot-cold-splits).*

The simulator's `creature` table has six columns: `pos`, `vel`, `energy`, `birth_t`, `id`, `gen`. The motion system reads three of the six (`pos`, `vel`, `energy`). The starvation system reads only `energy`. The cleanup system reads `id` and `gen`. The births log reads `birth_t`. *No system reads all six.*

If the columns are stored together — same memory region, same prefetcher pulls — every load brings in fields the inner loop ignores. At cache-spilling sizes, the ignored fields cost real bandwidth.

The fix is a split: fields touched on the hot path go in one table; fields read rarely go in another. Two tables, same length, same id alignment.

## Why this lesson is gentler in Python+numpy

In a Rust struct-of-fields-per-creature layout, `pos`, `vel`, `energy`, `birth_t`, `id`, `gen` all sit adjacent in memory. When the motion system reads `pos`, the cache line it pulls *also contains* `birth_t` and `id` and `gen` — the prefetcher loads them whether you want them or not. The hot/cold split breaks this adjacency by moving the cold fields to a different memory region.

In Python with numpy SoA, the situation is already different. Each column is its own contiguous numpy array, allocated by its own `np.empty(...)` call. When the motion system reads `pos_x`, the cache line it pulls *contains only `pos_x` values*. It does not touch `birth_t`'s memory at all. **The columns are already physically separated.** The hot/cold split is, for SoA-in-numpy, largely *organisational* — a way of naming and grouping columns that share an access pattern — not a memory-layout optimisation.

The split *does* matter when the layout is something other than SoA-in-numpy:

- **AoS dataclass lists** (`list[Creature]` with attributes). Reading `c.pos_x` from each instance pulls the full `Creature` object into cache, including `c.birth_t` and `c.id`. Splitting into two parallel lists of dataclasses (a hot Creature and a cold Creature) saves cache bandwidth — but you have already paid the bigger cost of being AoS in the first place. From [§11](11_the_tick.md)'s `tick_budget.py`, the AoS form costs 28 ms per tick at 1M creatures vs 0.6 ms for SoA. The hot/cold split inside the AoS form might recover some of that gap; switching to numpy SoA recovers all of it.
- **Numpy structured arrays** — `np.dtype([('pos', np.float32, 2), ('vel', np.float32, 2), ('birth_t', np.float64), ...])`. This is AoS in numpy clothing — the bytes for one creature are adjacent. Reading `arr['pos']` strides through the buffer, skipping past `birth_t`'s bytes one row at a time. The strided access is faster than a Python loop but slower than a contiguous numpy column. Splitting helps; using non-structured columns helps more.

The SoA-in-numpy discipline this book has built since [§7](07_structure_of_arrays.md) means **most of the bandwidth win the hot/cold split offers in Rust, Python+numpy already gives you for free.** The chapter exists for two reasons that are still load-bearing.

## What the split still buys you

**1. Code-organisational clarity.** A reader of `motion(pos_x, pos_y, vel_x, vel_y, energy, dt)` should not also have to know where `birth_t` lives. Putting the hot columns under one `CreatureHot` namespace and the cold columns under `CreatureCold` makes the read-set/write-set declarations from [§13](13_system_as_function.md) shorter and the dependency graph from [§14](14_systems_compose_into_a_dag.md) sparser. The compiler does not enforce it; the discipline does.

**2. Cleanup amortisation.** Cleanup ([§22](22_mutations_buffer.md)) writes every column when slots move. Six columns means six bulk-filter operations per cleanup. Splitting into hot (4 columns) and cold (2 columns) does not reduce the *total* work, but it lets you skip the cold-table cleanup *between* creatures-affecting and creatures-not-affecting cleanup phases. If a tick has only food deaths (no creature deaths), the creature_cold cleanup runs at zero cost ([§20](20_empty_tables_are_free.md)) — the empty-tables-free property compounds with the split.

**3. Persistence and inspection.** A snapshot for replay [§37](37_log_is_world.md) needs every column. A live debug inspector might want only the cold metadata. Splitting lets the inspector read only what it needs and avoid loading the hot columns for an interactive query.

The cost of the split is the cost of an extra table: one more name, one more bookkeeping point in cleanup, one more place where alignment must be maintained. Two tables of the same length share an id allocator; updates that affect both must be applied in lockstep.

## When the split is wrong

- **Pure SoA-in-numpy with sub-millisecond inner loops.** If the existing layout already has every column as its own numpy array and the inner loops are bandwidth-bound at numpy speed, splitting will not measurably help. The bandwidth wasn't being wasted to begin with.
- **All-fields workloads.** A debug-inspect system that reads every field reads everything; the split adds organisational overhead without reducing access cost.
- **Tiny rows.** If the full row is already 16-24 bytes, the split's overhead exceeds its benefit.
- **Frequently rebalancing.** If which fields are "hot" changes from tick to tick, a fixed split becomes unhelpful. Hot/cold is a static decision, made once for a given target workload.

The decision rests on measurement. Profile the simulator at the target size; identify the inner loop's actual touched columns; split when the split changes a measurable number. The split is earned by data, not by aesthetics.

A useful test: name the split *before* writing it. *"I am moving `birth_t` into a cold table because no inner loop reads it"* is a sound design choice. *"I am moving `birth_t` into a cold table because that's how ECS engines do it"* is not.

## Exercises

These extend the simulator's `creature` table.

1. **Audit access patterns.** For each system in your simulator, list which columns it reads and which it writes. Columns read every tick are hot; the rest are cold.
2. **Build the split, organisationally.** Refactor `creature` into `creature_hot` (a class holding `pos_x, pos_y, vel_x, vel_y, energy`) and `creature_cold` (a class holding `birth_t, id, gen`). Both share the id allocator. Verify each row's fields stay aligned across the two classes.
3. **Time motion at 1M creatures.** Pre-split: time motion. Post-split: time motion. The two should be *near identical* if you started from numpy SoA. The split was organisational, not bandwidth-saving.
4. **Time motion in numpy structured-array form.** Build the same world using `arr = np.zeros(N, dtype=np.dtype([('pos_x', 'f4'), ('pos_y', 'f4'), ('vel_x', 'f4'), ('vel_y', 'f4'), ('energy', 'f4'), ('birth_t', 'f8'), ('id', 'u4'), ('gen', 'u4')]))`. Run motion as `arr['pos_x'] += arr['vel_x'] * dt`. Time it. Compare to the unsplit SoA version. The structured-array version is slower because it strides past every cold field on every read — *this* is the layout where the hot/cold split would actually help.
5. **Cleanup must touch both.** Modify cleanup to apply the keep_mask ([§22](22_mutations_buffer.md)) to both `creature_hot` and `creature_cold` columns when a creature dies. Verify alignment after.
6. **A bad split.** Construct a split where the wrong fields go cold (e.g. `energy` in cold). Time motion. The cost of the cache-trip on `energy` per tick should bury any savings elsewhere.
7. *(stretch)* **The all-fields case.** Write a system that reads every field (e.g. a serialiser). Time the split version. Discuss why the split's overhead is real here, and why this is a fine tradeoff: most ticks do not run this system.

Reference notes in [26_hot_cold_splits_solutions.md](26_hot_cold_splits_solutions.md).

## What's next

[§27 — Working set vs cache](27_working_set_vs_cache.md) puts numbers on the question this section was implicitly asking: how big *is* the inner loop's footprint, and what cache level does it fit in?
