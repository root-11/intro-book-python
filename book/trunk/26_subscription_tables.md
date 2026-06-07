# 26 - Subscription tables, keyed by slot

<p align="center"><img src="../covers/phase_scale.jpg" alt="Scale phase" style="max-height: 380px; max-width: 100%;"></p>

> *Concept node: see the [DAG](../../concepts/dag.md) and [glossary entry 26](../../concepts/glossary.md#26--subscription-tables-keyed-by-slot).*

A system rarely touches every entity. Motion moves all of them, but starvation only reads the hungry, reproduction only the well-fed, a sleep timer only the sleeping. [§17](17_presence_replaces_flags.md) gave the tool for "which entities are in this set": a membership table. [§19](19_ebp_dispatch.md) measured the payoff: walk the 100,000 hungry instead of scanning 1,000,000 and masking, and the work is proportional to the subset, not the population.

Call that membership table a *subscription table*, and the loop that walks it a system's *hot loop*. A creature *subscribes* to `hungry` when its energy drops; it *unsubscribes* when it eats. The subscription is the system's input; the hot loop is the system. This section settles a question [§17](17_presence_replaces_flags.md) left open: what does a subscription table store, and how fast is the hot loop that reads it?

## A wrong turn first: splitting fields

The instinct many readers arrive with - and the one this chapter used to teach - is to split the *fields* of a creature into hot and cold: put the columns an inner loop touches in one group, the rarely-read ones in another, so a load does not drag in bytes the loop ignores. In a row-oriented (array-of-objects) world this is a real technique: reading `c.pos_x` pulls the whole `Creature` object into cache, `c.birth_t` and `c.id` and all.

But this book has been structure-of-arrays since [§7](07_structure_of_arrays.md), and in numpy SoA the split is already done for you, more completely than a hot/cold grouping would manage. Every column is its own `np.ndarray`, its own allocation. Reading `pos_x` touches `pos_x`'s memory and nothing else; `birth_t` is a different array entirely. There is no row to drag a cold field along with, so there is nothing to split. The bandwidth win that motivates a field split in array-of-objects is the same win SoA already banked back in [§7](07_structure_of_arrays.md); it is not a separate technique to apply here.

So the attribute columns are never split. They stay whole: every column, every slot, reachable by index `i`. What a system changes is not the columns but how it *reaches into* them. Rather than scan the whole column and mask, a system keeps a subscription table - the slots `i` it cares about - and indexes straight in: `energy[hungry]`, no scan, no field split. The rest of this section is about making that direct access fast.

## What a subscription stores: slots or ids

A creature has a stable [id](10_stable_ids_and_generations.md) and a current [slot](23_index_maps.md), its position in the columns. `id_to_slot` maps one to the other. A subscription table could hold either, and in numpy the choice is a measurable cost, not a style preference.

- Hold *slots*. The hot loop is one fancy-index gather: `energy[hungry]`. No redirection. But when [cleanup](24_append_only_and_recycling.md) moves entities, every slot in every subscription is stale and must be rewritten.
- Hold *ids*. The hot loop is two gathers: `energy[id_to_slot[hungry]]` - resolve each id to its slot, then gather the columns. The table survives relocation untouched; only `id_to_slot` changes.

The redirection is paid every tick. The rewrite is paid once per cleanup interval. Which loses?

**Measured.** At 1,000,000 creatures with a tenth subscribed, the id-keyed hot loop runs about twice as slow as the slot-keyed one on a modern desktop<sup>1</sup>. The extra cost is the inner gather: `id_to_slot` is a four-megabyte array and the subscribed ids are scattered through it, so `id_to_slot[hungry]` is a hundred-thousand-element random read before the column gather has even started. The slot key skips that gather entirely - `energy[hungry]` indexes the columns once. (In a compiled language the same gap is a scattered cache miss per element; numpy turns it into a second fancy-index pass, but the verdict is the same.)

The rewrite the slot key pays in return is small and bounded: when cleanup compacts, remap each subscription's `dense` array through the old→new slot map ([§23](23_index_maps.md)), once per cleanup interval, not once per tick. Across the realistic range - a handful of subscriptions, cleanup every few dozen ticks - the per-tick saving buries the per-interval rewrite. The benchmark is [`ebp_partition.py`](https://github.com/root-11/intro-book-python/blob/main/code/measurement/ebp_partition.py); numbers below.

So **subscription tables hold slots.** This is also *why* the lifecycle keeps [stable slots](24_append_only_and_recycling.md) and lets cleanup own the reindex: slot keys are only safe when one system is responsible for rewriting them when entities move. The cleanup can do that for any reference it owns - a subscription, or a cross-entity link stored in a column - remapping them all in one pass.

## So what is the id for?

For every reference cleanup *cannot* reach to rewrite. A save file ([§36](36_persistence_is_serialization.md)), a replay log ([§37](37_log_is_world.md)) whose events are `(tick, id, name)` - exactly the [§18](18_add_remove_insert_delete.md) boundary - a packet on the wire, an entity the UI has selected, a snapshot a slow background system is still reading ([§39](39_system_of_systems.md)): a slot is meaningless to all of them, because the next compaction moves it. They hold the id and resolve it through `id_to_slot` once, at the boundary ([§35](35_boundary_is_the_queue.md)), never per element. Slots are an internal, momentary fact; the id (with its generation, [§10](10_stable_ids_and_generations.md)) is the identity that survives a relocation, a save, and a network hop.

## Locality: a slot-keyed gather is fast only when its slots are dense

A slot-keyed hot loop gathers columns at the slots the subscription lists. If those slots are scattered through the column - which is what churn produces as deaths and births leave holes - `energy[hungry]` is a random-access gather. If they are contiguous, it streams. Compacting the live, subscribed entities to the front of the columns turns the scattered gather into a sequential one.

How much that buys depends on the machine, and the spread is wide. In numpy the gather's per-element machinery - bounds-check, dereference, copy into the output - is large, and on a fast desktop it dominates the cache-miss difference, so compaction buys only about 1.3x on the hot loop<sup>2</sup>. On hardware where memory latency dominates that machinery instead - the Pi 4 - the win is several-fold (4.9x; see Measurements). The reorder pass is about a millisecond at a million rows on the desktop, so on hot-loop locality *alone* it pays back in tens of ticks, on the order of a second at 30 Hz<sup>3</sup>; on the memory-bound machines the per-tick saving is larger and the payback shorter.

That payback sounds slow until you notice you were going to pay it anyway. The same pass reclaims dead slots ([§24](24_append_only_and_recycling.md)), which the simulator must do regardless as births and deaths leave holes, and the locality gain rides along on a move you already owed. So the rule for Python is: **compact on the GC cadence; the locality it also delivers ranges from a modest desktop bonus to a real win on memory-bound hardware.** [§28 - Proximity is a property of position](28_proximity.md) is that pass.

## The one case a split would help, in full view

There is a single scenario where grouping columns would still pay. A hot loop that gathers several columns at *scattered* slots issues one fancy-index per column, each its own random walk; interleaving those columns into one structured array would gather once. That case is real, and worth stating plainly rather than hiding behind the principle.

We keep the columns separate anyway. The book's answer to scatter is to remove it: compaction ([§28](28_proximity.md)) makes the subscribed slots dense, a dense gather streams each column, and the per-column cost the structured array would have saved is gone. A structured array would also forfeit what SoA bought in [§7](07_structure_of_arrays.md) - whole-column vectorised ops, and the per-column single-writer parallelism of [§31](31_disjoint_writes_parallelize.md) - on every loop that is *not* scattered, to win the one that is. So the rule stands with its exception in the open: keep the columns separate, and compact when the gather scatters.

## Name the subscription before you build it

A subscription is earned by a system that genuinely processes a subset. "Most creatures are not hungry on most ticks, so `hungry` is far smaller than the population" is a sound reason to build one. "Every creature is always in `alive`, but other engines keep an alive-set" is not. A subscription that holds the whole population is a scan-all with extra bookkeeping, and the measurement says so: at full participation the gather is *slower* than a plain vectorised pass over the column - it is the same crossover [§19](19_ebp_dispatch.md) measured, where presence loses to the bool mask once nearly everyone is a member. The subscription wins in proportion to how much it excludes, and not otherwise.

## Measurements

The prose quotes the modern-desktop figure; the spread across the reference machines is below. The keying verdict (row 1, slot beats id) holds on every machine. The locality win (row 2) varies widely - modest on the desktop, several-fold on the Pi - because it depends on how much memory latency dominates numpy's fixed gather overhead. Reproduce any column by running `ebp_partition.py` on that machine.

| # | measurement | Ryzen 9 (modern) | i7-3610QM (2012) | i3-5010U (2015) | Pi 4 |
|---|---|---|---|---|---|
| 1 | id-keyed ÷ slot-keyed hot loop, 1M @ 10% | 2.08x | 3.45x | 2.11x | 1.97x |
| 2 | scattered ÷ compacted gather, 1M @ 10% | 1.31x | 1.62x | 2.79x | 4.86x |
| 3 | compaction payback, hot-loop locality alone | ~30 ticks | ~20 ticks | ~12 ticks | ~7 ticks |

## Exercises

These extend the simulator's `creature` columns and the `id_to_slot` map from [§23](23_index_maps.md).

1. **Build a slot-keyed subscription.** Add `hungry = np.empty(0, dtype=np.uint32)` holding the *slots* of hungry creatures. Build it each tick with `np.flatnonzero(energy < HUNGER_THRESHOLD)`. Write the hot loop: `energy[hungry] -= burn * dt`. Verify it touches only the subscribed creatures.
2. **Key it by id instead, and time both.** Build a second version where `hungry` holds entity ids and the hot loop resolves each through `id_to_slot[hungry]` before the gather. At 1M creatures with 10% subscribed, time both hot loops with `timeit`. Reproduce the ~2x gap. Where does the id version's time go? Compare `id_to_slot`'s size with one cache line.
3. **Unsubscribe in O(1).** When a creature stops being hungry, remove its slot from `hungry`. What do you need alongside `hungry` to find the slot's *position in the table* without scanning it? (It is the [§23](23_index_maps.md) sparse set, one level up.)
4. **Reindex on compaction.** Relocate the live creatures to the front of the columns (a stand-in for the [§24](24_append_only_and_recycling.md)/[§28](28_proximity.md) cleanup), producing an `old_to_new` slot map. Rewrite the slot-keyed `hungry` with `hungry = old_to_new[hungry]`; confirm the hot loop still processes the same creatures. Now do the same for the id-keyed version: what has to change? Time both reindex passes.
5. **Dense vs scattered.** Time the slot-keyed hot loop with the subscription's slots scattered through the column, then again after sorting them to the front. Reproduce the ~1.3x speedup. How many ticks of hot-loop saving pay back one compaction pass? (Answer near 30 - and note this is *not* the reason to compact; reclaiming dead slots is.)
6. **The subscription that holds everyone.** Subscribe every creature and time the hot loop against a plain `energy -= burn * dt * mask` scan. The subscription should be no faster, and at full participation slower. Explain why, and state the rule for when a subscription is worth building.
7. *(stretch)* **Two subscriptions, one entity.** Put creatures in both `hungry` and `sleepy`. On compaction, both `dense` arrays need remapping through `old_to_new`. Measure how the reindex cost grows with the number of subscriptions an entity sits in, and argue why it stays cheaper than the id key's per-tick redirection for any realistic cleanup interval.

Reference notes in [26_subscription_tables_solutions.md](26_subscription_tables_solutions.md).

## What's next

[§27 - Working set vs cache](27_working_set_vs_cache.md) puts numbers on the question this section kept leaning on: how big *is* the hot loop's footprint, and what cache level does it fit in?
