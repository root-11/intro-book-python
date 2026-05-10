# 17 — Presence replaces flags

<p align="center"><img src="../covers/phase_ebp.jpg" alt="Existence-based processing phase" style="max-height: 380px; max-width: 100%;"></p>

> *Concept node: see the [DAG](../../concepts/dag.md) and [glossary entry 17](../../concepts/glossary.md#17--presence-replaces-flags).*

A creature can be hungry. Three ways to model it.

The instinct most Python programmers arrive with is a *boolean field on the object*: `is_hungry: bool` on every `Creature`, set to `True` when energy drops below a threshold, set to `False` when energy is restored. Every system that cares about hunger checks the flag: `for c in creatures: if c.is_hungry: ...`. This is everywhere; it is the natural choice; it is what most Python tutorials reach for. It is also the worst of the three options — it is both AoS (per-creature object) and flag-shaped (one bit per creature regardless of state), and it forces every consumer to scan all N creatures to find the K hungry ones.

The middle option — better than per-instance booleans, still not the disciplined choice — is a *boolean column*. `is_hungry = np.zeros(N, dtype=bool)`, indexed in lockstep with the rest of the creature table. This is what most readers will reach for after Part 2. It pays one byte per creature (numpy's bool is one byte, not one bit), but the bytes are contiguous; numpy vectorises the scan; SIMD reads forty creatures per cache line. Compared to the per-object form, it is one-to-two orders of magnitude faster. Compared to the disciplined form below, it still costs N bytes regardless of how few creatures are hungry.

The data-oriented alternative is *membership*. There is a `hungry` table — a `np.ndarray` of creature ids, of length K (the number of currently-hungry creatures), no longer than it has to be. A creature is hungry if and only if its id is in `hungry`. The flag does not exist as a field; it exists as a *fact about which table the creature appears in*.

```python
# Three representations of "is this creature hungry?"
is_hungry_attr     = creatures[i].is_hungry          # AoS bool field
is_hungry_mask     = bool(is_hungry[i])              # SoA bool column, O(N) bytes
is_hungry_presence = np.isin(creature_ids[i], hungry) # presence table, O(K) bytes
```

The substitution looks small: a `bool` field becomes a row in another table. The implications are not.

## Four shifts that follow

**Dispatch** changes shape. The flag version is a per-creature filter inside every consuming system — walk all creatures, check the flag, do work if true. The membership version skips the filter — walk `hungry`, do work for every entry. At 1,000,000 creatures with 100,000 hungry, the flag version processes 1,000,000 rows; the membership version processes 100,000 — a 10× difference in work, and a 10× difference in memory bandwidth. [§19](19_ebp_dispatch.md) names this.

**Storage** changes shape. A `np.bool_` column stores one byte per creature whether the flag is set or not. A creature with eight possible states needs eight bool columns = 8 bytes per creature; a million creatures store 8 MB of flags, most of which are `False`. Eight presence tables store only the entries that *are* set — if 10% of creatures are hungry, the `hungry` table is 10% the size of the flag column.

**Persistence** changes shape. Serialising a flag column writes the flag for every creature, including the ones where it is `False`. Serialising a presence table writes only the entries that exist. The latter is also closer to the natural shape of an event log ([§37](37_log_is_world.md)): a `hungry_added` event per entry, and that is the whole story.

**Concurrency** changes shape. Two bool columns on the same creature table sit adjacent in memory; concurrent writers to either column fight over the same cache lines ([§33](33_false_sharing.md) — false sharing). Two presence tables are physically separate numpy arrays; concurrent writers to disjoint tables never collide ([§31](31_disjoint_writes_parallelize.md)).

## The reversal

The clean way to phrase the move: **instead of asking each entity about its state, ask the state-table which entities have that state.** The query is reversed; the lookup is reversed; the work shrinks. Most programs spend their lives doing the wrong direction; the data-oriented mindset is to reverse it.

A production example: in a real ECS daemon, an admission decision is `is_admitted = peer_id in established_contacts`. There is no `is_admitted: bool` on a peer; there is only the question "is this peer's id in the table?". With an `id_to_slot` index map ([§23](23_index_maps.md)) this is O(1), no I/O, no enum.

## When flags are right

Presence is not the only valid representation. A bool column is sometimes right — when nearly every entity has the state set (a near-universal flag wastes nothing as a column and saves on the membership scan); when the predicate is so cheap to compute on the fly that materialising it is silly (`is_positive_x = pos_x > 0`); when the data is short-lived and persistence does not matter; when the lookup pattern is "give me this creature's hunger state" (per-creature query, where a column lookup is O(1) but a presence-table membership scan is O(K) without an index).

In this book, **presence is the default; flags are a tradeoff to earn**.

## Exercises

These extend the §0 simulator skeleton.

1. **Add a `hungry` table.** Add `hungry = np.empty(0, dtype=np.uint32)` to your world. It is empty at start.
2. **Populate it.** Write a system `def classify_hunger(energy, ids) -> np.ndarray` that returns the ids of all creatures with `energy[i] < HUNGER_THRESHOLD`. The body is one numpy line: `ids[energy < HUNGER_THRESHOLD]`. Replace the world's `hungry` with the result each tick.
3. **Build the flag version.** Add a parallel `is_hungry = np.zeros(N, dtype=bool)` indexed by creature slot. Write the equivalent classification system that sets the bool column.
4. **Build the AoS version.** Build a `list[Creature]` where `Creature` is a `@dataclass(slots=True)` with an `is_hungry: bool` field. Write the equivalent classification — a Python `for` loop. (Foreshadow: this is the version most tutorials teach.)
5. **Time all three at 1M creatures, 10% hungry.** Time `classify_hunger` (presence), the bool-column version (flag), and the AoS version. Note the ordering and the magnitudes. Presence and flag should be within ~2-5× of each other (both numpy); the AoS version should be one to two orders of magnitude slower than either (interpreter-bound, per §1).
6. **The membership query.** Write `def is_hungry_p(hungry, id) -> bool` (presence — `bool(np.any(hungry == id))`) and `def is_hungry_f(is_hungry_col, slot) -> bool` (flag — `bool(is_hungry_col[slot])`). Time both at 1M creatures. Note: presence is O(K) without an index map; the flag is O(1). [§23 — Index maps](23_index_maps.md) is the fix that makes presence O(1) too.
7. **"How many are hungry?"** Write it three ways. Presence: `len(hungry)`. Flag column: `int(is_hungry.sum())`. AoS: `sum(1 for c in creatures if c.is_hungry)`. Compare wall times at 1M creatures with 10% hungry. The presence version is constant-time; the flag-column version walks all 1M as a single numpy reduction; the AoS version walks all 1M with interpreter dispatch on every step.
8. *(stretch)* **Persist both.** Serialise the flag-column version with `np.save("is_hungry.npy", is_hungry)` and the presence version with `np.save("hungry.npy", hungry)`. Note the disk size for 1M creatures with 10% hungry. The presence file is ~400 KB; the flag-column file is ~1 MB even though 90% of the bits are `0`. (Compression closes some of this gap, but not all of it — `np.savez_compressed` will help on the flag column more than on the presence array, because the flag column has a long run of zeros to compress and the presence array is already small.)

Reference notes in [17_presence_replaces_flags_solutions.md](17_presence_replaces_flags_solutions.md).

## What's next

[§18 — Add/remove = insert/delete](18_add_remove_insert_delete.md) names what *changes* between the two representations: in the presence world, state transitions are structural moves between tables, not flag flips.
