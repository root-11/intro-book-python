# 10 — Stable IDs and generations

> *Concept node: see the [DAG](../../concepts/dag.md) and [glossary entry 10](../../concepts/glossary.md#10--stable-ids-and-generations).*

In [§9](09_sort_breaks_indices.md) you watched a player's reference go stale because they were holding *slots*, not *names*. The fix is to give each row a name — a stable identifier — that travels with the row when it moves.

A stable id is one extra column. For the deck:

```python
ids = np.arange(52, dtype=np.uint32)
```

Now every card has both a *slot* (its current index in the columns) and an *id* (its name). When you sort the columns, you reorder `ids` in lockstep with everything else:

```python
order = np.argsort(suits, kind="stable")
suits[:]     = suits[order]
ranks[:]     = ranks[order]
locations[:] = locations[order]
ids[:]       = ids[order]
```

The card with `id == 17` is still the same card — its suit, rank, and location are unchanged. It is just at a different *slot*.

To find a card by id, scan the `ids` column:

```python
def slot_of(ids: np.ndarray, target: int) -> int | None:
    matches = np.where(ids == target)[0]
    return int(matches[0]) if matches.size else None
```

That is O(N), which is fine for a 52-card deck and slow for a million creatures. The fix — an `id_to_slot` map maintained on every rearrangement — is [§23 — Index maps](23_index_maps.md). For now the linear scan is honest pedagogy.

## Generations: when slots are reused

The deck is constant-quantity. Always 52 cards, never more, never less. The simple `ids` column is enough.

For variable-quantity tables — creatures that are born and die, packets that arrive and are processed, sessions that come and go — slots get *reused*. A new creature is born in the slot that just held a dead one. The `ids` column for such a table behaves like an *auto-incrementing primary key* in a database: every new row gets a fresh, never-reused integer; old rows keep their original ids forever. The simulator differs from a database in one structural way — it recycles *slots* to keep memory bounded, while a database table just grows. That recycling is what generations exist for. Imagine code that held a reference to the dead creature: their reference points at a slot that may now hold a different creature with possibly the same id (if id reuse happens) or — worse — a *valid-looking* row that is no longer the row they cared about.

One more column fixes it: a `gens` (generation) counter that increments every time a slot is recycled. A reference is now a pair `(id, gen)`. To dereference it, you check that the row's stored `gen` still matches the reference's `gen`. If it does, the reference is live. If it does not, the slot has been recycled since the reference was taken, and the dereference returns `None`.

```python
from typing import NamedTuple

class CreatureRef(NamedTuple):
    id:  int
    gen: int

def get_slot(creatures, ref: CreatureRef) -> int | None:
    slot = creatures.id_to_slot.get(ref.id)
    if slot is None:
        return None
    if int(creatures.gens[slot]) != ref.gen:
        return None
    return slot
```

(This is one of the few places in the book where a `NamedTuple` earns its weight: a `CreatureRef` is a value passed through external code, and giving it field names makes the API readable. Per §6, the cost is real — a `NamedTuple` allocation per reference — but references are rare, not per-tick. Where the same lesson runs through hot data, the answer is still numpy columns.)

This is the pattern called a *generational arena*. It is the single mechanism behind every "handle" type in every ECS engine: Bevy's `Entity`, Rust's `slotmap::SlotMap`, C++'s `entt::registry`, and the indirect-handle pattern in databases. They differ in details — width of the id, packing into a `u64`, generation overflow handling — but the structural idea is the same: one column for identity, one for generation, a checked dereference.

That is enough machinery for the rest of the book to lean on. Sorting now works because the id column travels with the row. Deletion now works because the generation counter rejects stale references. Append-only and recycling tables ([§24](24_append_only_and_recycling.md)) are two policies on the same machinery.

> [!NOTE]
> *The strong form of [§5](05_identity_is_an_integer.md) still applies.* If your row has a natural key — `(suit, rank)`, `(date, ticker)`, `(species, position)` — you do not need a surrogate id. The card-game deck can be played without ids; the reference that survives is the `(suit, rank)` pair, because the data is unique by construction. Surrogate ids and generations earn their keep when the data has no natural unique tuple — which is most of the time once you start producing rows at runtime.

## Exercises

These extend the §5 deck once more, then take a step toward the simulator's variable-quantity case.

1. **Add the id column.** Add `ids = np.arange(52, dtype=np.uint32)` to your deck. Modify your sort so it reorders `ids` along with the other columns. Verify the original ids are still there, just in a new order.
2. **Find a card by id.** Implement `slot_of(ids, target)` as in the prose. Use it to look up the card with `id == 17` after a sort.
3. **Resolve the §9 bug.** With player 1 holding *ids* `[3, 17, 21, 28, 41]` (not slots), sort the deck. Use `slot_of` to translate ids to slots and print the hand. Confirm the cards are unchanged.
4. **Permutation-friendly hand query.** Rewrite `cards_held_by(locations, ids, player) -> np.ndarray` to return *ids*, not slots. The player now holds names. Test by sorting the deck after a deal and confirming `cards_held_by` still returns the same five cards.
5. **A first generation counter.** Add `gens = np.zeros(52, dtype=np.uint32)`. The 52-card deck does not actually recycle, but extend a small `swap_remove`-like operation: pop the last card from the deck (location 0), insert a "fresh" card at the freed slot, and bump that slot's `gens` by one. Take a `CreatureRef`-style `(id, gen)` reference *before* the operation. After the operation, look up the slot by id; check `gens[slot]` against the reference's `gen`. Confirm the dereference correctly reports stale.
6. *(stretch)* **A tiny generational arena.** Outside the deck, build a `Creatures` class with `pos: np.ndarray (float32)`, `gens: np.ndarray (uint32)`, plus `free: list[int]` of slots awaiting reuse. Implement `insert(pos) -> CreatureRef`, `remove(ref)`, and `get(ref) -> float | None`. Convince yourself by example that stale references cannot read a fresh creature's data.
7. *(stretch)* **The shape of `id_to_slot`.** Right now `slot_of` is O(N). Sketch (do not implement) the `id_to_slot` array — `np.full(N_ids, MAX, dtype=np.uint32)` — that lets you do the lookup in O(1). Note what has to happen on every reorder: when slot `i` is the new home of id `k`, `id_to_slot[k] = i`. This is a foreshadow of [§23 — Index maps](23_index_maps.md). The lookup speedup costs you another column to keep aligned.
8. *(stretch)* **Compare with a real ECS handle.** Read the `Entity` documentation for [bevy_ecs](https://docs.rs/bevy_ecs/latest/bevy_ecs/entity/struct.Entity.html) (Rust) or look at the `EntityHandle` docs of any Python ECS library. Identify which of your fields and operations correspond. What does the production library add that you didn't need for the simulator? Decide consciously whether to adopt it. (This is the from-scratch-then-price-the-crate move from [§41 — Compression-oriented programming](41_compression_oriented.md) and [§42 — You can only fix what you wrote](42_you_can_only_fix_what_you_wrote.md).)

Reference solutions for the deck exercises (1-5) in [10_stable_ids_and_generations_solutions.md](10_stable_ids_and_generations_solutions.md). The arena and library exercises follow the same shape and are worth working without reference.

## What's next

You now have stable references. The next thing the simulator will need is to look up a row by id in O(1) rather than O(N) — an `id_to_slot` map maintained on every reordering. That is [§23 — Index maps](23_index_maps.md). It is one extra `np.ndarray`, updated whenever the columns move.

Part 2 is closed. Identity is an integer; rows align in lockstep; SoA is the default; the singleton drops out; sort breaks indices and ids fix it. The next phase is *Time & passes*, starting with [§11 — The tick](11_the_tick.md). The ecosystem simulator from `code/sim/SPEC.md` is about to start running.
