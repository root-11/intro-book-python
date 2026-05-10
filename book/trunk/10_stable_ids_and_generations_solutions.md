# Solutions: 10 — Stable IDs and generations

The fix from §9: one extra column.

## Exercise 1 — Add the id column

```python
import numpy as np

def new_deck():
    suits     = np.repeat(np.arange(4, dtype=np.uint8), 13)
    ranks     = np.tile(np.arange(13, dtype=np.uint8), 4)
    locations = np.zeros(52, dtype=np.uint8)
    ids       = np.arange(52, dtype=np.uint32)        # the new column
    return suits, ranks, locations, ids

def reorder(suits, ranks, locations, ids, order):
    suits[:]     = suits[order]
    ranks[:]     = ranks[order]
    locations[:] = locations[order]
    ids[:]       = ids[order]                          # one extra line

# verify ids permute, not regenerate
suits, ranks, locations, ids = new_deck()
order = np.argsort(suits, kind="stable")
reorder(suits, ranks, locations, ids, order)
assert sorted(ids.tolist()) == list(range(52))         # same set, just permuted
```

The id column is just a numpy column. The reorder function gains one line.

## Exercise 2 — Find a card by id

```python
def slot_of(ids: np.ndarray, target: int) -> int | None:
    matches = np.where(ids == target)[0]
    return int(matches[0]) if matches.size else None

# after a sort, find the card with id = 17
slot = slot_of(ids, 17)
print(f"id 17 is now at slot {slot}: "
      f"{RANK[ranks[slot]]}{SUIT[suits[slot]]}")
```

O(N) on each lookup. Fine for 52 cards. For million-row tables, [§23](23_index_maps.md) caches the inverse map; that's an optimisation, not a correction.

## Exercise 3 — Resolve the §9 bug

```python
# fresh deck, pre-shuffled so positions are non-trivial
suits, ranks, locations, ids = new_deck()
rng = np.random.default_rng(42)
reorder(suits, ranks, locations, ids, rng.permutation(52))

# Player 1 records IDs [3, 17, 21, 28, 41] — names, not slots
held_ids = [3, 17, 21, 28, 41]
slots = [slot_of(ids, k) for k in held_ids]
locations[slots] = 1
print("before sort:",
      [f"{RANK[ranks[s]]}{SUIT[suits[s]]}" for s in slots])
# ['4♠', '5♥', '9♥', '3♦', '3♣']

# Sort the columns by suit (in lockstep with ids)
reorder(suits, ranks, locations, ids, np.argsort(suits, kind="stable"))

# Look up the same ids — get the new slots — read the cards
slots2 = [slot_of(ids, k) for k in held_ids]
print("after sort: ",
      [f"{RANK[ranks[s]]}{SUIT[suits[s]]}" for s in slots2])
# ['4♠', '5♥', '9♥', '3♦', '3♣']  — same cards!
```

The slots changed; the cards did not. Player 1's reference list is in the *id* domain — names, not addresses — and survives any rearrangement of the columns.

## Exercise 4 — Permutation-friendly hand query

```python
def cards_held_by(locations: np.ndarray, ids: np.ndarray, player: int) -> np.ndarray:
    return ids[locations == player]                    # return ids, not slots

# deal, then sort, then re-query — should return the same set
suits, ranks, locations, ids = new_deck()
locations[[0, 1, 2, 3, 4]] = 1
held_before = set(cards_held_by(locations, ids, 1).tolist())

reorder(suits, ranks, locations, ids, np.argsort(suits, kind="stable"))
held_after = set(cards_held_by(locations, ids, 1).tolist())

assert held_before == held_after                       # same five ids, regardless of sort
```

`locations == player` is a boolean mask of *slots* in the player's hand. Indexing the `ids` column with that mask returns the *names* of those cards. The set of names is invariant under reordering of the columns; the set of slots is not.

## Exercise 5 — A first generation counter

```python
from typing import NamedTuple

class CardRef(NamedTuple):
    id:  int
    gen: int

suits, ranks, locations, ids = new_deck()
gens = np.zeros(52, dtype=np.uint32)

# Take a reference to the card with id=17 BEFORE we recycle anything
slot = slot_of(ids, 17)
ref  = CardRef(id=17, gen=int(gens[slot]))             # gen=0

# A swap_remove-like operation: pop the card from slot 51, fill slot 17 with a "fresh" card
# (not realistic for a 52-card deck, but mimics the simulator pattern)
suits[17]     = suits[51]                              # recycle: move the last card here
ranks[17]     = ranks[51]
locations[17] = locations[51]
ids[17]       = 52                                     # fresh id (would be next sequence number)
gens[17]     += 1                                      # bump the generation: slot was reused

def deref(ids, gens, ref: CardRef) -> int | None:
    slot = slot_of(ids, ref.id)
    if slot is None:                                   # id no longer in the table
        return None
    if int(gens[slot]) != ref.gen:                     # slot recycled since ref taken
        return None
    return slot

print(deref(ids, gens, ref))                           # None — correctly stale
```

The `(id, gen)` pair is the read receipt. After the recycle, `slot_of(ids, 17)` returns `None` (id 17 was overwritten with id 52). Even if id 17 had been re-issued — e.g., into slot 17 — the generation bump (0 → 1) would have caught it: the reference's `gen=0` would not match the slot's `gens[slot]=1`, and `deref` would correctly report stale.

This is the *generational arena* pattern in 30 lines. The same shape carries the simulator's variable-quantity tables in the rest of the book.

## Exercise 6 — A tiny generational arena (stretch)

```python
import numpy as np
from typing import NamedTuple

class CreatureRef(NamedTuple):
    id:  int
    gen: int

class Creatures:
    def __init__(self, capacity: int = 1024):
        self.cap   = capacity
        self.pos   = np.zeros((capacity, 2), dtype=np.float32)
        self.ids   = np.full(capacity, np.iinfo(np.uint32).max, dtype=np.uint32)  # MAX = empty
        self.gens  = np.zeros(capacity, dtype=np.uint32)
        self.free: list[int] = list(range(capacity - 1, -1, -1))                  # stack of free slots
        self.next_id = 0

    def insert(self, x: float, y: float) -> CreatureRef:
        if not self.free:
            raise MemoryError("Creatures table full")
        slot = self.free.pop()
        self.pos[slot, 0] = x
        self.pos[slot, 1] = y
        new_id = self.next_id
        self.next_id += 1
        self.ids[slot] = new_id
        return CreatureRef(id=new_id, gen=int(self.gens[slot]))

    def _slot_of(self, target_id: int) -> int | None:
        m = np.where(self.ids == target_id)[0]
        return int(m[0]) if m.size else None

    def remove(self, ref: CreatureRef) -> bool:
        slot = self._slot_of(ref.id)
        if slot is None or int(self.gens[slot]) != ref.gen:
            return False
        self.ids[slot] = np.iinfo(np.uint32).max        # mark empty
        self.gens[slot] += 1                            # bump generation
        self.free.append(slot)
        return True

    def get(self, ref: CreatureRef) -> tuple[float, float] | None:
        slot = self._slot_of(ref.id)
        if slot is None or int(self.gens[slot]) != ref.gen:
            return None
        return float(self.pos[slot, 0]), float(self.pos[slot, 1])

# Stale-reference test
c = Creatures(capacity=4)
ref_a = c.insert(1.0, 2.0)             # id=0, gen=0
c.remove(ref_a)
ref_b = c.insert(99.0, 99.0)           # id=1, possibly in the same slot, gen=1 there

assert c.get(ref_a) is None             # stale ref correctly rejected
assert c.get(ref_b) == (99.0, 99.0)
```

A 70-line generational arena. The contract: a `CreatureRef` is the only valid handle into the table; the table guarantees that a `get` or `remove` against a stale ref returns `None`/`False` rather than reading or writing the wrong row.

## Exercise 7 — The shape of `id_to_slot` (stretch)

```python
# capacity-bounded inverse map: id → slot, kept in step with the columns
MAX_IDS = 1_000_000
id_to_slot = np.full(MAX_IDS, np.iinfo(np.uint32).max, dtype=np.uint32)

def slot_of_o1(id_to_slot: np.ndarray, target_id: int) -> int | None:
    s = int(id_to_slot[target_id])
    return None if s == np.iinfo(np.uint32).max else s
```

What the inverse map costs:

- **Memory**: `MAX_IDS × 4 bytes` — 4 MB at 1M ids. Constant per id, not per row in the table.
- **Update on every reorder**: when `order` is applied to the columns, also rebuild `id_to_slot` so `id_to_slot[ids[i]] = i` for every new slot. That's another loop of length N (one numpy primitive: `id_to_slot[ids] = np.arange(N)`).

What it buys: O(1) lookups at every dereference. For a simulator that does 100K+ dereferences per tick, this is the difference between a feasible inner loop and a broken one. [§23](23_index_maps.md) builds it properly with the lifecycle (ids issued, freed, recycled) handled.

## Exercise 8 — Compare with a real ECS handle (stretch)

[`bevy_ecs::entity::Entity`](https://docs.rs/bevy_ecs/latest/bevy_ecs/entity/struct.Entity.html) is conceptually two values packed into a `u64`:

- `index`: 32-bit slot in the entity table (≈ this chapter's id field)
- `generation`: 32-bit reuse counter (≈ this chapter's gen field)

Mapping:

| your column        | bevy_ecs                 | notes |
|--------------------|--------------------------|-------|
| `ids[slot]`        | `Entity::index()`        | same idea |
| `gens[slot]`       | `Entity::generation()`   | same idea |
| `(id, gen) tuple`  | `Entity` (one u64)       | bevy packs both into a u64 for cheap copying |
| `slot_of(ids, id)` | internal sparse-set      | bevy uses a `SparseSet` (an `id_to_slot` array) for O(1) lookup |

What bevy adds that you don't strictly need: packed handle (one u64 vs two integers), explicit `Entity::PLACEHOLDER` constant, deserialisation tagging, integration with bevy's reflection/inspector. None of these are *required* for a working ECS — they're ergonomics for a public API used by hundreds of downstream crates.

This is the [§41](41_compression_oriented.md) / [§42](42_you_can_only_fix_what_you_wrote.md) move. Build the small version yourself first; you now know what `Entity` *does*. When you later read bevy's source you can see what it adds and price each addition against your needs. Most simulators don't need a packed `u64` handle; some do. The cost-benefit is yours, with the from-scratch version in hand.
