# Solutions: 9 — Sort breaks indices

These exercises produce the bug, vary it, and quantify it. The structural fix is in [§10](10_stable_ids_and_generations.md).

## Exercise 1 — Reproduce the bug

```python
import numpy as np
suits     = np.repeat(np.arange(4, dtype=np.uint8), 13)
ranks     = np.tile(np.arange(13, dtype=np.uint8), 4)
locations = np.zeros(52, dtype=np.uint8)

# Shuffle once so positions are non-trivial
rng = np.random.default_rng(42)
order = rng.permutation(52)
suits[:] = suits[order]; ranks[:] = ranks[order]

# Player 1 records indices [3, 17, 21, 28, 41]
held = [3, 17, 21, 28, 41]
locations[held] = 1
print("before sort:",
      [f"{RANK[ranks[i]]}{SUIT[suits[i]]}" for i in held])
# ['K♦', '3♣', 'A♦', '4♥', 'A♠']

# Sort the columns themselves by suit
order = np.argsort(suits, kind="stable")
suits[:]     = suits[order]
ranks[:]     = ranks[order]
locations[:] = locations[order]

print("after sort:",
      [f"{RANK[ranks[i]]}{SUIT[suits[i]]}" for i in held])
# ['10♠', 'Q♥', '3♥', '9♦', 'J♣']
```

Slots `[3, 17, 21, 28, 41]` now hold completely different cards. Player 1's reference list has not changed; the slot contents have. The same line of code (`ranks[3]`) returns a different value before and after the sort. The bug is not in the sort; the bug is that **the index was never a name for the card.**

## Exercise 2 — A second rearrangement

```python
# fresh deck (rebuild from ex 1)
suits[[3, 17]] = suits[[17, 3]]
ranks[[3, 17]] = ranks[[17, 3]]
locations[[3, 17]] = locations[[17, 3]]

print([f"{RANK[ranks[i]]}{SUIT[suits[i]]}" for i in [3, 17, 21, 28, 41]])
```

Two cards swap. Player 1's references at indices 3 and 17 now point at each other's old contents. References at 21, 28, 41 are unchanged. Same shape of bug — index is a slot, not a name — different cause.

## Exercise 3 — A third rearrangement

```python
# swap_remove slot 7
suits[7] = suits[-1]
ranks[7] = ranks[-1]
locations[7] = locations[-1]
suits     = suits[:-1]
ranks     = ranks[:-1]
locations = locations[:-1]
```

Slot 7 now holds what *was* the last card (slot 51). Slot 51 no longer exists — the array is length 51. Player 1's references at indices 17, 21, 28, 41 still see the original cards (those slots untouched). Reference at 3 is unchanged because slot 3 was untouched too — but the card formerly at slot 51 has been silently removed from the universe of "cards."

This is the [§21 swap_remove](21_swap_remove.md) pattern: O(1) deletion at the cost of moving one row's worth of data, plus changing the index of one other row. Cheap, fast, and devastating to external references.

## Exercise 4 — Quantify the breakage

```python
def survival(rng):
    suits = np.repeat(np.arange(4, dtype=np.uint8), 13)
    ranks = np.tile(np.arange(13, dtype=np.uint8), 4)
    # shuffle once so the deck is non-trivial
    o = np.random.default_rng(42).permutation(52)
    suits[:] = suits[o]; ranks[:] = ranks[o]

    held = [3, 17, 21, 28, 41]
    pairs = [(suits[i], ranks[i]) for i in held]

    # rearrange and count survivors
    o = rng.permutation(52)
    suits[:] = suits[o]; ranks[:] = ranks[o]
    return sum(1 for (s, r), i in zip(pairs, held) if (suits[i], ranks[i]) == (s, r))

rng = np.random.default_rng(0)
print(f"survived: {sum(survival(rng) for _ in range(100))} / 500")
```

```
survived: 13 / 500
```

Expected value: each reference has probability `1/52 ≈ 1.9%` of pointing at its original card after a uniform shuffle. Five references × 100 trials × 1/52 ≈ 9.6 expected survivors. Empirically: 13 — within Poisson noise of the prediction.

98% of references are wrong after one shuffle. Not "occasionally broken in edge cases" — *catastrophically* broken in the common case.

## Exercise 5 — A reference that *can* survive

The reference that survives a shuffle is the one that does not depend on the slot. The natural-key reference `(suit, rank)` survives any rearrangement because `(suits[i], ranks[i])` is a property of the *card*, not of the slot. The dealer rearranges slots; the cards themselves are not changed.

But natural keys break in two cases:

- **Duplicates.** Variable-quantity tables (creatures, items, projectiles) routinely have rows with identical field values; "the creature with energy=10 at position (5,5)" can have many matches. A natural key needs to be a *guaranteed-unique* property of the row.
- **Re-issues.** A row removed and a new row added with the same values is indistinguishable by natural key. For variable-quantity tables this is a bug waiting to happen.

The structural fix in [§10](10_stable_ids_and_generations.md) is to *invent* a name and write it down: an `id` column whose values are guaranteed unique within the table, plus a generation counter to handle re-issues.

## Exercise 6 — The "object reference" non-fix

```python
from dataclasses import dataclass

@dataclass
class Card:
    suit: int
    rank: int
    location: int

# parallel lists
suits     = np.repeat(np.arange(4, dtype=np.uint8), 13)
ranks     = np.tile(np.arange(13, dtype=np.uint8), 4)
locations = np.zeros(52, dtype=np.uint8)
cards = [Card(int(suits[i]), int(ranks[i]), int(locations[i])) for i in range(52)]

# sort the numpy columns, NOT the object list
order = np.argsort(suits, kind="stable")
suits[:]     = suits[order]
ranks[:]     = ranks[order]
locations[:] = locations[order]

# now: numpy columns sorted, cards list still in original order
print(f"numpy slot 3:   {RANK[ranks[3]]}{SUIT[suits[3]]}")
print(f"object cards[3]: rank={cards[3].rank} suit={cards[3].suit}")
# they disagree
```

The object list and the numpy columns now describe two different decks. Player 1 reading from `cards[3]` sees one card; reading `(suits[3], ranks[3])` sees another. You have not fixed the index-into-slot problem — you have *added* a synchronisation problem on top of it.

This is why §9 explicitly rejects the parallel-object-list approach: it preserves stable references at the cost of doubling memory and inventing an alignment invariant the original problem didn't have. The cure is worse than the disease.

## Exercise 7 — The cost of never rearranging (stretch)

If the deck columns are never sorted, swapped, or compacted:

- **Shuffling**: must produce an `order` array each time and read through it indirectly. `for i in range(52): print(card_at(order[i]))`. Every read pays one extra indirection. Workable for 52 cards.
- **Discarding** a card: cannot remove it from the columns; must mark it dead via a status column (e.g., `locations[i] = 255`). The columns grow forever. For 52 cards over a single game session, fine.
- **Adding** a card: `np.concatenate` to grow each column. O(N) per addition.

Why this doesn't scale to 10,000 creatures (let alone the simulator's 100M):

1. **Forever-growing tables.** A simulator that runs for an hour and births 10K creatures per second has 36M dead rows by the end. Reading through them costs proportionally; bandwidth is the budget; you've spent it on tombstones.
2. **No compaction means no locality.** Live and dead rows are interleaved. Cache lines hold half-tombstones. The §28 *sort for locality* pattern is impossible.
3. **Parallel partition is impossible.** [§31-§32](31_disjoint_writes_parallelize.md) split the table by index range; if the live data is sparse and randomly distributed across a forever-growing array, you can't carve clean ranges.

The never-rearrange policy works for constant-quantity tables (52 cards, fixed grid sizes). It fails for everything that breathes — births, deaths, additions, removals. The book's simulator is variable-quantity, so the next chapter builds the fix.
