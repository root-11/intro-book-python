# Solutions: 5 — Identity is an integer

The exercises ask you to write three columns and a handful of small functions. The whole deck — shuffle, sort, deal, query — fits in about 50 lines. No `Card`, no `Deck`, no `Hand`.

## Exercise 1 — Build the deck

```python
import numpy as np

def new_deck() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    suits     = np.repeat(np.arange(4, dtype=np.uint8), 13)   # 0,0,...,1,1,...,3,3
    ranks     = np.tile(np.arange(13, dtype=np.uint8), 4)     # 0,1,..,12,0,1,..,12,...
    locations = np.zeros(52, dtype=np.uint8)                  # all in 'deck' (=0)
    return suits, ranks, locations
```

Total bytes: 156. The deck *is* three contiguous arrays of 52 unsigned bytes.

## Exercise 2 — Print a card

```python
SUIT = ['♠', '♥', '♦', '♣']
RANK = ['A','2','3','4','5','6','7','8','9','10','J','Q','K']

def card_to_string(suit: int, rank: int) -> str:
    return f"{RANK[rank]}{SUIT[suit]}"

suits, ranks, _ = new_deck()
for i in range(52):
    print(card_to_string(suits[i], ranks[i]))
```

The string-rendering layer is *outside* the deck. It looks up into two small lookup tables. The deck itself never deals in symbols.

## Exercise 3 — Shuffle

```python
rng = np.random.default_rng(seed=42)
order = rng.permutation(52)
for i in range(52):
    j = order[i]
    print(card_to_string(suits[j], ranks[j]))
```

`order` is a permutation of `[0, 1, ..., 51]`. Reading the deck through `order` reads the cards in shuffled order. `suits` and `ranks` are byte-for-byte unchanged after the shuffle — `(suits == new_deck()[0]).all()` is `True`. **The shuffle moved indices, not data.**

## Exercise 4 — Sort by suit then rank

```python
order = np.lexsort((ranks, suits))   # last key is primary; suit groups, ranks ascending within
for i in range(52):
    j = order[i]
    print(card_to_string(suits[j], ranks[j]))
```

`np.lexsort` returns indices that would sort by the keys (last key dominates). `(ranks, suits)` means: primary sort by suit, secondary by rank. Once again, `suits` and `ranks` are unchanged.

## Exercise 5 — Deal a hand

```python
locations[:5] = 1                                   # first 5 cards → player 1
hand = np.where(locations == 1)[0]                  # indices held by player 1
for i in hand:
    print(card_to_string(suits[i], ranks[i]))
```

One element write per card moved. The card data does not move; only the location markers change.

## Exercise 6 — Hand query

```python
def cards_held_by(locations: np.ndarray, player: int) -> np.ndarray:
    return np.where(locations == player)[0]
```

One line. Returns indices, not card data. The caller looks up the card data through those indices.

## Exercise 7 — Count by location

```python
def location_counts(locations: np.ndarray) -> np.ndarray:
    return np.bincount(locations, minlength=2)

counts = location_counts(locations)
assert counts[0] + counts[1:].sum() == 52
print(f"in deck: {counts[0]}, in hands: {counts[1:].sum()}")
```

`np.bincount` is the right primitive for "count by integer category" — one C-level pass over the locations array. For 52 cards the cost is negligible; the same primitive scales to 100M creatures with hunger states without changing shape.

## Exercise 8 — Deal four hands

```python
suits, ranks, locations = new_deck()
order = rng.permutation(52)

for player in range(1, 5):
    take = order[(player - 1) * 5 : player * 5]
    locations[take] = player

for player in range(1, 5):
    hand = cards_held_by(locations, player)
    cards = [card_to_string(suits[i], ranks[i]) for i in hand]
    print(f"player {player}: {cards}")
```

Twenty cards dealt; four arithmetic slices into a permutation; one assignment per slice. No object construction, no per-card branching.

## Exercise 9 — Drop the index (stretch)

```python
def cards_held_by_pairs(suits: np.ndarray, ranks: np.ndarray,
                        locations: np.ndarray, player: int) -> np.ndarray:
    mask = locations == player
    return np.column_stack([suits[mask], ranks[mask]])     # shape (N, 2)
```

What this makes easier: returning a self-contained snapshot of the hand. The caller can inspect `(suit, rank)` without holding a reference to the deck arrays. For *constant-quantity* tables (a 52-card deck never grows), this is fine.

What it makes harder: putting the cards *back*. To move a card from a hand to the discard pile you need to know the index, not the value — there are 52 distinct cards but no general way to invert from `(suit, rank)` to "which row in the deck arrays held this." For variable-quantity tables (creatures that are born and die), the index is what survives mutations to the table; the (suit, rank) "natural key" is brittle to anything that adds rows.

The book uses indices throughout because the simulator is variable-quantity. For constant-quantity domain (a fixed 52-card deck), dropping the index is a real option.

## Exercise 10 — The sort hazard (stretch)

```python
import numpy as np
suits     = np.repeat(np.arange(4, dtype=np.uint8), 13)
ranks     = np.tile(np.arange(13, dtype=np.uint8), 4)
locations = np.zeros(52, dtype=np.uint8)

# Shuffle the arrays in place so positions are non-trivial
rng = np.random.default_rng(42)
order = rng.permutation(52)
suits[:] = suits[order]
ranks[:] = ranks[order]

# Player 1 holds indices [3, 17, 21, 28, 41]
held = [3, 17, 21, 28, 41]
locations[held] = 1
print("Player 1 holds at indices", held, "→",
      [f"{RANK[ranks[i]]}{SUIT[suits[i]]}" for i in held])
# → ['K♦', '3♣', 'A♦', '4♥', 'A♠']

# Now sort the deck arrays in place by suit
order2 = np.argsort(suits, kind='stable')
suits[:]     = suits[order2]
ranks[:]     = ranks[order2]
locations[:] = locations[order2]

print("After in-place sort, player 1 looks at the SAME indices", held, "→",
      [f"{RANK[ranks[i]]}{SUIT[suits[i]]}" for i in held])
# → ['10♠', 'Q♥', '3♥', '9♦', 'J♣']
```

```
Player 1 holds at indices [3, 17, 21, 28, 41] → ['K♦', '3♣', 'A♦', '4♥', 'A♠']
After in-place sort, player 1 looks at the SAME indices [3, 17, 21, 28, 41] → ['10♠', 'Q♥', '3♥', '9♦', 'J♣']
```

Player 1 recorded *indices* `[3, 17, 21, 28, 41]` and stashed them somewhere outside the deck arrays. The sort moved cards around. Player 1's stored indices now point at whichever cards happened to land at those positions. They are *not* the cards player 1 was holding.

The `locations` column was reordered alongside `suits` and `ranks`, so internally `np.where(locations == 1)` correctly identifies player 1's cards at their new positions (`[8, 20, 27, 32, 44]`). The bug is in the *external* index list — the one the player code held outside the table. **Indices are not stable across reorderings.**

This is the bug [§9 — sort breaks indices](09_sort_breaks_indices.md) addresses. The fix is to issue every card a *stable id* (a number that travels with the card across reorderings) and let external code refer to cards by id, not by current position. The deck arrays then carry an id column whose contents are reordered along with the card data; `np.where(ids == card_id)` finds a card no matter how the rows have been shuffled.
