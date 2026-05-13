# 5 — Identity is an integer

<p align="center"><img src="../covers/phase_identity_structure.jpg" alt="Identity & structure phase" style="max-height: 380px; max-width: 100%;"></p>

> *Concept node: see the [DAG](../../concepts/dag.md) and [glossary entry 5](../../concepts/glossary.md#5--identity-is-an-integer).*

Hand a Python programmer fifty-two cards and tell them to write code that shuffles, sorts, and deals. Ask how long.

Most will start drawing classes. The "official" Python tutorial path leads here: define `class Card` with `__init__(self, suit, rank)`, then `class Deck` holding a `list[Card]`, then `class Hand`, then probably `class Player` and `class Game`. By the time the type hints are right and the `__repr__` methods print nicely, an evening has passed. There will be debates about whether `Hand` should *contain* `Card` instances or hold references to a shared `Deck`, whether `Deck.shuffle()` should mutate or return a new deck, whether `Card` should be a `@dataclass(frozen=True)` for hashability. None of these debates are wrong; all of them are work that has nothing to do with cards.

The whole problem fits in three lines of numpy. The way it fits is the lesson of this section.

A deck of cards has three pieces of information per card: its suit (♠ ♥ ♦ ♣), its rank (A, 2, ..., K), and its current location (in the deck, in someone's hand, in the discard pile). That is three columns. The deck itself is fifty-two rows.

```python
import numpy as np

suits     = np.zeros(52, dtype=np.uint8)  # 0..3
ranks     = np.zeros(52, dtype=np.uint8)  # 0..12
locations = np.zeros(52, dtype=np.uint8)  # 0=deck, 1..N=hands, 255=discard
```

That is the deck. The whole thing is **156 bytes** — three contiguous columns of 52 unsigned bytes. There is no `Card` class. There is no `Deck` class. The card at index `17` has its suit at `suits[17]`, its rank at `ranks[17]`, and its current location at `locations[17]`. The card *is* the index.

Filling the columns with a fresh, ordered deck is one assignment per column:

```python
suits[:] = np.repeat(np.arange(4, dtype=np.uint8), 13)
ranks[:] = np.tile(np.arange(13, dtype=np.uint8), 4)
locations[:] = 0
```

Dealing card 17 to player 1 is one element write:

```python
locations[17] = 1
```

Asking *what's in player 1's hand* is one numpy primitive:

```python
hand = np.where(locations == 1)[0]
```

`hand` is a numpy array of indices into the deck — a *list of card identities* — not a copy of any card data. Asking *how many cards are in each location* is also one primitive:

```python
counts = np.bincount(locations, minlength=2)  # counts[0] = deck, counts[1] = player 1, ...
```

Shuffling — the move students expect to be hard — is shuffling the order of indices. `0..52` becomes `[7, 32, 1, 19, ...]`, and you read your way through the cards in that order:

```python
order = np.random.permutation(52)
```

Look at what just happened. Nothing about the cards changed. `suits[17]`, `ranks[17]`, and `locations[17]` are exactly the values they were before. The shuffle moved indices, not data.

Sorting works the same way. To sort by suit then rank, you sort the indices by `(suits[i], ranks[i])`:

```python
order = np.lexsort((ranks, suits))  # last key is primary; sort by suit first, then rank
```

The cards do not move. Their identifiers are reordered.

That's the deck of cards in maybe fifteen lines of Python. It includes shuffle, sort, deal, and several queries. It is not a stylistic shortcut; it is what a deck of cards *is*. The class-hierarchy version's evening of work was the cost of pretending a card was an object that owned its suit and rank, when actually a card is one number — an index — and its suit and rank are values stored in arrays at that index.

We call this **identity-is-an-integer**, and it is the precondition for every economy the rest of this book buys you. Persistence will work because tables are easy to serialise — three `np.save` calls. Parallelism will work because indices are cheap to partition. Replay will work because a deck is just three arrays in a state. None of it works if you reach for `class Card`.

## Even *which* integer matters

Not every integer is the same integer for performance. From [`code/measurement/float_or_int_tuple.py`](https://github.com/root-11/intro-book-python/blob/main/code/measurement/float_or_int_tuple.py), looking up keys in a Python `dict` of 10,000 entries:

| key shape                  | lookups / sec |
|----------------------------|--------------:|
| `(int, int)`               |   42,800,637  |
| `(int, int, int)`          |   39,625,273  |
| `(float, float)`           |   26,461,898  |
| `(float, float, int)`      |   26,115,850  |
| `(float, float, float)`    |   17,630,435  |

A two-tuple of ints hashes and compares **2.4× faster** than a three-tuple of floats. Identity-is-an-integer is not just "use a number"; it is "use a small unsigned integer, ideally in a contiguous typed array." A `np.uint8` index packs 64 to a cache line and hashes in one CPU instruction. A `(float, float, float)` "identity" — the kind a Python tutorial might suggest for a 3D point in a dict — pays the price three times: more bytes, slower hash, slower compare.

The card-deck columns above use `np.uint8` deliberately: 0..255 covers everything (4 suits, 13 ranks, up to 254 locations), one byte per value, 64 cards per cache line. The width budget from §2 meets the identity choice from §5: a `np.uint8` column is the cheapest possible identity, the cheapest possible storage, and the cheapest possible lookup, all in one decision.

> [!NOTE]
> *The strong form, which we will return to later:* sometimes you do not even need the index. The pair `(suit, rank)` already uniquely identifies a playing card — there are only fifty-two such pairs. The index is a *surrogate key*; the pair is a *natural key*. For variable-quantity tables (creatures that come and go) you usually need a surrogate, because two creatures can be identical. For a constant-quantity 52-card deck, you do not. The book uses surrogates throughout because the simulator is variable-quantity, but knowing when you can drop the index is its own discipline.

## Exercises

The first time through, write everything from scratch in `deck.py`. Resist the urge to add a `Card` class or helper methods. Three numpy arrays.

1. **Build the deck.** Write `def new_deck() -> tuple[np.ndarray, np.ndarray, np.ndarray]` that returns the suits, ranks, and locations for a fresh, ordered deck (all 52 in `location 0 = deck`). All three arrays are `dtype=np.uint8`.
2. **Print a card.** Write `def card_to_string(suit: int, rank: int) -> str` that returns strings like `"A♠"`, `"10♥"`, `"K♦"`. Use it to print the whole deck.
3. **Shuffle.** Use `np.random.default_rng(seed).permutation(52)` to produce a shuffled order. Print the deck in shuffled order. Confirm by inspection that the `suits`, `ranks`, and `locations` arrays are unchanged.
4. **Sort by suit then rank.** Use `np.lexsort((ranks, suits))` to produce an `order` such that suits come out grouped, ranks ascending within each suit. Print again. Once again, the deck arrays are unchanged.
5. **Deal a hand.** Move the first 5 cards from the deck (location 0) to player 1 (location 1). Print player 1's hand using `card_to_string`.
6. **Hand query.** Write `def cards_held_by(locations: np.ndarray, player: int) -> np.ndarray` returning all card indices currently held by a given player. The body is one line.
7. **Count by location.** Write a function that returns counts grouped by location using `np.bincount`. Confirm `counts[0] + counts[1:].sum() == 52`.
8. **Deal four hands.** Deal 5 cards to each of players 1, 2, 3, 4. Print all four hands.
9. *(stretch)* **Drop the index.** Rewrite `cards_held_by` to return an `(N, 2)` numpy array of `(suit, rank)` pairs directly — no indices. What does this make easier? What does it make harder? (Hint: you cannot move the cards back to the deck without knowing which `i` they were.)
10. *(stretch)* **The sort hazard.** While player 1 is holding indices `[3, 17, 21, 28, 41]`, sort the deck arrays *themselves* in place by suit (`order = np.argsort(suits); suits[:] = suits[order]; ranks[:] = ranks[order]; locations[:] = locations[order]`). What does player 1 think they hold now? Print the cards at the indices `[3, 17, 21, 28, 41]` after the sort. This is the bug [§9 — sort breaks indices](09_sort_breaks_indices.md) was written for. Don't fix it yet — observe it.

Reference solutions for exercises 1-3 in [05_identity_is_an_integer_solutions.md](05_identity_is_an_integer_solutions.md). Solutions for the rest follow the same shape.

## What's next

Exercise 10 leaves you with a bug. The next several sections build the discipline that prevents it: [§6 — A row is a tuple](06_a_row_is_a_tuple.md) is the next vocabulary lesson, and [§9 — sort breaks indices](09_sort_breaks_indices.md) is the fix — keep a stable id alongside the position so external references survive reordering.
