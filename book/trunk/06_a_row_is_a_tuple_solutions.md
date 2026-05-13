# Solutions: 6 — A row is a tuple

These exercises extend the `deck.py` from §5. They demonstrate one rule: *every operation that reorders any column must reorder all columns together.*

## Exercise 1 — Print row 17

```python
def row(suits, ranks, locations, i):
    return (int(suits[i]), int(ranks[i]), int(locations[i]))

print(row(suits, ranks, locations, 17))
# (1, 4, 0)   — card 17 is suit 1 (♥), rank 4 (5), in deck (0)
```

The row is the implicit tuple `(col0[i], col1[i], col2[i])`. Casting to `int` strips the numpy dtype wrapper for cleaner printing — the underlying data is unchanged.

## Exercise 2 — Mishandle the alignment

```python
suits.sort()                                         # sorts only `suits`
print(row(suits, ranks, locations, 17))
# (1, 4, 0)  — but the (1, ...) is now from one card and (4, 0) from another
```

After `suits.sort()`, position 17 contains the *17th-smallest suit value* but `ranks[17]` and `locations[17]` still hold the rank and location of whichever card *originally* sat at index 17. Row 17 is now a Frankenstein composite of three different cards. Reading any row gives nonsense; the per-column data is internally consistent, but the table no longer has rows.

## Exercise 3 — Lockstep sort

```python
suits, ranks, locations = new_deck()                 # reset

order = np.argsort(suits, kind='stable')             # one permutation, used for all
suits[:]     = suits[order]
ranks[:]     = ranks[order]
locations[:] = locations[order]

print(row(suits, ranks, locations, 17))
# (1, 4, 0)  — values from one card again
```

A single `order` array, applied identically to every column, preserves alignment. The row at any new index is still a coherent tuple from one card.

The `[:]` matters. `suits = suits[order]` *rebinds the local name `suits` to a new array*; any other code holding the original `suits` array (a function parameter, an attribute, an element of a tuple) keeps the *unsorted* array. `suits[:] = suits[order]` writes through the existing buffer, so all aliases see the sort. Aliasing pitfalls live or die on the difference.

## Exercise 4 — Add a fourth column

```python
suits, ranks, locations = new_deck()
dealt_at = np.full(52, 255, dtype=np.uint8)          # 255 = not yet dealt

# example: deal card 17 at tick 7
locations[17] = 1
dealt_at[17]  = 7

# lockstep sort, now over four columns
order = np.argsort(suits, kind='stable')
suits[:]     = suits[order]
ranks[:]     = ranks[order]
locations[:] = locations[order]
dealt_at[:]  = dealt_at[order]

# spot-check: find where card 17 ended up via dealt_at = 7
moved_to = int(np.where(dealt_at == 7)[0][0])
print(row(suits, ranks, locations, moved_to), dealt_at[moved_to])
# (1, 4, 1) 7   — same card, new index, all four columns aligned
```

Adding a column adds one line to every place that reorders the table. That repetition is exactly what the next exercise factors out.

## Exercise 5 — The single-writer rule

```python
def reorder_deck(suits, ranks, locations, dealt_at, order):
    """The ONLY function permitted to reorder any column of the deck.

    Applies `order` (a permutation array) identically to every column,
    in place, so external references to these arrays continue to see
    aligned rows.
    """
    suits[:]     = suits[order]
    ranks[:]     = ranks[order]
    locations[:] = locations[order]
    dealt_at[:]  = dealt_at[order]


def shuffle(suits, ranks, locations, dealt_at, rng):
    reorder_deck(suits, ranks, locations, dealt_at,
                 rng.permutation(len(suits)))


def sort_by_suit_then_rank(suits, ranks, locations, dealt_at):
    reorder_deck(suits, ranks, locations, dealt_at,
                 np.lexsort((ranks, suits)))
```

The contract is in the docstring; future-you (or any other reader) sees in one place what every reordering must do. Adding a fifth column means editing one function. Forgetting to update one column at the call site stops being possible — there is only one call site.

This is the §25 *ownership-of-tables* discipline applied at the smallest scale: one writer per column, one reorder function per table.

## Exercise 6 — The construction cost, your machine

```sh
uv run code/measurement/classes_or_tuples.py
```

Source: [`code/measurement/classes_or_tuples.py`](https://github.com/root-11/intro-book-python/blob/main/code/measurement/classes_or_tuples.py). One million two-field rows, ordered fastest to slowest:

```
0.004 s  numpy SoA: two np.full(1_000_000, 10.0) calls (bulk)
0.011 s  bare tuple (10.0, 20.0) × 1M individual constructions
0.117 s  class with __slots__
0.157 s  typing.NamedTuple subclass
0.167 s  collections.namedtuple
0.178 s  @dataclass
```

Two readings:

- The slotted dataclass — the canonical "right" answer in modern Python — is the **slowest** of the named options. The slots win is real but small (it removes the per-instance `__dict__`); the dataclass overhead at construction (descriptor lookup, `__init__` call) dominates.
- Bulk numpy column allocation finishes 1M rows-worth of data in **3 ms**, half the time of a million bare-tuple constructions. The shape with no per-row construction cost is the cheapest shape *even when measured against the cheapest per-row option.*

A row is a tuple. The most useful version of that statement is: *a row is a tuple you do not have to build.*

## Exercise 7 — When alignment is moot (stretch)

```python
def is_ace_of_spades(suits, ranks):
    return np.where((suits == 0) & (ranks == 0))[0]

# returns the index (or indices, if duplicates) of the Ace of Spades
print(is_ace_of_spades(suits, ranks))
```

This query reads only `suits` and `ranks`. It is correct as long as those two columns are aligned *with each other*. It does *not* care about the alignment of `locations` or `dealt_at`. If a future reorder swaps two columns alongside `suits` and `ranks` — but for some reason fails to update `dealt_at` — this query still finds the Ace of Spades correctly.

This is the strong-form observation from §5: a `(suit, rank)` natural key uniquely identifies a card without an index. For *constant-quantity* tables (52 cards, fixed) this alternative works. For *variable-quantity* tables (creatures coming and going) you usually need a stable surrogate id, because the natural key may collide or fail to identify a row that has been re-issued. The book uses surrogates throughout because the through-line simulator is variable-quantity; this exercise is a reminder that not every table needs one.
