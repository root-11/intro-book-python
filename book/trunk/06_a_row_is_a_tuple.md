# 6 — A row is a tuple

> *Concept node: see the [DAG](../../concepts/dag.md) and [glossary entry 6](../../concepts/glossary.md#6--a-row-is-a-tuple).*

<p align="center"><img src="../illustrations/cad_bearing.jpg" alt="A bearing's dimensioned drawing names every field" style="max-height: 300px; max-width: 100%;"></p>

In §5 you built a deck of 52 cards as three numpy columns. The card at index 17 is the triple `(suits[17], ranks[17], locations[17])`. Together those three values are *the row*. There is no `Card` class. There is not even a tuple object — the row exists *implicitly* in the alignment: the same index, used in every column, recovers all the data about one card.

This is what we call a *row* throughout the rest of the book — a coherent set of values that belong to the same entity. In a `creature` table the row is `(pos[i], vel[i], energy[i], birth_t[i], id[i], gen[i])`. In a `food` table it is `(pos[i], value[i], id[i])`. The fields belong to the same entity by virtue of all sharing index `i`. There is no `dataclass` holding them; there is no `NamedTuple` instance; there is no `dict`. There is only the discipline that whatever index `i` you used to read one column, you also use to read every other column of the same table.

## Why "implicit" matters in Python

Python's tutorial reflex when it sees the word *row* is to reach for a class — `@dataclass class Row` or `class Row(NamedTuple)` or, if performance is mentioned, `class Row: __slots__ = (...)`. Each of these constructs the row as an *object*, with a header, a refcount, and field pointers. None of them are free. From [`code/measurement/classes_or_tuples.py`](../../code/measurement/classes_or_tuples.py), the time to materialise 1,000,000 two-field "rows" on this machine, ordered fastest to slowest:

| how the row is built                                     | time for 1M rows |
|----------------------------------------------------------|-----------------:|
| numpy SoA — two `np.full(N, value)` columns (bulk)       |       0.005 s    |
| `(x, y)` — bare tuple, 1M individual constructions       |       0.007 s    |
| `class` with `__slots__`                                 |       0.109 s    |
| `collections.namedtuple(...)`                            |       0.146 s    |
| `typing.NamedTuple` subclass                             |       0.151 s    |
| `@dataclass(frozen=True, slots=True)`                    |       0.164 s    |

Two readings of this table.

First reading: the bare tuple is **~16× faster** than a slotted class and **~23× faster** than a frozen+slots dataclass for per-row construction. The named alternatives all pay for an object header and per-field descriptor lookup that the tuple skips. From [`code/measurement/simple_namespace.py`](../../code/measurement/simple_namespace.py), even a `dict` (`{'x': 10.0, 'y': 20.0}`) constructs faster than any of the named-class options — about 0.036 s for the same million. *Naming the row* is the cost; the tuple is the cheapest row that is still recognisable as a row.

Second reading — and the one this book cares about — is the top line: **two bulk numpy column allocations construct 1,000,000 rows-worth of data faster than a million individual tuple literals.** Bulk allocation is roughly 30× faster than the named alternatives and is *not even slower than the cheapest per-row option*. The shape that lets you do this — pre-allocate a column once, fill it with values, and treat row `i` as the implicit tuple `(col0[i], col1[i], ...)` — has no per-row construction cost at all. The tuple at index `i` only exists when you ask for it explicitly; until then it lives in contiguous bytes inside numpy columns. From the §3 footprint exhibit, one million ten-field rows cost 99 MB as numpy SoA columns and 437 MB as a list of tuples — and the SoA version pays *zero* per-row construction cost on top of that, because there are no row objects.

A row is a tuple, but in Python the most useful version of that statement is: **a row is a tuple you do not have to build.**

## Alignment is the discipline

The cost of implicit binding is that you must *keep the indices aligned*. If you sort `suits` without also sorting `ranks` and `locations`, the row at every index is corrupted — the deck still has 52 entries in 52 slots, but each slot now holds the suit of one card, the rank of another, the location of a third. This is not a hypothetical bug; you produced it deliberately in §5 exercise 10, and [§9](09_sort_breaks_indices.md) will hand you the structural fix. The rule is simple: *every operation that reorders any column of a table must reorder all columns of that table together.*

The discipline that makes alignment maintainable is **single-writer-per-column**. If only one function writes to `locations`, and that function writes consistently, alignment is never violated. Multiple writers to the same column race against each other and produce inconsistent rows. This is what [§25 (ownership of tables)](25_ownership_of_tables.md) enforces: each table has exactly one writer, and a row is a tuple precisely because that one writer kept all its columns in step.

A row is a tuple — assembled from columns indexed by the same entity, kept aligned by discipline rather than by any container holding it together.

## Exercises

These extend your `deck.py` from §5.

1. **Print row 17.** Write `def row(suits, ranks, locations, i)` returning `(int(suits[i]), int(ranks[i]), int(locations[i]))`. Use it to print the suit, rank, and location of card 17.
2. **Mishandle the alignment.** Sort *only* `suits` in place: `suits.sort()`. Print row 17 again. The values are now from three different cards — exactly the bug.
3. **Lockstep sort.** Reset the deck. Now sort all three columns *together* using an order array: `order = np.argsort(suits); suits[:] = suits[order]; ranks[:] = ranks[order]; locations[:] = locations[order]`. Print row 17 again. The values are from one card. (The `[:]` matters — it is an in-place assignment that keeps the same backing array; `suits = suits[order]` would rebind the name to a new array and break aliases held elsewhere.)
4. **Add a fourth column.** Add `dealt_at = np.full(52, 255, dtype=np.uint8)` (when a card is dealt at tick `t`, write `t` into `dealt_at[i]`; the sentinel 255 means "not yet dealt"). Modify your lockstep sort to also reorder this column. Verify by spot-check that a row is still consistent after a sort.
5. **The single-writer rule.** Write `def reorder_deck(suits, ranks, locations, dealt_at, order)`. This function is the *only* one that should ever reorder any column of the deck. Document that contract in a docstring above the function. Refactor your shuffle and sort to call it.
6. **The construction cost, your machine.** Run `uv run code/measurement/classes_or_tuples.py` on your machine. Note the ratios. Confirm that the slotted-dataclass row, the canonical "right" answer in modern Python, is the *slowest* of the named options at construction.
7. *(stretch)* **When alignment is moot.** A query that uses only `(suits[i], ranks[i])` to identify a card — for instance, "is this the Ace of Spades?" — does not depend on `locations` or `dealt_at`. Write such a query (one line, using `np.where`). The natural-key view from §5's strong form means this query survives reorderings of unrelated columns; only `suits` and `ranks` need to be aligned with each other.

Reference notes in [06_a_row_is_a_tuple_solutions.md](06_a_row_is_a_tuple_solutions.md).

## What's next

[§7 — Structure of arrays (SoA)](07_structure_of_arrays.md) names the layout choice you have been making implicitly: each field its own column. The next section defends that choice against its alternative.
