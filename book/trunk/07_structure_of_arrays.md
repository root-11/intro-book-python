# 7 — Structure of arrays (SoA)

> *Concept node: see the [DAG](../../concepts/dag.md) and [glossary entry 7](../../concepts/glossary.md#7--structure-of-arrays-soa).*

Your deck has three numpy columns: `suits`, `ranks`, `locations`. Each field lives in its own array, indexed by entity. This layout is called *Structure of Arrays* — SoA. The opposite layout — a single `list[Card]` where each element is a `dataclass` holding all three fields — is called *Array of Structs* — AoS. They are different choices about *where the same data lives*.

```python
# SoA: three columns, indexed in lockstep
suits     = np.zeros(52, dtype=np.uint8)
ranks     = np.zeros(52, dtype=np.uint8)
locations = np.zeros(52, dtype=np.uint8)

# AoS: one list of objects
@dataclass
class Card:
    suit: int
    rank: int
    location: int

cards: list[Card] = [...]  # 52 instances
```

Most Python programmers reach for AoS by default. It is what every introductory tutorial teaches: define a class for the entity, put instances in a list. The trouble is that in a real loop "the entity" is whatever the inner loop reads, not whatever the data model says belongs together. A system that counts cards in player 1's hand reads only the location column — it does not need suits or ranks at all.

## What "reads only one column" actually costs

With SoA, that count is one numpy primitive:

```python
held_by_p1 = int(np.sum(locations == 1))
```

That call walks **N bytes** of `locations`, generates an N-byte boolean mask, and sums it — all inside C, no Python-level iteration. At N = 1,000,000 cards on this machine, the call takes ~0.5 ms.

With AoS, the same count is a Python `for` loop:

```python
held_by_p1 = sum(1 for c in cards if c.location == 1)
```

That loop pays for one bytecode dispatch per card, one `getattr` per card, one comparison per card, and one increment per card. From §1, interpreter dispatch is ~5 ns/element, and `getattr` adds more. At N = 1,000,000 the same count takes 30-50 ms — **two orders of magnitude slower** for the identical answer on the identical data.

This is the bandwidth-bound vs interpreter-bound regime distinction from §4. SoA pushes the inner loop into C and walks contiguous bytes; AoS keeps the inner loop in the interpreter. The SoA call can run inside a 30 Hz tick (33 ms budget) at 1 million entities and use under 2% of the budget. The AoS call uses the entire tick budget at 1 million entities, leaving no room for the rest of the simulation.

## The Python AoS penalty does not shrink with width

In a Rust AoS layout, the cost grows with the size of the struct: a 19-byte `Card` fills a cache line with three cards instead of sixty-four bytes of locations. A reader who does not need suits and ranks pays for them anyway because they ride in on the same cache line. Add a 16-byte `nickname` field and the gap widens.

In Python the story is different. Every field of a `dataclass` is a `PyObject*` pointer, so a "wider" `Card` does not put more *bytes* in the same cache line — it puts more pointers. The cost of `c.location` is not "extra cache traffic"; it is the fixed overhead of the Python attribute lookup. Adding fields you do not read makes each `Card` heavier in absolute terms (more allocation, more refcounts) but does not slow down the per-attribute access. The penalty is *fixed* by interpreter dispatch and `getattr`.

This makes the SoA win in Python *categorical*, not just *quantitative*. The numpy primitive escapes the interpreter entirely; the AoS loop does not. No amount of `@dataclass(slots=True)` discipline removes the per-attribute dispatch cost. From §6, slots reduce *construction* cost and per-instance memory, but every read of `c.location` still goes through Python's attribute machinery.

## SoA is the default

SoA is therefore the default in this book. AoS is sometimes the right choice — for example when every system reads every field of every entity on every tick (rare), or when N is so small that the loop overhead dominates regardless of layout (think dozens of items, not millions). But this is a tradeoff to *earn* by measurement, not to assume by habit. Write SoA first; switch to AoS only when a benchmark forces you to.

The §3 exhibit ([`code/measurement/aos_vs_soa_footprint.py`](../../code/measurement/aos_vs_soa_footprint.py)) is the reference measurement for this chapter. Re-read its sum-column-0 row: list-of-tuples (the AoS twin) summed column 0 of one million ten-field rows in 30 ms; numpy SoA did the same in 0.4 ms. **75× faster for the canonical "system reads one column" operation.** That is the regime your inner loops will live in for the rest of this book.

## Exercises

You will need `time.perf_counter()` for some of these.

1. **Build both layouts.** Take your `deck.py` from §5 and add an AoS twin: a `list[Card]` of 52 entries, where `Card` is a `@dataclass` with three int fields. Build both and verify they encode the same logical content.
2. **Count cards in a player's hand, both ways.** Write `count_held_soa(locations, player)` using `np.sum(locations == player)` and `count_held_aos(cards, player)` using a Python generator expression. Confirm they return the same number on the same deck.
3. **Time the count at 10,000 entries.** Replicate your deck to length 10,000. Time both functions with `timeit` (e.g., `number=1000` for the numpy one, `number=100` for the AoS one). Note the ratio in nanoseconds per element.
4. **Scale to 1,000,000 entries.** Repeat at length 1,000,000. The SoA version reads 1 MB of bytes; the AoS version walks a million pointer-chases through Python's attribute machinery. Note the ratio. On most machines it is in the 50-200× range.
5. **The hot/cold case, Python edition.** Extend `Card` with a `nickname: str = ""` field and a `dealt_at: int = -1` field — five fields total instead of three. Rebuild both. Time the count again. Note that the **SoA time is unchanged** (the count still walks only `locations`) and the **AoS time is also roughly unchanged** (interpreter dispatch dominates either way). Compare to the Rust version of this chapter, where the AoS time *grows* with row size — Python's penalty is fixed differently.
6. **A case where AoS does not lose.** Write a function that updates *every* field of one specific card. SoA writes to three (or five) different columns; AoS writes to one Python object. For the case "update every field of one card" — single entity, no loop — AoS is competitive or better. Time it. Note that this case has no inner loop, which is why the regime distinction from §4 doesn't apply.
7. **Construct, then read.** From §6 you know constructing `dataclass` instances is slow. Time *building* a million-entry AoS list once, then summing the location query 1000 times. Compare to building a million-entry SoA once, then summing 1000 times. The construction cost amortises over many reads; for short-lived data, even SoA construction time becomes a factor. (Hint: this is a foreshadowing of [§22 — mutations buffer](22_mutations_buffer.md).)
8. *(stretch)* **A from-scratch `SoaDeck` class.** Wrap the columns (suits, ranks, locations, dealt_at) in one Python class that owns them all. Provide `reorder(self, order)` as the only public mutator. What do you gain in correctness? What do you lose in flexibility? (Hint: you have just rebuilt the contract from [§25 — ownership of tables](25_ownership_of_tables.md), four chapters ahead of schedule.)

Reference notes in [07_structure_of_arrays_solutions.md](07_structure_of_arrays_solutions.md).

## What's next

[§8 — Where there's one, there's many](08_where_theres_one_theres_many.md) is the universalising principle. The deck taught it implicitly; the next section names it.
