# Solutions: 7 — Structure of arrays (SoA)

## Exercise 1 — Build both layouts

```python
import numpy as np
from dataclasses import dataclass

@dataclass
class Card:
    suit: int
    rank: int
    location: int

def make_soa():
    return (np.repeat(np.arange(4, dtype=np.uint8), 13),
            np.tile(np.arange(13, dtype=np.uint8), 4),
            np.zeros(52, dtype=np.uint8))

def make_aos():
    return [Card(i // 13, i % 13, 0) for i in range(52)]
```

The two layouts encode the same logical content. The SoA version costs 156 bytes of data plus three numpy header objects. The AoS version costs 52 `Card` instances (each ~88 bytes including header, refcount, and three `int` fields) plus a list pointing at them — close to 5 KB total. **30× memory difference at N=52**, before you've added any operations.

## Exercise 2 — Count cards in a player's hand

```python
def count_held_soa(locations, player):
    return int(np.sum(locations == player))

def count_held_aos(cards, player):
    return sum(1 for c in cards if c.location == player)

# both return the same number on the same logical deck
```

`np.sum(locations == player)` produces a boolean mask in C, sums its `True` entries in C, returns an int — no Python iteration. The generator-expression form pays per-element interpreter dispatch *plus* `getattr` (`.location`) on each `Card`.

## Exercise 3 — Time the count at N = 10,000

```python
import timeit
n = 10_000
cards = [Card(i%4, i%13, i%5) for i in range(n)]
suits = np.tile(np.arange(4, dtype=np.uint8),  n//4+1)[:n]
ranks = np.tile(np.arange(13, dtype=np.uint8), n//13+1)[:n]
locations = np.tile(np.arange(5, dtype=np.uint8), n//5+1)[:n]

t_soa = timeit.timeit(lambda: count_held_soa(locations, 1), number=1000) / 1000
t_aos = timeit.timeit(lambda: count_held_aos(cards, 1),     number=100)  / 100
print(f"SoA: {t_soa*1e6:.2f} µs   AoS: {t_aos*1e6:.1f} µs   ratio: {t_aos/t_soa:.0f}×")
```

```
SoA:    5.89 µs   AoS:   181.9 µs   ratio:   31×
```

At N=10,000 the SoA version is **31× faster** for the same answer.

## Exercise 4 — Scale to 1,000,000 entries

```
SoA:  226.07 µs   AoS: 12,008.3 µs   ratio:   53×
```

The ratio widens with N because the SoA call stays bandwidth-bound (a tight C loop reading int8s sequentially) while the AoS call stays interpreter-bound (one Python step per row). Doubling N doubles both costs, but they live in different regimes — at 1M, SoA finishes in 0.2 ms, AoS in 12 ms. AoS uses 36% of a 30 Hz tick budget on a single count-by-attribute query.

## Exercise 5 — The hot/cold case, Python edition

Add `nickname: str = ""` and `dealt_at: int = -1` to `Card`, rebuild, time again:

```
SoA:  226.07 µs   AoS5: 12,524.5 µs   (vs AoS3: 12,008.3)
```

The SoA time is *unchanged* (the count still walks only `locations`). The AoS time is *also roughly unchanged* (~4% slower from slightly larger objects, but not the multiplicative blowup the Rust edition's chapter shows).

This is the Python-specific shape of the SoA win the chapter prose names: in Rust, AoS pays for unread fields by dragging them into the cache line. In Python, AoS pays a fixed-per-attribute interpreter cost regardless of how wide the row is — adding fields you don't read makes each `Card` heavier in *memory* but does not slow the per-attribute *access*. The penalty is set by `getattr` and bytecode dispatch, not by cache-line traffic.

The categorical SoA win in Python is the *escape from the interpreter*. The numpy primitive runs in C; the AoS loop runs in CPython. No `slots=True`, no `__slots__`, no `@dataclass(frozen=True)` removes that gap.

## Exercise 6 — A case where AoS does not lose

```python
import time, numpy as np

# AoS: update one card, all five fields
cards = [Card(0, 0, 0) for _ in range(1_000_000)]
target = cards[42]
t0 = time.perf_counter()
target.suit = 1; target.rank = 5; target.location = 2
t1 = time.perf_counter()
print(f"AoS 1-card update: {(t1 - t0) * 1e9:.0f} ns")

# SoA: same update, three columns
suits = np.zeros(1_000_000, dtype=np.uint8)
ranks = np.zeros(1_000_000, dtype=np.uint8)
locations = np.zeros(1_000_000, dtype=np.uint8)
t0 = time.perf_counter()
suits[42] = 1; ranks[42] = 5; locations[42] = 2
t1 = time.perf_counter()
print(f"SoA 1-card update: {(t1 - t0) * 1e9:.0f} ns")
```

For a *single* row update, AoS and SoA are within noise of each other — both pay one or three Python attribute accesses, no inner loop, no scaling. The regime distinction from §4 doesn't apply because there is no loop to be inside or outside of. AoS is competitive whenever your access pattern is "touch one row, read or write all its fields" — for example, a UI inspector showing details of a selected entity.

The book's argument is not that AoS is always worse. It is that *the inner loop of every system in the simulator reads one or two columns across many rows* — exactly the case where SoA wins by the order of magnitude. AoS for the bookkeeping (the list of system names, the schedule); SoA for the rows.

## Exercise 7 — Construct, then read

```python
import time, numpy as np
from dataclasses import dataclass

@dataclass
class Card:
    suit: int
    rank: int
    location: int

n = 1_000_000

# AoS: build once, query 1000 times
t0 = time.perf_counter()
cards = [Card(i%4, i%13, i%5) for i in range(n)]
t1 = time.perf_counter()
build_aos = t1 - t0
t0 = time.perf_counter()
for _ in range(1000):
    sum(1 for c in cards if c.location == 1)
t1 = time.perf_counter()
read_aos = t1 - t0

# SoA: build once, query 1000 times
t0 = time.perf_counter()
suits = np.tile(np.arange(4, dtype=np.uint8), n//4+1)[:n]
ranks = np.tile(np.arange(13, dtype=np.uint8), n//13+1)[:n]
locations = np.tile(np.arange(5, dtype=np.uint8), n//5+1)[:n]
t1 = time.perf_counter()
build_soa = t1 - t0
t0 = time.perf_counter()
for _ in range(1000):
    int(np.sum(locations == 1))
t1 = time.perf_counter()
read_soa = t1 - t0

print(f"AoS: build {build_aos*1000:.1f} ms, read 1000× {read_aos*1000:.1f} ms, total {(build_aos+read_aos)*1000:.1f} ms")
print(f"SoA: build {build_soa*1000:.1f} ms, read 1000× {read_soa*1000:.1f} ms, total {(build_soa+read_soa)*1000:.1f} ms")
```

Build cost amortises across many reads. For long-lived data (a deck that exists for the duration of a game session), the construction cost is a one-off. For short-lived data (a list of "cards dealt this hand" rebuilt every tick), construction can dominate — and even SoA pays a non-trivial construction time for million-element columns. This foreshadows [§22 — mutations buffer](22_mutations_buffer.md): pre-allocate once, mutate in place, *never reconstruct in the inner loop.*

## Exercise 8 — A from-scratch `SoaDeck` class (stretch)

```python
class SoaDeck:
    """The single owner of the deck columns. The only mutation entry point is `reorder`."""

    def __init__(self):
        self.suits     = np.repeat(np.arange(4, dtype=np.uint8), 13)
        self.ranks     = np.tile(np.arange(13, dtype=np.uint8), 4)
        self.locations = np.zeros(52, dtype=np.uint8)
        self.dealt_at  = np.full(52, 255, dtype=np.uint8)

    def reorder(self, order: np.ndarray) -> None:
        """Apply `order` to every column in lockstep — the only function permitted to do so."""
        self.suits[:]     = self.suits[order]
        self.ranks[:]     = self.ranks[order]
        self.locations[:] = self.locations[order]
        self.dealt_at[:]  = self.dealt_at[order]

    def shuffle(self, rng: np.random.Generator) -> None:
        self.reorder(rng.permutation(len(self.suits)))

    def sort_by_suit_then_rank(self) -> None:
        self.reorder(np.lexsort((self.ranks, self.suits)))

    def deal(self, indices: list[int], player: int, tick: int) -> None:
        self.locations[indices] = player
        self.dealt_at[indices]  = tick
```

What you gain: *one* writer per column. Adding a fifth column means editing one method (`reorder`) and one constructor; every existing call site keeps working. Forgetting to reorder a column at a call site is impossible — there is only one site.

What you lose: explicit access to the columns from outside. Code that wants to read `suits` directly has to either reach through `deck.suits` (still allowed; reads are not the issue) or go through a method. For mostly-read systems this is fine; for diagnostic code that wants to peek at internals, the indirection adds friction.

The pattern is the [§25 ownership-of-tables](25_ownership_of_tables.md) discipline at the smallest scale. The simulator's actual tables are larger and have more callers, but the contract is identical: one writer, one reorder, columns read freely.
