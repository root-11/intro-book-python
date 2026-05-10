# Solutions: 8 — Where there's one, there's many

These exercises ask you to write the array version first and let the singleton fall out as the trivial case.

## Exercise 1 — The function over a slice

```python
def highest_rank_in_hand(hand, ranks):
    return int(ranks[hand].max())

ranks = np.tile(np.arange(13, dtype=np.uint8), 4)
print(highest_rank_in_hand(np.array([0, 13, 26, 39, 12]), ranks))   # 12 (K)
print(highest_rank_in_hand(np.array([12]), ranks))                  # 12 (K)
print(highest_rank_in_hand(np.array([], dtype=np.int64), ranks))    # raises — see ex 3
```

One function, three N values. The function does not branch on N; numpy's indexing primitive handles all three identically (modulo the empty case).

## Exercise 2 — Reverse the urge

```python
def face_cards(ranks):
    return ranks >= 10                  # J=10, Q=11, K=12

mask = face_cards(ranks)
print(int(mask.sum()))                  # 12 — three face cards × four suits
```

The OOP-shaped `def is_face_card(self) -> bool` would force every caller to write `for c in cards: if c.is_face_card(): ...` — back to the interpreter-bound regime. The array version `face_cards(ranks)` is one numpy primitive that returns a mask, costs ~25 µs at N=100K, and *also* answers the singleton case via `face_cards(np.array([rank]))[0]`.

## Exercise 3 — The N = 0 case

```python
def highest_rank_in_hand(hand, ranks):
    if hand.size == 0:
        return None                    # explicit "no answer" — caller decides
    return int(ranks[hand].max())
```

`arr.max()` on an empty array raises `ValueError: zero-size array ... has no identity`. Three reasonable resolutions:

- **Return `None`.** Forces the caller to handle the empty case explicitly. Best when "no cards" is a normal state.
- **Return a sentinel** (e.g., `-1` for ranks). Cheap; risks confusing data with metadata. Avoid unless the type already has a natural sentinel.
- **Raise.** Right when "empty hand" is a programming error in this code path (e.g., a function that should only be called when at least one card is held).

The book leans toward returning `None` for "no answer" cases because the type signature `Optional[int]` documents the possibility at the call site. The Rust edition has `Option<u8>` for the same reason.

## Exercise 4 — Predicate over a single value

```python
def red_mask(suits):
    return suits < 2                   # 0=♠, 1=♥, 2=♦, 3=♣ — wait, suits 0 and 1 are spades and hearts here

# the chapter assumes suit indexing where 0,1 are red. Use the project's convention.
# If suits 1 (♥) and 2 (♦) are red:
def red_mask(suits):
    return (suits == 1) | (suits == 2)

# singleton case naturally:
suit = 1
is_red = red_mask(np.array([suit]))[0]
```

The array version covers the singleton; the singleton wraps the array version's input in a one-element array. There is no separate code path. (The exact suit indexing — which numbers are red — is a convention to pick once and write down; the book's elsewhere-conventions can drift between editions.)

## Exercise 5 — Count overhead

```python
import timeit, numpy as np

# at N = 52
t_arr = timeit.timeit(lambda: int(face_cards(ranks).sum()), number=10_000) / 10_000
t_loop = timeit.timeit(lambda: sum((int(ranks[i]) >= 10) for i in range(52)), number=1_000) / 1_000
print(f"N=52:        array={t_arr*1e6:.2f} µs   loop={t_loop*1e6:.1f} µs   ratio={t_loop/t_arr:.0f}×")
```

```
N=52:        array=  2.29 µs   loop=    5.8 µs   ratio= 3×
N=100,000:   array= 25.00 µs   loop= 1199.0 µs   ratio=48×
N=1,000,000: array=228.70 µs   loop=12525.0 µs   ratio=55×
```

At N=52 the array version is only 3× faster — numpy's per-call overhead matters at small N. At N=100K the ratio settles at ~50× and stays there as N grows. The interpreter-vs-bandwidth gap from §1 *is* this ratio.

The lesson: even at N=52, where the array version's overhead is dominant, it is *still* faster. Where there's one, there's many; the array version is never slower beyond a couple dozen elements, and is wildly faster past a few hundred.

## Exercise 6 — The dataclass twin, revisited

```python
from dataclasses import dataclass

@dataclass
class Card:
    suit: int
    rank: int

def face_count_aos(cards):
    return sum(1 for c in cards if c.rank >= 10)

def face_count_soa(ranks):
    return int((ranks >= 10).sum())

n = 1_000_000
ranks_col = np.tile(np.arange(13, dtype=np.uint8), n // 13 + 1)[:n]
cards = [Card(0, int(ranks_col[i])) for i in range(n)]

t_aos = timeit.timeit(lambda: face_count_aos(cards), number=5) / 5
t_soa = timeit.timeit(lambda: face_count_soa(ranks_col), number=100) / 100
print(f"AoS face count: {t_aos*1e3:.1f} ms   SoA face count: {t_soa*1e3:.2f} ms   ratio: {t_aos/t_soa:.0f}×")
```

```
AoS face count: 12.5 ms   SoA face count: 0.23 ms   ratio: 55×
```

Same 55× ratio as §7's `count_held`. The cost gap is not query-specific; it is a property of *any* per-element work done in pure Python over `getattr`-accessed fields. Every loop you write in CPython that walks `for entity in entities: ... entity.field ...` lives in this cost regime. SoA + numpy primitives moves the loop into C and out of the regime.

## Exercise 7 — From a tutorial (stretch)

Pick almost any "Object-oriented programming in Python" tutorial that builds a card game (Real Python, Programiz, GeeksforGeeks, the Python docs themselves all have versions). The canonical shape is:

```python
class Card:
    SUITS = ['♠', '♥', '♦', '♣']
    RANKS = ['A', '2', ..., 'K']
    def __init__(self, suit, rank):
        self.suit = suit
        self.rank = rank
    def __repr__(self):
        return f"{self.RANKS[self.rank]}{self.SUITS[self.suit]}"
    def is_face(self):
        return self.rank >= 10

class Deck:
    def __init__(self):
        self.cards = [Card(s, r) for s in range(4) for r in range(13)]
    def shuffle(self):
        random.shuffle(self.cards)
    def deal(self, n):
        return [self.cards.pop() for _ in range(n)]
```

The numpy rewrite is approximately:

```python
import numpy as np

class Deck:
    def __init__(self):
        self.suits = np.repeat(np.arange(4, dtype=np.uint8), 13)
        self.ranks = np.tile(np.arange(13, dtype=np.uint8), 4)
        self.locations = np.zeros(52, dtype=np.uint8)
        self.dealt_at = np.full(52, 255, dtype=np.uint8)
    def shuffle(self, rng):
        order = rng.permutation(52)
        self.suits[:] = self.suits[order]
        self.ranks[:] = self.ranks[order]
        self.locations[:] = self.locations[order]
        self.dealt_at[:] = self.dealt_at[order]
```

Line counts: the OOP version is typically 30-50 lines for `Card` + `Deck`. The numpy version is ~15 lines. *And* "all face cards across the table" is one numpy call (`np.where(self.ranks >= 10)[0]`) instead of a loop over per-card method invocations.

Beyond line count: the numpy version is the precondition for everything in Phase 3+. Persistence is `np.savez(self.suits, self.ranks, self.locations, self.dealt_at)` — three or four arrays out, the same arrays in. Replay is "store the seed, replay the operations." Parallel partitioning is "split the index range." None of these work cleanly when the data lives behind `self.cards = list[Card]`. The savings show up not in this chapter but in the rest of the book.
