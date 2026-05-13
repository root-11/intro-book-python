# 8 — Where there's one, there's many

> *Concept node: see the [DAG](../../concepts/dag.md) and [glossary entry 8](../../concepts/glossary.md#8--where-theres-one-theres-many).*

<p align="center"><img src="../illustrations/tip_simplify_full.jpg" alt="Break complex problems into smaller parts — the singleton special-cased away" style="max-height: 300px; max-width: 100%;"></p>

Code is written for the array. A function that operates on one entity is just the special case of N = 1; it does not need its own abstraction. A card game with 52 cards is three arrays — suit, rank, location — not 52 objects. A simulation with 100 creatures is six arrays of length 100, not 100 instances of `Creature`. The plural is the primary unit; the singular is the trivial case.

The pattern is simple. Write the array version first. The singleton drops out as a one-element slice. To shuffle one card you swap two indices in the `order` array — same as shuffling the whole deck. To find the highest-rank card in player 1's hand you scan the (small) hand array — same shape as scanning all 52. To deal one card you write one cell in `locations` — same shape as dealing many cells.

## The OOP instinct, named

This stands against an instinct most Python programmers acquire on day one: the urge to write `card.shuffle()` or `creature.update()` and then puzzle over how to do it for many. Almost every Python tutorial models behaviour as methods on objects, then introduces lists of objects as the natural way to *have many*, then introduces `for c in creatures: c.update()` as the natural way to *do something for each*. Three steps, each locally sensible, that together build the pattern this chapter is asking you to drop.

The puzzle does not exist when you write for arrays from the start. `shuffle(deck)` is one function that works for any deck, including a deck of one. `update(creatures)` — taking the columns as numpy arrays — is one function that works for any population, including a population of one. The method-on-object form is *strictly more code* than the function-over-slice form: it requires a class, an `__init__`, a `self` argument that does nothing useful at the array level, and a calling convention that prevents the inner loop from ever leaving the interpreter.

A useful test: when you find yourself writing a method on a class, ask *what does this look like over an array?* If the array version is shorter, drop the method. If the array version is the same length, keep it as a free function over numpy arrays — `def shuffle(suits, ranks, locations, order)`, not `class Deck: def shuffle(self): ...`. Either way, the singleton was never the right unit of code.

## The performance argument

There is also a performance reason — sharper in Python than in any compiled language. A method that operates on one entity at a time forces the system that uses it to call the method N times. From [`code/measurement/cache_cliffs.py`](https://github.com/root-11/intro-book-python/blob/main/code/measurement/cache_cliffs.py), Python per-element work cost ~5 ns regardless of the size of the data; numpy bulk work cost ~0.2 ns/element. The ratio is **roughly 25×** at any size, and that is *just* the dispatch cost — before you add the cost of `getattr(creature, 'energy')` once per call, the refcount work on every return, and the lost opportunity for numpy to use SIMD instructions on contiguous bytes.

In a compiled language, an "obvious" inner loop over `creatures.iter().for_each(|c| c.update())` is something the optimizer can usually rescue — inline the method, fuse the body into the loop, autovectorize the result. In Python the optimizer is the bytecode dispatcher and it cannot do any of that. The per-method-call form is essentially the worst case the language offers. Writing for arrays first is a request the *interpreter* can fulfil — it can hand the work to numpy and step out of the loop entirely. Writing for singletons-and-iterate is a request that pins the work inside the interpreter for every element.

"Where there's one, there's many" is therefore not an architectural slogan but a daily practice. It costs nothing the first time. It costs everything the first time you forget.

## Exercises

These extend `deck.py` once more. The aim is to feel the array-first pattern in your fingertips before Part 3 turns into the rest of the book.

1. **The function over a slice.** Write `def highest_rank_in_hand(hand, ranks)` where `hand` is a numpy array of card indices and `ranks` is the deck's rank column. Body should be one line: `int(ranks[hand].max())`. Use it on a 5-card hand. Then use it on a 1-card hand. Then use it on an empty hand. Same function, three N values.
2. **Reverse the urge.** Given an OOP-style `def is_face_card(self) -> bool` that lives on a hypothetical `Card` class, rewrite it as `def face_cards(ranks)` returning a numpy boolean mask of shape `(N,)`. Apply it to all 52 cards in one call: `mask = face_cards(ranks); face_count = int(mask.sum())`.
3. **The N = 0 case.** What does `highest_rank_in_hand` do when `hand` is empty? `arr.max()` on an empty array raises. Pick a behaviour — return `None`, return a sentinel, raise — and justify the choice. (Hint: most uses can short-circuit with `if hand.size == 0: return None`.)
4. **Predicate over a single value.** Suppose you want `is_red(suit)` for a single card (suits 0 and 1 are hearts/diamonds). Write the array version `def red_mask(suits)` first — one line: `(suits < 2)`. Then convince yourself the singleton case is `red_mask(np.array([suit]))[0]` — the array version covers it.
5. **Count overhead.** Time `sum(is_face_card_per_row(suits[i], ranks[i]) for i in range(52))` against `int(face_cards(ranks).sum())`. The array version should be measurably faster at 52, much faster at 100,000. Document the ratio. (Repeat at N = 100,000 by replicating the deck.)
6. **The dataclass twin, revisited.** Take your `list[Card]` from §7 exercise 1. Write `face_count_aos(cards)` as a generator-expression sum and `face_count_soa(ranks)` as the numpy version. Time both at 1,000,000 entities. The ratio you measure here is the same ratio §7 measured for `count_held` — it is not specific to one query, it is the per-element dispatch cost of *any* inner loop you write in pure Python.
7. *(stretch)* **From a tutorial.** Find any Python tutorial that uses a `class Card` with methods (`__init__`, `is_face`, `__repr__`, etc.). Rewrite their full card game as three (or four) numpy arrays plus free functions. Compare line counts. Compare clarity. Compare what happens when you want to query "all face cards across the table" — one numpy call versus a loop over per-card method calls.

Reference notes in [08_where_theres_one_theres_many_solutions.md](08_where_theres_one_theres_many_solutions.md).

## What's next

You have closed Identity & structure. Cards behave; rows align; layouts are SoA; the singleton drops out. The next phase is *Time & passes*, starting with [§11 — The tick](11_the_tick.md). The ecosystem simulator from `code/sim/SPEC.md` is about to start running.
