# 9 — Sort breaks indices

> *Concept node: see the [DAG](../../concepts/dag.md) and [glossary entry 9](../../concepts/glossary.md#9--the-sort-breaks-indices).*

<p align="center"><img src="../illustrations/bridge_clipboard.jpg" alt="Engineer mouse with clipboard and F = ma — alignment is a structural property" style="max-height: 300px; max-width: 100%;"></p>

In [§5 — Identity is an integer](05_identity_is_an_integer.md), exercise 10 left you with a bug. Player 1 was holding the index list `[3, 17, 21, 28, 41]`. The dealer sorted the deck columns by suit. Player 1's hand was now wrong — the same indices, the same slots, but different cards.

That bug is the structural fact this section names. Sorting did not damage anything; the player's reference was never robust to begin with. **An index points at a *slot*, not at a *thing*.** When the slot's contents change, the index quietly changes meaning.

It is not only sorting. Any rearrangement does it: `swap_remove` (a O(1) deletion that moves the last row into the freed slot, coming in [§21](21_swap_remove.md)), reshuffling for locality ([§28](28_sort_for_locality.md)), compacting after a batch of deletions. The same index, the same array, the same line of code, now means a different card.

## "But Python objects are stable references — can't I just go back to that?"

This is the moment many readers feel the urge to retreat. The Python reflex from §6 — `class Card` with attributes — gave you object identity for free. A `Card` instance you held a reference to last week is still the same `Card` object today, regardless of what happened to the list it was in. `id(card)` does not change. The pointer through the Python interpreter to the heap-allocated `Card` is stable for the lifetime of the object.

So the temptation is real: keep the index-aligned numpy columns *and* a parallel `list[Card]` of object references, and use the objects when you need stability. Or just go back to `list[Card]` entirely — at least the references work.

This trade does not survive contact with the §3 footprint table or the §7 access-cost table. The numpy-SoA layout is **5× smaller** and **75× faster** at single-column queries than `list[Card]`; carrying a parallel object list to "rescue" reference stability gives back most of the footprint win and adds the synchronisation problem of keeping the column data in step with the object data. You have not solved the problem; you have hidden it inside an additional invariant.

The structural fix is the one [§10](10_stable_ids_and_generations.md) builds: an `id` column that travels with the row across rearrangements, plus (for variable-quantity tables) a generation counter on top. **The card itself is a slot; the card's *name* is an integer that we choose to be stable.** The cost is one extra `np.uint32` column. The benefit is that every rearrangement we will need from now on — sort, swap_remove, locality-driven reordering, compaction — works without breaking outside references.

This section's only job is to make the *slot vs name* distinction concrete enough that §10's solution feels inevitable rather than ceremonial.

> [!NOTE]
> *Why feel the pain first?* Because the fix in §10 is small — one extra column — and small fixes only stick if the student knows what they fix. Reading "always store an id" without first feeling the bug produces students who add ids cargo-culted, then drop them when the codebase looks too cluttered. Reading it after watching player 1 lose their hand produces students who never drop them.

## Exercises

You should still have your `deck.py` from §5. These exercises extend it.

1. **Reproduce the bug.** With player 1 holding `[3, 17, 21, 28, 41]`, sort the deck columns themselves (`suits`, `ranks`, and `locations` in lockstep) by suit. The pattern is `order = np.argsort(suits, kind="stable"); suits[:] = suits[order]; ranks[:] = ranks[order]; locations[:] = locations[order]`. Print player 1's hand using `card_to_string`. Confirm the cards have changed.
2. **A second rearrangement.** Instead of sorting, swap two cards' positions:
   ```python
   suits[[3, 17]] = suits[[17, 3]]
   ranks[[3, 17]] = ranks[[17, 3]]
   locations[[3, 17]] = locations[[17, 3]]
   ```
   Print player 1's hand again. Same bug shape, different cause.
3. **A third rearrangement.** Remove the card at slot 7 with the `swap_remove` pattern (move the last row into slot 7, then drop the last row): `suits[7] = suits[-1]; suits = suits[:-1]` and likewise for the other columns. Print player 1's hand. Note that the cards at slots `[17, 21, 28, 41]` are unchanged but slot 3 may now hold what was previously the last card; meanwhile slot 51 has silently been deleted.
4. **Quantify the breakage.** Write a function that takes the original `[3, 17, 21, 28, 41]` plus a freshly built deck, applies a Fisher-Yates shuffle to the deck columns themselves (`order = rng.permutation(52)` and reorder all three columns), and counts how many of the five references still point at the same `(suit, rank)` value. Run it 100 times. Roughly what fraction of references survive a random shuffle of the deck? (Spoiler: very small. With probability `1/52` per slot, the expected number that survive by accident is `5/52 ≈ 0.1`.)
5. **A reference that *can* survive.** Without writing any new code — on paper — describe what kind of reference would survive a shuffle. (Hint: you already know. The card's `(suit, rank)` is unique to that card. The reference that survives is the one that does not depend on the slot.)
6. **The "object reference" non-fix.** Build a parallel `list[Card]` (use a `@dataclass` if you wish) alongside the numpy columns. Fill them so that `cards[i]` mirrors `(suits[i], ranks[i], locations[i])`. Now sort the numpy columns by suit *without* updating the object list. What does player 1 see if they read from the object list? What if they read from the numpy columns? Note that you have introduced a new bug — *desynchronised* state — without fixing the old one.
7. *(stretch)* **The cost of never rearranging.** Suppose you decide to *never* sort, swap, or remove from the deck columns, to avoid this bug forever. How would shuffling work? How would discarding a card work? Why does this not scale to ten thousand creatures?

Reference notes for these exercises in [09_sort_breaks_indices_solutions.md](09_sort_breaks_indices_solutions.md).

## What's next

Exercise 5 points at the answer; exercise 7 makes the never-rearrange option look bad. The real fix is to store identity *separately from position* — an `id` column that travels with the row across rearrangements, with a generation counter on top for variable-quantity tables. [§10 — Stable IDs and generations](10_stable_ids_and_generations.md) builds it.
