# Solutions: 15 — State changes between ticks

## Exercise 1 — The list bug

```python
class Creature:
    def __init__(self, energy): self.energy = energy

import random
random.seed(0)
cs = [Creature(-1)] * 30 + [Creature(10)] * 70
random.shuffle(cs)

for c in cs:
    if c.energy <= 0:
        cs.remove(c)

remaining_starvers = sum(1 for c in cs if c.energy <= 0)
print(f"30 starvers initially → {remaining_starvers} survived after remove-during-iter")
```

```
30 starvers initially → 8 survived
```

The iterator advances by index. When `cs.remove(c)` shifts every later element down one slot, the index the iterator advances to *skips the next element*. So roughly every other starver is missed. The exact count depends on the shuffle (8 here, anywhere from 5-15 typical).

The bug is *silent*. The simulation runs, the program terminates, the answer is wrong. Nothing complains. The fact that you have to count the survivors to detect it is precisely why the discipline of buffered mutation exists.

## Exercise 2 — The dict bug

```python
cs = {i: Creature(-1 if i < 30 else 10) for i in range(100)}

# Naive remove-during-iter
try:
    for cid, c in cs.items():
        if c.energy <= 0:
            del cs[cid]
except RuntimeError as e:
    print(f"got expected: {e}")
# RuntimeError: dictionary changed size during iteration

# Local "fix": iterate a snapshot of the keys
cs = {i: Creature(-1 if i < 30 else 10) for i in range(100)}
for cid in list(cs.keys()):                    # snapshot — O(N) allocation
    if cs[cid].energy <= 0:
        del cs[cid]
print(f"survivors: {len(cs)}")                  # 70 — correct!
```

The dict version *crashes loudly*, which is better than silently wrong, but its lesson trains the reader to apply a *local* fix (`list(cs.keys())`) without recognising the structural problem. The local fix:

- Costs an O(N) allocation per tick (the snapshot).
- Hides the mutation pattern; the next reviewer assumes the iteration is safe.
- Doesn't fix the underlying issue: a system reading `cs` while another piece of code (you, in this case) writes to it.

Both bugs are the same lesson: **mutating a container while another piece of code is reading it is the bug**, regardless of whether the language catches it.

## Exercise 3 — The buffered fix

```python
import numpy as np

energy = np.array([-1.0]*30 + [10.0]*70, dtype=np.float32)
np.random.default_rng(0).shuffle(energy)
ids    = np.arange(100, dtype=np.uint32)

to_remove: list[int] = []

def apply_starve(energy: np.ndarray, to_remove: list[int]) -> None:
    starvers = np.where(energy <= 0)[0]
    to_remove.extend(starvers.tolist())

def cleanup(energy, ids, to_remove):
    """Apply queued removals via swap_remove (preview of §21)."""
    if not to_remove: return energy, ids
    keep = np.ones(len(energy), dtype=bool)
    keep[to_remove] = False
    return energy[keep], ids[keep]

apply_starve(energy, to_remove)
energy, ids = cleanup(energy, ids, to_remove)
print(f"30 starvers → {(energy <= 0).sum()} remain after one tick")    # 0
```

The starvation system is *read-only*: it scans `energy` and writes only to `to_remove`. The `energy` array does not change during `apply_starve` — it changes exactly once per tick, in `cleanup`. There is no window in which two systems could see different states.

## Exercise 4 — The cleanup pass

```python
def cleanup(world, to_remove: list[int], to_insert: list[dict]) -> None:
    """Apply removals first, then insertions.

    Removals first because:
    - swap_remove (§21) frees specific slots by moving the last row in.
    - Inserts can target those freed slots (§24 recycling).
    - Doing inserts first would force them to allocate fresh slots even
      when freed slots are about to become available.
    """
    # Removals via swap_remove
    for slot in sorted(to_remove, reverse=True):     # high-to-low avoids index shifting
        last = len(world.energy) - 1
        if slot != last:
            for col in (world.pos_x, world.pos_y, world.energy, world.ids, world.gens):
                col[slot] = col[last]
        # truncate (in real code: world.live_count -= 1)
        ...
    to_remove.clear()

    # Insertions into freed (or newly-allocated) slots
    for row in to_insert:
        slot = world.allocate_slot()                 # reuses recycled slots first
        for col_name, value in row.items():
            getattr(world, col_name)[slot] = value
    to_insert.clear()
```

Removals first means freed slots immediately host inserts on the same tick; the table doesn't grow when births and deaths balance. Inserts first would force every newborn to push the table to a fresh slot before the dying creatures return their slots — a one-tick high-water mark proportional to the death rate. [§24](24_append_only_and_recycling.md) makes this explicit.

## Exercise 5 — Show two ticks

```python
energy = np.array([-1.0, -1.0, 5.0, 8.0, -1.0], dtype=np.float32)
ids    = np.array([10, 11, 12, 13, 14], dtype=np.uint32)
to_remove: list[int] = []

# Tick 1
apply_starve(energy, to_remove)
energy, ids = cleanup(energy, ids, to_remove)
to_remove.clear()
print(f"after tick 1: ids={ids.tolist()}, energy={energy.tolist()}")
# ids=[12, 13], energy=[5.0, 8.0]

# Tick 2 — only the survivors run
apply_starve(energy, to_remove)         # nothing dies
print(f"tick 2 input: ids={ids.tolist()} (the dead ids 10,11,14 are not in this list)")
```

The dead creatures from tick 1 are gone *between* ticks. Tick 2's `apply_starve` sees only the survivors. The systems don't have to know about the death — the cleanup pass handled the bookkeeping at the tick boundary.

## Exercise 6 — Insertions are tick-delayed

```python
ages = np.zeros(N, dtype=np.uint16)

def age_creatures(ages):
    ages += 1                                   # one increment per tick

# Tick 5: a parent reproduces; offspring go into to_insert with age=0
to_insert.append({"pos_x": 1.0, "pos_y": 2.0, "energy": 5.0, "ages": 0})

cleanup(world, to_remove, to_insert)            # offspring now in `creatures`

# Tick 6: age_creatures runs over all live creatures, including the new ones
age_creatures(world.ages)                       # offspring goes 0 → 1
```

The newborn does not appear in any system's read-set during tick 5 — it is in `to_insert`, not in `creatures`. Its first tick of life is tick 6, where it is incremented from 0 to 1 by `age_creatures`. Every creature gets a full tick of life on its first tick, regardless of when in the previous tick it was born.

The alternative (in-tick insertion) would mean a creature born at the start of tick 5 ages from 0 → 1 in the same tick, while one born at the end of tick 5 ages 0 → 0. That arbitrariness is what the rule prevents.

## Exercise 7 — A bad design that almost works (stretch)

The "fix" of processing dead creatures in reverse-index order:

```python
# anti-pattern: bad!
def apply_starve_inplace(creatures):
    dead = [i for i, c in enumerate(creatures) if c.energy <= 0]
    for i in sorted(dead, reverse=True):
        del creatures[i]                        # high-to-low avoids index-shift
```

The case where it still corrupts state: **insertions during the same tick**. Suppose `apply_reproduce` runs *after* `apply_starve` and pushes new creatures onto the same `creatures` list:

```python
# tick body
apply_starve(creatures)                          # deletes some, frees indices
apply_reproduce(creatures)                       # appends new ones
inspect(creatures)                               # sees a mixed-state world
```

What `inspect` sees: a mix of creatures who were alive at the start of the tick (still here, at possibly-different indices), creatures born this tick (already alive in `creatures`), and the *gaps* if the death pattern wasn't a clean suffix. Other systems that captured indices at the start of the tick (e.g., a `pending_event` from `next_event`) now point at wrong rows.

The buffered approach prevents *all* of this by definition: nothing changes in `creatures` until `cleanup`, and `cleanup` applies removals + insertions in one consistent sweep.

## Exercise 8 — Read the simlog (stretch)

The vendored copy at [`.archive/simlog/logger.py`](https://github.com/root-11/intro-book-python/blob/main/.archive/simlog/logger.py) is the production-grade version of this chapter's pattern. Things to find:

- **Two `Container` instances.** The logger maintains two pre-allocated numpy column buffers (`active` and `inactive`). The simulation writes only to `active`.
- **The atomic swap.** When `active` fills, the logger atomically swaps the two references (`active`, `inactive` = `inactive`, `active`). The simulation's next write goes to the now-empty buffer; the previously-active buffer is now `inactive` and ready for flushing.
- **The background flush thread.** A worker thread sleeps until `inactive` is non-empty, then writes its contents to disk (`.npz` chunks) and clears it.

The simulator never holds both containers at once. The flush thread never sees a write in progress. The whole apparatus is the chapter's "writes accumulate, then commit" rule — at production scale, with a background flush, ~1 µs per logged event, and zero coordination cost on the hot path. Worth reading once you have written the toy version yourself.
