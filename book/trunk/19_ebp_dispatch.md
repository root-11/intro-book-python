# 19 — EBP dispatch

> *Concept node: see the [DAG](../../concepts/dag.md) and [glossary entry 19](../../concepts/glossary.md#19--ebp-dispatch).*

A system that needs to act on hungry creatures has two ways to find them.

**Filtered iteration.** Walk all creatures; for each, ask "is it hungry?"; do work if yes:

```python
for slot in range(len(creatures)):
    if is_hungry[slot]:
        drive_hunger_behaviour(slot)
```

**Existence-based dispatch.** Walk the `hungry` table directly; do work for every entry:

```python
for creature_id in hungry:
    drive_hunger_behaviour(creature_id)
```

In numpy, both shapes lift to one bulk operation:

```python
# filtered (mask-based)
energy[is_hungry] -= HUNGER_BURN_RATE * dt

# EBP (presence-based)
energy[hungry] -= HUNGER_BURN_RATE * dt
```

The two produce the same result. The two have very different costs.

The filtered version evaluates `is_hungry` for every creature — a 1,000,000-byte scan to find the 100,000 hungry ones. The EBP version reads the 100,000 entries of `hungry` and indexes directly. From [`code/measurement/alive_fraction.py`](../../code/measurement/alive_fraction.py) (the §18 exhibit), at 10% sparsity the presence version was **5× faster** than the bool mask version, and at 1% it was **10× faster**. Most simulator states are sparse — a small fraction of creatures are eating at any given tick, a small fraction are reproducing, a small fraction are dying — so EBP's compounding advantage shows up everywhere.

A useful intuition: it is the difference between a wandering shopper trying to remember what they need and a shopper with a list. The list version is shorter, faster, and correct by construction. You do not consult the list to ask "is this aisle on my list?" — you walk down the list and visit each aisle once.

## Three Python anti-shapes that collapse to "filtered iteration"

Python tutorials teach several patterns that all amount to filtered iteration. Each looks different on the page; they all consult a per-entity predicate instead of walking a presence table.

**1. `isinstance` chains.** When entities are modelled as a class hierarchy — `Hungry(Creature)`, `Sleepy(Creature)`, `Dead(Creature)` — dispatch usually walks one big list:

```python
# anti-pattern: bad!
for entity in entities:
    if isinstance(entity, Hungry):
        drive_hunger(entity)
    elif isinstance(entity, Sleepy):
        drive_sleep(entity)
    elif isinstance(entity, Dead):
        # nothing to do
        pass
```

The list contains every entity; the body asks the type-tag predicate per entity. The presence-table version splits this into three independent systems, each iterating its own table.

**2. Polymorphic method dispatch.** The "more Pythonic" version uses dynamic dispatch:

```python
# anti-pattern: bad!
for entity in entities:
    entity.update(dt)
```

Where `Creature.update` is overridden in `Hungry`, `Sleepy`, `Dead`. The `if/elif` is gone from the source code; it has been hidden inside Python's method resolution order. Every iteration still pays an attribute lookup, an MRO walk, and a function-call setup. The predicate is now invisible but it is still being consulted per entity, and the cache penalty for jumping into a different method body for each subclass type is real. EBP replaces this with three explicit functions, each over its own table.

**3. List-comprehension filters.** The Pythonic functional-flavoured version:

```python
# anti-pattern: bad!
hungry_creatures = [c for c in creatures if c.is_hungry]
for c in hungry_creatures:
    drive_hunger(c)
```

This *looks* like EBP — there is a list of just the hungry ones — but the list was built by scanning all N creatures and allocating a fresh Python list with K pointers. The filter pass is the same cost as the filtered-iteration version, *plus* a list allocation. EBP avoids the scan because the presence table was kept up to date as state transitions happened (§18); reads do not have to recompute it.

All three anti-shapes consult the predicate at iteration time. EBP arranges the world so the predicate has already been answered before the system runs — the table itself *is* the answer.

## What EBP looks like as a system

A system that uses EBP looks like:

```python
def drive_hunger(hungry: np.ndarray,
                 energy: np.ndarray,
                 id_to_slot: np.ndarray,
                 dt: float) -> None:
    """Read-set: hungry, id_to_slot.
       Write-set: energy (only at slots indexed by hungry)."""
    slots = id_to_slot[hungry]
    energy[slots] -= HUNGER_BURN_RATE * dt
```

Read-set declared. Write-set declared. No per-row branch; the table is the dispatcher. The signature is the contract — exactly the system shape from [§13](13_system_as_function.md). **EBP is not a separate idea; it is the natural shape that a system takes when its inputs are presence tables.**

EBP also composes cleanly with parallelism. A million creatures with 100,000 hungry can be split across multiple processes — each takes a slice of `hungry` and does its work. The processes never need to consult creatures that are not hungry; their reads do not interfere. [§31](31_disjoint_writes_parallelize.md) develops this under multiprocessing + shared_memory.

The takeaway: EBP is the dispatch that falls out of [§17](17_presence_replaces_flags.md)'s presence-replaces-flags substitution. You do not need to choose to use EBP — once your state is in presence tables, every system naturally iterates them. The filtered-iteration version does not even arise.

## Exercises

1. **Re-read your alive-fraction numbers.** From §18 exercise 2 you have measurements for AoS, bool mask, and presence at five alive-fractions. The same numbers tell the EBP story: the presence column *is* the EBP dispatch path. Confirm by mapping the §18 row labels to the §19 vocabulary — "presence" = "EBP," "bool mask" = "filtered iteration."
2. **Implement both, on creatures.** Implement `drive_hunger_filtered(creatures, is_hungry, dt)` (walks creatures, checks the bool column, applies the burn) and `drive_hunger_ebp(hungry, energy, id_to_slot, dt)` (walks the presence table). Run both on a 1M-creature world with 10% hungry. Time both with `timeit`. Note the ratio.
3. **The isinstance trap.** Build a `list[Creature]` where some are `Hungry(Creature)`, some are `Sleepy(Creature)`, some are plain `Creature`. Implement dispatch via `if isinstance(c, Hungry)` chains. Time it at 1M creatures with 10% Hungry. Now implement the EBP version: three numpy presence tables, three system functions. Time it. The ratio is the cost of consulting the predicate per entity.
4. **The polymorphic-method trap.** Convert exercise 3 to `class Hungry(Creature): def update(self): ...` and a single `for c in creatures: c.update()`. Time it. Note that the source-code complexity *fell* (the `if/elif` is gone), but the runtime cost did not — the predicate moved into Python's method resolution order, where it is still consulted on every iteration.
5. **The list-comprehension filter.** Implement `hungry = [c for c in creatures if c.is_hungry]` followed by `for c in hungry: drive(c)`. Time it. Compare against EBP. Note that the filter pass is the cost of the filtered-iteration version *plus* a list allocation; the EBP version pays neither, because the `hungry` table was maintained at state-transition time, not at read time.
6. **A multi-state system.** A creature can be in any combination of `hungry`, `sleepy`, `dead`. Write three EBP systems: `drive_hunger`, `drive_sleep`, `drive_death`. Each iterates *only its own* presence table. Compare with a single filtered loop that handles all three with `if/elif`. Note that the EBP version has no shared state between the three systems and could trivially run them in parallel ([§31](31_disjoint_writes_parallelize.md)).
7. *(stretch)* **A naive EBP bug.** A system that iterates `hungry` while also calling `hungry.append` on the table corrupts iteration. (You knew this from [§9](09_sort_breaks_indices.md) and [§15](15_state_changes_between_ticks.md).) Construct a small case that demonstrates the bug — a creature that "becomes hungry" mid-iteration. Then fix it via deferred cleanup: write to `to_become_hungry`, apply at tick boundary.

Reference notes in [19_ebp_dispatch_solutions.md](19_ebp_dispatch_solutions.md).

## What's next

[§20 — Empty tables are free](20_empty_tables_are_free.md) names the consequence at scale: cost is proportional to active rows, not to population.
