# 15 - State changes between ticks

> *Concept node: see the [DAG](../../concepts/dag.md) and [glossary entry 15](../../concepts/glossary.md#15--state-changes-between-ticks).*

<p align="center"><img src="../illustrations/microcontroller_loop.jpg" alt="Init / while { read; process; update } - the visible tick loop" style="max-height: 300px; max-width: 100%;"></p>

Inside a tick, the world is *frozen*. Systems read consistent snapshots of their inputs; mutations are *queued*, not applied; only at the tick boundary does the world step forward in one atomic transition.

This is the rule that makes the DAG from [§14](14_systems_compose_into_a_dag.md) actually work. If `motion` could mutate `pos` while `next_event` is reading `pos`, the data is inconsistent: half the creatures have moved, half have not. Even if the schedule is "correct" by topological order, what each system reads is no longer well-defined. By forbidding mutations to apply in-tick, the world becomes a clean function `world_{t+1} = step(world_t, inputs_t)`. Every system reads `world_t`; every system writes into a buffer that becomes `world_{t+1}` only at the tick boundary.

Concretely: `apply_starve` does not call `np.delete(creatures, slot)` or pop from a Python list. It writes the doomed slot into `to_remove`. The `creatures` columns are unchanged for the rest of the tick. After every system has run, `cleanup` consumes `to_remove` and `to_insert` together, applying every queued change in one sweep. *Now* the next tick begins with a consistent new world state.

This pattern is called *double buffering*: there is the world the systems read (`world_t`), and the buffer of changes that becomes the world the next tick reads (`world_{t+1}`). The pattern shows up everywhere - graphics frame buffers, database transactions, event-sourced systems. The rule is always the same: writes accumulate, then commit.

## The Python footguns this rule prevents

Python has two famous in-place-mutation footguns the discipline above eliminates.

**The list-during-iteration bug.** Removing from a list while iterating it silently skips elements. The iterator advances by index; `list.remove` shifts everything down by one; the next element is now at the index the iterator already passed:

```python
# anti-pattern: bad!
creatures = [c1, c2, c3, c4, c5]   # all five starving
for c in creatures:
    if c.energy <= 0:
        creatures.remove(c)        # skips c2 and c4 - they survive
# Surviving creatures: 2 out of 5. The starvation system is broken
# and the simulation will run forever.
```

**The dict-during-iteration bug.** Removing from a dict while iterating raises:

```python
# anti-pattern: bad!
for cid, c in creatures.items():
    if c.energy <= 0:
        del creatures[cid]
# RuntimeError: dictionary changed size during iteration
```

The list version is the dangerous one - it fails *silently* and hands you a wrong-but-finite simulation. The dict version is dangerous in a different way: the `RuntimeError` trains the reader to fix it locally (`for cid in list(creatures.keys()):`) without ever recognising the structural problem. Both are the same lesson: **mutating a container while another piece of code is reading it is the bug, regardless of whether the language catches it.**

The disciplined Python equivalent in numpy is one boolean mask per buffer:

```python
def apply_starve(energy: np.ndarray, ids: np.ndarray, to_remove: list[int]) -> None:
    starvers = np.where(energy <= 0)[0]          # read-only scan -> slots
    to_remove.extend(ids[starvers].tolist())      # buffer the *ids*, not the slots

def cleanup(world: World, to_remove: list[int], to_insert: list[CreatureRow]) -> None:
    # apply removals first (swap_remove pattern, §21), then inserts
    ...
```

`to_remove` holds *ids*, not slots. Slots move when rows are deleted or sorted; ids do not, so the buffer stays valid for cleanup to resolve through `id_to_slot` ([§22](22_mutations_buffer.md), [§23](23_index_maps.md)). The starvation system *only* writes to `to_remove`. It never touches `creatures`. The `creatures` columns are unchanged when `apply_starve` returns - they are unchanged when `apply_eat` and `apply_reproduce` return. They are mutated *exactly once per tick*, by `cleanup`, after every other system is done. There is no window in which a system could see an inconsistent world.

## What this looks like in code

The removal case buffered *which slots* to drop. The far more common case, a value that changes every tick, buffers *how much* it changed - and when several systems change the same value, each writes its own buffer and `cleanup` combines them in one commit:

```python
energy                  = np.array([1, 2, 3, 4, 5])     # world_t: every system reads this
energy_used             = np.array([-1, -1, 0, 0, -1])  # motion + apply_starve write here
energy_gained_from_food = np.array([1, 2, 0, 0, 3])     # apply_eat writes here

# ... the systems run during the tick. Each reads `energy` and writes only
#     its own buffer - never `energy`, never another system's buffer ...

# tick boundary - one atomic commit:
energy += energy_used + energy_gained_from_food         # energy is now world_{t+1}
```

No system reads a half-updated `energy`, because no system writes to it. `motion` and `apply_starve` write only `energy_used`; `apply_eat` writes only `energy_gained_from_food`. The buffers are separate, so the systems never collide and can run in parallel ([§14](14_systems_compose_into_a_dag.md)); `cleanup` sums them into `energy` in one step at the boundary. That is the whole rule: each system owns a buffer, structural edits go in `to_remove`/`to_insert`, value edits go in a delta column, and the writes combine exactly once.

When the commit must read the old values *while* computing the new ones - a diffusion step, an averaging filter, anything where element `i` depends on its neighbours - the in-place `+=` is unsafe, and you write into a second array and swap the two names at the boundary instead. That swap is the literal double buffer the production logger (exercise 8) is built on: read one array, write the other, exchange them between ticks. The delta-add above is its cheapest special case, for when each element's change is independent of the rest.

## Costs and trade

Two costs to absorb. First, every mutation is one extra entry pushed to a `to_remove` or `to_insert` buffer. Second, the cleanup pass is now its own system in the DAG. The benefit dwarfs the costs: every other system in the book composes cleanly, and parallelism becomes easy. With in-tick mutation, every parallel scheduling decision becomes a race condition. With buffered mutation, races are structurally impossible - disjoint write-sets are disjoint by construction.

A subtle case is *insertions*. A creature born during a tick (via `apply_reproduce`) does not appear in any system's read-set during that tick - it is in `to_insert`, not in `creatures`. The newborn lives its first life on the *next* tick. This is the right behaviour for almost every simulation: it gives every creature an equal first tick of life. The alternative - applying inserts mid-tick - is a closed-loop bug factory.

Within one system, the writes *can* be in-tick: a system that updates `pos_x[:] = pos_x + vel_x * dt` for every creature in one numpy call applies all writes "at once" inside that system, because the rest of the system is the only reader and the only writer. The buffering rule is between *systems*, not between iterations within one system. Inside a system, the writes are sequential (or vectorised); between systems, the writes are batched.

The shape that emerges is: read everything into local arrays at system entry; do work; write outputs to buffers at system exit; commit at tick boundary. It is the same shape as the audio engine's frame buffer, the database's transaction commit, and the version-controlled file system's commit-and-merge. They all solve the same problem: **how do you read consistent state while the world is changing?**

## Exercises

These build on the simulator skeleton. Your `to_remove: list[int]` and `to_insert: list[CreatureRow]` should already exist.

1. **The list bug.** Build a list of 100 creatures where 30 have `energy <= 0`. Iterate the list, calling `creatures.remove(c)` whenever `c.energy <= 0`. Count how many starvers survive. Why did the bug only affect *some* of them? (Hint: every removal shifts the iterator past one extra element.)
2. **The dict bug.** Build a `dict[int, Creature]` of 100 with the same 30 starvers. Iterate `creatures.items()`, calling `del creatures[cid]` whenever `c.energy <= 0`. Note the `RuntimeError`. Now "fix" it locally with `for cid in list(creatures.keys()):` - does the simulation now produce the right answer? Yes, but only because the local fix accidentally makes a complete copy first; you have papered over the structural problem at the cost of an O(N) allocation per tick.
3. **The buffered fix.** Rewrite the function to compute `starvers = np.where(energy <= 0)[0]` (read-only scan) and append the starvers' *ids* (`ids[starvers]`) to `to_remove` - the buffer holds ids, not slots, so it survives the rows moving during cleanup (§22/§23). After the loop completes, apply all removals in one pass using the swap_remove pattern (preview of [§21](21_swap_remove.md)). Verify all 30 starvers die.
4. **The cleanup pass.** Write `def cleanup(world, to_remove, to_insert)`. Apply removals first (using swap_remove on each affected column), then insertions. Why this order, and not the other? (Hint: insertions may reuse slots freed by removals - see [§24](24_append_only_and_recycling.md).)
5. **Show two ticks.** Run the loop for two ticks. After tick 1, log the population. After tick 2, log it again. Confirm that creatures killed in tick 1's `apply_starve` *do not* appear in tick 2's input - they were removed at the tick boundary, between the two ticks.
6. **Insertions are tick-delayed.** A creature reproduces in tick 5: parent in `creatures`, two offspring in `to_insert`. After cleanup, the offspring are in `creatures`. In tick 6 the offspring receive their first system pass. Confirm by adding an `age_in_ticks` column and watching offspring start at 0 in tick 6, not in tick 5.
7. *(stretch)* **A bad design that almost works.** Try to apply mutations in-tick *carefully* - collect dead creatures first, then process them in reverse-index order to avoid the iterator-skip bug. Show one specific case where this still corrupts state. (Hint: a reproduction produces an offspring whose new index conflicts with an in-progress death.)
8. *(stretch)* **Read the simlog.** Open [`.archive/simlog/logger.py`](https://github.com/root-11/intro-book-python/blob/main/.archive/simlog/logger.py). Find the two `Container` instances. Find the line where they swap. Find the function the background thread runs. Note that the logger never holds *both* containers locked simultaneously - the swap is atomic, the dump is on the inactive container. This is the production version of what exercise 3 teaches.

Reference notes in [15_state_changes_between_ticks_solutions.md](15_state_changes_between_ticks_solutions.md).

## What's next

[§16 - Determinism by order](16_determinism_by_order.md) is the property the buffering rule *guarantees*: same inputs, same system order, same outputs. Reproducibility is structural.
