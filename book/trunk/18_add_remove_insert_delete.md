# 18 — Add/remove = insert/delete

> *Concept node: see the [DAG](../../concepts/dag.md) and [glossary entry 18](../../concepts/glossary.md#18--addremove--insertdelete).*

In the flag world, a state transition is a write. To make a creature hungry, set `is_hungry = True`. To stop it being hungry, set `is_hungry = False`. The flag was always there; only its value changed.

In the presence world, a state transition is *a move between tables*. To make a creature hungry, *insert* a row into `hungry`. To stop it being hungry, *remove* the row. The state has no field to flip; it has only the question of which table the creature is currently a row of.

```python
# flag (canonical Python tutorial)
def become_hungry_flag(is_hungry: np.ndarray, slot: int) -> None:
    is_hungry[slot] = True

# presence
def become_hungry_presence(hungry: list[int], creature_id: int) -> None:
    hungry.append(creature_id)

def stop_being_hungry_presence(hungry: np.ndarray, creature_id: int) -> np.ndarray:
    pos = np.where(hungry == creature_id)[0]
    if pos.size:
        # swap_remove: move last entry into the freed slot, drop last
        hungry[pos[0]] = hungry[-1]
        return hungry[:-1]
    return hungry
```

## "But I just set the bool, what's the problem?"

The Python idiom that this chapter is asking you to abandon is older and more universal than `is_hungry`. It is `creature.alive = False` — the *soft delete*. Every Python tutorial that introduces classes teaches it: when a thing should stop being processed, set a bool, and check that bool before processing it. Tens of thousands of production codebases run on exactly this pattern.

The cost is real. From [`code/measurement/alive_fraction.py`](../../code/measurement/alive_fraction.py), one motion update over 1,000,000 creatures at varying alive-fraction:

| alive % | AoS (`for c if c.alive`) | numpy bool mask | numpy presence (ids) | mask/presence |
|--------:|-------------------------:|----------------:|---------------------:|--------------:|
|   1.0 % |                10.12 ms  |        0.684 ms |             0.067 ms |        10.2 × |
|  10.0 % |                25.65 ms  |        3.868 ms |             0.747 ms |         5.2 × |
|  50.0 % |                23.78 ms  |        9.470 ms |             2.426 ms |         3.9 × |
|  90.0 % |                32.03 ms  |        3.426 ms |             4.417 ms |         0.8 × |
| 100.0 % |                34.16 ms  |        1.616 ms |             4.968 ms |         0.3 × |

Read the rows. **At 1% alive — the typical case for a transient state like "hungry," "dying," or "just-spawned" — presence is 10× faster than the bool-mask version, and 150× faster than the AoS version.** As alive-fraction climbs, the gap closes; around 80-90% alive the bool mask starts winning, and at 100% alive it is faster (numpy spots the all-True mask and uses a contiguous slice path instead of fancy indexing).

The AoS column is flat at 25-35 ms regardless of alive-fraction. The interpreter is iterating *all* one million creatures and paying the `getattr(c, "alive")` cost on every one, even when 99% of them are skipped a moment later. The "soft delete" pattern saves the actual work but never escapes the per-element dispatch tax.

The honest reading of the table: *presence is the right default for transient state* (low alive-fraction, the common case for hungry/dying/sleeping-and-soon-to-wake); *bool masks are the right default for near-universal state* (alive ≥ 90%); *AoS is wrong at every alive-fraction*. There are no scale ranges where the interpreter loop wins.

> [!NOTE]
> *"Alive" generalises further than this chapter uses it.* In an MMORPG, the relevant set of creatures is the ones inside the player's render radius — and the radius itself can shrink dynamically when CPU is tight, trading visible-creature count against the tick-budget headroom from [§4](04_cost_and_budget.md). **The presence table is a query, not a metaphysical state**; its entries change when the system asks a different question. *"Alive," "hungry," "in-scope," "subscribed," "active-this-frame"* — same shape, different question. The crossover numbers above apply to whichever question your simulation is asking, with whichever fraction the answer happens to have.

## Two consequences worth naming

**The transition is structural.** When a creature crosses the hunger threshold, a row in `hungry` actually appears or disappears. There is no in-place mutation; the table grows by one or shrinks by one. This is why [§22](22_mutations_buffer.md) (mutations buffer; cleanup is batched) exists — adds and removes during a tick must be queued, then applied at the boundary, so that the iteration in progress does not see half the change. The deferred-cleanup pattern is born in this section.

**The vocabulary disappears.** There is no `set_hungry(True)`, no `set_hungry(False)`, no `is_hungry()` accessor pair. There is `become_hungry` (insert) and `stop_being_hungry` (remove), and even those are usually inlined into the system that detects the transition. **The data-oriented program does not have getters and setters; it has systems that move rows between tables.** No `@property`. No `__setattr__` hooks. No "validation lives on the model" decorators. The system that detects the threshold *is* the validation, *is* the transition, *is* the audit trail.

A useful test: can you describe the transition without naming a `bool`? *"This creature became hungry"* — well, did anything change? Yes: the `hungry` table grew by one entry. *"This creature stopped being hungry"* — the table shrank by one entry. Every state change in the system has a structural counterpart, and the structural counterpart is the canonical description.

## Multi-table transitions

The same pattern handles richer transitions. Imagine a creature that can be hungry, sleepy, or dead. Three tables: `hungry`, `sleepy`, `dead`. A creature transitions by moving between them. Becoming sleepy while hungry adds a row to `sleepy` (it can be in both). Dying removes the creature from `hungry` and `sleepy` (cleanup affects all relevant presence tables) and adds to `dead`. The transition is a multi-table operation, but each table is still just a numpy array of ids.

This shape — state changes as inserts and removes — is the precondition for everything else EBP gives you. The dispatch in [§19](19_ebp_dispatch.md) iterates *over the table directly*, so the table's contents *being* the canonical state of the world is structurally necessary. There is no flag to consult; there is only what is in the table right now.

## Exercises

1. **Hunger transitions.** Use your `hungry` table from [§17](17_presence_replaces_flags.md). Each tick: read `energy`; for any creature that crossed below the threshold, append to a `hungry_to_add` buffer; for any that crossed back above, append to a `hungry_to_remove` buffer; apply both at the tick boundary. Run for 100 ticks with energy varying randomly; verify `hungry` always contains exactly the creatures whose current energy is below threshold.
2. **Run the alive-fraction exhibit.** `uv run code/measurement/alive_fraction.py`. Note the crossover row — the alive-fraction at which the bool mask starts beating presence. Note that the AoS column does not have a crossover; it loses at every fraction.
3. **No bool, no setter.** Search your code for any boolean field on a creature. Replace it with a presence table. The setter and getter both disappear. Search for any `@property` decorator that wraps a state field; same fate.
4. **A second presence state.** Add a `sleepy` table. A creature is sleepy if its energy is *high enough that it does not need to eat right now*. A creature can be in both `sleepy` and `hungry`? No — by definition the conditions are mutually exclusive. (Or: design them so they are.) Verify the invariant by checking after each tick that `np.intersect1d(hungry, sleepy).size == 0`.
5. **Death.** Add a `dead` table. When a creature's energy drops below zero, append to `dead` *and* remove from `hungry` (and from `sleepy` if present). The cleanup logic is now multi-table; introduce a small `transition_to_dead(ids, hungry, sleepy, dead)` helper that handles all the affected presence tables.
6. **The transition log.** Add `events: list[tuple[int, int, str]]` (tick number, creature id, event name). Every insert/remove emits a row. After 100 ticks, the events log is the *canonical history* — every state change recorded. This is a preview of [§37 — The log is the world](37_log_is_world.md).
7. *(stretch)* **Reconstruct from the log.** Given only the events log and the initial creature ids, reconstruct the final `hungry`, `sleepy`, and `dead` tables. The reconstruction is a one-shot replay; if it produces the same tables as the live simulation, your transitions are correctly captured.
8. *(stretch)* **The crossover, on your machine.** Re-run the exhibit varying alive-fraction more finely between 70% and 95% — say at 70, 75, 80, 85, 90, 95%. Find the alive-fraction at which mask and presence cross over on *your* hardware. The exact crossover depends on cache size, branch predictor, and the specific numpy build.

Reference notes in [18_add_remove_insert_delete_solutions.md](18_add_remove_insert_delete_solutions.md).

## What's next

[§19 — EBP dispatch](19_ebp_dispatch.md) names the dispatch shape that the table-membership representation makes free.
