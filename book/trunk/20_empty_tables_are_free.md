# 20 — Empty tables are free

> *Concept node: see the [DAG](../../concepts/dag.md) and [glossary entry 20](../../concepts/glossary.md#20--empty-tables-are-free).*

<p align="center"><img src="../illustrations/tip_visualize_full.jpg" alt="Visualize the problem — the diagram of an empty table is free" style="max-height: 300px; max-width: 100%;"></p>

If a presence table is empty, the system that iterates it does nothing. No rows, no work. This is the consequence of [§19](19_ebp_dispatch.md) at the limit, and it is the property that lets the simulator scale gracefully under shifting state.

Concretely: a 1,000,000-creature simulation with zero hungry creatures right now spends *zero* cycles in `drive_hunger`. The system is wired into the DAG, runs every tick, takes a numpy array of `hungry` ids of length 0, executes one bulk op that operates on zero elements, returns. The overhead is one function call and one fancy-index of length zero — measured in microseconds, not milliseconds.

This is not "fast in the empty case as an optimisation". It is *free in the empty case as a structural consequence*. The flag-based version runs through the entire creature table even when no flags are set, paying full memory bandwidth to discover that no work is needed. The EBP version is told there is no work by the simple fact of an empty table.

## The Python-default failure: Optional fields on every entity

Python's tutorial reflex when an attribute might be absent is `disease: Optional[Disease] = None`. Every `Creature` carries the field; healthy creatures carry `None`. This looks free — `None` is a singleton, after all — but every instance still pays one slot, every iteration still pays one `getattr`, and the storage still scales with population, not with prevalence.

From [`code/measurement/empty_tables.py`](../../code/measurement/empty_tables.py), one million creatures with a `disease` field at four prevalence levels:

| prevalence | layout                              | RSS    | process tick | n diseased |
|-----------:|-------------------------------------|-------:|-------------:|-----------:|
|     0.00 % | `list[Creature]` with `Optional`    | 105.9 MB |   7.46 ms |          0 |
|     0.00 % | numpy SoA + `diseased` presence     |  26.5 MB |   0.02 ms |          0 |
|     0.10 % | `list[Creature]` with `Optional`    | 106.1 MB |  11.63 ms |      1,002 |
|     0.10 % | numpy SoA + `diseased` presence     |  26.7 MB |   0.06 ms |      1,002 |
|     1.00 % | `list[Creature]` with `Optional`    | 106.7 MB |   9.00 ms |     10,061 |
|     1.00 % | numpy SoA + `diseased` presence     |  26.5 MB |   0.12 ms |     10,064 |
|    10.00 % | `list[Creature]` with `Optional`    | 113.4 MB |  19.17 ms |     99,841 |
|    10.00 % | numpy SoA + `diseased` presence     |  26.6 MB |   0.48 ms |     99,714 |

Read the **0% row** first. With *zero diseased creatures*, the optional-field layout still costs 105.9 MB of RAM and 7.46 ms per tick of "process disease." It pays full population price for state that does not exist. The presence layout pays 0.02 ms — function call plus an empty fancy-index — and an extra ~0 KB for the empty `diseased` array. **At zero prevalence, the optional layout is 365× slower than the presence layout, and 4× heavier in memory.** The optional layout is not paying for what is happening; it is paying for what *might* happen.

Read the **10% row**. The presence layout pays 0.48 ms — proportional to the 100,000 active rows. The optional layout pays 19 ms — proportional to the *full population of one million*, because the loop walks every creature to check the `is None` predicate. The ratio shrinks from 365× to 40× as prevalence rises, but the presence layout always wins, and at typical sparsities (≪ 10% of population is in any specific state at any specific tick) the gap stays large.

The lesson generalises. For every condition you might think of as "optional state" — `disease`, `held_item`, `target`, `cooldown_until`, `aimed_at`, `fingerprint`, `last_login_ip`, `parent_pointer` — the disciplined Python form is **a separate presence table that contains only the entities that have it right now**, not an `Optional[X]` field on every entity.

## Activity-based costs

The effect compounds across many states. A simulation with twenty possible behaviours, each represented as a presence table, pays for the fraction of creatures actually exhibiting each behaviour. Most ticks, most tables are nearly empty. The total work is proportional to the *sum of active rows across all tables*, not to *population × number of behaviours*. For a sparsely active world this is one or two orders of magnitude cheaper than the equivalent flag-based design.

A subtle case worth naming: an *empty system* is not the same thing as a *missing system*. A `drive_hunger` system that iterates an empty `hungry` is still in the DAG, still scheduled, still part of the program's contract. It is just doing zero rows of work this tick. Removing it from the DAG entirely would change the contract; adding it back when the table next gains a row would require dynamic scheduling, which is harder than a no-op call. EBP gives you cheap idle systems, not absent ones.

## Three implications

**Activity-based costs.** A simulator's per-tick cost is set by what is *active*, not by what *exists*. A million dormant creatures cost nothing to ignore. Only behaving creatures consume budget. Most simulators in production rely on this — game worlds with hundreds of thousands of NPCs but only a few in active play, training simulations with millions of agents but few in critical phases, control systems with thousands of sensors but few in alarmed state.

**Structural sparsity.** The world is encouraged to be in mostly-resting states. Designs that scatter activity across many small presence tables (lots of cheap idle systems) outperform designs that concentrate activity in a single big "active creatures" flag. The data-oriented mindset is to multiply states (`hungry`, `sleepy`, `mating`, `fighting`, ...) rather than gate behaviour through one master switch.

**Persistence is also activity-based.** A snapshot of an empty `hungry` table is one row in the schema and zero rows of data. A snapshot of an `is_hungry: np.ndarray` of length 1,000,000 is 1 MB regardless of how many bits are set. Backups, replication, and replay all benefit from the same property.

The flag-based mind sees idle objects as "still present, just inactive". The data-oriented mind sees idle objects as *not in the table*. The difference is one of cost: the former pays for what exists; the latter pays for what is happening.

## Exercises

1. **Time the empty case.** With your simulator from [§19](19_ebp_dispatch.md), run a tick where `hungry` is empty. Time `drive_hunger`. It should be in the microseconds range — function call plus empty fancy-index, no inner work.
2. **Time the same case in flag form.** Run the bool-mask version of `drive_hunger` against a 1,000,000-creature world where `is_hungry.sum() == 0`. Time it. Should be milliseconds — the mask scan still walks the whole column, even though nothing matches.
3. **Run the exhibit.** `uv run code/measurement/empty_tables.py`. Read the 0% row first. Note the absolute cost of the optional layout when nothing is diseased. Note the ratio of optional/presence widening as prevalence drops.
4. **The cost-per-active-creature plot.** Run the EBP simulator with `hungry` size ranging over 0, 100, 1,000, 10,000, 100,000, 1,000,000. Time `drive_hunger` at each. Plot. The line is roughly linear in K, starting at near-zero.
5. **Add four more states.** Add `sleepy`, `mating`, `fighting`, `idle` as presence tables, each with its own driver system. Run a tick where most tables are empty (most creatures are in `idle`). Confirm the per-tick cost is roughly the cost of the `idle` driver only, plus negligible per-system overhead.
6. **Activity histogram.** At each tick, log `(tick, table_name, len)` for every presence table. After 1000 ticks, plot `len` over time. The plot is the simulator's *activity profile*; flat lines mean the world is at rest, bumps mean events are firing.
7. *(stretch)* **Idle systems removed?** Argue why removing an empty system from the DAG (rather than running it with zero work) is the wrong move. Hint: it changes the system DAG, breaks determinism if the table is non-empty next tick, and adds dynamic scheduling cost that exceeds the empty-call overhead.
8. *(stretch)* **The Optional[X] sweep.** Search any Python project you have. Count `Optional[`-typed fields on data classes. For each, ask: at runtime, what fraction of instances actually have it set? If the answer is "almost none," that field is a candidate for a presence table.

Reference notes in [20_empty_tables_are_free_solutions.md](20_empty_tables_are_free_solutions.md).

## What's next

You have closed Existence-based processing. The next phase is *Memory & lifecycle*, starting with [§21 — `swap_remove`](21_swap_remove.md). The simulator is about to start making structural changes to its tables — births and deaths, in production volumes — and the lifecycle phase makes those cheap.
