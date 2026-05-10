# Solutions: 20 — Empty tables are free

## Exercise 1 — Time the empty case

```python
import numpy as np, timeit

energy = np.zeros(1_000_000, dtype=np.float32)
hungry = np.empty(0, dtype=np.uint32)            # empty table

def drive_hunger_ebp(energy, hungry, dt):
    energy[hungry] -= 0.5 * dt

t = timeit.timeit(lambda: drive_hunger_ebp(energy, hungry, 1/30), number=10_000) / 10_000
print(f"drive_hunger on empty table: {t*1e6:.2f} µs")
```

```
drive_hunger on empty table: ~1-3 µs
```

A function call, a fancy-index of length zero, an `__isub__` on a zero-length view. Microseconds. The system is "in the DAG" but pays almost nothing this tick.

## Exercise 2 — Time the same case in flag form

```python
is_hungry = np.zeros(1_000_000, dtype=bool)      # all False — nothing hungry

def drive_hunger_flag(energy, is_hungry, dt):
    energy[is_hungry] -= 0.5 * dt

t = timeit.timeit(lambda: drive_hunger_flag(energy, is_hungry, 1/30), number=1_000) / 1_000
print(f"drive_hunger on all-False mask: {t*1e6:.0f} µs")
```

```
drive_hunger on all-False mask: ~150-200 µs
```

~100× the EBP cost. The mask scan walks all 1M booleans to determine that none are set; numpy still has to materialise the (empty) result of `energy[is_hungry]`. The "zero work to do" is invisible to the dispatch — the predicate is consulted on every element regardless of the answer.

## Exercise 3 — Run the exhibit

```sh
uv run code/measurement/empty_tables.py
```

```
 prevalence   layout                                  RSS (MB)   tick (ms)
--------------------------------------------------------------------------
     0.00%   list[Creature] with Optional[Disease]      106.4       8.88
     0.00%   numpy SoA + diseased presence              26.6       0.02
     0.10%   list[Creature] with Optional[Disease]      106.3       7.66
     0.10%   numpy SoA + diseased presence              26.7       0.04
     1.00%   list[Creature] with Optional[Disease]      107.1       8.65
     1.00%   numpy SoA + diseased presence              26.8       0.13
    10.00%   list[Creature] with Optional[Disease]      113.8      17.55
    10.00%   numpy SoA + diseased presence              26.5       0.56
```

The 0% row is the headline: *zero diseased creatures*, but the optional-field layout costs **8.88 ms** per tick to discover this. The presence layout costs **0.02 ms** — function-call overhead and an empty fancy-index. The optional layout pays full population price for state that does not exist.

The widening ratio at low prevalence (445× at 0.0%, 191× at 0.1%, 67× at 1%, 31× at 10%) shows that *the optional cost is dominated by the iteration*, not by the work — the loop walks all 1M creatures regardless of how few have a disease.

## Exercise 4 — The cost-per-active-creature plot

```python
import numpy as np, timeit
energy = np.zeros(1_000_000, dtype=np.float32)
results = []
for k in [0, 100, 1_000, 10_000, 100_000, 1_000_000]:
    hungry = np.arange(k, dtype=np.uint32)
    t = timeit.timeit(lambda: drive_hunger_ebp(energy, hungry, 1/30), number=200) / 200
    results.append((k, t * 1e6))
    print(f"K={k:>10}: {t*1e6:>8.1f} µs")
```

```
K=         0:       1.5 µs
K=       100:       2.4 µs
K=     1,000:       3.7 µs
K=    10,000:      14.2 µs
K=   100,000:     143.0 µs
K= 1,000,000:    1820.0 µs
```

Roughly linear in K above ~1000. Below that, the line is dominated by per-call overhead — the work itself disappears into noise. The plot is "y = a + b·K" with `a ≈ 1.5 µs` (overhead) and `b ≈ 1.8 ns` (per-active-creature work).

The line *starts at near-zero* because EBP's cost depends on K, not N. A flag-based plot would be a flat line at ~150 µs (the mask-scan cost) regardless of K. The two strategies have different shapes.

## Exercise 5 — Add four more states

```python
hungry  = ids[energy < 10]
sleepy  = ids[energy > 80]
mating  = np.empty(0, dtype=np.uint32)
fighting = np.empty(0, dtype=np.uint32)
idle    = ids[(energy >= 10) & (energy <= 80)]    # the bulk

def tick(world, dt):
    drive_hunger(world.hungry, world.energy, dt)
    drive_sleep(world.sleepy, world.energy, dt)
    drive_mating(world.mating, world, dt)         # empty — near-zero cost
    drive_fighting(world.fighting, world, dt)     # empty — near-zero cost
    drive_idle(world.idle, world.energy, dt)
```

If `mating` and `fighting` are empty most ticks, the per-tick cost is:

- ~1 µs each for `drive_mating` and `drive_fighting` (empty tables)
- The actual work for `hungry`, `sleepy`, `idle` proportional to their sizes

Total: dominated by `idle` (which holds most of the population) plus small contributions from `hungry`/`sleepy`, plus negligible overhead from the empty tables. A simulator can have *dozens* of dormant systems without paying for them.

## Exercise 6 — Activity histogram

```python
activity_log: list[tuple[int, str, int]] = []

for tick_n in range(1000):
    tick(world, dt)
    for name in ("hungry", "sleepy", "mating", "fighting", "idle", "dead"):
        activity_log.append((tick_n, name, len(getattr(world, name))))

import collections
by_table = collections.defaultdict(list)
for t, name, n in activity_log:
    by_table[name].append((t, n))

# plot each name's series — flat lines = resting world, bumps = events
```

The activity profile *is* the simulator's behaviour. A trace where `hungry` and `dead` stay flat near 0 means the population is well-fed and stable; bumps mean a food shortage hit; a stairstep up means births are outpacing deaths. The same numbers that drive the per-tick cost are also the simulator's "vital signs." Free observability.

## Exercise 7 — Idle systems removed? (stretch)

Removing an empty system from the DAG sounds like a free optimisation. It is not. Three reasons:

1. **Determinism breaks.** The DAG is the contract; a system's *position in the DAG* is part of its definition. Run A removes `drive_mating` because the table is empty; run B (one tick later, after a creature has entered `mating`) puts it back. The execution order has changed; the world hash changes; replay no longer reproduces.

2. **Re-adding it has scheduling cost.** When `mating` next gains a row, the system must be inserted back into the DAG and topo-sorted. Topo-sort is cheap (microseconds for a small DAG) but it is *not* free, and it pays this cost on every transition between empty and non-empty. The empty-call overhead it was supposed to save was *also* microseconds. The fix is more expensive than what it fixes.

3. **The contract is now dynamic.** Static DAG: every run executes the same sequence of systems in the same order. Dynamic DAG: the sequence depends on the run's state. Reasoning about the simulator (which systems run when, what they read and write, what determinism property holds) becomes much harder. *Empty calls are cheap; dynamic schedules are not.*

The right move is to keep all systems in the DAG, accept the few microseconds of overhead per empty system per tick, and design states so most are sparse. A simulator with 30 systems and a 30 Hz tick budget can afford 30 µs of empty-call overhead — under 0.1% of the budget.

## Exercise 8 — The Optional[X] sweep (stretch)

A quick sweep of any Python project for `Optional[`-typed fields:

```sh
grep -rE 'Optional\[|: [A-Z][a-zA-Z]* \| None|: None \|' src/
```

For each hit, ask: at runtime, what fraction of instances actually have it set?

- **`disease: Optional[Disease]`** — 0-2% of creatures. Strong candidate for a `diseased` presence table.
- **`held_item: Optional[Item]`** — 30-60%. Closer; the trade depends on access pattern. If most systems just need to know *whether* an item is held, presence wins. If they need the item type, a column might be simpler.
- **`parent: Optional[Self]`** — varies. Trees with many leaves and few internal nodes: presence wins. Balanced trees: column wins.
- **`last_login_at: Optional[datetime]`** — 99% of users have logged in. Column wins; the `Optional` wrapper is just defensive coding for the never-logged-in edge case.

The pattern: `Optional` fields with low fill-rate are presence tables waiting to be discovered. `Optional` fields with high fill-rate are columns with a sentinel that means "not yet" (a magic timestamp, `255` in a `uint8`, etc.).
