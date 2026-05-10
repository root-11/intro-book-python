# Solutions: 19 — EBP dispatch

## Exercise 1 — Re-read your alive-fraction numbers

The §18 alive-fraction exhibit is the EBP-vs-filtered comparison:

| §18 column        | §19 vocabulary           |
|-------------------|--------------------------|
| AoS (`for c if c.alive`) | filtered iteration in pure Python |
| numpy bool mask          | filtered iteration in numpy |
| numpy presence (ids)     | EBP dispatch in numpy |

At 1% sparsity (typical for transient state): EBP is **10×** faster than the filtered numpy version, **150×** faster than the AoS version. As sparsity rises, the EBP advantage shrinks; at 100% live the bool mask wins because the "filter" is a no-op.

The takeaway: EBP is the right default for sparse states; bool masks are the right default for near-universal states. Both happen to be correct on a wide range of hardware; AoS is wrong at every fraction.

## Exercise 2 — Implement both, on creatures

```python
import numpy as np, timeit

n = 1_000_000
rng = np.random.default_rng(0)
energy = rng.uniform(0, 100, n).astype(np.float32)
ids = np.arange(n, dtype=np.uint32)
hungry = ids[energy < 10]                            # 10% sparsity
is_hungry = energy < 10
HUNGER = 0.5
dt = 1/30

def filtered(energy, is_hungry, dt):
    energy[is_hungry] -= HUNGER * dt

def ebp(energy, hungry, dt):
    energy[hungry] -= HUNGER * dt

t_f = timeit.timeit(lambda: filtered(energy.copy(), is_hungry, dt), number=200) / 200
t_e = timeit.timeit(lambda: ebp(energy.copy(), hungry, dt),         number=200) / 200
print(f"filtered: {t_f*1e6:.0f} µs   EBP: {t_e*1e6:.0f} µs   ratio: {t_f/t_e:.1f}×")
```

```
filtered: 2756 µs   EBP: 421 µs   ratio: 6.5×
```

At 10% sparsity, EBP is 6.5× faster on this machine. The filtered version reads `is_hungry` in full (1M bytes scanned) plus `energy` at the masked positions. The EBP version reads only `hungry` (the K = 100K hungry indices, 400 KB) plus `energy` at those positions. EBP's working set is 90% smaller.

## Exercise 3 — The isinstance trap

```python
from dataclasses import dataclass

@dataclass(slots=True)
class Creature: energy: float
class Hungry(Creature): pass
class Sleepy(Creature): pass

# build a 1M list with three types mixed
ents = []
for i in range(n):
    e = float(energy[i])
    if e < 10:    ents.append(Hungry(e))
    elif e > 80:  ents.append(Sleepy(e))
    else:         ents.append(Creature(e))

def isinstance_dispatch(ents, dt):
    for e in ents:
        if isinstance(e, Hungry):
            e.energy -= HUNGER * dt

t_i = timeit.timeit(lambda: isinstance_dispatch(ents, dt), number=3) / 3
print(f"isinstance chain: {t_i*1e3:.1f} ms")
```

```
isinstance chain: 32.4 ms
```

At 1M entities with 10% Hungry, the `isinstance` chain costs **77× more than EBP** (32.4 ms vs 0.42 ms). The cost is not the `isinstance` call alone — it's per-entity interpreter dispatch *plus* `isinstance`, *plus* `getattr(e, "energy")`, *plus* the attribute write back to a heap-allocated object. Predicate-per-entity is the structural cost; `isinstance` is its idiomatic embodiment.

## Exercise 4 — The polymorphic-method trap

```python
class Creature:
    __slots__ = ("energy",)
    def __init__(self, e): self.energy = e
    def update(self, dt): pass

class Hungry(Creature):
    def update(self, dt):
        self.energy -= HUNGER * dt

# rebuild ents with subclass instances
ents = [Hungry(e) if e < 10 else Creature(e) for e in (float(x) for x in energy)]

def polymorphic(ents, dt):
    for e in ents:
        e.update(dt)

t_p = timeit.timeit(lambda: polymorphic(ents, dt), number=3) / 3
print(f"polymorphic dispatch: {t_p*1e3:.1f} ms")
```

Typical: ~50-80 ms. The source-code branching disappeared (no `if isinstance` in the loop body), but the *cost* moved into Python's method resolution. Each call:

1. Looks up `update` via the MRO chain (one for `Creature`, one for `Hungry`).
2. Sets up a Python frame for the method call.
3. Dispatches to a different code path depending on the runtime type — a *cache miss* every time the type changes.

The "cleaner code" form is *more* expensive than the visible-branch form — the predicate is consulted as often, and each consultation is more work than `isinstance`.

## Exercise 5 — The list-comprehension filter

```python
def list_comp_dispatch(creatures, dt):
    hungry_list = [c for c in creatures if isinstance(c, Hungry)]      # filter pass
    for c in hungry_list:                                              # work pass
        c.energy -= HUNGER * dt

# Two passes: one to filter, one to work. Plus a list allocation.
```

The cost is the filtered-iteration baseline *plus* the list allocation. At 1M entities with 10% hungry, expect ~30-40 ms — comparable to the `isinstance` chain, with extra allocation pressure.

The shape *looks* like EBP (a list containing only the hungry ones). The difference is *when* the filtering happens. EBP's `hungry` table is built when the *transition* occurs (energy crosses below threshold) — once per creature per state change. The list-comp form rebuilds it every read — once per query, on the entire population.

For a simulator with multiple consumers of "the hungry creatures" per tick, this gap compounds: EBP pays 1× the build cost, list-comp pays K× (K = number of consumers).

## Exercise 6 — A multi-state system

```python
hungry = ids[energy < 10]
sleepy = ids[energy > 80]
dead   = ids[energy < 0]

def drive_hunger(hungry, energy, dt):
    energy[hungry] -= HUNGER * dt

def drive_sleep(sleepy, energy, dt):
    pass     # sleepy creatures are at rest; no energy change

def drive_death(dead, world):
    world.live_count -= len(dead)

# Each system reads its own table. Disjoint write-sets where possible.
```

Three EBP systems, three independent write-sets:

- `drive_hunger`  reads `hungry`,  writes `energy[hungry slots]`
- `drive_sleep`   reads `sleepy`,  writes nothing (or a separate `rest_log`)
- `drive_death`   reads `dead`,    writes `world.live_count` (or `to_remove`)

Now compare to the filtered alternative:

```python
def drive_all_filtered(creatures, dt):
    for c in creatures:
        if c.is_hungry:    c.energy -= HUNGER * dt
        elif c.is_sleepy:  pass
        elif c.is_dead:    c.live = False
```

The filtered version is *one* loop with *three* shared write-sets (`energy`, `live`, etc.). The three EBP systems can run in parallel; the filtered loop cannot, because all three branches write through the same Python list.

The §31 multiprocessing pattern is the same systems, run on disjoint slices of `hungry`. The filtered version cannot be split that cleanly because the consumer can't tell, before reading each creature, which branch it will take.

## Exercise 7 — A naive EBP bug (stretch)

```python
hungry = list(np.arange(5, dtype=np.uint32))     # five creatures hungry
energy = np.array([5.0, 8.0, 3.0, 1.0, 7.0], dtype=np.float32)

# anti-pattern: bad! mutating hungry while iterating it
for cid in hungry:
    energy[cid] -= 1
    if energy[cid] < 2:                          # crossed a deeper threshold
        hungry.append(cid + 100)                 # also become *very_hungry*
```

The bug: the `for` loop's iteration is over `hungry`'s state at iteration start; appending to `hungry` mid-iteration may or may not extend the iteration depending on the iterator's implementation. With a Python list, appending *does* extend the iteration; with a generator over a numpy slice, it does not. Either way, the behavior is fragile — and reasoning about which creatures end up processed depends on the iteration's implementation detail.

The fix is the deferred-cleanup pattern from §15:

```python
to_add: list[int] = []
for cid in hungry:
    energy[cid] -= 1
    if energy[cid] < 2:
        to_add.append(cid + 100)

# After the iteration completes, apply the queued changes
hungry.extend(to_add)
```

The iteration sees a consistent snapshot. Mutations are queued and applied at a clear boundary. This is exactly the [§15](15_state_changes_between_ticks.md) and [§22](22_mutations_buffer.md) discipline scaling down to a single system.
