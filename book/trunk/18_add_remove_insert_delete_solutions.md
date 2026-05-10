# Solutions: 18 — Add/remove = insert/delete

## Exercise 1 — Hunger transitions

```python
import numpy as np

THRESHOLD = 10.0

def classify_transitions(prev_hungry, energy, ids):
    """Return (to_add, to_remove) for the hungry presence table."""
    is_hungry_now  = energy < THRESHOLD
    was_hungry     = np.zeros(len(energy), dtype=bool)
    if prev_hungry.size:
        # mark slots that were in prev_hungry — assumes ids[i] == i (dense table)
        was_hungry[prev_hungry] = True
    just_became   = ids[ is_hungry_now & ~was_hungry]
    just_recovered = ids[~is_hungry_now &  was_hungry]
    return just_became, just_recovered

def apply_hunger_changes(hungry: np.ndarray,
                          to_add: np.ndarray,
                          to_remove: np.ndarray) -> np.ndarray:
    if to_remove.size:
        hungry = hungry[~np.isin(hungry, to_remove)]
    if to_add.size:
        hungry = np.concatenate([hungry, to_add])
    return hungry

# Per-tick
to_add, to_remove = classify_transitions(world.hungry, world.energy, world.ids)
# (events are batched; apply at tick boundary, §22)
world.hungry = apply_hunger_changes(world.hungry, to_add, to_remove)

# Invariant after the tick:
assert set(world.hungry.tolist()) == set(world.ids[world.energy < THRESHOLD].tolist())
```

The invariant check verifies the table's contents match the predicate at the end of every tick. A simulator that respects this invariant has correctly implemented the structural transition.

## Exercise 2 — Run the alive-fraction exhibit

```sh
uv run code/measurement/alive_fraction.py
```

```
 alive %    AoS (ms)   mask (ms)   presence (ms)    mask/presence
-----------------------------------------------------------------
    1.0%        8.85       0.696           0.070             9.9×
   10.0%       17.48       3.908           0.607             6.4×
   50.0%       23.13       9.718           2.438             4.0×
   90.0%       31.19       3.512           4.559             0.8×
  100.0%       32.80       1.518           4.928             0.3×
```

The crossover is somewhere between 50% and 90% alive — at 50% presence is 4× faster, at 90% mask is 1.3× faster. The exact crossover depends on hardware (next exercise).

The AoS column has no crossover. At every alive-fraction it loses to both numpy versions by 5-50×. The interpreter loop is paying the per-creature dispatch tax regardless of how few creatures actually need work.

## Exercise 3 — No bool, no setter

A typical search shows fields like:

```python
class Creature:
    is_hungry: bool = False
    is_alive:  bool = True
    is_visible: bool = True
    is_in_combat: bool = False
```

After the refactor:

```python
class World:
    hungry:    np.ndarray = np.empty(0, dtype=np.uint32)
    visible:   np.ndarray = np.empty(0, dtype=np.uint32)
    in_combat: np.ndarray = np.empty(0, dtype=np.uint32)
    # `alive` becomes `live_count` + the implicit "all rows up to live_count are alive"
```

`@property` decorators that wrap state fields disappear too. The "validation" they encoded becomes part of the system that detects the transition — the system that *causes* a row to enter `in_combat` is also the only place where the validity of "this entity entered combat" gets checked. There's no separate setter to wrap.

The vocabulary shrinks. `creature.set_hungry(True)` is replaced by *whatever system produced the threshold crossing* appending to `hungry_to_add`. There is no setter; there is a transition.

## Exercise 4 — A second presence state

```python
SLEEP_THRESHOLD = 80.0   # high energy → sleepy (won't need to eat)
HUNGER_THRESHOLD = 10.0

def classify_states(energy, ids):
    hungry = ids[energy < HUNGER_THRESHOLD]
    sleepy = ids[energy >= SLEEP_THRESHOLD]
    return hungry, sleepy

# Invariant: a creature cannot be in both
hungry, sleepy = classify_states(energy, ids)
assert np.intersect1d(hungry, sleepy).size == 0
```

The mutual exclusion is structural (the predicate ranges don't overlap) — `energy < 10` and `energy >= 80` cannot both hold. If the predicates *could* overlap (e.g., `is_hungry` and `is_running`), one option is to enforce mutual exclusion in the apply step; another is to allow a creature to appear in both tables and let the dispatch code in [§19](19_ebp_dispatch.md) handle the overlap explicitly.

## Exercise 5 — Death

```python
def transition_to_dead(dying_ids: np.ndarray,
                      hungry: np.ndarray,
                      sleepy: np.ndarray,
                      dead:   np.ndarray):
    """A multi-table transition. Removes from all 'live state' tables, adds to dead."""
    new_hungry = hungry[~np.isin(hungry, dying_ids)]
    new_sleepy = sleepy[~np.isin(sleepy, dying_ids)]
    new_dead   = np.concatenate([dead, dying_ids])
    return new_hungry, new_sleepy, new_dead

dying = world.ids[world.energy < 0]
world.hungry, world.sleepy, world.dead = transition_to_dead(
    dying, world.hungry, world.sleepy, world.dead
)
```

A multi-table transition is *one* helper, not three independent updates. The helper is the audit trail: any change to the affected tables goes through it. If you later add a `frozen` table, you add it to the helper signature in one place. *No* place outside the helper writes to these tables — the [§25 ownership-of-tables](25_ownership_of_tables.md) discipline at the multi-table scale.

## Exercise 6 — The transition log

```python
events: list[tuple[int, int, str]] = []          # (tick, creature_id, event_name)

def log_transitions(events, tick, to_add_hungry, to_remove_hungry):
    for cid in to_add_hungry.tolist():
        events.append((tick, cid, "became_hungry"))
    for cid in to_remove_hungry.tolist():
        events.append((tick, cid, "stopped_being_hungry"))

# After 100 ticks, the events list is the canonical history of state transitions.
print(f"events captured: {len(events)}")
print(f"first 5: {events[:5]}")
```

Every state transition is now a row in the events log. The current state of `hungry` is *equivalent to* the sequence of `became_hungry` and `stopped_being_hungry` events applied in order. This equivalence is the §37 *log-is-world* claim — once you have it, replay, audit, and rollback all become projections of the log.

For a real simulator the events would be stored in numpy columns (timestamp, creature_id_int, event_kind_int) — see the simlog reference. For the exercise, a Python list is fine; converting at end-of-run is cheap.

## Exercise 7 — Reconstruct from the log (stretch)

```python
def replay(events: list[tuple[int, int, str]]) -> dict[str, set[int]]:
    """Reconstruct the live tables from an event log."""
    tables = {"hungry": set(), "sleepy": set(), "dead": set()}
    for _, cid, name in events:
        if name == "became_hungry":          tables["hungry"].add(cid)
        elif name == "stopped_being_hungry":  tables["hungry"].discard(cid)
        elif name == "became_sleepy":         tables["sleepy"].add(cid)
        elif name == "stopped_being_sleepy":  tables["sleepy"].discard(cid)
        elif name == "died":
            tables["hungry"].discard(cid)
            tables["sleepy"].discard(cid)
            tables["dead"].add(cid)
    return tables

# Compare with the live simulator
live_tables = {
    "hungry": set(world.hungry.tolist()),
    "sleepy": set(world.sleepy.tolist()),
    "dead":   set(world.dead.tolist()),
}
assert replay(events) == live_tables
```

If the assertion holds, the events captured every transition. The event log is now the canonical state; the in-memory tables are the *projection*. Snapshots of the tables become a performance optimisation (reading the snapshot is faster than replaying from t=0); the *truth* is the log.

This is exactly the architecture of every event-sourced system, every database WAL, every blockchain.

## Exercise 8 — The crossover, on your machine (stretch)

Finer sweep between 70% and 95%:

```python
# add to the SHARES list in alive_fraction.py:
SHARES = [0.01, 0.10, 0.50, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]
```

Expected pattern: presence and mask cross somewhere between 75% and 90% on most modern machines. The exact crossover varies by:

- **Cache size** — bigger L2/L3 means the bool mask stays warm at higher fractions.
- **Memory bandwidth** — more bandwidth helps the mask version (which reads more bytes).
- **Branch predictor quality** — modern predictors handle the regular branches in the bool sum well; older CPUs were worse at it.

The point of the exercise is *not* to memorise a number. The point is that the right layout depends on your alive-fraction and your hardware. Measure, then choose. Defaulting to presence (the chapter's stance) is right for transient state; defaulting to bool masks is right for near-universal state. Both happen to be correct on a wide range of hardware, just at different fractions.
