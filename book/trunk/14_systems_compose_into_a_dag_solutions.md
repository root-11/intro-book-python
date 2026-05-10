# Solutions: 14 — Systems compose into a DAG

## Exercise 1 — Draw the DAG

Reading each system's read-set and write-set:

```
food_spawn       reads {}                                         writes {food}
motion           reads {vel_x, vel_y, food}                       writes {pos_x, pos_y}
next_event       reads {pos_x, pos_y, food}                       writes {pending_event}
apply_eat        reads {pending_event}                            writes {energy_delta}
apply_reproduce  reads {pending_event}                            writes {to_insert}
apply_starve     reads {pending_event}                            writes {to_remove}
cleanup          reads {to_remove, to_insert, energy_delta}       writes {next_state}
inspect          reads {pos_x, pos_y, energy, food, ids}          writes {}
```

Edges (writer → reader):

```
food_spawn → motion             (food)
food_spawn → next_event         (food)
food_spawn → inspect            (food)
motion     → next_event         (pos_x, pos_y)
motion     → inspect            (pos_x, pos_y)
next_event → apply_eat          (pending_event)
next_event → apply_reproduce    (pending_event)
next_event → apply_starve       (pending_event)
apply_eat       → cleanup       (energy_delta)
apply_reproduce → cleanup       (to_insert)
apply_starve    → cleanup       (to_remove)
```

This matches the chapter's diagram. The three appliers form a "fan-out"; `cleanup` is the "fan-in" that consumes their outputs.

## Exercise 2 — Spot the cycle

If `apply_starve` writes `food` (returning fuel when a creature dies), the chain becomes:

```
food_spawn → next_event → apply_starve → food_spawn? (already ran this tick!)
```

`food_spawn` writes `food`; `apply_starve` reads `pending_event` (from `next_event`, which reads `food` from `food_spawn`); `apply_starve` writes `food` — but `food_spawn` already wrote `food` earlier this tick. The cycle is:

```
food_spawn → next_event → apply_starve → food_spawn   (back-edge: both write `food`)
```

A cycle of writers to the same column is the same-tick contradiction the chapter warns against. **Break it by buffering**: `apply_starve` writes to a `food_returns` buffer; `food_spawn` next tick reads `food_returns` and incorporates it into the new `food` table. The cycle becomes a tick boundary — the [§15 mutations buffer](15_state_changes_between_ticks.md) discipline.

## Exercise 3 — Topological sort by hand

```
A writes X
B reads X, writes Y
C reads X, writes Z
D reads Y and Z, writes W
```

Dependencies:

- B depends on A (X)
- C depends on A (X)
- D depends on B (Y) and C (Z)

**Parallelism**: B and C have the same predecessor (A) and disjoint write-sets (Y vs Z). They can run in parallel.

**Valid execution orders**:

- A, B, C, D
- A, C, B, D
- A, {B || C}, D       (B and C concurrent)

All three are correct; the schedule chooses one. Multiple valid sorts is the norm — any sort respecting the edges is correct, and the DAG itself does not pick.

## Exercise 4 — Topological sort in Python (Kahn's algorithm)

```python
def topo_sort(systems: list[tuple[str, set[str], set[str]]]) -> list[str]:
    """Kahn's algorithm. systems = [(name, read_set, write_set), ...]"""
    writers: dict[str, set[str]] = {}
    for name, _, ws in systems:
        for t in ws:
            writers.setdefault(t, set()).add(name)

    edges:  dict[str, set[str]] = {name: set() for name, _, _ in systems}
    in_deg: dict[str, int]      = {name: 0    for name, _, _ in systems}

    for name, rs, _ in systems:
        for t in rs:
            for w in writers.get(t, ()):
                if w != name and name not in edges[w]:
                    edges[w].add(name)
                    in_deg[name] += 1

    queue = sorted(n for n, d in in_deg.items() if d == 0)
    order: list[str] = []
    while queue:
        queue.sort()                                  # deterministic across runs
        n = queue.pop(0)
        order.append(n)
        for m in sorted(edges[n]):
            in_deg[m] -= 1
            if in_deg[m] == 0:
                queue.append(m)

    if len(order) != len(systems):
        raise ValueError("cycle in DAG")
    return order


# Apply to the sim DAG (with cleanup writing to a buffer to break the cycle from §2)
sim = [
    ("food_spawn",     set(),                                     {"food"}),
    ("motion",         {"vel_x","vel_y","food"},                  {"pos_x","pos_y"}),
    ("next_event",     {"pos_x","pos_y","food"},                  {"pending_event"}),
    ("apply_eat",      {"pending_event"},                         {"energy_delta"}),
    ("apply_reproduce",{"pending_event"},                         {"to_insert"}),
    ("apply_starve",   {"pending_event"},                         {"to_remove"}),
    ("cleanup",        {"to_remove","to_insert","energy_delta"},  {"next_state"}),
    ("inspect",        {"pos_x","pos_y","energy","ids","food"},   set()),
]

print(topo_sort(sim))
# ['food_spawn', 'motion', 'inspect', 'next_event', 'apply_eat',
#  'apply_reproduce', 'apply_starve', 'cleanup']
```

A valid order. `inspect` lands earlier than the chapter diagram suggests because it has no consumers — Kahn's algorithm pulls it as soon as its read-set is satisfied. Both placements (right after `motion` or right at the end) are correct topological sorts.

For exercise 3:

```python
sys2 = [("A", set(), {"X"}),
        ("B", {"X"}, {"Y"}),
        ("C", {"X"}, {"Z"}),
        ("D", {"Y","Z"}, {"W"})]
print(topo_sort(sys2))     # ['A', 'B', 'C', 'D']
```

## Exercise 5 — Compose two systems

```python
import numpy as np

class World:
    def __init__(self, n):
        rng = np.random.default_rng(0)
        self.pos_x = rng.uniform(0, 10, n).astype(np.float32)
        self.pos_y = rng.uniform(0, 10, n).astype(np.float32)
        self.vel_x = rng.uniform(-1, 1, n).astype(np.float32)
        self.vel_y = rng.uniform(-1, 1, n).astype(np.float32)
        self.pending_event = []          # list of (timestamp, kind, idx)


def motion(w: World, dt: float) -> None:
    w.pos_x += w.vel_x * dt
    w.pos_y += w.vel_y * dt


def next_event(w: World) -> None:
    w.pending_event.clear()
    # toy: an event for whichever creature is closest to (0, 0)
    d2 = w.pos_x ** 2 + w.pos_y ** 2
    i  = int(np.argmin(d2))
    w.pending_event.append((float(d2[i]), "closest", i))


def tick(w: World, dt: float) -> None:
    motion(w, dt)
    next_event(w)


w = World(100)
tick(w, 1.0 / 30.0)
print(w.pending_event)        # one event per tick
```

The tick is two function calls in topological order. The DAG is two nodes, one edge (`motion → next_event` via `pos_x`/`pos_y`).

## Exercise 6 — Add `cleanup`

```python
def cleanup(w: World) -> None:
    # toy: drop the closest-to-origin creature (an "eaten" event)
    if w.pending_event:
        _, _, i = w.pending_event[0]
        keep = np.ones(len(w.pos_x), dtype=bool)
        keep[i] = False
        w.pos_x = w.pos_x[keep]
        w.pos_y = w.pos_y[keep]
        w.vel_x = w.vel_x[keep]
        w.vel_y = w.vel_y[keep]


def tick(w: World, dt: float) -> None:
    motion(w, dt)
    next_event(w)
    cleanup(w)


w = World(100)
for _ in range(10):
    tick(w, 1.0 / 30.0)
print(f"after 10 ticks: {len(w.pos_x)} creatures left")    # 90
```

Three function calls, top to bottom in dependency order. Adding a fourth system means writing one line and re-running `topo_sort` if the order is non-trivial. There is no `register()`, no `subscribe()`. *The sequence is the program; the program is the sequence.*

## Exercise 7 — The wrong way: an observer

```python
class EventBus:
    def __init__(self):
        self.subs: dict[str, list] = {}
    def subscribe(self, event, handler):
        self.subs.setdefault(event, []).append(handler)
    def fire(self, event, *args, **kwargs):
        for h in self.subs.get(event, []):
            h(*args, **kwargs)

bus = EventBus()
bus.subscribe("tick", motion)
bus.subscribe("tick", next_event)
bus.subscribe("tick", cleanup)

w = World(100)
bus.fire("tick", w, 1.0 / 30.0)        # works — but only because we registered in order
```

Three subtle problems with this version:

1. **Order is implicit in registration order.** Swap the two `subscribe` lines for `next_event` and `motion` — the program runs without error, with stale data. There is no signal that the order is wrong.
2. **A new subscriber inserted at runtime can change the order silently.** Some plugin loads at startup, calls `bus.subscribe("tick", validate_invariants)`, and inserts itself in the middle. The loop now runs in a different order; whether that's correct depends entirely on the plugin's read/write set, which the bus doesn't know.
3. **Reading the program is harder.** To know what `bus.fire("tick", ...)` does, you must find every `bus.subscribe("tick", ...)` call across the entire codebase, in import order. Compare to `def tick(w, dt): motion(w, dt); next_event(w); cleanup(w)` — three lines, locally readable, ordering visible.

The function-call form tells you what runs when. The bus form tells you what *can* run when. The DAG-explicit version is the one that can be reasoned about, parallelised, tested, and trusted.

## Exercise 8 — A query planner (stretch)

Take five SQL queries and decompose into relational-algebra operators:

```sql
-- Query 1: "active users by country, top 10"
SELECT country, COUNT(*) AS n FROM users WHERE active = TRUE
GROUP BY country ORDER BY n DESC LIMIT 10;
```

Plan:
```
LIMIT(10,
  SORT(n DESC,
    AGGREGATE(GROUP BY country, COUNT(*),
      FILTER(active = TRUE, SCAN(users)))))
```

Each level is a *system* in the chapter's sense:

- `SCAN`         reads the underlying table, writes a stream of rows
- `FILTER`       reads the stream + predicate, writes a filtered stream
- `AGGREGATE`    reads the stream, writes grouped rows
- `SORT`         reads grouped rows, writes ordered rows
- `LIMIT`        reads ordered rows, writes prefix

Each operator declares its read-set (the input stream) and write-set (the output stream); the plan is a topo-sorted DAG. Database optimisers explore alternative plans (a different join order, an index scan instead of a full scan), pick the cheapest, and execute.

A simulator does the same thing every tick, but with the plan fixed at design time rather than chosen by an optimiser. Students who write five small plans by hand notice that a tick-loop and a query plan are the same shape: a DAG of small operators consuming and producing tables.
