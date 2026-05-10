# Solutions: 35 — The boundary is the queue

## Exercise 1 — Build the queues

```python
import numpy as np

class Queue:
    """A bounded SoA queue with parallel columns and a single n_active counter."""
    def __init__(self, capacity: int, schema: dict[str, np.dtype]):
        self.capacity = capacity
        self.columns = {name: np.zeros(capacity, dtype=dt) for name, dt in schema.items()}
        self.n_active = 0

    def push(self, **fields):
        i = self.n_active
        for name, value in fields.items():
            self.columns[name][i] = value
        self.n_active += 1

    def drain(self) -> dict:
        """Return a snapshot of every column up to n_active, then reset."""
        snapshot = {name: col[: self.n_active].copy() for name, col in self.columns.items()}
        self.n_active = 0
        return snapshot

# in the world
world.in_queue  = Queue(capacity=10_000, schema={
    "tick": np.uint32, "kind": np.uint8, "creature_id": np.uint32, "value": np.float32
})
world.out_queue = Queue(capacity=10_000, schema={
    "tick": np.uint32, "event": np.uint8, "id": np.uint32, "data": np.float32
})
```

The in-queue is filled by the tick driver *before* the tick runs. The out-queue is filled by systems *during* the tick. Both are drained at the tick boundary (the in-queue by the systems that consume it; the out-queue by the I/O layer that ships events outward).

## Exercise 2 — Refactor a system that reads time

```python
# Before
def schedule_event_bad(events):
    now = time.perf_counter()                # non-deterministic
    events.append((now + 0.5, "fire"))

# After
def schedule_event(events, current_time: float):
    events.append((current_time + 0.5, "fire"))

# The tick driver reads the clock once, passes it down
def run_tick(world):
    current_time = time.perf_counter()       # the ONLY clock read
    tick(world, current_time, dt=1.0/30.0)
```

The system is now a pure function of its inputs. The tick driver is the seam where the wall clock enters; everything inside the tick is deterministic.

## Exercise 3 — Refactor a system that prints

```python
# Before — print() from inside a system
def apply_starve_bad(creatures):
    for c in creatures:
        if c.energy <= 0:
            print(f"creature {c.id} starved")     # ← side effect; non-deterministic

# After — append to the out-queue
def apply_starve(world: World, out_queue: Queue):
    starvers = np.where(world.energy <= 0)[0]
    for s in starvers:
        out_queue.push(tick=world.current_tick, event=EVENT_STARVED,
                       id=world.id[s], data=0.0)

# The tick driver flushes the out-queue after the tick
def run_tick(world):
    tick(world)
    events = world.out_queue.drain()
    for e in events:                              # tick-driver-level I/O
        print(f"tick {e.tick}: creature {e.id} starved")
```

Logging is now deterministic: the events captured in the queue are bit-identical across two runs with the same seed. The actual writing-to-stdout is a separate concern handled by the tick driver, which is allowed to do I/O because it is *outside* the tick. Tests can assert on `world.out_queue.drain()` without redirecting stdout.

## Exercise 4 — Replay test

```python
import numpy as np

def record_run(seed, n_ticks):
    world = build_world(seed=seed)
    queue_log = []
    for _ in range(n_ticks):
        # feed inputs from a deterministic source
        inputs = generate_inputs(world.current_tick)
        for inp in inputs: world.in_queue.push(**inp)
        queue_log.append(world.in_queue.drain())
        tick(world)
    return world, queue_log

def replay_run(seed, queue_log):
    world = build_world(seed=seed)
    for queued in queue_log:
        for i in range(queued["tick"].size):
            world.in_queue.push(**{name: col[i] for name, col in queued.items()})
        tick(world)
    return world

original, log = record_run(seed=42, n_ticks=100)
replayed = replay_run(seed=42, queue_log=log)

assert hash_world(original) == hash_world(replayed)
```

The two worlds must be bit-identical. If they're not, somewhere a system reads from outside the queue. The queue *is* the input.

## Exercise 5 — Two simulators from one queue

```python
queue_recording = [...]                            # captured once

sim_a = build_world(seed=42)
sim_b = build_world(seed=42)

for queued in queue_recording:
    for sim in (sim_a, sim_b):
        for i in range(queued["tick"].size):
            sim.in_queue.push(**{name: col[i] for name, col in queued.items()})
        tick(sim)

assert hash_world(sim_a) == hash_world(sim_b)
```

Same queue, same seed, same world. The simulators must converge. If they diverge, find the system that reads from outside (exercise 6).

## Exercise 6 — Find every leak

```sh
grep -rEn 'time\.|print|logger|requests|os\.environ|input\(' code/sim/
```

Typical matches (and their fates):

| match | location | fate |
|-------|----------|------|
| `time.perf_counter()` | inside `motion` | refactor: take `dt` as parameter |
| `print(f"...")` | inside `apply_starve` | refactor: append to `out_queue` |
| `os.environ.get("BURN_RATE")` | inside `compute_burn` | refactor: pass `burn_rate` as parameter |
| `logger.info(...)` | inside any system | refactor: queue + tick-driver flush |
| `requests.get(...)` | inside any system | category error: I/O does not belong inside the tick at all; refactor as an out-of-tick task that feeds the in-queue |

Every match is a candidate determinism leak. The disciplined form: every system is a pure function of its parameter list; everything that comes from outside enters via the in-queue.

## Exercise 7 — Audit an open-source simulator (stretch)

Open a simulator from `mesa` (Mesa-ABM is one of Python's prominent ABM frameworks). Look at a `step()` method:

- **`self.random.random()`**: Mesa wraps Python's `random` in a per-model instance. *Deterministic given a seed.* Good.
- **`self.schedule.time`**: Mesa's scheduler keeps its own time. *Deterministic given the schedule.* Good.
- **`time.time()` for performance metrics**: usually inside `__main__` infrastructure, not the model. Good.
- **`self.datacollector.collect(self)`**: this is the *out-queue* in Mesa's vocabulary. Mesa explicitly separates "model step" from "data collection." Same pattern.

Mesa is actually fairly disciplined about the boundary. Many less mature ABM/simulation frameworks aren't — a common pattern is `logger.info(...)` calls scattered through agent step methods, plus `os.environ.get(...)` reads of configuration. Auditing for these is what makes a simulator into a *reproducible* simulator.

The audit is itself a system. Run it once before declaring the simulator deterministic; run it as a CI check on every PR that touches the simulator.
