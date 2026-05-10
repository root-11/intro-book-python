# Solutions: 34 — Order is the contract

## Exercise 1 — Build the schedule

```python
def tick(world, dt, scheduler):
    # Phase 1: serial (just one system)
    next_event(world)

    # Phase 2: parallel — three appliers, disjoint write-sets
    scheduler.run_phase([
        (apply_eat, world),
        (apply_reproduce, world),
        (apply_starve, world),
    ])                                            # waits for ALL three before returning

    # Phase 3: serial
    cleanup(world)

    # Phase 4: serial
    inspect(world)
```

The barrier is `scheduler.run_phase(...)`: it does not return until every system in the phase has completed. `cleanup` therefore cannot start before the three appliers all finish. The schedule is the document; `run_phase` is the enforcement.

## Exercise 2 — Test for determinism

```python
def hash_world(world) -> str:
    import hashlib
    h = hashlib.blake2b(digest_size=16)
    for col in (world.pos_x, world.pos_y, world.vel_x, world.vel_y,
                world.energy, world.id):
        h.update(col[: world.n_active].tobytes())
    return h.hexdigest()

a = run_simulator(seed=42, ticks=100)
b = run_simulator(seed=42, ticks=100)
assert hash_world(a) == hash_world(b)
```

The parallel ticks must produce a bit-identical world. If the assertion holds, the schedule is correct; the parallelism inside each phase is order-independent (disjoint write-sets), and the barriers between phases enforce the order across phases.

If the assertion fails, the next step is exercise 5's bisection — find which phase first introduces nondeterminism.

## Exercise 3 — Break the contract

```python
# anti-pattern: bad! cleanup races with apply_starve
def tick_broken(world, dt, scheduler):
    next_event(world)
    scheduler.run_phase_async([                  # does NOT wait
        (apply_eat, world),
        (apply_reproduce, world),
        (apply_starve, world),
    ])
    cleanup(world)                               # starts before phase 2 acks
    inspect(world)
```

Result on two runs:

```
run 1: hash = abc123...
run 2: hash = def456...
```

Sometimes the runs agree (if the appliers happen to finish before `cleanup` reads), sometimes they don't. The non-determinism is a *race*, and races present worst at the wrong time — they pass in CI, fail in production, then pass again when you go to debug. The fix is to keep the barrier. The intermittency is the cost of skipping it.

## Exercise 4 — Find your phase boundaries

For the §0 simulator's eight systems:

```
DAG:
  food_spawn → motion → next_event
  next_event → apply_eat, apply_reproduce, apply_starve   (fan-out)
  apply_eat, apply_reproduce, apply_starve → cleanup        (fan-in)
  cleanup → inspect

Phases (level-grouped):
  phase 0: {food_spawn}                          # 1 task
  phase 1: {motion}                              # 1 task
  phase 2: {next_event}                          # 1 task
  phase 3: {apply_eat, apply_reproduce, apply_starve}  # 3 tasks in parallel
  phase 4: {cleanup}                             # 1 task
  phase 5: {inspect}                             # 1 task
```

Each phase boundary is a barrier. The simulator's parallelism opportunity is phase 3 — three workers can run the three appliers. The other phases are serial (one task each).

For a wider simulator with more independent systems, more phases would have multiple tasks. The scheduler (exercise 7) is the algorithm that finds these.

## Exercise 5 — The asyncio trap, hands-on

```python
import asyncio

async def tick_async(world, dt):
    await asyncio.gather(
        motion(world, dt),
        next_event(world),
        apply_eat(world),
        apply_reproduce(world),
        apply_starve(world),
        cleanup(world),
    )

asyncio.run(tick_async(world, 1/30))
```

What happens: `asyncio.gather` schedules all six coroutines. Each runs until it hits an `await` (sleep, I/O, etc.). Since these are pure-Python CPU functions, none of them yield — whichever was scheduled first runs to completion, then the next, etc. The *order* is whatever `gather` happens to emit them in, which is *not* the DAG order.

Two runs of the simulator: `motion` happens to run first in run A and `apply_eat` happens to run first in run B (because the asyncio scheduler is allowed to choose). The world hashes diverge.

`gather` is the wrong shape for CPU work with dependencies. It is correct for *I/O* concurrency (request multiple URLs in parallel) where the order doesn't matter and waits are real. For CPU systems with a DAG, a *scheduler* (the ventilator) is the right tool.

## Exercise 6 — Cross-machine determinism

Set up two machines (e.g. your laptop and a server, or two cores in a CI matrix). Run the same simulator with the same seed. Hash the world after N ticks. Compare hashes.

If they diverge, candidates to investigate:

- **`PYTHONHASHSEED`**: set to `0` on both machines (or to the same explicit number) before launching. Without this, set iteration order differs across machines.
- **Wall clock**: any system that reads `time.perf_counter()` inside its body. Refactor to take `dt` from main.
- **Unseeded RNG**: any `random.random()` reading global state.
- **Hardware float behaviour**: some operations (e.g. `np.exp`, transcendentals) are not bit-identical across glibc versions. For most simulators this is below the noise; for high-precision physics, pin to a specific glibc + numpy version.
- **CPU determinism**: x86 floating-point should be bit-identical for the operations the book uses (basic +/-/*//, sum, comparisons). Watch out for `--fast-math`-style compiler flags in third-party libraries.

A simulator that is bit-identical across two machines is genuinely deterministic. Most simulators take some work to reach this; the work pays back in every test, every replay, every reproducible bug report.

## Exercise 7 — A minimal scheduler (stretch)

```python
def topo_phases(systems: list[tuple[str, set[str], set[str]]]) -> list[list[str]]:
    """Return systems grouped by DAG level — each list is a parallel phase."""
    writers: dict[str, set[str]] = {}
    for name, _, ws in systems:
        for t in ws:
            writers.setdefault(t, set()).add(name)

    edges:  dict[str, set[str]] = {n: set() for n, _, _ in systems}
    in_deg: dict[str, int]      = {n: 0 for n, _, _ in systems}
    for name, rs, _ in systems:
        for t in rs:
            for w in writers.get(t, ()):
                if w != name and name not in edges[w]:
                    edges[w].add(name)
                    in_deg[name] += 1

    phases = []
    current = sorted(n for n, d in in_deg.items() if d == 0)
    while current:
        phases.append(current)
        next_phase = []
        for n in current:
            for m in sorted(edges[n]):
                in_deg[m] -= 1
                if in_deg[m] == 0:
                    next_phase.append(m)
        current = sorted(next_phase)

    if sum(len(p) for p in phases) != len(systems):
        raise ValueError("cycle in DAG")
    return phases


systems = [
    ("food_spawn",     set(),              {"food"}),
    ("motion",         {"vel_x", "food"},  {"pos_x"}),
    ("next_event",     {"pos_x", "food"},  {"pending_event"}),
    ("apply_eat",      {"pending_event"},  {"energy_delta"}),
    ("apply_reproduce",{"pending_event"},  {"to_insert"}),
    ("apply_starve",   {"pending_event"},  {"to_remove"}),
    ("cleanup",        {"to_remove", "to_insert", "energy_delta"}, {"next_state"}),
    ("inspect",        {"pos_x"},          set()),
]
for i, phase in enumerate(topo_phases(systems)):
    print(f"phase {i}: {phase}")
```

```
phase 0: ['food_spawn']
phase 1: ['motion']
phase 2: ['inspect', 'next_event']                      # both can run after motion
phase 3: ['apply_eat', 'apply_reproduce', 'apply_starve']
phase 4: ['cleanup']
```

The phases drop out of Kahn's algorithm with a small tweak — instead of pulling one node per iteration, pull *all* nodes with `in_deg == 0` as a single phase. Each phase is the set of systems that can run in parallel without violating any dependency.

This is the scheduler. It is ~30 lines. Every ECS engine has a version of it; the structure is identical across languages.
