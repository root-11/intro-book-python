# 34 — Order is the contract

> *Concept node: see the [DAG](../../concepts/dag.md) and [glossary entry 34](../../concepts/glossary.md#34--order-is-the-contract).*

<p align="center"><img src="../illustrations/monte_carlo.jpg" alt="Monte Carlo simulation — reproducibility is the contract under concurrency" style="max-height: 300px; max-width: 100%;"></p>

§31, §32, and §33 unlocked parallelism. The natural temptation is to run *everything* in parallel — let the OS scheduler decide which system runs when, fan systems out across all available cores, push throughput up. This is wrong.

The system DAG ([§14](14_systems_compose_into_a_dag.md)) is the *contract* for the simulator's behaviour. Two systems with overlapping write-sets must run in a defined order. Two systems on the same DAG level may run in parallel — but they must both *complete* before any system that reads their outputs begins. **Parallelism is allowed inside a phase; it is never allowed across phases.**

The reason is determinism ([§16](16_determinism_by_order.md)). Same inputs + same system order = same outputs. If `apply_eat`, `apply_reproduce`, and `apply_starve` run in undefined order — say, the first one to finish gets to write `to_remove` first — then `cleanup` sees a different `to_remove` ordering on different runs, and the world state at the end of the tick is non-reproducible. Replay breaks. Tests become flaky. Distributed simulation drifts apart.

The schedule looks like:

```text
                  ┌── apply_eat ────┐
                  │                 │
   next_event ────┼── apply_repro ──┼─→ cleanup → inspect
                  │                 │
                  └── apply_starve ─┘
```

`next_event` runs first (its writes are needed by all three appliers). The three appliers run in parallel — their writes are disjoint (each writes to its own section of `to_remove` or its own table, [§31](31_disjoint_writes_parallelize.md)). `cleanup` runs after all three finish, never before any of them. `inspect` runs last.

The schedule is fixed by the DAG. Parallelism happens *within* the structure the DAG permits, not around it.

## Two anti-patterns to name

**The "let the OS decide" anti-pattern.** Fanning every system out as a process and letting them race is fast in the wrong way. Some runs produce one result; some produce another. The bug is intermittent, the cause is hard to find, and "fixing" it with locks reintroduces the costs §31-§33 worked to avoid.

```python
# anti-pattern: bad!
with Pool(processes=8) as pool:
    pool.starmap_async(motion, ...)
    pool.starmap_async(food_spawn, ...)        # runs concurrently with motion
    pool.starmap_async(next_event, ...)        # may finish before motion does
    pool.starmap_async(apply_eat, ...)         # reads pending_event, may see partial
    # ... no waits, no barriers ...
    pool.close()
    pool.join()                                # only barrier; everything raced
```

**The "early start" anti-pattern.** Starting a system before its prerequisites have finished — even if the data "looks ready" — is a bet that the schedule will not change. The bet often pays off in practice, until the day a buffer fills slightly later than usual and the world's state shifts in ways no test caught. Wait for the explicit completion of every prerequisite.

```python
# anti-pattern: bad!
def tick(world):
    motion_future = pool.apply_async(motion, ...)
    next_event(world)                          # starts before motion completes
    apply_eat(world)                           # reads pos, but motion is updating it!
    motion_future.wait()                       # too late; reads are already wrong
```

Python's third anti-shape — and the one most readers will be tempted by — is **`asyncio.gather` over the systems**:

```python
# anti-pattern: bad!
async def tick(world):
    await asyncio.gather(
        motion(world),
        next_event(world),
        apply_eat(world),
        apply_reproduce(world),
        apply_starve(world),
        cleanup(world),
    )
```

This shape *looks* like a scheduler. It is not. `asyncio.gather` runs awaitables to completion in *whatever order they cooperatively yield*, with no notion of dependency between them. The DAG's structure — *cleanup must wait for the appliers, the appliers must wait for next_event* — is invisible to `gather`. The first system to complete, completes; the rest race. Same failure mode as the multiprocessing version, with extra confusion because the surface syntax looks like the right shape.

## The ventilator IS the scheduler

[§32](32_partition_dont_lock.md)'s ventilator model is exactly the scheduler this chapter requires. Re-read the DAG-as-array:

```
phase 1: [1]                     # next_event
phase 2: [1, 2, 3]               # apply_eat, apply_reproduce, apply_starve in parallel
phase 3: [1]                     # cleanup
phase 4: [1]                     # inspect
```

The phases are barriers. Within a phase, work runs in parallel. Between phases, main waits for every worker to ack before bumping the generation. **Phase boundaries enforce the DAG; intra-phase parallelism uses the architecture from §31-§33.** One mechanism, two readings: the parallel schedule *and* the deterministic execution order are the same document.

Most production ECS engines implement exactly this — Bevy's `World::run_schedule`, Unity DOTS's `JobHandle.Complete`, Unreal's Mass Entities scheduler. The pattern is the same as a parallel `make`: build dependencies in order, build independents in parallel, never start a target before its prerequisites have finished.

## Determinism inside the parallel region

A subtler issue: even with phase boundaries respected, the *workers* themselves must produce deterministic output. From [§16](16_determinism_by_order.md), the recipe applies inside each worker:

- **No `random.random()` reading global state.** Each worker holds its own `np.random.default_rng(seed)`, seeded deterministically at startup (e.g. `default_rng(base_seed + my_id)`).
- **No system clock inside a system.** Time is passed as `dt` from main, not read from `time.perf_counter()` inside a worker.
- **Order-dependent reductions are wrong.** A worker that does `sum(arr)` is fine; a worker that does `for x in arr: total += float_func(x)` may produce different bit-level outputs depending on what `arr` happens to contain at that moment if `arr` is shared. Stick to numpy bulk operations for any reduction whose result feeds back into the world.
- **No set iteration.** The §16 set-iteration trap applies inside every worker independently.

The single-writer rule from [§25](25_ownership_of_tables.md) handles the rest: workers only write their own partition, so two workers cannot corrupt each other's bytes regardless of when they happen to run.

## The replay test

A useful test: *can you replay a tick to bit-identical output?* If yes, your scheduler respects the contract. If no, it does not — somewhere a system runs in undefined order, and the bug will surface in the worst possible debugging window.

The test is concrete:

```python
def replay_test(world_factory, n_ticks: int) -> bool:
    world_a = world_factory(seed=42)
    for _ in range(n_ticks):
        tick(world_a)
    hash_a = hash_world(world_a)

    world_b = world_factory(seed=42)
    for _ in range(n_ticks):
        tick(world_b)
    hash_b = hash_world(world_b)

    return hash_a == hash_b
```

Run it after every change to the simulator. Run it under N=1, N=2, N=4, N=8 workers. **Run it across machines.** If the hash diverges across machines, you have a non-deterministic dependency that one machine resolves one way and another machine resolves the other — almost always a `set` iteration, a wallclock read, or an unseeded RNG.

## Closing Part 7

This rule closes Concurrency. The simulator can now use every core on the machine without sacrificing the determinism that §16 guaranteed. The DAG is both the parallel schedule *and* the deterministic execution order; one document, two readings. The ventilator model implements both.

## Exercises

1. **Build the schedule.** Write a `tick(world, dt)` that runs `next_event`, then a parallel block of the three appliers (using your §32 ventilator pattern), then `cleanup`, then `inspect`. Verify the boundaries: `cleanup` must not start before all three appliers complete.
2. **Test for determinism.** Run the simulator twice with the same seed. Hash the world after 100 ticks. The hashes must be identical even though the appliers ran in parallel.
3. **Break the contract.** Construct a schedule where `cleanup` starts before `apply_starve` finishes (e.g. by skipping the wait-for-acks step in main between phases). Run twice. Hashes should differ — sometimes. The bug's intermittency is the lesson.
4. **Find your phase boundaries.** Sketch your simulator's full DAG from [`code/sim/SPEC.md`](../../code/sim/SPEC.md). Identify each *phase* (set of systems with no transitive dependency on each other). Each phase is a parallel batch; each boundary is a sync.
5. **The asyncio trap, hands-on.** Implement `tick` using `asyncio.gather` over the systems. Run the determinism test. Watch the hash diverge across runs. Note the failure shape: not a crash, just *wrong* answers.
6. **Cross-machine determinism.** If you have access to another machine, run the same simulator with the same seed there. The hashes must match. If they do not, find the difference — `PYTHONHASHSEED`, wall clock, glibc version, hardware float behaviour. Each is a possible source.
7. *(stretch)* **A minimal scheduler.** Write `def topo_phases(systems: list[tuple[str, set[str], set[str]]]) -> list[list[str]]` taking `(name, read_set, write_set)` triples and returning a list of phases (each phase is a list of system names that can run in parallel). Around 30 lines of Python. The scheduler is just a topological sort with level-grouping.

Reference notes in [34_order_is_the_contract_solutions.md](34_order_is_the_contract_solutions.md).

## What's next

You have closed Concurrency. The simulator now runs on multiple cores without losing determinism. The next phase is *I/O & persistence*, starting with [§35 — The boundary is the queue](35_boundary_is_the_queue.md). The simulator is about to begin talking to the world outside its tick.
