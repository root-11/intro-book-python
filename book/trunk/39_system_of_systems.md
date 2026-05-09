# 39 — System of systems

> *Concept node: see the [DAG](../../concepts/dag.md) and [glossary entry 39](../../concepts/glossary.md#39--system-of-systems).*

The trunk so far has assumed every system runs every tick and completes within the tick budget. That covers most of what the simulator does — motion, EBP dispatch, cleanup, persistence — and the surrounding chapters earned the assumption. But the assumption is not universal. Practical simulators have at least three classes of work that do not fit it.

- **Optimisation.** A scheduler choosing which tasks each warehouse robot should take next. A combat AI choosing a counter-strategy. A constraint solver finding a feasible plan. These can take seconds or minutes; they cannot fit in a 33 ms tick.
- **Search.** A path-finder over a large map. A neighbour query in a million-creature world. Even with [§28](28_sort_for_locality.md)'s spatial sort, some searches genuinely take longer than one tick can afford.
- **Out-of-process work.** A game AI evolving its strategy in a separate process. A pricing model running on a remote server. A precomputation handed off to a worker pool. The simulator never blocks waiting; results arrive when they arrive.

This chapter names the three patterns that cover these cases without breaking any of the trunk's previous rules. They are not new architecture. They are the trunk's existing rules, applied to a wider set of cadences.

The unifying principle: **a system has a cadence, and the cadence does not have to be one tick.** A system can run every tick (motion). It can run every N ticks (the spatial sort that [§28](28_sort_for_locality.md) re-runs every 50 frames). It can have a *deadline* and return its best current answer when the deadline arrives. It can be *suspended and resumed* across ticks, with its progress part of its state. It can be *out-of-loop* entirely, communicating with the simulator only through the queue from [§35](35_boundary_is_the_queue.md). The DAG generalises naturally: edges still represent dependencies, but some dependencies wait for promises rather than synchronous returns.

## Anytime algorithms

An *anytime* algorithm produces a valid answer at any time after it has started. The longer it runs, the better the answer. Monte Carlo Tree Search, simulated annealing, evolutionary algorithms, branch-and-bound, CP-SAT — all are anytime. They have a common shape: maintain a *best so far*; refine it as long as time permits; return *best so far* when the budget runs out.

```python
def plan_route(world: "World", deadline: float) -> Route:
    """Returns the best route found before `deadline` (a perf_counter() value)."""
    best = greedy_route(world)
    while time.perf_counter() < deadline:
        candidate = improve(best, world)
        if score(candidate) > score(best):
            best = candidate
    return best
```

The deadline is the budget. The algorithm respects it. Quality is a function of how much time was available — at 5 ms it is mediocre but valid; at 50 ms it is good; at 500 ms it is near-optimal. The simulator can give it whatever budget the tick allows and never get blocked.

This is [§4](04_cost_and_budget.md) applied to a long computation: the budget is named explicitly, and the algorithm honours it. The student who has internalised the budget calculus already knows how to design these algorithms; the only new vocabulary is the *anytime* contract.

## Time-sliced computation

Some work cannot be made anytime — there is no "best partial answer" until the work is complete. A spatial search that has examined 20% of the cells has a 20% chance of having found the answer; otherwise it has nothing useful to report. For these, the pattern is *time-slicing*: divide the work across many ticks, with the system's *progress* as part of its persistent state.

```python
@dataclass
class SpatialSearch:
    target_x: float
    target_y: float
    cursor: int = 0                    # next cell index to examine
    best_id: int = -1                  # best candidate so far
    best_dist: float = float("inf")

    def step(self, world: "World", max_cells: int) -> bool:
        """Examine up to `max_cells` cells. Return True when complete."""
        end = min(self.cursor + max_cells, len(world.cells))
        for cell_idx in range(self.cursor, end):
            for cid in world.cells[cell_idx]:
                d = (world.pos_x[cid] - self.target_x) ** 2 + \
                    (world.pos_y[cid] - self.target_y) ** 2
                if d < self.best_dist:
                    self.best_id = cid
                    self.best_dist = d
        self.cursor = end
        return self.cursor >= len(world.cells)
```

Each call examines `max_cells` cells. The simulator runs `step` every tick (or every N ticks); progress accumulates in `cursor` and the best-so-far fields; when `cursor` reaches the end, the search is complete and the result is delivered. From the simulator's perspective, the search is one system that takes its budget every tick until done.

This is [§15](15_state_changes_between_ticks.md) applied to a long computation: **the system's state at tick start includes its in-progress work.** The buffering rule that lets every system see consistent input also lets a system pick up where it left off.

## Out-of-loop computation

For work that is genuinely too large for *any* tick budget — a game AI re-planning its grand strategy, an offline machine-learning model, a remote optimisation service — the pattern is *out-of-loop*: the work runs in a separate process or machine, completely outside the simulator's tick. The simulator never blocks. When the work completes, its result enters the simulator through the input queue ([§35](35_boundary_is_the_queue.md)) like any other input event.

```python
# Out-of-loop, in a worker process:
def ai_planner_worker(snapshot_q, result_q):
    while True:
        snapshot = snapshot_q.get()
        if snapshot is None:
            break
        strategy = compute_counter_strategy(snapshot)   # may take seconds
        result_q.put(("strategy_update", strategy))

# Inside the simulator's tick:
def dispatch_ai(world, snapshot_q):
    if world.tick % 30 == 0:                            # every second at 30 Hz
        try:
            snapshot_q.put_nowait(snapshot_of(world))
        except queue.Full:
            pass                                         # last snapshot still in flight
```

The simulator dispatches a snapshot every second; the AI process chews on it; the strategy update lands in the input queue some time later. The strategy might be three ticks late, or three seconds late — the simulator does not know and does not care. The result is one more input event; the queue mechanism is the same.

This is [§35](35_boundary_is_the_queue.md) applied to a long computation: anything that crosses the boundary takes its own time, and the queue absorbs the latency. The discipline is not to wait — **never block the tick on an out-of-loop result.**

## Hierarchical scheduling

Production simulators usually combine these patterns. Game engines run physics at 60 Hz (every-tick), AI at 5 Hz (every-12-ticks), save-game at 0.1 Hz (every-300-ticks), and a strategic planner out-of-loop on a worker. Industrial control loops run inner loops at 1 kHz and outer loops at 10 Hz. The DAG generalises: each system is annotated with its cadence; the scheduler runs each according to its frequency or trigger; the result is a *system of systems* — one architecture, many cadences.

In Python the cadence dispatcher is one function:

```python
def schedule_for_tick(systems: list["System"], tick: int):
    return [s for s in systems if tick % s.period_ticks == 0]
```

Combined with §32's ventilator, this gives you a tick whose work-shape varies *by design* — motion runs every tick, the spatial sort runs every 50, AI dispatch runs every 30, snapshot runs every 1000. The DAG-as-array adapts in the same way it does for workload heterogeneity.

## Scale up before scaling out

> [!NOTE]
> The natural next question after "out-of-loop computation" is "what about across machines?" — splitting the simulator across nodes, with one machine running physics, another running AI, another running visualisation. **The default answer is no.** A network round-trip between machines costs ~5 ms (data centre) to ~100 ms (internet). For a 30 Hz tick (33 ms budget), a single network hop eats 15% of the budget at the best case and the entire tick at typical internet latencies. Modern boxes are large — server CPUs ship with 64-128 cores, terabytes of RAM, multi-channel DDR5. **It is almost always cheaper to rent a larger box than to coordinate many smaller ones.** Distribute only when one box genuinely cannot hold the workload, and accept that distribution forces architectural changes (eventual consistency, network failure handling, deployment complexity) that single-machine architectures do not need. The out-of-loop pattern in this chapter handles a *separate process on the same machine*; that is a different decision than *a separate machine across the network*. See Tristan Hume's "Production Twitter on One Machine" for a careful version of this argument applied to a famously distributed workload.

## Closing Part 9

The chapter is constructive: it names the three patterns and shows where each fits the simulator's existing structure. The next phase, *Discipline*, addresses what comes after: how to keep the architecture working as it ages, as people leave, as requirements change. *Making it work* is this chapter; *keeping it working* is the four chapters that follow.

## Exercises

1. **Audit cadence.** For each system in your simulator, name its cadence. Most are "every tick"; the ones that are not are candidates for the patterns in this chapter. Note any system whose work is currently capped or skipped because it would exceed the budget — these are unmet needs the patterns can serve.
2. **Anytime path-finder.** Implement `plan_route(world, deadline)` for one creature. The function returns the best path found within the deadline. With a 5 ms deadline, time how good the answers are; with 50 ms, how much better. Plot quality vs deadline.
3. **Time-sliced spatial search.** Implement `SpatialSearch` and `step` as in the prose. Run it across multiple ticks, advancing the cursor by a budget-bounded `max_cells` each tick. Verify the result is identical to a single-pass search done in one go.
4. **Out-of-loop AI.** Spawn a worker process via `multiprocessing.Process` that receives world snapshots through a `multiprocessing.Queue` and returns strategy updates through another. Dispatch a snapshot every second; let the worker take 5 seconds; observe that the simulator's tick rate is unaffected and the strategy update lands in the input queue when ready.
5. **Mixed cadence.** Run your simulator with motion at every tick, sort-for-locality every 50 ticks, snapshot every 1000 ticks, and a (mock) AI process updating strategy out-of-loop. Verify that determinism still holds: same seed plus same input queue produces identical hashes after 1000 ticks (per [§16](16_determinism_by_order.md) and [§34](34_order_is_the_contract.md)).
6. **The scale-up arithmetic.** For your simulator's expected workload at full scale, compute the per-tick budget and the working set. Does it fit in one modern box (1 TB RAM, 128 cores, multi-channel DDR5)? If yes, you do not need distributed scaling. If no, you have a real reason to look at it.
7. *(stretch)* **Anytime under varying budget.** Modify the path-finder so its caller passes the *remaining* tick budget each time. Some ticks have plenty of budget; some have very little. The path-finder still returns a valid answer in every case, and the answers improve when the budget allows. Plot quality over time as the simulator runs.

Reference notes in [39_system_of_systems_solutions.md](39_system_of_systems_solutions.md).

## What's next

[§40 — Mechanism vs policy](40_mechanism_vs_policy.md) opens *Discipline*: the rules that hold the architecture together over time. Where this chapter was about *making* the system work for problems that don't fit the standard tick, the next four chapters are about *keeping* it working as it ages.
