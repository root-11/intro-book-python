# Solutions: 39 — System of systems

## Exercise 1 — Audit cadence

A typical simulator's cadence audit:

| system            | cadence       | pattern        |
|-------------------|---------------|----------------|
| motion            | every tick    | standard       |
| food_spawn        | every tick    | standard       |
| next_event        | every tick    | standard       |
| apply_eat/repro/starve | every tick | standard      |
| cleanup           | every tick    | standard       |
| sort_for_locality | every 50 ticks | periodic      |
| snapshot          | every 1000 ticks | periodic    |
| log_flush         | every 10 ticks | periodic      |
| inspect (debug)   | every tick (debug-on) / never (debug-off) | conditional |
| path_planner      | per-creature, on demand, with deadline | anytime |
| spatial_search    | per-creature, time-sliced | time-sliced |
| strategy_ai       | out-of-loop, ~1 Hz update | out-of-loop |

The cadences that aren't "every tick" are candidates for one of the chapter's three patterns. Any system that's currently capped (e.g. "only the first 100 path-finds per tick") is a candidate for the anytime pattern.

## Exercise 2 — Anytime path-finder

```python
import time, random, numpy as np

def greedy_route(start, goal, obstacles):
    """Trivial baseline: take a straight line, ignoring obstacles."""
    return [start, goal]

def improve(route, obstacles):
    """Local search: try perturbing a random waypoint."""
    if len(route) < 3:
        # add a waypoint
        mid = ((route[0][0] + route[-1][0]) / 2 + random.uniform(-1, 1),
               (route[0][1] + route[-1][1]) / 2 + random.uniform(-1, 1))
        return [route[0], mid, route[-1]]
    i = random.randint(1, len(route) - 2)
    perturbed = list(route)
    perturbed[i] = (perturbed[i][0] + random.uniform(-0.5, 0.5),
                    perturbed[i][1] + random.uniform(-0.5, 0.5))
    return perturbed

def score(route, obstacles):
    """Lower is better. Penalises length and obstacle collisions."""
    length = sum(((route[i+1][0] - route[i][0])**2 + (route[i+1][1] - route[i][1])**2)**0.5
                 for i in range(len(route)-1))
    collisions = sum(1 for waypoint in route for ox, oy, r in obstacles
                     if (waypoint[0]-ox)**2 + (waypoint[1]-oy)**2 < r**2)
    return length + collisions * 100

def plan_route(start, goal, obstacles, deadline: float):
    best = greedy_route(start, goal, obstacles)
    best_score = score(best, obstacles)
    while time.perf_counter() < deadline:
        candidate = improve(best, obstacles)
        s = score(candidate, obstacles)
        if s < best_score:
            best = candidate
            best_score = s
    return best, best_score

# At 5 ms deadline
deadline = time.perf_counter() + 0.005
r1, s1 = plan_route((0, 0), (10, 10), [(5, 5, 1)], deadline)

# At 50 ms deadline  
deadline = time.perf_counter() + 0.050
r2, s2 = plan_route((0, 0), (10, 10), [(5, 5, 1)], deadline)

print(f"5ms:  {len(r1)} waypoints, score={s1:.2f}")
print(f"50ms: {len(r2)} waypoints, score={s2:.2f}")
```

Quality improves with deadline. Plot score-vs-deadline by repeating at 1 ms, 5 ms, 10 ms, 50 ms, 100 ms, 500 ms — typically logarithmic improvement (each doubling of time buys roughly the same quality increment).

## Exercise 3 — Time-sliced spatial search

```python
from dataclasses import dataclass
import numpy as np

@dataclass
class SpatialSearch:
    target_x: float
    target_y: float
    cells: list[np.ndarray]
    cursor: int = 0
    best_id: int = -1
    best_dist: float = float("inf")
    done: bool = False

    def step(self, world, max_cells: int):
        end = min(self.cursor + max_cells, len(self.cells))
        for i in range(self.cursor, end):
            for cid in self.cells[i]:
                d2 = (world.pos_x[cid] - self.target_x)**2 + \
                     (world.pos_y[cid] - self.target_y)**2
                if d2 < self.best_dist:
                    self.best_id = int(cid)
                    self.best_dist = float(d2)
        self.cursor = end
        if self.cursor >= len(self.cells):
            self.done = True
        return self.done

# Run a single-pass search
single = SpatialSearch(target_x=5.0, target_y=5.0, cells=world.cells)
single.step(world, max_cells=len(world.cells))

# Run a time-sliced search, 10 cells per tick
sliced = SpatialSearch(target_x=5.0, target_y=5.0, cells=world.cells)
while not sliced.done:
    sliced.step(world, max_cells=10)

assert single.best_id == sliced.best_id, "time-sliced version must match single-pass"
```

The time-sliced result is identical to the single-pass result. The work is the same; the *granularity* differs. The simulator can call `step(max_cells=budget_cells)` every tick with a budget computed from the remaining tick time.

## Exercise 4 — Out-of-loop AI

```python
import multiprocessing, time, queue

def ai_planner_worker(snapshot_q, result_q):
    while True:
        try:
            snapshot = snapshot_q.get(timeout=1.0)
        except queue.Empty:
            continue
        if snapshot is None:
            break
        time.sleep(5.0)                                # simulate 5-second compute
        result_q.put(("strategy_update", "new_strategy_from_snapshot"))

if __name__ == "__main__":
    snapshot_q = multiprocessing.Queue(maxsize=1)
    result_q   = multiprocessing.Queue()
    worker = multiprocessing.Process(target=ai_planner_worker, args=(snapshot_q, result_q))
    worker.start()

    for tick in range(200):                            # ~7 seconds at 30 Hz
        if tick % 30 == 0:                             # every 1 second
            try:
                snapshot_q.put_nowait({"tick": tick})
            except queue.Full:
                pass                                    # AI still working on previous one

        # Check for strategy updates without blocking
        try:
            event = result_q.get_nowait()
            print(f"tick {tick}: received {event}")
        except queue.Empty:
            pass

        time.sleep(1/30)                                # tick

    snapshot_q.put(None)
    worker.join()
```

The simulator's tick continues at 30 Hz. Snapshots dispatch every second. The AI takes 5 seconds; its result arrives in the result queue ~5 seconds late, and the simulator picks it up on the next polling cycle. No blocking. The tick rate is preserved exactly.

This is the architecture for "AI as a side process," "remote pricing service," "GPU model inference" — any work that takes longer than a tick. The queue is the seam.

## Exercise 5 — Mixed cadence

```python
def tick(world, current_tick):
    motion(world)
    food_spawn(world)
    next_event(world)
    # parallel block
    apply_eat(world); apply_reproduce(world); apply_starve(world)
    cleanup(world)
    if current_tick % 50 == 0:
        sort_for_locality(world)
    if current_tick % 1000 == 0:
        snapshot(world, f"snap_{current_tick}.npz")
    # Out-of-loop AI: dispatched separately, results enter via in_queue
```

Run twice with the same seed; hash after 1000 ticks. The hashes must match. The mixed cadences don't break determinism because:

- The cadence (every 50, every 1000) is deterministic given the tick number.
- The out-of-loop AI's result enters via the queue (§35), which is deterministic if the recorded queue matches.

For the AI to be deterministic across runs, the snapshot it processes and the time it takes must match. In practice this means either (a) testing with mocked AI that returns deterministic results based on snapshot content, or (b) accepting that real AI introduces stochasticity at the input queue and treating it as "another input source" rather than part of the simulator's determinism guarantee.

## Exercise 6 — The scale-up arithmetic

Suppose your simulator at full scale needs 1B creatures × 32 bytes/row = 32 GB of state, 30 Hz tick, 16-core parallelism.

Modern boxes:

| spec               | typical 2026 |
|--------------------|--------------|
| RAM                | 64 GB - 1 TB |
| cores              | 16-128       |
| memory channels    | 2-8 (DDR5)   |
| NVMe storage       | 4-30 TB      |

32 GB fits comfortably on any 64 GB box. 16 cores fits any modern desktop. The simulator can run on a single high-end laptop or a mid-range workstation. *No distributed system needed.*

For a workload of 100B creatures (3.2 TB state): now you've left single-machine territory. But the cost of one machine with 4 TB RAM (server class, ~$10K) is much less than the engineering cost of distributing the simulator. Rent or buy the bigger box.

The threshold where distribution becomes mandatory: when *one machine* can't physically hold the workload (~10 TB+ for most cloud providers, even larger on bare metal). Below that, every reasonable problem fits on one box.

## Exercise 7 — Anytime under varying budget (stretch)

```python
def plan_route_with_budget(start, goal, obstacles, remaining_ms: float):
    deadline = time.perf_counter() + remaining_ms / 1000
    return plan_route(start, goal, obstacles, deadline)

# In the tick:
def tick(world, current_tick):
    tick_start = time.perf_counter()
    motion(world); next_event(world); apply_eat(world); ...
    elapsed = (time.perf_counter() - tick_start) * 1000
    remaining_ms = max(1.0, 33.0 - elapsed)            # 30 Hz budget = 33 ms
    
    # Spend remaining budget on path planning
    if world.creatures_needing_paths:
        for creature in world.creatures_needing_paths:
            path = plan_route_with_budget(
                creature.pos, creature.goal, world.obstacles, remaining_ms)
            creature.path = path
            remaining_ms = max(1.0, 33.0 - (time.perf_counter() - tick_start) * 1000)
```

Some ticks have plenty of budget left (the rest of the work was cheap); some have very little (a heavy cleanup happened). The path-finder takes whatever's available.

Plot the path quality over a 1000-tick run. The line is jittery — quality varies tick-to-tick with the available budget — but the *trend* is positive: even the worst tick still produces a valid path; better ticks produce better paths; the simulator never blocks.

This is the production pattern for AI in real-time systems: *spend whatever budget you have, never block, never miss a deadline*. The simulator's overall frame rate is preserved; AI quality is a function of the time it gets.
