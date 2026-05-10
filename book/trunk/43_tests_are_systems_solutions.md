# Solutions: 43 — Tests are systems; TDD from day one

## Exercise 1 — A test as a system

```python
def test_no_creature_moves_too_far(world, max_step: float = 5.0) -> np.ndarray:
    """A read-only system that reports any creature whose move exceeded max_step."""
    dx = world.pos_x[: world.n_active] - world.prev_pos_x[: world.n_active]
    dy = world.pos_y[: world.n_active] - world.prev_pos_y[: world.n_active]
    return np.where(dx*dx + dy*dy > max_step*max_step)[0]

def tick_with_test(world):
    # Save previous positions
    world.prev_pos_x[: world.n_active] = world.pos_x[: world.n_active]
    world.prev_pos_y[: world.n_active] = world.pos_y[: world.n_active]
    # Run normal tick
    motion(world); next_event(world); apply_eat(world); ...
    # Run the test as a system
    suspicious = test_no_creature_moves_too_far(world, max_step=5.0)
    assert suspicious.size == 0, f"creatures {suspicious} teleported"
```

The test fits in the DAG with read-set `pos_x`, `prev_pos_x`, `pos_y`, `prev_pos_y` and empty write-set. It runs after `motion` (which it depends on) and asserts. In production, the system is gated behind a `--test` flag; in CI it runs every tick.

## Exercise 2 — A property test

```python
def property_test_population_bounded(seed: int, ticks: int, factor: float = 2.0):
    world = build_world(seed=seed)
    initial_n = world.n_active
    bound = factor * initial_n

    for t in range(ticks):
        tick(world)
        assert world.n_active <= bound, \
            f"population exploded at tick {t}: {world.n_active} > {bound}"
    return world

# Determinism check: same seed, same outcome
world_a = property_test_population_bounded(seed=0xCAFE, ticks=1000)
world_b = property_test_population_bounded(seed=0xCAFE, ticks=1000)
assert hash_world(world_a) == hash_world(world_b)
```

The property test runs the simulator and asserts an invariant after every tick. If the invariant fails, the assertion identifies the exact tick — the failure is *localised in time*, not just "test failed somewhere in the run."

The determinism check confirms the test itself is reproducible: same seed, same outcome, every run. This is what [§16](16_determinism_by_order.md) guarantees.

## Exercise 3 — A replay test

```python
def replay_test(seed: int, ticks: int):
    # Live run, recording the in-queue
    live = build_world(seed=seed)
    queue_log = []
    for _ in range(ticks):
        inputs = generate_inputs(live.tick)
        for inp in inputs:
            live.in_queue.push(**inp)
        queue_log.append(live.in_queue.drain())
        tick(live)

    # Save the recording
    np.savez("queue_log.npz", **{f"tick_{i}": q for i, q in enumerate(queue_log)})

    # Replay from a fresh simulator
    replayed = build_world(seed=seed)
    data = np.load("queue_log.npz")
    for i in range(ticks):
        recorded = data[f"tick_{i}"]
        for j in range(recorded.size):
            replayed.in_queue.push(...)             # un-pack each event
        tick(replayed)

    assert hash_world(live) == hash_world(replayed), \
        "replay diverged — non-deterministic dependency leaking"
```

The hashes must match. If they don't, somewhere a system reads from outside the queue — the §35 boundary is breached. The replay test is the catch-all for "did we accidentally make this non-deterministic?"

## Exercise 4 — TDD a new system

```python
# Step 1: write the test first
def test_slow_growth_when_high_energy(world):
    """Creatures with energy > 50 should grow more slowly than those with energy <= 50."""
    # Setup
    world.energy[:world.n_active] = np.full(world.n_active, 30.0, dtype=np.float32)
    world.energy[:10] = 80.0                          # first 10 are well-fed
    
    initial_age = world.age[:world.n_active].copy()
    
    # Run the (not-yet-written) system
    apply_slow_growth(world)
    
    delta = world.age[:world.n_active] - initial_age
    # well-fed creatures grow half as fast
    assert (delta[:10] < delta[10:].mean()).all()

# Step 2: minimal implementation
def apply_slow_growth(world):
    fast = world.energy[:world.n_active] <= 50
    slow = world.energy[:world.n_active] > 50
    world.age[:world.n_active][fast] += 1
    world.age[:world.n_active][slow] += 1  # bug! should be slower
    
# Step 3: run the test, see it fail, fix:
def apply_slow_growth_fixed(world):
    fast = world.energy[:world.n_active] <= 50
    slow = world.energy[:world.n_active] > 50
    world.age[:world.n_active][fast] += 1
    world.age[:world.n_active][slow] = world.age[:world.n_active][slow] + 1  # but only every other tick
    # actual implementation depends on the design — half-rate, threshold, etc.
```

The test is written first; the implementation follows. The test catches the bug; the implementation is iterated until the test passes. This is TDD's value: the test is the spec, refined until both the spec and the implementation agree.

For numpy/ECS-style code, TDD especially pays off because:

- The read-set / write-set declarations make tests trivially scoped.
- Pure functions of inputs are trivially testable.
- No mocks: tests set up real numpy arrays and read them.

## Exercise 5 — Read the simlog tests

`.archive/simlog/test_simlog.py` is the production-grade version of "tests as systems." Things to notice:

- **No `mock.patch` calls.** Every test fixture creates real `Simlog` instances, writes real events, and reads real `.npz` output. The simlog's interface is the queue; the queue is the test's input.
- **Property-style tests**: `test_log_round_trip` writes 100K events and verifies every one survives the codebook + write + read cycle. The test is a small simulator: produce events, consume events, assert equality.
- **1:1 line ratio**: 713 lines of tests for ~700 lines of library code. The ratio reflects how much the library depends on getting the contract right. Production code that takes user data and ships it durably needs this level of testing.
- **Tests are systems**: each test reads the world's state (a `Simlog` instance and its outputs) and asserts a property. Pytest is the *runner*; the assertions are the *systems*.

Reading the tests is a more useful exercise than reading the implementation. The tests *show* what the library guarantees; the implementation *delivers* those guarantees.

## Exercise 6 — The InspectionSystem connection

| feature                  | inspection system                       | test system                            |
|--------------------------|-----------------------------------------|-----------------------------------------|
| read-set                 | the columns of interest                  | the columns of interest                 |
| write-set                | nothing (or a "report" buffer)          | nothing (or a "report" buffer)          |
| schedule                 | every tick / on demand / `--debug`       | every tick (in CI) / on demand           |
| failure mode             | log the anomaly                          | raise AssertionError                    |
| production presence      | sometimes (gated by flag)                | absent (or in monitoring only)          |
| development presence     | always (helps debugging)                 | always (CI gate)                        |

The functions are structurally identical. The difference is in *what the report is used for*: an inspection system writes to logs or a dashboard; a test system writes to pytest's assertion mechanism.

In a mature simulator, the same function serves both roles. It returns a list of "violators"; in `--inspect` mode the caller prints them; in `--test` mode the caller asserts they're empty. *Same source code, different decision at the call site.*

## Exercise 7 — pytest-xdist as a determinism check

```sh
pip install pytest-xdist
pytest -n 8                                          # run 8 workers in parallel
```

Tests that pass under `pytest` but fail under `pytest -n 8` have a non-determinism leak. The leak surfaces in parallel because each worker has its own `PYTHONHASHSEED` (set when the worker forks); a test that iterates a `set` sees different orderings in each worker.

Common leaks pytest-xdist catches:

- `set` iteration in test setup or in production code under test.
- Wall-clock reads (`time.time()`) in test assertions.
- Global state shared between tests (one test mutates a module-level variable that another reads).
- Unseeded random calls in fixtures.

The fix is the §16 recipe — seeded RNG, no set iteration, no wall clock — applied to test code too. Tests are systems; the same discipline that keeps simulators reproducible keeps tests reproducible.

## Exercise 8 — A test runner that is the simulator's scheduler (stretch)

```python
def run_simulator(systems: list, world, ticks: int):
    """Run a list of systems for `ticks` ticks."""
    for _ in range(ticks):
        for system in systems:
            system(world)

# Production binary
PRODUCTION_SYSTEMS = [
    food_spawn, motion, next_event,
    apply_eat, apply_reproduce, apply_starve,
    cleanup,
]
run_simulator(PRODUCTION_SYSTEMS, world, ticks=10_000)

# Test binary
TEST_SYSTEMS = PRODUCTION_SYSTEMS + [
    test_no_creature_moves_too_far,
    test_population_bounded,
    test_energy_non_negative,
    inspect,
]
run_simulator(TEST_SYSTEMS, world, ticks=10_000)
```

The only difference between production and test is the system list. The scheduler is the same. The tick loop is the same. The world is the same.

Some test systems can fail loudly (raise AssertionError); others log and continue. Both are valid; the choice belongs to the test definition, not to a separate test framework.

For real-world use, pytest is still the right outer wrapper (discovery, reporting, parameterisation). But the *assertions inside* the pytest tests are systems over the simulator's tables. Pytest is plumbing; the systems are the logic.

This is the final connection. *Every concept in the book — systems, DAGs, single-writer ownership, determinism, ECS, EBP — applies to tests without translation, because tests are systems.* You have not learned a separate testing framework; you have learned that the simulator and its tests are one shape, instantiated twice with different system lists.

The trunk is closed. Forty-three concepts; one through-line; one shape applied at every scale.
