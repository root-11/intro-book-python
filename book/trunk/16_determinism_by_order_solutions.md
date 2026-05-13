# Solutions: 16 — Determinism by order

## Exercise 1 — Run the iteration-order exhibit

```sh
uv run code/measurement/set_iteration_order.py
```

Source: [`code/measurement/set_iteration_order.py`](https://github.com/root-11/intro-book-python/blob/main/code/measurement/set_iteration_order.py).

```
Set iteration order across runs:
  run 1: delta,bravo,foxtrot,echo,alpha,charlie
  run 2: alpha,foxtrot,delta,charlie,echo,bravo
  run 3: bravo,echo,alpha,foxtrot,charlie,delta
  → 3 distinct orders — sets are non-deterministic.

Dict iteration order across runs:
  run 1: alpha,bravo,charlie,delta,echo,foxtrot
  run 2: alpha,bravo,charlie,delta,echo,foxtrot
  run 3: alpha,bravo,charlie,delta,echo,foxtrot
  → orders match — dicts are insertion-ordered since CPython 3.7.
```

The dict survival is the *insertion order* of the keys, which is a property of how you populated the dict. It's *not* a guarantee against:

- `dict(some_set)` — values come from set iteration; first survival breaks.
- `frozenset` keys — same hash-bucket randomness applies.
- `dict.fromkeys(some_set)` — same.

Survive only what you can prove. When in doubt: `sorted(...)`.

## Exercise 2 — Hash the world

```python
import hashlib
import numpy as np

def hash_world(world) -> str:
    h = hashlib.blake2b(digest_size=16)
    for col in (world.pos_x, world.pos_y, world.vel_x, world.vel_y,
                world.energy, world.ids, world.gens):
        h.update(col.tobytes())
    h.update(np.array([len(world.pos_x)], dtype=np.int64).tobytes())  # length too
    return h.hexdigest()
```

`arr.tobytes()` returns the contiguous in-memory bytes. blake2b is fast and 16 bytes is plenty for run-to-run comparison. Including the length prevents two worlds with different sizes but identical-prefix data from hashing the same.

## Exercise 3 — Two identical runs

```python
def run(seed: int, ticks: int) -> str:
    rng = np.random.default_rng(seed)
    world = build_world(rng, n=100)
    for _ in range(ticks):
        tick(world, rng, dt=1.0/30.0)
    return hash_world(world)

assert run(42, 100) == run(42, 100)             # same seed → same hash
print(run(42, 100))                             # any deterministic 32-char hex
```

Bit-identical. If this fails, your simulator has a non-determinism source somewhere — exercise 5 is the diagnostic.

## Exercise 4 — Introduce non-determinism deliberately

```python
def run_unseeded(ticks: int) -> str:
    rng = np.random.default_rng()               # no seed — entropy from OS
    world = build_world(rng, n=100)
    for _ in range(ticks):
        tick(world, rng, dt=1.0/30.0)
    return hash_world(world)

print(run_unseeded(100))   # something
print(run_unseeded(100))   # something else
```

Without a seed, `default_rng()` reads `os.urandom`. Two consecutive runs draw from different entropy and produce different results. The hashes differ. The simulator is now untestable.

## Exercise 5 — Find the culprit

```python
def hash_world_per_system(world) -> dict:
    """Run each system and hash the world after each."""
    snapshots = {}
    snapshots["start"] = hash_world(world)
    food_spawn(world)
    snapshots["after food_spawn"] = hash_world(world)
    motion(world)
    snapshots["after motion"] = hash_world(world)
    next_event(world)
    snapshots["after next_event"] = hash_world(world)
    apply_eat(world)
    snapshots["after apply_eat"] = hash_world(world)
    # ... and so on
    return snapshots
```

Run twice with the same seed; compare the two `snapshots` dicts. The first key whose hash differs identifies the offending system. Look inside its body for:

- `for x in some_set:`
- `time.time()`, `time.perf_counter()`, `datetime.now()`
- `random.random()`, `np.random.random()` without an rng instance
- `hash(some_string)` used in any output computation
- `os.environ`, `os.getpid()`
- Iteration over a `frozenset`, `dict_keys`, or `dict.fromkeys(set)`

Most simulator non-determinism in the wild is one of these patterns hiding inside a "harmless" helper.

## Exercise 6 — Time as input

```python
# Before — system reads the OS clock:
def schedule_event_bad(events):
    now = time.perf_counter()                   # non-deterministic
    events.append((now + 0.5, "fire"))

# After — system takes time as a parameter:
def schedule_event(events, current_time: float):
    events.append((current_time + 0.5, "fire"))

# The tick loop scaffolding reads the wall clock — once, at the boundary:
def tick(world, current_time: float, dt: float):
    schedule_event(world.events, current_time)
    motion(world, dt)
    ...

current_time = 0.0
for _ in range(100):
    tick(world, current_time, dt)
    current_time += dt
```

The systems are pure functions of their inputs. The tick loop chooses what `current_time` is. For a real-time simulator, `current_time = time.perf_counter() - start`. For a deterministic replay, `current_time` comes from the event log. Same systems, two execution modes, the difference at the boundary, not in the body.

## Exercise 7 — The set trap up close

```python
import numpy as np
rng = np.random.default_rng(42)
s   = set(rng.integers(0, 1_000_000, size=1000).tolist())   # contents deterministic

# Three iterations IN THE SAME PROCESS
o1 = list(s); o2 = list(s); o3 = list(s)
print(o1 == o2 == o3)                # True — same process, same hash table layout
```

Within one process, set iteration order is stable (the hash table layout doesn't change between iterations). Run the same program in two fresh shells and you get two different orders, because `PYTHONHASHSEED` is randomised per process.

The trap: a single test run does not catch the bug. The CI worker that runs the same test in five parallel processes catches it. The user who reports "works on my machine, fails in CI" has hit it.

The fix: never iterate a `set` in a system. Always:

```python
for x in sorted(s):                  # deterministic across runs
    ...
```

Or store the data in something that's already ordered (a `list`, a numpy array). Or use a `dict` with insertion order if order matters and uniqueness matters.

## Exercise 8 — A property test (stretch)

```python
import numpy as np

def property_test_determinism(n_seeds: int = 100, ticks: int = 100):
    seeds = list(range(n_seeds))
    for seed in seeds:
        h1 = run(seed, ticks)
        h2 = run(seed, ticks)
        assert h1 == h2, f"non-deterministic at seed {seed}: {h1} vs {h2}"
    distinct = len({run(s, ticks) for s in seeds})
    assert distinct > n_seeds * 0.95, "different seeds collapse to same world — bug"
    print(f"OK: {n_seeds} seeds, all reproducible, {distinct} distinct worlds")

property_test_determinism()
```

Two assertions:

1. **Same seed → same world.** Catches non-determinism (a `set` iteration, a `time.time()` call).
2. **Different seeds → different worlds (mostly).** Catches accidental seed-loss (a global `random.seed()` overriding the per-run rng) — without it, every run produces the same hash regardless of the seed parameter.

This is the entire core of property-based simulation testing. Hypothesis (the Python library) builds elaborate generators and shrinkers around it; the underlying assertion is the same.
