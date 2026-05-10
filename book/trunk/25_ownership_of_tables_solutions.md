# Solutions: 25 — Ownership of tables

## Exercise 1 — Identify the writers

| table              | writer                              | notes                                |
|--------------------|-------------------------------------|--------------------------------------|
| `creatures` (live) | `cleanup`                           | every other system pushes to buffers |
| `food` (live)      | `cleanup`                           | same                                 |
| `food_spawner`     | `food_spawn` (the spawner system)   | a parameter table read-only elsewhere |
| `pending_event`    | `next_event`                        | rebuilt per tick                     |
| `eaten`            | `apply_eat`                         | append-only log                      |
| `born`             | `apply_reproduce`                   | append-only log                      |
| `dead`             | `apply_starve`                      | append-only log                      |
| `hungry` (presence)| `classify_hunger`                   | rebuilt per tick                     |
| `to_remove`        | many appenders, one consumer (`cleanup`) | per-system queues, drained at boundary |
| `to_insert_*`      | many appenders, one consumer (`cleanup`) | same                                 |

Audit any simulator project for tables with *two* writers — that's the rule violated. Common sources of the violation:

- An "update" function that *also* validates and corrects.
- A logging side-effect that mutates state.
- Two systems both setting a derived flag.

The fix is always one of: split into two systems with an intermediate buffer, or designate one as the writer and have the other request changes via a side buffer.

## Exercise 2 — The view trap, in your fingers

```python
import numpy as np

arr = np.arange(10)
view = arr[2:5]                      # view, NOT a copy
view[0] = 999
print(arr)
# [  0   1 999   3   4   5   6   7   8   9]   — arr was mutated through the view!

cpy = arr[2:5].copy()                # explicit copy
cpy[0] = 0
print(arr)
# [  0   1 999   3   4   5   6   7   8   9]   — arr unchanged
```

A slice of a numpy array is a view into the same backing buffer. `view[0] = 999` writes to byte offset 16 of `arr` (since `int64` × index 2). The `.copy()` allocates a new buffer; mutations there are isolated.

This is the classic memory-aliasing trap. The variable name (`view` vs `cpy`) gives no signal. The dtype gives no signal. The only ways to know: check `arr.base` (`view.base is arr` is `True`; `cpy.base is None`) or pass through the convention `*.copy()` whenever ownership transfers.

## Exercise 3 — The read-only-flag mitigation

```python
arr = np.arange(10)
arr.flags.writeable = False

try:
    arr[3] = 42
except ValueError as e:
    print(f"caught: {e}")
# caught: assignment destination is read-only

view = arr[2:5]
print(view.flags.writeable)          # False — read-only-ness propagates to views
```

Setting `writeable = False` is a runtime guard. Anyone with a reference to the array — including any view derived from it — can read but not write. This is the closest Python has to Rust's `&[T]` (immutable borrow). It does not guarantee correctness across function calls (a careless caller can still set `writeable = True` back), but it catches accidental writes loudly.

For library functions that accept arrays from outside, locking the input via `writeable = False` for the function body is a defensive practice. The cost is one attribute set; the protection is real.

## Exercise 4 — A constructed violation

```python
import numpy as np

def system_a(energy):
    energy[:] += 1.0                # writer 1

def system_b(energy):
    energy[:] -= 0.5                # writer 2 — same column!

# Sequentially: result depends on order
energy = np.zeros(10)
system_a(energy); system_b(energy)
print(energy)        # [0.5, 0.5, ...] — A first, then B

energy = np.zeros(10)
system_b(energy); system_a(energy)
print(energy)        # [0.5, 0.5, ...] — same end state because additions commute
                      # but the per-step state would differ
```

Sequentially: order matters and must be specified. With multiprocessing/shared_memory:

```python
# anti-pattern: bad! two writers, no synchronisation
from multiprocessing import Process
from multiprocessing.shared_memory import SharedMemory
import numpy as np

shm = SharedMemory(create=True, size=80)
energy = np.ndarray((10,), dtype=np.float64, buffer=shm.buf)
energy[:] = 0

def worker_a(shm_name):
    s = SharedMemory(shm_name)
    e = np.ndarray((10,), dtype=np.float64, buffer=s.buf)
    for _ in range(1_000_000): e[:] += 0.0001

def worker_b(shm_name):
    s = SharedMemory(shm_name)
    e = np.ndarray((10,), dtype=np.float64, buffer=s.buf)
    for _ in range(1_000_000): e[:] -= 0.0001

# Run them simultaneously
p1, p2 = Process(target=worker_a, args=(shm.name,)), Process(target=worker_b, args=(shm.name,))
p1.start(); p2.start(); p1.join(); p2.join()
print(energy)   # not [0, 0, ...] — race conditions ate some updates
```

Each `+=` involves a read, an add, and a write. Two processes interleaving these without coordination produce *lost updates*: process A reads x; process B reads x; A writes x+1; B writes x-1; the result is x-1 (or x+1) instead of x. No `ValueError`, no warning. Just silently wrong arithmetic.

The single-writer rule is the structural prevention. Two writers to the same column means *coordination is required*, and Python provides no enforcement. The rule eliminates the need for coordination at the architectural level.

## Exercise 5 — Refactor with a buffer

```python
def system_a(energy, energy_delta):
    energy_delta[:] += 1.0           # writer of energy_delta only

def system_b(energy, energy_delta):
    energy_delta[:] -= 0.5           # writer of energy_delta only

def cleanup(energy, energy_delta):
    energy[:] += energy_delta        # the SOLE writer of energy
    energy_delta[:] = 0
```

Now `system_a` and `system_b` are writer-disjoint with respect to `energy`; both write to `energy_delta` (which is also a violation, but a *contained* one — `energy_delta` is a side buffer, not load-bearing world state).

The architectural fix is one more level of buffering: each system writes to its *own* delta column.

```python
def system_a(energy_delta_a):  energy_delta_a[:] += 1.0
def system_b(energy_delta_b):  energy_delta_b[:] -= 0.5
def cleanup(energy, energy_delta_a, energy_delta_b):
    energy += energy_delta_a + energy_delta_b
    energy_delta_a[:] = 0; energy_delta_b[:] = 0
```

This is the canonical pattern for parallel mutation: each writer has its own column; the merge happens in cleanup, single-threaded, on disjoint inputs. [§31 — Disjoint write-sets parallelize freely](31_disjoint_writes_parallelize.md) develops it further.

## Exercise 6 — Build an InspectionSystem

```python
from contextlib import contextmanager

@contextmanager
def read_only_world(world):
    """Locks every column read-only for the duration of the inspection."""
    columns = (world.pos_x, world.pos_y, world.vel_x, world.vel_y, world.energy, world.id)
    for c in columns:
        c.flags.writeable = False
    try:
        yield
    finally:
        for c in columns:
            c.flags.writeable = True

def inspect(world) -> dict:
    """A read-only system; returns a snapshot."""
    with read_only_world(world):
        return {
            "n_active": world.n_active,
            "energy_min": float(world.energy[: world.n_active].min()),
            "energy_max": float(world.energy[: world.n_active].max()),
            "centre_of_mass": (float(world.pos_x[: world.n_active].mean()),
                               float(world.pos_y[: world.n_active].mean())),
        }
```

The system reads everything, writes nothing, locks the world for the duration. Any accidental write inside the inspection raises `ValueError` immediately. The lock is dropped on exit, so subsequent (non-inspection) systems can mutate normally.

This is the §43 *test as system* shape. A test that "verifies the world is consistent" runs in the same shape: lock, read, assert, unlock.

## Exercise 7 — The cleanup system as canonical writer (stretch)

```python
def write_audit(world, system_func):
    """Record which columns each system wrote during one tick."""
    snapshot_before = {name: getattr(world, name).tobytes() for name in world.column_names}
    system_func(world)
    written = []
    for name, before in snapshot_before.items():
        after = getattr(world, name).tobytes()
        if after != before:
            written.append(name)
    return written

# After running each system, assert which ones it should have written
expected = {
    "motion":         {"pos_x", "pos_y"},
    "next_event":     {"pending_event"},
    "apply_eat":      {"to_remove", "energy_delta"},     # buffers, not live tables
    "apply_starve":   {"to_remove"},
    "cleanup":        {"pos_x", "pos_y", "vel_x", "vel_y", "energy", "id", "n_active",
                       "id_to_slot", "to_remove", "to_insert_pos_x", ...},  # cleanup writes everything
}

for name, func in systems:
    written = write_audit(world, func)
    assert set(written) <= expected[name], f"{name} wrote {written} — unexpected: {set(written) - expected[name]}"
```

The audit is itself a system. It runs once per tick (or in a CI-only build) and asserts the structural property: *every system writes only what it claims to write*. A drift between expected and actual is the signal that someone added a side-effect — exactly the violation the single-writer rule forbids.

In Python this is the closest you get to a borrow checker. It runs at runtime, with O(N) overhead per tick (the byte snapshots), and it catches violations at the smallest mutation. Disable it in production; keep it on in CI.
