# Solutions: 13 — A system is a function over tables

## Exercise 1 — Identify the shape

| operation                                              | shape       |
|--------------------------------------------------------|-------------|
| Squaring every entry of a `np.ndarray[float32]`        | operation (1→1) |
| Filtering even integers from `np.ndarray[int32]`       | filter (1→{0,1}) |
| Splitting each `str` in a `list[str]` into words       | emission (1→N) |
| Summing a `np.ndarray[int32]`                          | reduction (N→1) — a fourth shape, distinct from the three |

Reductions deserve a footnote: they collapse a column into a scalar. They are systems too — read-set is the column, write-set is one scalar. The book mostly uses reductions inline (`sum`, `max`, `count_nonzero`) rather than as named systems, but the contract still applies.

## Exercise 2 — Write motion as a system

```python
import numpy as np

def motion(pos_x, pos_y, vel_x, vel_y, dt):
    pos_x += vel_x * dt
    pos_y += vel_y * dt

rng = np.random.default_rng(0)
n = 100
pos_x = rng.uniform(0, 10, n).astype(np.float32)
pos_y = rng.uniform(0, 10, n).astype(np.float32)
vel_x = rng.uniform(-1, 1, n).astype(np.float32)
vel_y = rng.uniform(-1, 1, n).astype(np.float32)

dt = 1.0 / 30.0
for t in range(10):
    print(f"t={t}: creature 17 at ({pos_x[17]:.3f}, {pos_y[17]:.3f})")
    motion(pos_x, pos_y, vel_x, vel_y, dt)
```

Two lines in the body. No per-creature loop, no method dispatch, no `self`. The `+=` operator on a numpy column is a single C-level pass.

## Exercise 3 — Declare the contract

```python
def motion(pos_x, pos_y, vel_x, vel_y, dt):
    """Advance every creature's position by one tick of motion.

    Read-set:  vel_x, vel_y, dt
    Write-set: pos_x, pos_y     (in-place)
    Contract:  pos_*.shape == vel_*.shape; arrays are float32 columns.
    """
    pos_x += vel_x * dt
    pos_y += vel_y * dt
```

The signature plus the docstring is the entire contract. A reader of `motion` does not need to inline the body to know it does not touch `energy` or `birth_t`; the docstring says so. The [§14 DAG construction](14_systems_compose_into_a_dag.md) reads exactly this declaration to schedule the system.

A test that the contract is honest (a [§43 test-as-system](43_tests_are_systems.md)) compares the declared write-set to the columns the function actually mutated. If `motion` ever silently writes to `energy`, the test catches it.

## Exercise 4 — Write a filter

```python
def starving(energy: np.ndarray) -> np.ndarray:
    """Return indices of creatures with energy <= 0.

    Read-set: energy
    Write-set: nothing  (returns indices for a separate apply step)
    """
    return np.where(energy <= 0)[0]

energy = np.array([3.0, -1.0, 5.0, 0.0, 7.0], dtype=np.float32)
print(starving(energy))     # [1 3]
```

The filter is read-only. It returns the indices that satisfy the predicate; a separate "apply" system writes them into `to_remove`. This separation is the [§22 mutations buffer](22_mutations_buffer.md) discipline applied at the smallest scale: filter and apply are separate systems with different read-sets and write-sets.

## Exercise 5 — Write an emission

```python
def reproduce(parent_energy: np.ndarray, threshold: float):
    """For each parent above threshold, produce two offspring (1→2 emission).

    Read-set: parent_energy, threshold
    Write-set: nothing  (returns parallel arrays for the apply step)
    Returns:
        parent_indices: which parent each offspring came from (length 2*K)
        offspring_energies: starting energy for each offspring (length 2*K)
    """
    mask = parent_energy > threshold
    idx  = np.where(mask)[0]
    parent_indices    = np.repeat(idx, 2)                    # parent appears twice
    offspring_energies = np.repeat(parent_energy[idx] / 2, 2) # half-energy each
    return parent_indices, offspring_energies

energies = np.array([3.0, 7.0, 1.0, 9.0, 5.0, 11.0], dtype=np.float32)
p, o = reproduce(energies, 5.0)
print(f"parents:    {p}")              # [1 1 3 3 5 5]
print(f"offspring:  {o}")              # [3.5 3.5 4.5 4.5 5.5 5.5]
```

`np.repeat(arr, 2)` is the emission primitive: each input row produces two output rows in column form. For a 1→N emission with variable N per row, `np.repeat(arr, counts)` takes a per-row count array. The shape is "filter, then expand"; the apply system later inserts the rows into the table.

## Exercise 6 — Observe non-systems

A canonical non-system from the wild:

```python
class GameObject:
    def update(self):
        self.pos += self.vel * GLOBAL_DT
        if self.energy <= 0:
            print(f"{self.name} died")          # side effect — not in any signature
            self.dead = True
            World.remove(self)                  # mutates global state
        for nearby in World.find_nearby(self):  # reads global state
            self.energy += nearby.value
```

What the signature `def update(self)` declares: nothing. What the body actually does:

- Reads `self.pos`, `self.vel`, `self.energy`, `self.name`, `World.objects` (implicit, through `find_nearby`)
- Reads global `GLOBAL_DT`
- Writes `self.pos`, `self.dead`, `self.energy`, `World.objects` (implicit, through `World.remove`)
- Writes stdout

You cannot tell any of this from the signature. To compose `update` with another system, you'd have to inline the body and trace every method call. Two `update` calls cannot run in parallel because both write `World.objects`. Tests cannot mock the read-set without mocking the world. The function has no contract anyone can read; it has *behaviour*, which is not the same thing.

## Exercise 7 — The OOP cost in your fingers

```sh
uv run code/measurement/tick_budget.py
```

The 1M-creatures row:

| layout                  | tick (ms) | 30 Hz       | 60 Hz        |
|-------------------------|----------:|-------------|--------------|
| numpy SoA               |  0.278    | fit (0.8%)  | fit (1.7%)   |
| Python dataclass list   | 27.525    | fit (82.6%) | **OVER 165%**|

The dataclass form has *one motion system* eating 82.6% of the 30 Hz budget; the simulator has 5.7 ms left for everything else (collision, energy, reproduction, rendering). At 60 Hz the loop has already missed its deadline. The system-as-function-over-numpy form runs the same logic in 0.278 ms and leaves 32.7 ms (98%) of the 30 Hz budget for the rest of the simulator.

The 100× cost gap is the cost of putting the per-creature loop inside the *interpreter* instead of inside *numpy*. There is no syntactic refactor of the OOP version that closes this gap — the cost is structural.

## Exercise 8 — A test as a system (stretch)

```python
def no_creature_moved_too_far(prev_pos_x, prev_pos_y,
                              cur_pos_x, cur_pos_y, max_step) -> np.ndarray:
    """Return indices of creatures that moved further than max_step between ticks.

    Read-set:  prev_pos_*, cur_pos_*, max_step
    Write-set: nothing  (the caller decides whether to assert, log, or correct)
    """
    dx = cur_pos_x - prev_pos_x
    dy = cur_pos_y - prev_pos_y
    return np.where(dx * dx + dy * dy > max_step * max_step)[0]


# Used as an assertion in a tick:
violators = no_creature_moved_too_far(prev_x, prev_y, x, y, 1.0)
assert violators.size == 0, f"creatures {violators} teleported"
```

This is a *system* by the chapter's definition: declared read-set, no write-set, no hidden state. Its presence in the program is what an *invariant* looks like — the rest of the program is required to keep `no_creature_moved_too_far` returning an empty array. Failing this test is the simulator telling you the motion system has a bug.

The Rust edition would write this as a `fn` taking slices; the only difference is that Python tests run inside the same loop while Rust tests usually run as `#[cfg(test)]` builds. The discipline is identical — the test is a system over the same tables, with the same contract shape, that happens to *report* rather than *transform*. [§43 — tests are systems](43_tests_are_systems.md) generalises this.
