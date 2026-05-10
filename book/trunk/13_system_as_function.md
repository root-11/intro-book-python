# 13 — A system is a function over tables

> *Concept node: see the [DAG](../../concepts/dag.md) and [glossary entry 13](../../concepts/glossary.md#13--a-system-is-a-function-over-tables).*

<p align="center"><img src="../illustrations/differential_equations.jpg" alt="A mouse at the chalkboard — systems are functions of state" style="max-height: 300px; max-width: 100%;"></p>

A *system* is a function that reads from one or more tables and writes to one or more tables. It declares its inputs (the *read-set*) and its outputs (the *write-set*). It has no hidden state, no global side effects, no interaction with the outside world during a tick. The signature is the contract.

```python
def motion(pos_x: np.ndarray, pos_y: np.ndarray,
           vel_x: np.ndarray, vel_y: np.ndarray,
           dt: float) -> None:
    pos_x += vel_x * dt
    pos_y += vel_y * dt
```

Read-set: `vel_x`, `vel_y`, `dt`. Write-set: `pos_x`, `pos_y`. That is the entire contract. This system can run any time those four columns and `dt` are available and nothing else is writing `pos_x` or `pos_y`. It runs once per tick over the whole population — there is no per-creature loop in the body. The for-loop disappeared into numpy.

## Three shapes

Every system takes one of three shapes.

An **operation** is 1→1: every input row produces exactly one output row. `motion` is an operation — each creature's position is updated to its new position. Most update functions are operations.

A **filter** is 1→{0, 1}: every input row produces zero or one output rows. `apply_starve` (from `code/sim/SPEC.md`) is a filter — each creature with `energy ≤ 0` produces an entry in `to_remove`; creatures with `energy > 0` produce nothing. The numpy form is one line:

```python
def starving(energy: np.ndarray) -> np.ndarray:
    return np.where(energy <= 0)[0]  # returns the indices to remove
```

An **emission** is 1→N: every input row produces zero or more output rows. `apply_reproduce` is an emission — a parent above the energy threshold produces two offspring (a 1→2 emission).

These three shapes are the same shapes a database query takes. `SELECT * FROM t WHERE p` is a filter, `SELECT a + b FROM t` is an operation, `SELECT explode(arr) FROM t` is an emission. A system is a database operation written in Python against numpy columns instead of SQL against tables. If you have ever written SQL, you already know the vocabulary; the work is recognising your simulation in those terms.

## The OOP method is the anti-shape

This is the moment to name what most Python tutorials teach instead. The method-on-object shape — `class Creature: def tick(self, dt): self.pos += self.vel * dt` — is the same lesson rotated through `self`, and the rotation costs you everything important. The signature `def tick(self, dt)` does not tell you what the method reads or writes. The body does, but only after you read it. The contract is no longer expressible at the call site; it is implicit in the body of the method, which means you cannot reason about composition without inlining every method.

It also costs you the loop. The natural caller for `Creature.tick` is `for c in creatures: c.tick(dt)` — a Python-level loop, one method dispatch per element, interpreter-bound at the floor of ~5 ns per element from §1, plus another ~50-100 ns of `getattr` and method-call overhead per attribute. From [`code/measurement/tick_budget.py`](../../code/measurement/tick_budget.py) the cost is **27.9 ms per tick at 1,000,000 creatures** for one motion system, against **0.6 ms** for the function-over-columns form. The system shape is not just clearer — it is the only one that fits inside a 30 Hz budget at scale.

The wider rule: **a function that takes `self` does not have a declared read-set or write-set.** A function that takes columns does. This is one of the two or three places where "OOP versus data-oriented" is not a stylistic choice — it is whether your system has a contract you can read.

## Logging is a separate system

The other reflex Python encourages is to write to stdout from inside the loop. `print(f"creature {i} starved")`, `logger.info(...)`, `traceback.print_exc()` — all of these are *side effects* that violate the system's no-hidden-output contract. The fix is the same shape as everything else in this book: there is a `log_events` table, a *logging system* writes to it, and a separate *flush* system writes the table to disk or stdout.

The book builds this discipline at [§37 — The log is the world](37_log_is_world.md). For now, the rule is: if a system needs to communicate with the outside, it does so through a column declared in its write-set. There are no surprise prints.

## Observability and tests are systems too

A debug inspector is a system whose read-set is "all relevant columns" and whose write-set is "nothing observable" — it gathers data for inspection and produces no side effects on the world. In production it is *absent*, not gated by a flag — the program simply does not contain it.

A test is also a system. `assert pos.shape == vel.shape and not np.any(np.isnan(pos))` is a system whose read-set is `pos` and `vel`, write-set is nothing, and whose effect is to *fail loudly* if the contract of the previous system was violated. Tests-as-systems is the [§43](43_tests_are_systems.md) topic, but you have been writing them since §5 exercise 1.

A system declares its inputs, declares its outputs, and does no more. That is the shape that lets every other discipline in the book work.

## A few patterns to watch for

A function that reads a column, writes to it, and reads it again in the same call is *not* a system — it has implicit ordering inside the body. Either split it into two systems with explicit ordering, or buffer the writes until the function exits. A function that takes a `world` object and mutates whatever it likes is *not* a system — it has no declared write-set, and you cannot reason about it from its signature.

The contract that the system has *no hidden state* is what makes systems compose. Two systems with disjoint write-sets can run in parallel without coordination ([§31](31_disjoint_writes_parallelize.md)). Two systems whose read-set and write-set form a chain must run in order ([§14](14_systems_compose_into_a_dag.md)). The contract is the basis for all of this.

## Exercises

Use the deck from §5, your `tick_lab` from §11, or the §0 simulator skeleton; any of them provides enough tables.

1. **Identify the shape.** Classify each as operation, filter, or emission:
   - Squaring every entry in a `np.ndarray` of `float32`.
   - Filtering even integers from a `np.ndarray` of `int32`.
   - Splitting each string in a `list[str]` into words, returning all words.
   - Computing the sum of a `np.ndarray` of `int32`.
2. **Write motion as a system.** With `pos_x, pos_y, vel_x, vel_y` as numpy `float32` columns of length 100, write `motion(pos_x, pos_y, vel_x, vel_y, dt)` as defined in the prose. Apply it to 100 creatures with random initial positions and velocities. Print the position of one creature across 10 ticks. The body is two lines.
3. **Declare the contract.** Add a docstring to `motion` listing its read-set and write-set explicitly. The signature plus the docstring is the system's contract.
4. **Write a filter.** With `energy: np.ndarray`, write `starving(energy)` returning a numpy array of indices where `energy[i] <= 0`. This is the read-only first half of `apply_starve`.
5. **Write an emission.** With `parent_energy: np.ndarray`, threshold `threshold: float`, write `reproduce(parent_energy, threshold)` returning two parallel arrays — `parent_indices` and `offspring_energies` — for each parent above threshold, with two entries each. This is a 1→2 emission. (Hint: `mask = parent_energy > threshold; idx = np.where(mask)[0]; np.repeat(idx, 2)`.)
6. **Observe non-systems.** Find a function in your previous work (or any Python tutorial) that takes `self` and mutates whatever it likes, or writes to a global, or calls `print` from inside the body. Note what makes it not a system. Try to express its read-set and write-set from the signature alone — confirm you cannot.
7. **The OOP cost in your fingers.** Run `uv run code/measurement/tick_budget.py`. Read the table. Note that you have just seen, at 1,000,000 creatures, what happens when the loop is in the body of a method instead of in numpy. The 30 Hz row is *over* for the Python dataclass version. The system-shaped version uses 1.8% of the budget.
8. *(stretch)* **A test as a system.** Write `def no_creature_moved_too_far(prev_pos_x, prev_pos_y, cur_pos_x, cur_pos_y, max_step)` returning indices where any creature moved further than `max_step` between two ticks. The "test" is just an inspection system reading the world. Hint: `dx = cur_pos_x - prev_pos_x; dy = cur_pos_y - prev_pos_y; np.where(dx*dx + dy*dy > max_step*max_step)[0]`.

Reference notes in [13_system_as_function_solutions.md](13_system_as_function_solutions.md).

## What's next

[§14 — Systems compose into a DAG](14_systems_compose_into_a_dag.md) takes the next step: when many systems run together, how do they fit?
