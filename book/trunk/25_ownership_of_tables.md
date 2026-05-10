# 25 — Ownership of tables

> *Concept node: see the [DAG](../../concepts/dag.md) and [glossary entry 25](../../concepts/glossary.md#25--ownership-of-tables).*

<p align="center"><img src="../illustrations/dag_planning_checklist.jpg" alt="One plan, one writer — PLAN, ANALYZE, DESIGN, BUILD, TEST, IMPROVE" style="max-height: 300px; max-width: 100%;"></p>

Every table has exactly one writer.

The rule is small. Its consequences are everything.

**Why it works.** A row is a tuple ([§6](06_a_row_is_a_tuple.md)) — its fields are aligned by index. A table's columns must be modified together to maintain alignment. A single writer guarantees this: only one place in the code mutates the table, so only one place can violate alignment, so testing one place is enough.

A table with two writers has two places where alignment can be violated. If they run concurrently, alignment is violated nondeterministically. If they run sequentially, the order matters and must be specified. Either way, the cost of getting it right grows superlinearly with the number of writers.

## The Python-specific problem: nothing enforces it

Rust has a borrow checker. `&mut [T]` is the type-level expression of single-writer ownership; only one mutable reference can exist at a time, and the compiler rejects code that violates it. Python has no equivalent. There is no `&mut`, no exclusive-access type, no compile-time check. Anyone who has a reference to a numpy array can mutate it. **The single-writer rule is a discipline you enforce by convention, not a constraint the language enforces for you.**

This makes the rule *more* important in Python, not less. Without compile-time enforcement, the violations show up at runtime as the bugs the rule was supposed to prevent: intermittent, silent, late-binding. The discipline is what stands between the architecture and the bug.

## The numpy view trap

The hardest version of the violation in Python is the numpy view. A slice of a numpy array is *not* a copy — it is a view into the same underlying bytes. Writing through the view mutates the parent:

```python
# anti-pattern: bad!
arr = np.zeros(10)
view = arr[2:5]    # looks like a new array; is actually a view
view[0] = 42       # also writes arr[2] = 42
```

A function receiving `view` has no way to know from the variable's name or its `np.ndarray` type that it shares memory with someone else's table. There is no compile-time signal. Mutating `view` looks local; the side effect on `arr` is invisible until something else reads it. This is the single-writer rule violated at the byte level, hidden behind a slice that looks like a fresh allocation.

Three mitigations:

```python
# explicit copy when handing data to a function that may mutate it
foreign_function(arr[2:5].copy())

# read-only flag on the parent (writes via any view raise ValueError)
arr.flags.writeable = False

# document the ownership in the function signature and let it live in the contract
def motion(pos_x: np.ndarray, vel_x: np.ndarray, dt: float) -> None:
    """Read-set: vel_x, dt.   Write-set: pos_x.
       pos_x and vel_x must not alias each other or any other column."""
```

The first two are runtime mechanisms. The third is the convention this book lives on. A function's docstring declares the read-set and write-set ([§13](13_system_as_function.md)); the *caller* is responsible for not handing aliasing arrays into a function that assumes none. If the caller cannot guarantee non-aliasing, they pass a copy.

## The disciplines that depend on it

All of these need single-writer ownership to work:

- **[§31 — Disjoint write-sets parallelize freely](31_disjoint_writes_parallelize.md).** Two systems with disjoint write-sets can run on different processes. The rule guarantees no shared mutation.
- **[§22 — Mutations buffer](22_mutations_buffer.md).** A side-table writer (cleanup) is the *only* writer of `creatures`. All other systems push to `to_remove` and `to_insert`, which they own.
- **[§43 — Tests are systems](43_tests_are_systems.md).** A test system reads everything and writes nothing. The ownership rule is what guarantees its reads see consistent state.
- **The InspectionSystem pattern.** A debug inspector holds read-only references to every table. Read-only access composes with single-writer ownership to make races structurally impossible.

## What the rule looks like in practice

```python
def motion(pos_x: np.ndarray, pos_y: np.ndarray,
           vel_x: np.ndarray, vel_y: np.ndarray, dt: float) -> None:
    """Read-set: vel_x, vel_y, dt.   Write-set: pos_x, pos_y."""
    pos_x += vel_x * dt
    pos_y += vel_y * dt

def next_event(pos_x: np.ndarray, food_x: np.ndarray,
               pending: np.ndarray) -> None:
    """Read-set: pos_x, food_x.   Write-set: pending."""
    ...

def apply_eat(pending: np.ndarray, food: np.ndarray,
              to_remove: list[int], energy: np.ndarray) -> None:
    """Read-set: pending, food.   Write-set: to_remove (append), energy."""
    ...
```

For each table, exactly one writer is allowed:

- `pos_x, pos_y`: written only by `motion`.
- `pending`: written only by `next_event`.
- `to_remove`, `to_insert`: written by *many* systems, but each system appends only its own queued mutations; no one reads them until cleanup.
- `creatures`, `food`: written only by `cleanup`, which materialises every other system's queued changes.

Multiple systems may *contribute* to a table by appending to its side buffer; the actual single writer of the live table is cleanup. The architecture preserves the rule even as many systems propose mutations.

## Bugs that arise from violations

Two systems writing the same column produce inconsistent state. The bug is usually *intermittent* (depends on schedule), *silent* (no error reported, just bad data), and *late-binding* (manifests far from the cause). They are among the hardest bugs in any concurrent system. The single-writer rule eliminates them by construction. In Python, where the language will not catch the violation, the rule is the only thing standing between you and the bug.

The rule applies recursively. A view table whose entries are derived from another table inherits the ownership rule: a `hungry: np.ndarray` is owned by the system that classifies hunger; no other system writes to it.

This is the rule that closes Memory & lifecycle. Without it, the buffering, swap_remove, index maps, and slot recycling are all unsafe in any concurrent or parallel context. With it, everything composes.

## Exercises

1. **Identify the writers.** For each table in your simulator (`creatures`, `food`, `food_spawner`, `pending_event`, `eaten`, `born`, `dead`, `hungry`, `to_remove`, `to_insert`), name the *one* system that writes it. If you find a table with two writers, the rule is violated — investigate.
2. **The view trap, in your fingers.** Build `arr = np.arange(10)`. Take `view = arr[2:5]`. Set `view[0] = 999`. Print `arr`. Confirm `arr[2] == 999`. Now take `cpy = arr[2:5].copy()`, set `cpy[0] = 0`, print `arr` — confirm `arr` is unchanged. The slice was a view; the `.copy()` was not.
3. **The read-only-flag mitigation.** Build `arr = np.arange(10)`. Set `arr.flags.writeable = False`. Try to assign `arr[3] = 42`. Catch the `ValueError`. Now derive `view = arr[2:5]` from the read-only parent — note that `view.flags.writeable` is also `False`. Read-only-ness propagates.
4. **A constructed violation.** Write two functions that both mutate `energy`. Call them in sequence on the same array; the result is whatever the second one wrote. Now run them in two `multiprocessing.Process` workers sharing the array via `multiprocessing.shared_memory`; observe that no error is raised and the bug is silent. This is the failure mode the single-writer rule prevents — Python will not warn you.
5. **Refactor with a buffer.** Take one of the violations from exercise 4 and add a side buffer that one function writes and the other reads. The two functions are now writer-disjoint, even though they touch the same logical concept.
6. **Build an InspectionSystem.** Write a function that takes a `World` (a dataclass holding all the tables), reads every column, and returns a snapshot dictionary. Mark every input array read-only via `arr.flags.writeable = False` for the duration of the call. The system is read-only by construction and cannot violate the rule.
7. *(stretch)* **The cleanup system as canonical writer.** In your simulator, audit: every mutation of `creatures`, `food`, etc. flows through cleanup. Every other system writes only to `to_remove`, `to_insert`, or its own outputs. Verify the audit holds for the simulator end-to-end. Note this is harder in Python than in Rust because nothing checks it for you — write a unit test that asserts no system other than cleanup mutates the live tables.

Reference notes in [25_ownership_of_tables_solutions.md](25_ownership_of_tables_solutions.md).

## What's next

You have closed Memory & lifecycle. The simulator's machinery is now complete: it can grow, shrink, recycle, parallelise, and replay. The next phase is *Scale*, starting with [§26 — Hot/cold splits](26_hot_cold_splits.md). The simulator's per-tick cost goes under the microscope.
