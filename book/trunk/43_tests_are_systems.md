# 43 — Tests are systems; TDD from day one

> *Concept node: see the [DAG](../../concepts/dag.md) and [glossary entry 43](../../concepts/glossary.md#43--tests-are-systems-tdd-from-day-one).*

<p align="center"><img src="../illustrations/dag_planning_checklist.jpg" alt="PLAN, ANALYZE, DESIGN, BUILD, TEST, IMPROVE — tests are part of the same loop, written first" style="max-height: 300px; max-width: 100%;"></p>

A test reads the world's state and asserts that some property holds. A system reads the world's state and writes a derived result. **The two are structurally the same.**

This is not a slogan. It is the structural fact that lets every other discipline in the book apply to tests without translation.

A test fixture is *the world at some tick*. A test is *a system whose write-set is empty*, or whose write-set is a small "report" table. A test runner is *the same scheduler that runs the simulator*, executing the test's read-set against the world.

```python
def no_creature_moves_too_far(
    pos_x_before: np.ndarray, pos_y_before: np.ndarray,
    pos_x_after: np.ndarray,  pos_y_after: np.ndarray,
    max_step: float,
) -> np.ndarray:
    """Returns indices of creatures whose move exceeded max_step.
    Read-set: the four position arrays, max_step.
    Write-set: empty (returns a report)."""
    dx = pos_x_after - pos_x_before
    dy = pos_y_after - pos_y_before
    dist_sq = dx * dx + dy * dy
    return np.where(dist_sq > max_step * max_step)[0]
```

This is a system. Read-set: the four position arrays plus `max_step`. Write-set: a report array. It runs over the simulator's tables. It asserts a property by returning the empty array on success and a non-empty one on failure. **The same code path serves test and inspection use** — at test time the assertion `assert result.size == 0` runs after; in production an inspection system might log non-empty results without failing.

## Three benefits compound

**Property tests over numpy columns fall out.** A property test fixes an RNG seed, runs the simulator for N ticks, and asserts that some property holds at every tick. If the property is "no creature moves more than `max_step` per tick", the assertion is the system above. If it is "the population stays bounded", the assertion is `world.n_active <= bound`. Each is a system.

**Replay tests over event logs fall out.** A replay test loads a recorded log via §37's triple-store, runs the replayer, and compares the resulting world to a snapshot. The "test" is the comparison; the comparison is a system over both worlds' columns.

**Integration tests do not need mocks.** A mock exists because the test cannot exercise the real component. The boundary-as-queue rule from [§35](35_boundary_is_the_queue.md) means there are no external components inside the simulator — every external interaction goes through the queues. A test fills the in-queue with synthetic input, runs the simulator, asserts on the out-queue. No `unittest.mock`, no `monkeypatch`, no "patch this import to return that fake" — the test reads the same data the simulator reads.

## The Python-specific calibrations

**pytest is fine.** Pytest is the universal Python testing tool, and it is genuinely good at the things this chapter does *not* cover: discovery, reporting, parameterisation, fixtures-as-setup. Use pytest. The lesson here is not anti-pytest; it is *write your assertions as systems, then put them inside a pytest function so pytest runs them.* The system shape and pytest's harness are orthogonal.

**`unittest.mock` is the wrong tool for ECS-style code.** The boundary-as-queue rule eliminates the things mocks exist to fake — there are no external services to patch, no `requests.get` to intercept, no clocks to freeze. If you find yourself reaching for `mock.patch`, the system you are testing has a leak from §35; the fix is to plumb the leaked dependency through the queue, not to mock it. The simlog's [test_simlog.py](https://github.com/root-11/intro-book-python/blob/main/.archive/simlog/test_simlog.py) (713 lines, full coverage of the simlog's contract) uses zero mocks — every test sets up real numpy arrays, runs real `log()` calls, and reads back the real `.npz` output.

**Property-based testing belongs here.** `hypothesis` is the Python ecosystem's property-based-testing library; it generates inputs and shrinks failures. For systems whose read-set is well-typed numpy columns, `hypothesis` integrates cleanly via `hypothesis-numpy`. The simulator's invariants ("population stays bounded", "energy is non-negative", "no slot has two ids") are perfect property-test material — let `hypothesis` generate the world states; assert the invariants on each.

## The TDD-from-day-one piece

From [§5](05_identity_is_an_integer.md) onward, every concept in the book is approached test-first. *What's the smallest case? What's the largest? What should the answer be for `np.uint8`, for `np.uint32`, for 10,000 entity ids?* The deck-game exercises start by asking "what should this return for a deck of 0 cards, of 1, of 52?" The simulator's exercises ask "what should population be after 100 ticks of zero food?" Tests come first; implementation follows.

The discipline pays off three ways:

- **Tests grow with the code.** Each new system has its tests as adjacent functions, sharing the same read/write conventions. A test refactor is no different from a system refactor.
- **Inspection and testing are the same code.** The inspection-system pattern from [§13](13_system_as_function.md) is identical to the test pattern: read-only access to all tables, output a report. In production, inspection is absent or running in `--debug` mode; in test, it is present and asserting. Same source code, different schedule.
- **Determinism makes tests trustworthy.** [§16](16_determinism_by_order.md)'s rule means tests are reproducible. A test that fails with seed `0xCAFE` fails with `0xCAFE` every time, on every machine — provided you respected the §16 recipe (no raw set iteration, no wall clock in systems, one seeded RNG). pytest-xdist running 8 parallel workers will surface set-iteration bugs that single-process pytest will not, exactly as §16 exercise 7 predicted.

## The book is closing

Forty-three concepts; ten phases; one through-line simulator. The disciplines named in this last phase — mechanism vs policy, compression-oriented programming, you-can-only-fix-what-you-wrote, tests-are-systems — are the rules that hold the rest together. They are not new architecture. They are how the architecture earlier chapters built stays maintainable.

A simulator that respects all forty-three nodes is one whose state is in numpy columns, whose transformations are systems, whose tick is a pure function, whose history is a log, whose persistence is transposition, whose tests are systems, and whose dependencies are bets you took with your eyes open.

That is the data-oriented program. That is the book.

## Exercises

1. **A test as a system.** Take the `no_creature_moves_too_far` system from the prose. Add it to your simulator's DAG behind a `--test` flag. Run for 100 ticks. The system should report zero suspicious creatures.
2. **A property test.** Run the simulator for 1000 ticks with seed `0xCAFE`. Assert: `world.n_active <= 2 * initial_n_active`. Run twice with the same seed; both runs should report the same outcome (passing or failing at the same tick).
3. **A replay test.** Save the in-queue of a 100-tick run via [§36](36_persistence_is_serialization.md)'s `np.savez`. Load it into a fresh simulator and replay. After 100 ticks, hash both worlds. They must match.
4. **TDD a new system.** Pick a piece of behaviour you have not built — say, "creatures with energy above 50 grow more slowly". Write the test first: what's the smallest case (one creature)? Largest (a million)? Then write the system. Confirm the test passes.
5. **Read the simlog tests.** Open `.archive/simlog/test_simlog.py`. Note the absence of mocks. Note that every test fixture is a real numpy array set up in the test body. The test file is 713 lines for a 700-line library — roughly 1:1, which is the right ratio for code that has to work.
6. **The InspectionSystem connection.** Take the test from exercise 1 and the inspection-system idea from [§13](13_system_as_function.md). Argue why they are structurally identical — same read-set, same lack of write-set, same scheduling slot.
7. **pytest-xdist as a determinism check.** Convert your test suite to run under `pytest -n 8` (parallel workers). Any test that passes under `pytest` but fails under `pytest -n 8` has a non-determinism leak (often a `set` iteration, often a wall clock). Fix the leak; the §16 recipe is the remedy.
8. *(stretch)* **A test runner that *is* the simulator's scheduler.** Implement a tiny test runner whose only difference from the simulator's scheduler is *which* systems it includes in the DAG: production systems for live runs, test-and-inspection systems for test runs. The two binaries share most of their code; the difference is the systems list.

Reference notes in [43_tests_are_systems_solutions.md](43_tests_are_systems_solutions.md).

## What's next

You have closed the trunk. [§44 — What you have built](44_closure.md) looks back at the shape of what you built and opens the questions the book deliberately did not settle.
