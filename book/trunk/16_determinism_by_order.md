# 16 — Determinism by order

> *Concept node: see the [DAG](../../concepts/dag.md) and [glossary entry 16](../../concepts/glossary.md#16--determinism-by-order).*

<p align="center"><img src="../illustrations/monte_carlo.jpg" alt="Monte Carlo estimate of π — same seed, same answer, every run" style="max-height: 300px; max-width: 100%;"></p>

A program is *deterministic* if the same inputs and the same execution produce the same outputs, every time. Sounds obvious. It is not — most modern Python programs are *not* deterministic by default. Threads run in OS-scheduled order. Sets iterate in randomised order across processes. The system clock differs by run. `random.random()` reads from a global instance whose state depends on import order and prior calls.

In an ECS architecture, determinism is structural. Same world state at tick start + same system order + same inputs (events, RNG seed) = same world state at tick end. Bit-identical. Every time.

This is not a quality goal; it is a precondition for almost everything the book builds on:

- **Replay.** The world is the log decoded ([§37](37_log_is_world.md)). Replay reconstructs world state by re-running the inputs through the same system sequence. Without determinism, replay is impossible.
- **Testing.** A property test fixes an RNG seed and asserts the simulator behaves identically across runs. Without determinism, every test is flaky.
- **Distributed simulation.** Multiple machines run identical copies of the world. Without determinism, they drift apart by tick 1.
- **Debugging.** A bug at tick 4783 should appear at tick 4783 every run. Without determinism, debugging real-time bugs becomes guesswork.

## The recipe, Python edition

The recipe for determinism is to forbid every source of non-determinism in the inner systems. In Python the sources have specific names.

**No raw set iteration.** From [`code/measurement/set_iteration_order.py`](https://github.com/root-11/intro-book-python/blob/main/code/measurement/set_iteration_order.py), three fresh subprocesses iterating the same six-element set produced three different orders:

```
run 1: delta,foxtrot,echo,bravo,charlie,alpha
run 2: bravo,foxtrot,delta,echo,alpha,charlie
run 3: echo,delta,foxtrot,charlie,bravo,alpha
```

CPython hashes strings using a per-process random seed (`PYTHONHASHSEED`), and `set` iteration order is a function of the hash table's bucket layout. Across processes, the layout differs; the iteration order differs. This is *by design* — it protects servers from hash-flooding attacks — but it is also a source of non-determinism that the simulator forbids. **Never iterate a set inside a system.** If you need an iteration order, use a sorted list, a numpy array, or a `dict` (which is insertion-ordered since CPython 3.7 and survives the same test):

```
run 1: alpha,bravo,charlie,delta,echo,foxtrot
run 2: alpha,bravo,charlie,delta,echo,foxtrot
run 3: alpha,bravo,charlie,delta,echo,foxtrot
```

**No system clock inside a system.** Get time from input events, not from `time.time()` or `time.perf_counter()`. Time is a value passed *into* the system, not read from the OS. The tick loop's outer scaffolding may read the wall clock; the systems inside the tick may not.

**One RNG, seeded.** A single `np.random.default_rng(seed)` per simulator instance, used in a defined order. Each system that needs randomness reads from it in DAG order. Never `random.random()` (reads global state), never `np.random.random()` without the rng object (uses a global). Pass the rng as a parameter — it has a declared read-set just like any other input.

**No threads inside a system.** A system runs single-threaded internally. The GIL does not save you from non-determinism here; it serialises Python bytecode but not the *order* in which threads acquire it. Parallelism happens *between* systems with disjoint write-sets ([§31](31_disjoint_writes_parallelize.md)) using `multiprocessing`, not inside one system using `threading`.

**Buffered mutations.** [§15](15_state_changes_between_ticks.md)'s rule: mutations apply at tick boundaries, not mid-tick.

**One Python-specific footnote: `hash()` itself.** Hash randomisation has been on by default since CPython 3.3 for `str` and `bytes` (and the containers that derive from them, including `frozenset`). If a system computes `hash(some_string)` and uses that value as part of its output, the output is non-deterministic across processes. Use `hashlib.blake2b(s.encode()).digest()` — or any deterministic hash — when you need a stable hash inside a system.

These rules are restrictive. They are also the price of every benefit listed above. Most modern Python programs decline to pay this price and accept the costs — flaky tests, unreproducible bugs, divergent distributed simulation. The book pays the price.

## The cost is at the boundary, not in the body

The cost of determinism is not absolute. *Within* a system, the implementation is free to use whatever it likes — vectorised numpy, low-level optimisations, even occasional non-deterministic libraries — as long as the inputs and outputs are bit-identical to what the abstract specification demands. The discipline is at the system boundary: between systems, everything must be reproducible.

Inside `motion`, you can use `pos_x += vel_x * dt` (numpy bulk op, deterministic) or `np.einsum` or write your own Cython kernel. As long as the output `pos_x` for given inputs is bit-identical across runs, the system is deterministic regardless of how its internals work. The contract is at the function boundary; the freedom is inside.

## Testing for determinism

A test for determinism is concrete. Run the simulator twice with the same seed, the same input event log, the same system order. After 1,000 ticks, hash the entire world state — feed every numpy column through `hashlib.blake2b(arr.tobytes()).hexdigest()` and combine. If the hashes match, you are deterministic. If they do not, find the system whose output first differs, and trace the source of variability. Often: a `set` iterated, a `time.time()` call, a `random.random()` reading global state.

A simulator that is deterministic is also a simulator that *can be tested*. Once that property holds, every other quality goal — performance, parallelism, distribution — becomes safe to optimise toward. Without determinism, every optimisation is a coin flip.

The full payoff of determinism arrives at the *save and load* phase named in [§11](11_the_tick.md). The simulator can be paused, its tables serialised to disk, reloaded later, and resumed — and the result must be indistinguishable from a run that never paused. The mechanics arrive in [§36 — Persistence is table serialization](36_persistence_is_serialization.md): a snapshot is the world's columns written as `.npz` files — the same bytes they have in memory. Combined with the input event log, replay is structural — read the snapshot, replay events through the same DAG with the same seed, you reconstruct the world at any later tick exactly. Determinism (this section), serialization ([§36](36_persistence_is_serialization.md)), and log-as-world ([§37](37_log_is_world.md)) are the three legs of replay.

## Exercises

1. **Run the iteration-order exhibit.** `uv run code/measurement/set_iteration_order.py`. Observe the set rows differ; the dict rows do not. Note that the dict survival is *not* a guarantee against `frozenset` keys, `dict.values()` derived from a set, or any operation that goes through hash bucket order — only the surface-level "I added these in order" pattern survives.
2. **Hash the world.** Write `def hash_world(world) -> str` that produces a hex digest by feeding every column through `hashlib.blake2b(arr.tobytes()).update(...)`. Use this to compare world states across runs.
3. **Two identical runs.** Run the simulator twice with the same RNG seed (`np.random.default_rng(42)`) and the same input events. Hash the world at tick 100. Confirm they are equal.
4. **Introduce non-determinism deliberately.** Replace your seeded `default_rng(42)` with `np.random.default_rng()` (no seed — uses entropy). Run twice. Show the hashes differ.
5. **Find the culprit.** Suppose your hashes differ. Hash the world after each system in the DAG. Identify which system's output first differs, and what source of non-determinism it pulls from. Common offenders: `for k in some_set:`, `time.time()`, `random.random()`, `hash(some_string)`.
6. **Time as input.** Find a system that uses `time.perf_counter()` and refactor it to instead take `current_time: float` as a parameter. The system is now deterministic; the source of `current_time` is the only place non-determinism can enter.
7. **The set trap up close.** Build a `set` of 1,000 random integers (use a `default_rng(42)` so the set contents are deterministic). Iterate it three times *in the same process*. Are the orders the same? Now run the program twice in two fresh shells. Are the orders the same across runs? (Hint: the answers are *yes* and *no*, in that order. The trap is that a single test run will not catch the bug; two test runs in two CI workers will.)
8. *(stretch)* **A property test.** Hand-roll a simple property test: generate 100 random seeds. For each, run the simulator for 100 ticks. Hash the resulting world. Verify that the same seed always produces the same hash, and that different seeds usually produce different hashes.

Reference notes in [16_determinism_by_order_solutions.md](16_determinism_by_order_solutions.md).

## What's next

You have closed Time & passes. Determinism is structural; replay is architectural; the next phase is *Existence-based processing*, starting with [§17 — Presence replaces flags](17_presence_replaces_flags.md). The simulator's hunger and starvation systems are about to lose their booleans.
