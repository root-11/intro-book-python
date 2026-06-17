# 44 - What you have built

The previous forty-three sections were a long climb. This one is a look down.

You have built a small ecosystem simulator that runs deterministically, scales from one hundred creatures to streaming workloads, and exposes its state to inspection at every tick. You did this with `numpy` arrays and functions - no class hierarchies, no ORM, no framework, no async runtime. The discipline that made it work is the entire content of the book.

## The shape that carried the whole thing

<p align="center"><img src="../illustrations/mathematics_describes.jpg" alt="Mathematics describes, models, implements, and improves the world." style="max-height: 300px; max-width: 100%;"></p>

Three patterns showed up everywhere:

**Tables, not objects.** A creature is not a `class` of fields with methods. It is a row across columns kept aligned by index - `pos_x[i]`, `pos_y[i]`, `energy[i]`. Each column is a numpy array. The columns have one writer each; they grow and shrink in lockstep. There is no container holding them together - only the discipline.

**Systems, not state.** Behaviour is a function over tables. `motion` reads `vel`, writes `pos`. `apply_starve` reads `energy`, pushes ids to `to_remove`. Each system has a name, a read-set, a write-set. The simulator is the DAG of systems composed in order. State changes happen between ticks, not inside them.

**Mechanism separated from policy.** The kernel exposes verbs (insert, remove, swap, push to buffer, batched cleanup). The rules live at the edges (when does a creature die, when does food spawn, what counts as a collision). The same kernel runs every variation; the policies change without it.

Those three are not Python-specific. They are not even ECS-specific. They are what data-oriented design names. The rest of the book - locality, parallelism, persistence, anytime algorithms - falls out of taking those three seriously.

## What this approach buys, in Python specifically

- **Speed by default**, because numpy SoA layout matches the machine and the inner loops escape the interpreter.
- **The answer to "Python is slow."** Python is slow when it is the inner loop. When numpy is the inner loop and Python is the orchestration, Python is *not* slow - it is exactly the right level of abstraction for the orchestration.
- **Determinism without locks**, because ordering is the contract and the GIL is no longer in the picture once you partition work into multiprocessing+shared_memory ([§31](31_disjoint_writes_parallelize.md)).
- **Testability**, because each system is a pure function over its inputs. No `unittest.mock`, no monkey-patching, no framework-specific magic.
- **Onboardability**, because the data is visible. A reader can `print(column[:10])` for any column and see the world.
- **Refactor cheap**, because there are no objects with hidden state to migrate, no `Optional[X]` fields whose meaning depends on context, no inheritance chains to follow.

## What this approach costs

- **Less abstraction.** You feel the machine. Some find this freeing; some find it exhausting.
- **More discipline.** Single-writer rules, mutation buffering, lockstep sorts - Python does not enforce these. You do. The borrow checker is not coming to save you.
- **Less idiomatic Python.** The book uses very little of what Python tutorials teach: no class hierarchies, very few decorators, no `Protocol`, no `pydantic`, no ORM. Idiomatic Python looks different. Engineers trained on the standard idioms will find this code surprising; the surprise is the point.
- **A different mental model.** Engineers trained in OOP will not naturally reach for tables. The translation cost is real.

## Open questions the book did not settle

The book made choices. Other books make different ones. Worth knowing where you sit:

- **Why not Bevy, or another existing ECS framework?** Faster to start, harder to see through. We did the slow thing on purpose. After §43 you can read Bevy's ECS source (or any production ECS) and tell whether its choices match yours.
- **Is a row really better than a class?** For a single creature, no - `class Card(suit, rank)` is fine. For a million, yes - the §3 measurements settle that. The crossover depends on your workload; the book named the tradeoff but did not prescribe.
- **Could this have been Rust, or Zig, or C?** Yes. The ideas are language-independent. Python contributes accessibility and the numpy ecosystem; the rest is layout discipline. The Rust edition of this book exists for readers who want compile-time guarantees on what this edition enforces by convention.
- **What about typing, dataclasses, async?** Two of Python's most-promoted features barely appear in the trunk. `typing` and `dataclass` show up at boundaries (function signatures, configuration objects, named references like `CreatureRef`); they do not earn their place inside hot loops. `async` does not appear at all - the simulator is CPU-bound and synchronous; async is for I/O-bound systems whose orchestration is genuinely waiting on external events. Future work might explore where each of these *does* pay rent in a Python ECS - usually at the edges (CLI parsing, configuration, network I/O at the boundary) rather than the kernel.
- **What about networking and rollback?** §31-§34 covers single-machine concurrency. Distributing the world across machines is a different book - the network-hop tax (§39) makes it the wrong default for tick-rate work; reach for it only when one box genuinely cannot hold the workload.
- **What about pandas, ORMs, async frameworks?** They earn their place when the workload genuinely fits their compression ([§41](41_compression_oriented.md), [§42](42_you_can_only_fix_what_you_wrote.md)). For a simulator whose data is columnar SoA and whose tick is CPU-bound, none of them fit. For other workloads they may. The discipline is to *decide consciously*, not to default to the popular tool.

## Two acts: building it, and living with it

Read back, the book has two acts. The first is *building something that works, and lasts*. Sections 1-39 made it run - deterministic, scaled from a hundred creatures past the million-entity wall, parallel across processes on disjoint writes, persisted and replayable. Sections 40-43 made it durable to *change*: mechanism vs policy, deferred abstraction, dependency pricing, tests-are-systems - the discipline that holds four of the five costs of ownership: extendibility, maintainability, performance, and memory.

The second act is *living with it* once it is in service - a different question entirely. The fifth cost of ownership, **operations** - recovering it, observing it, trusting it across machines and deadlines - only bites when the system is deployed and the human who used to watch it is gone. That act begins in [§45](45_living_with_it.md).

## The horizon: living with it at production scale

The open questions above are choices of taste - other books choose differently. This list is not. It is where what the first act built leaves a real gap the moment the system is in service. Each gap is named against the criterion it threatens; the second act sets out on them, beginning with the operations leg.

- **Crash consistency** (operations). "The log is the world" holds only while the log survives power loss. Torn writes, fsync barriers, atomic rename, idempotent replay after a half-written batch - [§38](38_storage_systems.md) names fsync once and stops. The second act builds the rest in [§46](46_log_survives_power_loss.md).
- **Observability** (operations). "The data is visible; `print()` every column" is a debugger's story, not an on-call engineer's at 2 AM. Metrics, tracing, structured logs, and alerting want to be read-only systems - [§47](47_observation_is_a_system.md).
- **Numerical determinism under parallelism** (operations). Same seed, different worker count, different bits - the parallel-reduction gotcha named in [§16](16_determinism_by_order.md). Replay across heterogeneous hardware needs a fixed reduction order or integer accumulation - [§48](48_reductions_dont_parallelize_freely.md).
- **Hard real-time** (operations). [§39](39_system_of_systems.md)'s anytime algorithms are soft real-time. Hard real-time - where a missed deadline is a fault - needs a worst-case bound that, as [§49](49_worst_case_is_the_only_case.md) shows, CPython cannot give. Knowing that line is the lesson.
- **Schema evolution** (extendibility). [§36](36_persistence_is_serialization.md) versions a save with a header. Renaming a column, splitting one, back-filling a derived column - each is a project, and every `.npz` in the wild is a hostage to today's layout. The triple-store of [§37](37_log_is_world.md) is the start of a fix; schema-as-data is the rest. A road for a later volume.
- **Heterogeneous compute** (performance). SoA is the precondition for SIMD, GPU offload, and accelerators; the book leaves the interpreter for numpy and stops at one box's bandwidth. The next bus - and its cost model of transfer bandwidth and launch latency - is a road for a later volume.
- **Where SoA does not pay** (memory, maintainability). Recursive structures dominated by topology, very small N where pointer-chasing's constant factor wins, and APIs that must hand structured rows to non-array consumers are where columns can cost more than they save. SoA is a default, not a law.
- **Floating-point geometry** (correctness). Data layout is orthogonal to degeneracies and robust predicates: a perfectly columnar geometry kernel is still wrong on collinear points. The book admits this exists for readers building CAD, GIS, or path planning.
- **The social layer** (maintainability). Code review, ownership transfer, deprecation, runbooks. "Onboardable because the data is visible" is one bullet; the rest of the team-scale layer is where every criterion above degrades fastest under turnover.

The first act is the harder problem, and the book finishes it. The second act - ship, evolve, observe, recover - begins now, in [§45](45_living_with_it.md).
