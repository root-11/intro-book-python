# 44 — What you have built

The previous forty-three sections were a long climb. This one is a look down.

You have built a small ecosystem simulator that runs deterministically, scales from one hundred creatures to streaming workloads, and exposes its state to inspection at every tick. You did this with `numpy` arrays and functions — no class hierarchies, no ORM, no framework, no async runtime. The discipline that made it work is the entire content of the book.

## The shape that carried the whole thing

<p align="center"><img src="../illustrations/mathematics_describes.jpg" alt="Mathematics describes, models, implements, and improves the world." style="max-height: 300px; max-width: 100%;"></p>

Three patterns showed up everywhere:

**Tables, not objects.** A creature is not a `class` of fields with methods. It is a row across columns kept aligned by index — `pos_x[i]`, `pos_y[i]`, `energy[i]`. Each column is a numpy array. The columns have one writer each; they grow and shrink in lockstep. There is no container holding them together — only the discipline.

**Systems, not state.** Behaviour is a function over tables. `motion` reads `vel`, writes `pos`. `apply_starve` reads `energy`, pushes ids to `to_remove`. Each system has a name, a read-set, a write-set. The simulator is the DAG of systems composed in order. State changes happen between ticks, not inside them.

**Mechanism separated from policy.** The kernel exposes verbs (insert, remove, swap, push to buffer, batched cleanup). The rules live at the edges (when does a creature die, when does food spawn, what counts as a collision). The same kernel runs every variation; the policies change without it.

Those three are not Python-specific. They are not even ECS-specific. They are what data-oriented design names. The rest of the book — locality, parallelism, persistence, anytime algorithms — falls out of taking those three seriously.

## What this approach buys, in Python specifically

- **Speed by default**, because numpy SoA layout matches the machine and the inner loops escape the interpreter.
- **The answer to "Python is slow."** Python is slow when it is the inner loop. When numpy is the inner loop and Python is the orchestration, Python is *not* slow — it is exactly the right level of abstraction for the orchestration.
- **Determinism without locks**, because ordering is the contract and the GIL is no longer in the picture once you partition work into multiprocessing+shared_memory ([§31](31_disjoint_writes_parallelize.md)).
- **Testability**, because each system is a pure function over its inputs. No `unittest.mock`, no monkey-patching, no framework-specific magic.
- **Onboardability**, because the data is visible. A reader can `print(column[:10])` for any column and see the world.
- **Refactor cheap**, because there are no objects with hidden state to migrate, no `Optional[X]` fields whose meaning depends on context, no inheritance chains to follow.

## What this approach costs

- **Less abstraction.** You feel the machine. Some find this freeing; some find it exhausting.
- **More discipline.** Single-writer rules, mutation buffering, lockstep sorts — Python does not enforce these. You do. The borrow checker is not coming to save you.
- **Less idiomatic Python.** The book uses very little of what Python tutorials teach: no class hierarchies, very few decorators, no `Protocol`, no `pydantic`, no ORM. Idiomatic Python looks different. Engineers trained on the standard idioms will find this code surprising; the surprise is the point.
- **A different mental model.** Engineers trained in OOP will not naturally reach for tables. The translation cost is real.

## Open questions the book did not settle

The book made choices. Other books make different ones. Worth knowing where you sit:

- **Why not Bevy, or another existing ECS framework?** Faster to start, harder to see through. We did the slow thing on purpose. After §43 you can read Bevy's ECS source (or any production ECS) and tell whether its choices match yours.
- **Is a row really better than a class?** For a single creature, no — `class Card(suit, rank)` is fine. For a million, yes — the §3 measurements settle that. The crossover depends on your workload; the book named the tradeoff but did not prescribe.
- **Could this have been Rust, or Zig, or C?** Yes. The ideas are language-independent. Python contributes accessibility and the numpy ecosystem; the rest is layout discipline. The Rust edition of this book exists for readers who want compile-time guarantees on what this edition enforces by convention.
- **What about typing, dataclasses, async?** Two of Python's most-promoted features barely appear in the trunk. `typing` and `dataclass` show up at boundaries (function signatures, configuration objects, named references like `CreatureRef`); they do not earn their place inside hot loops. `async` does not appear at all — the simulator is CPU-bound and synchronous; async is for I/O-bound systems whose orchestration is genuinely waiting on external events. Future work might explore where each of these *does* pay rent in a Python ECS — usually at the edges (CLI parsing, configuration, network I/O at the boundary) rather than the kernel.
- **What about networking and rollback?** §31-§34 covers single-machine concurrency. Distributing the world across machines is a different book — the network-hop tax (§39) makes it the wrong default for tick-rate work; reach for it only when one box genuinely cannot hold the workload.
- **What about pandas, ORMs, async frameworks?** They earn their place when the workload genuinely fits their compression ([§41](41_compression_oriented.md), [§42](42_you_can_only_fix_what_you_wrote.md)). For a simulator whose data is columnar SoA and whose tick is CPU-bound, none of them fit. For other workloads they may. The discipline is to *decide consciously*, not to default to the popular tool.

## Where to go next

- **Read Mike Acton's "Data-Oriented Design and C++"** (CppCon 2014). Forty-five minutes; the most concentrated case for this approach you will find.
- **Read Casey Muratori's *Handmade Hero*** episodes on grid storage and cache locality. Another route to the same conclusions.
- **Open Bevy's `bevy_ecs` crate** (Rust) or any production ECS in the language of your choice. You will recognise every pattern. The names will differ; the shapes are identical.
- **Read the Rust edition of this book.** Same architecture, different enforcement. Watching the borrow checker enforce what this edition asks you to do by discipline is a genuinely useful calibration.
- **Extend the simulator.** The genetics and predator-prey extensions flagged in the [simulator spec](../../code/sim/SPEC.md) break new ground without leaving the framework you have already built.
- **Apply the architecture beyond simulators.** §35 + §37 is event-sourced architecture with a deterministic reducer; the same pattern works for request handlers, control loops, agent systems, anything with state that has to evolve under load. The simulator was the worked example; the architecture is the lesson.

<p align="center"><img src="../illustrations/model_real_world.jpg" alt="Model the real world." style="max-height: 300px; max-width: 100%;"></p>

The book ends here. The simulator does not — it runs as long as you keep the discipline.
