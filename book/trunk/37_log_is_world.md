# 37 — The log is the world

> *Concept node: see the [DAG](../../concepts/dag.md) and [glossary entry 37](../../concepts/glossary.md#37--the-log-is-the-world).*

<p align="center"><img src="../illustrations/model_real_world.jpg" alt="Model the real world — the log is the world reconstructed step by step" style="max-height: 300px; max-width: 100%;"></p>

[§36](36_persistence_is_serialization.md) said persistence is transposition: the in-memory tables are written as their bytes, read back as their bytes. This section makes the deeper structural claim. **The log is the world**, and the world is the log decoded.

In an event-sourced simulator, every state change is an event:

```text
(tick=42, kind=become_hungry, creature_id=17)
(tick=42, kind=eat,           creature_id=23, food_id=8, energy_delta=+5.0)
(tick=43, kind=reproduce,     parent_id=14, offspring_id=400, offspring_energy=2.5)
(tick=43, kind=die,           creature_id=89)
```

The log is a sequence of such events. The world's tables can be reconstructed from the log: start from an empty world (or a snapshot), replay events in order, and the resulting tables are bit-identical to the world the live simulator produced.

The structural fact: **the log and the world have the same shape**.

A presence table `hungry: np.ndarray` is a list of creature ids. The log of `become_hungry` and `stop_being_hungry` events is a list of (tick, creature_id) pairs that, when replayed, produces the same array. A column `energy: np.ndarray` is the result of starting from an empty array plus the events that wrote each entry. The log holds these writes; the column is the cumulative effect of replaying them.

In the most explicit form — the *triple-store* shape — the log is three parallel numpy columns:

```python
rids: np.ndarray  # uint32 — which entity (the row id)
keys: np.ndarray  # uint8  — which column (a numeric code)
vals: np.ndarray  # float64 — the value to write
```

The triples form the log; transposed, they form the columns. **Transposition is the only translation. There is no impedance mismatch because there is no model gap.**

## Not the `logging` module

The Python instinct on hearing "log every state change" is to reach for the standard library's `logging` module. **The `logging` module is not the right tool for this job.** It is for human-readable diagnostic output — formatted strings, timestamps, severity levels, file rotation. The state-change log this chapter is about is structured, queryable, and replayable. Different tool for different job.

```python
# anti-pattern: bad!
import logging
logger = logging.getLogger("simulator")
logger.info(f"creature {cid} ate food {fid}, energy_delta={delta}")
```

What that line writes to disk is a string. To replay, a downstream tool would have to parse the string back into structured fields — exactly the translation [§36](36_persistence_is_serialization.md) said does not exist in this architecture. You have re-introduced the ORM trap one print call at a time.

The disciplined Python form: append the structured event to numpy columns, write the columns as bytes. The format on disk is the format in memory. No parsing, no parsing-bug, no cost.

## The simlog: a working specimen

The library [`.archive/simlog/logger.py`](../../.archive/simlog/logger.py) implements this triple-store shape directly, in Python, in 700 lines. Its design is worth walking through, because it meets three problems that recur whenever a simulator wants to log everything, and the conclusions it reaches are not specific to any one language or domain.

**The IOPS problem → batching.** A naive event logger calls `f.write` once per event. At a million events per minute, that is a million disk operations per minute — bound by IOPS, not bandwidth ([§38](38_storage_systems.md)). The disk's bandwidth sits mostly idle while it queues operations. The fix: collect events into an in-memory buffer; when the buffer fills, flush it as one large write. IOPS scales with "buffer flushes per second"; bandwidth absorbs the actual byte volume. Logging cost drops from disk-latency-bound to bandwidth-bound — typically 100-1000× faster. **This is the same pattern as §22's cleanup amortisation, applied at the disk boundary.**

**The redundancy problem → codebook and type inference.** Most fields in a simulator's event records repeat: the same kind code thousands of times, the same set of activity strings, the same handful of entity types. Storing each event's full payload wastes bytes. The fix: a *codebook* assigns each unique string a small integer code; the log stores the code, not the string. On read, the codebook reverses the mapping. simlog goes one step further with type inference — every value is stored as one `f64` (8 bytes), regardless of whether it began as an integer, a float, or a string code. Integers up to 2⁵³ round-trip exactly; the union format eliminates per-field type tags. The savings compound: at typical 5% field density, the format uses roughly 6× less memory than dense column arrays.

**The write-blocking problem → double-buffered pointer switch.** If the simulator blocks while the disk flushes, the simulation pauses on every flush. The fix: two `Container` instances, each holding a tunable number of rows (200,000 by default). When one fills, the foreground thread hands it to a background thread for flush; new events keep going to the other. When the flush completes, the containers' roles swap — a single pointer switch, often called the *revolver*. From the simulator's perspective, writing an event is one push to a numpy column, never a wait on disk. *This is the same pattern as §15's "world is frozen during a tick" applied at the producer/consumer boundary instead of the system/system boundary.*

The combined result: simlog's `log()` call costs roughly **0.9-1.9 µs** per event on this author's machine (faster at fewer fields per row, slower at many — published benchmarks show 934 ns at 5 fields, 1906 ns at 11). The hot-path output is a sequence of `.npz` chunks written sequentially by the background thread (`_write_chunk`); the simulator's `log()` never waits on disk. Auxiliary methods (`to_csv`, `to_sqlite`) read the `.npz` chunks back *after* the simulation and convert them for downstream consumers — post-processing, not part of the live logging path.

The structural identity — log = world — holds across all these formats; what changes is the storage system at the boundary ([§38](38_storage_systems.md)).

The library does not need to know what an "event" is. It stores triples; the consumer interprets them. That separation is what makes the same code serve as a simulation logger, an audit trail, and a replay source — three uses, one structural pattern.

## Why this matters in practice

**Replay is structural.** Snapshot + log = pause/resume. To recover the world at any tick T, load the most recent snapshot at tick S ≤ T, then replay the log from S to T. The cost is bounded by `T − S` events, which is small if snapshots are taken regularly.

**Auditability is free.** Every change in the world is in the log. To answer "why is creature 17 dead?", scan the log for events involving 17. The log is the system's complete history, in order.

**Testing is replay.** A test fixture is an initial world plus a log. A test is "replay this log; assert this property of the result". No `unittest.mock`, no setup fixtures, no `pytest.fixture` builders mocking out time and random.

**Distribution is structural.** Two nodes running identical code from the same log produce bit-identical worlds. Send the log; the worlds converge.

**The log is the system of record.** Snapshots are caches of the log's state; they exist for performance, not correctness. If snapshots are lost, the log can rebuild them. If the log is lost, no snapshot can recover events that have not been logged.

## The discipline

The discipline that makes this work is structural, not stylistic. *Every state change in the simulator is logged before being applied.* The cleanup pass ([§22](22_mutations_buffer.md)) is the natural place — it sees every mutation and can record each one as it commits. The [§38](38_storage_systems.md) storage system is the natural sink — log writes are sequential, batched, and amortised across the tick.

A simulator that respects this discipline is one whose history is the log, whose state is a projection of the log, and whose persistence is the log plus the most recent snapshot.

## §35 and §37 together

Read the last two chapters as one architecture. [§35](35_boundary_is_the_queue.md) says the simulator's external interface is a structured queue: inputs arrive in one place, outputs leave in one place, no system reads the environment directly. §37 says the simulator's historical record is a structured log: state changes are batched, deduplicated through a codebook, and written through a double-buffered revolver. **Together they describe an event-sourced architecture with the simulator as the deterministic reducer.**

The combination buys four properties that most Python systems give up because they are hard to maintain by hand:

- **Replay free.** Rerun the log; get the same world.
- **Testing free.** A fixture is `(initial_world, input_log)`; a test asserts on the result. No mocks, no fixture builders, no dependency injection.
- **Distribution free.** Send the log between nodes; worlds converge by construction.
- **Auditing free.** The log is the audit. The question "what happened to creature 17?" is one `np.where` away.

The high-performance properties fall out of the same shape:

- **Queues amortise syscalls** — no per-event kernel transition.
- **Logs amortise disk writes** — no per-mutation flush.
- **Cleanup batches both** — one pass per tick produces one queue drain and one log batch.
- **The worker pool stays warm** across all of it ([§31](31_disjoint_writes_parallelize.md)).

**Every architectural choice in Parts 1-7 was chosen so that this final architecture would compose.** Numpy SoA so the queue and the log share shape with the world. Single-writer ownership so cleanup can batch without races. Determinism so replay round-trips. EBP so the log of `become_hungry` events *is* the `hungry` table at any later tick. Index maps so id-based references survive the swap_remove pass that the cleanup applies. None of it was preparation; all of it was building toward this seam.

The remaining chapters — Part 8 closing with [§38](38_storage_systems.md), Part 9, Part 10 — are operational concerns and meta-discipline. The structural answer for a high-performance Python simulator is now in place.

## Exercises

1. **Log the simulator.** Add three parallel numpy columns (`rids: uint32`, `keys: uint8`, `vals: float64`) plus an `n_events` counter to your world. Modify the cleanup pass to push one triple per applied mutation. After 100 ticks, the log has roughly `active × ticks` triples.
2. **Reconstruct from the log.** Write `def replay(initial: World, events: TripleStore) -> World` that applies each triple in order. Verify: starting from an initial world and applying the log produces a world identical to the live simulator's output at the same tick. Hash both with the §16 `hash_world` function.
3. **Save and load the log.** Persist the triple-store via [§36](36_persistence_is_serialization.md)'s `np.savez`. Reload. Replay. Confirm bit-identical state.
4. **Snapshot + log.** Save a snapshot at tick S; save the log from tick S onward. Reconstruct any tick T > S by loading the snapshot and replaying the log from S to T. Verify against the live simulator.
5. **Run simlog.** Open `.archive/simlog/logger.py` and trace the `log()` call: what does it touch in memory, what does it not touch on disk, when does the swap happen, when does the disk write occur. Sketch the call graph on paper. The 700 lines you read are 700 lines you will not have to write.
6. **The codebook saving.** With 1,000,000 events of which all are `kind="eat"`, compare two storage forms: storing the literal string `"eat"` per event vs storing a `uint8` code with a one-row codebook. The codebook form is ~24× smaller (1 byte vs 24 bytes for the short string plus Python object overhead) and round-trips losslessly.
7. **The `logging` module trap.** Configure Python's standard `logging` module to write events to a file, one per `eat`. Generate 100,000 events. Now write the same events into a numpy triple-store. Compare: file size, write time, time to query "how many eat events involved creature 42?". The triple-store form is faster on every axis and the query is a single `np.where`.
8. *(stretch)* **The simlog API, three views.** Sketch the API for a hypothetical simlog-v2 in three forms:
   - **As a class.** `class Simlog: def log(self, **fields): ...; def to_arrays(self): ...`. Reusable across simulators; pip-installable.
   - **As a module** inside your simulator. Same shape, but accessing the simulator's existing types directly without crossing a package boundary. Less reusable, more efficient — no public API to keep stable.
   - **As an ECS system.** A logging system whose read-set is `to_remove`, `to_insert`, and any other commit-time tables, and whose write-set is the log columns. It runs in the same DAG as `cleanup`, perhaps merged with it. The two halves of cleanup — committing mutations and logging them — become one system.

   Implement none, sketch all three. Compare what each form gains and loses: reusability, performance, ease of testing, distance from the simulator's other concerns.

Reference notes in [37_log_is_world_solutions.md](37_log_is_world_solutions.md).

## What's next

[§38 — Storage systems: bandwidth and IOPS](38_storage_systems.md) names the cost of crossing the I/O boundary in concrete terms. The log lives there; so does the snapshot; so does every external connection.
