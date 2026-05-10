# 35 — The boundary is the queue

<p align="center"><img src="../covers/phase_io_persistence.jpg" alt="I/O & persistence phase" style="max-height: 380px; max-width: 100%;"></p>

> *Concept node: see the [DAG](../../concepts/dag.md) and [glossary entry 35](../../concepts/glossary.md#35--the-boundary-is-the-queue).*

The simulator is a pure function. Given the world at tick start (`world_t`) and the inputs that arrived during the tick (`inputs_t`), it produces the world at tick end (`world_t+1`) and the outputs that should leave (`outputs_t`). Between those endpoints, no system touches the outside world. No system reads `time.perf_counter()`, sends a packet, writes to disk, or prints to stdout. Inside, the simulator is a transformation. Outside, it is a queue.

```text
   ┌─────────────────────────────┐
   │      Simulator (pure)       │
   │  ┌──────────────────────┐   │
   │  │     systems run      │   │
   │  │   on world_t state   │   │
   │  └──────────────────────┘   │
   │     ↑                  ↓    │
   │ inputs_t           outputs_t│
   └─────↑──────────────────↓────┘
         │                  │
   ┌─────────┐        ┌─────────┐
   │ in queue│        │out queue│
   └─────────┘        └─────────┘
        ↑                  ↓
   environment        environment
```

Inputs arrive on the in-queue: events with timestamps, food-spawn requests from the policy, network packets in a multiplayer simulator, user input events. They wait in the queue until the next tick consumes them.

Outputs leave on the out-queue: state-change events for the log (`eaten`, `born`, `dead`), rendering data for the visualiser, packets for peers, replication updates for distributed nodes. They wait in the queue after the tick produces them, until the storage system or transport layer ships them.

What happens *inside* the boundary: pure transformation. Systems read from `inputs_t` (which is just another table by the time the systems start), update the world's tables, queue mutations to `to_remove`/`to_insert`, and write to `outputs_t` (also just a table). The inside is reproducible by construction; the outside is unpredictable, and the queue is the seam.

## Why this matters

**Determinism.** [§16](16_determinism_by_order.md)'s rule (same inputs + same order = same outputs) holds only if "inputs" is a complete description of the tick's environment. The queue *is* that complete description. Any system reading from outside the queue is a source of non-determinism the queue cannot capture.

**Replay.** Record the in-queue. Replay the tick from `world_t` with the recorded queue. Get bit-identical `world_t+1`. The queue is what makes replay possible.

**Testability.** A test fills the in-queue with a synthetic input, runs one tick, asserts on the out-queue. The test does not need to mock `open()`, `socket`, or the system clock; the queue interface is the only thing the simulator sees.

**Distribution.** A distributed simulator with multiple nodes communicates via queues — each node's out-queue feeds another node's in-queue. The queue interface is the same on a single machine and across a network.

**Auditability.** Every input that ever reached the simulator is in the in-queue's history. Every output is in the out-queue's history. The simulator's full external interface is two append-only logs.

## The Python anti-shapes the boundary forbids

Python's standard library makes I/O *frictionless to leak*. Five concrete leaks the boundary rule forbids inside the simulator's tick:

```python
# anti-pattern: bad!
print(f"creature {i} ate")              # 1. stdout from inside a system
logger.info("starvation event")         # 2. logging package, same problem
now = time.perf_counter()               # 3. wall clock read inside a system
response = requests.get(URL)            # 4. HTTP from a handler
threshold = float(os.environ["BURN"])   # 5. config read inside a system
```

Each one looks innocuous in isolation. Each one breaks determinism the moment two runs of the same simulator produce different output for "the same" inputs — because the inputs were not actually the same; one run saw a different clock, a different `BURN`, a different network response. The bug is silent and intermittent.

The disciplined Python form: every external read goes through the in-queue; every external write goes through the out-queue. Logging becomes a system that appends rows to a `log_events` column ([§37](37_log_is_world.md)). Time becomes a parameter, read once by the tick driver and passed down ([§16](16_determinism_by_order.md)). Config becomes part of `inputs_t` at the tick where it changes; the simulator never reads it directly.

## What the queue actually is, in Python

Three reasonable shapes for the queue itself. Pick the one that matches the data.

**Numpy parallel columns** for high-throughput, fixed-schema events. An `eaten` event is `(tick: u32, eater_id: u32, food_id: u32, energy_delta: f32)` — four columns, appended in lockstep. This is the simlog shape ([§30](30_streaming_wall.md)'s reference implementation), and the right pick when the simulator generates many events per tick. Bulk-numpy reads at consume-time; bulk-numpy writes at produce-time.

**A list of small dicts or named tuples** for low-volume, mixed-schema events arriving from the outside (user input, sparse network messages). The volume is small enough that the per-row construction cost from [§6](06_a_row_is_a_tuple.md) does not bind. Use named tuples if the schema is fixed; use a dict-of-columns approach if it varies.

**An sqlite table** when the queue itself must be durable across runs (audit logs, persisted requests). The §29/§38 sqlite numbers say it sustains ~830K-900K lookups per second on disk; that is enough headroom for any per-tick queue activity.

**One Python option that is *not* the right answer:** `multiprocessing.Queue`. Despite the name, it is the inter-process coordination mechanism from §32, not the simulator's external boundary. Its in-queue is for "main → worker" task dispatch, not for "outside world → simulator." Conflating the two means every external input pays kernel-call cost; worse, the queue's order is process-scheduler-dependent and not deterministic across runs. Use ordinary numpy columns or lists for the simulator's external queue; use `multiprocessing.Queue` only between main and workers.

## Composition with cleanup

The cleanup pattern from [§22](22_mutations_buffer.md) was the boundary at *tick scope* (mutations buffer, apply at tick boundary). The queue pattern at this scope is the same idea at *run scope* (I/O buffers, apply at the seam). The two compose: cleanup makes the tick atomic; the queue makes the run reproducible.

A useful test: can you run two simulators side-by-side from the same in-queue and get identical out-queues? If yes, the boundary holds. If no, somewhere a system reads the environment directly.

## Exercises

1. **Build the queues.** Add `in_events: dict[str, np.ndarray]` and `out_events: dict[str, np.ndarray]` to your simulator's world (one column per event field, plus an `n_active` counter per queue). Both fill at tick boundaries; both reset at the start of the next tick after their consumers have read them.
2. **Refactor a system that reads time.** Find any system that calls `time.perf_counter()` directly. Refactor: take `current_time: float` as a parameter. The tick driver reads `time.perf_counter()` once and passes it down. The system itself is now deterministic.
3. **Refactor a system that prints.** Find any system that calls `print(...)` or `logger.info(...)`. Refactor: append the message to `out_events["log"]`. The tick driver reads the queue after the tick and writes whatever's there. Logging is now deterministic; tests can assert on the queue.
4. **Replay test.** Save the in-queue across a 100-tick run (`np.savez("in_queue.npz", **in_events)`). Run the simulator a second time from the initial world state with the saved queue. Hash both worlds. They must match.
5. **Two simulators from one queue.** Run two simulators in parallel (or sequentially), feeding both from the same in-queue. After 100 ticks, hash both worlds. They must match. If they do not, somewhere a system reads from outside the queue.
6. **Find every leak.** Search your simulator's source: `grep -r "time\.\|print\|logger\|requests\|os.environ\|input(" code/sim/`. Each match is a candidate leak; each is a place where determinism could fail. Refactor the ones inside any system to go through the queue instead.
7. *(stretch)* **Audit an open-source simulator.** Open any Python simulator's tick function (mesa, agentpy, mesa-geo). Find every place it reads from the environment (clock, file, network, env vars). Each is a place where determinism leaks; each could be queue-ified.

Reference notes in [35_boundary_is_the_queue_solutions.md](35_boundary_is_the_queue_solutions.md).

## What's next

[§36 — Persistence is table serialization](36_persistence_is_serialization.md) takes the next step: when the simulator pauses and resumes, persistence is just writing the columns and reading them back. No translation, no impedance mismatch.
