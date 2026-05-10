# 32 — Partition, don't lock

> *Concept node: see the [DAG](../../concepts/dag.md) and [glossary entry 32](../../concepts/glossary.md#32--partition-dont-lock).*

<p align="center"><img src="../illustrations/bridge_clipboard.jpg" alt="Bridges drawn as independent spans — partition into disjoint write-sets" style="max-height: 300px; max-width: 100%;"></p>

[§31](31_disjoint_writes_parallelize.md) said "disjoint write-sets parallelise freely". What if the system has to write *one* table from many processes? Motion at 1M creatures wants to update `pos_x` and `pos_y` for every creature; the table is one. Eight processes, one table — looks like a lock case.

It is not. The fix is to *partition the data*, not to lock the access.

Each process takes a slice of the table. Process *t* writes slots `t * N/8 .. (t+1) * N/8` and only those slots. The slices are disjoint by construction; no process can write where another is writing. Inside each slice, a single process is the writer — [§25](25_ownership_of_tables.md)'s ownership rule still holds, just at the slice level instead of the table level. Numpy slicing into shared memory gives each worker a non-overlapping view of the same underlying bytes. No `Lock`, no `Semaphore`, no atomic. The bytes are physically partitioned; the writes cannot collide.

That is half the chapter. The other half is the question §31 left dangling: **how does main coordinate with the workers in the first place?**

## Coprocessors are IOPS-limited

A worker process is a CPU that can do work, but only after main has told it what work. Telling a worker something — sending a message, releasing a barrier, putting a task on a queue — has a cost, and that cost is a hard ceiling on how fast main can keep workers busy. From [`code/measurement/coordination_patterns.py`](../../code/measurement/coordination_patterns.py), three coordination patterns measured on this machine (8 physical cores, 7 workers + 1 main, 20,000 rounds × 7 workers = 140,000 round-trips per pattern):

| pattern                       |    msgs/sec | jitter p50 | jitter p99 |
|-------------------------------|------------:|-----------:|-----------:|
| 1. single shared `Queue`      |      88,016 |    32 µs   |    92 µs   |
| 2. per-worker `Queue`         |      57,083 |    77 µs   |   121 µs   |
| 3. shared numpy array         |   1,472,323 |   0.1 µs   |   0.6 µs   |

Three readings.

**Patterns 1 and 2 — both `multiprocessing.Queue` based — top out around 60K-90K msgs/sec.** That is the floor of "one kernel call per put, one kernel call per get, one pickle per message." It is not "Python is slow"; it is "anything that goes through the kernel costs ~10 microseconds, and at one round-trip per task you get 100K tasks per second per worker, and 7 workers do not multiply because main is the bottleneck."

**Per-worker queues are *slower* than the single shared queue here**, which is the chapter's first surprise. The contention argument from textbooks ("avoid lock contention by giving each worker its own queue") is real, but at this workload size the dominant cost is *main's serial calls* — one `q.put()` per worker per round, seven kernel transitions instead of seven enqueues into a single queue. Contention would matter at higher loads or with more workers; at the simulator's per-tick scale, pipelining is the thing.

**The shared numpy array runs at 1.47 million messages per second** — 17× faster than the single queue, with jitter two orders of magnitude tighter (0.6 µs at p99 vs 92 µs). No kernel involvement: main writes a generation counter to the shared array, workers spin-wait reading the array, do the work, increment their ack counter. The only synchronisation is x86's normal cache coherence on aligned 64-bit reads and writes. **This is the IOPS ceiling for in-process Python coordination on this machine.**

## Batching is forced by physics

Translate the IOPS ceiling into the simulator's tick budget. At 30 Hz, the budget is 33 ms. With the shared-array pattern at 1.5M msgs/sec, that is **~50,000 coordination events per tick**. With queue-based patterns at ~90K msgs/sec, it is **~3,000 events per tick**.

Compare against possible work shapes for a 1,000,000-creature, 20-system simulator:

| per-tick coordination shape                                  | events     | feasible? |
|--------------------------------------------------------------|-----------:|-----------|
| 1 message per creature per system: 20,000,000 events         | 20,000,000 | no — even shared-array is 400× short |
| 1 message per creature: 1,000,000 events                     |  1,000,000 | no — shared-array is 20× short |
| 1 message per system per partition × 7 partitions: 140 events |        140 | yes — three orders of magnitude under any pattern |
| 1 message per system: 20 events                              |         20 | yes — trivially |

The first two are off the table. The third is what the simulator actually does. **Batching is not an optimisation; it is forced by the IOPS ceiling.** A worker cannot be told "process this single creature" and then "process this next single creature" because the telling is much slower than the processing. A worker can be told "process your partition of the creature table" once, and then it does 100,000 creatures' worth of work before main needs to say anything to it again.

Once batching is forced, *partitioning is the natural batch shape*. Each batch is a slice of the table. Each worker owns its slice across many ticks. The coordination message is "run this system on your slice" — short enough to fit in any of the three patterns above, even the slowest.

## The ventilator model

Putting the pieces together gives the production-quality form of "partition, don't lock":

**Main owns** the tick clock, the I/O queue, the shared-memory arrays, and the system DAG. It does not allocate per tick; the buffers were sized at startup.

**Workers (`nprocs - 1`)** each hold their pre-assigned partition (slots `[my_id * chunk, (my_id+1) * chunk)`) and a numpy view onto the shared memory. They wait for signals from main, run the indicated system on their slice, signal completion. Workers do not allocate per tick either.

**The signal carries the system index, not the data.** A worker already knows which slice of the world it owns; main only needs to tell it *which system to run this phase*. The simulator's twenty systems become twenty small integers — one tells the worker "run motion on your partition", another tells it "run apply_starve on your partition", and so on.

The DAG itself, encoded as a shared array, becomes:

```
phase 1: [1]                     # one system runs (no parallelism this phase)
phase 2: [1, 2, 3, 4, 5, 6]      # 6 systems in parallel
phase 3: [1, 2, 3, 4, 5]         # 5 partitions of one system
phase 4: [1, 2, 3]               # 3 partitions
phase 5: [1, 2, 3]               # 3 systems
phase 6: [1]                     # cleanup
phase 7: [1]                     # inspection (if --debug is set)
```

Read it as a sequence of phases. Within a phase the entries are which-worker-runs-this-task; between phases there is a barrier (main waits for all acks before bumping the generation).

## DAG-as-line, sliced by phase

A tick is *an ordered sequence of atomic tasks*, partitioned into phases. Each atomic task is a (system, partition) pair. Phase boundaries are barriers — every task in phase N must complete before any task in phase N+1 starts, because of the data dependencies the DAG encodes ([§14](14_systems_compose_into_a_dag.md)).

Inside a phase, the work is independent and can run on as many workers as main has available.

The slicing question becomes concrete: **how do you snip the line of atomic tasks so that the DAG is respected (phase boundaries become barriers) and the work is as evenly spread across the available workers within each phase, given the jitter the table above measured?**

The DAG's structure is permanent — which systems exist, which depend on which — and is fixed at design time. *What varies tick to tick is the amount of work each system generates.* In an MMORPG the population of NPCs in a busy city demands more work in the AI system; a battlefield demands more in swarm coordination. The same DAG runs with the same phases; the partitioning of work *inside* each phase changes.

Main's job is to observe and rebalance: how long did each phase take last tick, how should this tick's partitions be assigned to spread work evenly given the per-worker jitter measured above?

## Load balancing at 30 Hz

A 30 Hz tick is 33 ms. The shared-array coordination round-trip is sub-microsecond at p99. Main has plenty of headroom — milliseconds, not microseconds — to reassign partitions every tick based on what it observed last tick.

The pattern: each phase, each worker stamps its completion timestamp in the shared array (the exhibit's `COORD_TIMESTAMP` slots). Main reads the timestamps, computes per-worker phase wall times, and adjusts the partition boundaries for the *next* tick. A worker that finished early gets a slightly larger slice next time; a worker that finished late gets a smaller slice. The DAG-as-array can also adjust *how many workers participate* in a phase — a short phase that only needs three workers releases the other four to start the next phase early.

This is closed-loop control over the tick budget. Main observes; main decides; main writes new partition boundaries before the next tick fires. *The partitioning is not a static decision; it is a quantity main maintains, like every other piece of simulator state.*

## Choosing the partition shape

Within the ventilator model, the *initial* partition shape is still a design choice. Four options worth naming:

**By entity range** (the default): each worker takes contiguous slot range `[i*N/W, (i+1)*N/W)`. Simple; works when access is uniform.

**By spatial cell** (after sort-for-locality, [§28](28_sort_for_locality.md)): each worker takes a region of the world. Useful when interactions are local — neighbours-only collisions, regional behaviours. Workers at boundary cells need a small synchronisation step (or a halo region copied into each worker's input).

**By hash**: each worker takes ids whose `hash(id) % n_workers` matches its index. Useful when access is uniform but you want stable worker-to-data mapping across ticks (worker caches stay warm on the same partition tick after tick).

**By workload weight** (the load-balanced form above): each worker takes a number of rows weighted by *expected work* per row. The 30-Hz observe-and-rebalance loop above implements this dynamically.

The partition shape is the design choice; the partition mechanism — numpy slicing into shared memory — is one line.

## A calibration

This chapter has covered a lot of ground at the architectural level. Three honest qualifications.

**The shared-array pattern is the principle, not a recipe.** The exhibit's pattern works; it is fast; it is also non-trivial to debug under load. Production implementations typically use `multiprocessing.shared_memory` plus `multiprocessing.Event` for the wake-up (instead of a busy-loop) to be friendlier to other processes on the machine. The IOPS ceiling drops from 1.5M to ~500K with the Event, which is still 5-10× the queue patterns.

**Python multiprocessing remains non-trivial.** As §31's calibration note said: this teaches the architecture, not a production recipe for workloads where every percent matters. The single-writer, partition-don't-lock, batched-coordination architecture *is* correct at every scale. If your tick budget cannot tolerate the operational complexity of debugging across N Python processes, the answer is to escalate to maturin (Rust + PyO3) and apply the same architecture in compiled code.

**Real ECS engines do this in compiled code.** Bevy, Unity DOTS, Unreal Mass Entities — they each implement variants of the ventilator model in C++ or Rust. The architecture is genuinely the right shape; the language is a tooling decision.

## Exercises

1. **Run the coordination exhibit.** `uv run code/measurement/coordination_patterns.py`. Read your three rates. Compute "coordination events per 30 Hz tick" for each pattern. The shared-array number is the budget you have for any per-tick orchestration.
2. **The batching threshold on your machine.** With your IOPS numbers, compute the smallest partition size that makes coordination cost ≤ 10% of partition work cost. Below that threshold, batching is the only option. Above it, you can afford to dispatch per-something.
3. **Pre-assigned partitions.** Modify your simulator so each worker holds its `(start, end)` once at startup, never receives it again. The signal it gets per phase is a small integer (system id). Compare the wall time to a version that re-sends `(start, end)` every phase. The difference is the marginal IPC saved.
4. **The DAG-as-array.** Build a length-20 numpy array of `int8` representing your simulator's DAG (system ids per phase, separators between phases). Have workers spin-wait on this array. Confirm correctness against a single-process baseline.
5. **Load-balanced partitioning.** Add per-worker timestamps after each phase (the `COORD_TIMESTAMP` slot pattern). After each tick, recompute partition boundaries proportionally to per-worker phase times. Run for 1000 ticks; observe the boundaries converge as the workload stabilises.
6. **Workload heterogeneity.** Construct a workload where 80% of work lives in 20% of the partitions (e.g. one MMORPG city dominates a flat world). Compare a fixed equal-sized partitioning to the load-balanced one from exercise 5. The load-balanced version should converge to slices of unequal size that all complete in roughly the same wall time.
7. **The boundary-builder lives in `__main__`.** Write a worker that *computes its own slice* from `(my_id, n_workers, N)`. Run it. Now change `N` mid-tick from `__main__` and observe the chaos. Confirm that the disciplined form (boundaries computed once in `__main__`) does not have this failure mode.
8. *(stretch)* **`Event` instead of busy-wait.** Replace the spin-loop in the shared-array worker with `multiprocessing.Event.wait()`. Measure the new throughput. The trade-off: lower CPU usage when idle, slightly higher latency per round-trip.
9. *(stretch)* **The 1 kHz physics-engine question.** Compute the per-tick budget at 1 kHz (1 ms). Compute how many shared-array coordination events fit in that budget. At what worker count does coordination overhead become unaffordable? This is the kind of arithmetic that decides whether your physics engine stays in Python multiprocessing or escalates to maturin.

Reference notes in [32_partition_dont_lock_solutions.md](32_partition_dont_lock_solutions.md).

## What's next

[§33 — False sharing](33_false_sharing.md) names the hardware-level pitfall that can sink the partition pattern: two processes writing different bytes in the same cache line slow each other down despite logical independence.
