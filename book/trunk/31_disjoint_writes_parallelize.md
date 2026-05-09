# 31 — Disjoint write-sets parallelize freely

> *Concept node: see the [DAG](../../concepts/dag.md) and [glossary entry 31](../../concepts/glossary.md#31--disjoint-write-sets-parallelize-freely).*

Two systems can run in parallel if and only if their write-sets do not overlap. That is the rule. It is small. It is what [§25](25_ownership_of_tables.md)'s single-writer ownership buys you.

Concretely: in the simulator's tick, `motion` writes `pos_x`, `pos_y`, `energy`; `food_spawn` writes `food`. Their write-sets are disjoint. They can run on two different processes with no coordination — no locks, no atomics, no message-passing. The data layout makes the parallelism free.

The same shape works at finer grain. The simulator's three appliers (`apply_eat`, `apply_reproduce`, `apply_starve`) all read `pending_event` and write disjoint things — `apply_eat` writes `food`, `to_remove`; `apply_reproduce` writes `to_insert`; `apply_starve` writes `to_remove`. Two of the three append to the same buffer. To parallelise them, give each its own *segment* of `to_remove` (one per process), then merge at cleanup. The merge is `np.concatenate` — O(N) in the merged total, free relative to the work that produced it.

## Not threading. Not asyncio.

This is the chapter where the GIL question finally lands. The Python reflex when a chapter says "parallel" is to reach for `threading.Thread` or `asyncio`. **Both are wrong for CPU-bound parallel work in CPython.**

`threading` does not give you parallel CPU. The Global Interpreter Lock serialises Python bytecode execution: one thread runs Python at a time, regardless of how many threads you started. *Numpy bulk operations release the GIL during their C-level work*, so a `threading.Thread` running `arr.sum()` can overlap with another thread doing the same — but only during the `sum()`'s C call, not during any Python around it. For workloads dominated by Python orchestration of numpy ops, threading delivers token speedup at best.

`asyncio` is a scheduler for I/O-bound work. CPU-bound systems give it nothing to overlap. The event loop adds dispatch overhead and removes nothing.

The disciplined alternative is **`multiprocessing` plus `shared_memory`**. `__main__` allocates the world's columns in a shared-memory region. Worker processes attach to that region, get a numpy view onto the same bytes, and write to *their slice only*. There is no copying across the process boundary; the bytes are shared. The GIL is no longer in the picture because each process has its own GIL, and each process is doing pure C-level numpy work on its own partition.

The shape (full version in [`code/measurement/parallel_motion.py`](../../code/measurement/parallel_motion.py)):

```python
# Worker globals — set once per worker by the Pool initializer.
_arr = None
_shm = None

def init_worker(shm_name: str) -> None:
    global _arr, _shm
    _shm = shared_memory.SharedMemory(name=shm_name)
    _arr = np.ndarray(SHAPE, dtype=DTYPE, buffer=_shm.buf)
    # _arr now views the same bytes as __main__'s array.

def worker(args: tuple[int, int]) -> None:
    start, end = args
    # Each worker writes only its slice; the writes go directly to
    # the shared bytes via the numpy view — no copy.
    _arr[0, start:end] += _arr[1, start:end] * DT

# In __main__:
shm = shared_memory.SharedMemory(create=True, size=arr.nbytes)
arr = np.ndarray(SHAPE, dtype=DTYPE, buffer=shm.buf)
# ... fill arr with the world's data ...
boundaries = [(i * chunk, (i + 1) * chunk) for i in range(n_workers)]
with Pool(processes=n_workers, initializer=init_worker, initargs=(shm.name,)) as pool:
    pool.map(worker, boundaries)
```

The shape: **`__main__` owns the memory; workers attach via `init_worker` and hold a numpy view onto the shared bytes; each worker writes only its slice; no shared writes, no locks, no message-passing.**

## What it costs and what it buys

From [`code/measurement/parallel_motion.py`](../../code/measurement/parallel_motion.py), two workloads applied 100 times to 10,000,000 `float32` creatures on this machine (8 physical cores, 16 logical with SMT):

**Workload A — memory-bound** (`pos += vel * dt`): 12 bytes accessed per element, 2 arithmetic ops. Memory traffic dominates.

| workers | wall (s) | speedup |
|--------:|---------:|--------:|
| serial  |    1.842 |    1.00 |
|       1 |    1.840 |    1.00 |
|       2 |    0.433 |    4.25 |
|       4 |    0.456 |    4.03 |
|       8 |    0.459 |    4.01 |
|      16 |    0.414 |    4.45 |

**Workload B — compute-bound** (`out += sin(x)**2 + cos(x)**2`): same byte accesses, much heavier per-element CPU work.

| workers | wall (s) | speedup |
|--------:|---------:|--------:|
| serial  |    7.749 |    1.00 |
|       1 |    7.778 |    1.00 |
|       2 |    2.575 |    3.01 |
|       4 |    1.608 |    4.82 |
|       8 |    1.412 |    5.49 |
|      16 |    1.427 |    5.43 |

Three readings.

**1 worker matches serial.** The pool round-trip cost is amortised across the run because the rig dispatches *once* per measurement (each worker runs all 100 ticks on its partition before returning) — a per-tick dispatch would add IPC overhead on top, capping speedup further. See exercise 6.

**Memory-bound caps at ~4×.** This is the **aggregate memory-bandwidth ceiling** on this machine. The 76 MB working set spills L3; once two cores are reading and writing flat-out, the DRAM bus is busy. Adding a third or fourth physical core helps slightly (some bandwidth comes from the cores' own L1/L2), but past that, more workers compete for the same bandwidth. *The ceiling is set by the memory subsystem, not by core count.* On a chip with more memory channels (server CPUs, modern desktops with quad-channel DDR5), the ceiling is higher; on a single-channel laptop or a Raspberry Pi, lower.

**Compute-bound caps at ~5.5×, with the plateau between 8 and 16 workers.** The plateau location matches the **physical core count** (8 here); the SMT-doubled logical count of 16 adds essentially nothing because both threads on the same core are now contending for the same arithmetic units. Compute-heavy work scales close to the physical core count; SMT helps work that has gaps the second thread can fill (mostly memory-stall waits), and pure compute has no gaps to fill.

The two ceilings are different shapes for different reasons. Measure your specific workload — neither is "wrong," they are different bottlenecks.

## Three things this rule does for you

**No locks.** A lock is a tax paid by every reader and writer of the locked thing. With single-writer ownership, locks are unnecessary; with disjoint write-sets across processes, they remain unnecessary at the parallel boundary. The simulator at this scale has zero `Lock`, zero `RLock`, zero `Semaphore` in its inner systems. The whole concurrency-primitive vocabulary you see in tutorials does not apply once the architecture is right.

**Speedup is structural, not promised.** N processes with disjoint work give close to N× speedup *until the bottleneck shifts*. Memory-bound work hits the bandwidth ceiling first; compute-bound work runs out of physical cores; per-tick dispatch hits IPC overhead. The ceilings are real and measurable; they are not reasons to avoid the architecture, only reasons to know which ceiling your workload hits.

**Tools without ceremony.** The Python ecosystem's standard tools — `multiprocessing.Pool`, `concurrent.futures.ProcessPoolExecutor`, `multiprocessing.shared_memory` — are stdlib. No third-party crate, no external service, no orchestrator. The rig in `parallel_motion.py` is ~150 lines. Build it once for your simulator; reuse it everywhere.

The single-writer rule ([§25](25_ownership_of_tables.md)) was the precondition. Disjoint write-sets is the rule applied across systems. Together, parallelism becomes a scheduling decision, not a design decision.

## A calibration note

Python multiprocessing is non-trivial. The clean speedup table above hides real complexity: pickle overhead at process boundaries, fork-vs-spawn semantics that vary by platform, signal handling, queue contention, the difficulty of reasoning about a system across N process boundaries when something goes wrong. The chapter has not lied — the architecture does work and the speedup is real — but it has presented the architecture without the operational cost.

**This chapter teaches the principles, not a production recipe.** Single-writer ownership, disjoint write-sets, partition-don't-lock, shared-memory-not-pickle: these are correct at every scale. Python multiprocessing is a fine implementation of those principles when your tick is comfortably above the IPC floor (≥ ~16 ms per tick, partitions of ≥ 100K elements). It stops being fine when every percent matters — a physics engine at 1 kHz, a real-time control loop, anything where the operational complexity above eats budget that a compiled language would not.

The escalation order is short: **numpy → maturin → leave Python.** Maturin (Rust + PyO3) gives you the same parallelism architecture without the Python orchestration tax — the inner loop, the dispatch, and the data are all in compiled Rust, exposed to Python through a thin binding. Past maturin, the answer is not to add another Python-side library; it is to leave Python entirely and write the application in Rust. The Rust standard library is enough for most parallel work; you do not need to reach for a parallel-iteration crate to do this well.

From-scratch-then-price-the-crate ([§41](41_compression_oriented.md), [§42](42_you_can_only_fix_what_you_wrote.md)) applies here too: build it in Python first to feel the architecture; price what the next tier gives you when the budget binds. *The book teaches the architecture; the language is a tooling decision.*

## Exercises

You will need a multi-core machine. Most desktops and laptops qualify.

1. **Run the rig.** `uv run code/measurement/parallel_motion.py`. Read your speedup column. Find the worker count where the curve flattens — that is your bandwidth ceiling.
2. **Threading falls short.** Rewrite `parallel_motion` using `threading.Thread` instead of `multiprocessing.Pool`. Keep the same partition pattern. Time it. The speedup is real but smaller (numpy releases the GIL during bulk ops, so threads can overlap during the `*= dt` step, but not during anything else). Compare to the multiprocessing version.
3. **A failing case.** Try to run motion and an `apply_eat` system in parallel where both write `energy`. Without single-writer discipline, two processes writing the same shared-memory region produce undefined behaviour. Construct the case; observe the corruption (it may be silent — that is the failure mode).
4. **Per-process segments.** Modify the rig so that, instead of motion, each worker runs `apply_starve` and produces *its own* `to_remove` segment as a separate shared-memory array. After all workers finish, `np.concatenate` the segments in `__main__`. Verify the merged result equals a single-process run.
5. **Find the bandwidth ceiling.** Run the rig at N = 100,000 (fits L2), N = 1,000,000 (fits L3), N = 10,000,000 (spills to RAM), N = 100,000,000 (deeply RAM-resident). Plot the memory-bound speedup vs N. The bandwidth-ceiling worker count shifts with N — small N is bandwidth-rich (per-core caches), large N is bandwidth-limited.
6. **Per-tick dispatch costs IPC.** Modify the rig so each worker runs *one* tick per `pool.map` call instead of all 100 in one call. Re-run. The speedup curve will plateau lower (~3-4× for memory-bound, ~4-5× for compute-bound on this machine) because every tick now pays for one IPC round-trip. The lesson: *batch when the access pattern allows*. The cost is small per call, real in aggregate.
7. **Find your physical core count.** `lscpu | grep 'Core(s) per socket'` (Linux). Compare to `os.cpu_count()`. The compute-bound ceiling lives near the physical count, not the logical count.
8. *(stretch)* **`concurrent.futures` comparison.** Rewrite the rig using `concurrent.futures.ProcessPoolExecutor.map`. Confirm equivalent performance. The two are largely interchangeable; pick the one whose API your team prefers.
9. *(stretch)* **A pure-Python anti-comparison.** Implement the same motion system as a per-creature Python loop (`for i in range(N): pos[i] += vel[i] * dt`). Run it serially. Run it under `threading.Thread` with 8 threads. Run it under `multiprocessing.Pool` with 8 workers. Note: the threading version is no faster than serial (GIL), the multiprocessing version is faster but still slower than the *bulk-numpy serial* version, because the bulk numpy version was already faster than any pure-Python form. Multiprocessing scales work that is already fast; it does not rescue work that was wrong-shaped.

Reference notes in [31_disjoint_writes_parallelize_solutions.md](31_disjoint_writes_parallelize_solutions.md).

## What's next

[§32 — Partition, don't lock](32_partition_dont_lock.md) takes the next step: when one system *must* write a single table from multiple processes, you split the table, not the access.
