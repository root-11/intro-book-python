# 33 — False sharing

> *Concept node: see the [DAG](../../concepts/dag.md) and [glossary entry 33](../../concepts/glossary.md#33--false-sharing).*

<p align="center"><img src="../illustrations/multimeter.jpg" alt="A mouse with a multimeter — false sharing is a precision-of-cost-measurement problem" style="max-height: 300px; max-width: 100%;"></p>

You partitioned the table. Each process writes its own disjoint slice. The work is balanced. The speedup is... 1.2× on 8 cores. Where did the parallelism go?

Probably to *false sharing*.

The CPU cache works on 64-byte *cache lines*. When a process writes to address X, the cache coherence protocol invalidates that line in every other core's cache — they must throw away their copy and reload. If two processes are writing to *different* addresses but in the *same* cache line, every write triggers an invalidation on the other process's cache. The processes slow each other down without ever logically conflicting.

A pathological case: eight processes each incrementing one entry in an `int64` array of length 8 in `multiprocessing.shared_memory`. The array is exactly 64 bytes — one cache line. All eight processes write to that line. Every write invalidates the other seven caches. The processes run *slower* together than one process alone — true negative scaling.

## Why this matters in Python+multiprocessing

The Python reflex is that the GIL is the only concurrency hazard. **False sharing is a hardware-level hazard that the GIL does not protect you from**, because once you are in `multiprocessing.shared_memory`, multiple OS-level processes are running on multiple cores, hitting the same physical bytes. The GIL does not enter — it never crosses the process boundary. The cache coherence protocol does.

The good news: the partition pattern from [§31](31_disjoint_writes_parallelize.md) and [§32](32_partition_dont_lock.md) avoids false sharing by default *because the partitions are huge*. The `parallel_motion.py` rig uses chunks of `N/n_workers = 10M/16 ≈ 625K` `float32` values per worker — 2.5 MB per chunk, **40,000 cache lines** per chunk. The boundaries between chunks are megabytes apart. False sharing requires *adjacent* writes within a 64-byte window, and the partition does not produce them.

False sharing shows up when the per-process state is small. Three cases worth naming:

**Per-process counters in shared memory.** If each worker writes to `counters[my_id]` in a shared array, and the array is `int64`, then 8 workers occupy 64 bytes — exactly one cache line. Every increment by any worker invalidates every other worker's cache copy. *True negative scaling.*

```python
# anti-pattern: bad!
counters = np.ndarray((8,), dtype=np.int64, buffer=shm.buf)
def worker(my_id: int) -> None:
    for _ in range(1_000_000):
        counters[my_id] += 1   # all 8 counters fit in one cache line
```

**Per-process accumulators near a boundary.** A worker that updates one row at the boundary of its partition (e.g. when applying boundary effects in a spatial sort, [§28](28_sort_for_locality.md)) can land in the same cache line as the neighbouring worker's first row. This is why halo regions in domain-decomposition codes are typically padded to cache-line size.

**Many small per-process buffers in one shared region.** If you put N small per-process scratch arrays adjacent in one shared-memory block, false sharing is likely at the boundaries. The fix is one shared-memory block per process, or padding between regions.

## Fixes

**Make per-process state structurally separate.** Each process gets its own `multiprocessing.shared_memory.SharedMemory` block, or its own private numpy array (the default — workers do not see each other's stack-allocated memory). Merge results in `__main__` after all workers complete. The `to_remove` per-process segments pattern in [§31](31_disjoint_writes_parallelize.md) does this — each process writes to its own `np.ndarray`, then `__main__` runs `np.concatenate` to merge.

**Pad shared per-process state to a cache line.** If you must have one shared array of per-process state, space the entries 64 bytes apart:

```python
# 8 workers, each owns counters_padded[my_id * 8] (one int64 per cache line)
counters_padded = np.ndarray((8 * 8,), dtype=np.int64, buffer=shm.buf)
def worker(my_id: int) -> None:
    for _ in range(1_000_000):
        counters_padded[my_id * 8] += 1   # each on its own cache line
```

**Partition at cache-line boundaries.** When dividing a typed array among workers, round the boundaries to multiples of `64 // dtype.itemsize` — 16 for `int32`/`float32`, 8 for `int64`/`float64`. The numpy partition above already does this for any chunk size larger than ~16 elements; only at very small chunks does the boundary land within a line.

False sharing is a hardware concern, not a Python concern. The Python interpreter sees no problem with eight processes writing eight disjoint addresses; the hardware sees one cache line and serialises the access. The bug is invisible at the language level. It shows up only as performance — the parallel version is mysteriously slow.

## Detection

Profile with `perf stat -e cache-references,cache-misses` (Linux) on your simulator:

```bash
perf stat -e cache-references,cache-misses -- python my_sim.py
```

False sharing produces high `cache-misses` despite supposedly disjoint writes. If profiling shows your parallel system has surprisingly high cache traffic — say, more cache misses per second than the working set could account for in one pass — false sharing is a likely cause.

The takeaway: physical layout matters even for logically disjoint data. Two writes to different shared-memory addresses do not parallelise freely if those addresses are within 64 bytes. The fix is separation or padding. The detection is profiling.

## Exercises

1. **The pathological counter.** Build the 8-process case with `multiprocessing.shared_memory`: an `int64` array of length 8, each worker incrementing its own slot in a tight loop. Time the parallel version against a single-process loop doing the same total work. The parallel version should be *slower* — true negative scaling. (Hint: at small enough work-per-tick, even spawning the processes is slower; pick a tight inner loop with millions of increments to see the cache effect dominate.)
2. **The padded version.** Pad each counter to its own cache line: use an `int64` array of length `8 * 8 = 64` and have each worker write to index `my_id * 8`. Re-run. The parallel version should now scale near-linearly with worker count.
3. **A real example.** In your simulator's per-process `to_remove` segments ([§31](31_disjoint_writes_parallelize.md) exercise 4), check whether two workers' segment-appending might land in the same cache line. They normally do not — separate per-process numpy arrays live in different shared-memory blocks — but if performance is unexpectedly poor, this is one place to look.
4. **Adjacent in shared memory.** Build a shared array of two `int64`s. Spawn two workers, one writing index 0, one writing index 1, in tight loops. Time vs. two workers each writing to its own separate `multiprocessing.shared_memory` block.
5. *(stretch)* **Find your cache-line size.** `getconf LEVEL1_DCACHE_LINESIZE` on Linux. Verify it is 64 bytes (some chips use 128 bytes — especially Apple Silicon at certain levels). If you are on one of those, padding to 64 is not enough; you need 128.
6. *(stretch)* **`perf stat` your rig.** `perf stat -e cache-references,cache-misses -- uv run code/measurement/parallel_motion.py`. Compare miss rates at 1 worker vs 8 workers. The miss rate should be roughly the same (no false sharing), confirming the rig's partition is large enough to avoid the trap.

Reference notes in [33_false_sharing_solutions.md](33_false_sharing_solutions.md).

## What's next

[§34 — Order is the contract](34_order_is_the_contract.md) ties parallelism back to the determinism rule from [§16](16_determinism_by_order.md): parallelism is allowed *inside* a step, never *across* steps.
