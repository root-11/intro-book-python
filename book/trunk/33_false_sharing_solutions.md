# Solutions: 33 — False sharing

## Exercise 1 — The pathological counter

```python
import numpy as np, time
from multiprocessing import Process
from multiprocessing.shared_memory import SharedMemory

ITERS = 5_000_000
N_WORKERS = 4

def worker_unpadded(shm_name, my_id):
    s = SharedMemory(shm_name)
    c = np.ndarray((N_WORKERS,), dtype=np.int64, buffer=s.buf)
    for _ in range(ITERS):
        c[my_id] += 1

if __name__ == "__main__":
    shm = SharedMemory(create=True, size=N_WORKERS * 8)
    np.ndarray((N_WORKERS,), dtype=np.int64, buffer=shm.buf)[:] = 0
    t = time.perf_counter()
    procs = [Process(target=worker_unpadded, args=(shm.name, i)) for i in range(N_WORKERS)]
    for p in procs: p.start()
    for p in procs: p.join()
    print(f"4 workers, all counters in one cache line: {time.perf_counter()-t:.2f} s")
    shm.close(); shm.unlink()
```

Expected: ~3-5 seconds for 4 workers × 5M increments. A *single-process* loop doing the same total work (20M increments) typically finishes in 2-3 seconds. **The parallel version is slower than serial** — true negative scaling. Every increment by any worker invalidates the cache line in the other workers' caches; the cache-coherence protocol serialises what looked like four independent loops.

This is the canonical pathological case. The fix is structural: separate or pad.

## Exercise 2 — The padded version

```python
def worker_padded(shm_name, my_id):
    s = SharedMemory(shm_name)
    # 8 int64 per worker = one cache line of slack per worker
    c = np.ndarray((N_WORKERS * 8,), dtype=np.int64, buffer=s.buf)
    for _ in range(ITERS):
        c[my_id * 8] += 1                                  # padded index

if __name__ == "__main__":
    shm = SharedMemory(create=True, size=N_WORKERS * 8 * 8)
    np.ndarray((N_WORKERS * 8,), dtype=np.int64, buffer=shm.buf)[:] = 0
    t = time.perf_counter()
    procs = [Process(target=worker_padded, args=(shm.name, i)) for i in range(N_WORKERS)]
    for p in procs: p.start()
    for p in procs: p.join()
    print(f"4 workers, padded to cache lines: {time.perf_counter()-t:.2f} s")
```

Expected: ~0.8-1.2 seconds — *near-linear speedup* from the serial baseline. Each worker now writes to its own cache line; no coherence traffic between cores. The wall time is roughly 1/N of the single-process equivalent.

The structural change: each counter sits on its own 64-byte boundary. The data the workers actually touch is non-adjacent in memory; the cache lines do not overlap.

## Exercise 3 — A real example

In the simulator's per-process `to_remove` segments pattern from §31 exercise 4: each worker writes to its own segment, allocated as its own `multiprocessing.shared_memory` block. The segments live at *different* OS-allocated virtual addresses; they cannot share a cache line because they are not within 64 bytes of each other.

The risk is only if you make the mistake of allocating one big shared-memory block and giving each worker a slice within it where the slice boundaries land mid-cache-line. With the per-process-shm pattern, this doesn't happen.

A diagnostic: write a small test that runs the `to_remove` build on 8 workers and compares wall time to a single-worker baseline doing 8× the work. Near-linear speedup → no false sharing. Sublinear → investigate.

## Exercise 4 — Adjacent in shared memory

```python
def worker_adjacent(shm_name, my_id):
    s = SharedMemory(shm_name)
    c = np.ndarray((2,), dtype=np.int64, buffer=s.buf)
    for _ in range(10_000_000): c[my_id] += 1

def worker_separate(shm_my_name, _):
    s = SharedMemory(shm_my_name)
    c = np.ndarray((1,), dtype=np.int64, buffer=s.buf)
    for _ in range(10_000_000): c[0] += 1

# adjacent: two workers, both writing into one shared block
# separate: two workers, each with its own private shared block
```

The adjacent version: both workers write to the same 64-byte cache line. The coherence protocol bounces the line between cores. Wall time: 2-3× a single-worker baseline.

The separate version: each worker writes to its own block at a different address. No coherence traffic. Wall time: 1× the single-worker baseline (parallel speedup is full).

The lesson: *physical separation in memory* is what matters, not *logical separation by index*. The Python interpreter sees no difference between the two cases; the cache hardware sees a different cache line, which is the difference.

## Exercise 5 — Find your cache-line size (stretch)

```sh
getconf LEVEL1_DCACHE_LINESIZE                   # usually 64 on x86, 64 or 128 on ARM
```

Most x86 desktops: 64 bytes.
Apple Silicon (M1/M2): 128 bytes at some cache levels (the "P-core" cluster's L1 was 128 in early reports, refined since).
Some server chips: 64 with a hint of false-sharing at 128 due to adjacent-line prefetching.

For portable code, padding to 128 bytes is a defensive choice — overpaying by 2× on x86, breaking even on ARM. For x86-only targets, 64 is exact.

## Exercise 6 — `perf stat` your rig (stretch)

```sh
perf stat -e cache-references,cache-misses -- uv run code/measurement/parallel_motion.py
```

For a *well-partitioned* simulator (large chunks, no false sharing):

- `cache-references` scales with the working set's cache-line count.
- `cache-misses` stays a small fraction (5-15%) regardless of worker count.

For a *false-sharing* version:

- `cache-misses` *grows* with worker count, often non-linearly.
- The miss rate (`cache-misses / cache-references`) can climb above 50% at 8 workers writing the same line.

The diagnostic: run perf at 1 worker and at 8 workers on the same workload. If miss rate is similar, the partition is healthy. If miss rate climbed substantially, look for adjacent writes within 64 bytes.

The `parallel_motion.py` rig uses ~625K-element chunks (2.5 MB per worker) for the motion case; partition boundaries are megabytes apart. False sharing is structurally impossible at that scale. The rig's near-linear speedup at the bandwidth ceiling is consistent with a clean cache-coherence profile.
