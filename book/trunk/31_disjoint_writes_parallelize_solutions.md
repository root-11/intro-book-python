# Solutions: 31 — Disjoint write-sets parallelize freely

## Exercise 1 — Run the rig

```sh
uv run code/measurement/parallel_motion.py
```

```
=== memory-bound: pos += vel * dt ===
 workers    wall (s)   speedup
--------------------------------------------------
  serial       3.199      1.00
       1       2.888      1.11
       2       0.588      5.44
       4       0.458      6.98
       8       0.463      6.92
      16       0.373      8.58

=== compute-bound: out += sin(x)**2 + cos(x)**2 ===
 workers    wall (s)   speedup
--------------------------------------------------
  serial      13.379      1.00
       1      12.978      1.03
       2       4.635      2.89
       4       2.784      4.81
       8       2.159      6.20
       16       1.830      7.31
```

On this machine, the memory-bound case plateaus around 4-8 workers (~7-8.5× speedup); compute-bound climbs more steadily to 7.3× at 16 workers. The memory-bound ceiling is *aggregate bandwidth*; compute-bound is *physical core count plus partial SMT overlap*.

Find your curve's flat spot — that's your machine's parallel ceiling for each regime.

## Exercise 2 — Threading falls short

```python
from threading import Thread
import numpy as np, time

def worker_thread(arr, start, end):
    arr[0, start:end] += arr[1, start:end] * 0.033

# Same partition, threads instead of processes
n_workers = 8
arr = np.zeros((2, 10_000_000), dtype=np.float32); arr[1] = 1.0
chunk = arr.shape[1] // n_workers

t = time.perf_counter()
threads = [Thread(target=worker_thread, args=(arr, i*chunk, (i+1)*chunk)) for i in range(n_workers)]
for th in threads: th.start()
for th in threads: th.join()
print(f"threading × 8: {(time.perf_counter()-t)*1000:.1f} ms")
```

Typical: ~1.5-2× speedup over serial — much less than the multiprocessing ~5×. Why?

- Numpy releases the GIL during bulk ops (`*= dt`), so threads can overlap during that C call.
- *Around* the bulk op, Python orchestration (slicing, attribute lookups, etc.) holds the GIL, serialising the threads.
- Net effect: parallelism only during the C calls themselves, not for the whole worker function.

For workloads that are pure numpy bulk ops on disjoint slices, threading gets a useful speedup but caps below multiprocessing. For workloads with any Python orchestration around the ops, threading caps near 1×.

## Exercise 3 — A failing case

```python
# anti-pattern: bad! two workers writing the same column without coordination
from multiprocessing import Process
from multiprocessing.shared_memory import SharedMemory
import numpy as np

shm = SharedMemory(create=True, size=80_000_000)
energy = np.ndarray((10_000_000,), dtype=np.float32, buffer=shm.buf)
energy[:] = 100.0

def motion_worker(shm_name, start, end):
    s = SharedMemory(shm_name)
    e = np.ndarray((10_000_000,), dtype=np.float32, buffer=s.buf)
    for _ in range(100):
        e[start:end] += 0.5            # writer 1: motion

def apply_eat_worker(shm_name, start, end):
    s = SharedMemory(shm_name)
    e = np.ndarray((10_000_000,), dtype=np.float32, buffer=s.buf)
    for _ in range(100):
        e[start:end] -= 1.0            # writer 2: starvation — SAME COLUMN

# Run them in parallel with overlapping slices
p1 = Process(target=motion_worker, args=(shm.name, 0, 5_000_000))
p2 = Process(target=apply_eat_worker, args=(shm.name, 0, 5_000_000))   # same slice!
p1.start(); p2.start(); p1.join(); p2.join()

print(energy[:10])
# Result is non-deterministic; some updates from each worker are lost
```

No `ValueError`, no warning. The two writes interleave at the cache-line level; some are lost. The wrong-result is silent.

The single-writer rule and disjoint write-sets are the *structural* prevention. There is no way to make this code correct without a lock, an atomic, or — the chapter's preferred answer — a different architecture where the two writers don't share a column.

## Exercise 4 — Per-process segments

```python
# Each worker writes to its own to_remove segment (per-process)
def starve_worker(shm_name, segment_shm_name, start, end):
    s = SharedMemory(shm_name)
    energy = np.ndarray(SHAPE, dtype=np.float32, buffer=s.buf)
    seg_shm = SharedMemory(segment_shm_name)
    seg = np.ndarray((SEGMENT_CAPACITY,), dtype=np.uint32, buffer=seg_shm.buf)
    n = 0
    for i in range(start, end):
        if energy[i] < 0:
            seg[n] = i
            n += 1
    return n   # the segment's used-count

# In __main__: pool.map yields one (segment, n_used) per worker
# Then concatenate all segments:
to_remove = np.concatenate([seg[:n] for seg, n in segments])
```

Each worker writes to its own segment — no contention. The `np.concatenate` at the end runs serially in `__main__`, but its cost is proportional to *total* removes, not to N. For 10,000 removes from a 1M table, the concat is microseconds.

This is the canonical pattern: *parallel filter, serial merge.* Same shape as MapReduce's shuffle step.

## Exercise 5 — Find the bandwidth ceiling

```
N           bandwidth-bound ceiling
100,000     ~8× (everything fits in per-core caches; aggregate scales)
1,000,000   ~6× (L3-resident; partial sharing)
10,000,000  ~4-5× (RAM-resident; bandwidth ceiling)
100,000,000 ~3-4× (deeply RAM; bus is the bottleneck)
```

Small N has *per-core bandwidth* (private L1/L2 plus shared L3 portion); workers don't compete much. Large N has *aggregate memory bandwidth*; all workers compete for the same DRAM bus.

Your machine's bus-bandwidth ceiling is the *maximum* parallel speedup at large N for memory-bound work. For a typical dual-channel desktop, that's 4-6×; quad-channel server class, 8-12×; single-channel laptop or Pi, 2-3×.

## Exercise 6 — Per-tick dispatch costs IPC

```python
# Per-tick dispatch — one pool.map per tick
for _ in range(100):
    pool.map(worker_one_tick, boundaries)
```

vs. the rig's *per-run* dispatch (one `pool.map` total, each worker runs all 100 ticks). The per-tick version pays one IPC round-trip per tick — typically 100-500 µs depending on platform. At 100 ticks × 8 workers, that's 80-400 ms of pure IPC. For a tick budget of 33 ms, you have spent the entire budget on dispatch.

The speedup curve sags lower for the per-tick version. The lesson: *batch when the access pattern allows*. If a worker can do 100 ticks worth of work on its partition before reporting back, IPC is amortised. If every tick needs a sync (e.g., the simulator's `cleanup` must see all workers' segments), then the IPC is unavoidable and the work-per-IPC must dominate it.

## Exercise 7 — Find your physical core count

```sh
lscpu | grep 'Core(s) per socket'                   # physical cores per socket
lscpu | grep 'Socket(s)'                             # how many sockets
python -c "import os; print(os.cpu_count())"        # logical (SMT-doubled)
```

Most desktops/laptops are single-socket; `Core(s) per socket × Socket(s)` is the physical count. `os.cpu_count()` returns logical (typically 2× physical on Intel/AMD SMT). For compute-bound work, target `n_workers = physical_count`; for memory-bound work, target around half-to-full physical (more workers compete for bandwidth without doing more work).

## Exercise 8 — `concurrent.futures` comparison

```python
from concurrent.futures import ProcessPoolExecutor

with ProcessPoolExecutor(max_workers=8) as ex:
    list(ex.map(worker, boundaries))
```

Performance is essentially the same as `multiprocessing.Pool` because they share the same underlying mechanics. `concurrent.futures` has a cleaner API for one-off submission (`submit` returns a `Future`) and integrates with asyncio (`run_in_executor`). `multiprocessing.Pool` has richer options for initializer, maxtasksperchild, and graceful shutdown.

Pick one and standardise. The choice is style, not performance.

## Exercise 9 — A pure-Python anti-comparison

```python
import time
N = 1_000_000
pos = [0.0] * N
vel = [1.0] * N

# Pure Python serial
t = time.perf_counter()
for i in range(N): pos[i] += vel[i] * 0.033
print(f"pure-Python serial: {(time.perf_counter()-t)*1000:.0f} ms")

# Pure Python threaded
from threading import Thread
def thread_motion(pos, vel, start, end):
    for i in range(start, end): pos[i] += vel[i] * 0.033
pos[:] = [0.0] * N
ts = [Thread(target=thread_motion, args=(pos, vel, i*N//8, (i+1)*N//8)) for i in range(8)]
t = time.perf_counter()
for th in ts: th.start()
for th in ts: th.join()
print(f"pure-Python × 8 threads: {(time.perf_counter()-t)*1000:.0f} ms")

# numpy bulk-op serial
import numpy as np
pos_np = np.zeros(N, dtype=np.float32)
vel_np = np.ones(N, dtype=np.float32)
t = time.perf_counter()
pos_np += vel_np * 0.033
print(f"numpy bulk-op serial: {(time.perf_counter()-t)*1000:.2f} ms")
```

Typical:

```
pure-Python serial:     150 ms
pure-Python × 8 threads: 155 ms     (no speedup — GIL serialises the loop)
numpy bulk-op serial:    0.3 ms     (500× faster than any pure-Python form)
```

The lesson is hard. Multiprocessing scales work that is *already shaped for it* (bulk numpy ops on disjoint slices). It does not rescue work that was wrong-shaped to begin with. **The right move is not to parallelise the Python loop; it is to leave the Python loop entirely.** Once you are inside numpy, parallelism is an architecture you can earn; until then, the gap to numpy is bigger than the gap from numpy to parallelism.
