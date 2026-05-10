# Solutions: 32 — Partition, don't lock

## Exercise 1 — Run the coordination exhibit

```sh
uv run code/measurement/coordination_patterns.py
```

```
pattern                            total (s)        msgs/sec  p50 jitter   p99 jitter
-------------------------------------------------------------------------------------
1. single shared Queue                4.48          31,283       54.7 µs    400.4 µs
2. per-worker Queue                  14.63           9,567      171.2 µs   3480.3 µs
3. shared numpy array                13.87          10,090        0.1 µs   5960.9 µs
```

Three readings of this particular run:

- **The single shared queue is fastest by throughput**; the shared numpy array has the lowest *p50 jitter* (sub-microsecond) but higher *p99* (multi-millisecond spikes when the OS preempts a spinning worker).
- **Per-worker queues are the slowest on every metric.** The lock-contention argument from textbooks loses to the pipelining-through-one-queue effect at this workload size.
- **Numbers shift by machine and CPython version.** On other hardware the shared-array pattern can run 5-20× faster on throughput too (the chapter prose's numbers); your run will likely sit between these two.

Coordination events per 30 Hz tick budget (33 ms):

| pattern              | events/tick |
|----------------------|------------:|
| single Queue         |     ~1,000  |
| per-worker Queue     |       ~300  |
| shared numpy array   |       ~330  |

Any of them is enough for a *batched* simulator (10-100 phase signals per tick). None of them is enough for *per-creature* signalling (1M events per tick).

## Exercise 2 — The batching threshold on your machine

At 30K msgs/sec coordination and ~30M ops/sec inner-loop numpy work (one motion update on 1M creatures in ~30 ms):

```
coordination cost per message: 33 µs
inner-loop work per message:   2 µs per creature × partition_size
For coordination ≤ 10% of work:
  33 µs ≤ 0.1 × 2 µs × partition_size
  partition_size ≥ 165 creatures
```

So **partitions ≥ 200 creatures** keep coordination cost under 10% of work cost. Below 200, coordination dominates; above, work dominates. For the simulator's 1M creatures over 8 workers, each partition is 125,000 — three orders of magnitude past the threshold. Coordination is negligible.

The threshold matters when *partition size shrinks* — e.g., a focal sub-system that only acts on 100 creatures should not be partitioned across 8 workers (coordination would dominate); it should be run by one worker.

## Exercise 3 — Pre-assigned partitions

```python
# At startup
boundaries = [(i*N//W, (i+1)*N//W) for i in range(W)]

def init_worker(my_id, my_boundaries, shm_name):
    global _start, _end, _shm, _arr
    _start, _end = my_boundaries
    _shm = SharedMemory(shm_name)
    _arr = np.ndarray((NUM_COLUMNS, N), dtype=np.float32, buffer=_shm.buf)

# Per phase, the only signal is "run system X"
def run_phase(system_id):
    # _start, _end already known
    if system_id == 0:   _arr[0, _start:_end] += _arr[1, _start:_end] * DT
    elif system_id == 1: _arr[2, _start:_end] *= 0.99
    # ...
```

Compared to a re-sending version (`pool.map(motion_worker, [(i*N//W, (i+1)*N//W) for i in range(W)])`):

- Pre-assigned: one signal per phase = one small int per worker (~1 µs)
- Re-sending: tuple of two ints per worker, pickled and unpickled (~10-30 µs)

At 100 phases per tick × 8 workers, that's 800-2400 µs vs ~800 µs. Real savings, but small in absolute terms — the *architectural* benefit (workers can keep state across phases, cached) matters more than the marginal IPC.

## Exercise 4 — The DAG-as-array

```python
# DAG_PROGRAM[phase, worker_id] = system_id_to_run (or 0 for "idle")
DAG_PROGRAM = np.array([
    [1, 0, 0, 0, 0, 0, 0, 0],     # phase 0: only worker 0 runs system 1
    [2, 2, 2, 2, 2, 2, 2, 0],     # phase 1: 7 workers run system 2 (partitioned)
    [3, 4, 5, 0, 0, 0, 0, 0],     # phase 2: three different systems run in parallel
    [6, 6, 6, 6, 0, 0, 0, 0],     # phase 3: 4 workers run system 6
    [7, 0, 0, 0, 0, 0, 0, 0],     # phase 4: cleanup
], dtype=np.int8)

# main bumps a generation counter; workers spin until it matches their phase index
def worker_loop(my_id, gen_array, dag_array, shm_name):
    expected_phase = 0
    while True:
        while int(gen_array[0]) != expected_phase: pass     # spin-wait
        system_id = int(dag_array[expected_phase, my_id])
        if system_id != 0:
            run_system(system_id, my_id)
        # signal done by incrementing the worker's ack counter
        gen_array[1 + my_id] += 1
        expected_phase += 1
```

Correctness is testable: pin the DAG, run for N ticks under the shared-array implementation, then run the same with a `for system in dag: run_system_serial(system)` baseline. Compare world hashes. They must match.

## Exercise 5 — Load-balanced partitioning

```python
# Per-worker timestamps stamped at phase end
# After each tick, main reads them and recomputes boundaries

def rebalance(boundaries, last_phase_durations, total_n):
    """Give larger slices to faster workers (smaller durations)."""
    inv_speed = 1.0 / np.maximum(last_phase_durations, 1e-6)
    weight = inv_speed / inv_speed.sum()
    new_sizes = (weight * total_n).astype(np.int64)
    # Build cumulative boundaries from sizes
    cum = np.cumsum(new_sizes)
    new_boundaries = []
    start = 0
    for end in cum:
        new_boundaries.append((start, int(end)))
        start = int(end)
    new_boundaries[-1] = (new_boundaries[-1][0], total_n)        # fix trailing
    return new_boundaries
```

Run for 1000 ticks. Plot per-worker phase times tick by tick. The boundaries oscillate at first, then converge to a steady state where every worker finishes its phase at roughly the same wall time. The convergence rate depends on the workload's stability — a flat-world uniform simulator converges fast; one with bursty events stays jittery.

This is *closed-loop scheduling*. Same pattern as TCP's congestion control: observe, react, repeat. Main has the timestamps; main decides.

## Exercise 6 — Workload heterogeneity

```python
# Construct: 80% of the work in 20% of the partitions
def heavy_partition(i, start, end, _arr):
    # workers 0,1 do expensive work; the rest do cheap work
    work_factor = 10 if i < 2 else 1
    for _ in range(work_factor):
        _arr[0, start:end] += _arr[1, start:end] * DT
```

Fixed equal-sized partitioning: workers 0 and 1 take 10× longer per phase than workers 2-7. The phase wall time is dominated by workers 0 and 1 — *the slowest worker sets the phase budget*. Workers 2-7 sit idle, wasting cores.

Load-balanced version (from exercise 5): boundaries converge to small slices for workers 0 and 1, large slices for workers 2-7. Steady state: all workers finish in roughly the same wall time. The phase budget shrinks ~3× because the slow workers got less work.

This is the right shape for any simulator where workload is non-uniform across space (MMORPGs with cities, fluid simulations with turbulence, traffic with congestion). Static partitioning is a special case that works only when the work is uniform.

## Exercise 7 — The boundary-builder lives in `__main__`

```python
# Worker tries to compute its own slice — fragile
def bad_worker(my_id, n_workers, shm_name):
    s = SharedMemory(shm_name)
    arr = np.ndarray(SHAPE, dtype=DTYPE, buffer=s.buf)
    N = arr.shape[1]                          # ← reads N from the buffer
    start = my_id * N // n_workers
    end   = (my_id + 1) * N // n_workers
    arr[0, start:end] += arr[1, start:end] * DT

# Main mutates N mid-tick:
# anti-pattern: bad!
shm = SharedMemory(create=True, size=(2 * 1_000_000 * 4 + 64))
# ...
# tick 1 fires with N=1_000_000
# main resizes the array somehow during tick 2 (in reality you can't easily resize shared memory, but if N is read from a counter:
shm_n = ...                                    # shared counter
shm_n[0] = 2_000_000                            # mid-tick — chaos
# now workers think they own [0, 1_000_000/W) but the data layout changed
```

The disciplined form: `__main__` computes boundaries once, writes them to a shared array, workers read their slice from the shared array. `__main__` is the single writer of the boundaries; workers are read-only consumers. If `__main__` wants to change boundaries (rebalance), it does so *between* phases, never during.

Letting workers compute their own slice from `(my_id, n_workers, N)` is fragile because `N` and `n_workers` must agree across all workers and main. Centralising the boundaries in `__main__` eliminates the disagreement.

## Exercise 8 — `Event` instead of busy-wait (stretch)

```python
# Worker spins on shared array
while gen_array[0] != expected:
    pass

# Worker uses Event.wait()
event.wait()
event.clear()
```

`event.wait()` puts the worker to sleep at the kernel level. The wakeup involves an inter-process signal — typically 50-200 µs of overhead. Compared to the spin-loop (~0.1-1 µs latency), the Event-based pattern is 50-500× slower per round-trip.

But: the spinning worker pins a CPU core at 100% even when there's no work. On a laptop, this means heat and battery drain. On a shared server, it crowds out other processes. **Event-based wakeup is the right choice for low-frequency coordination** (≤ a few hundred wakeups per second, e.g. background batch jobs). Spin-loop is right for *high-frequency* coordination on dedicated cores (a real-time simulator at 1 kHz).

## Exercise 9 — The 1 kHz physics-engine question (stretch)

```
Tick budget at 1 kHz: 1 ms = 1000 µs

If coordination is 1 µs/event (shared array, no contention):
  budget allows 1000 coordination events / tick
  but a typical physics simulator wants ~50 system phases × 8 workers = 400 events / tick
  fits — coordination uses 40% of the budget

If coordination is 30 µs/event (queue-based):
  budget allows 33 events / tick
  same simulator needs 400 events → exceeds budget by 12×
  does not fit
```

At 1 kHz the simulator must use shared-array coordination *and* still has 40% of its budget consumed by coordination alone. Most physics engines run at 1 kHz or higher (game physics often at 240 Hz, control systems at 1-10 kHz). The arithmetic above is why those engines are usually in C++ or Rust — the per-event coordination cost in those languages is ~10-100 ns, leaving room for the actual physics.

The escalation: at the point where Python coordination eats the budget, the work shifts to maturin (Rust + PyO3) for the inner loop. Same architecture, same partition-don't-lock pattern, but with sub-microsecond coordination via Rust's `crossbeam::channel` or `std::sync::atomic`. **The architecture is portable; the language is the tooling decision.**
