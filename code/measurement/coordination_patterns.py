# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy"]
# ///
"""
§32 exhibit — coprocessors are IOPS-limited; batching is forced.

Three measurements:

    1. Warmup cost. Spawning N worker processes and getting them
       attached to the shared-memory region takes time on the order
       of seconds. Long-running simulators amortise it; short-lived
       scripts cannot.

    2. Coordination throughput, three patterns.
       - Single shared multiprocessing.Queue (workers contend on the
         queue's lock).
       - Per-worker multiprocessing.Queue (no contention; one queue
         per worker, plus a shared ack queue).
       - Shared numpy array, busy-wait coordination (no kernel
         involvement; main and workers communicate by writing and
         spinning on shared bytes).

    3. Per-worker jitter. For each round (main signals all workers,
       all workers complete), measure the spread between the first
       and last worker to finish. High jitter means "all done" cannot
       be tightly scheduled; the slowest worker sets the phase floor.

The trivial task: each worker increments a per-worker counter in a
cache-line-padded shared region. The work is nanoseconds; the cost is
the coordination round-trip.

Run:
    uv run code/measurement/coordination_patterns.py
"""

import os
import statistics
import time
from multiprocessing import shared_memory, Process, Queue

import numpy as np


N_WORKERS = 7              # leave one logical core for main
N_ROUNDS = 20_000          # rounds of dispatch (each round = 1 task per worker)
PAD_INT64S = 8             # 64 bytes per worker — one cache line


def make_padded_arr(name: str, n_int64s: int) -> tuple[shared_memory.SharedMemory, np.ndarray]:
    shm = shared_memory.SharedMemory(create=True, name=name, size=n_int64s * 8)
    arr = np.ndarray((n_int64s,), dtype=np.int64, buffer=shm.buf)
    arr[:] = 0
    return shm, arr


# ---------------------------------------------------------------------
# Pattern 1: single shared queue
# ---------------------------------------------------------------------

def worker_single_q(my_id: int, task_q: Queue, ack_q: Queue, shm_name: str) -> None:
    shm = shared_memory.SharedMemory(name=shm_name)
    counters = np.ndarray((N_WORKERS * PAD_INT64S,), dtype=np.int64, buffer=shm.buf)
    while True:
        msg = task_q.get()
        if msg is None:
            break
        counters[my_id * PAD_INT64S] += 1
        ack_q.put((my_id, time.perf_counter_ns()))
    shm.close()


def run_single_queue() -> tuple[float, list[float]]:
    shm, _ = make_padded_arr("coord_single", N_WORKERS * PAD_INT64S)
    task_q: Queue = Queue()
    ack_q: Queue = Queue()
    workers = [
        Process(target=worker_single_q, args=(i, task_q, ack_q, shm.name))
        for i in range(N_WORKERS)
    ]
    for p in workers:
        p.start()

    jitter_per_round = []
    t0 = time.perf_counter()
    for _ in range(N_ROUNDS):
        for _ in range(N_WORKERS):
            task_q.put(1)
        completions = [ack_q.get()[1] for _ in range(N_WORKERS)]
        completions.sort()
        jitter_per_round.append((completions[-1] - completions[0]) / 1000.0)  # µs
    t1 = time.perf_counter()

    for _ in range(N_WORKERS):
        task_q.put(None)
    for p in workers:
        p.join()
    shm.close()
    shm.unlink()
    return t1 - t0, jitter_per_round


# ---------------------------------------------------------------------
# Pattern 2: per-worker queue
# ---------------------------------------------------------------------

def worker_own_q(my_id: int, my_q: Queue, ack_q: Queue, shm_name: str) -> None:
    shm = shared_memory.SharedMemory(name=shm_name)
    counters = np.ndarray((N_WORKERS * PAD_INT64S,), dtype=np.int64, buffer=shm.buf)
    while True:
        msg = my_q.get()
        if msg is None:
            break
        counters[my_id * PAD_INT64S] += 1
        ack_q.put((my_id, time.perf_counter_ns()))
    shm.close()


def run_per_worker_queue() -> tuple[float, list[float]]:
    shm, _ = make_padded_arr("coord_perworker", N_WORKERS * PAD_INT64S)
    qs: list[Queue] = [Queue() for _ in range(N_WORKERS)]
    ack_q: Queue = Queue()
    workers = [
        Process(target=worker_own_q, args=(i, qs[i], ack_q, shm.name))
        for i in range(N_WORKERS)
    ]
    for p in workers:
        p.start()

    jitter_per_round = []
    t0 = time.perf_counter()
    for _ in range(N_ROUNDS):
        for q in qs:
            q.put(1)
        completions = [ack_q.get()[1] for _ in range(N_WORKERS)]
        completions.sort()
        jitter_per_round.append((completions[-1] - completions[0]) / 1000.0)
    t1 = time.perf_counter()

    for q in qs:
        q.put(None)
    for p in workers:
        p.join()
    shm.close()
    shm.unlink()
    return t1 - t0, jitter_per_round


# ---------------------------------------------------------------------
# Pattern 3: shared-array coordination, busy-wait
# ---------------------------------------------------------------------
# Layout of the coordination array (all int64):
#   [0]                       : current generation (main increments)
#   [1 .. 1+N_WORKERS]        : per-worker task slot (-1 = none)
#   [1+N_WORKERS .. 1+2*N]    : per-worker ack counter
#   [1+2*N .. 1+2*N+N*PAD]    : per-worker completion timestamp (ns)

COORD_HDR = 1
COORD_TASK = 1
COORD_ACK = 1 + N_WORKERS
COORD_TIMESTAMP = 1 + 2 * N_WORKERS
COORD_SIZE = COORD_TIMESTAMP + N_WORKERS


def worker_shared_array(my_id: int, coord_name: str, data_name: str) -> None:
    coord_shm = shared_memory.SharedMemory(name=coord_name)
    coord = np.ndarray((COORD_SIZE,), dtype=np.int64, buffer=coord_shm.buf)
    data_shm = shared_memory.SharedMemory(name=data_name)
    counters = np.ndarray((N_WORKERS * PAD_INT64S,), dtype=np.int64, buffer=data_shm.buf)

    seen_gen = 0
    while True:
        # Spin-wait for the next generation to arrive.
        while coord[0] == seen_gen:
            pass
        gen = int(coord[0])
        if gen < 0:
            break
        # Read my task; do the work.
        if coord[COORD_TASK + my_id] >= 0:
            counters[my_id * PAD_INT64S] += 1
        # Stamp completion time, increment my ack.
        coord[COORD_TIMESTAMP + my_id] = time.perf_counter_ns()
        coord[COORD_ACK + my_id] += 1
        seen_gen = gen

    coord_shm.close()
    data_shm.close()


def run_shared_array() -> tuple[float, list[float]]:
    coord_shm, coord = make_padded_arr("coord_shared", COORD_SIZE)
    data_shm, _ = make_padded_arr("coord_shared_data", N_WORKERS * PAD_INT64S)
    workers = [
        Process(target=worker_shared_array, args=(i, coord_shm.name, data_shm.name))
        for i in range(N_WORKERS)
    ]
    for p in workers:
        p.start()

    jitter_per_round = []
    t0 = time.perf_counter()
    for tick in range(1, N_ROUNDS + 1):
        # Set tasks; release workers by bumping the generation.
        coord[COORD_TASK : COORD_TASK + N_WORKERS] = 1
        coord[0] = tick
        # Spin-wait for all acks to reach `tick`.
        while not (coord[COORD_ACK : COORD_ACK + N_WORKERS] == tick).all():
            pass
        completions = sorted(int(c) for c in coord[COORD_TIMESTAMP : COORD_TIMESTAMP + N_WORKERS])
        jitter_per_round.append((completions[-1] - completions[0]) / 1000.0)
    t1 = time.perf_counter()

    coord[0] = -1  # shutdown
    for p in workers:
        p.join()
    coord_shm.close()
    coord_shm.unlink()
    data_shm.close()
    data_shm.unlink()
    return t1 - t0, jitter_per_round


# ---------------------------------------------------------------------
# Warmup measurement
# ---------------------------------------------------------------------

def measure_warmup() -> float:
    shm, _ = make_padded_arr("coord_warmup", N_WORKERS * PAD_INT64S)
    qs: list[Queue] = [Queue() for _ in range(N_WORKERS)]
    ack_q: Queue = Queue()
    t0 = time.perf_counter()
    workers = [
        Process(target=worker_own_q, args=(i, qs[i], ack_q, shm.name))
        for i in range(N_WORKERS)
    ]
    for p in workers:
        p.start()
    # First round-trip: ensures all workers have imported numpy and
    # attached to shm.
    for q in qs:
        q.put(1)
    for _ in range(N_WORKERS):
        ack_q.get()
    t1 = time.perf_counter()
    for q in qs:
        q.put(None)
    for p in workers:
        p.join()
    shm.close()
    shm.unlink()
    return t1 - t0


# ---------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------

def report(label: str, total_s: float, jitter_us: list[float]) -> None:
    n_messages = N_ROUNDS * N_WORKERS  # main → worker round-trips
    rate = n_messages / total_s
    p50 = statistics.median(jitter_us)
    p95 = statistics.quantiles(jitter_us, n=20)[18]  # 95th percentile
    p99 = statistics.quantiles(jitter_us, n=100)[98]
    print(f"{label:<32}  {total_s:>8.2f}  {rate:>14,.0f}  "
          f"{p50:>9.1f}  {p95:>9.1f}  {p99:>9.1f}")


def main() -> None:
    cores = os.cpu_count() or 8
    print(f"Logical CPUs: {cores};  workers: {N_WORKERS} (one logical core left for main)")
    print(f"Rounds: {N_ROUNDS:,};  total round-trips per pattern: {N_ROUNDS * N_WORKERS:,}\n")

    warmup_s = measure_warmup()
    print(f"Warmup (spawn {N_WORKERS} workers + first round-trip): {warmup_s:.2f} s\n")

    header = (f"{'pattern':<32}  {'total (s)':>10}  {'msgs/sec':>14}  "
              f"{'jitter p50':>9}  {'p95':>9}  {'p99':>9}")
    print(header)
    print("-" * len(header))

    total_s, jitter_us = run_single_queue()
    report("1. single shared Queue", total_s, jitter_us)

    total_s, jitter_us = run_per_worker_queue()
    report("2. per-worker Queue", total_s, jitter_us)

    total_s, jitter_us = run_shared_array()
    report("3. shared numpy array", total_s, jitter_us)

    print("\nJitter columns are microseconds: spread between the fastest")
    print("and slowest worker to ack each round (lower = more predictable).")


if __name__ == "__main__":
    main()
