# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy"]
# ///
"""
§31 exhibit — the GIL is a creativity failure, not a language failure.

Two motion-system workloads run in the same multiprocessing+shared_memory
rig over N=10,000,000 creatures, for 100 ticks each:

    A. memory-bound:  pos += vel * dt
       12 bytes accessed per element (4 read vel + 8 read/write pos), 2
       arithmetic ops. The work is dominated by memory traffic; speedup
       caps at the aggregate memory-bandwidth ceiling.

    B. compute-bound: out += sin(x)**2 + cos(x)**2
       Same byte accesses, but the per-element CPU work is much larger.
       Speedup is bounded by physical-core count, with diminishing
       returns from SMT (logical cores past the physical count).

The rig:
    - __main__ allocates pos and vel in shared memory.
    - Workers attach to the same shared region and receive partition
      descriptors (start, end). Each worker writes only its slice; no
      copy crosses the process boundary.
    - One pool.map dispatch per run: each worker runs ALL 100 ticks on
      its partition before returning. This amortises IPC overhead — a
      naive form (one pool.map per tick) is slower; see the per-tick
      exercise in §31.

Two reasons the speedup curve does not reach physical-core count:
    1. Memory bandwidth saturation (workload A — caps below 8× even on
       8 physical cores, because the bus is full).
    2. SMT diminishing returns (workload B — 16 logical cores on 8
       physical cores; the second SMT thread per core adds 10-30%, not
       100%).

Run:
    uv run code/measurement/parallel_motion.py
"""

import os
import time
from multiprocessing import shared_memory, Pool

import numpy as np


N = 10_000_000
N_TICKS = 100
DT = np.float32(1.0 / 30.0)
SHAPE = (2, N)              # row 0 = pos, row 1 = vel
DTYPE = np.float32


# Worker-side context. Set by init_worker so the SharedMemory handle is
# opened once per worker (not per task).
_arr = None
_shm = None


def init_worker(shm_name: str) -> None:
    global _arr, _shm
    _shm = shared_memory.SharedMemory(name=shm_name)
    _arr = np.ndarray(SHAPE, dtype=DTYPE, buffer=_shm.buf)


def all_ticks_memory_bound(args: tuple[int, int]) -> None:
    """Run all N_TICKS ticks of `pos += vel * dt` on this partition."""
    start, end = args
    for _ in range(N_TICKS):
        _arr[0, start:end] += _arr[1, start:end] * DT


def all_ticks_compute_bound(args: tuple[int, int]) -> None:
    """Run all N_TICKS ticks of `out += sin² + cos²` on this partition.
    The result is ~1.0 per element — the chapter cares about the cost,
    not the answer."""
    start, end = args
    for _ in range(N_TICKS):
        s = np.sin(_arr[1, start:end])
        c = np.cos(_arr[1, start:end])
        _arr[0, start:end] += s * s + c * c


def cleanup_worker(_=None) -> None:
    global _arr, _shm
    _arr = None
    if _shm is not None:
        _shm.close()
        _shm = None


def setup_shared() -> tuple[shared_memory.SharedMemory, np.ndarray]:
    nbytes = int(np.prod(SHAPE) * np.dtype(DTYPE).itemsize)
    shm = shared_memory.SharedMemory(create=True, size=nbytes)
    arr = np.ndarray(SHAPE, dtype=DTYPE, buffer=shm.buf)
    return shm, arr


def reset_state(arr: np.ndarray, seed: int) -> None:
    rng = np.random.default_rng(seed)
    arr[0] = rng.random(N, dtype=np.float32)
    arr[1] = rng.random(N, dtype=np.float32) * np.float32(0.1)


def run_serial(fn) -> float:
    """Single-process baseline — runs in __main__, also through `fn` so
    apples-to-apples with the parallel runs."""
    t0 = time.perf_counter()
    fn((0, N))
    t1 = time.perf_counter()
    return t1 - t0


def run_parallel(shm_name: str, n_workers: int, fn) -> float:
    chunk = N // n_workers
    boundaries = [
        (i * chunk, (i + 1) * chunk if i < n_workers - 1 else N)
        for i in range(n_workers)
    ]
    warmup = [(0, 0)] * n_workers
    pool = Pool(processes=n_workers, initializer=init_worker, initargs=(shm_name,))
    try:
        pool.map(fn, warmup)             # spin up workers; do no work
        t0 = time.perf_counter()
        pool.map(fn, boundaries)          # ONE round-trip; each worker runs all ticks
        t1 = time.perf_counter()
    finally:
        pool.map(cleanup_worker, [None] * n_workers)
        pool.close()
        pool.join()
    return t1 - t0


def main() -> None:
    cores = os.cpu_count() or 4
    workers_to_try = sorted({1, 2, 4, min(8, cores), cores})
    SEED = 0xCAFE

    print(f"N = {N:,} creatures, {N_TICKS} ticks per run, dtype = {DTYPE.__name__}")
    print(f"Working set: {2 * N * 4 / 1024**2:.1f} MB (pos + vel, both float32)")
    print(f"Logical CPUs reported: {cores}\n")

    shm, arr = setup_shared()
    try:
        # init the worker globals in the main process for the serial runs.
        global _arr, _shm
        _arr = arr
        _shm = shm

        for label, fn in [
            ("memory-bound:  pos += vel * dt",            all_ticks_memory_bound),
            ("compute-bound: out += sin(x)**2 + cos(x)**2", all_ticks_compute_bound),
        ]:
            print(f"=== {label} ===")
            reset_state(arr, SEED)
            serial_s = run_serial(fn)
            serial_final = float(arr[0].sum())

            header = f"{'workers':>8}  {'wall (s)':>10}  {'speedup':>8}  {'final checksum':>18}"
            print(header)
            print("-" * len(header))
            print(f"{'serial':>8}  {serial_s:>10.3f}  {1.0:>8.2f}  {serial_final:>18.4f}")

            for nw in workers_to_try:
                reset_state(arr, SEED)
                par_s = run_parallel(shm.name, nw, fn)
                par_final = float(arr[0].sum())
                speedup = serial_s / par_s
                marker = " " if abs(par_final - serial_final) < 1.0 else " *MISMATCH*"
                print(f"{nw:>8}  {par_s:>10.3f}  {speedup:>8.2f}  "
                      f"{par_final:>18.4f}{marker}")
            print()

    finally:
        shm.close()
        shm.unlink()

    print("Read the curves: the memory-bound ceiling is set by aggregate")
    print("memory bandwidth; the compute-bound ceiling is set by physical")
    print("core count, with diminishing returns from SMT past that.")


if __name__ == "__main__":
    main()
