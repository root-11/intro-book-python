# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy"]
# ///
"""
§33 exhibit - false sharing through multiprocessing.shared_memory.

N worker processes each increment their own counter in a shared int64 array.
When the counters are packed (8 int64 = 64 bytes = one cache line) every
write invalidates the other workers' copies; the cache-coherence protocol
serialises writes that never logically conflict. Pad each counter onto its
own 64-byte line and the contention disappears.

Three timings of the same total work (N_WORKERS x ITERS increments):
    packed  - counters[i] for worker i; all in one cache line.
    padded  - counters[i * STRIDE]; each on its own 64-byte line.
    single  - one process does all the increments.

Caveat the chapter makes concrete: in *pure Python* the per-iteration
interpreter cost is large, so it partly masks the coherence penalty - the
same way the interpreter masks the cache cliffs in §27. The effect is real
but muted compared with the compiled/numpy regime.

    uv run code/measurement/false_sharing.py
"""

import time
from multiprocessing import Process, shared_memory
import numpy as np

ITERS = 2_000_000
STRIDE = 8  # 8 int64 = 64 bytes -> next cache line


def hammer(shm_name, index, iters):
    shm = shared_memory.SharedMemory(name=shm_name)
    arr = np.ndarray((64,), dtype=np.int64, buffer=shm.buf)
    for _ in range(iters):
        arr[index] += 1
    shm.close()


def run(n_workers, stride):
    shm = shared_memory.SharedMemory(create=True, size=64 * 8)
    try:
        arr = np.ndarray((64,), dtype=np.int64, buffer=shm.buf)
        arr[:] = 0
        t0 = time.perf_counter()
        procs = [Process(target=hammer, args=(shm.name, i * stride, ITERS))
                 for i in range(n_workers)]
        for p in procs: p.start()
        for p in procs: p.join()
        return time.perf_counter() - t0
    finally:
        shm.close()
        shm.unlink()


def run_single(n_workers):
    shm = shared_memory.SharedMemory(create=True, size=64 * 8)
    try:
        arr = np.ndarray((64,), dtype=np.int64, buffer=shm.buf)
        arr[:] = 0
        t0 = time.perf_counter()
        for _ in range(n_workers * ITERS):
            arr[0] += 1
        return time.perf_counter() - t0
    finally:
        shm.close()
        shm.unlink()


def main():
    import os
    n = min(8, os.cpu_count() or 8)
    print(f"false sharing: {n} processes x {ITERS:,} increments each\n")
    single = run_single(n)
    padded = run(n, STRIDE)
    packed = run(n, 1)
    print(f"  {'single process (baseline)':<28} {single*1000:8.1f} ms")
    print(f"  {'padded (own cache line)':<28} {padded*1000:8.1f} ms   {single/padded:5.2f}x speedup")
    print(f"  {'packed (false sharing)':<28} {packed*1000:8.1f} ms   {single/packed:5.2f}x speedup")
    print()
    print(f"  padding recovers {single/padded:.1f}x of the {n}x ideal.")
    if packed > single:
        print("  packed parallel run is SLOWER than one process - negative scaling.")
    else:
        print(f"  packed speedup is only {single/packed:.2f}x despite {n} cores.")


if __name__ == "__main__":
    main()
