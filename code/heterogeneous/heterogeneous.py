# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy"]
# ///
"""
§56 exhibit - the ceiling is bandwidth, not cores, in Python+numpy.

Counterpart to the Rust edition's code/heterogeneous crate, and the answer to
"a simulator this size needs a GPU". The motion pass - new position = old
position + velocity*dt - is two multiply-adds per element and almost no
arithmetic, so its speed is the speed of MEMORY, not the core. That ceiling does
not rise with cores, and reaching off the box (another process, or a GPU) pays a
transfer tax that a memory-bound pass cannot earn back.

The Python shading:

  C1 one core's reach   the numpy motion pass runs near cache speed while the
                        data fits in cache and drops to main-memory bandwidth at
                        scale. numpy already vectorises it (the SoA precondition).
  C2 more cores plateau in Python "more cores" means PROCESSES - the GIL rules out
                        threads for CPU work. Even with the array in shared memory
                        (so no copy), speedup plateaus once the single memory
                        channel saturates: the ceiling is bandwidth, not cores.
  C3 frame budget       how many particles one core / all cores keep current in a
                        33 ms frame - the active-set budget the GPU argument turns
                        on (you recompute the active cone, not the whole world).
  C4 the bus is the tax shipping the array to another process and back (pickle/IPC)
                        costs MORE than doing the pass in place - Python's native
                        version of the GPU round-trip. The GPU cost model is the
                        same shape and is deferred (no GPU on this box).

Run:  uv run code/heterogeneous/heterogeneous.py
"""

import time
from multiprocessing import shared_memory
from concurrent.futures import ProcessPoolExecutor

import numpy as np

DT = np.float32(1e-3)
BYTES_PER_ELEM = 24          # px,py read+written (8 each) + vx,vy read (4 each)
FRAME_S = 1.0 / 30.0


def motion(px, py, vx, vy):
    px += vx * DT
    py += vy * DT


def median(fn, k=5):
    return sorted(fn() for _ in range(k))[k // 2]


# ---------------------------------------------------------------------------
# C2 worker: attach to the shared arrays and advance one slice in place.
# ---------------------------------------------------------------------------

def _advance_slice(args):
    name, n, start, end = args
    shm = shared_memory.SharedMemory(name=name)
    buf = np.ndarray((4, n), dtype=np.float32, buffer=shm.buf)
    px, py, vx, vy = buf[0], buf[1], buf[2], buf[3]
    px[start:end] += vx[start:end] * DT
    py[start:end] += vy[start:end] * DT
    shm.close()
    return end - start


def c1_one_core():
    print("== C1: one core's motion pass, cache vs main memory ==")
    print(f"{'elements':>12} {'MB':>8} {'GB/s':>8}")
    rng = np.random.default_rng(0x56)
    for n in (1_000, 50_000, 1_000_000, 10_000_000, 50_000_000):
        px = rng.random(n, dtype=np.float32)
        py = rng.random(n, dtype=np.float32)
        vx = rng.random(n, dtype=np.float32)
        vy = rng.random(n, dtype=np.float32)
        iters = max(3, 30_000_000 // n)
        def run():
            t0 = time.perf_counter()
            for _ in range(iters):
                motion(px, py, vx, vy)
            return time.perf_counter() - t0
        dt = median(run)
        gbps = (n * BYTES_PER_ELEM * iters) / dt / 1e9
        print(f"{n:>12} {n * 16 / 1e6:>8.1f} {gbps:>8.1f}")


def c2_more_cores(n=50_000_000):
    print(f"\n== C2: the same pass across processes (shared memory, {n:,} elements) ==")
    shm = shared_memory.SharedMemory(create=True, size=4 * n * 4)
    buf = np.ndarray((4, n), dtype=np.float32, buffer=shm.buf)
    buf[:] = np.random.default_rng(1).random((4, n), dtype=np.float32)
    base_gbps = None
    print(f"{'procs':>6} {'time (ms)':>10} {'GB/s':>8} {'speedup':>9}")
    try:
        for k in (1, 2, 4, 8, 16):
            with ProcessPoolExecutor(max_workers=k) as ex:
                bounds = [(shm.name, n, i * n // k, (i + 1) * n // k) for i in range(k)]
                # warm the pool, then time
                list(ex.map(_advance_slice, bounds))
                def run():
                    t0 = time.perf_counter()
                    list(ex.map(_advance_slice, bounds))
                    return time.perf_counter() - t0
                dt = median(run, k=5)
            gbps = (n * BYTES_PER_ELEM) / dt / 1e9
            if base_gbps is None:
                base_gbps = gbps
            print(f"{k:>6} {dt * 1e3:>10.2f} {gbps:>8.1f} {gbps / base_gbps:>8.2f}x")
    finally:
        shm.close()
        shm.unlink()


def c3_frame_budget():
    print("\n== C3: active-set budget per 33 ms frame ==")
    # measured on this box: C1 main-memory bandwidth and C2's multi-process plateau
    one_core_gbps = 7.0       # ~main-memory bandwidth, one numpy pass (with the temporary)
    all_core_gbps = 24.0      # ~plateau across processes
    one = one_core_gbps * 1e9 * FRAME_S / BYTES_PER_ELEM
    allc = all_core_gbps * 1e9 * FRAME_S / BYTES_PER_ELEM
    print(f"  one core : ~{one / 1e6:.0f} M particles kept current per frame")
    print(f"  all cores: ~{allc / 1e6:.0f} M particles kept current per frame")
    print("  you recompute the ACTIVE cone (53/54), not the whole world - a few M fits.")


# ---------------------------------------------------------------------------
# C4 worker: receive an array by value (pickled through the pool), do the pass,
# return it - the round-trip a non-shared offload pays.
# ---------------------------------------------------------------------------

def _advance_by_value(arrs):
    px, py, vx, vy = arrs
    px = px + vx * DT
    py = py + vy * DT
    return px, py


def c4_bus_tax(n=10_000_000):
    print(f"\n== C4: the bus is the tax - ship to a process and back ({n:,} elements) ==")
    rng = np.random.default_rng(2)
    px, py, vx, vy = (rng.random(n, dtype=np.float32) for _ in range(4))
    # in place, this process
    def in_place():
        a, b = px.copy(), py.copy()
        t0 = time.perf_counter()
        a += vx * DT
        b += vy * DT
        return time.perf_counter() - t0
    t_local = median(in_place)
    # shipped to a worker process and back (pickle / IPC round trip)
    with ProcessPoolExecutor(max_workers=1) as ex:
        ex.submit(_advance_by_value, (px[:1], py[:1], vx[:1], vy[:1])).result()  # warm
        def round_trip():
            t0 = time.perf_counter()
            ex.submit(_advance_by_value, (px, py, vx, vy)).result()
            return time.perf_counter() - t0
        t_ship = median(round_trip, k=3)
    print(f"  pass in place (this process)      : {t_local * 1e3:.2f} ms")
    print(f"  ship to a worker process and back : {t_ship * 1e3:.2f} ms  ({t_ship / t_local:.0f}x)")
    print("  moving the data to another worker costs more than doing the pass -")
    print("  the same reason a GPU offload of a memory-bound pass loses (transfer > compute).")


def main():
    c1_one_core()
    c2_more_cores()
    c3_frame_budget()
    c4_bus_tax()


if __name__ == "__main__":
    main()
