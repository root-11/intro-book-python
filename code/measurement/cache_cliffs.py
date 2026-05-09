# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy"]
# ///
"""
§1 exhibit — the interpreter masks the cliffs; numpy reveals them.

For each working-set size N from 10K to 100M, sum N int64 values three ways:

    A. Python list:      sum(lst)
    B. numpy sequential: arr.sum()
    C. numpy gather:     arr[idx].sum()  with idx a shuffled permutation

We print *time per element*. Two lessons:

    1. Method A — the Python list — is roughly flat in ns/element across
       sizes. Interpreter dispatch (per-iteration `PyObject_Add`, `PyLong`
       boxing/unboxing, refcount work) dominates the per-element cost,
       so the cache hierarchy is invisible from inside pure Python.

    2. Methods B and C — numpy — reveal a staircase. Sequential sums are
       bandwidth-limited but well-prefetched. Random gather forces a fresh
       address per element; once the working set leaves L1, then L2, then
       L3, the cost stairsteps up. The ratio of B to C at large N is
       roughly the L1-to-RAM cost ratio for your machine.

Subprocess per size so RSS pressure does not interact across runs.

Run:
    uv run code/measurement/cache_cliffs.py
"""

import gc
import multiprocessing as mp
import time

SIZES = [10_000, 100_000, 1_000_000, 10_000_000, 100_000_000]
WARMUP_REPS = 1
MEASURE_REPS = 3


def time_call(fn):
    best = float("inf")
    for _ in range(WARMUP_REPS):
        fn()
    for _ in range(MEASURE_REPS):
        gc.collect()
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        best = min(best, t1 - t0)
    return best


def measure_size(n, q):
    import numpy as np
    rng = np.random.default_rng(seed=0xC0FFEE)
    arr = rng.integers(0, 1000, size=n, dtype=np.int64)
    lst = arr.tolist()
    idx = rng.permutation(n)

    t_list = time_call(lambda: sum(lst))
    t_seq = time_call(lambda: int(arr.sum()))
    t_gather = time_call(lambda: int(arr[idx].sum()))

    q.put({
        "n": n,
        "list_ns_per_elem": t_list / n * 1e9,
        "numpy_seq_ns_per_elem": t_seq / n * 1e9,
        "numpy_gather_ns_per_elem": t_gather / n * 1e9,
    })


def main():
    header = (f"{'N':>12}  {'Python list':>14}  {'numpy seq':>12}  "
              f"{'numpy gather':>14}  {'gather/seq':>11}")
    print(header)
    print("-" * len(header))

    for n in SIZES:
        q = mp.Queue()
        p = mp.Process(target=measure_size, args=(n, q))
        p.start()
        r = q.get()
        p.join()
        ratio = r["numpy_gather_ns_per_elem"] / r["numpy_seq_ns_per_elem"]
        print(f"{r['n']:>12,}  "
              f"{r['list_ns_per_elem']:>11.2f} ns  "
              f"{r['numpy_seq_ns_per_elem']:>9.3f} ns  "
              f"{r['numpy_gather_ns_per_elem']:>11.2f} ns  "
              f"{ratio:>10.1f}×")

    print()
    print("Read the columns:")
    print("  Python list — roughly flat across sizes; interpreter dispatch dominates.")
    print("  numpy seq   — staircase; cliffs reveal L1/L2/L3/RAM transitions.")
    print("  numpy gather — random access; gap to seq widens as working set spills caches.")


if __name__ == "__main__":
    main()
