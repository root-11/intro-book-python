# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy"]
# ///
"""
exhibit - float16 vs float32 vs float64: does half the footprint pay for itself?

float16 is a 2-byte IEEE 754 half. CUDA hardware runs it natively; most CPUs
do not. On a typical x86 / ARM core without AVX-512 FP16, numpy implements
float16 ops by widening each element to float32, computing, and (for stores)
narrowing back. So the trade-off per element is:

    + half the bytes through the cache hierarchy
    - two extra conversions wrapped around the arithmetic

The trade-off only resolves with a measurement. The crossover (if any) is
size-dependent: when the working set is in L1 the conversion penalty
dominates because cache bandwidth is not the bottleneck; once the working
set spills to RAM the bandwidth saving may overtake the conversion cost.

Three operations probe different ratios of bytes-to-arithmetic:

    sum     - one read per element, one accumulate. Bandwidth-bound when
              the array doesn't fit in cache.
    a*b→c   - two reads, one write, one multiply per element. Maximum
              bandwidth pressure.
    a @ b   - two reads, one multiply-add per element. Same memory traffic
              as a*b→c but the FMA is fused; arithmetic intensity matters
              more here.

Five sizes span L1 (~32 KB) to deep RAM (~128 MB at float64). Subprocess
per (size, dtype) so allocations don't interact.

Run:
    uv run code/measurement/float_widths.py
"""

import gc
import multiprocessing as mp
import time


# Element counts. At float32: 4_096 = 16 KB (L1), 65_536 = 256 KB (L2),
# 1_048_576 = 4 MB (L3 border), 8_388_608 = 32 MB (RAM), 33_554_432 = 128 MB
# (deep RAM). Double those byte counts for float64, halve for float16.
SIZES = [4_096, 65_536, 1_048_576, 8_388_608, 33_554_432]
DTYPES = ["float16", "float32", "float64"]

WARMUP_REPS = 1
MEASURE_REPS = 5


def time_call(fn):
    for _ in range(WARMUP_REPS):
        fn()
    best = float("inf")
    for _ in range(MEASURE_REPS):
        gc.collect()
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        best = min(best, t1 - t0)
    return best


def measure(n, dtype_name, q):
    import numpy as np
    rng = np.random.default_rng(seed=0xC0FFEE)
    # Scale to keep sums and dot products inside float16's representable
    # range (max ~65504) across all N. With values in [0, 2**-16), the
    # largest sum at N=33M is ~512 and the largest dot is ~8 - well inside
    # range, well above f16's subnormal threshold (~6e-5).
    scale = np.float32(2.0 ** -16)
    a = (rng.random(n, dtype=np.float32) * scale).astype(dtype_name, copy=False)
    b = (rng.random(n, dtype=np.float32) * scale).astype(dtype_name, copy=False)
    out = np.empty_like(a)

    t_sum = time_call(lambda: float(a.sum()))
    t_mul = time_call(lambda: np.multiply(a, b, out=out))
    t_dot = time_call(lambda: float(a @ b))

    q.put({
        "n": n,
        "dtype": dtype_name,
        "sum_ns": t_sum / n * 1e9,
        "mul_ns": t_mul / n * 1e9,
        "dot_ns": t_dot / n * 1e9,
    })


def run_grid():
    results = {}
    for n in SIZES:
        for dtype in DTYPES:
            q = mp.Queue()
            p = mp.Process(target=measure, args=(n, dtype, q))
            p.start()
            r = q.get()
            p.join()
            results[(n, dtype)] = r
    return results


def print_table(results, op_key, op_label):
    print(f"{op_label}  (ns per element, lower is faster)")
    header = (f"{'N':>12}  {'float16':>10}  {'float32':>10}  {'float64':>10}  "
              f"{'f16/f32':>9}  {'f64/f32':>9}")
    print(header)
    print("-" * len(header))
    for n in SIZES:
        r16 = results[(n, "float16")][op_key]
        r32 = results[(n, "float32")][op_key]
        r64 = results[(n, "float64")][op_key]
        ratio_16 = r16 / r32
        ratio_64 = r64 / r32
        print(f"{n:>12,}  "
              f"{r16:>9.3f}   "
              f"{r32:>9.3f}   "
              f"{r64:>9.3f}   "
              f"{ratio_16:>8.2f}×  "
              f"{ratio_64:>8.2f}×")
    print()


def main():
    print(__doc__.strip().splitlines()[0])
    print()
    print(f"Sizes (elements):   {SIZES}")
    print(f"Each cell:          best of {MEASURE_REPS} runs after {WARMUP_REPS} warmup")
    print()

    results = run_grid()

    print_table(results, "sum_ns", "sum(a)")
    print_table(results, "mul_ns", "a * b -> out (elementwise)")
    print_table(results, "dot_ns", "a @ b (dot product)")

    print("Read the columns:")
    print("  f16/f32 < 1: float16 is winning - halved bytes beat the conversion.")
    print("  f16/f32 > 1: conversion overhead dominates - the array fits well")
    print("               enough in cache that bandwidth savings don't pay.")
    print("  f64/f32 is the 'pure width' reference: no conversion penalty, just")
    print("               twice the bytes. If f64/f32 < 2 the op is partly")
    print("               compute-bound; if ~2 it is bandwidth-bound.")


if __name__ == "__main__":
    main()
