#!/usr/bin/env python3
import math
import random
from time import perf_counter
from typing import Iterable, List, Dict

# Optional NumPy
try:
    import numpy as np  # type: ignore
except Exception:
    np = None

# --------------------- Configuration ---------------------
SEED = 12345
N_RANDOM = 1_000_000
N_LARGE_SMALL = 2_000_000
N_ALTERNATING = 1_000_000
N_TINY_INCREMENTS = 5_000_000
N_NANS = 1_000_000
REPEATS = 5
WARMUP = 1

USE_DECIMAL_REFERENCE = False
DECIMAL_MAX_N = 200_000

# --------------------- Algorithms ------------------------
def builtin_sum(xs: Iterable[float]) -> float:
    return sum(xs)

def math_fsum(xs: Iterable[float]) -> float:
    return math.fsum(xs)

def kahan_sum(xs: Iterable[float]) -> float:
    total = 0.0
    c = 0.0
    for x in xs:
        y = x - c
        t = total + y
        c = (t - total) - y
        total = t
    return total

def neumaier_sum(xs: Iterable[float]) -> float:
    total = 0.0
    c = 0.0
    for x in xs:
        t = total + x
        if abs(total) >= abs(x):
            c += (total - t) + x
        else:
            c += (x - t) + total
        total = t
    return total + c

def pairwise_sum(xs: List[float]) -> float:
    n = len(xs)
    if n == 0:
        return 0.0
    if n == 1:
        return float(xs[0])
    mid = n // 2
    return pairwise_sum(xs[:mid]) + pairwise_sum(xs[mid:])

def decimal_reference(xs: List[float]) -> float:
    from decimal import Decimal, getcontext
    getcontext().prec = 200
    s = sum((Decimal.from_float(x) for x in xs), Decimal(0))
    return float(s)

# --------------------- Datasets --------------------------
random.seed(SEED)

def make_random_balanced(n: int) -> List[float]:
    return [random.uniform(-1.0, 1.0) for _ in range(n)]

def make_large_small(n_small: int) -> List[float]:
    return [1e16] + [1.0] * n_small + [-1e16]

def make_alternating(n: int) -> List[float]:
    return [(-1.0 if (k % 2) else 1.0) / (k + 1) for k in range(n)]

def make_tiny_increments(n: int) -> List[float]:
    eps = 1e-10
    return [1e10] + [eps] * n

def make_with_nans(n: int) -> List[float]:
    xs = [random.uniform(-1.0, 1.0) for _ in range(n - 3)]
    xs += [math.nan, math.nan, 1.0]
    random.shuffle(xs)
    return xs

DATASETS: Dict[str, List[float]] = {
    "random_balanced": make_random_balanced(N_RANDOM),
    "large_plus_small": make_large_small(N_LARGE_SMALL),
    "alternating": make_alternating(N_ALTERNATING),
    "tiny_increments": make_tiny_increments(N_TINY_INCREMENTS),
    "with_nans": make_with_nans(N_NANS),
}

def shuffled(xs: List[float]) -> List[float]:
    ys = xs.copy()
    random.seed(SEED)
    random.shuffle(ys)
    return ys

# --------------------- Benchmark -------------------------
if __name__ == "__main__":
    print("Configuration:")
    print(f"  SEED={SEED}")
    print(f"  Datasets={list(DATASETS.keys())}")
    print(f"  REPEATS={REPEATS}, WARMUP={WARMUP}")
    print(f"  NumPy={'available' if np is not None else 'NOT available'}")
    print(f"  Decimal reference={'ENABLED' if USE_DECIMAL_REFERENCE else 'DISABLED'} (max N={DECIMAL_MAX_N})")
    print("-" * 80)

    for ds_name, xs0 in DATASETS.items():
        print(f"\n=== DATASET: {ds_name} (N={len(xs0)}) ===")
        for order_name in ("original", "reversed", "shuffled"):
            if order_name == "original":
                xs = xs0
            elif order_name == "reversed":
                xs = list(reversed(xs0))
            else:
                xs = shuffled(xs0)

            # Reference
            if USE_DECIMAL_REFERENCE and len(xs) <= DECIMAL_MAX_N:
                try:
                    ref = decimal_reference(xs)
                except Exception:
                    ref = math.fsum(xs)
            else:
                ref = math.fsum(xs)
            print(f"\n-- Order: {order_name} --")
            print(f"Reference: {ref:.17g}")

            # Functions to test (pure Python)
            funcs = [
                ("builtin_sum", lambda a: builtin_sum(a)),
                ("math_fsum",   lambda a: math_fsum(a)),
                ("kahan_sum",   lambda a: kahan_sum(a)),
                ("neumaier_sum",lambda a: neumaier_sum(a)),
                ("pairwise_sum",lambda a: pairwise_sum(list(a))),
            ]

            for name, fn in funcs:
                # Warmup
                for _ in range(WARMUP):
                    _ = fn(xs)
                # Best-of timing
                best = float("inf")
                for _ in range(REPEATS):
                    t0 = perf_counter()
                    _ = fn(xs)
                    t1 = perf_counter()
                    dt = t1 - t0
                    if dt < best:
                        best = dt
                res = fn(xs)
                err = 0.0 if (math.isnan(res) and math.isnan(ref)) else abs(res - ref)
                print(f"{name:18s} | time_s: {best:.9f} | result: {res:.17g} | abs_err: {err:.3g}")

            # NumPy (if available)
            if np is not None:
                # Probe float128 availability
                dtypes = [("float32", np.float32), ("float64", np.float64)]
                try:
                    np.dtype(np.float128)
                    dtypes.append(("float128", np.float128))
                except Exception:
                    pass

                for dt_name, dt in dtypes:
                    # Pre-create array
                    arr = np.array(xs, dtype=dt)
                    # Warmup
                    for _ in range(WARMUP):
                        _ = float(np.sum(arr))
                    # Best-of timing on existing array
                    best_arr = float("inf")
                    for _ in range(REPEATS):
                        t0 = perf_counter()
                        _ = float(np.sum(arr))
                        t1 = perf_counter()
                        dta = t1 - t0
                        if dta < best_arr:
                            best_arr = dta
                    r_arr = float(np.sum(arr))
                    err_arr = 0.0 if (math.isnan(r_arr) and math.isnan(ref)) else abs(r_arr - ref)
                    print(f"numpy.sum[{dt_name}]-arr  | time_s: {best_arr:.9f} | result: {r_arr:.17g} | abs_err: {err_arr:.3g}")

                    # Include conversion cost from list
                    # Warmup
                    for _ in range(WARMUP):
                        _ = float(np.sum(np.array(xs, dtype=dt)))
                    best_list = float("inf")
                    r_list = None
                    for _ in range(REPEATS):
                        t0 = perf_counter()
                        r_tmp = float(np.sum(np.array(xs, dtype=dt)))
                        t1 = perf_counter()
                        dtl = t1 - t0
                        if dtl < best_list:
                            best_list = dtl
                            r_list = r_tmp
                    if r_list is None:
                        r_list = float(np.sum(np.array(xs, dtype=dt)))
                    err_list = 0.0 if (math.isnan(r_list) and math.isnan(ref)) else abs(r_list - ref)
                    print(f"numpy.sum[{dt_name}]-list | time_s: {best_list:.9f} | result: {r_list:.17g} | abs_err: {err_list:.3g}")

                # nansum only meaningful when NaN appears in reference path (others propagate NaN)
                if any(math.isnan(v) for v in xs):
                    for dt_name, dt in dtypes:
                        arr = np.array(xs, dtype=dt)
                        # Array case
                        for _ in range(WARMUP):
                            _ = float(np.nansum(arr))
                        best_narr = float("inf")
                        for _ in range(REPEATS):
                            t0 = perf_counter()
                            _ = float(np.nansum(arr))
                            t1 = perf_counter()
                            dtn = t1 - t0
                            if dtn < best_narr:
                                best_narr = dtn
                        r_narr = float(np.nansum(arr))
                        print(f"numpy.nansum[{dt_name}]-arr  | time_s: {best_narr:.9f} | result: {r_narr:.17g}")

                        # List (conversion)
                        for _ in range(WARMUP):
                            _ = float(np.nansum(np.array(xs, dtype=dt)))
                        best_nlist = float("inf")
                        r_nlist = None
                        for _ in range(REPEATS):
                            t0 = perf_counter()
                            r_tmp = float(np.nansum(np.array(xs, dtype=dt)))
                            t1 = perf_counter()
                            dtn = t1 - t0
                            if dtn < best_nlist:
                                best_nlist = dtn
                                r_nlist = r_tmp
                        if r_nlist is None:
                            r_nlist = float(np.nansum(np.array(xs, dtype=dt)))
                        print(f"numpy.nansum[{dt_name}]-list | time_s: {best_nlist:.9f} | result: {r_nlist:.17g}")

# Results
# (.venv) ) bjorn@LOCLAP932:~/github/root-11/journal$ /home/bjorn/github/root-11/journal/.venv/bin/python "/home/bjorn/github/root-11/journal/explorations/python benchmarks/sums.py"
# Configuration:
#   SEED=12345
#   Datasets=['random_balanced', 'large_plus_small', 'alternating', 'tiny_increments', 'with_nans']
#   REPEATS=5, WARMUP=1
#   NumPy=available
#   Decimal reference=DISABLED (max N=200000)
# --------------------------------------------------------------------------------

# === DATASET: random_balanced (N=1000000) ===

# -- Order: original --
# Reference: 129.52834728660494
# builtin_sum        | time_s: 0.003818549 | result: 129.52834728660494 | abs_err: 0
# math_fsum          | time_s: 0.005391490 | result: 129.52834728660494 | abs_err: 0
# kahan_sum          | time_s: 0.028997664 | result: 129.52834728660494 | abs_err: 0
# neumaier_sum       | time_s: 0.039794203 | result: 129.52834728660494 | abs_err: 0
# pairwise_sum       | time_s: 0.158596819 | result: 129.52834728660494 | abs_err: 0
# numpy.sum[float32]-arr  | time_s: 0.000145825 | result: 129.52827453613281 | abs_err: 7.28e-05
# numpy.sum[float32]-list | time_s: 0.019252580 | result: 129.52827453613281 | abs_err: 7.28e-05
# numpy.sum[float64]-arr  | time_s: 0.000163457 | result: 129.52834728660477 | abs_err: 1.71e-13
# numpy.sum[float64]-list | time_s: 0.019984279 | result: 129.52834728660477 | abs_err: 1.71e-13
# numpy.sum[float128]-arr  | time_s: 0.000717598 | result: 129.52834728660494 | abs_err: 0
# numpy.sum[float128]-list | time_s: 0.109870245 | result: 129.52834728660494 | abs_err: 0

# -- Order: reversed --
# Reference: 129.52834728660494
# builtin_sum        | time_s: 0.003649557 | result: 129.52834728660494 | abs_err: 0
# math_fsum          | time_s: 0.005701343 | result: 129.52834728660494 | abs_err: 0
# kahan_sum          | time_s: 0.029733055 | result: 129.52834728660494 | abs_err: 0
# neumaier_sum       | time_s: 0.039679071 | result: 129.52834728660494 | abs_err: 0
# pairwise_sum       | time_s: 0.160012228 | result: 129.528347286605 | abs_err: 5.68e-14
# numpy.sum[float32]-arr  | time_s: 0.000148542 | result: 129.52835083007812 | abs_err: 3.54e-06
# numpy.sum[float32]-list | time_s: 0.019734946 | result: 129.52835083007812 | abs_err: 3.54e-06
# numpy.sum[float64]-arr  | time_s: 0.000167244 | result: 129.52834728660505 | abs_err: 1.14e-13
# numpy.sum[float64]-list | time_s: 0.019933643 | result: 129.52834728660505 | abs_err: 1.14e-13
# numpy.sum[float128]-arr  | time_s: 0.000755652 | result: 129.52834728660494 | abs_err: 0
# numpy.sum[float128]-list | time_s: 0.112052446 | result: 129.52834728660494 | abs_err: 0

# -- Order: shuffled --
# Reference: 129.52834728660494
# builtin_sum        | time_s: 0.015089229 | result: 129.52834728660494 | abs_err: 0
# math_fsum          | time_s: 0.027619583 | result: 129.52834728660494 | abs_err: 0
# kahan_sum          | time_s: 0.099807856 | result: 129.52834728660494 | abs_err: 0
# neumaier_sum       | time_s: 0.130304344 | result: 129.52834728660494 | abs_err: 0
# pairwise_sum       | time_s: 0.255176892 | result: 129.52834728660486 | abs_err: 8.53e-14
# numpy.sum[float32]-arr  | time_s: 0.000153914 | result: 129.52841186523438 | abs_err: 6.46e-05
# numpy.sum[float32]-list | time_s: 0.059729976 | result: 129.52841186523438 | abs_err: 6.46e-05
# numpy.sum[float64]-arr  | time_s: 0.000167545 | result: 129.52834728660503 | abs_err: 8.53e-14
# numpy.sum[float64]-list | time_s: 0.059091933 | result: 129.52834728660503 | abs_err: 8.53e-14
# numpy.sum[float128]-arr  | time_s: 0.000739076 | result: 129.52834728660494 | abs_err: 0
# numpy.sum[float128]-list | time_s: 0.187039535 | result: 129.52834728660494 | abs_err: 0

# === DATASET: large_plus_small (N=2000002) ===

# -- Order: original --
# Reference: 2000000
# builtin_sum        | time_s: 0.011627461 | result: 2000000 | abs_err: 0
# math_fsum          | time_s: 0.011877968 | result: 2000000 | abs_err: 0
# kahan_sum          | time_s: 0.059549022 | result: 2000000 | abs_err: 0
# neumaier_sum       | time_s: 0.080016870 | result: 2000000 | abs_err: 0
# pairwise_sum       | time_s: 0.288560764 | result: 1999998 | abs_err: 2
# numpy.sum[float32]-arr  | time_s: 0.000303608 | result: 0 | abs_err: 2e+06
# numpy.sum[float32]-list | time_s: 0.036057356 | result: 0 | abs_err: 2e+06
# numpy.sum[float64]-arr  | time_s: 0.000311256 | result: 1999986 | abs_err: 14
# numpy.sum[float64]-list | time_s: 0.035757170 | result: 1999986 | abs_err: 14
# numpy.sum[float128]-arr  | time_s: 0.001924512 | result: 2000000 | abs_err: 0
# numpy.sum[float128]-list | time_s: 0.226034395 | result: 2000000 | abs_err: 0

# -- Order: reversed --
# Reference: 2000000
# builtin_sum        | time_s: 0.011931852 | result: 2000000 | abs_err: 0
# math_fsum          | time_s: 0.012219112 | result: 2000000 | abs_err: 0
# kahan_sum          | time_s: 0.062017343 | result: 2000000 | abs_err: 0
# neumaier_sum       | time_s: 0.082673034 | result: 2000000 | abs_err: 0
# pairwise_sum       | time_s: 0.290055757 | result: 1999998 | abs_err: 2
# numpy.sum[float32]-arr  | time_s: 0.000290289 | result: 0 | abs_err: 2e+06
# numpy.sum[float32]-list | time_s: 0.035983111 | result: 0 | abs_err: 2e+06
# numpy.sum[float64]-arr  | time_s: 0.000479886 | result: 1999986 | abs_err: 14
# numpy.sum[float64]-list | time_s: 0.036240653 | result: 1999986 | abs_err: 14
# numpy.sum[float128]-arr  | time_s: 0.001898072 | result: 2000000 | abs_err: 0
# numpy.sum[float128]-list | time_s: 0.223059058 | result: 2000000 | abs_err: 0

# -- Order: shuffled --
# Reference: 2000000
# builtin_sum        | time_s: 0.011904314 | result: 2000000 | abs_err: 0
# math_fsum          | time_s: 0.012047156 | result: 2000000 | abs_err: 0
# kahan_sum          | time_s: 0.059862487 | result: 2000000 | abs_err: 0
# neumaier_sum       | time_s: 0.081679114 | result: 2000000 | abs_err: 0
# pairwise_sum       | time_s: 0.290059988 | result: 2000000 | abs_err: 0
# numpy.sum[float32]-arr  | time_s: 0.000290779 | result: 0 | abs_err: 2e+06
# numpy.sum[float32]-list | time_s: 0.036089178 | result: 0 | abs_err: 2e+06
# numpy.sum[float64]-arr  | time_s: 0.000338255 | result: 1999992 | abs_err: 8
# numpy.sum[float64]-list | time_s: 0.036152954 | result: 1999992 | abs_err: 8
# numpy.sum[float128]-arr  | time_s: 0.001869574 | result: 2000000 | abs_err: 0
# numpy.sum[float128]-list | time_s: 0.223005756 | result: 2000000 | abs_err: 0

# === DATASET: alternating (N=1000000) ===

# -- Order: original --
# Reference: 0.69314668056019535
# builtin_sum        | time_s: 0.003642844 | result: 0.69314668056019535 | abs_err: 0
# math_fsum          | time_s: 0.005278207 | result: 0.69314668056019535 | abs_err: 0
# kahan_sum          | time_s: 0.029237090 | result: 0.69314668056019535 | abs_err: 0
# neumaier_sum       | time_s: 0.039993878 | result: 0.69314668056019535 | abs_err: 0
# pairwise_sum       | time_s: 0.161022824 | result: 0.69314668056019535 | abs_err: 0
# numpy.sum[float32]-arr  | time_s: 0.000147758 | result: 0.6931464672088623 | abs_err: 2.13e-07
# numpy.sum[float32]-list | time_s: 0.019588573 | result: 0.6931464672088623 | abs_err: 2.13e-07
# numpy.sum[float64]-arr  | time_s: 0.000157244 | result: 0.69314668056019535 | abs_err: 0
# numpy.sum[float64]-list | time_s: 0.019443792 | result: 0.69314668056019535 | abs_err: 0
# numpy.sum[float128]-arr  | time_s: 0.000813364 | result: 0.69314668056019535 | abs_err: 0
# numpy.sum[float128]-list | time_s: 0.112853688 | result: 0.69314668056019535 | abs_err: 0

# -- Order: reversed --
# Reference: 0.69314668056019535
# builtin_sum        | time_s: 0.003873221 | result: 0.69314668056019535 | abs_err: 0
# math_fsum          | time_s: 0.004378813 | result: 0.69314668056019535 | abs_err: 0
# kahan_sum          | time_s: 0.029581517 | result: 0.69314668056019535 | abs_err: 0
# neumaier_sum       | time_s: 0.040873199 | result: 0.69314668056019535 | abs_err: 0
# pairwise_sum       | time_s: 0.161387104 | result: 0.69314668056019513 | abs_err: 2.22e-16
# numpy.sum[float32]-arr  | time_s: 0.000148592 | result: 0.69314664602279663 | abs_err: 3.45e-08
# numpy.sum[float32]-list | time_s: 0.019342833 | result: 0.69314664602279663 | abs_err: 3.45e-08
# numpy.sum[float64]-arr  | time_s: 0.000175016 | result: 0.69314668056019546 | abs_err: 1.11e-16
# numpy.sum[float64]-list | time_s: 0.019839650 | result: 0.69314668056019546 | abs_err: 1.11e-16
# numpy.sum[float128]-arr  | time_s: 0.000728411 | result: 0.69314668056019535 | abs_err: 0
# numpy.sum[float128]-list | time_s: 0.112409463 | result: 0.69314668056019535 | abs_err: 0

# -- Order: shuffled --
# Reference: 0.69314668056019535
# builtin_sum        | time_s: 0.014484655 | result: 0.69314668056019535 | abs_err: 0
# math_fsum          | time_s: 0.029731747 | result: 0.69314668056019535 | abs_err: 0
# kahan_sum          | time_s: 0.077499398 | result: 0.69314668056019524 | abs_err: 1.11e-16
# neumaier_sum       | time_s: 0.109378215 | result: 0.69314668056019535 | abs_err: 0
# pairwise_sum       | time_s: 0.253783807 | result: 0.69314668056019568 | abs_err: 3.33e-16
# numpy.sum[float32]-arr  | time_s: 0.000150864 | result: 0.69314658641815186 | abs_err: 9.41e-08
# numpy.sum[float32]-list | time_s: 0.064172025 | result: 0.69314658641815186 | abs_err: 9.41e-08
# numpy.sum[float64]-arr  | time_s: 0.000171173 | result: 0.69314668056019546 | abs_err: 1.11e-16
# numpy.sum[float64]-list | time_s: 0.061417899 | result: 0.69314668056019546 | abs_err: 1.11e-16
# numpy.sum[float128]-arr  | time_s: 0.000700477 | result: 0.69314668056019535 | abs_err: 0
# numpy.sum[float128]-list | time_s: 0.192982922 | result: 0.69314668056019535 | abs_err: 0

# === DATASET: tiny_increments (N=5000001) ===

# -- Order: original --
# Reference: 10000000000.0005
# builtin_sum        | time_s: 0.029475617 | result: 10000000000.0005 | abs_err: 0
# math_fsum          | time_s: 0.031462108 | result: 10000000000.0005 | abs_err: 0
# kahan_sum          | time_s: 0.149747983 | result: 10000000000.0005 | abs_err: 0
# neumaier_sum       | time_s: 0.201032950 | result: 10000000000.0005 | abs_err: 0
# pairwise_sum       | time_s: 0.730980440 | result: 10000000000.0005 | abs_err: 0
# numpy.sum[float32]-arr  | time_s: 0.000801792 | result: 10000000000 | abs_err: 0.0005
# numpy.sum[float32]-list | time_s: 0.088509794 | result: 10000000000 | abs_err: 0.0005
# numpy.sum[float64]-arr  | time_s: 0.001706635 | result: 10000000000.0005 | abs_err: 0
# numpy.sum[float64]-list | time_s: 0.089804139 | result: 10000000000.0005 | abs_err: 0
# numpy.sum[float128]-arr  | time_s: 0.004963796 | result: 10000000000.0005 | abs_err: 0
# numpy.sum[float128]-list | time_s: 0.558820362 | result: 10000000000.0005 | abs_err: 0

# -- Order: reversed --
# Reference: 10000000000.0005
# builtin_sum        | time_s: 0.028565361 | result: 10000000000.0005 | abs_err: 0
# math_fsum          | time_s: 0.031274669 | result: 10000000000.0005 | abs_err: 0
# kahan_sum          | time_s: 0.147788228 | result: 10000000000.0005 | abs_err: 0
# neumaier_sum       | time_s: 0.200338568 | result: 10000000000.0005 | abs_err: 0
# pairwise_sum       | time_s: 0.734501911 | result: 10000000000.0005 | abs_err: 0
# numpy.sum[float32]-arr  | time_s: 0.000875906 | result: 10000000000 | abs_err: 0.0005
# numpy.sum[float32]-list | time_s: 0.088946843 | result: 10000000000 | abs_err: 0.0005
# numpy.sum[float64]-arr  | time_s: 0.001699797 | result: 10000000000.0005 | abs_err: 0
# numpy.sum[float64]-list | time_s: 0.089654100 | result: 10000000000.0005 | abs_err: 0
# numpy.sum[float128]-arr  | time_s: 0.004840314 | result: 10000000000.0005 | abs_err: 0
# numpy.sum[float128]-list | time_s: 0.557694674 | result: 10000000000.0005 | abs_err: 0

# -- Order: shuffled --
# Reference: 10000000000.0005
# builtin_sum        | time_s: 0.029787198 | result: 10000000000.0005 | abs_err: 0
# math_fsum          | time_s: 0.031319673 | result: 10000000000.0005 | abs_err: 0
# kahan_sum          | time_s: 0.148042741 | result: 10000000000.000502 | abs_err: 1.91e-06
# neumaier_sum       | time_s: 0.200243251 | result: 10000000000.0005 | abs_err: 0
# pairwise_sum       | time_s: 0.726077213 | result: 10000000000.0005 | abs_err: 0
# numpy.sum[float32]-arr  | time_s: 0.001084308 | result: 10000000000 | abs_err: 0.0005
# numpy.sum[float32]-list | time_s: 0.089859798 | result: 10000000000 | abs_err: 0.0005
# numpy.sum[float64]-arr  | time_s: 0.001859323 | result: 10000000000.0005 | abs_err: 0
# numpy.sum[float64]-list | time_s: 0.090408682 | result: 10000000000.0005 | abs_err: 0
# numpy.sum[float128]-arr  | time_s: 0.005083250 | result: 10000000000.0005 | abs_err: 0
# numpy.sum[float128]-list | time_s: 0.561989919 | result: 10000000000.0005 | abs_err: 0

# === DATASET: with_nans (N=1000000) ===

# -- Order: original --
# Reference: nan
# builtin_sum        | time_s: 0.016413410 | result: nan | abs_err: 0
# math_fsum          | time_s: 0.028913007 | result: nan | abs_err: 0
# kahan_sum          | time_s: 0.074341145 | result: nan | abs_err: 0
# neumaier_sum       | time_s: 0.103997342 | result: nan | abs_err: 0
# pairwise_sum       | time_s: 0.255269212 | result: nan | abs_err: 0
# numpy.sum[float32]-arr  | time_s: 0.000154944 | result: nan | abs_err: 0
# numpy.sum[float32]-list | time_s: 0.058561329 | result: nan | abs_err: 0
# numpy.sum[float64]-arr  | time_s: 0.000162885 | result: nan | abs_err: 0
# numpy.sum[float64]-list | time_s: 0.058812727 | result: nan | abs_err: 0
# numpy.sum[float128]-arr  | time_s: 0.000710558 | result: nan | abs_err: 0
# numpy.sum[float128]-list | time_s: 0.183420251 | result: nan | abs_err: 0
# numpy.nansum[float32]-arr  | time_s: 0.000658051 | result: 1276.467041015625
# numpy.nansum[float32]-list | time_s: 0.058184110 | result: 1276.467041015625
# numpy.nansum[float64]-arr  | time_s: 0.000996725 | result: 1276.4670360518999
# numpy.nansum[float64]-list | time_s: 0.059255765 | result: 1276.4670360518999
# numpy.nansum[float128]-arr  | time_s: 0.002297317 | result: 1276.4670360518996
# numpy.nansum[float128]-list | time_s: 0.185542817 | result: 1276.4670360518996

# -- Order: reversed --
# Reference: nan
# builtin_sum        | time_s: 0.016265892 | result: nan | abs_err: 0
# math_fsum          | time_s: 0.024753072 | result: nan | abs_err: 0
# kahan_sum          | time_s: 0.077722030 | result: nan | abs_err: 0
# neumaier_sum       | time_s: 0.100844801 | result: nan | abs_err: 0
# pairwise_sum       | time_s: 0.265180324 | result: nan | abs_err: 0
# numpy.sum[float32]-arr  | time_s: 0.000159391 | result: nan | abs_err: 0
# numpy.sum[float32]-list | time_s: 0.066387340 | result: nan | abs_err: 0
# numpy.sum[float64]-arr  | time_s: 0.000166203 | result: nan | abs_err: 0
# numpy.sum[float64]-list | time_s: 0.059378681 | result: nan | abs_err: 0
# numpy.sum[float128]-arr  | time_s: 0.000686921 | result: nan | abs_err: 0
# numpy.sum[float128]-list | time_s: 0.185905391 | result: nan | abs_err: 0
# numpy.nansum[float32]-arr  | time_s: 0.000640172 | result: 1276.467041015625
# numpy.nansum[float32]-list | time_s: 0.058415123 | result: 1276.467041015625
# numpy.nansum[float64]-arr  | time_s: 0.000934935 | result: 1276.4670360518996
# numpy.nansum[float64]-list | time_s: 0.059603259 | result: 1276.4670360518996
# numpy.nansum[float128]-arr  | time_s: 0.002516885 | result: 1276.4670360518996
# numpy.nansum[float128]-list | time_s: 0.181652389 | result: 1276.4670360518996

# -- Order: shuffled --
# Reference: nan
# builtin_sum        | time_s: 0.015344857 | result: nan | abs_err: 0
# math_fsum          | time_s: 0.024779549 | result: nan | abs_err: 0
# kahan_sum          | time_s: 0.078953058 | result: nan | abs_err: 0
# neumaier_sum       | time_s: 0.103980112 | result: nan | abs_err: 0
# pairwise_sum       | time_s: 0.254773215 | result: nan | abs_err: 0
# numpy.sum[float32]-arr  | time_s: 0.000154859 | result: nan | abs_err: 0
# numpy.sum[float32]-list | time_s: 0.059646981 | result: nan | abs_err: 0
# numpy.sum[float64]-arr  | time_s: 0.000166953 | result: nan | abs_err: 0
# numpy.sum[float64]-list | time_s: 0.060051396 | result: nan | abs_err: 0
# numpy.sum[float128]-arr  | time_s: 0.000697315 | result: nan | abs_err: 0
# numpy.sum[float128]-list | time_s: 0.183013053 | result: nan | abs_err: 0
# numpy.nansum[float32]-arr  | time_s: 0.000648943 | result: 1276.467041015625
# numpy.nansum[float32]-list | time_s: 0.057998494 | result: 1276.467041015625
# numpy.nansum[float64]-arr  | time_s: 0.000921311 | result: 1276.4670360518994
# numpy.nansum[float64]-list | time_s: 0.059675253 | result: 1276.4670360518994
# numpy.nansum[float128]-arr  | time_s: 0.002299529 | result: 1276.4670360518996
# numpy.nansum[float128]-list | time_s: 0.183475043 | result: 1276.4670360518996