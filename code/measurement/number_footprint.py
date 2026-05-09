# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy"]
# ///
"""
§2 exhibit — width budget exists in numpy, not in stdlib.

Three measurements:

    1. sys.getsizeof on individual Python ints and floats. PyLong is not
       fixed-width; the object grows in 4-byte digits with magnitude. PyFloat
       is fixed at 24 bytes. Both carry an object header and a refcount that
       a typed numpy element does not.

    2. RSS footprint of N=1,000,000 numbers stored four ways: a Python list
       of small ints (in the interning range), a Python list of large ints
       (escaping the interning range), a Python list of floats, and the
       equivalent numpy arrays at every typed width.

    3. Sum-time at each width. The narrower numpy types are bandwidth-bound;
       the speedup over the Python list is two orders of magnitude.

Each list-of-N row is measured in a fresh subprocess so RSS deltas don't
bleed.

Run:
    uv run code/measurement/number_footprint.py
"""

import gc
import multiprocessing as mp
import resource
import sys
import time


N = 1_000_000


def rss_kb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def section_per_value_sizeof():
    print("Per-value size (bytes) — what does one PyLong / PyFloat actually cost?")
    print("-" * 64)
    samples = [
        ("int 0",                     0),
        ("int 1",                     1),
        ("int 256 (last interned)",   256),
        ("int 257",                   257),
        ("int 1_000",                 1_000),
        ("int 2**31",                 2**31),
        ("int 2**63",                 2**63),
        ("int 2**127",                2**127),
        ("float 0.0",                 0.0),
        ("float 3.14",                3.14),
        ("float 1e300",               1e300),
    ]
    for label, value in samples:
        print(f"  {label:<28} {sys.getsizeof(value):>4} bytes")
    print()


def measure_layout(layout_name, build_fn, sum_fn, q):
    gc.collect()
    rss_before = rss_kb()
    t0 = time.perf_counter()
    container = build_fn()
    t1 = time.perf_counter()
    rss_peak = rss_kb()
    s = sum_fn(container)
    t2 = time.perf_counter()
    # data_kb: the intrinsic data cost. For numpy: arr.nbytes. For a Python
    # list: RSS-delta is the closest honest number (the list and its objects
    # *are* what we allocated). RSS for numpy rows includes ~20 MB of one-off
    # numpy init overhead, which is signal pollution for the comparison.
    data_kb = getattr(container, "nbytes", rss_peak - rss_before) / 1024 \
        if hasattr(container, "nbytes") else (rss_peak - rss_before)
    q.put({
        "layout": layout_name,
        "build_s": t1 - t0,
        "rss_kb": rss_peak - rss_before,
        "data_kb": data_kb,
        "sum_s": t2 - t1,
        "sum": float(s),
    })


def build_list_int_small():
    return [i & 0xFF for i in range(N)]  # values 0..255 — interned


def build_list_int_large():
    return [1000 + i for i in range(N)]  # all non-interned


def build_list_float():
    return [float(i) + 0.5 for i in range(N)]


def sum_list(lst):
    return sum(lst)


def make_numpy_builder(dtype):
    def build():
        import numpy as np
        return np.arange(N, dtype=dtype)
    return build


def sum_numpy(arr):
    return int(arr.sum())


def sum_numpy_float(arr):
    return float(arr.sum())


LAYOUTS = [
    ("Python list of small ints (interned)", build_list_int_small,    sum_list),
    ("Python list of large ints (non-interned)", build_list_int_large, sum_list),
    ("Python list of floats",                build_list_float,          sum_list),
    ("numpy int8",                           make_numpy_builder("int8"),    sum_numpy),
    ("numpy int16",                          make_numpy_builder("int16"),   sum_numpy),
    ("numpy int32",                          make_numpy_builder("int32"),   sum_numpy),
    ("numpy int64",                          make_numpy_builder("int64"),   sum_numpy),
    ("numpy float32",                        make_numpy_builder("float32"), sum_numpy_float),
    ("numpy float64",                        make_numpy_builder("float64"), sum_numpy_float),
]


def worker(idx, q):
    name, build_fn, sum_fn = LAYOUTS[idx]
    measure_layout(name, build_fn, sum_fn, q)


def main():
    section_per_value_sizeof()

    print(f"N = {N:,} numbers, each layout in a fresh subprocess")
    print("data:   for numpy rows = arr.nbytes;   for list rows = RSS delta")
    print("RSS:    process RSS delta — numpy rows include ~20 MB of one-off init")
    header = f"{'layout':<42}  {'data (MB)':>10}  {'RSS (MB)':>9}  {'build (s)':>9}  {'sum (ms)':>9}"
    print(header)
    print("-" * len(header))

    results = []
    for idx in range(len(LAYOUTS)):
        q = mp.Queue()
        p = mp.Process(target=worker, args=(idx, q))
        p.start()
        r = q.get()
        p.join()
        results.append(r)
        print(f"{r['layout']:<42}  "
              f"{r['data_kb']/1024:>10.2f}  "
              f"{r['rss_kb']/1024:>9.1f}  "
              f"{r['build_s']:>9.3f}  "
              f"{r['sum_s']*1000:>9.2f}")

    # Ratios using the honest data column: list-of-large-ints vs each numpy width.
    print()
    base = results[1]  # large ints
    print(f"Data-size ratio: '{base['layout']}' / numpy width")
    for r in results[3:7]:  # int variants
        print(f"  vs {r['layout']:<40}  {base['data_kb']/r['data_kb']:>6.1f}× more bytes in the list")


if __name__ == "__main__":
    main()
