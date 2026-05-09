# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy"]
# ///
"""
§12 exhibit — three ways to store event timestamps, three footprints.

N events, each carrying a microsecond-resolution timestamp, stored three ways:

    1. list of datetime objects            (the Python-default reflex)
    2. numpy datetime64[us] column          (typed temporal column)
    3. numpy float64 column, seconds-from-epoch (the simlog discipline)

Three measurements per layout:

    - data size (bytes that hold the timestamps)
    - sort time (full ascending sort of N timestamps)
    - "events before T" time (count of timestamps < a threshold)

Each layout is built and measured in a fresh subprocess so RSS deltas
don't bleed across runs. Values cover one hour of microsecond timestamps
to keep the numbers comparable across encodings.

Run:
    uv run code/measurement/event_time_storage.py
"""

import gc
import multiprocessing as mp
import resource
import time
from datetime import datetime, timedelta


N = 1_000_000


def rss_kb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def time_call(fn):
    gc.collect()
    t0 = time.perf_counter()
    result = fn()
    t1 = time.perf_counter()
    return (t1 - t0) * 1000.0, result


def measure(layout_name, build_fn, sort_fn, count_fn, q):
    gc.collect()
    rss_before = rss_kb()

    t0 = time.perf_counter()
    container = build_fn()
    t1 = time.perf_counter()
    rss_peak = rss_kb()

    sort_ms, _ = time_call(lambda: sort_fn(container))
    count_ms, count_result = time_call(lambda: count_fn(container))

    data_kb = getattr(container, "nbytes", rss_peak - rss_before) / 1024 \
        if hasattr(container, "nbytes") else (rss_peak - rss_before)

    q.put({
        "layout": layout_name,
        "build_ms": (t1 - t0) * 1000.0,
        "rss_kb": rss_peak - rss_before,
        "data_kb": data_kb,
        "sort_ms": sort_ms,
        "count_ms": count_ms,
        "count_result": count_result,
    })


def build_datetime_list():
    base = datetime(2026, 5, 9, 12, 0, 0)
    return [base + timedelta(microseconds=i * 3600) for i in range(N)]


def sort_datetime_list(lst):
    lst.sort()


def count_datetime_list(lst):
    threshold = datetime(2026, 5, 9, 12, 30, 0)
    return sum(1 for t in lst if t < threshold)


def build_np_datetime64():
    import numpy as np
    base = np.datetime64("2026-05-09T12:00:00.000000", "us")
    return base + (np.arange(N, dtype=np.int64) * 3600).astype("timedelta64[us]")


def sort_np(arr):
    arr.sort()


def count_np_datetime64(arr):
    import numpy as np
    threshold = np.datetime64("2026-05-09T12:30:00.000000", "us")
    return int((arr < threshold).sum())


def build_np_float64():
    import numpy as np
    return np.arange(N, dtype=np.float64) * 3600e-6  # seconds-from-base


def count_np_float64(arr):
    threshold = 30 * 60.0  # 30 minutes from base, in seconds
    return int((arr < threshold).sum())


LAYOUTS = [
    ("list of datetime objects",       build_datetime_list,  sort_datetime_list, count_datetime_list),
    ("numpy datetime64[us]",           build_np_datetime64,  sort_np,            count_np_datetime64),
    ("numpy float64 (seconds-from-base)", build_np_float64,  sort_np,            count_np_float64),
]


def worker(idx, q):
    name, build_fn, sort_fn, count_fn = LAYOUTS[idx]
    measure(name, build_fn, sort_fn, count_fn, q)


def main():
    print(f"N = {N:,} events covering one simulated hour at microsecond resolution.")
    print(f"Each layout built and measured in a fresh subprocess.\n")

    header = (f"{'layout':<40}  {'data (MB)':>10}  {'build (ms)':>10}  "
              f"{'sort (ms)':>10}  {'count (ms)':>10}")
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
        print(f"{r['layout']:<40}  "
              f"{r['data_kb']/1024:>10.2f}  "
              f"{r['build_ms']:>10.1f}  "
              f"{r['sort_ms']:>10.2f}  "
              f"{r['count_ms']:>10.3f}")

    counts = [r["count_result"] for r in results]
    assert all(c == counts[0] for c in counts), \
        f"layouts disagree on count: {counts}"
    print(f"\ncount agrees across layouts: {counts[0]:,} events before threshold")

    base = results[0]
    print(f"\nRatios vs '{base['layout']}':")
    for r in results[1:]:
        print(f"  {r['layout']:<40}  "
              f"{base['data_kb']/r['data_kb']:>5.1f}× smaller   "
              f"{base['sort_ms']/r['sort_ms']:>5.1f}× faster sort   "
              f"{base['count_ms']/r['count_ms']:>6.1f}× faster count")


if __name__ == "__main__":
    main()
