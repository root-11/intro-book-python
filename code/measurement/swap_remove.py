# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy"]
# ///
"""
§21 exhibit — four ways to remove 100,000 rows from a 1,000,000-row table.

Layouts:

    1. Python list, list.pop(i)
       Each pop shifts every subsequent element left by one. O(N) per remove.

    2. numpy array, np.delete(arr, i)
       np.delete returns a *new* array with the element removed. O(N) per
       remove plus an allocation; usually slower than list.pop for the
       same operation pattern, despite the typed bytes.

    3. numpy column with active counter, sequential swap_remove
       For each index (in reverse order so positions stay valid):
           arr[i] = arr[n_active - 1]; n_active -= 1
       O(1) per remove, but K trips through the Python interpreter.

    4. numpy bulk filter, arr[keep_mask]
       Build a boolean mask of survivors in one numpy call, then compress
       the column with one C-level pass. The natural pair to §22's
       buffered-cleanup pattern: collect K indices during the tick, apply
       the mask once at the tick boundary.

We remove K rows whose original positions are randomly distributed across
the middle of the table. Wall time per layout is the headline; we also
report ops/second so the orders-of-magnitude gaps are unambiguous.

Run:
    uv run code/measurement/swap_remove.py
"""

import gc
import time

import numpy as np


N = 1_000_000
K = 100_000  # rows to remove
SEED = 0xDEAD


def make_indices() -> np.ndarray:
    """K positions distributed across the middle of an N-row table."""
    rng = np.random.default_rng(SEED)
    return rng.integers(N // 4, 3 * N // 4, size=K).astype(np.int64)


def time_list_pop():
    lst = list(range(N))
    indices = make_indices()
    gc.collect()
    t0 = time.perf_counter()
    for raw in indices:
        # The table is shrinking, so clamp the index to the current length.
        i = int(raw) if raw < len(lst) else len(lst) - 1
        lst.pop(i)
    t1 = time.perf_counter()
    return (t1 - t0) * 1000.0, len(lst)


def time_np_delete():
    arr = np.arange(N, dtype=np.int64)
    indices = make_indices()
    gc.collect()
    t0 = time.perf_counter()
    for raw in indices:
        i = int(raw) if raw < arr.size else arr.size - 1
        arr = np.delete(arr, i)
    t1 = time.perf_counter()
    return (t1 - t0) * 1000.0, arr.size


def time_swap_remove_sequential():
    arr = np.arange(N, dtype=np.int64)
    n_active = N
    # Sort descending so each index is still valid as the table shrinks.
    indices = np.unique(make_indices())[::-1]
    gc.collect()
    t0 = time.perf_counter()
    for i in indices:
        arr[int(i)] = arr[n_active - 1]
        n_active -= 1
    t1 = time.perf_counter()
    return (t1 - t0) * 1000.0, n_active, indices.size


def time_swap_remove_bulk():
    arr = np.arange(N, dtype=np.int64)
    indices = np.unique(make_indices())
    gc.collect()
    t0 = time.perf_counter()
    keep_mask = np.ones(arr.size, dtype=bool)
    keep_mask[indices] = False
    arr = arr[keep_mask]
    t1 = time.perf_counter()
    return (t1 - t0) * 1000.0, arr.size, indices.size


def main():
    print(f"N = {N:,}, removing K = {K:,} mid-table rows.\n")
    header = f"{'layout':<46}  {'time (s)':>10}  {'remove rate (ops/s)':>22}"
    print(header)
    print("-" * len(header))

    rows = []
    list_ms, list_n = time_list_pop()
    rows.append(("Python list, list.pop(i)", list_ms, K))

    delete_ms, delete_n = time_np_delete()
    rows.append(("numpy, np.delete(arr, i)", delete_ms, K))

    seq_ms, seq_n, seq_k = time_swap_remove_sequential()
    rows.append(("numpy active counter, sequential swap_remove", seq_ms, seq_k))

    bulk_ms, bulk_n, bulk_k = time_swap_remove_bulk()
    rows.append(("numpy bulk filter, arr[keep_mask]", bulk_ms, bulk_k))

    for label, ms, k_done in rows:
        rate = k_done / (ms / 1000.0)
        print(f"{label:<46}  {ms/1000:>10.3f}  {rate:>22,.0f}")

    print()
    bulk_time = rows[-1][1]
    print("Speedup of bulk filter over the others:")
    for label, ms, _ in rows[:-1]:
        print(f"  {label:<46}  {ms / bulk_time:>8.0f}× slower")


if __name__ == "__main__":
    main()
