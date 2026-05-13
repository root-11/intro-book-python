# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy"]
# ///
"""
§3 exhibit — same payload, five layouts, five footprints.

N rows of K integers each, laid out five ways:

    1. list of tuples              (AoS, the canonical Python-default)
    2. list of lists               (AoS, the even-worse case)
    3. tuple of lists              (SoA, stdlib only)
    4. tuple of array.array        (SoA, stdlib typed — middle ground)
    5. tuple of numpy int64 arrays (SoA, the disciplined endpoint)

Each layout is built in a fresh subprocess so RSS readings do not bleed
across runs. Three numbers per layout: construction wall time, peak RSS
delta over the baseline, and the wall time of summing column 0.

Values are drawn from [1000, 1000+N) to escape CPython's small-int
interning ([-5, 256] are cached singletons). If we used 0..N the AoS
layouts would share PyLong objects across rows and the comparison would
flatter the wrong side.

The two SoA-stdlib rows (3 and 4) separate two effects that the four-row
version conflated: 'use SoA instead of AoS' (3 beats 1/2) and 'use typed
storage instead of PyLong boxes' (4 beats 3). The fifth row adds 'use
vectorised C primitives instead of Python-level iteration' (5 beats 4).
Three independent wins, three rows apart.

Run:
    uv run code/measurement/aos_vs_soa_footprint.py
"""

import gc
import multiprocessing as mp
import resource
import time

N = 1_000_000
K = 10
BASE = 1000  # past the small-int cache


def rss_kb():
    # ru_maxrss is high-water-mark RSS, in KB on Linux.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def measure(layout_name, build_fn, sum_fn):
    gc.collect()
    rss_before = rss_kb()
    t0 = time.perf_counter()
    container = build_fn()
    t1 = time.perf_counter()
    rss_peak = rss_kb()
    s = sum_fn(container)
    t2 = time.perf_counter()
    return {
        "layout": layout_name,
        "build_s": t1 - t0,
        "rss_kb": rss_peak - rss_before,
        "sum_s": t2 - t1,
        "checksum": int(s),
    }


def build_list_of_tuples():
    return [tuple(BASE + i + k for k in range(K)) for i in range(N)]


def sum_col0_list_of_tuples(rows):
    return sum(row[0] for row in rows)


def build_list_of_lists():
    return [[BASE + i + k for k in range(K)] for i in range(N)]


def sum_col0_list_of_lists(rows):
    return sum(row[0] for row in rows)


def build_tuple_of_lists():
    return tuple([BASE + i + k for i in range(N)] for k in range(K))


def sum_col0_tuple_of_lists(cols):
    return sum(cols[0])


def build_tuple_of_arrays():
    import array
    return tuple(array.array('q', (BASE + i + k for i in range(N))) for k in range(K))


def sum_col0_tuple_of_arrays(cols):
    return sum(cols[0])


def build_numpy_columns():
    import numpy as np
    return tuple(np.arange(BASE + k, BASE + k + N, dtype=np.int64) for k in range(K))


def sum_col0_numpy(cols):
    return int(cols[0].sum())


LAYOUTS = [
    ("1. list of tuples              (AoS)",        build_list_of_tuples,   sum_col0_list_of_tuples),
    ("2. list of lists               (AoS)",        build_list_of_lists,    sum_col0_list_of_lists),
    ("3. tuple of lists              (SoA stdlib)", build_tuple_of_lists,   sum_col0_tuple_of_lists),
    ("4. tuple of array.array        (SoA typed)",  build_tuple_of_arrays,  sum_col0_tuple_of_arrays),
    ("5. tuple of numpy int64 arrays (SoA numpy)",  build_numpy_columns,    sum_col0_numpy),
]


def worker(idx, q):
    name, build_fn, sum_fn = LAYOUTS[idx]
    q.put(measure(name, build_fn, sum_fn))


def main():
    print(f"N = {N:,} rows, K = {K} ints per row")
    print(f"values in [{BASE}, {BASE + N + K}) — past CPython small-int interning")
    print(f"each layout measured in a fresh subprocess\n")

    header = f"{'layout':<46}  {'build (s)':>9}  {'RSS (MB)':>9}  {'sum c0 (s)':>10}"
    print(header)
    print("-" * len(header))

    results = []
    for idx in range(len(LAYOUTS)):
        q = mp.Queue()
        p = mp.Process(target=worker, args=(idx, q))
        p.start()
        result = q.get()
        p.join()
        results.append(result)
        print(f"{result['layout']:<46}  "
              f"{result['build_s']:>9.3f}  "
              f"{result['rss_kb']/1024:>9.1f}  "
              f"{result['sum_s']:>10.4f}")

    print()
    checksums = [r["checksum"] for r in results]
    assert all(c == checksums[0] for c in checksums), \
        f"checksum mismatch — layouts disagree on the data: {checksums}"
    print(f"checksum (matches across all layouts): {checksums[0]:,}")

    print("\nRatios vs layout 5 (numpy SoA):")
    base = results[-1]
    for r in results[:-1]:
        print(f"  {r['layout']:<46}  "
              f"{r['rss_kb']/base['rss_kb']:>4.1f}× memory   "
              f"{r['build_s']/base['build_s']:>5.1f}× build   "
              f"{r['sum_s']/base['sum_s']:>5.1f}× sum-c0")


if __name__ == "__main__":
    main()
