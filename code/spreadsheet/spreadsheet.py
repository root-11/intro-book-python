# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy"]
# ///
"""
§54 exhibit - a spreadsheet is a dependency graph, in Python+numpy.

Counterpart to the Rust edition's code/spreadsheet crate (main.rs + scale.rs).
A cell holds a formula over other cells; the cells form a dependency graph;
recalculation is a topological sort of that graph. Dirty propagation - which the
scenegraph did down a tree - now runs through a graph, and the tree's
contiguous-subtree shortcut is gone: "what went stale" is a cone you compute.

The claims, and how Python shades them:

  C1 fill-down crossover   recompute the cone (only the edit's transitive
                           dependents) beats a full recompute when the fill-down
                           is small; full takes over once it covers most of the
                           sheet. A uniform fill-down vectorises per column, so
                           both sides are numpy here.
  C2 sum is not incremental a one-cell edit under =SUM(col) has a tiny cone but
                           re-reads the WHOLE column - the cost is the column, not
                           the cell. Touching one input does not make a sum cheap.
  C3 early cutoff          edit a below-max cell under =MAX(col) feeding a
                           dashboard: if the recomputed value is unchanged, STOP -
                           do not recompute the dashboard. Validation is cheaper
                           than recomputation. In Python the saved work is per-cell
                           dashboard recompute, so the win is large.
  C4 the program goes flat  one formula-object per cell at 1e9 cells is hundreds
                           of GB (Python objects are heavier than Rust's, so the
                           wall is closer); one template per column is kilobytes.
  C5 pegged memory         a tiled streaming sum holds peak memory at one tile,
                           whatever the column height - OOM becomes structurally
                           impossible. The dirty-column patch reads only the dirty
                           columns' bytes, not the whole file.

The >RAM disk-seconds pivot (Rust: ~16 s whole file vs ~0.1 s dirty columns at
36 GB > 30 GB RAM) needs a file larger than this box's RAM to measure honestly
(page cache would otherwise lie); it is deferred like the cross-machine columns.
What is measured here is the bytes-moved ratio and the pegged peak memory, which
are layout facts, not timing.

Run:  uv run code/spreadsheet/spreadsheet.py
"""

import sys
import time
import tracemalloc

import numpy as np

rng = np.random.default_rng(0x54)


def median(fn, k=3):
    return sorted(fn() for _ in range(k))[k // 2]


def time_call(fn):
    t0 = time.perf_counter()
    fn()
    return (time.perf_counter() - t0) * 1e3  # ms


# ---------------------------------------------------------------------------
# C1: the fill-down crossover. A grid where column c reads column c-1 row by row,
# so editing input rows dirties exactly those rows across all columns - a cone
# whose size is len(rows) * W. A uniform fill-down recomputes per column,
# vectorised over the dirty rows.
# ---------------------------------------------------------------------------

def c1_fill_down_crossover():
    H, W = 200_000, 50
    value = np.empty((H, W))
    value[:, 0] = rng.random(H)
    coef = 0.5 + 0.5 * rng.random(W)
    bias = rng.random(W) * 0.01

    def recompute(rows):
        for c in range(1, W):
            value[rows, c] = value[rows, c - 1] * coef[c] + bias[c]

    full_rows = np.arange(H)
    full_ms = median(lambda: time_call(lambda: recompute(full_rows)))
    print("== C1: fill-down crossover (200k x 50 grid) ==")
    print(f"  full recompute (all rows): {full_ms:.2f} ms\n")
    print(f"{'fill-down rows':>15} {'cone (ms)':>11} {'vs full':>9}")
    for k in (10, 1_000, 20_000, 100_000, 200_000):
        rows = np.arange(k)
        ms = median(lambda: time_call(lambda: recompute(rows)))
        print(f"{k:>15} {ms:>11.3f} {full_ms / ms:>8.2f}x")


# ---------------------------------------------------------------------------
# C2: the sum that is not incremental. =SUM over a tall column; edit one cell.
# The cone is {cell, SUM, SUM's dependents} - tiny in count - but recomputing the
# SUM re-reads the whole column.
# ---------------------------------------------------------------------------

def c2_sum_not_incremental():
    print("\n== C2: a one-cell edit under =SUM(1e6 column) ==")
    H = 1_000_000
    col = rng.random(H)
    # edit one cell, then recompute the SUM (its formula re-reads the whole range)
    def recompute_sum_after_one_edit():
        col[12345] = 0.5            # the edit: one cell
        return col.sum()            # the SUM: reads all H
    ms_1 = median(lambda: time_call(recompute_sum_after_one_edit))
    # editing 100k cells costs the SAME, because the sum re-reads the column either way
    def recompute_sum_after_many_edits():
        col[:100_000] = 0.5
        return col.sum()
    ms_many = median(lambda: time_call(recompute_sum_after_many_edits))
    print(f"  recompute SUM after 1 edit     : {ms_1:.3f} ms (reads all {H:,})")
    print(f"  recompute SUM after 100k edits : {ms_many:.3f} ms (same work)")
    print("  the cone is 1 cell; the work is the whole column - a sum is not incremental.")


# ---------------------------------------------------------------------------
# C3: early cutoff. =MAX over a column feeds D dashboard cells, each its own
# formula (recomputed per cell, as a real engine must for distinct formulas).
# Edit a below-max cell to another below-max value: MAX is unchanged, so the
# dashboard need not be touched - if you check.
# ---------------------------------------------------------------------------

def c3_early_cutoff():
    print("\n== C3: edit absorbed by =MAX(col), dashboard of 100k cells ==")
    H, D = 1000, 100_000
    col = rng.random(H) * 0.9          # all below 1.0
    col[500] = 1.0                     # the maximum
    dashboard_formula = rng.random(D)  # each dashboard cell: m * coef[i]

    old_max = col.max()

    def no_cutoff():
        col[123] = 0.4                 # edit a below-max cell, still below max
        m = col.max()                  # recompute MAX (rescan)
        # recompute every dashboard cell, one at a time (distinct formulas)
        out = 0.0
        for i in range(D):
            out += m * dashboard_formula[i]
        return out

    def with_cutoff():
        col[123] = 0.4
        m = col.max()                  # recompute MAX (rescan)
        if m == old_max:               # validation: did it actually change?
            return None                # absorbed - touch nothing downstream
        out = 0.0
        for i in range(D):
            out += m * dashboard_formula[i]
        return out

    nm = median(lambda: time_call(no_cutoff))
    wm = median(lambda: time_call(with_cutoff))
    print(f"  no cutoff  (recompute MAX + {D:,} dashboard): {nm:.3f} ms")
    print(f"  with cutoff (recompute MAX, value unchanged, stop): {wm:.4f} ms")
    print(f"  speedup: {nm / wm:.0f}x   - validation is cheaper than recomputation")


# ---------------------------------------------------------------------------
# C4: the program goes flat. One formula-object per cell vs one template per column.
# ---------------------------------------------------------------------------

def c4_templates():
    print("\n== C4: the 'program' for 1e9 cells ==")
    # a representative per-cell formula object: (op, left_ref, right_ref)
    formula = ("mul", 1234567, 2345678)
    per_cell = sys.getsizeof(formula) + sum(sys.getsizeof(x) for x in formula)
    cells = 1_000_000_000
    object_bytes = per_cell * cells
    # template-per-column: a few hundred columns, one formula object each
    template_bytes = 300 * per_cell
    print(f"  one formula object: ~{per_cell} bytes")
    print(f"  1e9 objects (one per cell): ~{object_bytes / 1e9:.0f} GB  (cannot allocate)")
    print(f"  one template per column (~300 columns): ~{template_bytes / 1e3:.1f} KB")
    print("  the cells become plain numpy columns; the program collapses to templates.")


# ---------------------------------------------------------------------------
# C5: pegged memory. A tiled streaming sum over an on-disk column holds peak
# memory at one tile, whatever the column height. The dirty-column patch reads
# only the dirty columns' bytes.
# ---------------------------------------------------------------------------

def c5_pegged_memory(tmpdir="/tmp"):
    print("\n== C5: tiled streaming sum, pegged memory ==")
    import os
    path = os.path.join(tmpdir, "intro_sheet_col.f32")
    H = 100_000_000             # 100M float32 = 400 MB on disk
    tile = 4_000_000            # 16 MB tile
    arr = np.memmap(path, dtype=np.float32, mode="w+", shape=(H,))
    arr[:] = rng.random(H).astype(np.float32)
    arr.flush()

    def tiled_sum():
        total = np.float64(0.0)
        for s in range(0, H, tile):
            total += arr[s:s + tile].sum(dtype=np.float64)
        return total

    tracemalloc.start()
    _ = tiled_sum()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"  column: {H:,} float32 on disk ({H * 4 / 1e6:.0f} MB)")
    print(f"  tiled sum peak python heap: ~{peak / 1e6:.1f} MB (tile is {tile * 4 / 1e6:.0f} MB)")
    print("  peak does not grow with column height: OOM cannot happen by construction.")

    # bytes-moved: dirty-column patch vs whole file (a W-column sheet)
    W = 50
    file_bytes = H * 4 * W
    dirty_cols = 2
    patch_bytes = H * 4 * dirty_cols
    print(f"  whole-file re-sum:   {file_bytes / 1e9:.1f} GB (all {W} columns)")
    print(f"  dirty-column patch:  {patch_bytes / 1e9:.2f} GB (only {dirty_cols} columns)")
    print(f"  ratio: {file_bytes / patch_bytes:.0f}x less data moved")
    del arr
    os.remove(path)


def main():
    c1_fill_down_crossover()
    c2_sum_not_incremental()
    c3_early_cutoff()
    c4_templates()
    c5_pegged_memory()


if __name__ == "__main__":
    main()
