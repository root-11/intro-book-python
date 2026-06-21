# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy"]
# ///
"""
§53 exhibit - staleness flows downhill, in Python+numpy.

Counterpart to the Rust edition's code/scenegraph crate. A scenegraph is a tree
of nodes, each with a *local* transform; a node's *world* transform is its
parent's world composed with its local. Every frame something moves and the
world transforms below it go stale. Recompute everything, or only the dirty
part? The answer is a crossover in the dirty fraction, and a second axis - is
the stale set packed or scattered - that matters more.

What carries over from the Rust edition, and how Python sharpens it:

  - The full recompute is a dependency chain (each node needs its parent first),
    so it cannot be one flat vectorised pass. The idiomatic Python move is to
    process the tree one DEPTH LEVEL at a time: all nodes at depth d have parents
    at depth < d, already done, so a whole level composes in one vectorised batch.
    The number of Python-level iterations is the tree DEPTH, not the node count.
    A scalar per-node sweep is interpreter-bound and loses by a wide margin -
    this is the Rust "flat sweep vs pointer walk" gap, blown up by the interpreter.

  - The dirty crossover holds. Measured: recompute-dirty still wins past 60%
    dirty and only loses near 100%, so the crossover sits around two-thirds -
    if anything higher than Rust's ~40-50%, because the vectorised recompute of a
    *contiguous* dirty subtree stays cheap deep into the tree.

  - Packed vs scattered is the same lesson, sharpened: a contiguous dirty subtree
    keeps its gather/scatter cache-local; the same count scattered hops the whole
    array. Measured ~4.4x apart (Rust ~10x - numpy's fixed fancy-index overhead
    dilutes the gap, but locality still decides). The gap is the next chapter.

Claims:
  C1 scalar == vectorised      both propagation paths agree bit for bit.
  C2 vectorise by level        the level-batched full sweep beats the scalar
                               per-node sweep by a wide, size-growing margin.
  C3 dirty crossover           recompute-dirty wins when little moved; recompute
                               -all takes over past some dirty fraction.
  C4 packed beats scattered    same dirty count, contiguous subtree vs scattered.

Run:  uv run code/scenegraph/scenegraph.py
"""

import random
import time

import numpy as np

SEED = 0x5CE9


# ---------------------------------------------------------------------------
# Tree generation: pre-order layout. parent[i] < i for every non-root node, so
# index order is parent-before-child and a subtree rooted at i is the contiguous
# range [i, i + subtree_size[i]).
# ---------------------------------------------------------------------------

NO_PARENT = -1
MAX_DEPTH = 24


def generate(target, rng):
    parent = []
    subtree = []
    depth = []

    def grow(parent_idx, d):
        me = len(parent)
        parent.append(parent_idx)
        subtree.append(0)
        depth.append(d)
        size = 1
        if d < MAX_DEPTH:
            for _ in range(rng.randrange(5)):  # fan-out 0..4: many leaves
                if len(parent) >= target:
                    break
                size += grow(me, d + 1)
        subtree[me] = size
        return size

    # grow a forest of roots until we hit the target node count
    import sys
    sys.setrecursionlimit(10000)
    while len(parent) < target:
        grow(NO_PARENT, 0)

    parent = np.array(parent, dtype=np.int64)
    subtree = np.array(subtree, dtype=np.int64)
    depth = np.array(depth, dtype=np.int64)
    return parent, subtree, depth


def rand_affine_cols(n, rng):
    """n near-identity 2D affines [a b c; d e f], as six columns. Bounded so a
    deep chain of compositions stays finite and the two paths agree bit for bit."""
    angle = (np.array([rng.random() for _ in range(n)]) - 0.5) * 0.1
    scale = 0.99 + 0.02 * np.array([rng.random() for _ in range(n)])
    cs, sn = np.cos(angle) * scale, np.sin(angle) * scale
    a = cs
    b = -sn
    c = (np.array([rng.random() for _ in range(n)]) - 0.5) * 0.2
    d = sn
    e = cs
    f = (np.array([rng.random() for _ in range(n)]) - 0.5) * 0.2
    return [a, b, c, d, e, f]


def compose_into(W, L, idx, parent):
    """world[idx] = compose(world[parent[idx]], local[idx]), vectorised over idx.
    idx must be an index set all of whose parents are already final this frame."""
    p = parent[idx]
    pa, pb, pc, pd, pe, pf = (W[k][p] for k in range(6))
    la, lb, lc, ld, le, lf = (L[k][idx] for k in range(6))
    W[0][idx] = pa * la + pb * ld
    W[1][idx] = pa * lb + pb * le
    W[2][idx] = pa * lc + pb * lf + pc
    W[3][idx] = pd * la + pe * ld
    W[4][idx] = pd * lb + pe * le
    W[5][idx] = pd * lc + pe * lf + pf


# ---------------------------------------------------------------------------
# Full recompute, two ways.
# ---------------------------------------------------------------------------

def propagate_full_vec(W, L, parent, levels):
    roots = levels[0]
    for k in range(6):
        W[k][roots] = L[k][roots]  # root: compose(identity, local) == local
    for lvl in levels[1:]:
        compose_into(W, L, lvl, parent)


def propagate_full_scalar(W, L, parent):
    a, b, c, d, e, f = W
    la, lb, lc, ld, le, lf = L
    pa = parent
    for i in range(a.size):
        p = pa[i]
        if p == NO_PARENT:
            a[i], b[i], c[i], d[i], e[i], f[i] = la[i], lb[i], lc[i], ld[i], le[i], lf[i]
        else:
            a[i] = a[p] * la[i] + b[p] * ld[i]
            b[i] = a[p] * lb[i] + b[p] * le[i]   # note: reads updated a[i]? no - a[p], p<i
            c[i] = a[p] * lc[i] + b[p] * lf[i] + c[p]
            d[i] = d[p] * la[i] + e[p] * ld[i]
            e[i] = d[p] * lb[i] + e[p] * le[i]
            f[i] = d[p] * lc[i] + e[p] * lf[i] + f[p]


def propagate_incremental(W, L, parent, dirty_levels):
    """dirty_levels: dirty indices grouped by ascending depth (precomputed, so
    this times the recompute, not the marking). A dirty node's parent is clean
    (valid from the last full frame) or dirty at a lower depth (already done)."""
    for lvl in dirty_levels:
        compose_into(W, L, lvl, parent)


# ---------------------------------------------------------------------------
# Plumbing.
# ---------------------------------------------------------------------------

def levels_of(depth):
    return [np.where(depth == d)[0] for d in range(depth.max() + 1)]


def group_by_depth(idx, depth):
    d = depth[idx]
    order = np.argsort(d, kind="stable")
    idx_sorted, d_sorted = idx[order], d[order]
    cuts = np.where(np.diff(d_sorted))[0] + 1
    return np.split(idx_sorted, cuts)


def median(fn, k=3):
    return sorted(fn() for _ in range(k))[k // 2]


def time_call(fn):
    t0 = time.perf_counter()
    fn()
    return (time.perf_counter() - t0) * 1e3  # ms


def main():
    rng = random.Random(SEED)

    # ---- C1: scalar == vectorised, bit for bit ----
    parent, subtree, depth = generate(20000, rng)
    L = rand_affine_cols(len(parent), rng)
    levels = levels_of(depth)
    Wv = [np.zeros(len(parent)) for _ in range(6)]
    Ws = [np.zeros(len(parent)) for _ in range(6)]
    propagate_full_vec(Wv, L, parent, levels)
    propagate_full_scalar(Ws, L, parent)
    agree = all(np.array_equal(Wv[k], Ws[k]) for k in range(6))
    print(f"C1 scalar == vectorised full recompute, bit for bit: {'PASS' if agree else 'FAIL'}\n")

    # ---- C2: full recompute, scalar vs level-vectorised ----
    print("== Full recompute: scalar per-node vs level-vectorised ==")
    print(f"{'nodes':>10} {'depth':>6} {'scalar (ms)':>13} {'vector (ms)':>13} {'speedup':>9}")
    for target in (10_000, 100_000, 1_000_000):
        g = random.Random(0x5CE9 ^ target)
        par, sub, dep = generate(target, g)
        Lc = rand_affine_cols(len(par), g)
        lev = levels_of(dep)
        W = [np.zeros(len(par)) for _ in range(6)]
        vc = median(lambda: time_call(lambda: propagate_full_vec(W, Lc, par, lev)))
        if len(par) <= 100_000:  # scalar per-node is slow; the gap is the point
            sc = median(lambda: time_call(lambda: propagate_full_scalar(W, Lc, par)), k=1)
            print(f"{len(par):>10} {dep.max():>6} {sc:>13.2f} {vc:>13.3f} {sc / vc:>8.0f}x")
        else:
            print(f"{len(par):>10} {dep.max():>6} {'(skipped)':>13} {vc:>13.3f} {'-':>9}")

    # ---- C3: dirty crossover (vectorised recompute-dirty vs recompute-all) ----
    print("\n== Recompute-dirty vs recompute-all, by dirty fraction (1M nodes) ==")
    g = random.Random(0xD117)
    par, sub, dep = generate(1_000_000, g)
    Lc = rand_affine_cols(len(par), g)
    lev = levels_of(dep)
    n = len(par)
    W = [np.zeros(n) for _ in range(6)]
    propagate_full_vec(W, Lc, par, lev)  # establish a valid baseline frame
    all_ms = median(lambda: time_call(lambda: propagate_full_vec(W, Lc, par, lev)))
    print(f"  recompute-all (vectorised by level): {all_ms:.3f} ms\n")
    print(f"{'dirty %':>9} {'dirty nodes':>12} {'dirty (ms)':>12} {'vs all':>9}")
    # dirty as a contiguous subtree of roughly the target size
    for frac in (0.001, 0.01, 0.10, 0.20, 0.40, 0.60, 1.00):
        want = int(n * frac)
        # find a node whose subtree is closest to `want`
        start = int(np.argmin(np.abs(sub - want)))
        idx = np.arange(start, start + sub[start])
        dl = group_by_depth(idx, dep)
        ms = median(lambda: time_call(lambda: propagate_incremental(W, Lc, par, dl)))
        print(f"{frac * 100:>8.1f}% {idx.size:>12} {ms:>12.3f} {all_ms / ms:>8.2f}x")

    # ---- C4: packed contiguous subtree vs same count scattered ----
    print("\n== Same dirty count: contiguous subtree vs scattered leaves (1M nodes) ==")
    want = n // 10  # ~10% dirty
    start = int(np.argmin(np.abs(sub - want)))
    packed = np.arange(start, start + sub[start])
    k = packed.size
    scattered = np.sort(np.random.default_rng(1).choice(n, size=k, replace=False))
    pl = group_by_depth(packed, dep)
    sl = group_by_depth(scattered, dep)
    pm = median(lambda: time_call(lambda: propagate_incremental(W, Lc, par, pl)))
    sm = median(lambda: time_call(lambda: propagate_incremental(W, Lc, par, sl)))
    print(f"  dirty count       : {k}")
    print(f"  contiguous subtree: {pm:.3f} ms")
    print(f"  scattered leaves  : {sm:.3f} ms")
    print(f"  scattered / packed: {sm / pm:.2f}x")


if __name__ == "__main__":
    main()
