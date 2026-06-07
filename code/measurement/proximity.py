# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "scipy"]
# ///
"""
§28 exhibit - proximity is a property of position, in Python+numpy.

Counterpart to the Rust edition's proximity.rs. The Rust lesson - "bin, don't
index; recompute from the position stream beats maintaining a bolt-on
structure" - HOLDS in numpy, with one Python-specific catch: the per-beast
O(1) bucket read must be run as ONE vectorised batch over all beasts. A naive
Python loop of O(1) reads is so slow it loses to scipy's cKDTree and tempts
you to the wrong conclusion ("the grid is slow, use the library"). Vectorised,
the grid beats cKDTree outright, because it does O(N) work where the tree does
O(N log N).

Claims:
  P1 all-pairs wall      O(N^2) neighbour test grows past the frame budget.
  P2 grid vs index, 1M   vectorised grid query vs cKDTree (+ the naive loop, the trap).
  P3 rebuild / query     the grid's CSR rebuild is a small fraction of its query.
  P4 pack-leader         O(N) centroid vs O(N^2) all-pairs cohesion.

    uv run code/measurement/proximity.py
"""
import time
import numpy as np
from scipy.spatial import cKDTree

RNG = np.random.default_rng(0)
BOX = 1000.0          # world is BOX x BOX
R = 2.0               # neighbour radius == cell size


def med(fn, trials=5):
    fn()
    ts = []
    for _ in range(trials):
        t = time.perf_counter(); fn(); ts.append(time.perf_counter() - t)
    return float(np.median(ts))


def world(n):
    return (RNG.random(n) * BOX).astype(np.float32), (RNG.random(n) * BOX).astype(np.float32)


# --- P1: all-pairs neighbour count, chunked so memory stays bounded ---
def all_pairs_count(px, py, r, chunk=1000):
    n = px.size; r2 = r * r; out = np.empty(n, dtype=np.int64)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        dx = px[s:e, None] - px[None, :]; dy = py[s:e, None] - py[None, :]
        out[s:e] = ((dx * dx + dy * dy) <= r2).sum(axis=1)
    return out


# --- dense CSR bin: the spatial structure, recomputed from the position stream ---
def build_csr(px, py, r):
    ncols = int(BOX / r) + 2
    cx = (px / r).astype(np.int64); cy = (py / r).astype(np.int64)
    cell = cx * ncols + cy
    order = np.argsort(cell, kind="stable")               # point indices grouped by cell
    offsets = np.zeros(ncols * ncols + 1, dtype=np.int64)
    np.cumsum(np.bincount(cell, minlength=ncols * ncols), out=offsets[1:])
    return cx, cy, ncols, order, offsets


def _expand(starts, lengths):
    """Flat (src, pos): for each i, the range [starts[i], starts[i]+lengths[i])."""
    mask = lengths > 0; starts = starts[mask]; lengths = lengths[mask]
    src_ids = np.nonzero(mask)[0]
    if starts.size == 0:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    total = int(lengths.sum()); out = np.ones(total, dtype=np.int64); out[0] = starts[0]
    csum = np.cumsum(lengths)
    out[csum[:-1]] = starts[1:] - (starts[:-1] + lengths[:-1]) + 1
    return np.repeat(src_ids, lengths), np.cumsum(out)


def grid_query_vectorised(px, py, r):
    """Every beast's 3x3 bucket read, run as ONE vectorised batch over all beasts."""
    n = px.size; r2 = r * r
    cx, cy, ncols, order, offsets = build_csr(px, py, r)
    total = np.zeros(n, dtype=np.int64)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            ncx = cx + dx; ncy = cy + dy
            valid = (ncx >= 0) & (ncx < ncols) & (ncy >= 0) & (ncy < ncols)
            nc = np.where(valid, ncx * ncols + ncy, 0)
            lengths = np.where(valid, offsets[nc + 1] - offsets[nc], 0)
            src, flat = _expand(offsets[nc], lengths)
            j = order[flat]
            d = (px[src] - px[j]) ** 2 + (py[src] - py[j]) ** 2
            total += np.bincount(src[d <= r2], minlength=n)
    return total


def grid_query_pyloop(px, py, r):
    """The trap: O(1) per query, but a Python loop over beasts. C-speed algorithm,
       interpreter-speed constant."""
    n = px.size; r2 = r * r
    cx, cy, ncols, order, offsets = build_csr(px, py, r)
    out = np.empty(n, dtype=np.int64)
    for i in range(n):
        cnt = 0; ix = int(cx[i]); iy = int(cy[i])
        for ox in (-1, 0, 1):
            for oy in (-1, 0, 1):
                c = (ix + ox) * ncols + (iy + oy)
                if 0 <= c < offsets.size - 1:
                    for k in range(int(offsets[c]), int(offsets[c + 1])):
                        jj = order[k]; ddx = px[i] - px[jj]; ddy = py[i] - py[jj]
                        if ddx * ddx + ddy * ddy <= r2:
                            cnt += 1
        out[i] = cnt
    return out


def main():
    print(f"numpy {np.__version__}   box {BOX:.0f}x{BOX:.0f}   radius {R}")

    print("\nP1  all-pairs neighbour count wall (O(N^2))")
    for n in (2000, 5000, 10000, 20000):
        px, py = world(n)
        print(f"    N={n:>6}:  {med(lambda: all_pairs_count(px, py, R), 3)*1e3:8.1f} ms")

    print("\nP2  query 1M (grid binned from the position stream)")
    px, py = world(1_000_000); pts = np.column_stack([px, py])
    t_grid = med(lambda: grid_query_vectorised(px, py, R), 3)
    t_kd = med(lambda: cKDTree(pts).query_ball_point(pts, R, return_length=True), 3)
    print(f"    vectorised grid                      : {t_grid*1e3:8.1f} ms")
    print(f"    cKDTree (bolt-on index, build+query) : {t_kd*1e3:8.1f} ms   grid {t_kd/t_grid:.2f}x faster")
    pxs, pys = world(100_000); ps = np.column_stack([pxs, pys])
    t_loop = med(lambda: grid_query_pyloop(pxs, pys, R), 1)
    t_kd_s = med(lambda: cKDTree(ps).query_ball_point(ps, R, return_length=True), 3)
    print(f"    [trap] naive Python-loop grid @100k  : {t_loop*1e3:8.1f} ms  vs cKDTree {t_kd_s*1e3:6.1f} ms "
          f"({t_loop/t_kd_s:.1f}x SLOWER - vectorise, or you will wrongly blame the grid)")

    print("\nP3  grid CSR rebuild vs its query, 1M (recompute is cheap)")
    t_build = med(lambda: build_csr(px, py, R), 3)
    print(f"    rebuild {t_build*1e3:6.1f} ms   full query {t_grid*1e3:7.1f} ms   rebuild/query {t_build/t_grid*100:4.1f}%")

    print("\nP4  pack-leader: O(N) centroid vs O(N^2) all-pairs cohesion")
    n = 20000; px, py = world(n)
    def cohesion():
        out = np.empty(n, dtype=np.float32); chunk = 1000
        for s in range(0, n, chunk):
            e = min(s + chunk, n)
            out[s:e] = (px[s:e, None] - px[None, :]).mean(axis=1)
        return out
    def centroid():
        return px - px.mean(), py - py.mean()
    t_ap = med(cohesion, 2); t_ld = med(centroid, 50)
    print(f"    N={n}:  all-pairs cohesion {t_ap*1e3:8.1f} ms   centroid {t_ld*1e6:7.1f} us   {t_ap/t_ld:9.0f}x")


if __name__ == "__main__":
    main()
