# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy"]
# ///
"""
measure_forage_layout.py - is forage's wall the memory bus, or the scattered access pattern?

forage gathers rep[cell] and then fx[f], both at scattered indices. If we order foragers AND
targets by cell, those gathers stream (consecutive cells sit at consecutive memory), so:
  * if single-core gets faster, the kernel was cache/latency-bound, not compute-bound;
  * if the 8-worker speedup climbs past ~5.6x, the saturation was the scattered access wasting
    cache lines, not the raw bus;
  * if nothing moves, it is the bus (or compute) and we are genuinely near the machine limit.

Also probes int16 positions (half the bytes per gather) as a second bandwidth lever.

    uv run code/sim/measure_forage_layout.py
"""
import time
from multiprocessing import Pool, shared_memory

import numpy as np

WORLD = 1000.0
CS = 2.0
NCOL = int(WORLD / CS) + 1
_SH: dict = {}


def cells(x, y):
    return ((x.astype(np.float64) / CS).astype(np.int64) % NCOL) * NCOL + \
           ((y.astype(np.float64) / CS).astype(np.int64) % NCOL)


def make_rep(fx, fy):
    rep = np.full(NCOL * NCOL, -1, np.int64)
    rep[cells(fx, fy)] = np.arange(fx.size)
    return rep


def forage_idx(fx, fy, tx, ty, rep, out, tidx):
    tpx, tpy = tx[tidx], ty[tidx]
    tcx = (tpx.astype(np.float64) / CS).astype(np.int64) % NCOL
    tcy = (tpy.astype(np.float64) / CS).astype(np.int64) % NCOL
    half = WORLD * 0.5
    ti = np.arange(tidx.size)
    pt, pf, pd = [], [], []
    for ox in (-1, 0, 1):
        for oy in (-1, 0, 1):
            r = rep[((tcx + ox) % NCOL) * NCOL + ((tcy + oy) % NCOL)]
            has = r >= 0
            if not has.any():
                continue
            t, f = ti[has], r[has]
            dx = (tpx[t].astype(np.float32) - fx[f] + half) % WORLD - half
            dy = (tpy[t].astype(np.float32) - fy[f] + half) % WORLD - half
            pt.append(t); pf.append(f); pd.append(dx * dx + dy * dy)
    if not pt:
        return
    tgt, fpos, d2 = np.concatenate(pt), np.concatenate(pf), np.concatenate(pd)
    keep = d2 <= CS ** 2
    tgt, fpos, d2 = tgt[keep], fpos[keep], d2[keep]
    best = np.full(tidx.size, np.inf)
    np.minimum.at(best, tgt, d2)
    bi = np.flatnonzero(d2 == best[tgt])
    u, fi = np.unique(tgt[bi], return_index=True)
    out[tidx[u]] = fpos[bi[fi]]


def _init(specs):
    for k, (name, shape, dt) in specs.items():
        shm = shared_memory.SharedMemory(name=name)
        _SH[k] = (shm, np.ndarray(shape, dtype=dt, buffer=shm.buf))


BLOCK = 20_000  # process each worker's slice in cache-resident blocks (0 = whole slice)


def _worker(args):
    k, w = args
    out = _SH["out"][1]
    nt = out.size
    step = (nt + w - 1) // w
    lo, hi = k * step, min((k + 1) * step, nt)
    blk = BLOCK or (hi - lo)
    for b in range(lo, hi, blk):
        forage_idx(_SH["fx"][1], _SH["fy"][1], _SH["tx"][1], _SH["ty"][1],
                   _SH["rep"][1], out, np.arange(b, min(b + blk, hi)))


def _publish(arr, store):
    shm = shared_memory.SharedMemory(create=True, size=arr.nbytes)
    np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)[:] = arr
    store.append(shm)
    return (shm.name, arr.shape, arr.dtype)


def best_ms(fn, reps=3):
    b = float("inf")
    for _ in range(reps):
        t = time.perf_counter(); fn(); b = min(b, time.perf_counter() - t)
    return b * 1000


def measure(label, fx, fy, tx, ty):
    nt = tx.size
    rep = make_rep(fx, fy)
    serial = np.full(nt, -1, np.int64)
    s_ms = best_ms(lambda: forage_idx(fx, fy, tx, ty, rep, serial, np.arange(nt)))

    store = []
    out_shm = shared_memory.SharedMemory(create=True, size=serial.nbytes)
    out = np.ndarray(nt, dtype=np.int64, buffer=out_shm.buf)
    store.append(out_shm)
    specs = {"fx": _publish(fx, store), "fy": _publish(fy, store),
             "tx": _publish(tx, store), "ty": _publish(ty, store),
             "rep": _publish(rep, store), "out": (out_shm.name, (nt,), np.int64)}
    try:
        with Pool(8, initializer=_init, initargs=(specs,)) as p:
            tasks = [(k, 8) for k in range(8)]
            p.map(_worker, tasks)                 # warm
            out[:] = -1
            p_ms = best_ms(lambda: p.map(_worker, tasks), reps=3)
        ok = np.array_equal(out, serial)
    finally:
        for shm in store:
            shm.close(); shm.unlink()
    print(f"{label:22} serial {s_ms:7.1f} ms   8-core {p_ms:7.1f} ms   {s_ms / p_ms:4.1f}x   identical={ok}")
    return s_ms, p_ms


def main():
    rng = np.random.default_rng(0)
    n = 2_000_000
    fx = rng.uniform(0, WORLD, n).astype(np.float32)
    fy = rng.uniform(0, WORLD, n).astype(np.float32)
    tx = rng.uniform(0, WORLD, n).astype(np.float32)
    ty = rng.uniform(0, WORLD, n).astype(np.float32)

    fo = np.argsort(cells(fx, fy), kind="stable")
    to = np.argsort(cells(tx, ty), kind="stable")

    print(f"foragers=targets={n:,}, world={WORLD:.0f}\n")
    measure("random float32", fx, fy, tx, ty)
    measure("cell-sorted float32", fx[fo], fy[fo], tx[to], ty[to])
    # int16 positions: half the bytes per gather (world fits in int16)
    measure("random int16", fx.astype(np.int16), fy.astype(np.int16),
            tx.astype(np.int16), ty.astype(np.int16))
    measure("cell-sorted int16", fx[fo].astype(np.int16), fy[fo].astype(np.int16),
            tx[to].astype(np.int16), ty[to].astype(np.int16))


if __name__ == "__main__":
    main()
