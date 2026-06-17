# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy"]
# ///
"""
parallel_forage.py - slice the WORK, not the world, and prove the single-core result equals
the multi-core result bit-for-bit, whatever the OS does with the schedule. Also: contiguous
slice vs modulo (strided) partition, to separate correctness (partition-independent) from speed
(partition-dependent), and to locate the bandwidth saturation.

The design (Bjorn's): the world arrays live in shared memory, every worker reads all of them
(no spatial boundary, no halo), and each worker writes only its own disjoint subset of a shared
write-zone. No locks, because the writes never overlap.

Why it is deterministic, as a contract:
  * The partition is FIXED before the run (worker k always owns the same target indices), so it
    does not depend on timing.
  * forage's per-target choice is a pure map - each target picks the nearest of its cell's
    representatives independently - so there is NO float reduction across targets; the only
    order-dependent operation (the one that would drift) is absent here.
  * The write-zone is indexed by target, so worker writes are disjoint by construction. The
    final array is identical regardless of which core wrote which subset or in what order,
    AND regardless of whether the partition is a contiguous slice or a strided (modulo) one.

That last clause is the point of the slice-vs-modulo comparison: the RESULT is the same either
way (asserted), but the SPEED is not, because the access pattern is.

    uv run code/sim/parallel_forage.py
"""
import time
from multiprocessing import Pool, shared_memory

import numpy as np

WORLD = 1000.0
CS = 2.0
NCOL = int(WORLD / CS) + 1
_SH: dict = {}


def make_rep(fx, fy):
    fcell = ((fx / CS).astype(np.int64) % NCOL) * NCOL + ((fy / CS).astype(np.int64) % NCOL)
    rep = np.full(NCOL * NCOL, -1, np.int64)
    rep[fcell] = np.arange(fx.size)
    return rep


def forage_idx(fx, fy, tx, ty, rep, out, tidx):
    """For the targets named by tidx, write each one's chosen forager into out[tidx]. Pure map:
    target g's result depends only on g's own <=9 candidates, so it is identical no matter which
    other targets share the batch or how tidx is shaped (contiguous or strided)."""
    tpx, tpy = tx[tidx], ty[tidx]
    tcx = (tpx / CS).astype(np.int64) % NCOL
    tcy = (tpy / CS).astype(np.int64) % NCOL
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
            dx = (tpx[t] - fx[f] + half) % WORLD - half
            dy = (tpy[t] - fy[f] + half) % WORLD - half
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


def partition(mode, k, w, nt):
    """The FIXED set of target indices worker k owns. Contiguous slice or strided modulo - both
    disjoint partitions of [0, nt)."""
    if mode == "slice":
        step = (nt + w - 1) // w
        return np.arange(k * step, min((k + 1) * step, nt))
    return np.arange(k, nt, w)  # modulo: worker k owns k, k+w, k+2w, ...


def _init(specs):
    for key, (name, shape, dt) in specs.items():
        shm = shared_memory.SharedMemory(name=name)
        _SH[key] = (shm, np.ndarray(shape, dtype=dt, buffer=shm.buf))


def _worker(args):
    mode, k, w = args
    out = _SH["out"][1]
    tidx = partition(mode, k, w, out.size)
    forage_idx(_SH["fx"][1], _SH["fy"][1], _SH["tx"][1], _SH["ty"][1], _SH["rep"][1], out, tidx)


def _publish(arr, store):
    shm = shared_memory.SharedMemory(create=True, size=arr.nbytes)
    np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)[:] = arr
    store.append(shm)
    return (shm.name, arr.shape, arr.dtype)


def run_mode(mode, out, serial, specs, serial_ms):
    print(f"\n{mode} partition:")
    print(f"{'workers':>7} {'ms':>9} {'speedup':>8}   identical")
    for w in (1, 2, 4, 8):
        tasks = [(mode, k, w) for k in range(w)]
        with Pool(w, initializer=_init, initargs=(specs,)) as p:
            out[:] = -1
            p.map(_worker, tasks)        # warm
            out[:] = -1
            t = time.perf_counter()
            p.map(_worker, tasks)        # fixed tasks, order-preserving collection
            ms = (time.perf_counter() - t) * 1000
        ok = np.array_equal(out, serial)
        print(f"{w:>7} {ms:>9.1f} {serial_ms / ms:>7.1f}x   {ok}")
        assert ok, f"NON-DETERMINISM: {mode} w={w} != serial"


def main():
    rng = np.random.default_rng(0)
    nf = nt = 2_000_000
    fx = rng.uniform(0, WORLD, nf).astype(np.float32)
    fy = rng.uniform(0, WORLD, nf).astype(np.float32)
    tx = rng.uniform(0, WORLD, nt).astype(np.float32)
    ty = rng.uniform(0, WORLD, nt).astype(np.float32)
    rep = make_rep(fx, fy)

    serial = np.full(nt, -1, np.int64)
    t = time.perf_counter()
    forage_idx(fx, fy, tx, ty, rep, serial, np.arange(nt))
    serial_ms = (time.perf_counter() - t) * 1000
    print(f"foragers={nf:,} targets={nt:,} fed={int((serial >= 0).sum()):,}  serial {serial_ms:.1f} ms")

    store = []
    out_shm = shared_memory.SharedMemory(create=True, size=serial.nbytes)
    out = np.ndarray(nt, dtype=np.int64, buffer=out_shm.buf)
    store.append(out_shm)
    specs = {"fx": _publish(fx, store), "fy": _publish(fy, store),
             "tx": _publish(tx, store), "ty": _publish(ty, store),
             "rep": _publish(rep, store), "out": (out_shm.name, (nt,), np.int64)}
    try:
        run_mode("slice", out, serial, specs, serial_ms)
        run_mode("modulo", out, serial, specs, serial_ms)
    finally:
        for shm in store:
            shm.close(); shm.unlink()
    print("\nResult identical under slice AND modulo; speed differs because the access pattern "
          "does. The leftover saturation is the scattered forager gather, not the schedule.")


if __name__ == "__main__":
    main()
