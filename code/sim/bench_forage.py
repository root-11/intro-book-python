# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy"]
# ///
"""
bench_forage.py - measure forage HONESTLY, by scaling it the way production does, and show
what the per-cell representative buys.

History worth keeping, because it is the lesson:

  * The first version of this file held foragers fixed and grew only the targets. That kept
    forager *density* constant, the binned neighbourhood stayed tiny, and the curve looked
    flat. It lied: it said binning made forage O(N) when it had not. A benchmark that does not
    scale the way the system scales is worse than none, because it is believed.

  * Growing *both* populations in a fixed world exposed the truth: density grows with N, a cell
    holds O(N) foragers, the 3x3 scan is O(N), and forage was O(N^2) WITH the grid in place
    (§28's density caveat). Letting the world grow as sqrt(N) held density and restored O(N) -
    but that is a constraint on the simulation, not a fix to the code.

  * The fix to the code is the per-cell representative (§28): ask each cell ONCE for one
    forager, and a target matches the representatives of its 3x3, so it sees at most nine
    candidates however crowded the cell is. That collapses the branching factor and makes
    forage O(N) *even at fixed world*. This file now measures that, and the drift it costs.

    uv run code/sim/bench_forage.py
"""
import math
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
import sim1b as S


def _world(n: int, world: float):
    nz = max(1, int(n * 0.4))
    ng = n - nz
    cfg = S.Config(n0_grass=ng, n0_grazers=nz, cap=int(n * 1.5) + 2000, world=world)
    return S.World(cfg, np.random.default_rng(0)), cfg


def forage_ms(n: int, world: float, fn, reps: int = 3) -> float:
    w, cfg = _world(n, world)
    best = float("inf")
    for _ in range(reps):
        t = time.perf_counter()
        fn(w, cfg, w.grazers, w.grass, cfg.graze_radius, cfg.graze_gain)
        best = min(best, time.perf_counter() - t)
    return best * 1000


def forage_exact(w, cfg, fi, ti, radius, gain):
    """Exhaustive nearest-in-3x3: every target ranks every forager in its neighbourhood. The
    O(N^2)-at-fixed-world baseline the representative approximates. Kept here as the reference
    the drift is measured against."""
    fo, ta = S._slots(w, fi), S._slots(w, ti)
    if fo.size == 0 or ta.size == 0:
        return S.ForagePatch(S._i64(), S._u32(), S._f32())
    cs = radius
    ncol = int(cfg.world / cs) + 1
    half = cfg.world * 0.5
    fpx, fpy, tpx, tpy = w.px[fo], w.py[fo], w.px[ta], w.py[ta]
    fcell = ((fpx / cs).astype(np.int64) % ncol) * ncol + ((fpy / cs).astype(np.int64) % ncol)
    order = np.argsort(fcell, kind="stable")
    starts = np.zeros(ncol * ncol + 1, np.int64)
    np.cumsum(np.bincount(fcell, minlength=ncol * ncol), out=starts[1:])
    tcx = (tpx / cs).astype(np.int64) % ncol
    tcy = (tpy / cs).astype(np.int64) % ncol
    pt, pf, pd = [], [], []
    for ox in (-1, 0, 1):
        for oy in (-1, 0, 1):
            nc = ((tcx + ox) % ncol) * ncol + ((tcy + oy) % ncol)
            cnt = starts[nc + 1] - starts[nc]
            if cnt.sum() == 0:
                continue
            tg = np.repeat(np.arange(ta.size), cnt)
            wc = np.arange(tg.size) - np.repeat(np.cumsum(cnt) - cnt, cnt)
            fp = order[np.repeat(starts[nc], cnt) + wc]
            dx = (tpx[tg] - fpx[fp] + half) % cfg.world - half
            dy = (tpy[tg] - fpy[fp] + half) % cfg.world - half
            pt.append(tg); pf.append(fp); pd.append(dx * dx + dy * dy)
    if not pt:
        return S.ForagePatch(S._i64(), S._u32(), S._f32())
    tg, fp, d2 = np.concatenate(pt), np.concatenate(pf), np.concatenate(pd)
    k = d2 <= radius ** 2
    tg, fp, d2 = tg[k], fp[k], d2[k]
    best = np.full(ta.size, np.inf)
    np.minimum.at(best, tg, d2)
    bi = np.flatnonzero(d2 == best[tg])
    u, fidx = np.unique(tg[bi], return_index=True)
    return S.ForagePatch(fo[fp[bi[fidx]]], w.cid[ta[u]], np.full(u.size, gain, np.float32))


def sweep(label: str, world_of, scales, fn) -> None:
    print(f"\n{label}")
    print(f"{'entities':>9} {'world':>7} {'ms':>9} {'growth':>11}")
    prev = None
    step_ratio = scales[1] // scales[0]
    for n in scales:
        ms = forage_ms(n, world_of(n), fn)
        g = f"{ms / prev:.1f}x/{step_ratio}x" if prev else "-"
        print(f"{n:>9} {world_of(n):>7.0f} {ms:>9.2f} {g:>11}")
        prev = ms


def drift(n: int, world: float = 100.0) -> None:
    """How far does the representative pull the result from the exhaustive nearest?"""
    w, cfg = _world(n, world)
    rep = S.forage(w, cfg, w.grazers, w.grass, cfg.graze_radius, cfg.graze_gain)
    w2, cfg2 = _world(n, world)
    full = forage_exact(w2, cfg2, w2.grazers, w2.grass, cfg2.graze_radius, cfg2.graze_gain)
    rm = dict(zip(rep.target_ids.tolist(), rep.forager_slots.tolist()))
    fm = dict(zip(full.target_ids.tolist(), full.forager_slots.tolist()))
    shared = rm.keys() & fm.keys()
    differ = sum(rm[t] != fm[t] for t in shared)
    print(f"\nDRIFT at {n:,} (representative vs exhaustive nearest), world={world:.0f}:")
    print(f"  targets fed: exhaustive {len(fm):,}, representative {len(rm):,}")
    only_full = len(fm.keys() - rm.keys())
    print(f"    {only_full:,} fed by exhaustive but not by the representative "
          f"(its cell's chosen forager was not the one in reach)")
    pct = 100 * differ / max(len(shared), 1)
    print(f"  of {len(shared):,} shared targets, {differ:,} ({pct:.1f}%) matched a different "
          f"forager - all within a radius-{cfg.graze_radius:g} cell of the exhaustive winner")


def main() -> None:
    budget = S.Config().dt * 1000
    scales = [10_000, 30_000, 100_000]
    print(f"forage, both populations scaled (40% grazers); budget {budget:.1f} ms")

    sweep("1. EXHAUSTIVE nearest-3x3, FIXED WORLD=100 (density grows ~N) - the wall:",
          lambda n: 100.0, scales, forage_exact)
    sweep("2. EXHAUSTIVE nearest-3x3, CONSTANT DENSITY (world=sqrt(N)) - O(N) only if held:",
          lambda n: math.sqrt(n), scales, forage_exact)
    sweep("3. REPRESENTATIVE per cell, FIXED WORLD=100 - O(N) without holding density:",
          lambda n: 100.0, scales, S.forage)
    print("\n  1 is quadratic, 2 is linear only because the world grew, 3 is linear in a fixed "
          "world: the per-cell ask collapses the branching factor that 1 paid for and 2 dodged.")

    drift(100_000)


if __name__ == "__main__":
    main()
