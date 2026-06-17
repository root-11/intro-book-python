# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy"]
# ///
"""
scale_sweep.py - the §4 instrument. Sweep every system in the tick across scale and read
each one's cost against the budget, so the binding system is found in development, by a
curve, not in production, by a dropped frame.

Method (per CAPEX_OPEX_NOTES.md): build a representative *churning* world at N live entities
(energies spread so a slice reproduces and a slice starves, positions random so a slice
forages), run one tick instrumented per system, take the min over a few reps (the OS only
adds time, so the minimum is the machine's floor). Sweep N across decades.

    uv run code/sim/scale_sweep.py
"""
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
import sim1b as S

SYSTEMS = ["regenerate", "herd_move", "forage", "reproduce", "die", "apply", "cleanup", "inspect"]


def stressed_world(n: int, rng):
    """N live entities in a churning state: ~40% grazers, energies spread across the
    reproduce/starve thresholds so both fire this tick, positions random so foraging fires."""
    nz = max(1, int(n * 0.4))
    ng = n - nz
    cfg = S.Config(n0_grass=ng, n0_grazers=nz, cap=int(n * 2) + 2000)
    w = S.World(cfg, rng)
    w.energy[:n] = rng.uniform(0.0, cfg.repro_threshold * 1.2, n).astype(np.float32)
    return w, cfg


def occupancy(w, cfg):
    """The branching factor b: foragers per occupied cell at forage's cell size (= graze
    radius). forage is O(N*b), so b is what picks the data structure, not the wall-clock:
    b~1 is a pure lookup, b~10-100 a scan-or-sample, b in the thousands wants a subtree.
    A tight mean==max says uniform density (a flat grid is enough); a heavy tail (max >>
    mean) is clumping, the regime that wants a hierarchical grid."""
    slots = S._slots(w, w.grazers)
    if slots.size == 0:
        return 0.0, 0, 0
    cs = cfg.graze_radius
    ncol = int(cfg.world / cs) + 1
    cell = ((w.px[slots] / cs).astype(np.int64) % ncol) * ncol + \
           ((w.py[slots] / cs).astype(np.int64) % ncol)
    nz = np.bincount(cell)
    nz = nz[nz > 0]
    return float(nz.mean()), int(np.percentile(nz, 99)), int(nz.max())


def regime(b: float) -> str:
    return "1:1 lookup" if b <= 1.5 else "1:10-100 scan/sample" if b <= 100 else "1:1000+ subtree"


def timed_tick(w, cfg, rng) -> dict:
    """step(), unrolled with a timer per system."""
    ms = {}

    def clock(name, fn):
        a = time.perf_counter()
        r = fn()
        ms[name] = (time.perf_counter() - a) * 1000
        return r

    log = S.Log()
    w.d_energy[: w.n_active] = 0.0
    clock("regenerate", lambda: S.regenerate(w, cfg, rng))
    clock("herd_move", lambda: S.herd_move(w, cfg, rng, w.grazers, cfg.herd_speed, cfg.herd_burn))
    foraged = [clock("forage", lambda: S.forage(w, cfg, w.grazers, w.grass,
                                                cfg.graze_radius, cfg.graze_gain))]
    born = clock("reproduce", lambda: S.reproduce(w, cfg, rng))
    dead = clock("die", lambda: S.die(w, cfg))
    clock("apply", lambda: S.apply(w, log, cfg, foraged, born, dead))
    clock("cleanup", lambda: S.cleanup(w, cfg))
    clock("inspect", lambda: S.inspect(w, log))
    return ms


def main() -> None:
    budget = S.Config().dt * 1000
    scales = [10_000, 30_000, 100_000]
    reps = 3
    table = {s: {} for s in SYSTEMS}
    for n in scales:
        best = {s: float("inf") for s in SYSTEMS}
        for _ in range(reps):
            w, cfg = stressed_world(n, np.random.default_rng(0))
            ms = timed_tick(w, cfg, np.random.default_rng(1))
            for s in SYSTEMS:
                best[s] = min(best[s], ms[s])
        for s in SYSTEMS:
            table[s][n] = best[s]

    print(f"per-system tick cost, min of {reps} reps, budget {budget:.1f} ms/tick\n")
    print("system".ljust(11) + "".join(f"{n:>11,}" for n in scales) + "    growth (cost x / scale x)")
    for s in SYSTEMS:
        row = s.ljust(11) + "".join(f"{table[s][n]:>11.2f}" for n in scales)
        lo, hi = table[s][scales[0]], table[s][scales[-1]]
        row += f"    {hi / max(lo, 1e-9):>5.0f}x / {scales[-1] // scales[0]}x"
        print(row)
    totals = {n: sum(table[s][n] for s in SYSTEMS) for n in scales}
    print("total".ljust(11) + "".join(f"{totals[n]:>11.2f}" for n in scales))
    over = [s for s in SYSTEMS if table[s][scales[-1]] > budget]
    binding = max(SYSTEMS, key=lambda s: table[s][scales[-1]])
    print(f"\nbinding system at {scales[-1]:,}: {binding} ({table[binding][scales[-1]]:.1f} ms)")
    print(f"over budget at {scales[-1]:,}: {', '.join(over) if over else 'none'}")

    print(f"\nforage branching factor b (foragers per occupied cell), the number that picks "
          f"the structure:\n{'scale':>9} {'mean':>7} {'p99':>6} {'max':>6}   regime (by tail)")
    for n in scales:
        w, cfg = stressed_world(n, np.random.default_rng(0))
        mean, p99, mx = occupancy(w, cfg)
        print(f"{n:>9,} {mean:>7.1f} {p99:>6} {mx:>6}   {regime(mx)}")
    print("tight mean~max -> uniform density, a flat grid suffices; heavy tail (max>>mean) "
          "-> clumping, which wants the hierarchical grid.")


if __name__ == "__main__":
    main()
