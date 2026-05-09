# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy"]
# ///
"""
§11 exhibit — what fits in a tick at increasing N?

Define a minimal motion system: position += velocity * dt. Run it once per
tick. Measure wall time per tick at N from 10K to 10M. Report the cost as
a fraction of the 30 Hz budget (33 ms) and the 60 Hz budget (16.67 ms).

Two layouts at each N:
    1. numpy SoA: two contiguous float32 columns, one bulk update.
    2. Python AoS: list of dataclass instances, per-instance attribute write.

The numpy loop is bandwidth-bound (per §4); the dataclass loop is
interpreter-bound. The point of this exhibit is to make the budget
binding visible: at what N does each layout stop fitting in a 30 Hz tick?

Run:
    uv run code/measurement/tick_budget.py
"""

import gc
import time
from dataclasses import dataclass

import numpy as np


SIZES = [10_000, 100_000, 1_000_000, 10_000_000]
BUDGET_30HZ_MS = 1000.0 / 30.0
BUDGET_60HZ_MS = 1000.0 / 60.0
DT = 1.0 / 30.0
WARMUP = 2
REPEATS = 5


@dataclass(slots=True)
class Creature:
    pos: float
    vel: float


def build_soa(n, rng):
    pos = rng.random(n, dtype=np.float32)
    vel = rng.random(n, dtype=np.float32) * 0.1
    return pos, vel


def tick_soa(pos, vel, dt):
    pos += vel * dt  # in-place; one bulk numpy op


def build_aos(n, rng):
    pos = rng.random(n).tolist()
    vel = (rng.random(n) * 0.1).tolist()
    return [Creature(p, v) for p, v in zip(pos, vel)]


def tick_aos(creatures, dt):
    for c in creatures:
        c.pos += c.vel * dt


def time_tick(setup_fn, tick_fn):
    """Run tick_fn(*setup_fn()) once, measuring wall time. Best of REPEATS."""
    times = []
    state = setup_fn()
    for _ in range(WARMUP):
        tick_fn(*state, DT) if isinstance(state, tuple) else tick_fn(state, DT)
    for _ in range(REPEATS):
        gc.collect()
        if isinstance(state, tuple):
            t0 = time.perf_counter()
            tick_fn(*state, DT)
            t1 = time.perf_counter()
        else:
            t0 = time.perf_counter()
            tick_fn(state, DT)
            t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)  # ms
    return min(times)


def fits(ms, budget_ms):
    """Return a small string indicator: 'fit', 'tight' (>50%), or 'over'."""
    if ms > budget_ms:
        return f"OVER ({ms / budget_ms * 100:5.0f}%)"
    pct = ms / budget_ms * 100
    return f"fit  ({pct:5.1f}%)"


def main():
    rng = np.random.default_rng(0xC0FFEE)
    print(f"Motion system: pos += vel * dt   (dt = 1/30 s)")
    print(f"Budgets: 30 Hz = {BUDGET_30HZ_MS:.2f} ms,  60 Hz = {BUDGET_60HZ_MS:.2f} ms")
    print()
    header = f"{'N':>11}  {'layout':<22}  {'tick (ms)':>9}   {'30 Hz':<14}   {'60 Hz':<14}"
    print(header)
    print("-" * len(header))

    for n in SIZES:
        soa_ms = time_tick(lambda: build_soa(n, rng), tick_soa)
        print(f"{n:>11,}  {'numpy SoA':<22}  {soa_ms:>9.3f}   "
              f"{fits(soa_ms, BUDGET_30HZ_MS):<14}   {fits(soa_ms, BUDGET_60HZ_MS):<14}")
        # Skip dataclass at 10M — would take minutes per tick.
        if n <= 1_000_000:
            aos_ms = time_tick(lambda: build_aos(n, rng), tick_aos)
            print(f"{n:>11,}  {'Python dataclass list':<22}  {aos_ms:>9.3f}   "
                  f"{fits(aos_ms, BUDGET_30HZ_MS):<14}   {fits(aos_ms, BUDGET_60HZ_MS):<14}")
        else:
            print(f"{n:>11,}  {'Python dataclass list':<22}  {'(skipped)':>9}   "
                  f"{'extrapolates over':<14}   {'extrapolates over':<14}")


if __name__ == "__main__":
    main()
