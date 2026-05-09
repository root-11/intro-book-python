# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy"]
# ///
"""
§18 exhibit — scan cost vs alive-fraction across three layouts.

A motion update — pos += vel * dt — applied only to alive creatures, at
N = 1,000,000 creatures, varying the alive fraction from 1% to 100%.

Three layouts:

    1. AoS list of dataclass instances with an `alive: bool` field
       (the Python tutorial default for "soft delete")
    2. Numpy SoA + boolean mask column      (`pos[alive_mask] += ...`)
    3. Numpy SoA + presence id array         (`pos[alive_ids] += ...`)

The lesson:
- Layout 1 scales with N regardless of alive-fraction (interpreter dispatch
  on every iteration plus per-attribute getattr).
- Layout 2 has to scan all N to evaluate the mask, then operate on K alive
  rows. Cost is dominated by the N-sized scan at low alive-fraction.
- Layout 3 reads K alive ids and operates on K rows. Cost scales with K.

At alive = 100% all three numpy paths are similar. At alive = 1% the
presence layout does roughly 1% of the work the mask layout does. The
AoS layout is interpreter-bound and barely cares about alive-fraction.

Run:
    uv run code/measurement/alive_fraction.py
"""

import gc
import time
from dataclasses import dataclass

import numpy as np


N = 1_000_000
DT = 1.0 / 30.0
ALIVE_FRACTIONS = [0.01, 0.10, 0.50, 0.90, 1.00]
WARMUP = 1
REPEATS = 3


@dataclass(slots=True)
class Creature:
    pos: float
    vel: float
    alive: bool


def best(times):
    return min(times)


def time_run(setup_fn, body_fn, *body_args):
    state = setup_fn()
    for _ in range(WARMUP):
        body_fn(state, *body_args)
    times = []
    for _ in range(REPEATS):
        gc.collect()
        t0 = time.perf_counter()
        body_fn(state, *body_args)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)
    return best(times)


# AoS variant
def setup_aos(alive_frac):
    rng = np.random.default_rng(0xA105)
    pos = rng.random(N).tolist()
    vel = (rng.random(N) * 0.1).tolist()
    alive = (rng.random(N) < alive_frac).tolist()
    return [Creature(p, v, a) for p, v, a in zip(pos, vel, alive)]


def tick_aos(creatures, dt):
    for c in creatures:
        if c.alive:
            c.pos += c.vel * dt


# Bool-mask variant
def setup_mask(alive_frac):
    rng = np.random.default_rng(0xA105)
    pos = rng.random(N, dtype=np.float32)
    vel = rng.random(N, dtype=np.float32) * np.float32(0.1)
    alive_mask = rng.random(N) < alive_frac
    return pos, vel, alive_mask


def tick_mask(state, dt):
    pos, vel, alive_mask = state
    pos[alive_mask] += vel[alive_mask] * dt


# Presence variant
def setup_presence(alive_frac):
    rng = np.random.default_rng(0xA105)
    pos = rng.random(N, dtype=np.float32)
    vel = rng.random(N, dtype=np.float32) * np.float32(0.1)
    alive_mask = rng.random(N) < alive_frac
    alive_ids = np.where(alive_mask)[0].astype(np.uint32)
    return pos, vel, alive_ids


def tick_presence(state, dt):
    pos, vel, alive_ids = state
    pos[alive_ids] += vel[alive_ids] * dt


def main():
    print(f"N = {N:,} creatures, motion: pos += vel * dt   (dt = 1/30)")
    print(f"Three layouts; varying alive-fraction.")
    print()
    header = f"{'alive %':>8}  {'AoS (ms)':>10}  {'mask (ms)':>10}  {'presence (ms)':>14}  {'mask/presence':>15}"
    print(header)
    print("-" * len(header))

    for frac in ALIVE_FRACTIONS:
        # AoS is slow at any fraction; we still measure for honesty.
        aos_ms = time_run(lambda: setup_aos(frac), tick_aos, DT)
        mask_ms = time_run(lambda: setup_mask(frac), tick_mask, DT)
        presence_ms = time_run(lambda: setup_presence(frac), tick_presence, DT)
        ratio = mask_ms / presence_ms
        print(f"{frac*100:>7.1f}%  "
              f"{aos_ms:>10.2f}  "
              f"{mask_ms:>10.3f}  "
              f"{presence_ms:>14.3f}  "
              f"{ratio:>14.1f}×")


if __name__ == "__main__":
    main()
