# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy"]
# ///
"""
§20 exhibit — optional fields on every entity vs presence table.

A 1,000,000-creature world where some have a disease. Two layouts:

    1. AoS with optional field. Every Creature instance carries a
       `disease` slot, even though most are None. Every creature pays
       the slot's pointer cost regardless of whether the disease exists.

    2. Numpy SoA with presence table. A main creature table (numpy
       columns) plus a tiny `diseased` array holding the affected
       creature ids and a parallel `severity` column. Healthy creatures
       pay nothing for the disease state.

Three measurements per (layout, prevalence) cell:
    - footprint (RSS delta over baseline)
    - "process diseased" wall time (one tick of disease-driven energy drain)
    - count of creatures actually processed

Each cell measured in a fresh subprocess so RSS readings don't bleed.
Prevalences include 0% — where the empty-tables-free property is the
whole point.

Run:
    uv run code/measurement/empty_tables.py
"""

import gc
import multiprocessing as mp
import resource
import time
from dataclasses import dataclass

import numpy as np


N = 1_000_000
PREVALENCES = [0.0, 0.001, 0.01, 0.10]


def rss_kb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


@dataclass(slots=True)
class Disease:
    severity: float


@dataclass(slots=True)
class Creature:
    pos: float
    vel: float
    energy: float
    disease: object  # Optional[Disease] — None for healthy


def setup_optional(prevalence: float) -> list:
    rng = np.random.default_rng(42)
    creatures = []
    for i in range(N):
        d = Disease(severity=float(rng.random())) if rng.random() < prevalence else None
        creatures.append(Creature(pos=float(i), vel=0.1, energy=1.0, disease=d))
    return creatures


def process_diseased_optional(creatures: list) -> int:
    n = 0
    for c in creatures:
        if c.disease is not None:
            c.energy -= c.disease.severity * 0.01
            n += 1
    return n


def setup_presence(prevalence: float) -> tuple:
    rng = np.random.default_rng(42)
    pos = np.arange(N, dtype=np.float32)
    vel = np.full(N, 0.1, dtype=np.float32)
    energy = np.ones(N, dtype=np.float32)
    diseased_mask = rng.random(N) < prevalence
    diseased_ids = np.where(diseased_mask)[0].astype(np.uint32)
    severity = rng.random(diseased_ids.size).astype(np.float32)
    return (pos, vel, energy, diseased_ids, severity)


def process_diseased_presence(state: tuple) -> int:
    pos, vel, energy, diseased_ids, severity = state
    energy[diseased_ids] -= severity * 0.01
    return int(diseased_ids.size)


def worker(name: str, prevalence: float, q) -> None:
    gc.collect()
    rss_before = rss_kb()
    if name == "optional":
        state = setup_optional(prevalence)
        process = process_diseased_optional
        layout = "list[Creature] with Optional[Disease]"
    else:
        state = setup_presence(prevalence)
        process = process_diseased_presence
        layout = "numpy SoA + diseased presence"
    rss_after = rss_kb()

    process(state)  # warmup
    gc.collect()
    t0 = time.perf_counter()
    n = process(state)
    t1 = time.perf_counter()

    q.put({
        "layout": layout,
        "prevalence": prevalence,
        "rss_kb": rss_after - rss_before,
        "process_ms": (t1 - t0) * 1000.0,
        "n_processed": n,
    })


def main() -> None:
    print(f"N = {N:,} creatures.   Each cell measured in a fresh subprocess.\n")
    header = (f"{'prevalence':>11}  {'layout':<40}  {'RSS (MB)':>9}  "
              f"{'process (ms)':>13}  {'n diseased':>11}")
    print(header)
    print("-" * len(header))

    for prev in PREVALENCES:
        for name in ("optional", "presence"):
            q = mp.Queue()
            p = mp.Process(target=worker, args=(name, prev, q))
            p.start()
            r = q.get()
            p.join()
            print(f"{prev*100:>10.2f}%  {r['layout']:<40}  "
                  f"{r['rss_kb']/1024:>9.1f}  {r['process_ms']:>13.4f}  "
                  f"{r['n_processed']:>11,}")
        print()


if __name__ == "__main__":
    main()
