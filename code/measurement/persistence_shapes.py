# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy"]
# ///
"""
§36 exhibit — four ways to persist a 1M-row world.

Same data, four serialisation layouts:

    1. pickle of list[dataclass]
       The Python tutorial form. Every Creature is a Python object;
       pickle walks the list, encoding each object with type tags,
       attribute names, and refcounts. AoS plus per-row protocol.

    2. pickle of dict-of-numpy-columns
       SoA in memory, but going through pickle's protocol. Pickle
       knows about numpy and uses numpy's reduce method, so this is
       roughly "raw bytes plus pickle wrappers" — much faster than (1)
       but still not the fastest.

    3. np.savez
       Numpy's native format. Each named array becomes a .npy file
       inside a .zip; the .npy bytes are exactly the array's bytes.
       No per-row work; one bulk write per column.

    4. np.savez_compressed
       Same as (3) with DEFLATE compression. Trades CPU for disk
       space; useful when the file ships across a network or sits
       around for a long time.

Three measurements per layout: file size, write time, read time.
A round-trip checksum verifies correctness.

Run:
    uv run code/measurement/persistence_shapes.py
"""

import gc
import os
import pickle
import tempfile
import time
from dataclasses import dataclass

import numpy as np


N = 1_000_000
SEED = 0xCAFE


@dataclass(slots=True)
class Creature:
    pos_x: float
    pos_y: float
    vel_x: float
    vel_y: float
    energy: float
    birth_t: float
    id: int
    gen: int


def make_world():
    rng = np.random.default_rng(SEED)
    columns = {
        "pos_x":   rng.random(N, dtype=np.float32),
        "pos_y":   rng.random(N, dtype=np.float32),
        "vel_x":   rng.random(N, dtype=np.float32) * np.float32(0.1),
        "vel_y":   rng.random(N, dtype=np.float32) * np.float32(0.1),
        "energy":  rng.random(N, dtype=np.float32),
        "birth_t": rng.random(N, dtype=np.float64),
        "id":      np.arange(N, dtype=np.uint32),
        "gen":     np.zeros(N, dtype=np.uint32),
    }
    return columns


def world_to_aos(columns) -> list:
    return [
        Creature(
            pos_x=float(columns["pos_x"][i]),
            pos_y=float(columns["pos_y"][i]),
            vel_x=float(columns["vel_x"][i]),
            vel_y=float(columns["vel_y"][i]),
            energy=float(columns["energy"][i]),
            birth_t=float(columns["birth_t"][i]),
            id=int(columns["id"][i]),
            gen=int(columns["gen"][i]),
        )
        for i in range(N)
    ]


def checksum_columns(columns) -> float:
    return float(sum(c.sum() for c in columns.values()))


def checksum_aos(creatures) -> float:
    return float(sum(c.pos_x + c.pos_y + c.vel_x + c.vel_y +
                     c.energy + c.birth_t + c.id + c.gen
                     for c in creatures))


def time_call(fn):
    gc.collect()
    t0 = time.perf_counter()
    result = fn()
    t1 = time.perf_counter()
    return result, (t1 - t0) * 1000.0  # ms


def measure_pickle_aos(creatures, path):
    _, write_ms = time_call(
        lambda: pickle.dump(creatures, open(path, "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    )
    size = os.path.getsize(path)
    loaded, read_ms = time_call(lambda: pickle.load(open(path, "rb")))
    return size, write_ms, read_ms, checksum_aos(loaded)


def measure_pickle_columns(columns, path):
    _, write_ms = time_call(
        lambda: pickle.dump(columns, open(path, "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    )
    size = os.path.getsize(path)
    loaded, read_ms = time_call(lambda: pickle.load(open(path, "rb")))
    return size, write_ms, read_ms, checksum_columns(loaded)


def measure_savez(columns, path, compressed: bool):
    saver = np.savez_compressed if compressed else np.savez
    _, write_ms = time_call(lambda: saver(path, **columns))
    actual_path = path + ".npz"
    size = os.path.getsize(actual_path)

    def load_all():
        with np.load(actual_path) as data:
            return {k: data[k] for k in data.files}

    loaded, read_ms = time_call(load_all)
    return size, write_ms, read_ms, checksum_columns(loaded)


def main():
    print(f"N = {N:,} creatures, 8 columns (5 × float32 + 1 × float64 + 2 × uint32)")
    print(f"In-memory column total: {(5*4 + 8 + 2*4) * N / 1024**2:.1f} MB\n")

    columns = make_world()
    expected = checksum_columns(columns)

    with tempfile.TemporaryDirectory() as tmpdir:
        results = []

        # AoS pickle: build the list once (it's slow), measure pickle separately.
        print("Building AoS list (1M dataclass instances)...", flush=True)
        creatures, build_ms = time_call(lambda: world_to_aos(columns))
        print(f"AoS build alone: {build_ms:.0f} ms (one-time pre-cost; not counted in 'write')\n")

        path = os.path.join(tmpdir, "aos.pkl")
        size, w, r, ck = measure_pickle_aos(creatures, path)
        results.append(("pickle of list[dataclass]", size, w, r, ck))
        del creatures

        path = os.path.join(tmpdir, "cols.pkl")
        size, w, r, ck = measure_pickle_columns(columns, path)
        results.append(("pickle of dict-of-numpy-columns", size, w, r, ck))

        path = os.path.join(tmpdir, "save")
        size, w, r, ck = measure_savez(columns, path, compressed=False)
        results.append(("np.savez", size, w, r, ck))

        path = os.path.join(tmpdir, "save_compressed")
        size, w, r, ck = measure_savez(columns, path, compressed=True)
        results.append(("np.savez_compressed", size, w, r, ck))

    header = (f"{'layout':<35}  {'file (MB)':>10}  {'write (ms)':>11}  "
              f"{'read (ms)':>10}  {'round-trip':>12}")
    print(header)
    print("-" * len(header))
    for label, size, w, r, ck in results:
        ok = "OK" if abs(ck - expected) < 1.0 else "MISMATCH"
        print(f"{label:<35}  {size/1024**2:>10.2f}  {w:>11.1f}  "
              f"{r:>10.1f}  {ok:>12}")

    print(f"\nExpected checksum: {expected:.4f}")


if __name__ == "__main__":
    main()
