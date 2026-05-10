# Solutions: 36 — Persistence is table serialization

## Exercise 1 — Snapshot the world

```python
import numpy as np
from pathlib import Path

def snapshot(world, path: str) -> None:
    np.savez(path,
        tick      = np.array([world.tick], dtype=np.uint64),
        n_active  = np.array([world.n_active], dtype=np.uint32),
        pos_x     = world.pos_x[: world.n_active],
        pos_y     = world.pos_y[: world.n_active],
        vel_x     = world.vel_x[: world.n_active],
        vel_y     = world.vel_y[: world.n_active],
        energy    = world.energy[: world.n_active],
        id        = world.id[: world.n_active],
        gen       = world.gen[: world.n_active],
        birth_t   = world.birth_t[: world.n_active],
    )

def load(path: str, capacity: int) -> "World":
    data = np.load(path)
    world = build_world(capacity=capacity)
    world.tick     = int(data["tick"][0])
    world.n_active = int(data["n_active"][0])
    for name in ("pos_x", "pos_y", "vel_x", "vel_y", "energy", "id", "gen", "birth_t"):
        getattr(world, name)[: world.n_active] = data[name]
    return world

snapshot(world, "world.npz")
restored = load("world.npz", capacity=world.capacity)
```

File size: `n_active × bytes_per_row + small zip overhead`. For 1M creatures × 36 bytes = 36 MB plus ~80 KB of zip metadata. Slicing to `[: n_active]` avoids saving the unused tail.

## Exercise 2 — Round-trip test

```python
def hash_world(world) -> str:
    import hashlib
    h = hashlib.blake2b(digest_size=16)
    for name in ("pos_x", "pos_y", "vel_x", "vel_y", "energy", "id", "gen"):
        h.update(getattr(world, name)[: world.n_active].tobytes())
    return h.hexdigest()

# Round-trip
h_before = hash_world(world)
snapshot(world, "rt.npz")
restored = load("rt.npz", capacity=world.capacity)
h_after = hash_world(restored)
assert h_before == h_after

# Continue from the loaded state — should match a never-paused run
for _ in range(100): tick(restored)
control = build_world(seed=42); restore_from(world)   # same starting state
for _ in range(100): tick(control)
assert hash_world(restored) == hash_world(control)
```

The snapshot/load round-trip must be bit-identical. Combined with the [§16](16_determinism_by_order.md) deterministic rules, this gives you full pause-and-resume capability — the loaded world runs forward identically to one that never paused.

## Exercise 3 — Run the persistence exhibit

```sh
uv run code/measurement/persistence_shapes.py
```

```
layout                                file (MB)   write (ms)   read (ms)
-------------------------------------------------------------------------
pickle of list[dataclass]                 85.72      2185.4       927.1
pickle of dict-of-numpy-columns           34.33         2.9        14.5
np.savez                                  34.33        26.2        26.3
np.savez_compressed                       25.52       989.2        95.9
```

Plus the AoS-list construction cost itself: ~1050 ms. So pickling a million dataclass instances costs 3.3 seconds total (build + write); the equivalent numpy SoA snapshot is **3 ms** for the write (~1000× faster) without the construction step at all (the columns *are* the data).

The pickle-of-columns row is fastest for the simulator's per-tick snapshot use case. `np.savez` adds 7× the write time for cross-language portability — a fair price for a checkpoint format you'd like to read from Rust or Julia. Compression adds another 38× write time for 25% disk savings — only worth it for archival.

## Exercise 4 — The OOP comparison, in your fingers

```python
from pydantic import BaseModel
import json, time

class CreatureRecord(BaseModel):
    pos_x: float; pos_y: float
    vel_x: float; vel_y: float
    energy: float
    id: int

# Build records (this alone is slow)
records = [CreatureRecord(pos_x=float(world.pos_x[i]), pos_y=float(world.pos_y[i]),
                          vel_x=float(world.vel_x[i]), vel_y=float(world.vel_y[i]),
                          energy=float(world.energy[i]), id=int(world.id[i]))
           for i in range(world.n_active)]

# Serialise
t = time.perf_counter()
with open("oop.json", "w") as f:
    json.dump([r.model_dump() for r in records], f)
print(f"pydantic+json write: {(time.perf_counter()-t)*1000:.0f} ms")
```

Typical: ~5-15 seconds for 1M creatures, file size ~250+ MB. Two-to-three orders of magnitude slower than `np.savez`. The pydantic + json combination pays for: per-row instance construction, per-field validation, per-row dict construction, per-field JSON encoding, per-row JSON boundary.

The numpy-column form does none of this — the bytes are written verbatim. The OOP version's "advantages" (human-readable JSON, validation) are mostly mirages for a million-row simulator state: nobody reads it by hand, and validation should live at the queue boundary (§35), not at every snapshot.

## Exercise 5 — Schema versioning

```python
SCHEMA_VERSION = 2

def snapshot_v2(world, path):
    np.savez(path,
        schema_version = np.array([SCHEMA_VERSION], dtype=np.uint32),
        # ... existing columns ...
        hunger_buildup = world.hunger_buildup[: world.n_active],  # NEW in v2
    )

def load(path, capacity):
    data = np.load(path)
    version = int(data["schema_version"][0]) if "schema_version" in data.files else 1
    world = build_world(capacity=capacity)
    # ... load common columns ...
    if version >= 2:
        world.hunger_buildup[: world.n_active] = data["hunger_buildup"]
    else:
        world.hunger_buildup[: world.n_active] = 0.0           # zero-fill for old snapshots
    return world
```

The migration is *additive at load time*: old snapshots load with the new column zero-filled; new snapshots load all columns. Renaming columns or changing dtypes requires a real migration (read the old name, write to the new column at the right dtype). The version field is the disambiguator.

In practice most simulators bump the version on every breaking change and write a one-shot script to migrate old snapshot files when needed.

## Exercise 6 — Pickle-version stability

```python
import pickle
with open("p4.pkl", "wb") as f:
    pickle.dump(world.columns, f, protocol=4)         # stable since Python 3.4
with open("phighest.pkl", "wb") as f:
    pickle.dump(world.columns, f, protocol=pickle.HIGHEST_PROTOCOL)
```

File size difference: usually <5%. The wire format is similar; the main difference is `protocol=5` (added in 3.8) supports out-of-band buffers for large arrays, slightly more efficient for huge payloads.

The question is *forward compatibility*: in CPython 3.20, will `protocol=4` still load? Almost certainly yes — protocol 4 has been stable for over a decade and `pickle` maintains backward compatibility. Will `protocol=pickle.HIGHEST_PROTOCOL` from today still load in 3.20? Probably yes too, but the guarantee is weaker.

For long-term archives, prefer `np.savez` (`.npy` format frozen since 2007) over pickle at any protocol. For short-term internal snapshots where the same Python process reads what it wrote: protocol=HIGHEST is fine.

## Exercise 7 — Memory-mapped snapshot (stretch)

```python
import numpy as np, time

# 100 MB file with one column
path = "/tmp/big.npy"
arr = np.zeros(12_500_000, dtype=np.float64)         # 100 MB
np.save(path, arr)

# Full load
t = time.perf_counter()
full = np.load(path)
print(f"np.load full: {(time.perf_counter()-t)*1000:.0f} ms")

# Memory-mapped — does no actual I/O until first access
t = time.perf_counter()
mm = np.load(path, mmap_mode='r')
print(f"np.load mmap: {(time.perf_counter()-t)*1000:.2f} ms")

# Touch one element — pages get faulted in
t = time.perf_counter()
val = float(mm[1_000_000])
print(f"first read:   {(time.perf_counter()-t)*1e6:.0f} µs")

# Touch the whole thing — pays the I/O now
t = time.perf_counter()
s = float(mm.sum())
print(f"full sum:     {(time.perf_counter()-t)*1000:.0f} ms")
```

Typical:

```
np.load full: 60 ms       (reads the whole file into memory)
np.load mmap: 0.1 ms      (just opens the file; no I/O)
first read:   80 µs       (faults in one 4-KB page)
full sum:     50 ms       (faults in all pages — pays I/O now)
```

The mmap form is *much* faster at *open time* and faster overall *if the program only reads part of the data*. For the simulator: if a snapshot has 20 columns and the inspector only wants one, mmap reads only that column's bytes from disk. For a full restore, mmap pays the same total I/O — just amortised across first accesses.

For per-tick snapshots that get fully restored, the standard `np.load` is fine. For large checkpoints where you might want to inspect one column without paying for all of them, mmap wins.
