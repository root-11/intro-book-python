# 36 — Persistence is table serialization

> *Concept node: see the [DAG](../../concepts/dag.md) and [glossary entry 36](../../concepts/glossary.md#36--persistence-is-table-serialization).*

The simulator pauses. The world is in memory: eight columns of `creatures` (`pos_x`, `pos_y`, `vel_x`, `vel_y`, `energy`, `birth_t`, `id`, `gen`), a `food` table, presence tables (`hungry`, `dead`, etc.), the index map (`id_to_slot`), and the cleanup buffers. To pause durably, all of this must be written to disk; to resume, all of this must be read back.

The instinct most Python programmers bring: design a "persistence format" with a schema, marshalling logic, version handling, and a translation layer between in-memory objects and on-disk records. Sometimes via `pydantic`, sometimes via `dataclasses.asdict` plus `json.dumps`, sometimes via SQLAlchemy ORMs. **This is wrong on the data-oriented side. There is no translation. There is only *transposition*.**

A snapshot is the columns, written sequentially. A recovery is the columns, read sequentially. The on-disk format is the same shape as memory.

```python
import numpy as np

def snapshot(world, path: str) -> None:
    np.savez(path, tick=np.int64(world.tick), **world.columns)

def load(path: str) -> "World":
    with np.load(path) as data:
        tick = int(data["tick"])
        columns = {k: data[k] for k in data.files if k != "tick"}
    return World(tick=tick, columns=columns)
```

That is the snapshot. Recovery is the inverse. No type conversion, no field mapping, no schema discrimination at the row level. The file is exactly what the memory was; the memory is exactly what the file is.

## What it costs, four ways

From [`code/measurement/persistence_shapes.py`](../../code/measurement/persistence_shapes.py), 1,000,000 creatures across 8 columns (34 MB in memory), persisted four ways on this machine:

| layout                              | file (MB) | write (ms) | read (ms) |
|-------------------------------------|----------:|-----------:|----------:|
| pickle of `list[Creature]` (AoS)    |     85.72 |    2,105.3 |     938.5 |
| pickle of dict-of-numpy-columns     |     34.33 |        2.7 |      13.9 |
| `np.savez`                          |     34.33 |       18.8 |      62.9 |
| `np.savez_compressed`               |     25.52 |    1,004.7 |      98.5 |

Plus an unpaid invoice: **building the `list[Creature]` for the AoS variant cost 1,314 ms** before pickle even started — the construction tax from [§6](06_a_row_is_a_tuple.md). If your in-memory representation is already AoS, you carry that cost on every snapshot.

Three readings.

**The AoS form is catastrophic.** 86 MB on disk for 34 MB of data — pickle adds ~2.5× of per-row metadata, type tags, and refcount overhead. 2.1 seconds to write, 0.9 seconds to read. **778× slower writing than pickle-of-columns** for the same logical content. This is the `pickle.dump(creatures, ...)` form most Python tutorials demonstrate. It is the single most expensive way to persist a million-row world that the language offers.

**Pickle-of-numpy-columns is genuinely fast.** Numpy's `__reduce__` protocol means pickle writes the array bytes directly with thin wrappers around them — no per-row work. 2.7 ms write, 13.9 ms read for 34 MB of data is bandwidth-bound. The format is **smaller and faster than `np.savez`** in this measurement.

**`np.savez` pays for portability.** It is 7× slower to write than pickle-of-columns (18.8 ms vs 2.7 ms) because it builds a zip archive with each array as a `.npy` member. The cost buys two things pickle cannot offer:

- **Stability.** The `.npy` format is documented, versioned, and unchanged in non-breaking ways since 2007. Pickle protocols change; pickled data from one CPython version may fail to load in another, especially across major version jumps.
- **Cross-language.** `.npy` files load from Rust (`ndarray-npy`), Julia (`NPZ.jl`), and C (any of half a dozen libraries). Pickle does not.

**Compression buys ~25% disk for ~50× write time.** `np.savez_compressed` is the right choice when the file ships across a network or sits on storage that bills by the byte. It is the wrong choice when the snapshot stays on the same machine and is rewritten often.

The honest recommendation:
- **For a simulator's per-tick snapshots** (frequent, local, internal): pickle-of-numpy-columns is fastest. The portability concerns do not apply when the snapshot's only reader is the same Python process or a fork of it.
- **For checkpoint/restore across runs, machines, or language boundaries**: `np.savez`. The 7× write cost is amortised against future you not having to reverse-engineer a pickle format from a different CPython version.
- **For long-term archives or distributed transfer**: `np.savez_compressed`. The 50× write cost is paid once; the disk savings are paid forever.
- **For AoS pickle of a dataclass list**: never. The chapter's first row exists to discourage it.

## What you save by not translating

**No schema design.** The schema is whatever the columns are. Schema documentation is the column declarations.

**No object marshalling.** No `__getstate__`, no `__setstate__`, no `pydantic.BaseModel`, no `Marshmallow` schemas. The numpy array is written as bytes; bytes are read as a numpy array.

**No translation bugs.** ORMs, JSON-with-coercion, and pickled-class-hierarchies are famous sources of subtle correctness issues — fields renamed, types coerced, edge cases mishandled. Here the in-memory and on-disk forms are bit-identical; the load is `np.load(path)` and that is all.

**Deterministic recovery.** A snapshot taken in a deterministic simulator round-trips exactly. The hashed world after `snapshot → load` is identical to the hashed world before. Combined with [§16](16_determinism_by_order.md)'s rules and [§35](35_boundary_is_the_queue.md)'s queue, replay is structural.

## What it does *not* save you from

**Schema versioning.** A new column added between snapshots breaks the load. Three things can break a snapshot across environments: the *schema* changed (you added a column or renamed a type), the *byte order* differs (you saved on a little-endian machine and loaded on a big-endian one — rare on Linux/Mac/Windows but possible on certain ARM configurations), or the *Python version* differs (rare for `.npy`, common for pickle). All three have the same fix: write a small header with every snapshot — a `schema_version: int` column with one element — and at load time, run the matching migration if the field disagrees with current code. Most simulators target a single architecture and skip the migrations until they are needed; the mechanism is there from day one for the cost of a single integer.

**The pickle-version trap.** Every CPython release that adds a new pickle protocol risks invalidating pickled data from older versions. `protocol=pickle.HIGHEST_PROTOCOL` keeps you on the latest, which is great for speed and dangerous for archival. If you are picking pickle-of-columns over `np.savez` for snapshot speed, set `protocol` to a stable older version (e.g. `protocol=4`, supported since CPython 3.4) so a new Python version cannot strand your archive.

The pattern shows up everywhere this scale matters. Write-ahead logs in databases, save-game files in games, checkpoint files in HPC, frame snapshots in video editing. They all dodge the ORM trap by writing the columns directly.

The simulator's snapshot is roughly five lines of Python per direction (the code block at the top). The OOP equivalent — define a `CreatureRecord` `pydantic` model, walk the world serialising one creature at a time — is ten times the code, **two-to-three orders of magnitude slower at runtime**, and prone to the translation bugs the column-direct version cannot have.

## Exercises

1. **Snapshot the world.** Implement `snapshot(world, path)` and `load(path)` for your simulator using `np.savez`. Save to `snapshot.npz`. Note the file size; it should match `bytes per column × N` for hot tables, plus a small zip overhead per column.
2. **Round-trip test.** Save the world; reload from disk into a fresh `World`; run the simulator from the loaded state and compare the hash to the original at the same tick. They must match.
3. **Run the persistence exhibit.** `uv run code/measurement/persistence_shapes.py`. Note the catastrophic AoS-pickle row. Note that `np.savez` is *not* the fastest, but it is the most portable. Decide for your use case which row to copy.
4. **The OOP comparison, in your fingers.** Implement a per-row serialiser using `pydantic.BaseModel` or `dataclasses.asdict` plus `json.dumps`. Time it at 1M creatures. The per-row version is typically two orders of magnitude slower than `np.savez` and produces files several times larger.
5. **Schema versioning.** Add a new column (`hunger_buildup: float32`) to the simulator. Save with the new column; modify the loader to handle both old (no `hunger_buildup` key in the loaded `.npz`) and new (key present) snapshots. Old snapshots get the new column zero-filled at load. Verify both round-trip cleanly.
6. **Pickle-version stability.** Save a snapshot with `pickle.dump(world.columns, f, protocol=4)`. Save another with `protocol=pickle.HIGHEST_PROTOCOL`. Note the file sizes (small difference). Now consider: which file will still load in CPython 3.20? `protocol=4` is supported since 3.4; `HIGHEST_PROTOCOL` keeps moving.
7. *(stretch)* **Memory-mapped snapshot.** Use `np.load(path, mmap_mode='r')` to map the snapshot file directly. The arrays' bytes are the file's bytes; loading is zero-copy until the first read of each column. Compare load times for a 100 MB snapshot. The mmap form may not be faster on first read (the OS still has to fault pages in) but is *much* faster when the simulator only needs one of the columns.

Reference notes in [36_persistence_is_serialization_solutions.md](36_persistence_is_serialization_solutions.md).

## What's next

[§37 — The log is the world](37_log_is_world.md) makes the structural argument explicit: the log of events and the world's tables share a shape; one is a projection of the other.
