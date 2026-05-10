# Solutions: 30 — Moving beyond the wall

## Exercise 1 — Compute your streaming threshold

```
Per-creature footprint (hot SoA):
  pos_x  float32  4 bytes
  pos_y  float32  4 bytes
  vel_x  float32  4 bytes
  vel_y  float32  4 bytes
  energy float32  4 bytes
                  ─────
  hot total      20 bytes / creature

Per-creature footprint (full SoA, with cold):
  + birth_t  float64  8 bytes
  + id       uint32   4 bytes
  + gen      uint32   4 bytes
                      ──
  full total         36 bytes / creature

Plus id_to_slot (4 bytes per id ever issued) and side buffers (~20% pad).

RAM available to the simulator: assume 8 GB on a 16 GB laptop.
  Hot only:   8 × 10^9 / 20  = 400 million creatures
  Full SoA:   8 × 10^9 / 36  = 222 million
```

The streaming threshold is in the *hundreds of millions* for hot-only data on a typical laptop, but every cold column you add chips away. Adding a `name: object` column (one Python string per creature) blows the budget at ~50M because each string is 50+ bytes.

This is why the §2 dtype discipline and the §26 hot/cold split bind together. Wider dtypes pull the wall inward; the split pushes the *motion-system* wall outward by isolating the hot working set.

## Exercise 2 — Predict the cost

| storage          | latency per read |
|------------------|-----------------:|
| NVMe SSD         |     ~100 µs      |
| SATA SSD         |    300-500 µs    |
| spinning HDD     |       ~10 ms     |
| network (LAN)    |       ~500 µs    |
| network (WAN)    |     50-200 ms    |

Within a 33 ms tick budget:

| storage          | max reads per tick |
|------------------|-------------------:|
| NVMe SSD         |       ~300         |
| SATA SSD         |        ~70         |
| spinning HDD     |          3         |
| LAN              |        ~60         |
| WAN              |     0.1-0.5        |

A simulator that wants to make *thousands* of disk reads per tick fits on no storage tier. The fix is *batched* reads: gather all the indices needed this tick, issue one big read for the contiguous range, parse the bytes locally. One read of 1 MB on NVMe costs ~1 ms; reading 1000 individual 1 KB chunks costs ~100 ms.

The disk's bandwidth-per-second is high; its operations-per-second is low. Match the access pattern to the bandwidth, not to the IOPS.

## Exercise 3 — Snapshot a small world

```python
import numpy as np

def snapshot(world, path):
    np.savez_compressed(path,
        pos_x = world.pos_x[: world.n_active],
        pos_y = world.pos_y[: world.n_active],
        vel_x = world.vel_x[: world.n_active],
        vel_y = world.vel_y[: world.n_active],
        energy = world.energy[: world.n_active],
        id = world.id[: world.n_active],
        n_active = np.array([world.n_active], dtype=np.uint32),
    )

def restore(world, path):
    data = np.load(path)
    n = int(data["n_active"][0])
    world.n_active = n
    world.pos_x[:n]  = data["pos_x"]
    world.pos_y[:n]  = data["pos_y"]
    # ...

snapshot(world, "checkpoint.npz")
# ... continue simulation ...
restore(world, "checkpoint.npz")
# the world is byte-identical to the snapshot
```

`np.savez_compressed` ships the typed bytes verbatim, zip-deflated. The file is portable, language-readable, and self-describing (named arrays). For a 1M-creature world: ~30 MB uncompressed, ~15-25 MB compressed depending on entropy.

The simulator's continuation after `restore` is indistinguishable from the original run — *this is determinism* (§16) plus *persistence-as-serialisation* (§36). The combination is replay.

## Exercise 4 — A windowed log

```python
import numpy as np, sqlite3

class WindowedLog:
    def __init__(self, window_size: int, db_path: str):
        self.window_tick = np.zeros(window_size, dtype=np.uint32)
        self.window_id   = np.zeros(window_size, dtype=np.uint32)
        self.window_kind = np.zeros(window_size, dtype=np.uint8)
        self.window_head = 0
        self.window_size = window_size
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("CREATE TABLE IF NOT EXISTS events "
                          "(tick INTEGER, id INTEGER, kind INTEGER)")

    def append(self, tick: int, creature_id: int, kind: int):
        if self.window_head >= self.window_size:
            self.flush()
        i = self.window_head
        self.window_tick[i] = tick
        self.window_id[i]   = creature_id
        self.window_kind[i] = kind
        self.window_head += 1

    def flush(self):
        rows = list(zip(self.window_tick[:self.window_head].tolist(),
                        self.window_id[:self.window_head].tolist(),
                        self.window_kind[:self.window_head].tolist()))
        self.conn.executemany("INSERT INTO events VALUES (?, ?, ?)", rows)
        self.conn.commit()
        self.window_head = 0

    def query_window(self, tick: int):
        mask = self.window_tick[:self.window_head] == tick
        return self.window_id[:self.window_head][mask]

    def query_archive(self, tick: int):
        cur = self.conn.execute("SELECT id FROM events WHERE tick = ?", (tick,))
        return np.array([row[0] for row in cur], dtype=np.uint32)
```

Window queries are O(K) numpy scans (~1 µs at K=10K). Archive queries are O(log N) sqlite reads (~5-30 µs after the page is in cache). The window is the hot path; the archive is the cold path.

## Exercise 5 — Log-as-world

```python
def replay_to_tick(log: WindowedLog, target_tick: int, snapshots_dir: Path):
    """Reconstruct world state at target_tick using the most recent snapshot ≤ target_tick
       plus a replay of the log from the snapshot's tick to target_tick."""
    # Find the most recent snapshot ≤ target_tick
    snaps = sorted(snapshots_dir.glob("snap_*.npz"))
    chosen = max((s for s in snaps if int(s.stem.split("_")[1]) <= target_tick), default=None)
    if chosen is None:
        world = build_world_initial()
        start_tick = 0
    else:
        world = restore_snapshot(chosen)
        start_tick = int(chosen.stem.split("_")[1])
    # Replay events from start_tick to target_tick
    for tick in range(start_tick, target_tick):
        events_in_window = log.query_window(tick)
        events_in_archive = log.query_archive(tick) if tick < (target_tick - log.window_size) else np.empty(0, dtype=np.uint32)
        # apply events to world
        apply_events(world, np.concatenate([events_in_archive, events_in_window]), tick)
    return world
```

The reconstruction time depends on `target_tick - start_tick`: more events to replay = more work. Periodic snapshots cap the replay length; a snapshot every 1000 ticks means at most 1000 ticks of replay per query.

This is the architecture of every event-sourced system, every git, every database WAL.

## Exercise 6 — Read the simlog seriously

The vendored simlog at [`.archive/simlog/logger.py`](https://codeberg.org/root-11/intro-book-python/src/branch/main/.archive/simlog/logger.py) implements the windowed-log pattern in 700 lines. Trace one `log(...)` call:

1. **Inside the simulation**: `log(time, value, **fields)` is called.
2. **Active container write**: the call writes a row to the *active* `Container` (a pre-allocated numpy SoA buffer). Counter increments.
3. **Container full check**: if the container has hit its capacity, the swap fires.
4. **Atomic swap**: `active, inactive = inactive, active`. Both are pre-allocated; no allocation happens.
5. **Background thread**: a worker thread waiting on `inactive` notices its `n_used > 0`, opens an `.npz` file, dumps the columns, marks `inactive.n_used = 0`.
6. **Simulation continues**: next `log()` call writes to the (now-empty, swapped-in) active container.

The 700 lines you don't have to write include: codebook compression for repeated string fields, type inference (one f64 column holds ints, floats, and string codes), throughput benchmarks, and the auxiliary `to_csv` / `to_sqlite` exporters. The reference implementation is the production version of every chapter from §15 to §30.

## Exercise 7 — Chunked numpy

```python
import numpy as np, time

# Build a 2 GB file
path = "/tmp/big.npy"
n_total = 2_000_000_000 // 8                     # 250M float64 = 2 GB
np.save(path, np.zeros(n_total, dtype=np.float64))    # write zeros once

# Approach 1: load the whole thing
t = time.perf_counter()
arr = np.load(path)
m = arr.mean()
print(f"full load: {time.perf_counter() - t:.2f} s, mean={m}")
del arr

# Approach 2: chunked via mmap-less fromfile
t = time.perf_counter()
total = 0.0; n = 0
with open(path, "rb") as f:
    f.read(128)                                   # skip header (.npy magic + dtype info)
    chunk_bytes = 100 * 1024 * 1024               # 100 MB
    while True:
        raw = f.read(chunk_bytes)
        if not raw: break
        chunk = np.frombuffer(raw, dtype=np.float64)
        total += float(chunk.sum())
        n += chunk.size
print(f"chunked: {time.perf_counter() - t:.2f} s, mean={total/n}")
```

The chunked version's wall time is similar (~3-5 s on NVMe for 2 GB) but caps RAM at 100 MB instead of 2 GB. For files larger than RAM, chunking is the only option; for files smaller than RAM, the full-load is usually slightly faster (fewer syscalls).

## Exercise 8 — Document your bound (stretch)

A simulator's deployment bound is a one-paragraph document:

> **Simulator deployment bound.** On the reference hardware (16 GB RAM, NVMe SSD, 8-core Ryzen 5800), the simulator runs N ≤ 8,000,000 creatures at 30 Hz with the hot-path memory footprint of 160 MB (20 bytes × 8M). Above 8M, the L3 → RAM cliff begins to bind motion's inner loop; we project N=20M to run at 15 Hz (50% deadline missed). The streaming architecture (windowed log + snapshots every 1000 ticks) is required above 50M, where the full SoA exceeds typical desktop RAM.

The document is what tells future readers (including you) when to escalate the architecture and when to just buy more RAM. It is the closing artifact of the Scale phase — the explicit price tag on running at each scale.
