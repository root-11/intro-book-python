# Solutions: 37 — The log is the world

## Exercise 1 — Log the simulator

```python
import numpy as np

class TripleStore:
    def __init__(self, capacity: int):
        self.rids = np.zeros(capacity, dtype=np.uint32)
        self.keys = np.zeros(capacity, dtype=np.uint8)
        self.vals = np.zeros(capacity, dtype=np.float64)
        self.n = 0

    def append(self, rid: int, key: int, val: float):
        i = self.n
        self.rids[i] = rid
        self.keys[i] = key
        self.vals[i] = val
        self.n += 1

# Key codes — a 1-byte enum
KEY_POS_X     = 0
KEY_POS_Y     = 1
KEY_ENERGY    = 2
KEY_BIRTH_T   = 3
KEY_DIED      = 4
KEY_BORN      = 5

log = TripleStore(capacity=100_000_000)

# Cleanup pushes triples for every applied mutation
def cleanup_with_log(world, buffer, log):
    for slot in buffer.to_remove:
        log.append(int(world.id[slot]), KEY_DIED, float(world.tick))
    for i in range(len(buffer.to_insert_id)):
        cid = buffer.to_insert_id[i]
        log.append(cid, KEY_BORN, float(world.tick))
        log.append(cid, KEY_POS_X, float(buffer.to_insert_pos_x[i]))
        log.append(cid, KEY_POS_Y, float(buffer.to_insert_pos_y[i]))
    # ... apply mutations as before ...
```

Each triple is `(rid, key, val)` — entity id, column code, value. The log is three parallel numpy columns. After 100 ticks of a 1000-creature simulation with moderate churn: ~100K-1M triples, depending on event rate.

## Exercise 2 — Reconstruct from the log

```python
def replay(initial_state: dict, events: TripleStore, up_to_tick: int = None) -> dict:
    """Apply every event in the log to the initial state. Returns the resulting world tables."""
    world = {k: v.copy() for k, v in initial_state.items()}
    alive = set(world["id"].tolist())

    for i in range(events.n):
        rid, key, val = int(events.rids[i]), int(events.keys[i]), float(events.vals[i])
        if up_to_tick is not None:
            # If your log includes a tick column, gate on it; else assume sequential
            pass
        if key == KEY_BORN:
            alive.add(rid)
            # extend world arrays — left as exercise; in a real implementation use slot recycling
        elif key == KEY_DIED:
            alive.discard(rid)
        elif key == KEY_POS_X:
            # locate slot for rid and write val
            pass
        # ... etc ...
    return world

# Compare:
live_world = run_live(seed=42, ticks=100)
replayed_world = replay(initial_state(seed=42), log)
assert hash_world(live_world) == hash_world(replayed_world)
```

If the replay matches the live world bit-for-bit, the log captures every mutation. If it doesn't, an event type is missing from the log (or the apply logic differs between live and replay). The cleanup pass is the canonical place to record events; *every* mutation flows through it (§22), so logging there gives complete coverage.

## Exercise 3 — Save and load the log

```python
def save_log(log: TripleStore, path: str):
    np.savez(path,
        rids = log.rids[: log.n],
        keys = log.keys[: log.n],
        vals = log.vals[: log.n],
    )

def load_log(path: str, capacity: int) -> TripleStore:
    data = np.load(path)
    log = TripleStore(capacity=capacity)
    n = len(data["rids"])
    log.rids[:n] = data["rids"]
    log.keys[:n] = data["keys"]
    log.vals[:n] = data["vals"]
    log.n = n
    return log

save_log(log, "events.npz")
reloaded = load_log("events.npz", capacity=100_000_000)
replayed = replay(initial_state(seed=42), reloaded)
assert hash_world(live) == hash_world(replayed)
```

The log is just three numpy columns; the [§36](36_persistence_is_serialization.md) `np.savez` pattern applies unchanged. Round-trip is byte-identical because the log is *only* bytes — no objects, no pointers, no schema mismatches.

## Exercise 4 — Snapshot + log

```python
def reconstruct_at(tick_T, snapshots_dir, log_path):
    """Return the world state at tick T, using the most recent snapshot ≤ T plus log replay."""
    snaps = sorted(Path(snapshots_dir).glob("snap_*.npz"))
    chosen = max((s for s in snaps if int(s.stem.split("_")[1]) <= tick_T), default=None)
    if chosen is None:
        world = initial_state(seed=42)
        start_tick = 0
    else:
        world = load_snapshot(chosen)
        start_tick = int(chosen.stem.split("_")[1])
    log = load_log(log_path, capacity=100_000_000)
    # filter to events with tick in [start_tick, tick_T]
    return replay_in_range(world, log, start_tick, tick_T)

# Snapshots every 1000 ticks; log keeps growing
# Worst-case replay: 1000 ticks worth of events — much faster than replaying from t=0
```

This is the production replay architecture. Snapshots cap the replay window; the log holds everything in between. Storage scales with `O(events) + O(snapshots × world_size)`; recovery time is `O(events_per_snapshot_interval)`.

## Exercise 5 — Run simlog

Tracing one `log(time, value, **fields)` call through `.archive/simlog/logger.py`:

1. **Field code lookup**: each `**fields` key is converted to its uint8 code via the codebook (`self.codebook` dict). New strings get a fresh code; existing ones reuse the prior code. O(1) per field.
2. **Value normalisation**: each value is cast to `f64`. Strings become codebook codes packed into `f64` (a uint32 code fits inside the int53 mantissa exactly).
3. **Write to active container**: the row is appended to `self.active.rids`, `self.active.keys`, `self.active.vals` at index `self.active.n_used`. Counter increments.
4. **Capacity check**: if `self.active.n_used == self.active.capacity` (200K rows), trigger the swap.
5. **The swap (revolver)**: `self.active, self.inactive = self.inactive, self.active`. Both are pre-allocated `Container` objects; no allocation. The previously-active container is now waiting for the background thread.
6. **Background flush**: the worker thread (`_write_chunk`) notices `self.inactive.n_used > 0`, opens an `.npz` file, writes the three columns, sets `self.inactive.n_used = 0`.

Cost: ~0.9-1.9 µs per `log()` call, almost all in steps 1-3. Steps 4-6 amortise across 200K calls.

The 700 lines you don't have to write: codebook serialisation, `to_csv` and `to_sqlite` post-processors, type-coercion edge cases, capacity tuning, signal handling for graceful shutdown.

## Exercise 6 — The codebook saving

```python
import numpy as np
n_events = 1_000_000

# Literal-string form
strings = np.array(["eat"] * n_events, dtype=object)
# size: each "eat" is a Python str — ~50 bytes object + 3 bytes content
# total: ~50 MB

# Codebook form  
codes = np.full(n_events, 0, dtype=np.uint8)         # all the same code
codebook = {"eat": 0}                                 # one-row codebook
# size: 1 MB for codes + 50 bytes for the codebook
```

24-50× smaller. The codebook overhead is *fixed* (size of unique strings × ~50 bytes), not per-event. With 100 unique kinds and 1M events, the codebook is 5 KB and the codes are 1 MB; the literal-string form is 50 MB.

This is the structural argument for codebooks: as the corpus grows, the codebook stays the same size while the event log doubles. The ratio improves linearly with corpus size.

## Exercise 7 — The `logging` module trap

```python
import logging, time

# logging module form
logging.basicConfig(filename="events.log", level=logging.INFO)
t0 = time.perf_counter()
for cid in range(100_000):
    logging.info(f"creature {cid} ate food {cid+1000} energy_delta=0.5")
t_log = time.perf_counter() - t0

# numpy triple-store form  
log = TripleStore(capacity=100_000)
t0 = time.perf_counter()
for cid in range(100_000):
    log.append(cid, KEY_EAT, 0.5)
t_npy = time.perf_counter() - t0
```

Typical results:

| metric | logging module | numpy triple-store |
|--------|---------------:|-------------------:|
| write time      | 1.5-3 s        |     1-5 ms         |
| file size       | 6 MB (strings) | 0.3 MB (typed columns) |
| query "events for creature 42" | parse every line (~100 ms) | `np.where(rids == 42)` (~50 µs) |

The logging module is a string-formatting + per-event-flush + level-filtering machine. None of those features helps the simulator. The triple-store form is faster on every axis and queryable without parsing.

## Exercise 8 — The simlog API, three views (stretch)

**As a class (`class Simlog`)**: pip-installable, reusable across simulators. Public API stays stable across versions. Best for code that crosses package boundaries — used by Mesa-like frameworks, audit-log tools, third-party simulators. Cost: a layer of indirection between simulator and log; can't access simulator internals.

**As a module inside your simulator**: same shape, no external boundary. The logger knows about your simulator's specific table shapes and field codes. Faster (no abstraction layer); not reusable. Best for a single bespoke simulator that doesn't ship its logger.

**As an ECS system**: a logging system whose read-set is `to_remove`, `to_insert`, and other commit-time tables; whose write-set is the log columns. Runs in the DAG, possibly merged with `cleanup`. Fastest (the logging *is* part of the tick); most coupled (can't be unplugged without removing the system). Best for production simulators where logging is essential, not optional.

The three forms map to a familiar tradeoff: reusability vs. integration. Pick the form that matches the deployment context. For Bjorn's reference simulator: the ECS-system form is right — the simulator and the log are one architecture. For a library aimed at other simulators: the class form. For a one-off prototype: the module form.

The same structural pattern (triple-store, codebook, double-buffer) supports all three. The choice is *packaging*, not *design*.
