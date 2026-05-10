# Solutions: 26 — Hot/cold splits

## Exercise 1 — Audit access patterns

A typical simulator audit:

| system            | reads                              | writes              |
|-------------------|------------------------------------|---------------------|
| `food_spawn`      | `food`, RNG, `spawner_params`      | `food`              |
| `motion`          | `vel_x`, `vel_y`, `dt`             | `pos_x`, `pos_y`    |
| `next_event`      | `pos_x`, `pos_y`, `food`           | `pending_event`     |
| `apply_eat`       | `pending_event`                    | `energy_delta`, `to_remove` (food) |
| `apply_reproduce` | `pending_event`, `energy`          | `to_insert_*`       |
| `apply_starve`    | `energy`                           | `to_remove`         |
| `cleanup`         | buffers + every column             | every column        |
| `inspect`         | every column                       | nothing             |

**Hot columns**: `pos_x`, `pos_y`, `vel_x`, `vel_y`, `energy` — read by 4-5 systems per tick.
**Cold columns**: `birth_t`, `id`, `gen` — read only by cleanup and logging.

`food` and `pending_event` are their own tables; not part of the creature split.

## Exercise 2 — Build the split, organisationally

```python
class CreatureHot:
    def __init__(self, capacity):
        self.pos_x  = np.zeros(capacity, dtype=np.float32)
        self.pos_y  = np.zeros(capacity, dtype=np.float32)
        self.vel_x  = np.zeros(capacity, dtype=np.float32)
        self.vel_y  = np.zeros(capacity, dtype=np.float32)
        self.energy = np.zeros(capacity, dtype=np.float32)

class CreatureCold:
    def __init__(self, capacity):
        self.birth_t = np.zeros(capacity, dtype=np.float64)
        self.id      = np.zeros(capacity, dtype=np.uint32)
        self.gen     = np.zeros(capacity, dtype=np.uint32)

class Creatures:
    def __init__(self, capacity):
        self.hot  = CreatureHot(capacity)
        self.cold = CreatureCold(capacity)
        self.n_active = 0
```

Both tables share the same capacity and `n_active`. Cleanup must apply rearrangement to *both* in lockstep.

## Exercise 3 — Time motion at 1M creatures

```
unsplit numpy SoA:   ~0.6 ms per call
split numpy SoA:     ~0.6 ms per call
```

Identical. Each numpy column is already its own contiguous buffer; the split changed no bytes' addresses, only their namespace. The bandwidth win the chapter's prose describes for Rust does not materialise in Python+numpy because *the layout is already optimal*.

This is the chapter's main point: the split is organisational, not bandwidth-saving, in the Python edition.

## Exercise 4 — Time motion in numpy structured-array form

```python
import numpy as np, time
n = 1_000_000
dtype = np.dtype([('pos_x', 'f4'), ('pos_y', 'f4'),
                  ('vel_x', 'f4'), ('vel_y', 'f4'),
                  ('energy', 'f4'), ('birth_t', 'f8'),
                  ('id', 'u4'), ('gen', 'u4')])
arr = np.zeros(n, dtype=dtype)
arr['vel_x'] = 1.0; arr['vel_y'] = 1.0

t = time.perf_counter()
for _ in range(100):
    arr['pos_x'] += arr['vel_x'] / 30.0
    arr['pos_y'] += arr['vel_y'] / 30.0
print(f"structured array motion: {(time.perf_counter()-t)*10:.2f} ms/call")
```

```
SoA columns:       0.62 ms
structured array:  4.93 ms
ratio:             8×
```

The structured array is **8× slower** than separate columns. Why? Each `arr['pos_x']` returns a *strided* view — it walks the buffer with stride 32 bytes (the size of the full row), reading 4-byte `pos_x` values every 32 bytes. The prefetcher pulls 32 bytes per row even though only 4 are used; the remaining 28 are wasted bandwidth (and the cold fields, especially `birth_t` at 8 bytes, are dragged through cache anyway).

This is the AoS-pattern in numpy clothing. The split *would* help here — splitting the structured array into two structured arrays, hot and cold, reduces the stride from 32 to 20 (40% bandwidth saving). But the simpler fix is to leave the structured-array world and use one numpy column per field, which is what the simulator does.

## Exercise 5 — Cleanup must touch both

```python
def cleanup(world: Creatures, to_remove: list[int]):
    if not to_remove: return
    ids = np.unique(np.array(to_remove, dtype=np.uint32))
    slots = world.id_to_slot[ids]
    keep_mask = np.ones(world.n_active, dtype=bool)
    keep_mask[slots] = False

    # Apply to both hot and cold columns in lockstep
    for col_name in ("pos_x", "pos_y", "vel_x", "vel_y", "energy"):
        col = getattr(world.hot, col_name)
        col[:keep_mask.sum()] = col[:world.n_active][keep_mask]
    for col_name in ("birth_t", "id", "gen"):
        col = getattr(world.cold, col_name)
        col[:keep_mask.sum()] = col[:world.n_active][keep_mask]
    world.n_active = int(keep_mask.sum())
```

The same keep_mask, applied to every column of every sub-table. Missing one column → misalignment between hot and cold (§9 bug across tables).

## Exercise 6 — A bad split

```python
# anti-pattern: bad! energy is hot, but we put it in cold
class CreatureHot:  pos_x, pos_y, vel_x, vel_y      # missing energy!
class CreatureCold: energy, birth_t, id, gen        # energy stored here

# motion now reads from cold:
def motion_bad(hot, cold, dt):
    hot.pos_x += hot.vel_x * dt
    hot.pos_y += hot.vel_y * dt
    # apply_starve still reads cold.energy every tick — extra column-access overhead
```

In numpy SoA the timing penalty is small (`cold.energy` is still its own contiguous array). In *structured-array* layout the penalty would be real — `energy` would be at stride-32-bytes inside the cold record. The categorical error is *naming hot fields cold*: code that follows the convention "cold means rarely read" will draw wrong conclusions about which fields can be omitted from inspection-time reads, persistence, etc.

The lesson: hot/cold is not about names; it is about *access frequency*. Audit, then split.

## Exercise 7 — The all-fields case (stretch)

```python
def serialize_world(world):
    """Read every column to disk via np.savez."""
    np.savez("snapshot.npz",
             pos_x  = world.hot.pos_x[: world.n_active],
             pos_y  = world.hot.pos_y[: world.n_active],
             vel_x  = world.hot.vel_x[: world.n_active],
             vel_y  = world.hot.vel_y[: world.n_active],
             energy = world.hot.energy[: world.n_active],
             birth_t = world.cold.birth_t[: world.n_active],
             id     = world.cold.id[: world.n_active],
             gen    = world.cold.gen[: world.n_active])
```

The split's overhead here is the function signature — eight columns spread across two namespaces — but the *runtime cost* is identical to the unsplit version: every column gets read once, written to disk once. The serialiser does not benefit from the split.

This is a fine tradeoff: serialisation runs once per checkpoint (every minute? every hour?), not once per tick. Paying eight extra characters in the function call to keep the inner loop's namespace clean is cheap. The split is earned by the *hot path*, not the cold path.
