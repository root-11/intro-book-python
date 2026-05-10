# Solutions: 22 — Mutations buffer; cleanup is batched

## Exercise 1 — Implement the side buffers

```python
from dataclasses import dataclass, field

@dataclass
class CleanupBuffer:
    to_remove: list[int] = field(default_factory=list)
    to_insert_pos_x:   list[float] = field(default_factory=list)
    to_insert_pos_y:   list[float] = field(default_factory=list)
    to_insert_vel_x:   list[float] = field(default_factory=list)
    to_insert_vel_y:   list[float] = field(default_factory=list)
    to_insert_energy:  list[float] = field(default_factory=list)
    to_insert_id:      list[int]   = field(default_factory=list)

# tick boundary: clear everything
buffer = CleanupBuffer()
```

The insert side is *parallel column lists*, not a list of objects. The whole point of the simulator's SoA discipline is that "a row to insert" is six values across six lists with the same index — exactly like the live tables, just on the side.

For tighter packing, the insert lists could be pre-allocated numpy arrays with their own `n_pending` counter; for typical mutation rates (hundreds to thousands per tick), Python lists are plenty fast.

## Exercise 2 — Push from `apply_starve`

```python
def apply_starve(world: World, buffer: CleanupBuffer) -> None:
    """Read-set: world.energy, world.id, world.n_active.
       Write-set: buffer.to_remove (only)."""
    starvers = np.where(world.energy[: world.n_active] <= 0)[0]
    starver_ids = world.id[starvers]
    buffer.to_remove.extend(starver_ids.tolist())
```

The system does not call `world.delete_creature()`. It does not modify `world.energy` or `world.n_active`. It writes only to `buffer.to_remove` — the live world is untouched until cleanup. A diff between this version and the previous shows: every line that mutated a live column is gone; one `extend` line replaces all of them.

## Exercise 3 — Push from `apply_reproduce`

```python
THRESHOLD = 100.0

def apply_reproduce(world: World, buffer: CleanupBuffer, rng) -> None:
    """Read-set: world.energy, world.pos_x, world.pos_y, world.id, world.n_active.
       Write-set: buffer.to_insert_* (only). Parent's energy unchanged here;
                  splitting energy is a separate consideration handled by cleanup
                  or a follow-on system."""
    parents = np.where(world.energy[: world.n_active] > THRESHOLD)[0]
    if parents.size == 0:
        return
    n = parents.size
    # offspring inherit parent pos with tiny jitter
    jitter_x = rng.uniform(-0.1, 0.1, n).astype(np.float32)
    jitter_y = rng.uniform(-0.1, 0.1, n).astype(np.float32)
    new_ids  = world.next_ids(n)                              # see §24

    buffer.to_insert_pos_x.extend((world.pos_x[parents] + jitter_x).tolist())
    buffer.to_insert_pos_y.extend((world.pos_y[parents] + jitter_y).tolist())
    buffer.to_insert_vel_x.extend([0.0] * n)
    buffer.to_insert_vel_y.extend([0.0] * n)
    buffer.to_insert_energy.extend([world.energy[parents].mean()] * n)  # half-energy variant in §13
    buffer.to_insert_id.extend(new_ids.tolist())
```

Reproduction has no direct effect on the world during the tick. The offspring exist as parallel entries in the buffer lists. Cleanup will materialise them.

## Exercise 4 — Implement bulk cleanup

```python
def cleanup(world: World, buffer: CleanupBuffer) -> None:
    # 1. Removals (deletes first so freed slots can host inserts in §24's recycling)
    if buffer.to_remove:
        ids = np.unique(np.array(buffer.to_remove, dtype=np.uint32))
        slots = world.id_to_slot[ids]                     # see §23
        keep_mask = np.ones(world.n_active, dtype=bool)
        keep_mask[slots] = False
        n_keep = int(keep_mask.sum())
        for col_name in world.column_names:
            col = getattr(world, col_name)
            col[:n_keep] = col[: world.n_active][keep_mask]
        world.n_active = n_keep
        buffer.to_remove.clear()
        # update id_to_slot — see §23

    # 2. Insertions (one slice-write per column)
    n_inserts = len(buffer.to_insert_id)
    if n_inserts:
        new_n = world.n_active + n_inserts
        world.pos_x[world.n_active : new_n]  = buffer.to_insert_pos_x
        world.pos_y[world.n_active : new_n]  = buffer.to_insert_pos_y
        world.vel_x[world.n_active : new_n]  = buffer.to_insert_vel_x
        world.vel_y[world.n_active : new_n]  = buffer.to_insert_vel_y
        world.energy[world.n_active : new_n] = buffer.to_insert_energy
        world.id[world.n_active : new_n]     = buffer.to_insert_id
        world.n_active = new_n
        # update id_to_slot for new ids — see §23
        for lst in (buffer.to_insert_pos_x, buffer.to_insert_pos_y,
                    buffer.to_insert_vel_x, buffer.to_insert_vel_y,
                    buffer.to_insert_energy, buffer.to_insert_id):
            lst.clear()
```

Two bulk ops. The world is consistent at the end. Spot-check after a tick:

```python
assert len(set(world.id[: world.n_active].tolist())) == world.n_active     # no duplicates
```

## Exercise 5 — Compare cleanup forms

```python
import time, numpy as np
N, K = 1_000_000, 1_000

# Bulk cleanup: arr[keep_mask]
def bulk_cleanup(arr, indices_to_remove):
    keep = np.ones(len(arr), dtype=bool)
    keep[indices_to_remove] = False
    return arr[keep]

# Per-element swap_remove in a Python loop
def per_element_cleanup(arr, indices_to_remove):
    n = len(arr)
    for i in sorted(indices_to_remove, reverse=True):
        arr[i] = arr[n - 1]
        n -= 1
    return arr[:n]

arr = np.arange(N, dtype=np.int64)
indices = np.random.default_rng(0).choice(N, size=K, replace=False)

t = time.perf_counter()
for _ in range(100):
    bulk_cleanup(arr.copy(), indices)
print(f"bulk:        {(time.perf_counter()-t)*10:.2f} ms / call")

t = time.perf_counter()
for _ in range(100):
    per_element_cleanup(arr.copy(), indices.tolist())
print(f"per-element: {(time.perf_counter()-t)*10:.2f} ms / call")
```

Typical ratio at K=1000: bulk ~3-5× faster. At K=100,000: bulk ~5-10× faster (the boundary-crossing cost grows linearly with K for the per-element version, while the bulk form pays it once).

The bulk form is the right default for the Python edition. If you find yourself writing a per-element swap_remove loop inside cleanup, consider whether you have a buffer of indices in hand — if you do, use the mask.

## Exercise 6 — The dedup question

```python
# anti-pattern: bad! no dedup
buffer.to_remove.append(42)                      # apply_starve appends it
buffer.to_remove.append(42)                      # apply_disease appends it too
# both systems independently noticed creature 42 should die

# cleanup without np.unique:
slots = world.id_to_slot[buffer.to_remove]      # [slot_of_42, slot_of_42] — same slot twice
keep_mask = np.ones(world.n_active, dtype=bool)
keep_mask[slots] = False                         # idempotent — same slot zeroed twice is fine
```

For *removals via mask*, dedup happens to be implicit — assigning `False` to the same index twice is the same as once. So the boolean-mask form is robust to duplicate `to_remove` entries.

The risk is for **per-element swap_remove**: removing slot 42 once moves the last row into 42; removing it again moves the *new* last row into 42, deleting an *unintended* row. The cleanup function above protects via `np.unique` regardless of which deletion form is used.

## Exercise 7 — Tick-delayed visibility

```python
@dataclass
class World:
    age_in_ticks: np.ndarray = ...
    # ...

def end_of_tick(world):
    """Increment all live ages."""
    world.age_in_ticks[: world.n_active] += 1

# Tick 5: parent reproduces; offspring goes into to_insert with age_in_ticks=0
buffer.to_insert_age_in_ticks.append(0)
cleanup(world, buffer)                            # offspring now in live columns

end_of_tick(world)                                # offspring goes 0 → 1 (counts as full tick of life)

# Tick 6: age_in_ticks of newborn is 1 at start of tick
print(world.age_in_ticks[-1])                     # 1
```

The offspring did *not* live a partial tick of tick 5. It became part of the world *between* tick 5 and tick 6. Tick 6 is its first full tick; `end_of_tick` on tick 6 makes its `age_in_ticks` = 2.

Whether the increment happens before or after cleanup is a policy decision. The convention here: increment after cleanup, so newborns start at 0 and reach 1 at the end of their first tick. The choice should be written down once (in the simulator's contract) and applied consistently.

## Exercise 8 — A graphics pipeline analogy (stretch)

A double-buffered renderer:

- **Front buffer**: the framebuffer the display reads.
- **Back buffer**: the framebuffer the renderer writes.
- At vsync (the frame boundary), the buffers swap. The display now reads what the renderer just wrote; the renderer starts writing what the display previously had.

Map to the simulator:

| renderer concept             | simulator concept                          |
|------------------------------|--------------------------------------------|
| front buffer                 | live columns (`pos_x`, `pos_y`, ...) — what systems read |
| back buffer                  | `to_remove`, `to_insert_*` — where mutations queue |
| vsync (frame boundary)       | tick boundary                              |
| swap (front ↔ back)          | cleanup (apply queued changes to live columns) |

The shapes are identical. Both solve "many independent operations want to mutate shared state; how do they not step on each other?" by *accumulating in a side buffer and applying atomically at a boundary*. Database transactions, version-controlled file systems, audio engines (frame buffers for samples), and real-time-safety control systems (double-buffered set-points) all share this pattern.

A simulator that buffers its mutations is a simulator that has discovered transaction processing without naming it. Once you see the shape, every "atomic commit" boundary in software is a tick-boundary in disguise.
