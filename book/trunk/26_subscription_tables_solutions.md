# Solutions: 26 - Subscription tables, keyed by slot

## Exercise 1 - Build a slot-keyed subscription

```python
import numpy as np
HUNGER_THRESHOLD = 10.0

def classify_hungry(energy: np.ndarray) -> np.ndarray:
    return np.flatnonzero(energy < HUNGER_THRESHOLD)   # the slots that pass

def drive_hunger(hungry: np.ndarray, energy: np.ndarray, burn: float, dt: float) -> None:
    energy[hungry] -= burn * dt                        # one gather; slots index the columns directly
```

The subscription holds slots, so the hot loop indexes the columns with no redirection. Only the K subscribed rows are touched - the work is proportional to the subset, exactly as [§19](19_ebp_dispatch.md) promised.

## Exercise 2 - Key it by id instead, and time both

```python
import numpy as np, timeit
N = 1_000_000
rng = np.random.default_rng(0)
energy = rng.random(N).astype(np.float32)
slots  = rng.choice(N, size=N // 10, replace=False).astype(np.uint32)   # 10%, scattered

# id-keyed: hungry holds ids; id_to_slot resolves id -> slot
ids = rng.permutation(N)[:N // 10].astype(np.uint32)
id_to_slot = np.empty(N, dtype=np.uint32)
id_to_slot[ids] = slots

slot_keyed = lambda: energy[slots]                # one fancy index
id_keyed   = lambda: energy[id_to_slot[ids]]      # two fancy indexes

t_slot = timeit.timeit(slot_keyed, number=200) / 200
t_id   = timeit.timeit(id_keyed,   number=200) / 200
print(f"slot {t_slot*1e6:.0f} us   id {t_id*1e6:.0f} us   id/slot {t_id/t_slot:.2f}x")
```

```
slot 149 us   id 310 us   id/slot 2.08x      (Ryzen 9 270)
```

The id version's time goes into `id_to_slot[ids]` - a 100,000-element random read of a 4 MB array, a complete gather *before* the column gather starts. `id_to_slot` is 4 MB; one cache line is 64 bytes, so the subscribed ids are scattered across tens of thousands of cache lines. The slot key never touches it. Four-machine spread: `ebp_partition.py` claim C1, and the [§26 Measurements table](26_subscription_tables.md#measurements).

## Exercise 3 - Unsubscribe in O(1)

You need the [§23](23_index_maps.md) sparse set alongside `hungry`'s dense list: a `sparse` array, indexed by slot, holding each present slot's position in `dense`.

```python
def unsubscribe(dense, sparse, n, i):          # remove slot i in O(1), no scan
    p = sparse[i]
    last = dense[n - 1]
    dense[p] = last                            # backfill the hole
    sparse[last] = p
    sparse[i] = INVALID
    return n - 1
```

Without `sparse` you would scan `hungry` to find slot `i`'s position - O(K). The sparse set makes both "is `i` subscribed?" and "remove `i`" O(1), and it is the same index-map shape as `id_to_slot`, one level up: pointing into the subscription table instead of into the columns.

## Exercise 4 - Reindex on compaction

```python
# compaction relocates the live creatures; build the old -> new slot map once
order = np.argsort(~alive_mask, kind="stable")[:n_live]   # live slots, front to back
old_to_new = np.full(N, INVALID, dtype=np.uint32)
old_to_new[order] = np.arange(n_live, dtype=np.uint32)

# slot-keyed subscription: one bulk remap
hungry = old_to_new[hungry]

# id-keyed subscription: hungry (holding ids) is untouched; only id_to_slot is rewritten
id_to_slot[ids_in_new_order] = np.arange(n_live, dtype=np.uint32)
```

The slot-keyed reindex is K writes (one per subscribed slot) per cleanup. The id-keyed table needs *no* reindex - that is the id key's one advantage, and it buys it by paying the per-tick redirection from exercise 2. Across realistic cleanup intervals the slot key wins the trade (exercise 7). Both reindex passes are bulk numpy assignments; time them and you will find both are a small fraction of one tick, so the contest is decided by the per-tick hot loop, not the per-interval reindex.

## Exercise 5 - Dense vs scattered

```python
scattered = rng.choice(N, size=N // 10, replace=False).astype(np.uint32)
compacted = np.arange(N // 10, dtype=np.uint32)          # subscribed slots at the front

t_sc = timeit.timeit(lambda: energy[scattered], number=200) / 200
t_co = timeit.timeit(lambda: energy[compacted], number=200) / 200
print(f"scattered {t_sc*1e6:.0f} us   compacted {t_co*1e6:.0f} us   ratio {t_sc/t_co:.2f}x")
```

```
scattered 149 us   compacted 114 us   ratio 1.31x      (Ryzen 9 270)
```

Compaction buys ~1.3x on the Ryzen 9 270 - the gather's per-element machinery dominates and masks most of the cache-miss difference (the same reason numpy mutes the cache cliffs in [§27](27_working_set_vs_cache.md) and false sharing in [§33](33_false_sharing.md)). The reorder pass is ~1 ms at a million rows, so payback on locality *alone* is ~30 ticks, about a second at 30 Hz. (On memory-bound hardware the cache-miss difference dominates the gather overhead instead: the Pi 4 sees ~4.9x with payback ~7 ticks - see the [§26 Measurements table](26_subscription_tables.md#measurements).) That is acceptable - but on the desktop it is not the reason to compact. The same pass reclaims dead slots ([§24](24_append_only_and_recycling.md)), which the simulator owes regardless; the locality rides along free.

## Exercise 6 - The subscription that holds everyone

```python
all_slots = np.arange(N, dtype=np.uint32)
mask = np.ones(N, dtype=bool)

t_sub  = timeit.timeit(lambda: energy[all_slots], number=200) / 200   # gather all N
t_scan = timeit.timeit(lambda: energy * mask,     number=200) / 200   # branchless scan
print(f"gather-all {t_sub*1e6:.0f} us   scan {t_scan*1e6:.0f} us")
```

At 100% participation the gather is several times *slower* than the branchless scan: the gather still pays per-element machinery across all N, while `energy * mask` streams the whole contiguous column at full bandwidth. A subscription that holds everyone is a scan-all with extra bookkeeping. **Rule:** build a subscription only when it excludes most of the population. Its win is proportional to what it leaves out - the same crossover [§19](19_ebp_dispatch.md) measured, where presence loses to the bool mask once nearly everyone is a member.

## Exercise 7 - Two subscriptions, one entity (stretch)

The reindex cost scales with `S`, the number of subscriptions an entity sits in: each compaction remaps `S` dense arrays through `old_to_new`. The id key pays its redirection every tick regardless of `S`. Amortized over a cleanup interval `G` ticks:

- slot key: `S × reindex / G` per tick
- id key:   `S × redirect` per tick

Since one redirect (a full `id_to_slot[ids]` gather) is far more expensive than one reindex amortized over tens of ticks, the slot key wins at every `S` tested. From `ebp_partition.py` claim C4 (S=2, G=30) on the Ryzen 9 270, the id key costs ~30x the slot key per tick, amortized. The slot key's reindex burden grows with `S`, but it grows slowly and only at the cleanup cadence; the id key's redirection is paid in full every tick.
