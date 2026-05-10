# Solutions: 17 — Presence replaces flags

## Exercise 1 — Add a `hungry` table

```python
import numpy as np

class World:
    def __init__(self, n):
        self.energy = ...                                 # existing
        self.ids    = np.arange(n, dtype=np.uint32)
        self.hungry = np.empty(0, dtype=np.uint32)        # the new presence table
```

Empty array. No bool column, no flag, no boolean attribute. The world starts with zero hungry creatures.

## Exercise 2 — Populate it

```python
HUNGER_THRESHOLD = 10.0

def classify_hunger(energy: np.ndarray, ids: np.ndarray) -> np.ndarray:
    return ids[energy < HUNGER_THRESHOLD]
```

One numpy line. The system's read-set is `energy` and `ids`; the write-set is whatever the caller assigns the result to. Each tick:

```python
world.hungry = classify_hunger(world.energy, world.ids)
```

## Exercise 3 — Build the flag version

```python
is_hungry = np.zeros(N, dtype=bool)              # one byte per creature

def classify_flag(energy, is_hungry):
    is_hungry[:] = energy < HUNGER_THRESHOLD     # in-place, broadcasts
```

Same data, parallel column shape. Length N regardless of how many are actually hungry; one byte per creature wasted on the false ones.

## Exercise 4 — Build the AoS version

```python
from dataclasses import dataclass

@dataclass(slots=True)
class Creature:
    energy: float
    is_hungry: bool = False

creatures = [Creature(float(e), False) for e in energy]

def classify_aos(creatures):
    for c in creatures:
        c.is_hungry = c.energy < HUNGER_THRESHOLD
```

The Python tutorial canonical version. Every consumer of "is this creature hungry" reads `c.is_hungry` and pays for `getattr` on every access.

## Exercise 5 — Time all three at 1M creatures

```
classify presence:  1.41 ms  (100K hungry of 1M)
classify flag:      0.05 ms
classify AoS:      13.1  ms
```

| layout    | classify time | comment |
|-----------|--------------:|---------|
| flag (numpy bool column)    |  0.05 ms | fastest — pure C bulk op |
| presence (numpy id array)   |  1.41 ms | extra step: scan + boolean indexing |
| AoS (Python loop)           | 13.1  ms | interpreter-bound; ~250× slower than flag |

Two surprises:

- **Flag is *faster* than presence at the classification step.** Building the boolean mask alone is cheap; building the *list of ids that pass the mask* needs an extra pass to materialise the index array. For one-shot classification, the flag column wins.
- **AoS is ~10× slower than the worst numpy version.** That's the cost of the per-element interpreter loop, exactly as §13 promised.

The presence advantage shows up *downstream* — at the consumer step, not the classifier. Next exercise.

## Exercise 6 — The membership query

```python
def is_hungry_p(hungry: np.ndarray, target_id: int) -> bool:
    return bool(np.any(hungry == target_id))         # O(K)

def is_hungry_f(is_hungry: np.ndarray, slot: int) -> bool:
    return bool(is_hungry[slot])                     # O(1)
```

```
flag:     ~50 ns
presence: O(K) ms — proportional to len(hungry)
```

The flag wins for *single-creature lookup* — direct array indexing is faster than scanning. Presence wins for *whole-table operations* (count, iterate the hungry set) because there is no scanning of the false rows. The right answer depends on the query pattern; the wrong reflex is to assume one is always faster than the other.

[§23 — index maps](23_index_maps.md) is the fix that makes presence O(1) for membership too: an `id_to_slot` array lets you check membership in one read. With the index map, presence beats flag on *every* operation that matters in the simulator.

## Exercise 7 — "How many are hungry?"

```
count presence:  30 ns         (len(hungry))
count flag:     204 µs         (int(is_hungry.sum()))
count AoS:        10 ms         (sum(1 for c in creatures if c.is_hungry))
```

| version  | time at 1M | regime |
|----------|-----------:|--------|
| presence |       30 ns | constant — `len()` is O(1) |
| flag     |      204 µs | bandwidth-bound numpy reduction |
| AoS      |       10 ms | interpreter-bound Python loop |

Presence is **6800× faster** than flag here. Why? `len(hungry)` is a single Python attribute read on a numpy array — it does not iterate. The flag version *has* to iterate (sum a million booleans). The AoS version pays for it 50,000× over.

This is where presence pays back. The classification cost is paid once per tick; the count is read by every system that needs to know "how many are hungry?" If even one consumer asks for the count per tick, the presence form pays back its classification overhead instantly. Most simulators have several such consumers (UI display, log entry, decision in the food-spawn policy, etc.).

## Exercise 8 — Persist both (stretch)

```python
np.save("is_hungry.npy", is_hungry)               # 1 MB (1 byte × 1M)
np.save("hungry.npy",    hungry)                  # ~400 KB (4 bytes × 100K)

np.savez_compressed("is_hungry.npz", is_hungry)   # ~120 KB (compresses runs of zeros)
np.savez_compressed("hungry.npz",    hungry)      # ~395 KB (already dense)
```

Uncompressed: presence is 2.5× smaller. Compressed: flag becomes smaller because 90% of its bytes are zeros that compress almost to nothing; presence is essentially incompressible random integers.

This reverses the conclusion at storage time but not at I/O time: writing the flag column requires reading 1 MB of bytes from RAM to compress, while writing the presence array reads 400 KB. *In RAM, presence wins; on disk after compression, flag wins (sometimes); at write time, presence wins.* Pick the layout that matches your dominant access pattern; persistence is one consideration among several.

For the simulator's case — frequent in-memory operations, infrequent persistence — presence is the right default. For an archive that's mostly written once and read rarely, the trade is closer.
