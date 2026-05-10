# Solutions: 29 — The wall at 10K → 1M

These exercises ask you to *find the wall*, not to remove it abstractly. The fixes are techniques you have from §26-§28; the diagnostic is the new content.

## Exercises 1 & 2 — Calibration and scale-up

```sh
time python my_sim.py --n 10000 --ticks 1000
time python my_sim.py --n 1000000 --ticks 100
```

Both runs do **the same total entity-ticks** (10⁷). The wall-clock ratio is the diagnostic:

| ratio        | meaning |
|--------------|---------|
| ~1×          | Inner loop is bandwidth-bound at numpy speed across both scales. No wall. |
| 2-3×         | L2 → L3 / L3 → RAM transitions. Working set spilled; per-element cost rose by ~3×. Hot/cold splits help. |
| 10-30×       | A non-numpy hot loop scaled with N rather than amortising. Use cProfile to find it. |
| 100×+        | Quadratic blow-up: a per-creature operation that scans the whole table. Use the index map. |

A 1.5-3× wall is normal and the chapter's techniques close it. A 100× wall is a structural bug; nothing in this chapter fixes it short of recognising it.

## Exercise 3 — Profile with cProfile

```sh
python -m cProfile -o profile.out -s cumulative my_sim.py
python -m pstats profile.out
> sort cumulative
> stats 30
```

Typical hot-list culprits at 1M:

- `list.append`  — `to_insert.append` in a loop; pre-size to fix.
- `numpy.ndarray.__getitem__`  — accidental Python-level fancy indexing.
- `<dict iteration>`  — id lookup via `dict.get` per creature when an `id_to_slot` array would be O(1).
- One named system that wasn't supposed to be hot but is.

`cProfile` sees Python-level calls. Numpy primitives show up as one C-call entry (`numpy.add` or similar) regardless of how many elements they process. For numpy-internal hot spots, use py-spy.

## Exercise 4 — Profile with py-spy

```sh
pip install py-spy
py-spy record -o flame.svg -- python my_sim.py
# then open flame.svg in a browser
```

py-spy samples the C stack, which surfaces numpy hot spots that cProfile lumps together. Typical findings:

- A `np.where(...)` over a column that could be a presence table.
- A bool-mask reduction (`(arr > 0).sum()`) that compiles to a slow path on int8.
- A `np.argsort` inside the tick that should run every 10 ticks (§28 cadence).

The flame graph's *width* is wall time. Widest function is your bottleneck.

## Exercise 5 — Pre-size cleanup buffers

```python
# Before
class CleanupBuffer:
    to_insert: list[CreatureRow] = field(default_factory=list)

# After
class CleanupBuffer:
    def __init__(self, capacity: int):
        self.to_insert_pos_x = np.zeros(capacity, dtype=np.float32)
        self.to_insert_pos_y = np.zeros(capacity, dtype=np.float32)
        # ...
        self.n_inserts = 0

    def add_insert(self, pos_x, pos_y, ...):
        i = self.n_inserts
        self.to_insert_pos_x[i] = pos_x
        self.to_insert_pos_y[i] = pos_y
        self.n_inserts += 1
```

The Python list `append` is amortised O(1) but each doubling is an N-byte copy. At 10K inserts per tick that's a 80K-byte copy every few ticks (negligible). At 100K inserts per tick the doublings happen often enough to be one of the hottest calls in the profile. Pre-sized arrays remove the doubling entirely.

## Exercise 6 — Hot/cold split

In pure numpy SoA (where every column is its own array), splitting the row organisationally does *not* change the profile — the bytes were already separated. §26's framing applies: the split is naming, not bandwidth.

If the simulator uses *numpy structured arrays* (one combined dtype for the whole row), the split shows up immediately. Motion's `arr['pos_x'] += arr['vel_x'] * dt` runs at structured-array stride; splitting into `hot_arr['pos_x'] += hot_arr['vel_x'] * dt` runs at SoA speed. Expect ~8× improvement at 1M creatures.

## Exercise 7 — Use index maps

```python
# Before
def find_creature(world, target_id):
    return np.where(world.id == target_id)[0]   # O(N) per call

# After  
def find_creature(world, target_id):
    return int(world.id_to_slot[target_id])     # O(1) per call
```

For 100 lookups per tick at N=1M, the linear-scan version costs ~100 × 5 ms = 500 ms per tick (orders-of-magnitude over budget). The index-map version costs ~100 × 50 ns = 5 µs.

The 100,000× speedup vanishes from the profile after this fix. The id_to_slot maintenance in cleanup is paid once per cleanup pass, in the form of one bulk numpy assignment — invisible in the profile.

## Exercise 8 — The pandas wall, hands-on

```python
import pandas as pd, numpy as np, sqlite3, time
n = 5_000_000

# pandas
df = pd.DataFrame({f"col{i}": np.random.rand(n).astype(np.float64) for i in range(10)})
pandas_mb = df.memory_usage(deep=True).sum() / 1e6
print(f"pandas:   {pandas_mb:.0f} MB ({n} rows × 10 cols × float64)")

# numpy float32
cols = {f"col{i}": np.random.rand(n).astype(np.float32) for i in range(10)}
numpy_mb = sum(c.nbytes for c in cols.values()) / 1e6
print(f"numpy f32: {numpy_mb:.0f} MB")

# sqlite
conn = sqlite3.connect(":memory:")
conn.execute(f"CREATE TABLE t (id INTEGER PRIMARY KEY, " + ", ".join(f"c{i} REAL" for i in range(10)) + ")")
# ... insert and measure ...
```

Typical results:

| layout              | memory |  comment |
|---------------------|-------:|----------|
| pandas (float64)    |  400 MB | default — float64 inflates the bytes |
| numpy float32 cols  |  200 MB | half the bytes per value |
| sqlite (disk)       |  ~150 MB on disk | typed, indexed, queryable |

If queries are random by primary key: sqlite wins (the index makes it O(log N) per lookup, ~830K-900K lookups/sec on this hardware).
If queries are full-column reductions: numpy wins (one bandwidth-bound pass).
If queries are joins or groupbys: it depends — for small results, pandas/numpy; for large results, sqlite or polars.

The decision is the access pattern. Default to numpy SoA when the data fits RAM and queries are scans. Default to sqlite when queries are point lookups or the data exceeds RAM.

## Exercise 9 — Find one new wall (stretch)

A specific finding pattern:

1. Run the simulator at N=1M and at N=2M.
2. If the 2M version takes more than 2× the 1M version's time, you have a non-linear cost.
3. Profile both with py-spy.
4. Compare flame graphs. The function whose share of total time grew between the two runs is the suspect.
5. Map the suspect to one of the §26-§28 techniques. Fix it. Re-profile.

In practice, the first one or two passes find the easy walls. Subsequent passes find subtler ones — a `np.unique` inside cleanup that scales O(K log K) on the unique count, a sort that runs on a slowly-changing key, a Python-level `for` loop over a list that should have been a numpy primitive. Every fix is a chapter you have read. The diagnostic is the constant.
