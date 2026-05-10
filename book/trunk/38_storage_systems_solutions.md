# Solutions: 38 — Storage systems: bandwidth and IOPS

## Exercise 1 — Measure your bandwidth

```sh
dd if=/dev/zero of=/tmp/test bs=1M count=1024 oflag=direct
# example output: "1073741824 bytes (1.1 GB) copied, 1.42 s, 757 MB/s"
```

Typical 2026 hardware:

| storage           | sustained sequential write |
|-------------------|---------------------------:|
| NVMe Gen3         |       1-2 GB/s             |
| NVMe Gen4         |       3-5 GB/s             |
| NVMe Gen5         |       5-12 GB/s            |
| SATA SSD          |       400-550 MB/s         |
| spinning HDD      |       100-200 MB/s         |

Read the number off your machine; that's your *bandwidth ceiling*. No workload writes faster than this.

## Exercise 2 — Measure your IOPS

```python
import os, time
path = "/tmp/iops_test"
n_ops = 10_000
chunk = b"X" * 4096                                # 4 KB

with open(path, "wb") as f:
    t = time.perf_counter()
    for _ in range(n_ops):
        f.write(chunk)
        f.flush()
        os.fsync(f.fileno())                       # force durable write
    elapsed = time.perf_counter() - t

print(f"{n_ops/elapsed:,.0f} IOPS")
```

Typical: 100-2000 fsync-IOPS on consumer NVMe. The IOPS rate is *much* lower than the bandwidth number suggests because every `fsync` blocks until the SSD's internal buffers are durably committed — that's microseconds per call, even though the data itself is tiny.

Without `fsync`, raw write IOPS to a file in the page cache can be 100K+ per second. Durable IOPS (the kind a database needs) are 10-100× lower.

## Exercise 3 — Batched vs unbatched

```python
import time, os
n = 1_000_000
data = b"X" * 32

# 1M separate writes
with open("/tmp/many.bin", "wb") as f:
    t = time.perf_counter()
    for _ in range(n): f.write(data)
print(f"1M writes:  {(time.perf_counter()-t)*1000:.0f} ms")

# 1 bulk write
with open("/tmp/one.bin", "wb") as f:
    t = time.perf_counter()
    f.write(data * n)
print(f"1 bulk write: {(time.perf_counter()-t)*1000:.0f} ms")
```

Typical: many-writes ~200-500 ms; one bulk write ~20-50 ms. The Python `for` loop's per-call cost dominates the actual disk traffic at this size.

If you add `f.flush()` and `os.fsync()` after every write, the gap widens to **1000-5000×** — the bulk version still pays one fsync, the many-writes version pays a million.

This is the simlog's batching argument made concrete. Per-mutation writes are infeasible; batched writes are bandwidth-bound and fast.

## Exercise 4 — SQLite throughput, three forms

```python
import sqlite3, time

conn = sqlite3.connect(":memory:")
conn.execute("CREATE TABLE t (a INTEGER, b INTEGER, c INTEGER)")
rows = [(i, i*2, i*3) for i in range(1_000_000)]

# Form 1: one INSERT per row, separate transactions
t = time.perf_counter()
for r in rows: conn.execute("INSERT INTO t VALUES (?, ?, ?)", r)
conn.commit()
print(f"per-row INSERT: {(time.perf_counter()-t)*1000:.0f} ms")

# Form 2: executemany inside a single transaction
conn.execute("DELETE FROM t")
t = time.perf_counter()
with conn:
    conn.executemany("INSERT INTO t VALUES (?, ?, ?)", rows)
print(f"executemany:    {(time.perf_counter()-t)*1000:.0f} ms")

# Form 3: INSERT-FROM-SELECT over a separate table
conn.execute("DELETE FROM t")
conn.execute("CREATE TABLE source (a INTEGER, b INTEGER, c INTEGER)")
conn.executemany("INSERT INTO source VALUES (?, ?, ?)", rows)
t = time.perf_counter()
conn.execute("INSERT INTO t SELECT * FROM source")
print(f"INSERT FROM SELECT: {(time.perf_counter()-t)*1000:.0f} ms")
```

Typical:

```
per-row INSERT:        ~20-30 s     (50-100K rows/sec)
executemany:             1-2 s      (500K-1M rows/sec)
INSERT FROM SELECT:    100-300 ms   (3-10M rows/sec)
```

Three orders of magnitude span. The difference: the per-row form pays SQL parsing, locking, and (without a transaction) per-row commit overhead on every call. `executemany` parses once, batches the per-row work. INSERT-FROM-SELECT keeps everything inside SQLite's engine; no Python boundary crossing.

For the simulator's exporter to SQLite (after a run), INSERT-FROM-SELECT is the right shape — get the data into an in-memory SQLite table first (via column-direct bulk writes), then have SQLite move it to the on-disk table.

## Exercise 5 — Run the SQLite warm-disk exhibit

```sh
uv run code/measurement/sqlite_performance_test.py
```

The script requires an external CSV file that the repo doesn't ship; you'd populate it from your own data first. The expected pattern when run:

```
backing             lookups/sec
:memory:               ~900,000
local NVMe (warm)      ~830,000
local NVMe (cold)      ~50-200K  (after page-cache drop)
```

The cold/warm gap is the disk's real cost — once pages are in the OS page cache, "disk" is RAM. The cold reads pay actual seek time; the warm reads pay only SQLite's dispatch overhead.

For most simulator workloads, this means: a recently-written log file behaves like memory. Reading it weeks later, after the OS has evicted its pages, behaves like a disk. *Cold I/O is the wall; warm I/O is not.*

## Exercise 6 — Compute your tick budget

```
30 Hz tick = 33 ms = 33,000 µs
1,000 mutations per tick = 33 µs/mutation budget

NVMe latency per random read: ~100 µs   → too slow without batching (would consume 3 ticks/mutation)
Memory access:                ~100 ns   → fits 330 per mutation slot

Verdict: each mutation cannot afford an individual disk read.
Must batch — one batched write per tick → 1 IOP per tick → ~100 µs → ~0.3% of budget.
```

The batching pattern (§22 cleanup amortising disk writes) is what makes the simulator durable at 30 Hz. Without it, every mutation would block on disk; one tick would take seconds.

## Exercise 7 — The pandas-OOM-to-sqlite migration

```python
import pandas as pd, sqlite3, time, numpy as np

n = 5_000_000
df = pd.DataFrame({f"col{i}": np.random.rand(n).astype(np.float64) for i in range(10)})
print(f"pandas memory: {df.memory_usage(deep=True).sum() / 1e6:.0f} MB")
# ~400 MB

# Migrate to SQLite
conn = sqlite3.connect("/tmp/data.db")
df.to_sql("t", conn, index=False, if_exists="replace")
del df

# Query against pandas (if you can still hold it in memory)
# ... vs query against SQLite
t = time.perf_counter()
result_sqlite = conn.execute("SELECT col0, col1 FROM t WHERE col0 > 0.99").fetchall()
print(f"SQLite query: {(time.perf_counter()-t)*1000:.0f} ms, {len(result_sqlite)} rows")
```

The migration is one `df.to_sql(...)` call. After it, the data lives in a typed indexed disk-backed table that supports relational queries without consuming RAM. Query times: ~10-100 ms for a million-row filter, similar to pandas warm.

The pandas form is faster at *unrestricted in-memory operations* (a join, a groupby). The SQLite form is faster at *random point queries with indices* and doesn't blow up on memory. Pick the tool that matches the workload. For analyst-style queries against simulation output: SQLite is the safer default.

## Exercise 8 — A second storage system (stretch)

```python
import time, urllib.request

# Latency to a remote: round-trip per read
url = "https://your-network-filesystem/path/file.bin"

t = time.perf_counter()
for _ in range(100):
    with urllib.request.urlopen(url) as r:
        r.read(1024)
print(f"100 sequential reads: {(time.perf_counter()-t)*1000:.0f} ms")
# typical: 10-50 seconds (100-500 ms per round-trip)

# Concurrent reads via aiohttp or httpx
# (skipping the implementation — the point is the order-of-magnitude difference)
# concurrent 100 reads: ~500 ms-2s — bounded by aggregate bandwidth
```

The bandwidth-delay product is the bound. For 100 ms latency and 1 KB reads, throughput per connection is 10 KB/s. Concurrency multiplies that — 100 concurrent connections give 1 MB/s aggregate. For a simulator that depends on a remote storage system, *concurrency is the only knob*; you can't make the latency smaller.

This is why distributed simulations partition the world by location (each node owns its region) and only cross the boundary at the edges. Per-tick remote reads are infeasible past a handful per tick; per-snapshot remote reads (one large transfer at checkpoint time) are fine.
