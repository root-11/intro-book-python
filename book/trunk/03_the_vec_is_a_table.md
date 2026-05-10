# 3 — The `Vec` is a table

> *Concept node: see the [DAG](../../concepts/dag.md) and [glossary entry 3](../../concepts/glossary.md#3--the-vec-is-a-table).*

<p align="center"><img src="../illustrations/linear_algebra.jpg" alt="Linear algebra: Ax = b — a table is a matrix of columns indexed in lockstep" style="max-height: 300px; max-width: 100%;"></p>

A `list` in Python is a header object on the heap that stores three things: a length, a capacity (over-allocated by a small fraction), and a pointer to a contiguous run of `PyObject*` *pointers*. That last word is the lesson. The `list` does not contain your integers; it contains pointers to integer *objects*, each allocated separately on the heap. `lst[i]` reads a pointer from the contiguous run, then dereferences it to find the actual `PyLong` (28 bytes per int, 24 per float) somewhere else in memory.

If you used Python last week, this is the container you reached for, and it is the right shape for *some* problems. It is also the wrong shape for almost everything the trunk of this book teaches, which is "process all the rows of a table." A `list` of N rows-as-tuples is one big jump table sitting in front of N+10N small objects scattered across the heap. Walking it is pointer-chasing, not sequential reading.

A `numpy` array — `np.array(..., dtype=...)` — is the same three-things-on-the-heap shape, but the contiguous run holds *values*, not pointers. Ten million int64s in a numpy array is 80 MB of contiguous bytes; ten million ints in a list is 280 MB of `PyLong` objects plus 80 MB of pointers, scattered. `arr[i]` computes `base + i * 8` and reads — once. No object dereference. No allocation per element.

The trunk of this book uses two containers: `list` for the small bookkeeping (the names of your tables, the schedule of your systems) and `numpy.ndarray` for the rows. There are no `dict`s of objects, no class hierarchies, no `dataclasses` with `__slots__` for the things that need to scale. Not because they don't exist, but because every container that wraps a `PyObject` per row pays the pointer-chase tax on every read, and the rest of the book is about not paying that tax.

## The flip, measured

Take the same data — N rows, K integers per row — and lay it out four ways. The first two are what the official tutorial teaches. The third is a stdlib-only flip. The fourth is the disciplined endpoint.

| layout                                      | what it is                       |
|---------------------------------------------|----------------------------------|
| 1. `[(i, i+1, …) for i in range(N)]`        | list of tuples — AoS, default    |
| 2. `[[i, i+1, …] for i in range(N)]`        | list of lists — AoS, mutable inner |
| 3. `tuple([i+k for i in range(N)] for k …)` | tuple of lists — SoA, stdlib     |
| 4. `tuple(np.arange(...) for k in range(K))`| tuple of numpy columns — SoA, typed |

[`code/measurement/aos_vs_soa_footprint.py`](../../code/measurement/aos_vs_soa_footprint.py) builds each, in a fresh subprocess so RSS readings don't bleed, with N=1,000,000 and K=10. Values past the small-int cache so `PyLong` objects aren't shared singletons across rows. Three numbers per layout: peak RSS, construction time, time to sum column 0.

| layout                              |  RSS    | build  | sum c0 |
|-------------------------------------|--------:|-------:|-------:|
| list of tuples            (AoS)     | 437 MB  | 0.76 s | 30.0 ms |
| list of lists             (AoS)     | 499 MB  | 0.57 s | 26.8 ms |
| tuple of lists            (SoA)     | 383 MB  | 0.44 s |  3.7 ms |
| tuple of numpy int64 cols (SoA)     |  99 MB  | 0.20 s |  0.4 ms |

> [!NOTE]
> Measured on this author's machine; reproduce on yours with `uv run code/measurement/aos_vs_soa_footprint.py`. Order-of-magnitude is the durable claim. Numbers will shift with K, N, value range, and CPython version, but the shape — that going SoA shrinks footprint and that going to typed columns collapses both footprint and per-column-op time — is stable across machines.

Three things to notice.

**The mutable AoS is worse than the immutable AoS.** Replacing the inner tuples with lists costs ~60 MB of additional list-header overhead at this scale. The "list of lists" pattern is the most-taught layout in introductory Python and the most-expensive one in this comparison.

**The stdlib SoA flip is already worth doing.** Tuple-of-lists is the same code an intermediate Python programmer might write without ever touching numpy. It saves ~12% memory over the canonical AoS, builds 1.7× faster, and — the surprise — sums column 0 about 8× faster. The win comes from walking *one* contiguous list of 1M `PyLong` pointers instead of walking 1M tuple objects and dereferencing through each one to reach `row[0]`. No numpy required.

**The typed-numpy step is the order-of-magnitude move.** Going from SoA-stdlib to SoA-numpy shrinks footprint another ~4× and speeds column-sum another ~10×. The `PyLong` dereference is gone; the bytes are typed; the inner loop is C, not Python. This is the layout that the simulator (§11+) and every system after it depends on.

## The Python-default trap, named

The official tutorial is not wrong. It's optimised for *teaching the language*, not for teaching layout. The path it teaches looks like this:

1. Make a class for the row.
2. Put instances in a list.
3. Reach for `dataclass` when the class gets noisy.
4. Reach for `__slots__` when memory pressure shows up.

Each step is a local improvement and a global trap. Step 1 commits you to AoS. Step 2 puts pointers between the rows. Step 3 makes the AoS more ergonomic. Step 4 saves a per-instance `__dict__` but does nothing about the fundamental shape — every row is still its own heap object reached through a pointer. The `__slots__` win is real and small; the SoA win is the same data costing 4-5× less memory, and you don't need a class at all.

There is no such thing as a cost-free abstraction. Every pointer has a cost, and in a `list` of rows that cost multiplies linearly with the row count. The four-step path stacks pointers: an outer list of N row-pointers, each row pointing to K field objects, each field a separately allocated value somewhere else on the heap. `__slots__` removes one layer (the per-instance `__dict__`); the SoA flip removes the rest. The next several phases of this book teach the alternative.

## Exercises

1. **Pointer-chase or value-read.** Print `sys.getsizeof(0)`, `sys.getsizeof(1000)`, `sys.getsizeof(10**100)`. Note that even a small Python int costs 28 bytes. Now print `np.array([0, 1000, 10**18], dtype=np.int64).nbytes`. Three int64s = 24 bytes, and there are no per-element headers.
2. **The interning trap.** Repeat exercise 1 with values 0 and 1, then again with values 257 and 1000. Use `id()` to confirm that `[0] * 1_000_000` shares one `PyLong` object across all positions, but `[1000 + i for i in range(1_000_000)]` does not. The "list of small ints is cheap" intuition only holds inside CPython's small-int cache `[-5, 256]`.
3. **Capacity vs length.** Build `lst = []`. In a loop, append 0..1000 and print `len(lst)` and `sys.getsizeof(lst)` after each step. Observe the over-allocation pattern — `list` grows in chunks, like `Vec::push`, but the chunks are CPython implementation detail (currently `~1.125 ×` growth).
4. **Run the §3 exhibit.** `uv run code/measurement/aos_vs_soa_footprint.py`. Read the output. The sum-c0 column matters: even if you ignore the memory line, the column-sum cost gap between layouts 1 and 4 is two orders of magnitude on the same data.
5. **The dict trap.** Build `d = {i: i*i for i in range(1_000_000)}` and time looking up 100,000 random keys. Build `arr = np.arange(1_000_000) ** 2` and time the same access pattern via `arr[idx]`. Note that you have replaced "look up by integer" with "index by integer," and the structures cost different amounts.
6. **swap-remove vs remove.** Build `lst = list(range(1_000_000))`. Time removing 100 elements from the middle by `lst.pop(500_000)` (slow — every pop shifts ~half the list). Time the equivalent via `lst[i] = lst[-1]; lst.pop()`. Note the orders-of-magnitude difference. This trick will earn its keep at [§21](21_swap_remove.md).
7. *(stretch)* **Read your own array.** Use `np.frombuffer(arr.tobytes(), dtype=np.int64)` and confirm that `arr.data.tobytes()` is exactly `arr.size * 8` bytes long. The bytes you would write to disk *are* the bytes already in memory. This is what [§36 — persistence](36_persistence_is_serialization.md) means by "tables serialise themselves."

Reference notes in [03_the_vec_is_a_table_solutions.md](03_the_vec_is_a_table_solutions.md).

## Applied reference

If you want to see this discipline carried through a real piece of code, read [`.archive/simlog/logger.py`](../../.archive/simlog/logger.py). It is a 700-line columnar logger that parks dict payloads into pre-allocated numpy columns, with a double-buffered design that lets the simulation write to one buffer while a background thread dumps the other to disk. The book does not require you to read it now. It's the destination this chapter and the next several point at.

## What's next

[§4 — Cost is layout, and you have a budget](04_cost_and_budget.md) takes the layout reasoning into per-tick territory: how many bytes can you actually move per tick on your machine, and what does that buy you in entities? After that, [§5 — Identity is an integer](05_identity_is_an_integer.md) is where the through-line simulator gets its first concrete shape.
