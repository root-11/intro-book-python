# Python edition — chapter-by-chapter analysis

Same book as the Rust edition. Same 44 sections, same DAG, same simulator
through-line. The variation is per-chapter commentary on **what Python's
OOP-AoS defaults push the reader into, and why ECS/EBP wins anyway**.

The thesis the edition has to carry, in three words each:
- **EBP** — process more efficiently (operations grouped over arrays).
- **Systems** — more extendible (data-oriented composition over class graphs).
- **Arrays** — smaller memory footprint (typed columns over object graphs).

What carries this edition is the **evidence**. Each chapter below names the
Python-default failure to call out, and the exhibit that proves the win.
Exhibits in `~/code/intro-py/` are tagged `[have]`; missing exhibits are
tagged `[write]` with one line on what they should measure.

---

## Part 1: Foundation

| § | Chapter | Python-default to call out | Exhibit |
|---|---------|----------------------------|---------|
| 1 | The machine model | "The interpreter abstracts the machine." It abstracts *some* of it; constant factors and rates leak through. | `try_except.py` [have] — cost depends on hit-rate, not on syntax. `string_methods.py` [have] — `%`-format beats f-string narrowly; idiom ≠ speed. |
| 2 | Numbers and how they fit | "`int` is `int`, `float` is `float`, size doesn't matter." Python ints are PyLong objects (28 B for small, more for big); floats are PyFloat (24 B). The numpy `i4`/`f8` collapse is the lesson. | `sums.py` [have] — Kahan/Neumaier/fsum vs naive across pathological inputs; FP is not the reals. **[write]** PyLong/PyFloat footprint vs `np.int32`/`np.float64` at 1M elements (`sys.getsizeof` + array.nbytes). |
| 3 | The `Vec` is a table | "A `list` is universal — put anything in it." A `list` of N objects is N pointers + N object headers + N payloads, scattered. A `list` of tuples is even worse per row. | **[write]** AoS-vs-SoA headline: 1M rows of 10-int tuple-in-list vs 10 lists of 1M ints; `tracemalloc` peak. The ~30% saving demo. |
| 4 | Cost is layout, and you have a budget | "Memory is free." Container headers and per-element pointer-overhead dominate small payloads. | **[write]** list-of-lists vs list-of-tuples vs tuple-of-lists: same data, three layouts, three footprints. |

---

## Part 2: Identity & structure

| § | Chapter | Python-default to call out | Exhibit |
|---|---------|----------------------------|---------|
| 5 | Identity is an integer | "Use a UUID string / object identity / `id()`." Identity choice is measurable: int keys hash faster than float keys, and string keys are worst. | `float_or_int_tuple.py` [have] — (int,int) → 42M lookups/s; (float,float,float) → 17M. Same dict, different identity shape. |
| 6 | A row is a tuple | "Make a class for it." Construction tax: tuple ≪ namedtuple < slots-class < frozen+slots dataclass. `__slots__` does not save you. | `classes_or_tuples.py` [have] — 17–25× spread on construction. `simple_namespace.py` [have] — dict beats SimpleNamespace on creation and mutation. |
| 7 | Structure of arrays (SoA) | "Make a list of objects, each with x/y/z." SoA in Python = parallel typed numpy columns; the dict-of-columns is the natural shape. | `simlog/logger.py` [have] — `Container` is the canonical SoA exhibit. **[write]** dict-of-objects → dict-of-numpy-columns over the same payload, footprint and bulk-op time. |
| 8 | Where there's one, there's many | "Reach for `defaultdict`/`OrderedDict`." A sparse *map* is a dict; a sparse *matrix* is not — match the structure to the access pattern. | `csr_matrix or python dict.py` [have] — dict 108× faster than scipy CSR on random scalar lookups. **Honest framing required:** CSR optimises for SpMV, not point access. |
| 9 | Sort breaks indices | "Sort the list, references update themselves." Sorting a list of objects shuffles row positions; any int-index held elsewhere now points to the wrong row. | **[write]** sort-then-deref demo: list of rows + parallel index dict; sort the list, watch the index lie. |
| 10 | Stable IDs and generations | "`is` vs `==` confusion; GC handles identity." Hash equality is not value equality, and Python lets you build collisions trivially. | `hash collision.py` [have] — three lines, two distinct tuples with equal hash. The foundational trap. |

---

## Part 3: Time & passes

| § | Chapter | Python-default to call out | Exhibit |
|---|---------|----------------------------|---------|
| 11 | The tick | "Use `asyncio` / threads for time." A tick is a single-threaded loop over columns; concurrency is partition, not interleaving. | **[write]** minimal tick over numpy columns; per-tick wall time at N=10K, 100K, 1M. |
| 12 | Event time vs tick time | "Log with `datetime` objects everywhere." `datetime` is 48+ B per instance; `f8` seconds-since-epoch is 8 B and sortable. | `simlog/logger.py` [have] — time stored as `f8` column. **[write]** companion: `datetime` list vs `f8` array footprint at 1M events. |
| 13 | A system is a function over tables | "Methods on objects." A system in Python = a function `(columns, …) → columns`. | **[write]** `update_position(pos, vel, dt)` as numpy column operation vs `creature.tick()` per-object loop; speed and lines-of-code. |
| 14 | Systems compose into a DAG | "Callbacks / observers / pub-sub." Composition is read-set / write-set, not message subscription. | **[write]** small DAG with explicit read/write column declarations; ordering by data dependency. |
| 15 | State changes between ticks | "Mutate in place." Double-buffering is the pattern: write to back, swap; reads see a consistent world. | `simlog/logger.py` [have] — `Container` double-buffer with background dumper is the reference exhibit. |
| 16 | Determinism by order | "Set/dict iteration is fine." Dict iteration is insertion-ordered since 3.7; **set** iteration varies across processes (`PYTHONHASHSEED`). | **[write]** identical program, same input, run in two processes; show set-iteration divergence and how column-iteration is immune. |

---

## Part 4: Existence-based processing

| § | Chapter | Python-default to call out | Exhibit |
|---|---------|----------------------------|---------|
| 17 | Presence replaces flags | "Boolean attribute on each object." Presence = membership in a column / mask array. | `simlog/logger.py` [have] — per-field `_mask_<name>` boolean column is presence-as-data. |
| 18 | Add/remove = insert/delete | "`obj.alive = False`." Add is append; remove is `swap_remove`. The flag is the bug. | **[write]** flag-based "alive" filter scan vs partitioned alive/dead columns; scan cost as alive-fraction drops. |
| 19 | EBP dispatch | "`isinstance` chains." Dispatch is membership in a table, not type-tagging. | **[write]** `isinstance(creature, Predator)` branch loop vs `for i in predator_ids: …`; constant-factor and code-shape comparison. |
| 20 | Empty tables are free | "Every entity carries optional fields." Optional → small dedicated table; absent rows cost nothing. | **[write]** `[Creature(..., disease=None) for _ in range(1M)]` footprint vs main table + tiny `diseased` table at 0.1% prevalence. |

---

## Part 5: Memory & lifecycle

| § | Chapter | Python-default to call out | Exhibit |
|---|---------|----------------------------|---------|
| 21 | `swap_remove` | "`list.remove(x)` / `list.pop(i)`." Both are O(n) with shift; `swap_remove` is O(1). | **[write]** 1M-element list, remove 100K from the middle by both methods; wall time + shift count. |
| 22 | Mutations buffer; cleanup is batched | "Mutate during iteration, hit `RuntimeError: dictionary changed size`." Buffer mutations, apply between ticks. | **[write]** the canonical `RuntimeError` reproduction + the buffered alternative. |
| 23 | Index maps | "Dict-of-object-pointers as universal index." A sparse index is a dict; pick keys (and dtypes) deliberately. | `csr_matrix or python dict.py` [have] — revisited: when "sparse" means "scattered point access," dict wins. `float_or_int_tuple.py` [have] — key shape matters. |
| 24 | Append-only and recycling | "`del obj`, GC frees memory." `list.append` doesn't shrink; recycling slots is the disciplined alternative. | **[write]** append-only with free-list at 1M churn; RSS over time vs `del`+`append`. |
| 25 | Ownership of tables | "Shared mutable state, fingers crossed." One owner per table; everyone else gets read-only views. | **[write]** numpy view aliasing trap (writing through a view mutates the parent) and the explicit-owner discipline. |

---

## Part 6: Scale

| § | Chapter | Python-default to call out | Exhibit |
|---|---------|----------------------------|---------|
| 26 | Hot/cold splits | "One big object with all the fields." Split per-tick fields from once-per-creature fields; hot table stays in cache. | **[write]** combined-column scan vs hot-only scan at 1M rows; wall time. |
| 27 | Working set vs cache | "Python doesn't have caches." Yes it does — at the numpy/CPU level. Random vs sequential numpy access is measurable. | **[write]** sequential vs shuffled-index gather over a 1M `f8` column; throughput ratio. |
| 28 | Sort for locality | "Sort to make output pretty." Sort to make subsequent scans sequential; the cost amortises across the next 100 ticks. | **[write]** unsorted vs sort-once-then-scan-100×; total time. |
| 29 | The wall at 10K → 1M | "Throw it at pandas, then OOM." This is the chapter where Bjorn's lived migration story lands: pandas → sqlite or `.npz`. | `sqlite_performance_test.py` [have] — sqlite as the actual answer when "doesn't fit in RAM" hits. |
| 30 | Moving beyond the wall | "Read the whole file into RAM." Stream from `.npz` / sqlite; bulk-load a partition at a time. | `simlog/logger.py` [have] — streaming write + bulk read API. `numpy_unique_args_permutations.py` [have] — chunkable unique. |

---

## Part 7: Concurrency

| § | Chapter | Python-default to call out | Exhibit |
|---|---------|----------------------------|---------|
| 31 | Disjoint write-sets parallelize freely | "GIL, give up, use a faster language." `multiprocessing.shared_memory` + numpy + main-as-coordinator beats threading wholesale. | **[write]** the headline GIL-creativity exhibit: `__main__` owns a shared-memory numpy array; N workers each get a partition; speedup on physical cores. |
| 32 | Partition, don't lock | "`threading.Lock` everywhere." Disjoint slices need no lock. | **[write]** same harness as §31, with the partition boundaries explicit and zero shared writes. |
| 33 | False sharing | "Doesn't apply in Python — GIL serialises." With `multiprocessing` + `shared_memory`, false sharing is back on the menu. | **[write]** workers writing adjacent indices vs cache-line-spaced indices; speedup gap. |
| 34 | Order is the contract | "Ordering is a UI concern." Order is part of the type when readers depend on it (sorted ID column for binary search, etc.). | **[write]** binary-search-by-ID vs linear-scan once order is part of the contract. |

---

## Part 8: I/O & persistence

| § | Chapter | Python-default to call out | Exhibit |
|---|---------|----------------------------|---------|
| 35 | The boundary is the queue | "`asyncio` / a message bus." `multiprocessing.Queue` between owner and workers; the queue is the literal type-checked boundary. | **[write]** owner+worker harness with a `Queue` carrying partition descriptors, not data. |
| 36 | Persistence is table serialization | "`pickle` / JSON dump." Pickle is opaque, version-fragile, and slow on large numeric data. `.npz` is typed, schema-visible, fast. | **[write]** 1M-row table → pickle vs `.npz` vs sqlite: file size, write time, read time, schema visibility. |
| 37 | The log is the world | "`logging.info(f"…")` formatted strings." A log of typed columns is queryable; a log of strings is grep. | `simlog/logger.py` [have] — the canonical exhibit. This chapter *is* the simlog. |
| 38 | Storage systems: bandwidth and IOPS | "Disk is slow, RAM is fast." Once the page cache is warm, in-memory and on-disk sqlite are within 10%. The cost is dispatch, not the platter. | `sqlite_performance_test.py` [have] — 906K vs 826K lookups/s, in-memory vs on-disk. |

---

## Part 9: System of systems

| § | Chapter | Python-default to call out | Exhibit |
|---|---------|----------------------------|---------|
| 39 | System of systems | "Microservices for everything." Composition first; one process can run a system-of-systems before any RPC enters the picture. | **[write]** two systems sharing one numpy table via shared_memory; the "service boundary" earns its keep only when it has to. |

---

## Part 10: Discipline

| § | Chapter | Python-default to call out | Exhibit |
|---|---------|----------------------------|---------|
| 40 | Mechanism vs policy | "Decorators that hide policy in surprising places." Mechanism is the column op; policy is which rows it runs over. | **[write]** decorator-buried policy vs explicit `select(table, predicate)` + `apply(rows, op)`. |
| 41 | Compression-oriented programming | "DRY at all costs; abstract early." Three similar lines beat a premature `class Strategy`. | **[write]** three near-duplicate updates side-by-side, then the disciplined factoring after the third. |
| 42 | You can only fix what you wrote | "Monkey-patch the library." If you can't fix it, you don't own it; rewrite the small piece you need. | **[write]** small-pure-rewrite-of-third-party-helper vs monkey-patch with a version bump that breaks it. |
| 43 | Tests are systems; TDD from day one | "`pytest` fixtures, mock heavily." Tests are systems over tables: fixed input, expected output column, diff. | `simlog/test_simlog.py` [have] — 713 lines of contract tests against the columnar logger. |

---

## Closure

| § | Chapter | Note |
|---|---------|------|
| 44 | What you have built | No Python-failure call-out; same closure as the Rust edition, with the simlog as the artefact. |

---

## Coverage summary

**Already in `~/code/intro-py/`** (carries the chapter's evidence as-is):
§1, §2 (partial), §5, §6, §8, §10, §15, §17, §23 (partial), §29, §30 (partial), §37, §38, §43.

**To write** (specific exhibit named per row above):
§2 (footprint), §3, §4, §7 (companion), §9, §11, §12 (companion), §13, §14, §16, §18, §19, §20, §21, §22, §24, §25, §26, §27, §28, §31, §32, §33, §34, §35, §36, §39, §40, §41, §42.

That's roughly 30 exhibits to write. Several share a harness (§31/32/33 are
one shared_memory rig with three measurements; §3/4 are one footprint rig
with three layouts), so the actual file count is closer to 18–22.

**One framing fix required** to existing material:
- `csr_matrix or python dict.py` — current conclusion ("CSR is 108× slower")
  is a teaching trap. Reframe as "scipy gave you a sparse *matrix*, not a
  sparse *map*; pick the structure that matches your access pattern."

---

## What this analysis is *not*

- Not the book. This is the map from the existing 44 chapters to the
  evidence the Python edition needs to carry. Drafting prose comes after
  the exhibits exist and have measured numbers on real hardware
  (`reference_hardware.md` rigs).
- Not a structural redesign. Same DAG, same trunk, same closure.
- Not a recommendation to ship two parallel books. This analysis lets us
  decide later whether the Python material lives as (a) a sister site,
  (b) a long appendix to the Rust edition, or (c) per-chapter sidebars in
  one combined book. That decision is downstream of the exhibit work.
