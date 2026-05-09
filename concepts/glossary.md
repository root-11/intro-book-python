# Glossary

Canonical wording for the 43 nodes in [`concepts/dag.md`](dag.md). Each entry gives the teaching definition (the words the book will use), one concrete example drawn from the through-line simulator, the card-game milestone, or one of the track openings, the anti-pattern the concept exists to displace, and cross-references to related nodes.

This file is paired with the DAG: change a definition here, change the node there.

## Format

Each entry has four parts:

- **Definition** — what we say.
- **Example** — how it shows up in an exercise. Drawn from the through-line simulator, the card-game milestone, or one of the five track openings.
- **Anti-pattern** — what students reach for instead, and why this concept rejects it.
- **See also** — cross-references by node number.

---

## 1 — The machine model

**Definition.** A computer is a long array of bytes with a CPU that reads and writes them. Reading from cache (L1/L2/L3) is fast; reading from main memory is roughly 100× slower; chasing a pointer is reading from memory at an unknown address. The cost of an operation is dominated by *where* the data is, not by how clever the algorithm is. *Python adds a layer: per-iteration interpreter dispatch (~5 ns) is comparable to or higher than an L3 read, which masks the cache hierarchy from inside pure-Python code and makes it visible only when numpy bulk ops take over.*

**Example.** In the §0 toy simulator, 100 creatures × four `np.float32` fields × 4 bytes is 1.6 KB — comfortably in L1 cache. The motion loop runs without ever leaving the cache. At §1 with 10,000 creatures, the same fields total 16 KB — still in L1. At §2 with a million, you are at 16 MB — past L2, fits in L3 — and the loop costs change by an order of magnitude. None of this is hypothetical; the [`code/measurement/cache_cliffs.py`](../code/measurement/cache_cliffs.py) exhibit produces the numbers on the reader's machine.

**Anti-pattern.** Programming as if memory access were free. The cost asymmetry shows up the moment the simulator gets non-trivial; treating it as a footnote leads to programs that are unfixably slow at the scales the rest of the book targets.

**See also.** 2 (numbers), 4 (cost & budget), 27 (working set), 29 (10K-to-1M wall).

---

## 2 — Numbers and how they fit

**Definition.** Integers and floats come in widths: `np.uint8` (0..256), `np.uint16`, `np.uint32`, `np.uint64`, `np.int32`, `np.int64`, `np.float32`, `np.float64`. Width is a budget choice — narrower types fit more values per cache line. Floats are not real numbers; they have a finite set of representable values and edges where arithmetic stops behaving (denormals, infinities, NaN). *Python's stdlib `int` (PyLong) is 28+ bytes regardless of value; the width budget exists in numpy, not in Python's native integers.*

**Example.** A 52-card deck stores `suits: np.uint8` and `ranks: np.uint8` because four suits and thirteen ranks fit easily; `np.uint32` would waste 75% of every cache line. The simulator's `creature.energy` is `np.float32` — fast, fits twice as many entries per line as `float64`, and the precision is more than enough for fuel accounting.

**Anti-pattern.** Reaching for `np.int64` and `np.float64` reflexively because they are "safe defaults". They are safe; they are also half the throughput on cache-bound loops. Pick the narrowest type that holds your range and document the choice.

**See also.** 1 (machine model), 3 (ndarray is a table), 27 (working set vs cache).

---

## 3 — The `np.ndarray` is a table

**Definition.** `np.ndarray` of fixed dtype is a contiguous run of typed values in memory, addressed by index. It is the primitive out of which every component table in this book is built. A `np.ndarray` of dtype `np.uint32` and length N is N × 4 bytes laid out in order; `arr[i]` is one pointer addition and one memory load.

**Example.** The card-game `suits`, `ranks`, and `locations` are three `np.ndarray` of dtype `np.uint8`, length 52. The §0 simulator's `creature.pos_x` and `creature.pos_y` are two `np.ndarray` of dtype `np.float32`, length 100. Every concept in the book lands on one or more `np.ndarray`. There are no other primitive containers in the trunk.

**Anti-pattern.** Reaching for `dict`, `list`, `pandas.DataFrame`, or any allocator-per-element structure when an `np.ndarray` and an integer index will do. These all break sequential access (which is what nodes 1 and 4 are about) or pay per-element overhead (Python lists are pointer arrays, not value arrays). Use them only when the access pattern genuinely demands it — and demonstrate that in a benchmark first.

**See also.** 1 (machine model), 5 (id is an integer), 7 (SoA), 27 (working set).

---

## 4 — Cost is layout — and you have a budget

**Definition.** The same algorithm runs at different speeds depending on where its data lives in memory. Asymptotic complexity tells you whether the algorithm scales; *layout* decides the constant factor that dominates at the scales we care about. Every program has a frequency target — a game runs at 30 Hz, a control loop at 1 kHz, a market data system at 1 MHz — which sets a per-tick *budget* in milliseconds. Operations are counted against that budget in microseconds, or in nanoseconds for tight inner loops. *Python adds a fourth regime to the standard three (compute-bound, bandwidth-bound, latency-bound): interpreter-bound, where per-iteration dispatch dominates and the cache hierarchy becomes invisible.*

**Example.** The simulator's main loop targets 30 Hz, giving 33 ms per tick. From [`code/measurement/tick_budget.py`](../code/measurement/tick_budget.py): one motion update on 1,000,000 creatures costs 0.6 ms in numpy SoA (1.8% of budget), 28 ms in a `list[Creature]` AoS form (84% of budget — almost the entire tick on one system).

**Anti-pattern.** Treating performance as something to "optimise later". The layout decisions made early decide whether the program ever has a chance of meeting its budget; refactoring an OOP graph to SoA is a project, not a tweak.

**See also.** 1 (machine model), 7 (SoA), 11 (the tick), 27 (working set), 29-30 (scale walls).

---

## 5 — Identity is an integer

**Definition.** An entity is a small integer — usually a `np.uint32`. It names a slot in the world's tables, not a thing in itself. There is no entity *object* and no "where the entity lives". An entity is one number, and that number is an index into every table that has something to say about it.

*The strong form: sometimes you don't even need that number.* If the row's own fields uniquely identify it — `(suit, rank)` for a playing card, `(date, ticker)` for a market quote — the identity is already in the data. A separate `entity_id` is then a *surrogate key*; before adding one, ask whether the data carries a *natural key* you can use directly. The card game can be played using `(suit, rank)` and no entity id at all. Variable-quantity tables (creatures, packets, sessions) usually have no natural key — two creatures can be identical — so a surrogate id is necessary, and nodes 9-10 follow.

**Example.** In the card-game milestone (after node 10), an entity is one of the indices `0..52`. The card at index 17 has its suit at `suits[17]`, its rank at `ranks[17]`, and its current location — deck, hand, or discard — at `locations[17]`. Dealing a card means writing one cell in `locations`. There is no `Card` class.

**Anti-pattern.** Treating the entity as a class instance with methods. The moment an entity has methods, the data is scattered across `PyObject` allocations, mutation is hidden behind setters, and the rest of the book's economies — SoA, parallelism, persistence, replay — become impossible. Most students arriving in this book have written exactly this code before; the card game is where they first feel the alternative.

**See also.** 3 (ndarray is a table), 6 (row is a tuple), 9 (sort breaks indices), 10 (stable IDs and generations).

---

## 6 — A row is a tuple

**Definition.** A coherent set of values that describe one entity travels together — but only if you keep them together. In ECS, "together" means *at the same index in every component table that has something to say about that entity*. Split a row across tables and you must keep the indices aligned; rearrange one without rearranging the others and you have corrupted the world. *In Python the most useful version of this statement is: a row is a tuple you do not have to build. The tuple at index `i` only exists when you ask for it explicitly; until then it lives in contiguous bytes inside numpy columns.*

**Example.** A creature at index 17 has its position at `pos_x[17]`, `pos_y[17]`, its velocity at `vel_x[17]`, `vel_y[17]`, its energy at `energy[17]`, and its birth time at `birth_t[17]`. Together they are the row. There is no `Creature` class holding all four; the row is implicit in the alignment.

**Anti-pattern.** Keeping a `list[Creature]` (AoS — Array of Structs). It works, but it sacrifices the layout reasoning of nodes 4 and 7: the inner loop reads all six fields whether it needs them or not, doubling cache pressure for systems that only touch position. *Plus a per-instance Python overhead of ~100 bytes that doesn't exist in numpy SoA.*

**See also.** 5 (id is integer), 7 (SoA), 23 (index maps), 25 (ownership of tables).

---

## 7 — Structure of arrays (SoA)

**Definition.** Each field of a row gets its own `np.ndarray`, indexed by entity. The row is reconstructed at access time by reading position `i` from each field's array. The opposite layout — `list[Creature]` (AoS), or `np.dtype([('field', ...)])` structured arrays — bundles the row's fields into one record; SoA splits them. SoA is the default in this book because most systems read only a few fields, and SoA gives them sequential access to exactly those fields.

**Example.** The `creature` table is six `np.ndarray`s — `pos_x`, `pos_y`, `vel_x`, `vel_y`, `energy`, `birth_t`, plus `id` and `gen`. The `motion` system reads only `pos_*`, `vel_*`, `energy`. With SoA those vectors are sequentially scanned; AoS would force the loop to read all eight fields whether it needs them or not.

**Anti-pattern.** Reaching for `list[Creature]` because "it's neater". Neatness is not a layout property. The cost is real and shows up at §1 onwards — the §3 exhibit measures it at 75× slower for sum-column-0 at 1M rows.

**See also.** 4 (cost & budget), 6 (row is a tuple), 26 (hot/cold splits), 31 (disjoint writes parallelize).

---

## 8 — Where there's one, there's many

**Definition.** Code is written for the array. The single-instance case is simply N=1; it does not need its own abstraction. A function that takes one entity and returns one result is a special case of a function over an `np.ndarray`; write the array version first and the singleton drops out.

**Example.** "Update one creature's position" is `pos_x[i:i+1] += vel_x[i:i+1] * dt`. "Update all creatures" is `pos_x += vel_x * dt`. Same operation, different slice. The card game illustrates the singularity case from the other side: a card game with 52 cards is three numpy arrays — suit, rank, location — not 52 objects.

**Anti-pattern.** Writing `Card.shuffle(self)` and then puzzling over how to shuffle a deck. The deck is three `np.ndarray`s; shuffling is permuting an order array; the per-card operation never appears.

**See also.** 3 (ndarray is a table), 13 (system as function over tables), 31 (disjoint writes parallelize).

---

## 9 — The sort breaks indices

**Definition.** Rearranging the rows of a table — sorting, swap-removing, compacting — breaks any external reference that pointed at a slot. The card you held at index 17 is still there, but index 17 may now be a different card. The student must feel this pain before the next node makes sense.

**Example.** In §5's exercise 10, player 1 holds card indices `[3, 17, 21, 28, 41]`. The dealer sorts the deck columns themselves by suit. Player 1's hand is now wrong: index 17 used to be the 5♥, but is now the 4♣. The student observes the bug; they don't fix it yet.

**Anti-pattern.** Saving an index across a reordering. The fix — coming next — is to save a stable id, not a slot index. *Tempting Python-specific escape hatch: hold object references to `Card` instances instead. This works only if you accept the AoS layout, giving back the §3 footprint and §7 access wins.*

**See also.** 5 (id is integer), 10 (stable IDs and generations), 23 (index maps), 28 (sort for locality).

---

## 10 — Stable IDs and generations

**Definition.** A separate `id` column gives a name that survives sorting. A `generation` counter on top gives a name that survives recycling: when a slot is reused, its generation increments, so any reference holding the old `(slot, gen)` pair can detect that it is stale. *The id behaves like an auto-incrementing primary key in a database; the generation is the part that isn't there, because a database table grows and a recycling simulator does not.*

**Example.** In the §1 simulator, every `creature` carries `id: np.uint32` and `gen: np.uint32`. A reference to creature `(id=42, gen=3)` survives sorting (the column is reordered, but the pair persists), and survives recycling (if slot 17 is freed and reused for a fresh creature, that fresh creature has `gen=4`, so the old `gen=3` reference no longer matches).

**Anti-pattern.** Treating slot index as identity. This works until the first sort, after which it never works again. The stable-id pattern is the cheapest possible fix and is the structural backbone of every "handle" type in modern systems software.

**See also.** 5 (id is integer), 9 (sort breaks indices), 23 (index maps), 24 (append-only & recycling).

---

## 11 — The tick

**Definition.** Programs run in discrete passes. State at the start of a tick is read; state at the end is written; nothing is half-updated mid-tick. The tick has two natural shapes: *turn-based* — the loop advances when an event arrives (a card game, a chess engine, a discrete-event simulator); and *time-driven* — the loop runs at a fixed rate (30 Hz, 1 kHz) with a per-tick budget. *In Python: not asyncio (an I/O scheduler that adds dispatch overhead and removes nothing), not threading (the GIL serialises bytecode and adds nondeterminism) — a synchronous loop with `time.perf_counter` and `time.sleep` is the disciplined answer.*

**Example.** The card game is turn-based: a tick is "deal one card" or "play one move". The §1 simulator is time-driven: a tick is one 33 ms step, during which all systems run in order. Both are tick loops; the difference is what drives the next pass.

**Anti-pattern.** Threading "real time" through the program as a global clock. The tick is the right unit because it makes determinism cheap (node 16) and bounds the work per pass.

**See also.** 4 (cost & budget), 12 (event time vs tick time), 13 (system as function), 14 (systems compose into a DAG).

---

## 12 — Event time is separate from tick time

**Definition.** The tick rate is how often the loop runs — typically a fixed number per second (30 Hz, 1 kHz). The *event clock* is the simulation's internal time, which lives on the events themselves. A 30 Hz loop can resolve microsecond-precision events because the timestamp travels with the event, not with the loop. *In Python the discipline is to store time as `np.float64` seconds-since-base, not as `datetime` objects — `datetime` is 7× the footprint and 17× slower at the per-tick "events before T" query that decides what gets processed this tick.*

**Example.** In the multi-agent track, 10,000 delivery drones each carry an arrival timestamp at their next stop. The loop runs at 30 Hz, but inside one tick the simulator may process events whose timestamps differ by four microseconds. The visualisation samples at tick rate; the underlying physics runs at event-clock resolution. The same pattern recurs in the multiplayer track, where rollback works only because event time is not tick time.

**Anti-pattern.** Conflating the two — usually expressed as "my model can only resolve `dt = 1/30s` because the loop runs at 30 Hz". This is the most common confusion in physical simulation and event-driven systems work, and it imposes a false ceiling on the model's time resolution. The fix is structural: put the timestamp on the data.

**See also.** 11 (the tick), 16 (determinism by order), 37 (the log is the world).

---

## 13 — A system is a function over tables

**Definition.** A system declares its inputs (read-set) and outputs (write-set). It has no hidden state. The signature is the contract. Every system takes one of three shapes: an *operation* (1→1, every input row produces one output), a *filter* (1→{0,1}, every input row produces zero or one), or an *emission* (1→N, every input row produces zero or more). These are the same shapes as familiar database operations — `sort`, `groupby`, `filter`, `join`, `aggregate` — over component arrays. Even observability is a system: `inspect` holds read references to other systems' tables, instantiated only when transparency is needed; in production it is *absent*, not gated.

*In Python, a function that takes `self` does not have a declared read-set or write-set; a function that takes columns does. The OOP method form is the universal anti-shape this rule replaces.*

**Example.** `motion` is an operation: read `(pos_x, pos_y, vel_x, vel_y)`, write `pos_x, pos_y`. `apply_eat` is a filter: read pending eat events, output an updated energy and a removed food row. `apply_reproduce` is an emission: one parent input row, two offspring output rows. The simulator's eight systems split cleanly into the three shapes.

**Anti-pattern.** A system that touches global state, mutates input parameters, or carries cross-tick state in a closure. None of these compose, none of these parallelize, and none of these can be tested without a fixture. *Python adds: `print()` from inside a system, `logger.info(...)`, `requests.get(...)`, `os.environ[...]` — all are I/O leaks that violate the boundary rule and break determinism.*

**See also.** 8 (one to many), 14 (systems compose into a DAG), 25 (ownership of tables), 31 (disjoint writes parallelize).

---

## 14 — Systems compose into a DAG

**Definition.** The order of systems is given by who reads what who wrote. A system that reads a table must run after every system that writes that table within the tick. The program is a *topological sort* of this graph; choose the sort, and the program runs. Designing the system order is the same problem as designing a database query plan: each system is a stage, the DAG is the plan, and the program executes the plan.

**Example.** The §1 simulator's tick DAG: `food_spawn → motion → next_event → {apply_eat, apply_reproduce, apply_starve} → cleanup → inspect`. Drawing this DAG is the first thing to do when adding a new system; the question "what do I read?" forces the right edges.

**Anti-pattern.** Calling systems in the order they were written in the file. This works for the first three systems; by the tenth, the read/write dependencies are tangled and one bad ordering corrupts state in ways that are hard to find. *In Python the looks-right-but-isn't anti-shape is `asyncio.gather` over the systems — runs them concurrently with no notion of dependencies; the first to complete, completes; the rest race.*

**See also.** 13 (system as function), 25 (ownership of tables), 34 (order is the contract), 31 (disjoint writes parallelize).

---

## 15 — State changes between ticks

**Definition.** Mutations buffer; the world transitions atomically at tick boundaries. Inside a tick, systems read consistent snapshots of their inputs and *queue* changes to their outputs. At the end of the tick, the queued changes are applied. This is the structural reason systems compose at all.

**Example.** When a creature dies in `apply_starve`, its id is appended to `to_remove`. The creature row is *not* yet gone; the rest of the tick's systems still see it. After all systems complete, `cleanup` applies `to_remove` (and `to_insert` from `apply_reproduce`) in one bulk-mask filter operation, and the next tick begins with the world in a consistent state.

**Anti-pattern.** Mutating the table inside a system pass. *In Python this rule eliminates two famous footguns: `creatures.remove(c)` during `for c in creatures` (silently skips elements; the simulation runs forever with half the deaths missed), and `del d[k]` during `for k in d` (raises `RuntimeError`). The structural fix is buffered cleanup — same shape regardless of which footgun the language presents.*

**See also.** 14 (systems compose into a DAG), 16 (determinism by order), 22 (mutations buffer), 37 (the log is the world).

---

## 16 — Determinism by order

**Definition.** Same inputs + same system order = same outputs. Reproducibility is structural, not a quality goal. It is what makes replay possible (you can rerun any tick from a snapshot), testing trustworthy (a property test can fix a seed), and the simulator's regression test (the population graph) reliable.

**Example.** Two runs of the §1 simulator with the same seed and the same system order produce bit-identical population graphs. Reorder two systems with overlapping write-sets, and the runs diverge — which is exactly the bug that node 34 ("order is the contract") is written to prevent.

**Anti-pattern.** Relying on ad-hoc randomness, system threads scheduled by the OS, or "good enough" reproducibility. These are fine for debugging but fatal for replay and for distributed extensions. *Python-specific recipe: no raw `set` iteration (`PYTHONHASHSEED` randomises across processes — the bug fails CI workers but passes local), no `time.perf_counter` inside systems, one `np.random.default_rng(seed)` per simulator. `pytest-xdist` is a determinism-leak detector.*

**See also.** 14 (systems compose into a DAG), 34 (order is the contract), 37 (the log is the world), 43 (tests are systems).

---

## 17 — Presence replaces flags

**Definition.** "Is hungry" is membership in a `hungry` table, not a `bool` field on `Creature`. State is structural — a row exists or it does not — rather than a flag stored alongside other data. The change reads as small in code and turns out large in consequence: dispatch, parallelism, and persistence all simplify.

*In Python the spectrum is wider than in Rust. Three representations on a continuum: per-instance bool field on `Creature` (worst — AoS plus flag), `np.bool_` column indexed in lockstep (better — SoA with the flag still O(N) bytes), and a presence index `hungry: np.ndarray[np.uint32]` of just the affected ids (disciplined — O(K) bytes when sparse). Crossover between bool column and presence index lands around 80-90% prevalence on this machine; below that, presence wins decisively.*

**Example.** In the simulator, a creature becomes hungry by having its id appended to `hungry`. The system that drives hunger-related behaviour iterates `hungry` directly via `energy[hungry] -= burn * dt`; it does not scan `creatures` checking a flag. The same pattern appears in production ECS daemons: `is_admitted = peer_id in established_contacts` — O(1), no I/O, no enum.

**Anti-pattern.** `if creature.is_hungry: ...`. The flag forces every system that cares about hunger to filter the entire creature table; the table grows linearly with population whether or not anyone is hungry. *In Python the additional cost is per-iteration `getattr` and interpreter dispatch — the AoS form runs 30-50× slower than the presence form at 1M creatures in the [`code/measurement/alive_fraction.py`](../code/measurement/alive_fraction.py) exhibit.*

**See also.** 13 (system as function over tables), 18 (add/remove = insert/delete), 19 (EBP dispatch), 20 (empty tables are free).

---

## 18 — Add/remove = insert/delete

**Definition.** A state transition is a structural move: insert a row in one table, remove a row from another. There is no `set_hungry(True)`. To make a creature hungry, you append to `hungry`; to make it stop being hungry, you swap-remove the row.

**Example.** When a creature eats food in §1, `apply_eat` appends the food's id to `to_remove` and updates the creature's `energy`. There is no `food.is_eaten = True` flag — the food simply ceases to be in the table after cleanup.

**Anti-pattern.** Tombstoning rows with `is_alive = False` or `is_eaten = True` flags. The flag forces every reader to filter the table; the table grows linearly with history. *In Python the same anti-pattern wears extra disguises: `@property` setters that flip-and-commit (validation buried inside attribute assignment), `__setattr__` overrides that intercept writes (policy hidden inside object identity), `update(self, **fields)` methods that take `**kwargs` (signature hides the read-set and write-set).*

**See also.** 17 (presence replaces flags), 19 (EBP dispatch), 21 (`swap_remove`), 22 (mutations buffer).

---

## 19 — EBP dispatch

**Definition.** A system iterates over the table whose presence defines its applicability. There is no per-row branch checking *"does this case apply to me"*; if a row is in the table, the system runs on it.

**Example.** The "process all hungry creatures" system iterates the `hungry` table directly: `energy[id_to_slot[hungry]] -= burn * dt`. There is no `for c in creatures: if c.is_hungry: ...`. The dispatcher *is* the table; iterating means processing.

**Anti-pattern.** Iterating a master table and filtering inside the loop. *Three Python forms that all reduce to filtered iteration: `isinstance` chains over a heterogeneous list (`if isinstance(e, Hungry):`), polymorphic method dispatch via inheritance (`for c in creatures: c.update()` where each subclass overrides `update`), and list-comprehension filters (`hungry = [c for c in cs if c.is_hungry]`). Each consults the predicate per entity; each pays interpreter dispatch on every visit.*

**See also.** 13 (system as function), 17 (presence replaces flags), 18 (add/remove = insert/delete), 20 (empty tables are free).

---

## 20 — Empty tables are free

**Definition.** No rows means no work. A simulation with 90% inactive entities does no work for the inactive ones — the dispatcher never visits them.

**Example.** The §1 simulator may have 10,000 creatures and 9,000 of them are not hungry yet (their energy is full). The hunger system iterates `hungry` (1,000 ids), not `creatures` (10,000 rows). Cost scales with active rows, not with population. *The [`code/measurement/empty_tables.py`](../code/measurement/empty_tables.py) exhibit measures the 0% prevalence case directly: at zero diseased creatures, the optional-field layout still costs 105 MB and 7.5 ms per tick; the presence layout costs 0.02 ms — 365× faster, and 4× smaller.*

**Anti-pattern.** Iterating the master table "just to be safe", or putting an `Optional[Disease]` field on every creature so the data lives in one place. The 9,000 healthy creatures cost as much as the 1,000 affected ones, and no number of branch hints fixes that.

**See also.** 17 (presence replaces flags), 18 (add/remove = insert/delete), 19 (EBP dispatch), 29 (10K-to-1M wall).

---

## 21 — `swap_remove`

**Definition.** Deletion in O(1) by moving the last row of a table into the deleted slot, then shrinking the active range by one. Order is sacrificed for speed; the next two nodes fix the consequences. This and the rest of the Memory & lifecycle phase only matter for *variable-quantity* tables; constant-quantity tables like the 52-card deck need none of it.

*In Python the in-place form pairs a fixed-capacity numpy array with an `n_active: int` counter; the "table" is the prefix `arr[:n_active]`. Removing is `arr[i] = arr[n_active - 1]; n_active -= 1`. For batched removal — the §22 cleanup case — the bulk-mask filter `arr[keep_mask]` is even faster than per-element swap_remove because it pays the Python-numpy boundary cost once instead of K times.*

**Example.** When a creature dies in §1, `cleanup` builds one `keep_mask = np.ones(n_active, dtype=bool); keep_mask[doomed_slots] = False`, then applies it to every column: `for col in cols: col[:new_n] = col[:n_active][keep_mask]`. One mask, six columns, one numpy pass per column.

**Anti-pattern.** `np.delete(arr, slot)`, which allocates a fresh array each call and copies the survivors over. From the [`code/measurement/swap_remove.py`](../code/measurement/swap_remove.py) exhibit at 100K removes from a 1M-row column: `np.delete` is 1,097× slower than the bulk-mask filter; `list.pop(i)` is 1,129× slower.

**See also.** 18 (add/remove = insert/delete), 22 (mutations buffer), 23 (index maps), 24 (append-only & recycling).

---

## 22 — Mutations buffer; cleanup is batched

**Definition.** Inserts and removes during a tick are not applied immediately; they are recorded as dirty markers in side tables (commonly `to_insert` and `to_remove`). At the tick boundary, a single sweep applies them all. Structural changes happen *between* passes, not during them.

**Example.** In the through-line simulator, when a creature dies its entity id is appended to `to_remove`. The system that detected the death does not call any swap_remove on the position table — that would corrupt the iteration the system is in the middle of. After every system in the tick has run, `cleanup` builds one `keep_mask` from `to_remove`, applies it to every column in lockstep, and clears the marker lists. *Python edition uses bulk-mask filter at cleanup, not per-element swap_remove; from §21's exhibit, the bulk form is 5× faster at K=100K removes per tick.*

**Anti-pattern.** Mutating tables in place inside a system pass. Either the iteration breaks (because the indices it is using just got rearranged) or you pay per-mutation interpreter dispatch (because numpy bulk ops aren't an option when each mutation is decided one at a time). In a simulation with steady birth and death, the cost is at least O(N) interpreter trips per tick — orders of magnitude over what the budget allows.

**See also.** 15 (state changes between ticks), 18 (add/remove = insert/delete), 21 (`swap_remove`), 23 (index maps).

---

## 23 — Index maps

**Definition.** When external references must survive reordering, an `id_to_slot` map maintains the mapping. It is updated on every move — whether by `swap_remove`, by sort-for-locality, or by the buffered-cleanup sweep. Looking up a creature by id is O(1) through the map; no scanning required.

**Example.** A player holds creature id 42. The `creature` columns get sorted for locality (node 28). The `id_to_slot` map is also rewritten in lockstep — one bulk numpy assignment: `id_to_slot[ids[order]] = np.arange(n_active)`. The player's reference still works.

**Anti-pattern.** Scanning the id column to find a row by id. This is O(N) per lookup, which is fine at §0 and slow at §1. *In Python there are two right shapes — `id_to_slot: np.ndarray[np.uint32]` for dense ids, `dict[int, int]` for sparse — and one wrong tool: `scipy.sparse.csr_matrix` for point lookups. The CSR form is 108× slower than dict, not because CSR is slow but because it is built for sparse-matrix-vector products, not point queries.*

**See also.** 5 (id is integer), 9 (sort breaks indices), 10 (stable IDs and generations), 28 (sort for locality).

---

## 24 — Append-only and recycling

**Definition.** Two strategies for slot reuse. *Append-only* tables grow forever; old slots stay valid forever. *Recycling* tables reuse vacated slots; the generation counter (node 10) prevents stale references. The choice is decided by access pattern: append-only is simpler but wastes memory under churn; recycling pays a small bookkeeping cost and bounds memory.

**Example.** The simulator's `eaten`, `born`, `dead` logs are append-only — they record history and never delete. The `creature` table itself is recycling — slots are reused as creatures die and new ones are born. Two strategies, same simulator, different access patterns.

**Anti-pattern.** Always append-only "to keep things simple". For a long-running simulator with steady churn, the table grows without bound and the working set blows the cache. Always recycling, conversely, breaks the history that node 37 wants to lean on. *In Python, the recycling discipline is a `free_slots: list[int]` LIFO stack plus a `gens: np.ndarray` of generation counters — Python does not have automatic GC for numpy column slots.*

**See also.** 10 (stable IDs and generations), 21 (`swap_remove`), 35 (boundary is the queue), 37 (the log is the world).

---

## 25 — Ownership of tables

**Definition.** Each table has exactly one writer. Many readers are fine. This is the rule that makes parallelism possible without locks, and it is the precondition for the inspection-system pattern: read-only access to all tables, no risk of races. *Python has no borrow checker; the discipline is convention. The [§25 chapter](../book/trunk/25_ownership_of_tables.md) names the conventions: explicit `.copy()`, `arr.flags.writeable = False`, and the docstring read/write-set contract.*

**Example.** In the §1 simulator, `motion` is the only writer of `creature.pos_*`. `apply_eat` is the only writer of `food`. `cleanup` is the only writer of the `creature` table's structure (insertions and removals). When two systems have disjoint write-sets, they parallelize freely (node 31).

**Anti-pattern.** Two systems writing the same field. *Python adds a subtle variant: the numpy view trap. `view = arr[2:5]` is not a copy — it is a view into the same underlying bytes. Writing through the view mutates the parent; if a function receives a view as `arr` and assumes ownership, it has silently violated the rule. The mitigation is explicit: pass `arr[2:5].copy()` when the receiver is supposed to own its slice, or set `arr.flags.writeable = False` on shared parents.*

**See also.** 13 (system as function), 14 (systems DAG), 31 (disjoint writes parallelize), 40 (mechanism vs policy).

---

## 26 — Hot/cold splits

**Definition.** Fields touched in the inner loop go in one table; metadata read rarely goes in another. The inner loop's footprint shrinks; cache works. SoA is the prerequisite — you cannot split fields you have already bundled into a struct.

**Example.** The §2 simulator splits `creature` into `creature_hot` (`pos_*`, `vel_*`, `energy` — read every tick by `motion` and `next_event`) and `creature_cold` (`birth_t`, `species`, `name` — read only when logging or debugging). The hot table fits in L2; the cold table does not have to.

**Anti-pattern.** A single fat table where every system reads all fields whether it uses them or not. The cold fields are paid for in cache traffic at every hot-path read. *In Python+numpy SoA, this lesson is gentler than in Rust AoS — columns are already physically separate allocations, so reading `pos_x` does not bring `birth_t` into cache. The split's primary value in numpy SoA is organisational (the §13 read-set and write-set contracts get shorter), not bandwidth-saving. The exception: numpy structured arrays (`np.dtype([(...)])`) are AoS in numpy clothing; the split helps there.*

**See also.** 4 (cost & budget), 7 (SoA), 27 (working set vs cache), 28 (sort for locality).

---

## 27 — Working set vs cache

**Definition.** The size of the data the inner loop touches per pass decides speed more than the algorithm. If it fits in L1/L2, the loop is fast; if it does not, no algorithm saves you. This is what every other Scale-phase node serves: keeping the working set in cache.

**Example.** The §2 simulator's `motion` loop reads two `np.float32` × 2 fields (`pos_*`, `vel_*`) per creature. At 1,000,000 creatures × 16 bytes = 16 MB — bigger than L2, fits in L3. The loop is L3-bound. Splitting hot/cold (node 26) and sorting for locality (node 28) shrinks the per-pass touch and brings the loop back into L2. *The [`code/measurement/cache_cliffs.py`](../code/measurement/cache_cliffs.py) exhibit measures the cliff for the reader's hardware: at 100M elements on this machine, random gather is 72× slower than sequential.*

**Anti-pattern.** Optimising the algorithm without measuring the working set. A 2× algorithmic speedup that doubles the working set is a slowdown.

**See also.** 1 (machine model), 4 (cost & budget), 26 (hot/cold splits), 28 (sort for locality).

---

## 28 — Sort for locality

**Definition.** Reordering rows so that frequently co-accessed entities sit together turns random access into sequential access. This is the technique that node 9 (sort breaks indices) was the prerequisite pain for: once you have stable ids and an index map (nodes 10, 23), you can sort the table without breaking external references.

**Example.** The §2 simulator sorts creatures by spatial cell so that creatures-likely-to-collide are adjacent in the column. The `next_event` system's per-creature work now reads neighbours from the same cache line. The id-to-slot map is rewritten in the same pass: `order = np.argsort(spatial_cell); for col in cols: col[:] = col[order]; id_to_slot[id_col] = np.arange(n_active)`.

**Anti-pattern.** Skipping the sort because of node 9. The fear of breaking references is solved by node 10's stable ids, not by leaving the table unsorted forever.

**See also.** 9 (sort breaks indices), 10 (stable IDs and generations), 23 (index maps), 27 (working set vs cache).

---

## 29 — The wall at 10K → 1M

**Definition.** What changes when allocations cannot be casual: pre-sized buffers, no per-frame heap traffic, `swap_remove` instead of `remove`, batched cleanup, consciously chosen layouts. The design budget from node 4 starts to bind. Code that worked at 10,000 stops working at 1,000,000 not because of complexity class, but because of constant factors.

**Example.** §1's `apply_reproduce` calls `to_insert.append(offspring)` once per reproducing parent. At 10,000 creatures with 1% reproducing per tick, that is 100 appends per tick — fine. At 1,000,000 with the same rate, it is 10,000 appends per tick, and the list's geometric growth becomes visible. §2 pre-sizes `to_insert` to a typical batch capacity and the resizes disappear.

**Anti-pattern.** Treating §1 code as ready for §2 scale without measurement. The wall is *constant factor*, not algorithm — profilers find it (`cProfile`, `py-spy`); intuition does not. *The pandas-specific wall: a `DataFrame` of 10M rows × 20 columns at default dtypes occupies 1.6 GB+ before any operation. The migration from pandas to numpy SoA or sqlite is usually a one-day project that gives back days of OOM debugging per quarter.*

**See also.** 4 (cost & budget), 21 (`swap_remove`), 22 (mutations buffer), 30 (1M-to-streaming wall).

---

## 30 — The wall at 1M → streaming

**Definition.** What changes when the table no longer fits in main memory at all. Snapshots, sliding windows, log-orientation. The world becomes a window over the log; only the relevant slice is in memory at any one time.

**Example.** §3's simulator may simulate a year of population history at 30 Hz — close to a billion ticks. The `eaten`, `born`, and `dead` logs alone are too big to keep in memory. The simulator writes them through to disk (a storage system, node 38) and re-reads windows on demand. The world becomes a function of the log over a time range.

**Anti-pattern.** Treating "doesn't fit in memory" as a problem to solve with a bigger machine. The streaming pattern scales to anything the log itself can hold; a bigger machine just postpones the same redesign. *The Python toolkit: `np.savez` for snapshots; sqlite for queryable archives; the simlog as the canonical streaming logger. Not `np.memmap` — it works but rarely beats explicit chunked reads, because the OS's prefetch heuristics don't match the simulator's access patterns.*

**See also.** 27 (working set vs cache), 35 (boundary is the queue), 37 (the log is the world), 38 (storage systems).

---

## 31 — Disjoint write-sets parallelize freely

**Definition.** Two systems that write to disjoint tables can run in parallel without coordination. No locks, no atomics. This is what node 25's ownership rule buys: every table has one writer, so any two systems with non-overlapping writes are by construction race-free.

**Example.** In the §1 simulator, `apply_eat` writes `food` and `creature.energy`; `apply_starve` writes only `creature` removals via `to_remove`. Disjoint write-sets — they can run in parallel. *In Python the parallelism is via `multiprocessing.Pool` plus `multiprocessing.shared_memory`, not threads. From the [`code/measurement/parallel_motion.py`](../code/measurement/parallel_motion.py) exhibit on this 8-physical-core machine: ~4× speedup for memory-bound work, ~5.5× for compute-bound — the ceiling shifts with workload type.*

**Anti-pattern.** Threading shared memory under the GIL — the GIL serialises bytecode and gives no parallel CPU for Python-level work. `asyncio.gather` over CPU-bound systems is the same trap with extra ceremony. The correct form is processes with shared memory and pre-assigned partitions.

**See also.** 13 (system as function), 25 (ownership), 32 (partition not lock), 34 (order is the contract).

---

## 32 — Partition, don't lock

**Definition.** When one system must write a single table from multiple threads, split the table by entity range (or by spatial cell, or by hash) and give each thread its own slice to write. You partition the data, not the access. Each thread's slice has a single writer; nodes 25 and 31 still hold within each slice.

**Example.** The §2 `motion` system writes `creature.pos_x` for a million creatures across 7 worker processes. Instead of locking, the loop is split: worker `t` writes slots `t*N/7 .. (t+1)*N/7`. No lock, no atomic, no contention. *The disciplined production form is the Beazley ventilator: pre-assigned partitions held by each worker for its lifetime, signal-only dispatch (system index, not data), shared-array DAG for coordination. From the [`code/measurement/coordination_patterns.py`](../code/measurement/coordination_patterns.py) exhibit: ~1.5M msgs/sec via shared array, ~90K via single shared `multiprocessing.Queue`.*

**Anti-pattern.** A `Lock` shared across processes. Even when correct, the lock serialises the write under contention; you have re-introduced the single-writer rule the long way around.

**See also.** 25 (ownership), 28 (sort for locality), 31 (disjoint writes), 33 (false sharing).

---

## 33 — False sharing

**Definition.** Two threads writing to *different* fields that happen to land in the same cache line slow each other down through hardware. The cache coherency protocol forces every write to invalidate the line on the other thread, even though the writes don't conflict logically.

**Example.** Eight threads each accumulate a counter in `counters: np.ndarray((8,), dtype=np.int64)`. Naive layout puts all 8 counters in one 64-byte cache line — the threads thrash the line. Padding each counter to its own cache line (`counters: np.ndarray((64,), dtype=np.int64)` with each worker writing index `my_id * 8`) eliminates the contention.

**Anti-pattern.** Laying out per-thread state as adjacent bytes/words. Almost always a footgun. *In Python with `multiprocessing.shared_memory`, false sharing applies the same way it does in compiled languages — the GIL does not protect across processes. Detection: `perf stat -e cache-references,cache-misses` on Linux.*

**See also.** 27 (working set vs cache), 31 (disjoint writes), 32 (partition, don't lock).

---

## 34 — Order is the contract

**Definition.** Parallelism is allowed *inside* a step (between systems with disjoint writes), never *across* steps. Determinism (node 16) depends on this discipline. The system DAG (node 14) defines the permitted concurrency; anything outside the DAG is undefined behaviour.

**Example.** In the §1 simulator, `apply_eat`, `apply_reproduce`, and `apply_starve` may run in parallel because their writes are disjoint. They must all complete before `cleanup` starts. They must all run after `next_event`. The order is the contract; parallelism happens inside the contract, never around it. *In Python the §32 ventilator IS the scheduler: phase boundaries enforce the DAG; intra-phase parallelism uses §31's disjoint-writes architecture. One mechanism, two readings — the parallel schedule and the deterministic execution order are the same document.*

**Anti-pattern.** "Optimising" by running systems out of DAG order because the test passed once. Determinism is a property of structure, not of testing. *The Python looks-right-but-isn't form: `await asyncio.gather(motion(world), next_event(world), apply_eat(world), ...)` — runs them concurrently, no notion of dependencies, the first to complete completes; the bug surfaces as wrong answers, not crashes.*

**See also.** 14 (systems compose into a DAG), 16 (determinism by order), 31 (disjoint writes parallelize), 32 (partition not lock).

---

## 35 — The boundary is the queue

**Definition.** Events flow into the world on one queue, results flow out on another. Inside, the world is pure transformation — no I/O, no time, no environment. Everything that crosses the boundary goes through a storage system (node 38). The queue is the seam.

**Example.** The §1 simulator's input queue carries food-spawn events from the `food_spawner` policy; the output queue carries `eaten`, `born`, `dead` events to the population log. The simulator's tick reads the input queue, transforms the world, writes to the output queue. Nothing else crosses the boundary. *The queue shape: numpy parallel columns for high-throughput events, a list of small dicts/named tuples for low-volume mixed-schema, sqlite when the queue must be durable.*

**Anti-pattern.** Sprinkling I/O calls inside systems. Logging from `apply_eat`, calling out to a metrics service from `motion`. *Five Python forms specifically: `print(...)`, `logger.info(...)`, `time.perf_counter()`, `requests.get(...)`, `os.environ[...]` — each one inside a system body breaks determinism the moment two runs see different values. One subtle Python-specific clarification: `multiprocessing.Queue` is for §32's main-to-worker coordination, not for the simulator's external boundary; conflating the two means every external input pays kernel-call cost and the queue's order is process-scheduler-dependent.*

**See also.** 13 (system as function), 36 (persistence is table serialization), 37 (the log is the world), 38 (storage systems).

---

## 36 — Persistence is table serialization

**Definition.** A snapshot is the world's tables written as a stream of `(entity, key, value)` triples — the same shape the world has in memory. Recovery is reading them back. There is no separate "domain model" to map; serialisation is *transposition*, not *translation*.

**Example.** The simulator can write `creature.pos_x`, `creature.pos_y`, ..., `creature.id` to a single `.npz` archive: `np.savez(path, **world.columns)`. To recover: `with np.load(path) as data: world.columns = {k: data[k] for k in data.files}`. No ORM, no schema migration, no impedance mismatch — the file is the same shape as the memory. *From [`code/measurement/persistence_shapes.py`](../code/measurement/persistence_shapes.py): for the same 1M-row 8-column world, AoS pickle is 86 MB and 2.1s to write; pickle of dict-of-numpy-columns is 34 MB and 2.7 ms; `np.savez` is 34 MB and 19 ms — the fastest is pickle-of-columns, but `np.savez` is the disciplined pick because the format is portable, version-stable, and cross-language.*

**Anti-pattern.** Building a separate persistence layer with its own object model — `pydantic.BaseModel`, ORMs, `dataclasses.asdict` plus `json.dumps`. The translation between the persistence object and the in-memory state is friction; every change to one requires a change to the other; a class of bugs lives in that translation forever. *Plus a Python-specific footnote: pickle protocols change across CPython versions; `protocol=4` is the stable archive choice when the file might be read by a future Python.*

**See also.** 7 (SoA), 35 (boundary is the queue), 37 (the log is the world), 38 (storage systems).

---

## 37 — The log is the world

**Definition.** An append-only log of events is the canonical state; the world's tables are the log decoded into SoA. They share a structure — `(rid, key, val)` triples either way — so replaying the log builds the tables, and serialising the tables produces a log. The two are not analogues; they are two views of one thing.

**Example.** [`.archive/simlog/logger.py`](../.archive/simlog/logger.py) stores rows as three parallel arrays: `rids` (which entity), `keys` (which component code), `vals` (the value, as `f64`). On read, the triples are re-densified into per-field arrays plus presence masks — the canonical SoA-plus-EBP shape. Any simulation that logs every event automatically has a replayable history; recovery is not a separate code path, it is the read path. Three design insights worth naming from the simlog: *batching* (one bulk write per buffer-fill, not per event); *codebook* (string field names interned to small ints, type inference unifies values to f64); *double-buffered revolver* (two containers, atomic swap, background dumper thread — the simulator's `log()` never waits on disk).

**Anti-pattern.** Treating logs as ledger / audit records and the world as the "real" state, with translation code on each side. The translation is friction; it implies impedance mismatch where there is none. *In Python the universal anti-shape is the standard `logging` module: it produces formatted strings (for human diagnostics), not structured triples (for replay). Different tool for different job; the boundary rule sends `logging.info(...)` to the queue, not to the log.*

**See also.** 16 (determinism by order), 30 (1M-to-streaming wall), 36 (persistence is table serialization), 38 (storage systems), 43 (tests are systems).

---

## 38 — Storage systems: bandwidth and IOPS

**Definition.** A storage system is the part of the program that crosses I/O — to disk (HDD/SSD/NVMe), to network, to a service. Its limits are *bandwidth* (bytes per second) and *IOPS* (operations per second), and both must be counted against the tick budget from node 4. SQLite is one specimen; a TCP socket is another; a network filesystem is a third. The pattern — single owner, batched writes, asynchronous flush — is the same across all of them.

**Example.** The §3 streaming simulator's storage system writes the `eaten`/`born`/`dead` logs to disk in batches of 50,000 rows via SQLite WAL-mode + executemany. At 30 Hz with one batch per tick, that is roughly 1.5 million rows/second — well within an SSD's IOPS budget. Compare with one row per `INSERT`: 30 Hz × thousands of events = a different order of magnitude on the IOPS counter (~50K rows/sec at the row-by-row form versus ~1M rows/sec at the batched-transaction form versus ~5M rows/sec at the `INSERT FROM SELECT` form on a populated in-memory source — three orders of magnitude across three forms of the same operation). *From [`code/measurement/sqlite_performance_test.py`](../code/measurement/sqlite_performance_test.py): once warm, on-disk SQLite is only ~9% slower than `:memory:` (906K vs 826K random lookups/sec), which contradicts the common Python intuition that "disk is much slower than memory" — the gap is for cold reads only.*

**Anti-pattern.** Treating I/O as free at the call site. Every row written through a single-row `INSERT` is one IOP; budgets that ignore IOPS hit the floor without warning. *SQL's role in this architecture is at the boundary — for queryable archives of the log, for external inputs/outputs, and for the pandas-OOM migration — not for the simulator's hot path. The hot path is `np.savez` snapshots and the simlog's triple-store.*

**See also.** 4 (cost & budget), 35 (boundary is the queue), 36 (persistence is table serialization), 37 (the log is the world).

---

## 39 — System of systems

**Definition.** Not all systems run every tick to completion. Some computations exceed the tick budget; some run on a different cadence; some live entirely outside the simulator. A system has a *cadence* — every tick, every N ticks, on a deadline, suspended-and-resumed across ticks, or out-of-loop entirely — and the cadence does not have to be one tick. Three patterns handle the cases that do not fit the simple model: *anytime algorithms* (return best-current answer when the deadline arrives), *time-sliced computation* (divide work across ticks, with progress as part of state), and *out-of-loop computation* (run on a separate thread or process, deliver results via the input queue).

**Example.** A path-finding system for a creature has a 5 ms budget per tick. A real path-finder may take much longer for a complex map. The anytime version returns its best partial path at 5 ms; the next tick refines it. A spatial search for the nearest task scans cells across multiple ticks, with `cursor: int` tracking progress. A game AI evolving counter-strategy runs in a separate `multiprocessing.Process`, reads a snapshot every few seconds, and delivers a `strategy_update` event into the simulator's input queue. None of these break the trunk's rules; each respects [§4](../book/trunk/04_cost_and_budget.md)'s budget, [§15](../book/trunk/15_state_changes_between_ticks.md)'s state-as-progress framing, or [§35](../book/trunk/35_boundary_is_the_queue.md)'s queue boundary.

**Anti-pattern.** Forcing every computation into the per-tick model. A path-finder that blocks for 100 ms freezes a 30 Hz simulator for three ticks; a synchronous AI call to a remote service stalls the entire loop on network latency. Both bugs come from refusing to acknowledge that some work has its own cadence; the fix is structural, not algorithmic. *Scale up before scaling out: a network hop costs ~5 ms (data centre) to ~100 ms (internet) per request — 15-300% of a 30 Hz tick budget per hop. Distribute only when one box genuinely cannot hold the workload.*

**See also.** 4 (cost & budget), 11 (the tick), 13 (system as function over tables), 15 (state changes between ticks), 35 (boundary is the queue).

---

## 40 — Mechanism vs policy

**Definition.** The kernel of a system exposes raw verbs. Rules — what is allowed, what triggers what — live at the edges, not in the kernel. Confusing the two is how systems calcify: a kernel that knows about a rule cannot drop the rule without rewriting itself.

**Example.** The simulator's `cleanup` is mechanism: it applies whatever is in `to_remove` and `to_insert`, no opinions. The `food_spawn` system is policy: it decides *when* and *where* food appears, expressed as a set of rules over the `food_spawner` table. Replacing `food_spawn` with a different policy (a fixed schedule, an LLM, a player input) requires no change to `cleanup`.

**Anti-pattern.** Encoding policy decisions in the kernel — `if hungry && food_nearby: eat`. *Three Python-specific forms that bury policy in mechanism: `@property` setters that validate-and-commit (rule lives inside attribute assignment), decorators that hide control flow (`@cache_for(seconds=60)`, `@require_role("admin")` — the function's read-set and write-set are no longer derivable from its signature), `__getattr__`/`__setattr__` overrides (an arbitrary read of `creature.foo` triggers I/O — the boundary from §35 is breached at the most innocuous-looking line).*

**See also.** 13 (system as function), 25 (ownership), 35 (boundary is the queue), 41 (compression-oriented).

---

## 41 — Compression-oriented programming

**Definition.** Write the concrete case three times before extracting. Don't pre-architect. The from-scratch version is also the dependency-pricing test (node 42): most packages lose the comparison because they generalise more than your case requires.

**Example.** A student building three small functions to filter creatures by hunger, by age, and by location is tempted to extract a generic `filter_by` taking a closure. Don't — yet. The three concrete versions are easier to read and benchmark, and they expose what is actually shared. Once a fourth case shows up, the genuine abstraction emerges from the pattern of the four, not from imagined future needs.

**Anti-pattern.** Designing the abstraction before the third use. *Five Python forms of premature abstraction: inheritance hierarchies (`class Creature(Entity, Updatable, Persistable, Drawable)`), `Protocol` interfaces designed before two implementations exist, `*args, **kwargs` "for flexibility" (the function's signature documents nothing), generic helpers parameterised over `Callable` (one helper plus three call sites that all read worse than the three concrete two-liners), and plugin systems with no plugins (carrying the architectural cost of a plugin point that no one is plugging into).*

**See also.** 13 (system as function), 40 (mechanism vs policy), 42 (you can only fix what you wrote).

---

## 42 — You can only fix what you wrote

**Definition.** Foreign libraries are allowed; this is not a prohibition. But every dependency is a bet that someone else will keep it working, and the bet has a cost: if the library is wrong, abandoned, or breaking-changed, you cannot fix it. You can only replace it, fork it, or live with it. The discipline is to take the bet *consciously* — knowing how much code the dependency saves and how much risk it carries.

**Example.** In the multicore track, the student is tempted to add a third-party parallel-iteration package for the parallel-sum opening. The exercise asks them to first write the 50-line manual `multiprocessing.Pool` version (with the §31 rig), time it, then read the package's relevant source. Most students discover the package does about 200 lines more than they need; some still adopt it. The difference is that they now know what they bet on. *The Python escalation order when single-process numpy isn't fast enough: numpy → multiprocessing+shared_memory (§31) → maturin (Rust + PyO3) → leave Python entirely. Skipping straight from numpy to GPU-from-Python disappoints because the orchestration loop stays Python and the GIL stays in the picture for the dispatch.*

**Anti-pattern.** Reaching for `pip install` reflexively, by name recognition or because a tutorial used the package. The dependency arrives with no measurement, no reading, and no appraisal of what its absence would have cost. *The Python-specific traps: pandas (between mid-size and ecosystem-scale, often unquestioned), ORMs (`sqlalchemy`, `peewee`, Django ORM — when the workload is SoA and the ORM is being used out of habit), `pickle` of complex objects (version-fragile across CPython releases), async frameworks (`asyncio`/`trio`/`anyio` each make architectural commitments that propagate through your code).*

**See also.** 38 (storage systems), 41 (compression-oriented programming), 43 (tests are systems).

---

## 43 — Tests are systems; TDD from day one

**Definition.** From the first exercise onward, every concept is approached test-first: *what's the smallest case? what's the largest? what should the answer be for `np.uint8`, for `np.uint32`, for 10,000 agent ids?* Tests are not a separate framework — they are systems that read tables and assert. A test rig is structurally identical to an inspection system. Property tests over component arrays and integration tests by replay log fall out of the structure rather than being a separate effort.

**Example.** §5's first exercise — "build the deck" — has a test: after `new_deck()`, every (suit, rank) pair appears exactly once across the 52 rows. The test is a system: read `suits` and `ranks`; output an assertion result. The same shape is the InspectionSystem pattern: read references to all tables, assertions in test mode, transparency in `--debug` mode, identical code path. *In Python: pytest is fine for the harness; `hypothesis` is the standard property-based-testing tool; `pytest-xdist` running parallel workers surfaces non-determinism that single-process pytest hides ([§16's set-iteration trap](../book/trunk/16_determinism_by_order.md)).*

**Anti-pattern.** Testing as a separate concern bolted on at the end. *In Python specifically: `unittest.mock` is the wrong tool for ECS-style code — the §35 boundary eliminates the things mocks exist to fake (no external services to patch, no `requests.get` to intercept, no clocks to freeze). If you find yourself reaching for `mock.patch`, the system you are testing has a leak from §35; the fix is to plumb the leaked dependency through the queue, not to mock it. The simlog's [`test_simlog.py`](../.archive/simlog/test_simlog.py) (713 lines, full coverage of the simlog's contract) uses zero mocks.*

**See also.** 13 (system as function), 16 (determinism by order), 37 (the log is the world), 41 (compression-oriented).
