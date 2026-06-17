# Through-line Simulator: A Simple Ecosystem

A specification for the simulator the book is written backwards from. It is the *autobiography reference* - every chapter either adds a feature to this simulator or asks a question only it can answer.

This is M2 in `PLAN.md`. The simulator must use every node in `concepts/dag.md` at least once before the book reaches it.

## Premise

A 2D world populated by creatures, with food appearing from §1 onward. On each tick, creatures may:

- **wander** - take a step in a chosen direction; movement *burns* fuel,
- **eat** food they encounter - fuel *tanks*; the food row is removed,
- **reproduce** when their fuel is high - the parent fissions into two offspring, each carrying half the parent's remaining fuel; the parent is consumed,
- **starve** when their fuel runs out - the creature row is removed.

A food-spawning policy at the edge of the world keeps the population from collapsing or exploding. The story of the simulator is a story of *variable-quantity tables under closed-loop control* - births, deaths, and the resulting need for `swap_remove`, dirty markers, generations, and log-orientation.

§0 is a stripped-down first version: 100 creatures wandering on a grid. No food, no fuel, no births and no deaths. Food, fuel, reproduction, and starvation all arrive together in §1.

> [!NOTE]
>
> The *shape* - variable quantity under closed-loop control, with reproduction as a 1→N emission - comes from a different domain. The author was asked, twenty years ago, to simulate a sub-critical fissile assembly with active control rods. The OOP version was painful; the ECS version is much simpler. The book uses an ecosystem instead because every learner has the vocabulary for it; the shape is the same, *including* reproduction-as-fission.

## Why this through-line

- **Universal vocabulary.** Every learner has been taught ecology in school. No prior physics, finance, or networking knowledge required.
- **Variable quantity is the default from §1.** Population grows (reproduction) and shrinks (starvation) every tick. The book's lifecycle machinery (`swap_remove`, dirty markers, generations) is not introduced because the curriculum says so - it is introduced because the simulator stops working without it.
- **All three system shapes appear naturally.** Motion is an *operation* (1→1). Eat and starve are *filters* (1→{0,1}). Reproduce is an *emission* (1→2 in §1, 1→{2,3} sampled in §2). Students meet all three before chapter 4.
- **Discrete event clocks land cleanly.** A creature's next-eat, next-starve, and next-reproduce times carry arbitrary microsecond precision within a 30 Hz loop. The model resolves event time independently of loop rate - exactly the confusion node 12 is written to address.
- **The log is the world.** Every birth, death, and meal is one row in an append-only log. The world's tables are the log decoded; replay reconstructs the population's state.
- **Control is policy at the boundary.** The food-spawn rate is a separate system at the edge - mechanism-vs-policy made visible. The policy can change without touching the kernel.
- **Visceral.** Births and deaths are unambiguous. Students attend.

## Scale spine

The simulator grows with the book. Each scale step adds features and forces a new set of techniques.

| Stage          | Population    | What appears at this stage                                                                       | What it forces                                                                                  |
|----------------|---------------|--------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------|
| §0 - toy       | 100           | motion only on a 2D grid; no food, no fuel, no births, no deaths                                  | identity & structure (nodes 1-10); constant-quantity tables; the card-game milestone applies   |
| §1 - alive     | 10,000        | food, fuel (burns in motion, tanks at food), reproduction (fission-style 1→2), starvation         | variable-quantity arrives; `swap_remove`, dirty markers, lifecycle nodes earn their keep        |
| §2 - crowded   | 1,000,000     | sampled fission (1→{2,3}), spatial structure                                                     | hot/cold splits, working-set discipline, sort for locality                                      |
| §3 - streaming | 100,000,000   | append-only history, sliding windows                                                             | log-orientation; the world becomes a window on the log                                          |

## Initial schema

Field types are indicative; the book may sharpen them as it goes. Some fields and tables appear only at later stages - noted in each row.

### `creature` (constant in §0; variable-quantity from §1)

| field     | type  | from | notes                                          |
|-----------|-------|------|------------------------------------------------|
| `id`      | u32   | §0   | surrogate key                                  |
| `gen`     | u32   | §1   | generation counter (recycling arrives in §1)   |
| `pos`     | f32×2 | §0   | (x, y) on the grid                             |
| `vel`     | f32×2 | §0   | direction × speed                              |
| `energy`  | f32   | §1   | *fuel*: tanks at food, burns in motion         |
| `birth_t` | f64   | §1   | μs since simulation start                      |
| `alive`   | bool  | §1   | dirty marker: live row, or dead hole (§22/§24) |
| `herd`    | u32   | §28  | which herd the creature follows (pack-leader cohesion) |

### `food` (variable-quantity, from §1)

| field   | type  | notes                              |
|---------|-------|------------------------------------|
| `id`    | u32   |                                    |
| `pos`   | f32×2 |                                    |
| `value` | f32   | fuel yielded when eaten            |

### `food_spawner` (constant-quantity, from §1)

| field    | type  | notes                              |
|----------|-------|------------------------------------|
| `id`     | u8    |                                    |
| `region` | f32×4 | bounding box                       |
| `rate`   | f32   | food per second                    |

### `pending_event` (variable; rebuilt each tick; from §1)

| field         | type | notes                                |
|---------------|------|--------------------------------------|
| `t`           | f64  | event timestamp                      |
| `kind`        | u8   | eat / reproduce / starve             |
| `creature_id` | u32  |                                      |
| `target_id`   | u32  | food id for eat; unused otherwise    |

### Append-only logs (EBP and history; from §1)

`eaten`, `born`, `dead` - one row per event. These are simultaneously the world's history and the input to replay.

### Dirty markers (lifecycle, applied at tick boundary; from §1)

`to_remove: Vec<u32>` - creature ids slated for removal.
`to_insert: Vec<CreatureRow>` - fresh creatures from reproduction.

### Index map (§23; maintained only by cleanup)

`id_to_slot: u32[]` - maps a stable creature id to its current slot, with a sentinel for "absent". The EBP appliers hold ids (from `pending_event` and `to_remove`) and *read* this map to reach the columns; no hot-path system writes it. Only `cleanup` writes it: an append adds one entry, a death drops one entry, and the GC compaction rewrites every survivor's entry in one bulk pass.

### Population log (visualisation; from §0)

`population: Vec<(t, count_creatures, count_food)>` - one row per tick, written by `inspect`. The basis for the canonical population graph below.

## Systems

| Name              | Read-set                                          | Write-set                                              | Shape       | From |
|-------------------|---------------------------------------------------|--------------------------------------------------------|-------------|------|
| `herding`         | `creature.pos`, `creature.herd`, `creature.birth_t` | `creature.vel`, `creature.herd` (on a split)         | operation   | §28  |
| `motion`          | `creature.pos`, `creature.vel`                    | `creature.pos`, `creature.vel`, `d_energy_burn`        | operation   | §0 (energy from §1) |
| `food_spawn`      | `food_spawner`, `food`                            | `food`                                                 | operation (policy) | §1   |
| `next_event`      | `creature`, `food`                                | `pending_event`                                        | operation   | §1   |
| `apply_eat`       | `pending_event` (kind=eat), `food`                | `to_remove`(food), `creature.energy`, `eaten`          | filter      | §1   |
| `apply_reproduce` | `pending_event` (kind=reproduce), `creature`      | `to_remove`(parent), `to_insert`(offspring), `born`    | emission (1→2 in §1; 1→{2,3} in §2) | §1 |
| `apply_starve`    | `pending_event` (kind=starve)                     | `to_remove`(creature), `dead`                          | filter      | §1   |
| `cleanup`         | `to_remove`, `to_insert`                          | `creature`, `food`                                     | meta        | §1   |
| `inspect`         | all                                                | `population`                                           | debug-only  | §0   |

System DAG (per tick, from §1):

```
food_spawn
  └── motion
        └── next_event
              ├── apply_eat
              ├── apply_reproduce
              └── apply_starve
                    └── cleanup
                          └── inspect
```

In §0, only `motion` and `inspect` exist; `inspect` runs last and reads only.

## Final architecture (reference implementation)

The reference implementation is a single commented script, [`sim.py`](https://github.com/root-11/intro-book-python/blob/main/code/sim/sim.py) (`uv run code/sim/sim.py`). It is the endpoint the Part 2-5 chapters build toward, and it fixes the following decisions.

- **SoA, fixed capacity, live prefix.** Every field is its own typed column, allocated once at capacity. The live table is the prefix `[0:n_active]`; the `alive` dirty marker distinguishes live rows from dead holes awaiting compaction (§7, §21, §24).
- **Ids, not slots, for anything that outlives a tick.** Creatures carry a stable `id`; buffers and events reference creatures by id. Slots move under `swap_remove` and compaction; ids do not (§9, §10). A slot-only design (no surrogate id) was considered and rejected: it makes every buffered reference brittle at the first compaction.
- **The index map is read in the hot path, written only by cleanup.** `id_to_slot` turns an id into a slot in one bulk gather. The EBP appliers read it; the GC pass maintains it (§23).
- **Mutations buffer; cleanup commits.** Systems append to `to_remove` (ids) and the parallel insert columns; nothing mutates a live table mid-tick. Cleanup applies the batch at the boundary (§15, §22).
- **Deferred GC.** A death marks its slot dead every tick (bump generation, drop the `id_to_slot` entry, flip `alive`); the columns are compacted on a slower cadence, and only that compaction rewrites `id_to_slot` (§22, §24).
- **Food earns less machinery than creatures.** No reference to a food row survives the tick it is eaten in, so food has no surrogate id, no generation, and no index map; it is bulk-filtered every tick. Reference lifetime decides which machinery a table earns (§10).
- **Deterministic.** One seeded RNG plus the fixed system order give a reproducible run (§16); `sim.py --check` asserts two runs are identical.
- **Energy is multi-writer, so it buffers.** `motion` burns and `apply_eat` refuels, so neither writes `energy` in place; each writes a delta buffer and `cleanup` commits `energy += burn + gain` at the boundary (§15). `pos`/`vel` have `motion` as sole writer, so they are written in place (the §15 exception).
- **Proximity is a pack-leader, not all-pairs (§28).** `herding` steers each creature toward its herd's eldest (one position read per creature, O(N)), and splits a herd past `max_herd`. The naive O(C×F) eat query remains for clarity; binning is §28's other half.
- **Persistence is serialization (§36).** `save_world`/`load_world` write the SoA columns with `np.savez` and read them back bit-for-bit. No ORM, no schema migration.
- **The log is the world (§37).** Births/deaths/meals append to logs; replaying births minus deaths reconstructs the live population bit-for-bit. `sim.py --log DIR` routes the log through the production columnar logger and replays from disk.
- **Event time is separate from tick time (§12).** Events carry sub-tick timestamps: a starvation is logged at `energy / burn_rate` seconds into the tick, not at the 33 ms boundary, so the recorded time is independent of the loop rate. Eat and reproduce are instantaneous (t = 0), so eating beats starving by the clock, not by a hand-picked priority.
- **Tests are systems (§43).** `tests.py` runs `check_invariants` - a read-only pass over the tables, the same shape as a system - after *every* tick, plus behaviour tests for determinism, replay, save/load, sub-tick event time, GC reclamation, and the herd split. `uv run code/sim/tests.py`.

## Visualisation: the population graph

The canonical output of the simulator is a *time-series plot of the population size*. Every tick, `inspect` appends the current creature count (and food count, from §1) to the `population` log. After the run, the student plots that log as a line chart.

This is enough visualisation for every stage of the book. It is also one of the cleanest data-viz exercises available: the inspect system writes a tidy three-column table; the plot is a one-liner.

The population graph doubles as the simulator's *regression test*: a stable closed-loop population is a passing run; a population that explodes or collapses is a failing run. Students who tune the food-spawn rate (a policy at the boundary) can watch the curve change in real time.

Other visualisations (a 2D heatmap of creature density, a real-time window) are optional and arrive later, if at all.

## What this simulator is not

- A correct biology simulation. Fuel and food work like accounting balances, not metabolism. Geometry is a 2D box. No metabolism, no genetics, no learning, no behavioural variation.
- A teaching tool for ecology. Population dynamics will emerge, but they are not the focus.
- A game. There is no player.

The point is the *shape*. The simulator is the canonical case for every concept in the book - nothing more, nothing less.

## Extensions for the enthusiastic student

Deliberately *not* in the main book. These are exercises for the student who wants to push further.

- **Predators and prey.** Add a `predator` table with its own motion, hunting, and reproduction. Trophic dynamics emerge. The student exercises every concept twice in the same simulator - once with herbivores, once with carnivores - which is the surest way to know they have understood, not memorised.
- **Sexual reproduction.** Reproduction requires two creatures to meet. Emission becomes collision-mediated rather than threshold-mediated, exercising a different shape of the same node.
- **Genetics.** Each creature carries a small genome; offspring inherit with mutation. Selection often favours phenotypes the student did not intend. The result is usually surprising and educational.
- **Policy-driven wandering.** The motion system reads a per-creature policy table. Connects directly to the multi-agent track.

## Resolved decisions

1. **§0 minimum schema.** §0 has motion only - no food, no fuel, no lifecycle. Food, fuel, reproduction, and starvation all arrive together in §1.
2. **Reproduction trigger.** Energy threshold (asexual). Movement burns fuel; reproduction consumes the parent and produces 2 offspring carrying half the parent's remaining fuel each. This is the fission shape - one row in, multiple rows out, parent consumed. §2 generalises to a sampled 2-or-3.
3. **Visualisation.** A time-series plot of population size, generated from the `inspect` system's per-tick `population` log. Doubles as the simulator's regression test.
4. **Energy.** Fuel metaphor: tanks at food, burns in motion. Carried from §1 onward; absent from §0.
