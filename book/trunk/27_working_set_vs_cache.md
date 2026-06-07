# 27 - Working set vs cache

> *Concept node: see the [DAG](../../concepts/dag.md) and [glossary entry 27](../../concepts/glossary.md#27--working-set-vs-cache).*

<p align="center"><img src="../illustrations/bridge_clipboard.jpg" alt="Engineer mouse with clipboard - load capacity is what fits in the working set" style="max-height: 300px; max-width: 100%;"></p>

The *working set* of a loop is the data it touches per pass. The *cache hierarchy* (§1) is what holds that data. The two together decide the loop's speed - *once you are in numpy*. In pure Python, the interpreter-dispatch tax dominates and the cliff is invisible. The moment your inner loop drops into a bulk numpy op, the cliff is real and exactly where the hardware says it is.

Sequential numpy access stays bandwidth-bound and cheap at every size - the prefetcher reaches forward and amortises the cost. The cliff is in *random* (gather) access, where each element is an unpredictable jump the prefetcher cannot hide. The exact ns depend on the chip; the numbers below are what this machine does, measured in [§1's `cache_cliffs.py` exhibit](https://github.com/root-11/intro-book-python/blob/main/code/measurement/cache_cliffs.py) (3-run medians):

| N           | numpy seq | numpy gather | gather/seq |
|------------:|----------:|-------------:|-----------:|
|     10,000  |  0.65 ns  |   3.07 ns    |    4.7 ×   |
|    100,000  |  0.37 ns  |   2.01 ns    |    5.4 ×   |
|  1,000,000  |  0.21 ns  |   3.53 ns    |   17.0 ×   |
| 10,000,000  |  0.15 ns  |  10.06 ns    |   66.0 ×   |
|100,000,000  |  0.15 ns  |  11.72 ns    |   80.0 ×   |

The cliff is in the gather column. The 10K and 100K rows fit in L1 / L2 (gather ratio ~5×); 1M sits in L3 (~17×); the 10M and 100M rows spill to RAM (ratio 66-80×). The numpy *sequential* row stays flat under ~0.7 ns/element throughout - that is what bandwidth-bound looks like on this machine. (The sub-nanosecond seq column is noisy run-to-run; the gather column and the RAM ratios are the stable claims.)

## Computing your working set

The arithmetic is mechanical. Motion's inner loop reads `pos_x: float32 = 4 bytes`, `pos_y: float32 = 4 bytes`, `vel_x: float32 = 4 bytes`, `vel_y: float32 = 4 bytes`, `energy: float32 = 4 bytes`. Total: 20 bytes per creature. At N creatures, working set = 20 × N bytes.

| N           | working set | regime (this machine)            |
|------------:|------------:|----------------------------------|
|       1,000 |        20 KB | fits L1                         |
|      10,000 |       200 KB | fits L2                         |
|     100,000 |         2 MB | borderline L2/L3                |
|   1,000,000 |        20 MB | fits L3, spills L2              |
|  10,000,000 |       200 MB | spills L3, hits RAM             |

Each transition costs roughly 3-5× in per-element time when the access pattern is random. Sequential access is largely insulated by the prefetcher, but only up to RAM bandwidth - at 10M creatures and beyond, the prefetcher is no longer hiding latency, just keeping pace with what RAM can deliver.

This is what [§4](04_cost_and_budget.md)'s "cliff" was about, made concrete for your simulator. The transition points are not magic - they are arithmetic over your cache sizes. From [§1 exercise 1](01_the_machine_model.md#exercises) you have those numbers written down.

## Why this lesson still matters when numpy hides it

Most numpy code never thinks about cache size because the inner loops are bandwidth-bound and "fast enough." That intuition holds until the working set leaves L3 - at which point per-element cost rises 5-10× *with no change to the source code*. A simulator written for 1M creatures and tested at 100K never notices the cliff; it shows up the day the simulator is sized to 10M and the deadline is missed.

What sets the working set is which columns a system reads, and SoA already decides that for you ([§26](26_subscription_tables.md)). Motion reads `px, py, vx, vy, energy` - five columns, 20 bytes per creature - and never touches `birth_t` or `species`, because those are different arrays entirely. There is no full row to trim and no hot/cold split to apply: in numpy SoA a system's footprint is its read-set, already. The levers that move the cliff are reading *fewer* columns and storing them in *narrower* dtypes ([§2](02_numbers_and_how_they_fit.md)), not regrouping fields.

## Design discipline

- Decide the target N before the schema. The schema must fit the cache that fits N.
- Audit the inner loops. Sum the bytes per row touched. Compare to your cache sizes.
- When you cross a transition, *measure* - do not assume. The prefetcher and the OS will sometimes save you, sometimes not. Numpy's bulk-op threshold also shifts with version; benchmark on the exact stack you ship.
- The narrowest dtype that holds the value ([§2](02_numbers_and_how_they_fit.md)) is not aesthetic; it is the cliff's distance. `np.float32` over `np.float64` doubles the headroom; `np.uint8` for indices from 0 to 255 packs 64 to a cache line.

This is not premature optimisation. It is *layout-aware design* - making the schema fit the machine that will run it. A schema that ignores the cache works for small N and breaks at the scales the simulator was meant for.

## Exercises

1. **Compute your working sets.** For each system in your simulator, compute `bytes per row × N` for N = 1K, 10K, 100K, 1M, 10M. Note which cache level each falls into on your machine (use `lscpu | grep -i cache` from §1 exercise 1).
2. **Find your cliff.** `uv run code/measurement/cache_cliffs.py` (the §1 exhibit) gives you ns/element across sizes for sequential and gather access. Plot the gather column. The transitions should match your cache sizes.
3. **The unused column costs nothing.** Add two columns motion never reads (`birth_t: float64`, `species: uint8`). Recompute motion's working set and re-time motion at the cliff size from exercise 2. It should not move: in SoA the extra columns are separate arrays motion never loads. This is why there is no hot/cold split to apply in numpy ([§26](26_subscription_tables.md)) - a system's footprint is already just the columns it reads.
4. **A wider dtype.** Change `energy: float32` to `energy: float64`. Recompute the working set. Time motion. The cliff should move inward (closer to smaller N).
5. **Random vs sequential, your machine.** Re-read the gather/seq ratio in the cache_cliffs table for *your* output. The factor ~5× → 80× growth across sizes is your machine's cache-vs-RAM cost gap. Memorise this number; it is the answer to "how much does a random access cost compared to a sequential one on this hardware?".
6. *(stretch)* **The L1 sweet spot.** Find the N at which motion's working set fills L1 to roughly 75%. Run the motion loop in tight repetition (call it 1,000 times in a row, no other work between calls). The L1-resident loop should run at a stable ~0.2 ns/element for the entire run. The closest L2-only neighbour should be 3-5× slower.

Reference notes in [27_working_set_vs_cache_solutions.md](27_working_set_vs_cache_solutions.md).

## What's next

[§28 - Proximity is a property of position](28_proximity.md) computes neighbour structure from the position stream, and its cell-ordered compaction (with §26) makes the spatial gather sequential.
