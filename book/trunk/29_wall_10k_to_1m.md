# 29 — The wall at 10K → 1M

> *Concept node: see the [DAG](../../concepts/dag.md) and [glossary entry 29](../../concepts/glossary.md#29--the-wall-at-10k--1m).*

<p align="center"><img src="../illustrations/hard_hat_repeat.jpg" alt="Construction mouse — scale up the build, MEASURE / CALCULATE / DESIGN / BUILD / REPEAT" style="max-height: 300px; max-width: 100%;"></p>

A simulator that runs cleanly at 10,000 creatures often grinds to a halt at 1,000,000. Not because the algorithm changed — because constant factors that were invisible at the smaller scale now bind.

This chapter is about *finding the wall*. The fixes are techniques you already have: hot/cold splits ([§26](26_hot_cold_splits.md)), working-set discipline ([§27](27_working_set_vs_cache.md)), sort for locality ([§28](28_sort_for_locality.md)), pre-sized buffers, batched cleanup. The chapter's job is to teach the reader to *measure* — to find which constant factors blew up.

## Walls Python hits, named

- **Pre-allocation skipped.** A `to_insert: list[CreatureRow]` that grew lazily was fine at 100 appends per tick (10K creatures × 1% reproduction). At 10K appends per tick (1M × 1%), Python list `append` is amortised O(1) but each capacity doubling is an N-byte copy; at this scale the doublings dominate. Fix: pre-size with `[None] * estimated_max` plus an `n_inserts` counter, the same pattern §22 already uses.
- **Linear scans in pure Python.** A list comprehension `[c for c in creatures if c.id == target_id]` was 0.1 ms at 10K, but tens of milliseconds at 1M. Fix: the `id_to_slot` map ([§23](23_index_maps.md)) plus parallel presence flags. *In Python the linear-scan cost is sharper than in Rust* — you pay interpreter dispatch on every iteration, ~5 ns per step from §1.
- **Cache spillover.** A `creature` working set at 10K is 200 KB (L2-resident). At 1M it is 20 MB (L3-resident). Per-element time triples. Fix: hot/cold splits + narrower numpy dtypes.
- **The pandas wall.** A `pandas.DataFrame` of 10M rows × 20 columns at default dtypes occupies 1.6 GB+ before any operation. A `DataFrame.merge` allocates intermediate copies; a `groupby.apply` materialises Python objects per row; both can OOM long before the data itself would. Fix: drop pandas. Either move to numpy SoA (when the working set still fits in RAM with explicit columns) or to **sqlite** (when it doesn't, or won't long-term). [`code/measurement/sqlite_performance_test.py`](https://github.com/root-11/intro-book-python/blob/main/code/measurement/sqlite_performance_test.py) shows sqlite delivers ~830K-900K random lookups per second on disk — fast enough to be the production answer for many workloads that pandas was struggling with. **The migration is usually a one-day project that gives back days of OOM debugging per quarter.**
- **Per-tick allocation.** A system that calls `np.zeros(N)` per tick was fine when N was 10,000 (40 KB). At N = 1,000,000 it is 4 MB allocated and zero-filled every tick — the malloc cost alone is significant. Fix: allocate the buffer once at startup, fill or reuse in place.
- **Logging.** A `print(f"creature {i} ate")` per event was tolerable at 10K. At 1M events it is the simulator's bottleneck — `print` flushes, formats, dispatches the GIL. Fix: write to a numpy event log per [§37](37_log_is_world.md), flush in bulk; or simply turn it off.

The pattern: any cost that was O(1) per creature, multiplied by 1M, is no longer free. Anything that was O(N) per tick at 10K is now O(N²)-equivalent in wall time. The fixes are local — each cost is a single-line change — but finding them requires measurement.

## Measurement tools

The right tool is a profiler. In Python, three good options:

- **`cProfile`** (stdlib). `python -m cProfile -o profile.out my_sim.py` records every Python-level function call. Read with `python -m pstats profile.out` or `snakeviz`. Fine for finding hot Python functions; opaque to numpy internals (numpy ops show up as one C call).
- **`py-spy`** (third-party). `py-spy record -o flame.svg -- python my_sim.py` produces a flame graph similar to `perf`. Sees the C stack inside numpy ops, which `cProfile` does not. The right tool when the bottleneck is *inside* numpy.
- **`perf`** (Linux). The same tool the Rust edition uses. `perf record -- python my_sim.py; perf report` reads at the OS level; sees everything but interprets nothing — you read raw symbols.

The same simulator at 10K and 1M produces different flame graphs; the wall is the difference.

## Calibration

A useful exercise: run your simulator at 10K for 1,000 ticks; time it. Run at 1M for 100 ticks (same total entity-ticks); time it. The 1M version should take **roughly 10× longer**, not 100×. If it takes 100×, something has crossed a constant-factor wall and the profiler will show you what.

The fix is structural. Apply the techniques: hot/cold, working set, sort for locality, pre-sized buffers, batched cleanup, deterministic structures. Each is a chapter you have already read. The wall is the moment they all become non-optional.

## Exercises

1. **Calibration.** Run your simulator at N = 10,000 for 1,000 ticks. Time it. Note the wall-clock total.
2. **Scale up.** Run at N = 1,000,000 for 100 ticks (same total entity-ticks). Time it. Compute the ratio.
3. **Profile with cProfile.** `python -m cProfile -s cumulative my_sim.py | head -30`. Identify the top three hottest functions.
4. **Profile with py-spy.** `py-spy record -o flame.svg -- python my_sim.py`. Open the flame graph in a browser. Identify hot regions inside numpy that `cProfile` did not surface.
5. **Pre-size cleanup buffers.** Replace `to_insert = []` plus `to_insert.append(...)` with a pre-sized array plus an `n_inserts` counter (the §22 pattern). Re-run; re-profile. The list-resize calls should disappear from the hot list.
6. **Hot/cold split.** Apply the [§26](26_hot_cold_splits.md) split organisationally. Re-run; re-profile. In numpy SoA you may see no change in the profile (per §26's framing); in numpy structured-array form you should see a visible improvement.
7. **Use index maps.** Replace any linear `np.where(arr == target)[0]` lookup with the [§23](23_index_maps.md) `id_to_slot` form. Re-run; re-profile.
8. **The pandas wall, hands-on.** Build a pandas DataFrame of 5M rows × 10 float64 columns. Note its memory (`df.memory_usage(deep=True).sum() / 1e6` MB). Now move the same data into 10 numpy `float32` columns; note the memory ratio. Now move it into a sqlite table; note the disk size and a sample lookup time using `sqlite_performance_test.py` as a template. Decide consciously which form fits your workload.
9. *(stretch)* **Find one new wall.** Pick any system in your simulator and find one constant factor that scales worse than expected. The fix is usually one of the techniques above; identifying *which* one is the lesson.

Reference notes in [29_wall_10k_to_1m_solutions.md](29_wall_10k_to_1m_solutions.md).

## What's next

[§30 — Moving beyond the wall](30_streaming_wall.md) takes the next step: when even your fastest, tightest, hot/cold-split, sorted-for-locality simulator no longer fits in RAM, the architecture itself shifts.
