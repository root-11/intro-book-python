# 49 - The worst case is the only case - reference notes

## Exercises 1-2: measure jitter, and watch the collector

An empty tick loop at a fixed period, recorded with `time.perf_counter` over a few million iterations, has a tight mean and a long ugly tail. The p99.9 and max are tens to hundreds of times the mean on a stock desktop - and hard real-time counts the max, not the mean. Re-run with `gc.disable()` for the measured window and the tail shrinks; force a `gc.collect()` mid-run and the collection event appears *in* the tail. That is the lesson made visible: a runtime that can pause to walk the object graph at any allocation cannot offer a worst-case bound. You can shrink the tail; you cannot prove it bounded.

## Exercises 3-4: allocation is a spike; hunt the unbounded operation

Add one `[0] * 1000` per tick and the tail grows; remove it and it shrinks - the numpy-columns, no-per-element-allocation discipline ([§7](07_structure_of_arrays.md)) was buying tail latency all along. Auditing a system for unbounded operations turns up the usual Python suspects: a `dict` that can rehash, a `list` that reallocates, a data-dependent loop, an `f"{...}"` that allocates a string. Each is a spike in the tail. Replacing one with a pre-sized numpy array and re-measuring shows it flatten - but never to a guarantee.

## Exercises 5-6: the cache is a worst case; soft, not hard

Run a system over L1-sized data and over RAM-sized data ([§27](27_working_set_vs_cache.md)) and compare the *maxima*, not the means: the cache miss is the worst case, and average-case layout tuning gives you a good mean, not a WCET. Writing down the honest worst-case time of your anytime system, you find CPython gives you none you can prove - name the three reasons: the GC pauses, the GIL, the allocator. Conclude the deadline it may be trusted with (soft: a dropped frame, a coarser answer) and the one it may not (hard: a brake, a motor current, a flight surface). If a miss is a fault, the loop is not Python; Python sits outside it.

## Exercise 7: degrade gracefully (the soft side)

The priority order is not arbitrary: shed in increasing order of how much the world depends on the work. The inspection system writes nothing the simulation reads, so it goes first. The GC's cadence can stretch because dead slots linger harmlessly. Deferring reproduction is the strongest lever because it is *back-pressure*: the births grew the population that overran the budget, so deferring them reduces the next tick's load - the shed attacks the cause, and the system walks itself back under budget.

Two properties must hold. **Integrity survives**: because mutation is buffered ([§22](22_mutations_buffer.md)) and committed at the tick boundary, a long or shed tick applies a whole, consistent world - you observe a late world, never a torn one. **The run still replays**: the shed must be a *logged decision*, not `if over_budget:` branching on the wall clock, or the run stops being a function of its inputs and the [§37](37_log_is_world.md)/[§48](48_reductions_dont_parallelize_freely.md) replay guarantee evaporates - and that bug is invisible until the slower machine sheds differently and diverges. Bounded staleness is the last check: each deferral has a fixed horizon, so the degraded world is never more than a known distance from the budget-met one, and it heals when load drops. Soft real-time managed this way is not "it got slow"; it is "it chose, predictably and reversibly, what to slow."
