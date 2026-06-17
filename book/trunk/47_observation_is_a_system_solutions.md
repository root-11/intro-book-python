# 47 - Observation is a read-only system - reference notes

## Exercises 1-2: a metrics system, and proving it read-only

The metrics system is one vectorised pass: `population = world.n_active`, `mean_e = world.energy[:n].mean()`, append the row to a `metrics` array indexed by tick. Read-set: the world columns. Write-set: the `metrics` array, which nothing else touches.

The proof is a hash. Run 1,000 ticks with the metrics system in the DAG, hash the world; run again with it removed, hash again. They must be identical, because a disjoint write-set ([§31](31_disjoint_writes_parallelize.md)) cannot perturb the world. If they differ, the observer wrote a column it should not have - find it. The simulator's own `inspect` is exactly this system, and the determinism gate already proves it does not perturb (the run is bit-identical with it running).

## Exercise 3: measure the thermometer

Time the tick with metrics on and off; vectorised reductions cost a fraction of a percent and the two are indistinguishable. Now make the metrics system `np.sort(world.energy[:n])` each tick to report a median, and the reported tick time climbs - the thermometer is now heating the water. `np.partition` (O(N), no full sort) or a streaming quantile estimate recovers the budget. The Python-specific trap underneath: a metrics reduction written as a Python `for` loop is interpreter-bound and shows against the tick immediately; keep it in numpy.

## Exercises 4-5: trace, and structured query

A trace is a filter, not new storage: query the [§37](37_log_is_world.md) log for one entity id across the tick range and read its life off the events. A structured-log query - "every tick where population fell more than 10%" - is a vectorised diff over the population column; the same question over `print()` text is `grep` and hope, because the events were flattened to strings that no longer carry their types. A string is for one human reading one line; a structured event is for a system reading a million.

## Exercises 6-7: an alert is a system; behind the queue

The alert reads the `metrics` table and emits a threshold crossing (population zero, tick-over-budget for T ticks) - one more read-only system, its read-set another system's write-set. Shipping metrics out through the [§35](35_boundary_is_the_queue.md) queue and pausing the sink shows the inversion: the tick rate is unaffected and samples are *dropped*, not stalled. Backpressure on observability degrades observability, never the system.

## Exercise 8: the guaranteed metric

A counter you must not lose - a billing total - is not fire-and-forget; it obeys the [§46](46_log_survives_power_loss.md) rule. Do not advance the "reported" watermark until the sink confirms the batch durable, and on restart resend from the watermark. Contrast the cost: the fire-and-forget metric pays nothing and may gap a chart; the guaranteed one pays an `fsync`-class confirmation per batch and may never lose a count. The decision is per metric: a dropped frame-rate sample is a gap, a dropped charge is theft.
