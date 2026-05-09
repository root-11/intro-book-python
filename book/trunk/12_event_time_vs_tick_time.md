# 12 — Event time is separate from tick time

> *Concept node: see the [DAG](../../concepts/dag.md) and [glossary entry 12](../../concepts/glossary.md#12--event-time-is-separate-from-tick-time).*

Most beginners assume the loop's frequency sets the model's time resolution. If the loop runs at 30 Hz, surely the model can only resolve events at 1/30 s = 33 ms? This is wrong, and the confusion costs many simulations their precision.

The tick rate is *how often the loop runs*. It says nothing about what the loop does inside one tick. Inside one tick, the loop can process events at arbitrary timestamps — microsecond, picosecond, whatever the data carries. The clock lives on the events, not on the loop.

Concretely: a 30 Hz loop receiving 1,000 events per tick, each with microsecond-precision timestamps, processes them in timestamp order — applying each event's effect with the precision the timestamp implies. Output to the rest of the world (rendering, logging, network) happens at 30 Hz, but the *physics inside* runs at microsecond resolution. The tick is a *sampling* rate; the events are the actual phenomena.

This is the model used by:

- **Discrete-event simulators** (queueing networks, traffic, supply chains): events fired at exact times.
- **Game replay systems** (rollback netcode, multiplayer): events arrive late but with their original timestamps.
- **Trade execution engines**: orders carry nanosecond timestamps; the loop processes them in order.
- **Logic simulators** in chip design: gate transitions at picosecond resolution; the simulator advances one transition at a time.

In each case, the tick rate of the host loop is irrelevant to the simulation's resolution. The data carries the time.

## How time wants to be stored

The Python reflex when a chapter mentions "timestamps" is to reach for `datetime`. It is the obvious choice — the standard library provides it, every tutorial uses it, comparisons work with `<` and `>`, subtractions return a readable `timedelta`. It is also one of the most expensive ways to store time at scale.

From [`code/measurement/event_time_storage.py`](../../code/measurement/event_time_storage.py), one million events covering an hour at microsecond resolution, on this machine:

| layout                               |  data    | build   | sort    | count <T |
|--------------------------------------|---------:|--------:|--------:|---------:|
| `list[datetime]`                     |  53.6 MB |  406 ms |  8.5 ms | 22.1 ms  |
| `np.array(dtype="datetime64[us]")`   |   7.6 MB |  209 ms |  6.1 ms |  1.3 ms  |
| `np.array(dtype=np.float64)` (sec)   |   7.6 MB |   86 ms | 36.7 ms |  1.3 ms  |

The headline numbers, both ways:

- **7× smaller** footprint moving from `datetime` list to either typed numpy column. Each `datetime` instance is ~56 bytes (header, refcount, eight integer fields, pointer); each numpy element is 8 bytes (an `int64` micro-since-epoch under `datetime64[us]`, or a `float64` second-from-base for the `f8` representation).
- **17× faster** count of "how many events happened before time T?" — the per-tick query that decides what gets processed this tick. The numpy versions evaluate the comparison as one bandwidth-bound bulk op; the datetime version pays per-element interpreter dispatch and a `<` method call.
- Sort time is mixed and dtype-sensitive — measure your specific case. On this run numpy's float64 sort was slower than its datetime64 sort, which was slightly faster than Python's Timsort on the already-sorted datetime list. Sort cost matters for ingestion; count cost matters per tick. The tick is the binding budget.

The simlog reference implementation (vendored at [`.archive/simlog/logger.py`](../../.archive/simlog/logger.py)) stores time as `f8` — float64 seconds. That is the disciplined choice for an event log: small, sortable, amenable to bulk numpy ops, and the same width as everything else in the column store. `datetime64[us]` is a reasonable alternative when you need to read the timestamps as wall-clock dates without conversion. Use `datetime` objects only at the boundary — formatting a string for a log line, comparing against a user-supplied timestamp from a request — never as your in-memory storage at simulation scale.

## The decoupling, in code

The pitfall is hard-coding the tick interval as the simulation's clock granularity. Code that says

```python
# anti-pattern: bad!
creature.energy -= 1.0 / 30.0  # "one tick worth of fuel"
```

is conflating the two clocks. The right shape is

```python
energy[mask] -= elapsed_event_seconds * burn_rate[mask]
```

using the actual elapsed event-time, not the tick interval. The numpy form is also column-shaped — `mask` is a boolean filter selecting the affected creatures, `burn_rate` is per-creature. The same computation works for one event affecting one creature and a thousand events affecting a thousand creatures, because *event time and tick time are decoupled*. The same model can be sampled at any tick rate the application needs — visualisation at 30 Hz, recording at 60 Hz, fast-forward replay at 1 kHz — without changing what the model means.

This separation is what makes the simulator's `pending_event` table possible. Each tick, the loop builds a list of events that should fire — collisions, eats, reproductions — each tagged with its predicted timestamp as an `f8`. The events fire in timestamp order regardless of which tick they were *predicted in*. A creature that "would have eaten 2 µs into the tick" has its eat applied at that exact moment, not at the start or end of the tick.

## Exercises

These extend the discrete-event loop from §11 exercise 7.

1. **A tiny event queue.** Use `numpy` arrays: `times = np.array([...], dtype=np.float64)` of timestamps and `messages = np.array([...], dtype=object)` of strings. Push 10 events with random timestamps in `[0, 10]` seconds. Pop them in time order using `order = np.argsort(times)`. Print each as `[t=<sec>] <message>`. Verify the output is timestamp-sorted.
2. **The wrong way: tick-rate clock.** Run a 30 Hz loop. In each tick, advance a counter by `1.0 / 30.0`. Use this counter as your "simulation time". Try to fire an event at `t = 0.005 s` (5 ms). What happens? When does the event fire? (Hint: 5 ms < 33 ms; the event waits for the next tick boundary, losing 28 ms of resolution.)
3. **The right way: timestamp on events.** Run the same 30 Hz loop, but each tick pop *all* events with timestamp ≤ current real time, applied in timestamp order. Fire an event at `t = 0.005 s`. Show that the event applies at exactly that time, not at the next tick boundary.
4. **Sampling at different rates.** Run the same model under a 30 Hz loop, then a 60 Hz loop, then a 1 Hz loop. The events should fire at the same simulation times in all three runs (down to whatever precision the loop allows).
5. **Float and time.** What is the smallest time step `np.float32` can represent for events at `t ≈ 1 hour`? At `t ≈ 1 day`? At `t ≈ 1 year`? When do you need `np.float64`? (See [§2](02_numbers_and_how_they_fit.md). Hint: `np.spacing(np.float32(3600))` is a fast way to find the answer for one hour.)
6. **Run the storage exhibit.** `uv run code/measurement/event_time_storage.py`. Note the count-time row — that is the per-tick query cost in three layouts. Note where the `datetime` list lands and where the numpy columns land.
7. *(stretch)* **A budget-aware loop.** Modify your 30 Hz loop: at the start of each tick, pop events until either (a) the queue is empty or (b) you have used 25 ms of the 33 ms budget. Defer remaining events to the next tick. This is the soft-real-time pattern used in interactive simulators.

Reference notes in [12_event_time_vs_tick_time_solutions.md](12_event_time_vs_tick_time_solutions.md).

## What's next

[§13 — A system is a function over tables](13_system_as_function.md) introduces the building block of every tick: the system. Read-set in, write-set out, no hidden state, no surprises.
