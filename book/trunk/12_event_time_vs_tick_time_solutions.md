# Solutions: 12 — Event time is separate from tick time

## Exercise 1 — A tiny event queue

```python
import numpy as np

rng = np.random.default_rng(0)
times    = rng.uniform(0.0, 10.0, size=10).astype(np.float64)
messages = np.array([f"event_{i}" for i in range(10)], dtype=object)

order = np.argsort(times)
for t, m in zip(times[order], messages[order]):
    print(f"[t={t:.3f}] {m}")
```

The `order` array is the only thing that's "sorted." `times` and `messages` are unchanged. The decoupling pattern from §5: data lives in columns; iteration goes through an index array.

## Exercise 2 — The wrong way: tick-rate clock

```python
import time
TICK_S = 1.0 / 30.0

# anti-pattern: bad! "simulation time" advances in tick-sized steps
sim_time = 0.0
target   = 0.005       # 5 ms event
fired    = False

while sim_time < 1.0 and not fired:
    if sim_time >= target:
        print(f"event fired at sim_time={sim_time:.4f}")
        fired = True
    sim_time += TICK_S                  # 33 ms granularity
    time.sleep(TICK_S)
```

Output: `event fired at sim_time=0.0333` — 28 ms late. The event's "true time" was 5 ms; the sim_time clock cannot resolve below 33 ms because that is the step size. Every event between two tick boundaries gets snapped to the next boundary, losing precision proportional to the tick rate.

This is the conflation the chapter warns against. The 30 Hz tick rate is *how often the loop wakes up*; it is not the *resolution of the model*. Hard-coding `1.0/30.0` as the simulation's time delta makes them the same thing — and pins the simulation's accuracy to the loop's wake-up rate.

## Exercise 3 — The right way: timestamp on events

```python
import time, heapq, numpy as np

events: list[tuple[float, str]] = []
heapq.heappush(events, (0.005, "early_event"))
heapq.heappush(events, (0.040, "second_event"))
heapq.heappush(events, (0.080, "third_event"))

start = time.perf_counter()
TICK_S = 1.0 / 30.0

while events:
    now = time.perf_counter() - start
    while events and events[0][0] <= now:
        t, msg = heapq.heappop(events)
        print(f"[real={now:.4f}, sim={t:.4f}] {msg}")
    time.sleep(TICK_S)
```

Each tick processes *all* events whose timestamp has passed. The 5 ms event fires inside the *first* tick (the loop has been running for >5 ms by the time the tick finishes). The event's `t` is preserved — `print(...)` shows the original 0.005, not the snapped tick boundary.

The simulator processes the event with its own time, not the loop's. Same model, sub-tick precision, no overhead beyond a heap pop per event.

## Exercise 4 — Sampling at different rates

```python
def run(tick_hz, events_in):
    import time, heapq
    events = list(events_in)
    heapq.heapify(events)
    start = time.perf_counter()
    tick_s = 1.0 / tick_hz
    fired_times = []
    while events:
        now = time.perf_counter() - start
        while events and events[0][0] <= now:
            t, msg = heapq.heappop(events)
            fired_times.append(t)
        time.sleep(tick_s)
    return fired_times

events_in = [(0.005, "a"), (0.040, "b"), (0.080, "c"), (0.150, "d")]
for hz in (30, 60, 1):
    print(f"{hz:>3} Hz fires at: {run(hz, events_in)}")
```

The list of fired *event* times is the same in all three runs (modulo floating-point comparison): `[0.005, 0.040, 0.080, 0.150]`. The 30 Hz, 60 Hz, and 1 Hz runs differ only in *how often the loop checked* — they all see and apply the same set of events at the same simulation timestamps. The model is sample-rate-independent.

## Exercise 5 — Float and time

```python
import numpy as np
print(np.spacing(np.float32(3600)))          # ~2.4e-04 = 244 µs
print(np.spacing(np.float32(86400)))         # ~7.8e-03 = 7.8 ms
print(np.spacing(np.float32(31_536_000)))    # 2.0 s
print(np.spacing(np.float64(31_536_000)))    # ~3.7e-09 = 3.7 ns
```

| at time of           | smallest representable step |  | usable for ms-resolution? |
|----------------------|----------------------------:|--|---------------------------|
| 1 hour, `float32`    |    244 µs                   |  | yes (just barely)          |
| 1 day, `float32`     |    **7.8 ms**               |  | no — coarser than a 100 Hz tick |
| 1 year, `float32`    |    **2 seconds**            |  | absolutely not             |
| 1 year, `float64`    |    3.7 ns                   |  | yes, with vast headroom    |

`float32` runs out of precision *fast* once the absolute time grows. A simulation that runs for more than a day at sub-millisecond resolution needs `float64`. This is the §2 catastrophic-cancellation lesson re-applied: precision is a function of the *magnitude* of the values you're representing, not just the size of the differences you care about.

## Exercise 6 — Run the storage exhibit

```sh
uv run code/measurement/event_time_storage.py
```

Source: [`code/measurement/event_time_storage.py`](https://codeberg.org/root-11/intro-book-python/src/branch/main/code/measurement/event_time_storage.py).

```
layout                                          data (MB)   build (ms)   sort (ms)   count <T (ms)
---------------------------------------------------------------------------------------------------
list of datetime objects                          53.62       387.6        5.71       19.980
numpy datetime64[us]                               7.63        92.3        6.12        1.198
numpy float64 (seconds-from-base)                  7.63        44.6       42.53        0.894

vs 'list of datetime objects':
  numpy datetime64[us]                       7.0× smaller    0.9× faster sort    16.7× faster count
  numpy float64 (seconds-from-base)          7.0× smaller    0.1× faster sort    22.3× faster count
```

The per-tick query is `count <T`: 22× faster on `float64` vs the `datetime` list. That is the column the simulator hits every tick to decide what events fire. Sort cost is one-off (ingestion); count cost compounds across millions of ticks. *The tick is the binding budget, so the count column is the one to optimise.*

## Exercise 7 — A budget-aware loop (stretch)

```python
import time, heapq

TICK_S       = 1.0 / 30.0
SOFT_BUDGET  = 0.025                              # 25 ms of the 33 ms tick

events: list[tuple[float, str]] = [...]           # populated from outside
heapq.heapify(events)

while True:
    tick_start = time.perf_counter()
    deadline   = tick_start + SOFT_BUDGET

    processed = 0
    while events and events[0][0] <= time.perf_counter() - tick_start:
        if time.perf_counter() > deadline:
            break                                 # over budget — defer the rest
        t, msg = heapq.heappop(events)
        apply_event(msg)
        processed += 1

    elapsed = time.perf_counter() - tick_start
    if events and elapsed > SOFT_BUDGET:
        print(f"deferred {len(events)} events; tick used {elapsed*1000:.1f}ms")

    sleep_for = TICK_S - elapsed
    if sleep_for > 0:
        time.sleep(sleep_for)
```

This is the *soft real-time* pattern: the loop *prefers* to process every due event each tick, but *guarantees* it will return within budget. Surplus events spill into the next tick.

This shape is what runs game engines, animation systems, and interactive simulators. It is also what the simulator's [§35 — boundary is the queue](35_boundary_is_the_queue.md) builds on — events at the edge of the tick belong to the next tick's queue, not this one's stretch goal.

The pattern fails gracefully when overloaded: latency degrades but the loop continues. The alternative — process every event whatever it costs — fails *catastrophically* when overloaded: the loop blows its tick budget, drops the next deadline, and either loses real-time properties or piles up an ever-growing deficit.
