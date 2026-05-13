# Solutions: 11 — The tick

## Exercise 1 — A 30 Hz time-driven loop

```python
import time

TICK_S = 1.0 / 30.0
start = time.perf_counter()
end   = start + 10.0
ticks = 0
while time.perf_counter() < end:
    t0 = time.perf_counter()
    ticks += 1
    print(f"t={t0 - start:6.3f}s  tick={ticks}")
    elapsed = time.perf_counter() - t0
    if elapsed < TICK_S:
        time.sleep(TICK_S - elapsed)

print(f"{ticks} ticks in {time.perf_counter()-start:.2f}s")
```

Expected: 300 ticks ± 1 in 10 seconds. The loop sleeps for `TICK_S - work_done`, so each iteration ends *exactly* `TICK_S` after it began (modulo OS scheduling). `time.perf_counter()` is monotonic; `time.time()` can step backwards on NTP corrections and is the wrong tool here.

## Exercise 2 — The naive sleep mistake

```python
while True:
    do_some_work()
    time.sleep(1/30)            # always 33 ms, regardless of work time
```

Each iteration takes `work_ms + 33 ms`, not `33 ms` total. If the work consistently takes 5 ms, the loop ticks at `1 / (0.005 + 0.033)` ≈ **26.3 Hz**, not 30. Over a minute that is 1,580 ticks instead of 1,800 — a 12% deficit, and the program reports "running at 30 Hz" because that's what it asked for.

The drift is silent: nothing in the program complains. Only an external observer (the wall clock, an event log, an animation that runs slow) notices. The fix is to *measure work time and subtract*, as in exercise 1.

## Exercise 3 — Dropped frames

```python
while running:
    t0 = time.perf_counter()
    do_some_work()                    # may take longer than TICK_S
    elapsed = time.perf_counter() - t0
    if elapsed > TICK_S:
        print(f"missed deadline by {(elapsed - TICK_S)*1000:.1f} ms")
    else:
        time.sleep(TICK_S - elapsed)
```

If `do_some_work()` sleeps 50 ms (longer than the 33 ms budget), the loop runs at 20 Hz and prints `missed deadline by 16.7 ms` every iteration. Detecting missed deadlines is half the battle; *responding* to them is the rest. The simplest response is "log it and continue"; smarter responses (skip a frame's interpolation, drop secondary work, lower the visible LOD) live at the application layer.

A simulator that has missed its tick budget is a simulator running on the wrong hardware or with the wrong N. Naming the deadline-miss is how you know.

## Exercise 4 — A turn-based loop

```python
while running:
    line = input("> ")
    if line.strip() in {"quit", "exit"}: break
    print(f"you said: {line}")
```

Each `input()` blocks until a line arrives. The loop has no fixed rate — its pace is whatever the typist provides. The same shape carries a chess engine (one tick per move), a card game (one tick per play), a discrete-event simulator (one tick per event timestamp). The trigger is *"a thing happened"*, not *"33 ms passed"*.

## Exercise 5 — Run the tick-budget exhibit

```sh
uv run code/measurement/tick_budget.py
```

Source: [`code/measurement/tick_budget.py`](https://github.com/root-11/intro-book-python/blob/main/code/measurement/tick_budget.py).

```
          N  layout                  tick (ms)   30 Hz            60 Hz
--------------------------------------------------------------------------------
     10,000  numpy SoA                   0.005   fit  (  0.0%)    fit  (  0.0%)
     10,000  Python dataclass list       0.272   fit  (  0.8%)    fit  (  1.6%)
    100,000  numpy SoA                   0.019   fit  (  0.1%)    fit  (  0.1%)
    100,000  Python dataclass list       2.750   fit  (  8.3%)    fit  ( 16.5%)
  1,000,000  numpy SoA                   0.278   fit  (  0.8%)    fit  (  1.7%)
  1,000,000  Python dataclass list      27.525   fit  ( 82.6%)    OVER (  165%)
 10,000,000  numpy SoA                  16.609   fit  ( 49.8%)    fit  ( 99.7%)
 10,000,000  Python dataclass list   (skipped)   extrapolates over   extrapolates over
```

The 60 Hz line on `1M dataclass`: 165% over budget. The 30 Hz line on `1M dataclass`: 82.6% used by **one motion system**, leaving 5.7 ms for everything else the simulator needs to do per tick. The book is asking you to keep the numpy line because that is the population at which Python becomes feasible. Below 100K entities the layout choice doesn't matter much; above 100K it determines whether the simulator runs at all.

## Exercise 6 — The asyncio comparison

```python
import asyncio, time

TICK_S = 1.0 / 30.0

async def loop():
    start = time.perf_counter()
    end   = start + 10.0
    while time.perf_counter() < end:
        t0 = time.perf_counter()
        # do_work()
        elapsed = time.perf_counter() - t0
        if elapsed < TICK_S:
            await asyncio.sleep(TICK_S - elapsed)

asyncio.run(loop())
```

Tick rate: same as the synchronous version (~30 Hz). Memory: ~1-2 MB more, for the event loop, the task object, and the awaitable infrastructure. Wall time per tick: 5-20 µs higher because each iteration goes through the event loop's task-stepping machinery to schedule the next wakeup.

What you got for the cost: nothing. The work is CPU-bound; there are no other awaitables to interleave; the event loop has no useful work to do during the sleep. `await asyncio.sleep` becomes a slightly more expensive `time.sleep`. The asyncio scheduler is the right shape for *I/O-bound* programs (web servers, network clients) where a single thread juggles many waiting connections; it is the wrong shape for a CPU-bound tick loop.

The lesson is the chapter's: *reach for the simplest tool that gives you the property you actually need.* Asyncio is correct for many programs. This is not one of them.

## Exercise 7 — A discrete-event tick loop (stretch)

```python
import heapq

events: list[tuple[float, str]] = []
heapq.heappush(events, (1.0, "creature_birth"))
heapq.heappush(events, (0.5, "food_spawn"))
heapq.heappush(events, (2.5, "starvation_check"))
heapq.heappush(events, (1.5, "creature_birth"))

clock = 0.0
while events:
    t, msg = heapq.heappop(events)
    clock = t                                   # advance to the event's timestamp
    print(f"t={clock:.2f}  event: {msg}")
```

```
t=0.50  event: food_spawn
t=1.00  event: creature_birth
t=1.50  event: creature_birth
t=2.50  event: starvation_check
```

Two properties to notice:

- **The clock advances *with* the events**, not in fixed steps. There is no "tick" between t=0.5 and t=1.0; the simulation simply jumps. Long quiet periods cost nothing.
- **No external time reference is needed.** Everything is internal — events have timestamps, the clock follows them. This is the discrete-event-simulation (DES) shape that production tools (SimPy, NS-3, OMNeT++, MATLAB Simulink) build on.

[§12 — Event time vs tick time](12_event_time_vs_tick_time.md) names this distinction: the clock the *simulation* uses doesn't have to be the clock the *loop* uses. A 30 Hz time-driven loop with a discrete-event subsystem inside it is a common shape — the outer loop advances the world by 33 ms; the inner DES processes all events with timestamps in the next 33 ms.
