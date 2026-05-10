# 11 — The tick

<p align="center"><img src="../covers/phase_time_passes.jpg" alt="Time & passes phase" style="max-height: 380px; max-width: 100%;"></p>

> *Concept node: see the [DAG](../../concepts/dag.md) and [glossary entry 11](../../concepts/glossary.md#11--the-tick).*

A program's life has a shape:

- **Start-up** — initialisation. Tables are allocated, inputs are opened, the RNG is seeded, the world reaches a known state.
- **Steps** — ticks of the clock in a simulation, turns in a card game, requests in a server. The repeating unit of forward motion.
- **Save and load** — the in-memory state is preserved to disk so a future run can resume from where this one left off. Optional, but if you want it, it lives here.
- **Exit** — resources are returned to the kernel. Memory, file handles, sockets, lockfiles. Failure to do this cleanly is called a *memory leak* (or a stale lock, or a broken socket).

This section is about the step. The step is where the time budget binds, where the system DAG runs, where determinism either holds or breaks. The other phases are real — the book returns to save and load when persistence is named at [§36](36_persistence_is_serialization.md), and exit is mostly the operating system's job — but the inner step is what makes or breaks every other property the book builds on.

Each step is a *tick*. State at the start of a tick is read; state at the end is written; nothing is half-updated mid-tick. Even an interactive program — a card game waiting for the next move, a text editor waiting for a keystroke — is a tick loop, just with an external trigger driving it. A program that does a single pass over a file and exits is a degenerate tick loop with N=1.

## Two shapes of tick

A **time-driven** tick fires at a fixed rate. The simulator from [`code/sim/SPEC.md`](../../code/sim/SPEC.md) runs at 30 Hz: one tick every 33 ms. The loop wakes up, advances every system by one step, sleeps until the next tick. Most simulations, games, control loops, audio engines, and animation systems are time-driven. The rate is a contract with the rest of the world: at this rate, output appears.

A **turn-based** tick fires when an event arrives. A card game ticks when a player makes a move. A chess engine ticks when its opponent moves. A discrete-event simulator ticks at the timestamp of the next pending event, however far in the future that is. The clock advances *with* the events, not under them. Turn-based ticks have no fixed rate; their pace is set by the input stream.

Both are ticks. The difference is what triggers the next pass:

```python
# time-driven
import time

TICK_S = 1.0 / 30.0  # 33.3 ms

while running:
    start = time.perf_counter()
    run_all_systems(world)
    elapsed = time.perf_counter() - start
    if elapsed < TICK_S:
        time.sleep(TICK_S - elapsed)
```

```python
# turn-based
while running:
    event = wait_for_next_event()
    apply_event(world, event)
```

The §0 simulator runs time-driven. The card game from §5 ran turn-based — every card you dealt was one tick. Both are valid; both fit the same framework.

## Not asyncio. Not threads.

Two reflexes the modern Python reader will reach for, and neither is the right tool here.

The **asyncio** reflex says "control loops are async." `asyncio` is a scheduler for I/O-bound work — code that spends most of its time waiting for sockets, files, or sleeps. A simulation tick is **CPU-bound**: every tick, you have computation to do, and the goal is to do it as fast as possible and then sleep precisely until the next deadline. The asyncio event loop adds dispatch overhead (awaitable wrapping, task stepping, the event loop's own bookkeeping) without giving you anything in return — you are not waiting on external I/O. A synchronous `while True:` loop with `time.sleep` is the correct shape, and it is shorter.

The **threading** reflex says "use a `Timer` thread to fire ticks." This is worse. CPython's GIL means the timer thread and the main thread cannot run Python code simultaneously; the timer thread firing the tick at 33 ms intervals contends for the same lock the simulation needs. You add scheduler nondeterminism (the OS picks who gets the GIL after each tick interval), you add the GIL-acquisition cost on every wakeup, and you gain nothing — you could have called `time.sleep` from the main thread directly.

A simulation tick wants three things: precision (sleep until exactly the next deadline), determinism (the same input produces the same output), and simplicity (one place to read to understand the loop). A synchronous loop with `time.perf_counter` and `time.sleep` provides all three. The two reflexes above provide none of them. *Reach for the simplest tool that gives you the property you actually need.*

## What fits in a tick

The budget binds the design. From [`code/measurement/tick_budget.py`](../../code/measurement/tick_budget.py), one motion system (`pos += vel * dt`) measured on this machine:

|             N  | layout              | tick time | 30 Hz budget | 60 Hz budget |
|---------------:|---------------------|----------:|:-------------|:-------------|
|        10,000  | numpy SoA           |  0.011 ms | 0.03%        | 0.07%        |
|        10,000  | Python dataclass    |  0.280 ms | 0.84%        | 1.7%         |
|       100,000  | numpy SoA           |  0.023 ms | 0.07%        | 0.14%        |
|       100,000  | Python dataclass    |  2.858 ms | 8.6%         | 17.1%        |
|     1,000,000  | numpy SoA           |  0.613 ms | 1.8%         | 3.7%         |
|     1,000,000  | Python dataclass    | 27.947 ms | **84%**      | **OVER**     |
|    10,000,000  | numpy SoA           | 28.965 ms | 87%          | **OVER**     |

Read the rows. At 100,000 entities, both layouts fit comfortably at 30 Hz, but the dataclass loop already uses *125× more of the budget* than the numpy version. At 1,000,000 entities, the dataclass version eats 84% of the 30 Hz budget on **one** system — the rest of the simulator has 5 ms left for everything else. It does not fit at 60 Hz at all. The numpy version still has 98% of the budget free. At 10,000,000 entities, even the numpy version is at 87% of the 30 Hz budget; the simulation has hit a scale limit on this hardware, and the next move is either reducing the work per element, partitioning the work across processes ([§31](31_disjoint_writes_parallelize.md)), or accepting a slower tick rate.

The dataclass version at 10,000,000 was skipped because it would extrapolate to ~280 ms per tick — eight ticks of 30 Hz budget — for one system, before any other work. The right reading of that gap is not "numpy is fast" but "an interpreter-bound inner loop puts a hard ceiling on the population your tick can sustain, and the ceiling is much lower than most readers expect."

The budget is also where mixing turn-based and time-driven thinking in the same loop produces *drift*: the turn-based subsystem's pace bleeds into the time-driven subsystem's budget. The fix is to keep the two cleanly separated — typically one outer loop and the other as an event source feeding it.

A tick is the unit of forward motion in any program that has forward motion. The next sections name what *fits* in one tick, in what order, and what does not.

## Exercises

You will need a fresh project for these. `mkdir tick_lab && cd tick_lab && uv init` is enough.

1. **A 30 Hz time-driven loop.** Write a `main()` that loops at 30 Hz. Each iteration, print the elapsed time since program start. Sleep between ticks to maintain the rate. Run it for 10 seconds. Did you actually get 300 iterations? Use `time.perf_counter()` — `time.time()` can go backwards on clock corrections.
2. **The naive sleep mistake.** Replace your sleep logic with `time.sleep(1/30)` (no measurement of work time). Run for 30 seconds. Does the program drift over time? Why? (Hint: each iteration's work + sleep is now `33 ms + work_ms`, not `33 ms` total.)
3. **Dropped frames.** Inside the loop, sleep for 50 ms — longer than the budget. The loop is now running at 20 Hz; it has *missed frames*. Print a warning when this happens. The right way to detect: `if elapsed > TICK_S: print(f"missed deadline by {elapsed - TICK_S:.3f} s")`.
4. **A turn-based loop.** Write a tiny REPL: print `> `, read a line with `input()`, print `you said: <line>`. Each line is one tick. Run it. Note that the loop has no fixed rate — its pace is your typing.
5. **Run the tick-budget exhibit.** `uv run code/measurement/tick_budget.py`. Note the row where the dataclass version stops fitting at 60 Hz. Note the row where it stops fitting at 30 Hz. Note that the numpy version is still fine at both N values. The book is asking you to keep the numpy line running for the next thirty chapters.
6. **The asyncio comparison.** Rewrite exercise 1 using `asyncio.run` and `await asyncio.sleep`. Measure: does it tick at the same rate? Does the program use more memory? More wall time per tick? Compare your two implementations side by side. Most readers will find the asyncio version harder to read and not measurably faster — exactly the calibration the prose above predicts.
7. *(stretch)* **A discrete-event tick loop.** Maintain a list of `(timestamp, message)` events sorted by timestamp. Pop the smallest-timestamp event, advance a "simulation clock" to that timestamp, print the message, repeat until the queue is empty. This is the structure of a discrete-event simulator and a preview of [§12](12_event_time_vs_tick_time.md). Use `heapq` for the priority queue.

Reference notes in [11_the_tick_solutions.md](11_the_tick_solutions.md).

## What's next

Exercise 7 hints at the next section. The clock can live on the events themselves, independent of how often the loop fires. [§12 — Event time vs tick time](12_event_time_vs_tick_time.md) names that separation.
