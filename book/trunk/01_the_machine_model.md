# 1 — The machine model

<p align="center"><img src="../covers/phase_foundation.jpg" alt="Foundation phase" style="max-height: 380px; max-width: 100%;"></p>

> *Concept node: see the [DAG](../../concepts/dag.md) and [glossary entry 1](../../concepts/glossary.md#1--the-machine-model).*

Most explanations of "how a computer works" use a diagram with a CPU and a single big block called *memory*. The diagram is wrong. Memory is many things at different speeds, and which one your data sits in decides whether your program is fast or slow.

Inside the CPU there is **L1 cache** — small, sometimes only 32 KB per core, but a read from it costs about one nanosecond. Around it sits **L2** — a few hundred KB, around 3-4 ns. Then **L3** — measured in megabytes, around 10 ns. Outside the CPU sits **main memory (RAM)** — gigabytes, around 100 ns per read. The numbers vary by chip; the *ratios* are stable. L1 is roughly a hundred times faster than RAM.

When your code reads `arr[17]`, the CPU does not pull just byte 17. It pulls a whole 64-byte chunk — a *cache line* — and keeps that line in L1. The next read of `arr[18]` is then almost free. Reading sequentially is fast because every line that gets loaded is mostly used before it gets evicted. Reading at random is slow because every read costs a fresh trip to RAM.

A pointer is an address in memory. Following one is one memory read at an address the CPU does not get to predict. If the address is in cache, the read is fast; if not, you wait the full ~100 ns. A program with many objects and many pointers between them is a program with many of those waits.

## Why you have not had to think about this

If you used Python last week, none of the above came up. The interpreter ran your code, the operating system handed it memory, and *it worked*. You felt no cliff at 100 KB or 100 MB. You wrote a `for` loop, the loop ran, and the cost per element was whatever it was.

That experience is real, and it is hiding the machine from you. The cost of one iteration of a Python `for` loop — `PyObject_Add`, the refcount increment, the `PyLong` boxing, the bytecode dispatch — is around 5 nanoseconds per element on this machine. That number is *higher* than an L3 cache miss. So when you iterate over a Python list, the cache hierarchy is invisible to you: you spend so long in the interpreter on every step that whether the next byte was in L1 or had to come from RAM is rounding error.

This is the missing piece of the machine model in Python. The hierarchy is still there; the bottleneck just moved. To *see* the machine, you have to look in places where the interpreter dispatch isn't dominating. Two such places, both measurable on your laptop:

**1. Sum a million int64s, three ways.** [`code/measurement/cache_cliffs.py`](https://github.com/root-11/intro-book-python/blob/main/code/measurement/cache_cliffs.py) walks N from 10K to 100M and times: `sum(lst)` on a Python list, `arr.sum()` on a contiguous numpy array, and `arr[idx].sum()` where `idx` is a shuffled permutation. On this machine:

| N           | Python list | numpy seq | numpy gather | gather/seq |
|------------:|------------:|----------:|-------------:|-----------:|
|      10,000 |    4.85 ns  |  0.54 ns  |   1.47 ns    |    2.7×    |
|     100,000 |    4.60 ns  |  0.18 ns  |   2.88 ns    |   16.4×    |
|   1,000,000 |    4.60 ns  |  0.21 ns  |   3.51 ns    |   17.0×    |
|  10,000,000 |    4.62 ns  |  0.19 ns  |  10.33 ns    |   53.7×    |
| 100,000,000 |    4.60 ns  |  0.16 ns  |  11.80 ns    |   72.2×    |

Read the columns. The Python list is **flat at ~4.6 ns/element across five orders of magnitude**. From inside the interpreter the cache hierarchy does not exist. The numpy sequential column is 25-30× faster and reveals the bandwidth — the inner loop is C, the bytes are typed, the prefetcher works. The numpy gather column is the same data accessed in a shuffled order; once the working set leaves L1 (between 10K and 100K), the per-element cost climbs, and by 100M the gap to sequential is **72×**. That ratio is the L1-to-RAM cost gap on this machine, measured.

**2. Take an exception once vs a million times.** [`code/measurement/try_except.py`](https://github.com/root-11/intro-book-python/blob/main/code/measurement/try_except.py) compares `try/except ZeroDivisionError` against an explicit `if value != 0` check, across hit rates from 0.0001% to 99.9999%. At 50/50 the `try/except` form is 4× slower; at 99.9999% (almost no exceptions raised) the `try/except` form is *faster* than the `if`. The difference is the CPU's branch predictor: a taken branch with high frequency is essentially free; a mispredicted one costs ~10-20 cycles. The lesson is not "use try/except" or "use if" — it is that constant factors are rate-dependent, and even Python inherits this.

**3. Constant factors leak through.** [`code/measurement/string_methods.py`](https://github.com/root-11/intro-book-python/blob/main/code/measurement/string_methods.py) compares `%`-format, f-strings, and `.format` for the same output. On this machine `%`-format is ~20% faster than f-strings, which are ~5% faster than `.format`. None of this matters in a one-off log line. All of it matters in a tight loop. The "modern idiomatic" choice is not automatically the cheap choice.

## What this chapter is asking you to do

The dominant fact about modern CPUs is that arithmetic is virtually free; the cost is *getting the data to the arithmetic*. A program that respects this is fast. A program that ignores it can be a hundred times slower than a program that does the same work, with the same number of additions, in a layout the cache likes.

In Python this fact wears a disguise: the interpreter is so slow that the machine appears to have no cliff. The disguise comes off the moment you leave pure Python — and almost everything this book teaches involves leaving pure Python for typed contiguous columns where the cliff is right where it always was.

This is also what makes "complexity class" misleading on its own. An O(N log N) algorithm that hits the cache hard can outrun a "faster" O(N) algorithm that scatters reads across RAM. Big-O describes how cost grows with N; layout describes the constant factor that gets multiplied in. At the scales this book targets, the constant factor often wins.

## Exercises

These exercises are calibrations. Run them on your machine and write the numbers down — the rest of the book references them.

1. **Look up your cache sizes.** On Linux, `lscpu | grep -i cache` lists L1d, L1i, L2, L3 per core. (On macOS: `sysctl -a | grep cache`.) Write them down. These are the budgets [§27](27_working_set_vs_cache.md) will hold you to later.
2. **Run the cache-cliffs exhibit.** `uv run code/measurement/cache_cliffs.py`. Read the output. Note the size at which the numpy gather column starts climbing — that is where you spilled out of L1. Note where it climbs again — L2, L3.
3. **Confirm the interpreter mask.** Modify the exhibit to print `arr.tolist()` sum at every size step alongside the existing measurements. Confirm that the Python list cost is still flat — the cliffs do not appear, even though the data is the same.
4. **Run the try/except exhibit.** `uv run code/measurement/try_except.py`. Note the cross-over: at what hit-rate does `try/except` become faster than `if`? On most machines it lands above 99%.
5. **Run the string-format exhibit.** `uv run code/measurement/string_methods.py`. Note the ranking on your machine. The order can shift across CPython versions — measure, do not memorise.
6. **A linked list of pointers.** Build a chain of 1,000,000 nodes as `class Node: __slots__ = ("value", "next")`, then sum `value` by walking `.next` from the head. Compare against the same sum on a numpy `int64` array of the same length. The ratio you see is roughly the L1-to-RAM ratio for *one* level of indirection in Python — note that this ratio compounds when objects nest deeper.
7. *(stretch)* **Read your `lscpu` output to your benchmarks.** With your cache sizes from exercise 1 and your timings from exercise 2, identify which level of cache each step in the gather column is leaving. The transitions are not always clean — annotate where they are noisy.

> [!NOTE]
> Numbers in this chapter were measured on this author's machine. The shape — flat Python list, staircase numpy, widening gather/seq ratio — is robust across hardware. The exact ratios shift with CPU generation: older or smaller chips (Raspberry Pi 4, 2012-era Intel) show a graded staircase across L1/L2/L3, while modern desktop chips often show one big cliff at the L3-to-RAM boundary. Measure on your own machine; reproduce shapes, not specific numbers.

Reference notes for these exercises in [01_the_machine_model_solutions.md](01_the_machine_model_solutions.md).

## What's next

The cache sizes you wrote down in exercise 1 and the cliffs you found in exercise 2 are the constants behind the whole book. [§2 — Numbers and how they fit](02_numbers_and_how_they_fit.md) takes the next step: how big is each unit of data, and how many fit in a cache line?
