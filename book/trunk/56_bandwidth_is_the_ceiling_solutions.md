# Solutions: 56 - The ceiling is bandwidth, not cores

CPU figures are the Ryzen 9 270 / 16 cores / CPython 3.14.5 / numpy 2.4.4 figures from [`code/heterogeneous/heterogeneous.py`](https://github.com/root-11/intro-book-python/blob/main/code/heterogeneous/heterogeneous.py); cross-machine is pending and the GPU figures are a labelled cost model, not a measurement.

## Exercise 1 - Watch the hierarchy

The motion pass `px += vx * dt` across sizes:

| elements | working set | GB/s |
|---:|---:|---:|
| 50,000 | 0.8 MB | ~83 (in cache) |
| 1,000,000 | 16 MB | ~54 |
| 10,000,000 | 160 MB | ~7 (main memory) |
| 50,000,000 | 800 MB | ~7 (main memory) |

The step down from ~80 GB/s to ~7 GB/s is the data spilling out of the last-level cache; on this box that happens between 1 MB and 16 MB working sets, so the L3 is a few MB. The main-memory floor is what matters, because real working sets do not fit in cache. Fusing the temporary (`np.multiply(vx, dt, out=tmp); px += tmp`, or one expression with `out=`) avoids allocating and writing a whole `vx * dt` array each pass and recovers a chunk of bandwidth - the idiomatic one-liner moves more memory than the arithmetic needs.

## Exercise 2 - More cores, less help

The same pass over a 50-million-element shared-memory array, split across worker processes:

| processes | GB/s | speedup |
|---:|---:|---:|
| 1 | 5.4 | 1.0x |
| 2 | 9.8 | 1.8x |
| 4 | 16.8 | 3.1x |
| 8 | 23.9 | 4.4x |
| 16 | 23.5 | 4.4x |

Speedup plateaus around eight processes at ~24 GB/s; sixteen do no better than eight. A memory-bound pass stops scaling once the processes saturate the single memory channel - past that they queue for memory rather than compute in parallel, so adding cores adds nothing. It had to be processes, not threads, because the GIL serialises CPU-bound Python threads onto one core; the array is in `multiprocessing.shared_memory` so the processes share it without copying, which is the only way the comparison is about bandwidth and not about pickling.

## Exercise 3 - The frame budget

At 24 bytes moved per element (two `float32` positions read and written, two velocities read), a 33 ms frame at the measured bandwidths keeps current:

- one core at ~7 GB/s: `7e9 * 0.0333 / 24` ~ **10 million** particles
- all cores at ~24 GB/s: `24e9 * 0.0333 / 24` ~ **33 million** particles

That is the active-set budget: how much of the world one box can bring up to date per frame. Compare it not to the whole world but to the *active* part - the cone that changed this frame ([§53](53_dirty_propagation.md), [§54](54_recompute_the_cone.md)). A few million active cells sits comfortably inside the budget, which is why the incremental discipline, not more hardware, is what keeps a large world live.

## Exercise 4 - The argument against the GPU

A million-cell active cone is a tenth of one core's frame budget and a thirtieth of the whole box's. It does not need an accelerator; it does not even need all the cores. The case where you *would* reach off the box is precise: when the *active set itself* - the part that genuinely changes each frame, after all the incremental pruning - exceeds what the box can feed in a frame (tens of millions of elements, here). Until then, the GPU is answering "recompute everything fast", a question [§53](53_dirty_propagation.md)/[§54](54_recompute_the_cone.md) arranged for you never to ask.

## Exercise 5 - The bus is the tax

Shipping ten million particles to a worker process and back takes ~408 ms; doing the pass in place takes ~34 ms - about **12x** slower to offload. The transfer moves the same bytes the computation reads, and at IPC/pickle speed that costs far more than two multiply-adds. The GPU cost model has the same shape: `bytes_to_device / bus + launch_latency + compute + bytes_back / bus` versus `cpu_time`. Offload breaks even only when arithmetic intensity - flops per byte transferred - is high enough that `compute` dominates the transfer terms. Two multiply-adds per 24 bytes is an intensity near zero, so a memory-bound pass is the worst possible offload candidate, on a GPU or to another process.

## Exercise 6 - Touch less, not more (stretch)

Take the scenegraph recompute or the spreadsheet pivot and speed it up two ways. Processes: split the array, hit the bandwidth ceiling, get ~4x at best and then nothing. Shrink the working set: recompute only the active cone ([§53](53_dirty_propagation.md)/[§54](54_recompute_the_cone.md)) and the cost drops by the dirty fraction - often 10x to 1000x, far past what cores can give, and with no coordination. The arc has been pulling the second lever the whole time. The hardware lever raises the ceiling by a small constant and then stops; the touch-less lever changes how much work there is at all, and it is unbounded in a way more cores can never be. For these workloads, doing less always beats doing the same amount faster.
