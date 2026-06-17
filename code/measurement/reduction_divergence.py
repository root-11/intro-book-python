# /// script
# requires-python = ">=3.11"
# ///
"""
reduction_divergence.py - the specimen behind §48. A parallel floating-point reduction's result
depends on the worker count, because float addition is not associative and the partition grouping
changes with the number of workers. Then the two fixes.

    uv run code/measurement/reduction_divergence.py

In Python the GIL means CPU-bound parallelism is multiprocessing, but the bug is about GROUPING,
not threads: summing in `w` chunks rounds differently from summing in `w'` chunks, whatever ran
them. This specimen shows the grouping divergence directly (no processes needed) and the two fixes.

Sceptic's note this settles: "fix the reduction order" buys worker-count independence ONLY if the
partition count is fixed *independent of worker count*. Partitioning by worker count still changes
the grouping and still diverges.
"""
import struct

N = 1_000_000


def values() -> list[float]:
    # harmonic series 1, 1/2, 1/3, ... - the textbook case where grouping moves the low bits.
    return [1.0 / (i + 1) for i in range(N)]


def fold(xs) -> float:
    s = 0.0
    for x in xs:
        s += x
    return s


def bits(x: float) -> int:
    # raw IEEE-754 bits, so the comparison is exact rather than eyeballed.
    return struct.unpack("<Q", struct.pack("<d", x))[0]


def racy(v: list[float], workers: int) -> float:
    # partition into `workers` contiguous chunks (grouping = worker count), sum each in order, fold
    # the partials in order. Deterministic for a fixed worker count; the grouping changes with it.
    chunk = (N + workers - 1) // workers
    partials = [fold(v[i : i + chunk]) for i in range(0, N, chunk)]
    return fold(partials)


def fixed_order(v: list[float], parts: int = 64) -> float:
    # a FIXED number of partitions regardless of worker count: the grouping is always `parts`, so
    # the result does not depend on how many workers ran it.
    chunk = (N + parts - 1) // parts
    partials = [fold(v[i : i + chunk]) for i in range(0, N, chunk)]
    return fold(partials)


def integer(v: list[float], scale: float = 1e9) -> float:
    # integer addition is associative: exact, order-independent, identical for any grouping.
    return sum(int(x * scale) for x in v) / scale


def main() -> None:
    v = values()
    s = fold(v)
    print(f"§48 specimen - parallel float reduction, N = {N}, harmonic values\n")
    print(f"serial (index order):  bits={bits(s):016x}\n")
    print(f"{'workers':>7}  {'racy(parts=wrk)':>16}  {'fixed-order(64)':>16}  {'integer':>16}")
    for w in (1, 2, 4, 8):
        print(f"{w:>7}  {bits(racy(v, w)):016x}  {bits(fixed_order(v)):016x}  {bits(integer(v)):016x}")
    print("\nRACY bits change with worker count - a world hashed after this reduction differs across workers.")
    print("FIXED-ORDER (64 partitions, NOT equal to the worker count) and INTEGER are identical across all.")
    print("Python note: real CPU-bound parallelism here is multiprocessing (the GIL); the divergence is the")
    print("grouping, not the threads - and numpy.sum's own pairwise/SIMD grouping differs from a Python fold too.")


if __name__ == "__main__":
    main()
