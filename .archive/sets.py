import statistics, timeit

print("="*70)
print("ISOLATING s.pop() performance by measuring copy overhead")
print("="*70)

# Measure just the copy operation
copy_times = timeit.repeat(
    "s_local = s.copy()",
    globals={'s': {42}},
    number=3_000_000,
    repeat=7
)

# Measure copy + pop
copy_and_pop_times = timeit.repeat(
    "s_local = s.copy(); s_local.pop()",
    globals={'s': {42}},
    number=3_000_000,
    repeat=7
)

# Measure just pop on a pre-created local set (best we can do)
# We need to recreate the set each iteration
pop_with_recreation_times = timeit.repeat(
    "s_local = {42}; s_local.pop()",
    number=3_000_000,
    repeat=7
)

copy_median = statistics.median(copy_times)
copy_and_pop_median = statistics.median(copy_and_pop_times)
pop_with_recreation_median = statistics.median(pop_with_recreation_times)

# Calculate isolated pop time
isolated_pop_median = copy_and_pop_median - copy_median

copy_ns = (copy_median / 3_000_000) * 1e9
copy_and_pop_ns = (copy_and_pop_median / 3_000_000) * 1e9
isolated_pop_ns = (isolated_pop_median / 3_000_000) * 1e9
pop_with_recreation_ns = (pop_with_recreation_median / 3_000_000) * 1e9

print(f"\nCopy only:           {copy_median:.5f}s ({copy_ns:.2f}ns/op)")
print(f"Copy + pop:          {copy_and_pop_median:.5f}s ({copy_and_pop_ns:.2f}ns/op)")
print(f"Pop with recreation: {pop_with_recreation_median:.5f}s ({pop_with_recreation_ns:.2f}ns/op)")
print(f"\nIsolated pop (copy+pop - copy): {isolated_pop_median:.5f}s ({isolated_pop_ns:.2f}ns/op)")

print("\n" + "="*70)
print("COMPARISON WITH OTHER METHODS:")
print("="*70)

# Re-run the other methods for comparison
other_results = {}

other_results["value, = s"] = timeit.repeat(
    "value, = s",
    globals={'s': {42}},
    number=3_000_000,
    repeat=7
)

other_results["next(iter(s))"] = timeit.repeat(
    "next(iter(s))",
    globals={'s': {42}},
    number=3_000_000,
    repeat=7
)

# Add our isolated pop result
all_results = {
    "s.pop() (isolated)": isolated_pop_median,
    "value, = s": statistics.median(other_results["value, = s"]),
    "next(iter(s))": statistics.median(other_results["next(iter(s))"]),
}

sorted_results = sorted(all_results.items(), key=lambda x: x[1])
fastest = sorted_results[0][1]

for i, (name, median) in enumerate(sorted_results, 1):
    relative = median / fastest
    ns_per_op = (median / 3_000_000) * 1e9
    print(f"{i}. {name:25} {median:.5f}s ({ns_per_op:.2f}ns/op, {relative:.2f}x)")

print("\n" + "="*70)
print("ANALYSIS:")
print("="*70)
print(f"""
Is this fair? Let's consider:

PRO: 
- Isolates the actual pop() operation
- Removes the copy overhead we added for testing
- Shows true performance of just pop()

CON:
- Subtracting measurements can amplify noise/variance
- The copy operation itself might have caching effects
- Small measurement errors become magnified

Result: Isolated pop is {isolated_pop_ns:.2f}ns/op

The isolated pop() time is {"FASTER" if isolated_pop_ns < all_results["value, = s"] / 3_000_000 * 1e9 else "SLOWER"} than unpacking.

However, in your real use case where the set already exists and you 
don't need it afterward, you'd just call s.pop() directly with no
copy needed at all!
""")

print("\n" + "="*70)
print("PRACTICAL CONCLUSION:")
print("="*70)
print("""
For your use case (set exists, don't need to keep it):

Just use: value = s.pop()

No copy needed, no measurement tricks needed.
It's the simplest, fastest, and most memory-efficient solution.
""")

