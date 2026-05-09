import timeit
import random

# Size of the dictionary to ensure cache misses
N = 10_000

# Generate dictionaries where keys differ only on the last element
dict_ff = {(1.0, float(i)): i for i in range(N)}
dict_ii = {(1, i): i for i in range(N)}
dict_ffi = {(1.0, 1.0, i): i for i in range(N)}
dict_fff = {(1.0, 1.0, float(i)): i for i in range(N)}
dict_iii = {(1, 1, i): i for i in range(N)}

# Create shuffled lists of keys to provide entropy during lookup
keys_ff = list(dict_ff.keys())
keys_ii = list(dict_ii.keys())
keys_ffi = list(dict_ffi.keys())
keys_fff = list(dict_fff.keys())
keys_iii = list(dict_iii.keys())

# Shuffle once so the order is random but consistent across benchmark runs
random.seed(42)
random.shuffle(keys_ff)
random.shuffle(keys_ii)
random.shuffle(keys_ffi)
random.shuffle(keys_fff)
random.shuffle(keys_iii)

# Test functions to minimize timeit overhead while processing the whole key set
def lookup_ff():
    for k in keys_ff: _ = dict_ff[k]

def lookup_ii():
    for k in keys_ii: _ = dict_ii[k]

def lookup_ffi():
    for k in keys_ffi: _ = dict_ffi[k]

def lookup_fff():
    for k in keys_fff: _ = dict_fff[k]

def lookup_iii():
    for k in keys_iii: _ = dict_iii[k]

# Adjust iterations: each iteration performs N lookups
ITERATIONS = 200
TOTAL_LOOKUPS = N * ITERATIONS

tests = [
    (lookup_ff, "(float, float)"),
    (lookup_ii, "(int, int)"),
    (lookup_ffi, "(float, float, int)"),
    (lookup_fff, "(float, float, float)"),
    (lookup_iii, "(int, int, int)"),
]

print(f"Dictionary size: {N:,} entries")
print(f"Total lookups per test: {TOTAL_LOOKUPS:,} (Shuffled order)")
print("-" * 75)

results = []
for func, label in tests:
    exe_time = timeit.timeit(func, globals=globals(), number=ITERATIONS)
    results.append((exe_time, label))

# Sort by fastest execution time (lowest total time)
results.sort()

for exe_time, label in results:
    throughput = TOTAL_LOOKUPS / exe_time
    print(f"{exe_time:.6f}s | {throughput:15,.0f} lookups/s | {label}")

# Dictionary size: 10,000 entries
# Total lookups per test: 2,000,000 (Shuffled order)
# ---------------------------------------------------------------------------
# 0.046728s |      42,800,637 lookups/s | (int, int)
# 0.050473s |      39,625,273 lookups/s | (int, int, int)
# 0.075580s |      26,461,898 lookups/s | (float, float)
# 0.076582s |      26,115,850 lookups/s | (float, float, int)
# 0.113440s |      17,630,435 lookups/s | (float, float, float)