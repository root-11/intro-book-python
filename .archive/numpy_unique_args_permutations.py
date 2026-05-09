import numpy as np
import timeit
import itertools
import sys

# Configuration
N = 10_000
ITERATIONS = 20

# Sample values
def val_int(i): return i
def val_float(i): return float(i)
def val_str(i): return f"item_{i}"

type_map = {
    'int': val_int,
    'float': val_float,
    'str': val_str
}

test_types = ['int', 'float', 'str']
# Single types first, then all permutations of length 2
all_combinations = [(t,) for t in test_types] + list(itertools.product(test_types, repeat=2))

# Check if 'sorted' is a valid argument for np.unique in this environment
try:
    np.unique(np.array([1, 2]), sorted=True)
    HAS_SORTED_PARAM = True
except TypeError:
    HAS_SORTED_PARAM = False

prepared_data = []
for combo in all_combinations:
    data = []
    for i in range(N):
        if len(combo) == 1:
            data.append(type_map[combo[0]](i))
        else:
            data.append(tuple(type_map[t](i) for t in combo))
    
    arr = np.array(data)
    # If 1D, reshape to (N, 1) to support axis=0 tests consistently
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
        
    type_label = str(combo[0]) if len(combo) == 1 else "(" + ", ".join(combo) + ")"
    prepared_data.append((arr, type_label))

def run_test(arr, axis, use_sorted):
    """Wrapper for np.unique call."""
    if axis == 'void':
        # View the row as a single void blob
        # This only works reliably for homogeneous types (same width) and contiguous arrays
        try:
            # Ensure continuity and view as a single byte-block per row
            row_size = arr.dtype.itemsize * arr.shape[1]
            void_arr = np.ascontiguousarray(arr).view(np.dtype((np.void, row_size)))
            
            kwargs = {}
            if HAS_SORTED_PARAM:
                kwargs["sorted"] = use_sorted
            return len(np.unique(void_arr, **kwargs))
        except Exception:
            # Fallback or skip if void view is impossible (e.g. mixed string lengths in some np versions)
            return None

    kwargs = {"axis": axis}
    if HAS_SORTED_PARAM:
        kwargs["sorted"] = use_sorted
    return len(np.unique(arr, **kwargs))

print(f"NumPy version: {np.__version__}")
print(f"Has 'sorted' parameter: {HAS_SORTED_PARAM}")
print(f"Benchmark: len(np.unique(arr, axis=..., sorted=...))")
print(f"Array size: {N:,} rows x 2 columns")
print("-" * 90)

# Define variations to test
variations = [
    (0, True, "axis=0, sorted=True"),
    (None, True, "axis=None, sorted=True"),
    ('void', True, "axis=void, sorted=True"),
]
if HAS_SORTED_PARAM:
    variations.append((0, False, "axis=0, sorted=False"))
    variations.append((None, False, "axis=None, sorted=False"))
    variations.append(('void', False, "axis=void, sorted=False"))

results = []
for arr, type_label in prepared_data:
    for axis, use_sorted, var_label in variations:
        try:
            exe_time = timeit.timeit(lambda: run_test(arr, axis, use_sorted), number=ITERATIONS)
            # If the result was actually None (failed void view), skip it
            # timeit will still return a time, but we should verify the operation worked
            if axis == 'void' and run_test(arr, axis, use_sorted) is None:
                continue
                
            full_label = f"{type_label:20} | {var_label}"
            results.append((exe_time, full_label))
        except Exception as e:
            continue

# Sort by fastest execution time
results.sort()

header = f"{'Time':<12} | {'Throughput':<15} | {'Types':<20} | {'Parameters'}"
print(header)
print("-" * 90)

for exe_time, label in results:
    throughput = (N * ITERATIONS) / exe_time
    print(f"{exe_time:.6f}s | {throughput:12,.0f} r/s | {label}")

if not HAS_SORTED_PARAM:
    print("\nNote: The 'sorted' parameter was not detected in this version of NumPy.")

# NumPy version: 2.3.5
# Has 'sorted' parameter: True
# Benchmark: len(np.unique(arr, axis=..., sorted=...))
# Array size: 10,000 rows x 2 columns
# ------------------------------------------------------------------------------------------
# Time         | Throughput      | Types                | Parameters
# ------------------------------------------------------------------------------------------
# 0.001043s |  191,768,160 r/s | float                | axis=None, sorted=False
# 0.001062s |  188,259,391 r/s | float                | axis=None, sorted=True
# 0.002808s |   71,223,752 r/s | (int, float)         | axis=None, sorted=False
# 0.002809s |   71,198,068 r/s | (float, float)       | axis=None, sorted=False
# 0.002815s |   71,038,948 r/s | (int, float)         | axis=None, sorted=True
# 0.002823s |   70,845,287 r/s | (float, int)         | axis=None, sorted=False
# 0.002916s |   68,584,119 r/s | (float, int)         | axis=None, sorted=True
# 0.003091s |   64,712,333 r/s | (float, float)       | axis=None, sorted=True
# 0.007323s |   27,312,252 r/s | (int, int)           | axis=None, sorted=False
# 0.007371s |   27,132,529 r/s | int                  | axis=None, sorted=False
# 0.008738s |   22,888,504 r/s | (int, int)           | axis=None, sorted=True
# 0.008897s |   22,479,728 r/s | int                  | axis=None, sorted=True
# 0.012781s |   15,648,137 r/s | str                  | axis=None, sorted=True
# 0.012819s |   15,601,946 r/s | str                  | axis=None, sorted=False
# 0.018422s |   10,856,802 r/s | str                  | axis=void, sorted=False
# 0.018857s |   10,605,897 r/s | str                  | axis=void, sorted=True
# 0.020291s |    9,856,709 r/s | (str, str)           | axis=void, sorted=False
# 0.020385s |    9,811,062 r/s | (str, str)           | axis=void, sorted=True
# 0.020968s |    9,538,260 r/s | (int, int)           | axis=void, sorted=False
# 0.021236s |    9,417,936 r/s | (int, float)         | axis=void, sorted=False
# 0.021242s |    9,415,147 r/s | (float, float)       | axis=void, sorted=False
# 0.021312s |    9,384,554 r/s | (int, float)         | axis=void, sorted=True
# 0.021339s |    9,372,323 r/s | (float, float)       | axis=void, sorted=True
# 0.021366s |    9,360,596 r/s | (float, int)         | axis=void, sorted=False
# 0.021443s |    9,326,973 r/s | (int, int)           | axis=void, sorted=True
# 0.021539s |    9,285,449 r/s | (float, int)         | axis=void, sorted=True
# 0.023824s |    8,395,013 r/s | float                | axis=void, sorted=False
# 0.023972s |    8,343,113 r/s | int                  | axis=void, sorted=False
# 0.023988s |    8,337,458 r/s | float                | axis=void, sorted=True
# 0.024220s |    8,257,628 r/s | (str, int)           | axis=void, sorted=False
# 0.024379s |    8,203,945 r/s | int                  | axis=void, sorted=True
# 0.024453s |    8,179,092 r/s | (str, int)           | axis=void, sorted=True
# 0.026301s |    7,604,398 r/s | (str, str)           | axis=None, sorted=True
# 0.026791s |    7,465,121 r/s | (str, str)           | axis=None, sorted=False
# 0.033461s |    5,977,094 r/s | (int, str)           | axis=void, sorted=True
# 0.033463s |    5,976,682 r/s | (int, str)           | axis=void, sorted=False
# 0.036102s |    5,539,812 r/s | (str, int)           | axis=None, sorted=False
# 0.037009s |    5,404,123 r/s | (str, int)           | axis=None, sorted=True
# 0.040021s |    4,997,403 r/s | float                | axis=0, sorted=False
# 0.040256s |    4,968,254 r/s | float                | axis=0, sorted=True
# 0.040425s |    4,947,490 r/s | (int, float)         | axis=0, sorted=False
# 0.040609s |    4,924,982 r/s | (int, float)         | axis=0, sorted=True
# 0.040658s |    4,919,132 r/s | int                  | axis=0, sorted=False
# 0.040664s |    4,918,416 r/s | (float, float)       | axis=0, sorted=True
# 0.040731s |    4,910,260 r/s | int                  | axis=0, sorted=True
# 0.040807s |    4,901,060 r/s | (float, int)         | axis=0, sorted=False
# 0.041007s |    4,877,268 r/s | (float, float)       | axis=0, sorted=False
# 0.041082s |    4,868,304 r/s | (int, int)           | axis=0, sorted=False
# 0.041086s |    4,867,876 r/s | (int, int)           | axis=0, sorted=True
# 0.041367s |    4,834,822 r/s | (float, int)         | axis=0, sorted=True
# 0.047802s |    4,183,955 r/s | (int, str)           | axis=None, sorted=False
# 0.048092s |    4,158,659 r/s | (float, str)         | axis=void, sorted=True
# 0.048400s |    4,132,262 r/s | (int, str)           | axis=None, sorted=True
# 0.051366s |    3,893,612 r/s | (float, str)         | axis=void, sorted=False
# 0.051863s |    3,856,303 r/s | (str, float)         | axis=void, sorted=True
# 0.052799s |    3,787,969 r/s | (str, float)         | axis=void, sorted=False
# 0.061590s |    3,247,264 r/s | (float, str)         | axis=None, sorted=True
# 0.061790s |    3,236,750 r/s | (str, float)         | axis=None, sorted=True
# 0.062503s |    3,199,870 r/s | (float, str)         | axis=None, sorted=False
# 0.063127s |    3,168,234 r/s | (str, float)         | axis=None, sorted=False
# 0.066726s |    2,997,316 r/s | str                  | axis=0, sorted=False
# 0.066800s |    2,994,010 r/s | str                  | axis=0, sorted=True
# 0.070303s |    2,844,832 r/s | (str, str)           | axis=0, sorted=False
# 0.070922s |    2,820,012 r/s | (str, str)           | axis=0, sorted=True
# 0.074401s |    2,688,144 r/s | (str, int)           | axis=0, sorted=False
# 0.074642s |    2,679,458 r/s | (str, int)           | axis=0, sorted=True
# 0.080887s |    2,472,579 r/s | (int, str)           | axis=0, sorted=False
# 0.080897s |    2,472,288 r/s | (int, str)           | axis=0, sorted=True
# 0.089664s |    2,230,549 r/s | (float, str)         | axis=0, sorted=False
# 0.089792s |    2,227,374 r/s | (float, str)         | axis=0, sorted=True
# 0.093509s |    2,138,837 r/s | (str, float)         | axis=0, sorted=True
# 0.095484s |    2,094,582 r/s | (str, float)         | axis=0, sorted=False