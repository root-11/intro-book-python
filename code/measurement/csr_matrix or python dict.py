# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "scipy"]
# ///
import timeit
import numpy as np
from scipy.sparse import csr_matrix

# Configuration for the experiment
MATRIX_SIZE = 1000  # Size of the square matrix (MATRIX_SIZE x MATRIX_SIZE)
DENSITY = 0.01      # Percentage of non-zero elements
NUM_LOOKUPS = 10000 # Number of lookups to perform

# 1. Generate sparse data
np.random.seed(42)
num_non_zero = int(MATRIX_SIZE * MATRIX_SIZE * DENSITY)
rows = np.random.randint(0, MATRIX_SIZE, num_non_zero)
cols = np.random.randint(0_0, MATRIX_SIZE, num_non_zero)
data = np.random.rand(num_non_zero)

# Ensure unique (row, col) pairs for dictionary keys and proper CSR construction
# If there are duplicates, csr_matrix sums them, and dict would only keep the last.
# For a fair comparison, let's make them unique for dict and let csr handle duplicates by summing.
unique_coords = set()
sparse_data_for_dict = []
for r, c, val in zip(rows, cols, data):
    if (r, c) not in unique_coords:
        sparse_data_for_dict.append(((r, c), val))
        unique_coords.add((r, c))

# --- CORRECTED PART ---
# 2. Construct csr_matrix
# Extract row indices, column indices, and values separately as 1-D arrays
coords_for_csr_rows = np.array([item[0][0] for item in sparse_data_for_dict]) # Get the 'row' from (row, col)
coords_for_csr_cols = np.array([item[0][1] for item in sparse_data_for_dict]) # Get the 'col' from (row, col)
values_for_csr = np.array([item[1] for item in sparse_data_for_dict])        # Get the 'value'

csr_mat = csr_matrix((values_for_csr, (coords_for_csr_rows, coords_for_csr_cols)),
                     shape=(MATRIX_SIZE, MATRIX_SIZE))
# --- END CORRECTED PART ---


# 3. Construct a Python dict
python_dict = {coord_pair: value for coord_pair, value in sparse_data_for_dict}

# 4. Define lookups
# We'll generate random coordinates for lookup.
# Some will exist in our sparse data, some might be zero.
lookup_coords = []
for _ in range(NUM_LOOKUPS):
    r = np.random.randint(0, MATRIX_SIZE)
    c = np.random.randint(0, MATRIX_SIZE)
    lookup_coords.append((r, c))

# --- Benchmarking ---

# Function for csr_matrix lookup
def csr_lookup(matrix, coords_list):
    for r, c in coords_list:
        _ = matrix[r, c]

# Function for python dict lookup (using .get for non-existent keys, similar to csr returning 0)
def dict_lookup(dictionary, coords_list):
    for r, c in coords_list:
        _ = dictionary.get((r, c), 0.0) # .get is safer for non-existent keys

print(f"Benchmarking with a {MATRIX_SIZE}x{MATRIX_SIZE} matrix, {DENSITY*100}% density ({len(sparse_data_for_dict)} non-zero elements).")
print(f"Performing {NUM_LOOKUPS} random lookups.\n")

# Measure csr_matrix lookup time
csr_time = timeit.timeit(lambda: csr_lookup(csr_mat, lookup_coords), number=1)
print(f"CSR Matrix lookup time: {csr_time:.6f} seconds")

# Measure python dict lookup time
dict_time = timeit.timeit(lambda: dict_lookup(python_dict, lookup_coords), number=1)
print(f"Python Dictionary lookup time: {dict_time:.6f} seconds")

# Compare
if dict_time < csr_time:
    print(f"\nConclusion: Python Dictionary is faster for lookups by approximately {csr_time / dict_time:.2f} times.")
else:
    print(f"\nConclusion: CSR Matrix is faster for lookups by approximately {dict_time / csr_time:.2f} times.")

#----------- OUTPUT ---------

# Benchmarking with a 1000x1000 matrix, 1.0% density (9954 non-zero elements).
# Performing 10000 random lookups.

# CSR Matrix lookup time: 0.069129 seconds
# Python Dictionary lookup time: 0.000638 seconds

# Conclusion: Python Dictionary is faster for lookups by approximately 108.40 times.