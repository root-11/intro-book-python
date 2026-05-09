
import timeit
import collections
from typing import NamedTuple
from types import SimpleNamespace

# Define namedtuple and typing.NamedTuple for comparison
CoordinateNamedTuple = collections.namedtuple("CoordinateNamedTuple", ["x", "y"])

class CoordinateTypingNamedTuple(NamedTuple):
    x: float
    y: float

# Number of iterations for the test
NUM_ITERATIONS = 1_000_000

# --- Creation Benchmarks ---
dict_creation_time = timeit.timeit(
    "obj = {'x': 10.0, 'y': 20.0}", globals=globals(), number=NUM_ITERATIONS
)
named_tuple_creation_time = timeit.timeit(
    "obj = CoordinateNamedTuple(10.0, 20.0)", globals=globals(), number=NUM_ITERATIONS
)
typing_named_tuple_creation_time = timeit.timeit(
    "obj = CoordinateTypingNamedTuple(10.0, 20.0)", globals=globals(), number=NUM_ITERATIONS
)
simple_namespace_creation_time = timeit.timeit(
    "obj = SimpleNamespace(x=10.0, y=20.0)", globals=globals(), number=NUM_ITERATIONS
)

print(f"dict creation time:                   {dict_creation_time:.6f} seconds")
print(f"collections.namedtuple creation time: {named_tuple_creation_time:.6f} seconds")
print(f"typing.NamedTuple creation time:      {typing_named_tuple_creation_time:.6f} seconds")
print(f"SimpleNamespace creation time:        {simple_namespace_creation_time:.6f} seconds")

# --- Attribute Access Benchmarks ---
dict_access_time = timeit.timeit(
    "obj['x']; obj['y']",
    globals={"obj": {'x': 10.0, 'y': 20.0}},
    number=NUM_ITERATIONS
)
named_tuple_access_time = timeit.timeit(
    "obj.x; obj.y",
    globals={"obj": CoordinateNamedTuple(10.0, 20.0)},
    number=NUM_ITERATIONS
)
typing_named_tuple_access_time = timeit.timeit(
    "obj.x; obj.y",
    globals={"obj": CoordinateTypingNamedTuple(10.0, 20.0)},
    number=NUM_ITERATIONS
)
simple_namespace_access_time = timeit.timeit(
    "obj.x; obj.y",
    globals={"obj": SimpleNamespace(x=10.0, y=20.0)},
    number=NUM_ITERATIONS
)

print(f"dict access time:                     {dict_access_time:.6f} seconds")
print(f"collections.namedtuple access time:   {named_tuple_access_time:.6f} seconds")
print(f"typing.NamedTuple access time:        {typing_named_tuple_access_time:.6f} seconds")
print(f"SimpleNamespace access time:          {simple_namespace_access_time:.6f} seconds")

# --- Attribute Mutation Benchmarks ---
dict_mutation_time = timeit.timeit(
    "obj['x'] = 30.0; obj['y'] = 40.0",
    globals={"obj": {'x': 10.0, 'y': 20.0}},
    number=NUM_ITERATIONS
)
simple_namespace_mutation_time = timeit.timeit(
    "obj.x = 30.0; obj.y = 40.0",
    globals={"obj": SimpleNamespace(x=10.0, y=20.0)},
    number=NUM_ITERATIONS
)

print(f"dict mutation time:                   {dict_mutation_time:.6f} seconds")
print(f"SimpleNamespace mutation time:        {simple_namespace_mutation_time:.6f} seconds")

### Results

# dict creation time:                   0.035596 seconds
# collections.namedtuple creation time: 0.100861 seconds
# typing.NamedTuple creation time:      0.103934 seconds
# SimpleNamespace creation time:        0.079200 seconds
# dict access time:                     0.025760 seconds
# collections.namedtuple access time:   0.026598 seconds
# typing.NamedTuple access time:        0.026512 seconds
# SimpleNamespace access time:          0.030687 seconds
# dict mutation time:                   0.028512 seconds
# SimpleNamespace mutation time:        0.033870 seconds

### Creation Performance

# Fastest: dict (≈0.036 s)
# Slowest: typing.NamedTuple (≈0.104 s)
# SimpleNamespace: Faster than named tuples but slower than dict (≈0.079 s)

# Interpretation:  
# If object creation speed is critical, `dict` is the best choice. 
# `SimpleNamespace` is acceptable for dynamic attribute-style objects 
# but incurs overhead compared to `dict`.

### **Attribute Access**

# All types are close:
# `dict`: ≈0.026 s
# `namedtuple` / `typing.NamedTuple`: ≈0.026 s
# `SimpleNamespace`: ≈0.031 s

# Interpretation:  
# Access speed differences are negligible for most applications. 
# `SimpleNamespace` is slightly slower than others but still efficient.

### **Mutation**

# `dict`: ≈0.029 s
# `SimpleNamespace`: ≈0.034 s

# Interpretation:
# `dict` remains faster for updates. 
# `SimpleNamespace` is slower but offers cleaner syntax (`obj.x` vs `obj['x']`).

### **Overall Conclusion**

# Use `dict`` when raw speed and flexibility matter.
# Use `namedtuple` / `typing.NamedTuple` for immutability and lightweight structured data.
# Use `SimpleNamespace`when you need dynamic attributes with dot-access and readability, and performance is not the primary concern.

