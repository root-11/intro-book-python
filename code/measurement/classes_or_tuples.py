# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy"]
# ///
import collections
import timeit
from typing import NamedTuple, Tuple
from dataclasses import dataclass
import math
import numpy as np


class Point:  # Dummy Point class for the Coordinate class to work
    def __init__(self, x, y):
        self.x = x
        self.y = y


class Coordinate:
    """Vector friendly mirror object to Points. """
    __slots__ = ['x', 'y', '_hash']

    def __init__(self, x, y):
        self.x, self.y = x, y
        self._hash = hash((self.x, self.y))

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __hash__(self):
        return self._hash

    def __repr__(self):
        return f"Coordinate({self.x}, {self.y})"

    def __lt__(self, other):
        return self._hash < other._hash

    def __sub__(self, other):
        return Coordinate(self.x - other.x, self.y - other.y)

    def __add__(self, other):
        return Coordinate(self.x + other.x, self.y + other.y)  # not used. Kept for completeness.

    def __mul__(self, scalar: int | float):
        return Coordinate(self.x * scalar, self.y * scalar)  # not used. Kept for completeness.

    def cross(self, other):
        return self.x * other.y - self.y * other.x

    def dot(self, other):
        return self.x * other.x + self.y * other.y  # not used. Kept for completeness.

    def distance(self, other):
        return float(math.hypot(other.x - self.x, other.y - self.y))

    def distance_squared(self, other) -> float:  # to be used for performance critical distance comparisons
        dx: float = other.x - self.x
        dy: float = other.y - self.y
        return dx * dx + dy * dy

    @classmethod
    def from_point(cls, point: Point):
        return Coordinate(point.x, point.y)

    def to_point(self) -> Point:
        return Point(self.x, self.y)

    def to_tuple(self) -> Tuple[float, float]:
        return (self.x, self.y)

    def __iter__(self):
        yield self.x
        yield self.y

    def __len__(self):
        return 2


# Define a named tuple for comparison
CoordinateNamedTuple = collections.namedtuple("CoordinateNamedTuple", ["x", "y"])


# Define a typing.NamedTuple for comparison
class CoordinateTypingNamedTuple(NamedTuple):
    x: float
    y: float


@dataclass(frozen=True, slots=True)
class CoordinateDataclass:
    x: float
    y: float


# Number of iterations for the test
NUM_ITERATIONS = 1_000_000

tests = [
    "test_tuple = (10.0, 20.0)",
    "test_named_tuple = CoordinateNamedTuple(10.0, 20.0)",
    "test_typing_named_tuple = CoordinateTypingNamedTuple(10.0, 20.0)",
    "test_coordinate = Coordinate(10.0, 20.0)",
    "test_dataclass = CoordinateDataclass(10.0, 20.0)",
]
results = []
for string in tests:
    exe_time = timeit.timeit(string, globals=globals(), number=NUM_ITERATIONS)
    results.append([exe_time, string])

# Numpy SoA: a single bulk allocation for two columns produces all
# NUM_ITERATIONS rows-worth of data in one call. Per-row time is
# total_time / NUM_ITERATIONS. We measure with number=1, repeat=10
# and take the minimum to get a stable estimate of the bulk cost.
numpy_stmt = (
    f"a = np.full({NUM_ITERATIONS}, 10.0); "
    f"b = np.full({NUM_ITERATIONS}, 20.0)"
)
numpy_times = timeit.repeat(numpy_stmt, globals=globals(), number=1, repeat=10)
results.append([min(numpy_times),
                f"numpy SoA: a, b = np.full({NUM_ITERATIONS}, 10.0) ×2  (one bulk call)"])

results.sort()
for exe_time, string in results:
    print(f"{exe_time:.6f} for {string}")

# Pre-recorded numbers (this author's machine) — your absolute numbers will
# differ but the ordering and order-of-magnitude are stable:
#   ~0.0006 s for numpy SoA (two bulk np.full calls, 1M rows-worth)
#   ~0.005  s for tuple (1M individual constructions)
#   ~0.093  s for collections.namedtuple
#   ~0.097  s for typing.NamedTuple
#   ~0.103  s for class with __slots__
#   ~0.144  s for @dataclass(frozen=True, slots=True)