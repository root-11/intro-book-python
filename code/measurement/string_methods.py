# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
import statistics, timeit

setup = "x, y = 123, 456.789"

def bench(stmt, setup=setup, number=3_000_000, repeat=7):
    times = timeit.repeat(stmt, setup=setup, number=number, repeat=repeat)
    return {
        "median": statistics.median(times),
        "stdev": statistics.pstdev(times),
        "runs": times,
    }

results = {
    "%-format": bench('"values: x=%s, y=%.2f" % (x, y)'),
    "f-string": bench('f"values: x={x}, y={y:.2f}"'),
    ".format":  bench('"values: x={}, y={:.2f}".format(x, y)'),
}

for k, v in results.items():
    print(k, v)

# results:
# --------
# %-format {'median': 0.43349, 'stdev': 0.0025740, 'runs': [0.43306, 0.43371, 0.43349, 0.43317, 0.43355, 0.44018, 0.43152]}
# f-string {'median': 0.47408, 'stdev': 0.0051570, 'runs': [0.46964, 0.47502, 0.47408, 0.47973, 0.47428, 0.46447, 0.46522]}
# .format  {'median': 0.54085, 'stdev': 0.0005862, 'runs': [0.54156, 0.54199, 0.54044, 0.54042, 0.54034, 0.54085, 0.54112]}
