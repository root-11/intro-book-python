# Solutions: 42 — You can only fix what you wrote

These exercises are audits, not measurements. Answers reflect typical project patterns; your specific project's audit produces specific answers.

## Exercise 1 — Audit your dependencies

A typical `pyproject.toml`:

```toml
[project]
dependencies = [
    "numpy>=1.24",
    "pandas>=2.0",
    "requests>=2.31",
    "pydantic>=2.0",
    "click>=8.1",
    "tomli>=2.0",
    "tqdm>=4.65",
]
```

Classified by the chapter's categories:

| dependency  | size              | forkable? |
|-------------|-------------------|-----------|
| numpy       | ecosystem-scale   | no (millions of LOC, huge team) |
| pandas      | ecosystem-scale   | no, but worth questioning if it earns its place |
| requests    | small-to-mid      | technically forkable; rarely needed |
| pydantic    | mid-to-ecosystem  | hard to fork; deep adoption in ecosystem |
| click       | mid-size          | forkable in a week |
| tomli       | trivial           | inlinable in a day; or use stdlib `tomllib` in 3.11+ |
| tqdm        | small             | forkable; many forks exist |

The trivial ones (tomli) can sometimes be replaced by stdlib if Python version allows. The mid-size ones (click) deserve a "would I fork it if I had to" decision. The ecosystem-scale ones are commitments; pretend otherwise at your peril.

## Exercise 2 — The from-scratch test

Pick `tqdm` (small, ~5K LOC). The relevant 80% (a basic progress bar):

```python
import sys, time

class SimpleTqdm:
    def __init__(self, iterable, total=None):
        self.iterable = iter(iterable)
        self.total = total if total is not None else len(iterable)
        self.n = 0
        self.start = time.perf_counter()

    def __iter__(self):
        return self

    def __next__(self):
        try:
            item = next(self.iterable)
            self.n += 1
            if self.n % 100 == 0 or self.n == self.total:
                elapsed = time.perf_counter() - self.start
                rate = self.n / elapsed if elapsed > 0 else 0
                pct = 100 * self.n / self.total if self.total else 0
                eta = (self.total - self.n) / rate if rate > 0 else 0
                sys.stderr.write(f"\r{pct:5.1f}% [{self.n}/{self.total}] {rate:.0f} it/s ETA {eta:5.1f}s")
                sys.stderr.flush()
            return item
        except StopIteration:
            sys.stderr.write("\n")
            raise

# usage: for x in SimpleTqdm(range(10_000)): work(x)
```

~25 lines for the relevant 80% of tqdm. The full library handles edge cases (Jupyter, nested bars, dynamic resize, customisation, threading) that this version omits. For a simulator that just wants a progress bar in a CLI: this is enough.

The exercise reveals two things: how much code the dependency *actually* saves (small — most of tqdm's value is the edge cases), and how cheaply you could fork (a day to rewrite the 80%). The dependency is fine to keep, but you now know the replacement cost.

## Exercise 3 — A breakage drill

Pick one dependency — say, `pydantic`. Pretend it's been abandoned tomorrow.

**Migration plan**:

1. **Identify the use case.** What does pydantic do for this project? Probably: parse JSON inputs at API boundaries, validate types, convert nested dicts to typed objects.
2. **Evaluate alternatives.**
   - `attrs` + `cattrs` (still maintained; smaller API surface).
   - `dataclasses` + manual validation (stdlib; no validation built in).
   - `msgspec` (faster, smaller; less mature).
   - Roll your own (a couple hundred lines for the parts we use).
3. **Migration cost.** ~3-5 days for a medium project with hundreds of pydantic models. Models migrate one-by-one; tests catch regressions.
4. **Documentation.** Write the plan into the project README: "If pydantic breaks, we go to `msgspec` (preferred) or roll our own (~300 LOC). Estimated migration: 1 week."

The drill takes an hour. The documented plan saves you a panic when the actual breakage happens.

## Exercise 4 — Small over big

Two packages doing the same job:

| job             | small option            | big option         |
|-----------------|-------------------------|--------------------|
| CLI parsing     | `argparse` (stdlib)     | `click` (mid)      |
| HTTP            | `httpx` (small-mid)     | `requests` (mid)   |
| TOML reading    | `tomllib` (stdlib 3.11+) | `tomli` (trivial)  |
| Progress bar    | rolled (~25 LOC)        | `tqdm` (small)     |
| JSON validation | `msgspec` (small)       | `pydantic` (mid)   |

The small options are usually 70-90% of the functionality with much less surface area. The big options earn their place when their *additional* features are genuinely needed — but most projects don't need them. Default to the small option; upgrade when you hit a specific limitation.

The argparse vs click question is canonical. argparse has a clunkier API; click is friendlier. For a small CLI, the clunkiness is a one-time write; for a large CLI, click's compression earns its place. Pick by *project size*, not by *popularity*.

## Exercise 5 — The pandas question

```python
# pandas form
import pandas as pd
df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
filtered = df[df["x"] > 1]
result = filtered["y"].sum()

# numpy SoA form
import numpy as np
x = np.array([1, 2, 3], dtype=np.int64)
y = np.array([4, 5, 6], dtype=np.int64)
mask = x > 1
result = y[mask].sum()
```

Lines roughly equal; the numpy form is slightly more explicit (no implicit column-name lookup). At runtime:

- pandas: ~50-100 µs (creates a new DataFrame for the filter).
- numpy: ~5-10 µs (no intermediate object).

10× difference for trivial operations. At larger scales, the gap widens because pandas has more per-row overhead.

When does pandas earn its keep?

- *Interactive data exploration*: pandas's pretty printing, .head(), .describe(), .to_csv() are real conveniences.
- *Heterogeneous columns* (mixing float, string, datetime, bool): pandas handles the polymorphism cleanly; numpy structured arrays are worse.
- *Group-by aggregations*: pandas's `.groupby().agg()` is concise; numpy needs explicit handling.
- *Joins between DataFrames*: pandas's merge/join is concise; numpy needs explicit handling.

When pandas is a habit, not a need:

- Inner-loop work on numeric columns. Use numpy.
- High-throughput per-row operations. Use numpy.
- Anywhere the working set is past 100K rows and the operations are simple. Use numpy.

A useful audit: count how many DataFrame columns are pure numeric. If most are, the project is better served by numpy SoA + a thin formatting layer for the times it wants pretty output.

## Exercise 6 — Vendoring (stretch)

```sh
# Copy a small dependency's source into your repo
mkdir -p vendor
cp -r .venv/lib/python3.*/site-packages/tomli vendor/

# Update pyproject.toml
[tool.uv.sources]
tomli = { path = "vendor/tomli" }
```

The package is now under your control. Future maintenance items:

- **Security patches**: you must apply them yourself (the upstream's CVE alerts no longer fix your version).
- **Bug fixes**: you cherry-pick from upstream or write your own.
- **New features**: upstream's improvements don't automatically arrive.

Document in the project README:

```markdown
## Vendored dependencies

- `vendor/tomli`: vendored at v2.0.1 on 2026-05-04. Rationale: stdlib's `tomllib` is sufficient
  in Python 3.11+, but we support 3.10 in this codebase. Future migration plan: drop `tomli`
  when we set 3.11 as minimum.
```

The trade: more work for more agency. Worth it for small packages you depend on at the bottom of your stack; not worth it for ecosystem-scale ones.

The pattern is the same one the book applies to `simlog`: vendor a small, complete reference implementation under your repo's control. Future readers can read it; future you can fix it. The maintenance is *yours* — explicitly chosen, not absorbed by accident.
