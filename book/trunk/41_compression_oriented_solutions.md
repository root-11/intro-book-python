# Solutions: 41 — Compression-oriented programming

These exercises are reflective; the work is *audit and rewrite*, not measurement. The answers reflect typical patterns rather than any specific run.

## Exercise 1 — Find a too-early abstraction

A frequent finding in code reviews: a `class WorldComponent(ABC)` with abstract `update`, `serialize`, `inspect` methods, implemented by exactly one subclass (`Creature`). The hierarchy was designed for a hypothetical "future components"; future-them never arrived. Inlining the abstract methods directly into `Creature` deletes the hierarchy and makes the code shorter.

Other shapes that turn out speculative on close inspection:

- A `Protocol` named `Movable` implemented by one class.
- A `Strategy` pattern with one strategy.
- A `Factory` that always returns the same concrete type.
- A `Repository` interface with no second implementation.

All can be inlined. The cost of the inlining is small (a few lines deleted); the benefit is large (one fewer concept to track).

## Exercise 2 — Three concrete versions

```python
def filter_by_hunger(world, hunger_threshold: float) -> np.ndarray:
    """Returns ids of creatures whose energy is below threshold."""
    mask = world.energy[: world.n_active] < hunger_threshold
    return world.id[: world.n_active][mask]

def filter_by_age(world, age_threshold: int) -> np.ndarray:
    """Returns ids of creatures older than threshold."""
    age = world.tick - world.birth_t[: world.n_active]
    mask = age > age_threshold
    return world.id[: world.n_active][mask]

def filter_by_location(world, x: float, y: float, radius: float) -> np.ndarray:
    """Returns ids of creatures within radius of (x, y)."""
    dx = world.pos_x[: world.n_active] - x
    dy = world.pos_y[: world.n_active] - y
    mask = dx*dx + dy*dy < radius*radius
    return world.id[: world.n_active][mask]
```

Three two-line functions. Each is self-documenting; each reads cleanly. The shared shape is "compute mask, index ids".

## Exercise 3 — Resist extraction

The "obvious" abstraction:

```python
def filter_by(world, condition: callable) -> np.ndarray:
    mask = condition(world)
    return world.id[: world.n_active][mask]

# Usage:
filter_by(world, lambda w: w.energy[: w.n_active] < 10.0)
filter_by(world, lambda w: w.tick - w.birth_t[: w.n_active] > 100)
filter_by(world, lambda w: ((w.pos_x[: w.n_active] - 5)**2 + (w.pos_y[: w.n_active] - 5)**2) < 4)
```

Compare:

- The three concrete functions read directly. Each name describes what it does.
- The lambda-based abstraction reads worse. The call site has to inline what was previously a named function; the closures obscure the intent.

The abstraction is *not* a compression — it does not save code (the call sites are now longer than the function bodies); it does not improve clarity (named functions beat anonymous lambdas); it does not enable composition (the lambdas don't have natural names to reuse).

Resist. Keep the three concrete functions. The "DRY" instinct here is wrong; the named functions are easier to read, test, and maintain than the generic helper.

## Exercise 4 — Add a fourth case

```python
def filter_by_proximity_to_food(world) -> np.ndarray:
    """Returns ids of creatures within range of any food."""
    # creatures × food cross-product to find nearest distance
    cx = world.pos_x[: world.n_active]
    cy = world.pos_y[: world.n_active]
    fx, fy = world.food_x, world.food_y
    # broadcasting: shape (n_creatures, n_food)
    dx = cx[:, None] - fx[None, :]
    dy = cy[:, None] - fy[None, :]
    nearest_dist = np.sqrt((dx*dx + dy*dy).min(axis=1))
    mask = nearest_dist < EAT_RADIUS
    return world.id[: world.n_active][mask]
```

This case is *different*. It needs *two* data sources (creature positions + food positions); the earlier three cases needed only one (creatures). The computation involves cross-product broadcasting; the earlier three are flat element-wise comparisons.

The `filter_by` abstraction from exercise 3 *can't handle this* without major changes. The lambda would need to accept both `creatures` and `food`, and the cross-product reshaping doesn't fit the "predicate returns mask" shape. Trying to force-fit produces awkward code; leaving the proximity filter as its own concrete function reads cleanly.

This is exactly the failure mode the chapter warns about: an abstraction that fits three cases is *not* a guarantee it'll fit the fourth. The discipline is to wait for the fourth (and a fifth, a sixth) before committing to the abstraction.

## Exercise 5 — Audit a `Protocol`

Searching a typical codebase for `typing.Protocol`:

```python
class HasUpdate(Protocol):
    def update(self, dt: float) -> None: ...

# Only one class implements it: Creature.
```

Verdict: speculative. Delete the protocol; the type annotation in the caller becomes `Creature` directly. The protocol was a hedge against a future case that never materialised.

When does a protocol earn its place?

- *Three or more independent implementations exist*. (Plural is the test; one is not enough; two is borderline.)
- *The implementations come from different parties* — your code, a third-party library, a test mock. If all three are in your control, you can just refactor; if one is third-party, the protocol is the only seam available.
- *The interface is stable across implementations*. A protocol that grows to fit every new case turns into the `@property` setter trap: every change costs every consumer.

Without these conditions, a protocol is over-engineering. Delete it; replace with the concrete type; you can always add the protocol back when the third implementation arrives.

## Exercise 6 — A library audit (stretch)

Pick a well-regarded library: `requests`, `httpx`, `polars`, `attrs`.

**`requests`**: The `Session` abstraction is a real compression — every HTTP-heavy project rewrote "keep a connection alive, attach default headers, handle cookies" before `requests` existed. The library captured the pattern. `requests.get`, `requests.post`, etc. fit the dominant case (one-shot request) and the cumulative case (a session). Real compression.

**`polars`**: A re-thinking of `pandas` from a columnar-execution perspective. The patterns it abstracts (lazy query plans, column-store, streaming) were extracted from concrete experience with big-data workflows. Some abstractions feel speculative (the eager-vs-lazy split has had ergonomic issues); the core compression is real.

**`pydantic`**: Real compression of "parse JSON / validate / type-check" workflows. Earned its place because the pattern existed everywhere by hand before. Has accreted features (settings management, validators, computed fields) that drift past the original compression; the core remains useful.

**`attrs`**: Predates `dataclasses` and was the canonical compression of "boilerplate class definitions." When `dataclasses` shipped in stdlib (3.7), much of `attrs`'s mandate was absorbed. `attrs` survived by adding features `dataclasses` lacked. Real compression that the stdlib eventually adopted.

**A counter-example**: many "framework" packages with one major user (the author's own application) are speculative compressions. They impose abstractions that fit only the original use case; downstream users either bend their problem to fit or replace the framework.

The pattern: real compressions look inevitable in retrospect because they were extracted, not invented. Premature abstractions look clever and frustrating in practice because they were invented before the patterns they claim to compress existed.
