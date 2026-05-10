# Solutions: 40 — Mechanism vs policy

## Exercise 1 — Find the mechanism

| system            | role               | what's buried where                       |
|-------------------|--------------------|-------------------------------------------|
| `motion`          | mechanism + policy | mechanism: update position from velocity; policy: assumes velocity is correct (could be wrong if integration scheme matters) |
| `food_spawn`      | policy             | decides *when* and *where* food appears; pushes to `to_insert` |
| `next_event`      | policy             | decides which events to fire; pushes to `pending_event` |
| `apply_eat`       | policy             | decides who eats (highest-priority overlap, ties broken by id); pushes `to_remove` + `energy_delta` |
| `apply_reproduce` | policy             | decides who reproduces (threshold); pushes to `to_insert` |
| `apply_starve`    | policy             | decides who dies (threshold); pushes `to_remove` |
| `cleanup`         | mechanism          | applies buffered changes; doesn't know what they mean |
| `inspect`         | observer           | reads everything, writes nothing; pure mechanism (no policy) |

`motion` is the trickiest: the per-tick update is mechanism, but the *integration scheme* (Euler vs Verlet vs Runge-Kutta) is a policy. For most simulators the scheme is fixed, but in physics-focused work it's a policy variable that should be extractable.

## Exercise 2 — Replace a policy

```python
# Before
def apply_starve(world, buffer):
    starvers = np.where(world.energy[: world.n_active] <= 0)[0]
    for s in starvers:
        buffer.to_remove.append(int(world.id[s]))

# After — different rule, same mechanism
def apply_starve_v2(world, buffer):
    starvers = np.where(
        (world.energy[: world.n_active] < -10) &
        (world.age[: world.n_active] > 100)
    )[0]
    for s in starvers:
        buffer.to_remove.append(int(world.id[s]))
```

`cleanup` is unchanged. The new rule replaces the old; nothing else cares. This is the test of clean mechanism-policy separation: a policy change is a one-file diff.

## Exercise 3 — Add a new policy on the same mechanism

```python
def apply_predation(world, buffer):
    """A new policy: creatures within predation_range of a predator are eaten."""
    for pred in world.predators:
        nearby = np.where(
            ((world.pos_x[: world.n_active] - pred.x)**2 +
             (world.pos_y[: world.n_active] - pred.y)**2) < pred.range**2
        )[0]
        for s in nearby:
            buffer.to_remove.append(int(world.id[s]))

# Both apply_starve and apply_predation push to the same to_remove
# cleanup applies both batches without knowing which policy contributed which ids
```

Two policies, one mechanism. The cleanup pass deduplicates (`np.unique` inside cleanup, per [§22](22_mutations_buffer.md)) so a creature that's both starving *and* predated is correctly removed once. Two policies could disagree (one wants to remove, another wants to keep alive); resolving that disagreement is a *third* policy that runs before either — *meta-policy* — and it lives at the cleanup boundary just like the other two.

## Exercise 4 — Spot the anti-pattern

Common offenders:

```python
# anti-pattern: bad! food_spawn writes directly to live food table
def food_spawn(food, world, rng):
    if rng.uniform() < 0.1:
        food.append(rng.uniform(0, 100, 2))      # ← live mutation, no buffer

# Fix: push to_insert_food
def food_spawn(world, buffer, rng):
    if rng.uniform() < 0.1:
        buffer.to_insert_food.append(rng.uniform(0, 100, 2))
```

```python
# anti-pattern: bad! cleanup contains a rule (a policy)
def cleanup_bad(world, buffer):
    for cid in buffer.to_remove:
        if world.is_special(cid):
            continue                              # ← policy: "special creatures don't die"
        # ... apply the remove ...

# Fix: the special-handling is its own policy that runs before cleanup
def filter_specials(world, buffer):
    buffer.to_remove = [cid for cid in buffer.to_remove if not world.is_special(cid)]

def cleanup_clean(world, buffer):
    # no policy here; just commit what's in the buffers
    ...
```

The audit pattern: read each system. Ask "what decision is this making?" and "what action is this taking?" If both, split into a decider and an applier.

## Exercise 5 — Audit your decorators

```python
# Decorator that hides control flow
@cache_for(seconds=60)
@require_role("admin")
def remove_creature(world, cid):
    ...
```

Three policy decisions baked in:

1. The function's *result is cached* (no actual call if recent result exists). Policy: "cache for 60 seconds." Where does this rule belong? Almost never at the function definition; it's a deployment concern.
2. The function *only runs for admins*. Policy: authorisation. Where does it belong? At the *caller* or at a request-routing layer, not at the function definition.
3. The function *applies a removal*. Mechanism. This is the legitimate concern.

Refactor:

```python
def remove_creature(world, cid):
    """Mechanism only: applies a removal. No caching, no auth."""
    ...

# Caller decides whether to call:
if user.has_role("admin") and not cache.has(cid, ttl=60):
    remove_creature(world, cid)
    cache.set(cid)
```

Policy lives at the call site, where the context is. The function does one thing.

## Exercise 6 — A second mechanism (stretch)

```python
def cleanup_with_archive(world, buffer):
    """A different mechanism: 'removed' creatures move to a `dead` table instead of being deleted."""
    if buffer.to_remove:
        ids = np.unique(np.array(buffer.to_remove, dtype=np.uint32))
        slots = world.id_to_slot[ids]
        # Copy the soon-to-be-removed rows into the dead table
        n_dead_before = world.dead_count
        n_dying = len(ids)
        for col_name in world.column_names:
            getattr(world.dead, col_name)[n_dead_before : n_dead_before + n_dying] = \
                getattr(world, col_name)[slots]
        world.dead_count += n_dying
        # Now do the regular remove (compact the live table)
        keep_mask = np.ones(world.n_active, dtype=bool)
        keep_mask[slots] = False
        for col_name in world.column_names:
            col = getattr(world, col_name)
            col[: keep_mask.sum()] = col[: world.n_active][keep_mask]
        world.n_active = int(keep_mask.sum())
        # ... update id_to_slot ...
        buffer.to_remove.clear()
    # ... insertions same as before ...
```

`apply_starve` and `apply_predation` are unchanged. They still push to `to_remove`. The mechanism that interprets `to_remove` now archives instead of dropping. Swap mechanisms by changing one entry in the DAG (`cleanup` → `cleanup_with_archive`); the policies don't notice.

This is the architectural payoff. Mechanism is a *plugin*; policies are *consumers*. Each can change independently of the other.
