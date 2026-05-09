# 40 — Mechanism vs policy

> *Concept node: see the [DAG](../../concepts/dag.md) and [glossary entry 40](../../concepts/glossary.md#40--mechanism-vs-policy).*

The kernel of a system exposes verbs. The rules — what's allowed, what triggers what — live at the edges. Confusing the two is how systems calcify; once a kernel knows about a rule, the rule cannot change without rewriting the kernel.

The principle is older than ECS. It is named in operating-system kernel design (Mach, X11, Plan 9 all teach this rule), in network-protocol design (TCP is mechanism, congestion control is policy), and in file-system design (read/write/seek is mechanism, access control is policy). The same shape applies to ECS systems.

In the simulator:

- `cleanup` is **mechanism**. It takes `to_remove` and `to_insert`, applies them via the bulk-mask filter and append patterns from [§22](22_mutations_buffer.md), and updates `id_to_slot`. It has no opinion about *which* creatures should be removed or *why*. It just commits the changes its callers asked for.
- `apply_starve` is **policy**. It reads `energy` and pushes ids of creatures with `energy <= 0` to `to_remove`. The rule "creatures die when energy reaches zero" lives here. Change the rule to `energy < -10` or `energy < threshold for 100 ticks` and only `apply_starve` changes; cleanup stays the same.

The separation pays off in three places.

**Replaceable rules.** A new gameplay variant — "creatures don't die, they hibernate" — is a new policy on top of unchanged mechanism. `apply_starve` becomes `apply_hibernate`; cleanup still works because cleanup does not know what these systems are doing. The kernel is stable; rules are mobile.

**Composable rules.** Two policies acting on the same kernel compose: one system marks "expired" creatures, another marks "predated" creatures. Both push to `to_remove`. Cleanup applies both batches without knowing why either was set.

**Testable rules.** A test fixture sets up `to_remove` and `to_insert` directly, runs `cleanup` alone, and asserts on the result. The mechanism is testable in isolation. Each policy's test fixture sets up `creatures` and asserts on what the policy pushes to the buffer. Mechanism tests and policy tests don't need each other.

## Three Python anti-shapes that bury policy in mechanism

Python makes mechanism-policy entanglement easy to reach for. Three patterns worth naming.

**`@property` setters that validate and commit.** A `@property` that runs business rules in its setter is policy buried inside attribute assignment:

```python
# anti-pattern: bad!
class Creature:
    @property
    def energy(self): return self._energy
    @energy.setter
    def energy(self, v):
        if v < 0:
            self._dead = True              # policy: "below zero is dead"
            self._world.dead_table.add(self.id)   # mechanism: live-table mutation
        self._energy = v
```

Two roles fused into one assignment. Replacing the policy ("hibernate at zero") requires editing the setter; replacing the mechanism (buffered cleanup instead of live-table mutation) requires editing the same setter. They have become the same change.

**Decorators that hide control flow.** `@lru_cache`, `@retry`, `@require_auth`, `@validate_input` all run code around the function they wrap — by definition, hidden from the call site. When the decorator decides *whether* the function runs, it is a policy embedded in mechanism:

```python
# anti-pattern: bad!
@cache_for(seconds=60)
@require_role("admin")
def remove_creature(world, cid): ...
```

The function's read-set and write-set are no longer derivable from its signature. Whether it runs depends on cache state and role state — invisible at the call site. The §13 contract is gone.

**`__getattr__` / `__setattr__` overrides.** When an arbitrary read of `creature.foo` triggers a database lookup or a network call, the simulator's tick is no longer pure. Every `getattr` could now be I/O. The boundary from §35 is breached at the most innocuous-looking line.

The fix in all three cases is the same shape as the §22 cleanup pattern: separate the *deciding* (policy) from the *committing* (mechanism). The decision goes into a system whose write-set is a buffer; the committing system reads the buffer and applies it. Two functions, two read-sets, two write-sets — and the rule lives in exactly one of them.

## The book's anti-pattern, in one line

A system that mutates a "live" table directly:

```python
# anti-pattern: bad!
def food_spawn(food, world):
    if some_condition(world):
        food.append(...)         # bypasses to_insert; cleanup is now redundant
```

Now `food_spawn` is doing both the *deciding* (when food appears) and the *committing* (writing to `food`). Two changes need rewriting it: a new spawn rule (policy change) and a new cleanup mechanism (mechanism change). They have become the same change. The kernel is married to its current rule.

The fix is to push to `to_insert` instead, letting cleanup commit. The two roles are separable because they were designed to be — through the buffering pattern from [§22](22_mutations_buffer.md), which is itself a mechanism-vs-policy separation. The *mechanism* is "apply changes at the boundary"; the *policy* is "what changes to apply".

Mechanism vs policy is therefore not a separate discipline. It is the rule that every previous chapter has been respecting implicitly. Naming it makes it visible.

## Exercises

1. **Find the mechanism.** For each system in your simulator (motion, food_spawn, next_event, apply_eat, apply_reproduce, apply_starve, cleanup, inspect), classify: is this *mechanism* (committing what something else asked for), *policy* (deciding what to ask for), or both? Note where each role lives.
2. **Replace a policy.** Change `apply_starve`'s rule from `energy <= 0` to `(energy < -10) & (age > 100)`. Confirm: only `apply_starve` changes; `cleanup` stays untouched.
3. **Add a new policy on the same mechanism.** Write a new system `apply_predation` that pushes ids of "predated" creatures (some other rule) to `to_remove`. The two policies' outputs both flow to cleanup, which applies them without distinction.
4. **Spot the anti-pattern.** Find any place in your simulator where a system writes directly to a "live" table instead of to `to_insert` or `to_remove`. Refactor.
5. **Audit your decorators.** Search your code for `@property` with side-effecting setters, `@cached` decorators on stateful functions, or `__getattr__`/`__setattr__` overrides. Each is a candidate for the policy-buried-in-mechanism trap. For each, ask: can the policy be extracted into a system whose write-set is a buffer?
6. *(stretch)* **A second mechanism.** Suppose you want a "soft delete" — creatures move to a `dead` table instead of being removed. Implement a new mechanism (`cleanup_with_archive`) without touching the existing policies. The same `to_remove` ids; different mechanism applied. Switch between them by swapping the system in the DAG, not by editing the systems that produce the data.

Reference notes in [40_mechanism_vs_policy_solutions.md](40_mechanism_vs_policy_solutions.md).

## What's next

[§41 — Compression-oriented programming](41_compression_oriented.md) is the discipline for writing the kernel-and-policies in the first place: write three concrete cases before extracting any abstraction.
