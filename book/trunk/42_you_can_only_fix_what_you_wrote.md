# 42 — You can only fix what you wrote

> *Concept node: see the [DAG](../../concepts/dag.md) and [glossary entry 42](../../concepts/glossary.md#42--you-can-only-fix-what-you-wrote).*

Foreign libraries are allowed in this book. They are not banned. They are *priced*.

Every dependency is a bet. The bet is that someone else will keep the library working — fix bugs, ship versions, respond to security issues, support future Python releases, not abandon the project. The bet has a cost: if the library breaks, you cannot fix it. You can only replace it, fork it, or live with the breakage.

The discipline is to take the bet *consciously*, knowing how much code the dependency saves you and how much risk it carries.

## What risk looks like in Python

**The leftpad equivalent.** An eleven-line npm package was unpublished by its author over a naming dispute, breaking thousands of build pipelines worldwide. Python has had its own versions: `python-twitter` going stale, smaller PyPI packages disappearing or changing maintainers, the `simplejson` / `json` standoff. Every project that depended on these was, structurally, depending on someone else's emotional state.

**Major-version cascade.** A transitive dependency makes a breaking change. Your code does not change. The dependency's dependency does. The build is now broken, sometimes for days, while you wait for an upstream fix or pin a workaround. Python's loose version pinning conventions (`requirements.txt` with `>=` everywhere) make this category larger than it is in stricter ecosystems. You have lost agency over your own build.

**The slow fade.** A package works in production for two years, then its author switches careers, the package stops getting updates, and a future Python release deprecates a feature it relies on. The package still installs for now, but its days are numbered. Migration is on you.

**The Python-version trap.** CPython's deprecation cycle is long but real. A package that uses `imp` (removed in 3.12), or relies on `distutils` (removed in 3.12), or depends on a now-deprecated C-API, will break on a future interpreter. Even active maintainers run out of time; a "we'll fix it before 3.13" is sometimes a promise no one is left to keep.

These are not edge cases. They are the *typical* lifecycle of a dependency relationship. Some libraries beat the curve — `numpy`, `requests`, `pytest`, `sqlite3` (stdlib) — because they are maintained by ecosystems too large to fail. Most do not.

## The discipline

The discipline that follows from this is not "use no dependencies". It is:

1. **Write the from-scratch version first.** If it is fifty lines and two hours, often you do not need the dependency at all. The from-scratch version is also the calibration: how much code does the package actually save?
2. **Read the dependency's source.** Not the docs — the source. How much code is it? Who maintains it? What's its history? Is it actively maintained or coasting? `pip show foo` plus a quick browse of the GitHub repo answers most of these questions in five minutes.
3. **Decide consciously.** Adopt for the right reasons (genuine code savings, ecosystem alignment, escape from your own bug-prone reimplementation). Reject for the wrong reasons (it is there, it is popular, no one questioned it).

## A useful classification by size

- **Trivial** (a few hundred lines or less). Easy to fork, easy to inline. Often easier to write yourself than to take the dependency. Examples: `colorama`, `python-dateutil`'s parts you actually use, half the "utilities" packages on PyPI.
- **Small** (around a thousand lines). Forkable in a day or two. Reasonable to depend on; reasonable to vendor. Examples: `tqdm`, `tomli`.
- **Mid-size** (a few thousand lines, e.g. `attrs`, `click`). Forkable but a real commitment. Adopt cautiously; have a migration plan.
- **Ecosystem-scale** (tens of thousands of lines, large team — `numpy`, `requests`, `pytest`, `sqlalchemy`). Not realistically forkable. Adoption is a commitment to the ecosystem; pretending otherwise is the bug.

## The Python-specific traps

**`pandas`** sits awkwardly between mid-size and ecosystem-scale. The codebase is enormous; the API is huge; the maintainers are competent but the surface area means breaking changes happen regularly. The book's tooling memory says pandas is out for the simulator's hot path; this chapter says: *if you are using pandas because nobody questioned it, that is the wrong reason.* Read the from-scratch alternative — numpy SoA columns plus targeted helpers — and decide consciously.

**ORMs (`sqlalchemy`, `peewee`, Django ORM)** earn their place when the workload genuinely fits the relational model and the ORM's compression matches your access patterns. They do not earn their place when the simulator's data is columnar SoA and the ORM is being used as "the way one talks to a database" out of habit. The §38 framing applies: SQL is at the boundary, not in the hot path.

**`pickle` of complex objects.** §36 covered this. The version-skew risk is real; `protocol=4` is the stable choice when archive longevity matters.

**Async frameworks (`asyncio`, `trio`, `anyio`).** Each is large; each makes architectural commitments that propagate through your code. §31 said async is the wrong tool for CPU-bound work; this chapter adds: even for I/O work, picking an async framework is a decision worth making consciously, not by default.

## The book's worked example

The book's through-line example is the simlog. The simlog implements the generational arena pattern from [§10](10_stable_ids_and_generations.md), the index map from [§23](23_index_maps.md), the buffered cleanup from [§22](22_mutations_buffer.md), the double-buffered serialisation from [§37](37_log_is_world.md), and the np.savez output from [§36](36_persistence_is_serialization.md) — in 700 lines, vendored at `.archive/simlog/logger.py`. Most simulators benefit from it because the from-scratch version is non-trivial. But the from-scratch version is *also* small enough that you could fork and own it if needed. **That balance — small enough to fix, complex enough to want — is the sweet spot.**

The opposite end is `numpy`. Adoption is a commitment to the maintainer team. For most projects this is fine — the team is competent and the ecosystem is durable. But the commitment is real.

The middle ground is uncomfortable. A 2,000-line single-author package on PyPI that is exactly what you need: too big to fork comfortably, too small for ecosystem support. Adopt cautiously; consider vendoring (copying into your repo); be ready to maintain.

The book's discipline lives at this evaluation. **Not "no deps" — "consciously chosen deps, sized to the maintenance you can do".**

## Exercises

1. **Audit your `pyproject.toml` (or `requirements.txt`).** For each direct dependency, classify by the size categories above. The small ones are easiest to fork; the ecosystem-scale ones are too big to fork.
2. **The from-scratch test.** Pick one mid-size or small dependency. Estimate: how long would it take to write the relevant 80% of it from scratch? If less than two days, you have an alternative — keep it in mind for the day the dependency breaks.
3. **A breakage drill.** Pick one dependency. Pretend it is unmaintained. What is your migration path? (Fork? Replace? Live with the bug?) Write the answer in your project's README. The drill is cheap; the breakage is not.
4. **Small over big.** When two packages do the same job, prefer the smaller. A small package is forkable; a large one usually is not. The bigger package's extra features are someone else's needs, not yours.
5. **The pandas question.** If your project uses pandas, audit one DataFrame in your code. Could the same operation be expressed as numpy SoA columns? How much code grows; how much code shrinks; how does the runtime change? You may find pandas earns its keep — or you may find it is a habit no one questioned.
6. *(stretch)* **Vendoring.** Copy one small package's source into `vendor/foo` in your repo. Update `pyproject.toml` to install it from `path = "vendor/foo"` (uv supports this; pip does too via local paths). The package is now under your control. Future breakages are yours to fix; future improvements are yours to apply. The trade is more work for more agency. Document the decision so future maintainers know why.

Reference notes in [42_you_can_only_fix_what_you_wrote_solutions.md](42_you_can_only_fix_what_you_wrote_solutions.md).

## What's next

[§43 — Tests are systems; TDD from day one](43_tests_are_systems.md) is the closing discipline: tests are not a separate framework, they are systems. The same shape that runs the simulator runs its tests.
