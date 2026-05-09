# 41 — Compression-oriented programming

> *Concept node: see the [DAG](../../concepts/dag.md) and [glossary entry 41](../../concepts/glossary.md#41--compression-oriented-programming).*

The instinct most programmers acquire from training is *abstract early*. See a case; imagine the second case; design an interface that handles both. The early abstraction feels tidy. It also breaks down the moment the third or fourth case turns out not to fit.

The data-oriented discipline is the opposite. *Write the concrete case three times before extracting anything.* Then look at the three concrete versions and ask whether the abstraction that fits all three is obvious. Often it is, and the extraction is mechanical. Sometimes it is not — the three cases share less than expected, and the right move is to leave them concrete.

Walk through the failure mode. You write the simulator's `motion` system. You can already see motion would also apply to food drift, particle effects, projectile trajectories. The instinct says: design a generic `Movable` protocol or base class. The discipline says: don't yet. Write motion. Move on.

When the second case arrives — say, food drift — you write it concretely. Maybe it shares 80% of motion's structure. Maybe only 60%. You see this clearly because both versions exist as concrete code, not as imagined cases.

When the third case arrives, look at all three. Now the shared structure is *measured*, not imagined. If the abstraction is obvious, extract it. If the three cases share only a vague shape, leave them. **A bad abstraction is more expensive than three concrete versions of similar code.**

## The Python forms of premature abstraction

Python's flexibility makes premature abstraction especially tempting. Five common forms:

**Inheritance hierarchies.** `class Creature(Entity, Updatable, Persistable, Drawable)` — multiple inheritance offered as a way to compose behaviours that have not yet been written concretely. Each base class declares an abstract method that all subclasses override; each override is a concrete implementation that *would have been written anyway*. The hierarchy adds dispatch overhead and obscures which methods actually run.

**`Protocol` and ABC interfaces designed before two implementations exist.** `class Movable(Protocol): def update(self, dt) -> None: ...` — declared because "we'll have lots of movable things", written without concrete callers. The first concrete `Creature.update` fits the protocol because the protocol was shaped to fit it; the protocol guarantees nothing about a hypothetical second implementation that does not exist.

**`*args, **kwargs` "for flexibility".** A function that takes arbitrary keyword arguments and dispatches inside its body is the runtime form of a premature interface. The signature does not document what it accepts; the body is a switch statement disguised as flexibility.

**Generic helpers parameterised over a `Callable`.** `apply_to_all(creatures, fn)` where `fn` is a one-line lambda — three cases later you have one helper plus three call sites that all read worse than the three concrete two-liners they replaced.

**Plugin systems with no plugins.** A `register(plugin)` API designed before any third party will plug into it. The system carries the architectural cost of a plugin point — abstract interface, lifecycle hooks, configuration — for zero plugins. By the time a plugin arrives, the design no longer fits.

In every case the cost is in the *avoided* abstractions. A library of premature interfaces is a library of code-shaped scar tissue. Each interface fits some of its uses well and others poorly. The misfits add casts, branches, defaults, and special cases. Concrete code has none of these.

## What real compressions look like

The Python ecosystem demonstrates compression-oriented programming repeatedly. `collections.namedtuple` is the abstraction over many concrete row-like tuples; it earned its place because the concrete patterns existed first. `pathlib.Path` is the abstraction over the dozen things you do with file paths; it earned its place because every project was rewriting the same string manipulations. **These abstractions feel inevitable because they are *compressions* of patterns the community had already written by hand many times.**

The opposite — abstractions that did not earn their place — also live in the ecosystem: deep ORM hierarchies designed for hypothetical schemas; "framework" packages with one user; metaclass machinery that solves problems the codebase does not have. They are recognisable by the gap between their surface complexity and their actual use.

The discipline is structural, not stylistic. *Compress when you can see the shape, not before.* The book's own through-line uses it. The simulator was built one concrete piece at a time. The DAG was named after the systems were built, not before. The trunk vocabulary is the compression of patterns that actually emerged.

A useful test: after extracting an abstraction, can the abstraction handle a *fourth* case without a special branch? If yes, the compression is real. If no — if the abstraction grew an `if`/`elif` for the fourth case — the abstraction was wrong, and the fourth case is the case showing it.

The connection to the next chapter is concrete. A third-party library is somebody else's compression — an abstraction they extracted from *their* concrete cases. If your three concrete cases match theirs, the library fits and adopting it saves real work. If they do not, the library is friction at every use. [§42](42_you_can_only_fix_what_you_wrote.md) develops this into the dependency-pricing discipline.

## Exercises

1. **Find a too-early abstraction.** Look at code you have written. Find a class hierarchy, a `Protocol`, or a generic helper with fewer than three concrete uses. Could it be inlined? Often the answer is yes; the abstraction was speculative.
2. **Three concrete versions.** Write `filter_creatures_by_hunger`, `filter_creatures_by_age`, `filter_creatures_by_location`. Three independent functions, two or three lines each. Look at them. Is there an obvious shared abstraction?
3. **Resist extraction.** Even with an obvious abstraction in exercise 2, ask: do the three concrete versions read more clearly *as concrete versions*? In some cases yes — three numpy one-liners (`creatures[ids][energy[ids] < THRESHOLD]`, etc.) are more legible than a generic `filter_by(creatures, ids, predicate)` with a closure that hides the actual condition.
4. **Add a fourth case.** Suppose you also want `filter_creatures_by_proximity_to_food`. Does this fit the abstraction from exercise 2? If yes, the abstraction holds. If no (the proximity calculation needs `food`, which the others do not), the abstraction was a tight fit, and the fourth case requires either a new abstraction or a different concrete shape.
5. **Audit a `Protocol`.** If your code uses `typing.Protocol`, find one. Count how many concrete classes implement it. If only one does, the protocol was speculative; consider inlining the interface and deleting the protocol.
6. *(stretch)* **A library audit.** Look at one Python package you have used (not stdlib, not numpy/scipy). Identify the abstractions it offers. For each, ask: does it match three or more concrete cases that came before it, or is it an abstraction of one case generalised on speculation? The answer says whether the package is a real compression or a guess.

Reference notes in [41_compression_oriented_solutions.md](41_compression_oriented_solutions.md).

## What's next

[§42 — You can only fix what you wrote](42_you_can_only_fix_what_you_wrote.md) extends compression-oriented programming to dependencies: every package is somebody else's abstraction; adopting it is a bet that their compression matches yours.
