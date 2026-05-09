# An Introduction to Programming *using entity-component-systems & existence-based processing in `python`*

_written by [Bjorn Madsen](mailto:dr.bjorn.madsen@gmail.com)_
_updated: 2026-05-09_

> **Read online:** [Codeberg](https://root-11.codeberg.page/intro-book-python/) · [GitHub Pages](https://root-11.github.io/intro-book-python/)
> **Clone source:** `git clone https://codeberg.org/root-11/intro-book-python.git` · `git clone https://github.com/root-11/intro-book-python.git`
> **Issues:** [Codeberg](https://codeberg.org/root-11/intro-book-python/issues) · [GitHub](https://github.com/root-11/intro-book-python/issues)

This book teaches programming from first principles of data-oriented design, entity-component-systems (ECS), and existence-based processing (EBP). It uses Python and `numpy` as the only languages.

The book is structured around forty-three concepts ([the DAG](../concepts/dag.md)) and their canonical wording ([the glossary](../concepts/glossary.md)). Sections are short — two to three pages of prose followed by four to twelve compounding exercises. Concepts are *named* only after they are *built*: every section earns its vocabulary through working code, not the other way around.

The through-line is a small ecosystem simulator built in stages from one hundred wandering creatures to a hundred million streamed ones. The simulator's specification is at [`code/sim/SPEC.md`](../code/sim/SPEC.md).

This is the **Python edition** — a sister volume to the Rust edition of the same book. Same forty-four sections, same DAG, same simulator. The variation is per-chapter commentary on what Python's defaults push the reader into, and why ECS and EBP win even in a slow language. The thesis the edition carries: **ECS and EBP beat OOP because they process more efficiently (operations grouped over arrays), they extend more cleanly (data-oriented composition over class graphs), and they have smaller memory footprint (typed columns over object graphs).**

What carries this edition is the **evidence**. Every load-bearing claim is backed by a measurement the reader can reproduce on their own laptop in under a minute. The exhibits live in [`code/measurement/`](../code/measurement/) and run via `uv run code/measurement/<file>.py`.

This is a work in progress. Section ordering is by the DAG; reading order can be linear (front to back) or by following the cross-links wherever they lead.

## Who this book is for

You used Python last week. You wrote a class, put instances in a list, iterated over them, maybe reached for `pandas` when the list got too big. Your code worked, but it was slower than you expected, and you have started wondering whether the standard idioms are the bottleneck.

This book is for people who want to find out. The premise is that they are — and that the architecture this book teaches is what Python is fast in, when Python is fast at all.

## Background

You should be comfortable with high-school algebra and a command line — running a command, changing directories, reading error messages without panic. A laptop with internet is enough; the book uses Python 3.11+, `numpy`, and `uv` for environment management. Everything else is standard library.

You do *not* need prior expertise in numerics, parallel computing, or game development. The book teaches numpy and the simulator together; the language is a vehicle, not the subject.

## The companion edition

If you already know Python well and want compile-time enforcement of the discipline this book teaches by convention, the [Rust edition](https://root-11.codeberg.page/intro-book/) covers the same forty-four sections in Rust. The architecture is identical; the language differs. Many readers find that watching the borrow checker enforce in Rust what this edition asks for as discipline is a useful calibration in the other direction too.
