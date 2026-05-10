# An Introduction to Programming *using entity-component-systems & existence-based processing in `python`*

_written by [Bjorn Madsen](mailto:dr.bjorn.madsen@gmail.com)_
_updated: 2026-05-09_

> **Read online:** [Codeberg](https://root-11.codeberg.page/intro-book-python/) · [GitHub Pages](https://root-11.github.io/intro-book-python/)
>
> **Clone source:** `git clone https://codeberg.org/root-11/intro-book-python.git` · `git clone https://github.com/root-11/intro-book-python.git`
>
> **Issues:** [Codeberg](https://codeberg.org/root-11/intro-book-python/issues) · [GitHub](https://github.com/root-11/intro-book-python/issues)

<p align="center"><img src="illustrations/classroom.jpg" alt="A classroom: Understand, Model, Solve, Validate, Improve" style="max-height: 360px; max-width: 100%;"></p>

This book teaches programming from first principles of data-oriented design, entity-component-systems (ECS), and existence-based processing (EBP). It uses Python and `numpy` as the only languages.

The book is structured around forty-three concepts ([the DAG](../concepts/dag.md)) and their canonical wording ([the glossary](../concepts/glossary.md)). Sections are short — two to three pages of prose followed by four to twelve compounding exercises. Concepts are *named* only after they are *built*: every section earns its vocabulary through working code, not the other way around.

The through-line is a small ecosystem simulator built in stages from one hundred wandering creatures to a hundred million streamed ones. The simulator's specification is at [`code/sim/SPEC`](../code/sim/SPEC.md).

This is the **Python edition** — a sister volume to the Rust edition of the same book. Same forty-four sections, same DAG, same simulator. The variation is per-chapter commentary on what Python's defaults push the reader into, and why ECS and EBP win even in a slow language. The thesis the edition carries: **ECS and EBP beat OOP because they process more efficiently (operations grouped over arrays), they extend more cleanly (data-oriented composition over class graphs), and they have smaller memory footprint (typed columns over object graphs).**

What carries this edition is the **evidence**. Every load-bearing claim is backed by a measurement the reader can reproduce on their own laptop in under a minute. The exhibits live in [`code/measurement/`](https://codeberg.org/root-11/intro-book-python/src/branch/main/code/measurement) and run via `uv run code/measurement/<file>.py`.

This is a work in progress. Section ordering is by the DAG; reading order can be linear (front to back) or by following the cross-links wherever they lead.

## Who this book is for

You used Python last week. You wrote a class, put instances in a list, iterated over them. Your code worked, but it was slower than you expected, and you have started wondering whether the standard idioms are the bottleneck.

This book is for people who want to find out. The premise is that they are — and that the architecture this book teaches is what Python is fast in, when Python is fast at all.

Many online books include a playground that runs the code in your browser. This one does not, on purpose: the measurements only mean something when they come from *your* hardware.

## Background

You should be comfortable with high-school algebra and a command line — running a command, changing directories, reading error messages without panic. A laptop with internet is enough; the book uses Python 3.11+, `numpy`, and `uv` for environment management. Everything else is standard library.

You do *not* need prior expertise in numerics, parallel computing, or game development. The book teaches numpy and the simulator together; the language is a vehicle, not the subject.

## A first taste

Before any vocabulary is named, here is what an ECS world looks like in fifteen lines of Python. One hundred creatures, each with a position and a velocity, moving for thirty ticks of simulated time. No classes, no instances, no method calls — four `numpy` arrays indexed in lockstep, and a function (the per-tick update) that advances every creature in one stride.

```python
import numpy as np

n = 100
x  = np.arange(n, dtype=np.float32) * 0.1
y  = np.sin(np.arange(n, dtype=np.float32))
vx = ((np.arange(n) * 7) % 11).astype(np.float32) * 0.01 - 0.05
vy = ((np.arange(n) * 13) % 7).astype(np.float32) * 0.01 - 0.03

for tick in range(30):
    x += vx
    y += vy
    if tick % 10 == 0:
        print(f"tick {tick}: creature 17 at ({x[17]:.2f}, {y[17]:.2f})")
```

Run it locally. Three lines print, the script stops. That is the entire shape of what the rest of the book grows: tables (the four arrays), a tick (the outer loop), a system (the per-tick update). Everything that follows is the discipline that lets this same shape carry a hundred million creatures without falling apart.

The familiar Python shape — a `Creature` class, a list of instances, a `step()` method — works at this size too. It stops working at a million, and the reason is in [§2](trunk/02_numbers_and_how_they_fit.md): an order of magnitude more memory per creature, an order of magnitude slower per tick. The book teaches the layout that survives the next zero.

## Running the code

Python has no equivalent of the Rust Playground — there is no browser-hosted runner that reproduces the numbers a chapter quotes. Every measurement and exhibit in this book runs locally, using [`uv`](https://docs.astral.sh/uv/) to manage the Python toolchain and environment. To run anything, you will want a clone of the book's repo:

```sh
git clone https://codeberg.org/root-11/intro-book-python.git
cd intro-book-python
uv run code/measurement/cache_cliffs.py
```

Each `code/measurement/<name>.py` file is one exercise group, runnable in isolation. The numbers it prints are *yours* — they come from your hardware. The exercise asks "how fast does *your* machine run this?", and that question only has a real answer locally.

From the simulator chapters onward (§11+), the exercises stop being self-contained scripts. They build the through-line: a Python program that grows from one hundred wandering creatures to a hundred million streamed ones. That program holds state between runs, which is what `uv run` and the project layout buy you.

## The companion edition

If you already know Python well and want compile-time enforcement of the discipline this book teaches by convention, the [Rust edition](https://root-11.codeberg.page/intro-book/) covers the same forty-four sections in Rust. The architecture is identical; the language differs. Many readers find that watching the borrow checker enforce in Rust what this edition asks for as discipline is a useful calibration in the other direction too.
