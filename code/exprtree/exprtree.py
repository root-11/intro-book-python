# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy"]
# ///
"""
§52 exhibit - flattening a tree is compiling it, in Python+numpy.

Counterpart to the Rust edition's code/exprtree crate. Same project: one
arithmetic expression, several representations, measured against each other to
find a *crossover*, not a verdict.

The Rust lesson is about cache: a flat post-order "stack machine" reads memory
front to back and pulls away from the pointer tree once the tree outgrows cache,
while the index-arena buys nothing because it still hops in tree order.

The Python lesson DIVERGES, and the divergence is the chapter:

  - All three scalar forms (boxed pointer tree, index arena, flat stack machine)
    are interpreter-bound. Every node is a handful of Python bytecodes, ~100 ns
    of pure dispatch, which swamps the cache effects the Rust edition measures.
    So the layout barely matters: flat is modestly ahead because it drops the
    recursive call per node, but there is no cache-resident band to see.

  - The real escape in Python is NOT a better layout for one evaluation. It is
    to stop paying the interpreter per node at all: run the SAME flat program
    over a whole numpy array of inputs at once, so each op's dispatch is
    amortised across thousands of values. That vectorised stack machine is the
    Python "compile" that actually pays - tens to hundreds of times per value.

  - The edit-vs-evaluate crossover holds in kind: the boxed tree swings one
    reference in O(depth); the flat program has no cheap edit and must
    re-linearise the whole expression, O(N). Compile when you evaluate far more
    than you edit.

Claims:
  C1 all forms agree           scalar boxed == arena == flat, bit for bit, and
                               the vectorised form agrees elementwise.
  C2 layout barely matters     the three scalar forms stay within a small factor
                               across sizes; no cache crossover appears.
  C3 vectorising is the win    the vectorised stack machine is many times cheaper
                               per value than the best scalar form.
  C4 edit vs evaluate          boxed edit is O(depth); flat edit is O(N); a
                               break-even edit-fraction follows from the two.

Run:  uv run code/exprtree/exprtree.py
"""

import random
import time

import numpy as np

SEED = 0x12345678

CONST, VAR, ADD, SUB, MUL = 0, 1, 2, 3, 4
OPS = (ADD, SUB, MUL)


# ---------------------------------------------------------------------------
# Representation 1: boxes and arrows - a mutable node is a list [tag, a, b].
# Leaves: [CONST, value, None] / [VAR, None, None].  Internal: [op, left, right].
# Lists (not tuples) so a structural edit can swing one child reference in place.
# ---------------------------------------------------------------------------

def gen_boxed(depth, rng):
    if depth == 0:
        if rng.random() < 0.5:
            return [VAR, None, None]
        return [CONST, 0.5 + rng.random(), None]
    left = gen_boxed(depth - 1, rng)
    right = gen_boxed(depth - 1, rng)
    return [rng.choice(OPS), left, right]


def eval_boxed(e, x):
    t = e[0]
    if t == CONST:
        return e[1]
    if t == VAR:
        return x
    a = eval_boxed(e[1], x)
    b = eval_boxed(e[2], x)
    if t == ADD:
        return a + b
    if t == SUB:
        return a - b
    return a * b


def clone_boxed(e):
    if e[0] in OPS:
        return [e[0], clone_boxed(e[1]), clone_boxed(e[2])]
    return [e[0], e[1], e[2]]


def mutate_boxed(root, steps, new):
    """Swing the child reference at the end of `steps` to a fresh clone of `new`
    (the Rust edition clones on graft too, so the clone is part of the edit cost).
    O(len(steps)) navigate plus O(|new|) clone, O(1) swap. Returns the new root."""
    graft = clone_boxed(new)
    if not steps:
        return graft
    parent, side, node = None, 1, root
    for go_right in steps:
        if node[0] not in OPS:
            break  # hit a leaf early - replace it via its parent
        parent, side = node, (2 if go_right else 1)
        node = node[side]
    if parent is None:
        return graft
    parent[side] = graft
    return root


# ---------------------------------------------------------------------------
# Representation 2: the flat arena - parallel lists, children named by index.
# build_arena appends in post-order, so the freshly built arena's node order
# already IS a valid post-order program (the flat form, below, exploits that).
# ---------------------------------------------------------------------------

class Arena:
    __slots__ = ("tag", "lhs", "rhs", "val", "root")

    def __init__(self):
        self.tag, self.lhs, self.rhs, self.val = [], [], [], []
        self.root = 0


def build_arena_into(e, ar):
    t = e[0]
    if t == CONST:
        ar.tag.append(CONST); ar.lhs.append(0); ar.rhs.append(0); ar.val.append(e[1])
    elif t == VAR:
        ar.tag.append(VAR); ar.lhs.append(0); ar.rhs.append(0); ar.val.append(0.0)
    else:
        li = build_arena_into(e[1], ar)
        ri = build_arena_into(e[2], ar)
        ar.tag.append(t); ar.lhs.append(li); ar.rhs.append(ri); ar.val.append(0.0)
    return len(ar.tag) - 1


def arena_from_boxed(e):
    ar = Arena()
    ar.root = build_arena_into(e, ar)
    return ar


def eval_arena(ar, i, x):
    t = ar.tag[i]
    if t == CONST:
        return ar.val[i]
    if t == VAR:
        return x
    a = eval_arena(ar, ar.lhs[i], x)
    b = eval_arena(ar, ar.rhs[i], x)
    if t == ADD:
        return a + b
    if t == SUB:
        return a - b
    return a * b


def mutate_arena(ar, steps, new):
    """Append the new subtree's nodes (old ones orphaned - deferred GC) and
    repoint one parent index. O(len(steps) + |new|) - the array order is now
    no longer post-order, which is exactly why the flat form must recompile."""
    new_root = build_arena_into(new, ar)
    if not steps:
        ar.root = new_root
        return
    parent, side, idx = -1, 0, ar.root
    for go_right in steps:
        if ar.tag[idx] not in OPS:
            break
        parent, side, idx = idx, (1 if go_right else 0), (ar.rhs[idx] if go_right else ar.lhs[idx])
    if parent == -1:
        ar.root = new_root
    elif side:
        ar.rhs[parent] = new_root
    else:
        ar.lhs[parent] = new_root


# ---------------------------------------------------------------------------
# Representation 3: the flat post-order program (a stack machine / RPN).
# code is a list of (tag, value) pairs in compute order.
# ---------------------------------------------------------------------------

def compile_flat(ar):
    out = []
    # iterative post-order so a deep arena cannot overflow the Python stack
    order = []
    visit = [ar.root]
    while visit:
        i = visit.pop()
        order.append(i)
        if ar.tag[i] in OPS:
            visit.append(ar.lhs[i])
            visit.append(ar.rhs[i])
    for i in reversed(order):
        out.append((ar.tag[i], ar.val[i]))
    return out


def eval_flat(code, x):
    stack = []
    push, pop = stack.append, stack.pop
    for t, v in code:
        if t == CONST:
            push(v)
        elif t == VAR:
            push(x)
        elif t == ADD:
            b = pop(); a = pop(); push(a + b)
        elif t == SUB:
            b = pop(); a = pop(); push(a - b)
        else:
            b = pop(); a = pop(); push(a * b)
    return stack[0]


def eval_flat_vec(code, xarr):
    """The same program, run once over a whole array of inputs. The stack holds
    numpy arrays; each op is one whole-array operation, so the per-op interpreter
    dispatch is amortised across len(xarr) values. This is the Python 'compile'."""
    stack = []
    push, pop = stack.append, stack.pop
    for t, v in code:
        if t == CONST:
            push(v)  # python scalar; numpy broadcasts it against the arrays
        elif t == VAR:
            push(xarr)
        elif t == ADD:
            b = pop(); a = pop(); push(a + b)
        elif t == SUB:
            b = pop(); a = pop(); push(a - b)
        else:
            b = pop(); a = pop(); push(a * b)
    out = stack[0]
    return np.broadcast_to(out, xarr.shape)  # in case the tree held no Var


# ---------------------------------------------------------------------------
# Benchmark plumbing.
# ---------------------------------------------------------------------------

def nodes_in(depth):
    return (1 << (depth + 1)) - 1


def median(fn, k=3):
    return sorted(fn() for _ in range(k))[k // 2]


def bench_scalar(evalfn, tree_arg, xs, iters):
    acc = 0.0
    t0 = time.perf_counter()
    for i in range(iters):
        acc += evalfn(tree_arg, xs[i & 255])
    dt = time.perf_counter() - t0
    if acc == 123456.789:  # defeat dead-code elimination cheaply (never true)
        print(acc)
    return dt / iters * 1e9  # ns per evaluation


def gen_edits(n, max_depth, rng):
    edits = []
    for _ in range(n):
        ln = rng.randrange(max_depth) + 1
        steps = [rng.random() < 0.5 for _ in range(ln)]
        edits.append((steps, gen_boxed(2, rng)))
    return edits


def main():
    rng = random.Random(SEED)
    xs = [0.5 + rng.random() for _ in range(256)]

    # ---- C1: all forms agree ----
    ok = True
    for depth in (0, 1, 2, 5, 8, 11):
        g = random.Random(0xC0FFEE ^ depth)
        boxed = gen_boxed(depth, g)
        ar = arena_from_boxed(boxed)
        code = compile_flat(ar)
        xarr = np.array(xs, dtype=np.float64)
        vec = eval_flat_vec(code, xarr)
        for k, x in enumerate(xs):
            b = eval_boxed(boxed, x)
            a = eval_arena(ar, ar.root, x)
            f = eval_flat(code, x)
            if not (b == a == f and b == float(vec[k])):
                ok = False
                print(f"  DISAGREE depth={depth} x={x}: boxed={b} arena={a} flat={f} vec={vec[k]}")
                break
    print(f"C1 all forms agree (scalar + vectorised), bit for bit: {'PASS' if ok else 'FAIL'}\n")

    # ---- C2/C3: bulk evaluation across tree sizes ----
    print("== Workload 1: bulk evaluation ==")
    print("ns per scalar evaluation (lower better); last column is the vectorised")
    print("stack machine's ns PER VALUE over a 100k-input batch.\n")
    print(f"{'depth':>5} {'nodes':>10} {'boxed':>11} {'arena':>11} {'flat':>11} {'flat_vec/val':>13} {'vec speedup':>12}")
    xarr = np.array([0.5 + rng.random() for _ in range(100_000)], dtype=np.float64)
    for depth in (3, 5, 7, 9, 11, 13, 15, 17):
        nodes = nodes_in(depth)
        iters = max(3, 3_000_000 // nodes)
        g = random.Random(0x51A11ED5 ^ depth)
        boxed = gen_boxed(depth, g)
        ar = arena_from_boxed(boxed)
        code = compile_flat(ar)
        b = median(lambda: bench_scalar(eval_boxed, boxed, xs, iters))
        a = median(lambda: bench_scalar(lambda t, x: eval_arena(t, t.root, x), ar, xs, iters))
        f = median(lambda: bench_scalar(eval_flat, code, xs, iters))
        # vectorised: one call over the whole batch; ns per value
        def vec_once():
            t0 = time.perf_counter()
            r = eval_flat_vec(code, xarr)
            dt = time.perf_counter() - t0
            if r[0] == 123456.789:
                print(r[0])
            return dt / xarr.size * 1e9
        fv = median(vec_once)
        print(f"{depth:>5} {nodes:>10} {b:>11.1f} {a:>11.1f} {f:>11.1f} {fv:>13.2f} {f / fv:>11.1f}x")

    # ---- C4: edits vs evaluation, and the crossover ----
    edit_depth, n_edits = 12, 2000
    g = random.Random(0xED175EED)
    boxed = gen_boxed(edit_depth, g)
    ar = arena_from_boxed(boxed)
    code = compile_flat(ar)
    edits = gen_edits(n_edits, edit_depth, g)
    nodes = nodes_in(edit_depth)

    def edits_boxed():
        tree = gen_boxed(edit_depth, random.Random(0xED175EED))  # fresh, not timed... rebuilt below
        t0 = time.perf_counter()
        for steps, new in edits:
            tree = mutate_boxed(tree, steps, new)
        return (time.perf_counter() - t0) / n_edits * 1e9

    def edits_arena():
        a2 = arena_from_boxed(gen_boxed(edit_depth, random.Random(0xED175EED)))
        t0 = time.perf_counter()
        for steps, new in edits:
            mutate_arena(a2, steps, new)
        return (time.perf_counter() - t0) / n_edits * 1e9

    def edits_flat():
        a2 = arena_from_boxed(gen_boxed(edit_depth, random.Random(0xED175EED)))
        t0 = time.perf_counter()
        for steps, new in edits:
            mutate_arena(a2, steps, new)
            _ = compile_flat(a2)  # the O(N) re-linearise that flat cannot avoid
        return (time.perf_counter() - t0) / n_edits * 1e9

    eb = median(edits_boxed)
    ea = median(edits_arena)
    ef = median(edits_flat)
    iters = max(3, 3_000_000 // nodes)
    vb = median(lambda: bench_scalar(eval_boxed, boxed, xs, iters))
    vf = median(lambda: bench_scalar(eval_flat, code, xs, iters))

    print(f"\n== Workload 2: structural mutation (depth {edit_depth}, {nodes} nodes, {n_edits} edits) ==\n")
    print(f"{'rep':>8} {'ns / edit':>14} {'ns / eval':>14}")
    print(f"{'boxed':>8} {eb:>14.1f} {vb:>14.1f}")
    print(f"{'arena':>8} {ea:>14.1f} {'':>14}")
    print(f"{'flat':>8} {ef:>14.1f} {vf:>14.1f}")

    # crossover: r* = (eval_b - eval_f) / ((edit_f - edit_b) + (eval_b - eval_f))
    num = vb - vf
    den = (ef - eb) + num
    print("\n== Crossover (flat vs boxed, derived) ==")
    if num <= 0 or den <= 0:
        print(f"  no crossover in [0,1] (flat eval advantage {num:.1f} ns, edit penalty {ef - eb:.1f} ns)")
    else:
        r = num / den
        print(f"  r* = {r:.4f}  (about 1 edit per {(1 - r) / r:.1f} evaluations)")
    print("\nReading: the scalar forms are interpreter-bound and close; the win is")
    print("the vectorised stack machine, which is the compile that pays in Python.")


if __name__ == "__main__":
    main()
