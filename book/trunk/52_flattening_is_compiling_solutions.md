# Solutions: 52 - Flattening a tree is compiling it

Numbers below are the Ryzen 9 270 / CPython 3.14.5 / numpy 2.4.4 figures from [`code/exprtree/exprtree.py`](https://github.com/root-11/intro-book-python/blob/main/code/exprtree/exprtree.py); cross-machine capture is pending, so treat the shape as the claim, not the digits.

## Exercise 1 - Four forms, one number

```python
CONST, VAR, ADD, SUB, MUL = 0, 1, 2, 3, 4

def eval_flat(code, x):                 # code is [(tag, value), ...] in post-order
    stack = []
    for t, v in code:
        if t == CONST:  stack.append(v)
        elif t == VAR:  stack.append(x)
        elif t == ADD:  b = stack.pop(); a = stack.pop(); stack.append(a + b)
        elif t == SUB:  b = stack.pop(); a = stack.pop(); stack.append(a - b)
        else:           b = stack.pop(); a = stack.pop(); stack.append(a * b)
    return stack[0]

def eval_vec(code, xs):                 # xs is a numpy array; stack holds arrays
    stack = []
    for t, v in code:
        if t == CONST:  stack.append(v)
        elif t == VAR:  stack.append(xs)
        elif t == ADD:  b = stack.pop(); a = stack.pop(); stack.append(a + b)
        elif t == SUB:  b = stack.pop(); a = stack.pop(); stack.append(a - b)
        else:           b = stack.pop(); a = stack.pop(); stack.append(a * b)
    return stack[0]
```

The contract check builds all four from one expression and asserts `boxed(x) == arena(x) == flat(x)` bit for bit across many `x`, and `vec(xs)` elementwise against the array. It passes. Both the scalar ops and numpy's `float64` ops are IEEE-754 in the same associative order, so the results are identical to the last bit - which is the floor: every timing below compares four implementations of the *same* arithmetic, so a divergence would mean you are timing four different sums.

## Exercise 2 - Trace the stack by hand

For `(x + 2) * 3` the compute-order list is `x 2 + 3 *`, traced in the chapter to 18 at `x = 4`. Take a subtraction, `(x - 5) * (x + 1)`. Post-order writes every child before its parent: `x 5 - x 1 + *`. At `x = 4`:

```
x  -> push           pad: [4]
5  -> push           pad: [4, 5]
-  -> pop two, sub   pad: [-1]        (4 - 5)
x  -> push           pad: [-1, 4]
1  -> push           pad: [-1, 4, 1]
+  -> pop two, add   pad: [-1, 5]     (4 + 1)
*  -> pop two, mul   pad: [-5]        (-1 * 5)
```

The list never looks back. Each operator consumes values already on the pad and leaves one behind, so the read head only ever moves forward. Subtraction is the one to be careful with: it is not commutative, so the order of the two pops matters - the first pop is the right operand, the second is the left (`a - b` with `a` popped second), which is exactly how the evaluator is written.

## Exercise 3 - The size sweep, and the flat interpreter floor

Bulk-evaluate each scalar form and read nanoseconds per evaluation:

| nodes | boxed | arena | flat | flat vs boxed |
|---:|---:|---:|---:|---:|
| 15 | 681 | 738 | 755 | 0.90x |
| 255 | 11,200 | 12,200 | 10,900 | 1.03x |
| 4,095 | 187,000 | 215,000 | 175,000 | 1.07x |
| 65,535 | 4,150,000 | 4,160,000 | 3,210,000 | 1.29x |
| 262,143 | 26,700,000 | 20,100,000 | 12,900,000 | 2.07x |

The three stay within about a factor of two, and there is no cache-resident band where the pointer tree decisively wins - the Rust edition's whole middle regime is absent. What they are all paying for is the interpreter: every node is a tag check, a dispatch, and an arithmetic op, on the order of a hundred nanoseconds, and that per-node tax is the floor under all three. The flat form is fastest because it is the only one without a recursive function call per node, and its lead widens at the top as the boxed tree's scattered objects begin to miss cache as well - but the layout never produces the clean crossover it produces in Rust, because the interpreter cost is larger than the memory cost it would otherwise expose.

## Exercise 4 - The vectorised win

Evaluate the run-in-order program over a 100k-input array in one pass, ns per input against the scalar flat form:

| nodes | flat scalar (ns/eval) | flat vectorised (ns/input) | speedup |
|---:|---:|---:|---:|
| 255 | 10,900 | 121 | 90x |
| 4,095 | 175,000 | 1,610 | 109x |
| 65,535 | 3,210,000 | 25,500 | 126x |
| 262,143 | 12,900,000 | 72,900 | 177x |

The op count does not change - the vectorised evaluator runs the same list of operations - but each op now does the arithmetic for a hundred thousand inputs in a single numpy call. The interpreter dispatch that the scalar form pays once per input is paid once per *batch*, so it divides away as the batch grows. That is where the per-element Python cost goes, and it is the same move the trunk made everywhere: do the loop in C over a column, not in Python over an element. The layout was never the win in Python; leaving per-element Python was.

## Exercise 5 - The cost of editing compiled code

Swap one subtree on each form, 2000 edits at 8191 nodes:

| rep | ns / edit |
|---|---:|
| boxed | 830 |
| arena | 970 |
| flat | 11,500 |

The boxed tree navigates to the changed node (O(depth)) and swings one child reference to a fresh clone of the graft (O(graft)). The arena appends the graft's nodes and repoints one index, about the same. The flat form has no cheap edit: its order *is* the program, so any change to the shape invalidates the linear sequence, and you re-linearise the whole expression from scratch - O(N), about fourteen times the pointer edit here and widening with the tree. This is the weakness of compiled code stated mechanically: you cannot edit it in place, because the thing that made it cheap to run was committing to one traversal order ahead of time.

## Exercise 6 - The break-even

A workload is some fraction `r` of edits and the rest evaluations. The boxed tree has the cheap edit and the slow walk; the flat form has the slow edit and the fast walk. They break even where the flat form's faster evaluations stop repaying its expensive rebuilds. From the per-op costs at 8191 nodes (boxed: 830 ns edit, 451 µs eval; flat: 11,500 ns edit, 401 µs eval) the crossover is `r* = (451 - 401) / ((11.5 - 0.83) / 1000 + (451 - 401)) ≈ 0.82`.

That is far higher than the Rust edition's ~0.2, and the reason is the evaluation side, not the edit side. In Rust the compiled form's per-evaluation advantage is modest (the pointer walk is fast until it leaves cache), so it only repays its O(N) rebuild in the compute-many corner. In Python the pointer tree pays a full recursive function call per node, so its evaluation is markedly slower than the flat loop at every size; the compiled form's per-evaluation advantage is large, and it repays the rebuild across almost the whole range. The pointer tree wins only when the workload is nearly all edits. And the gap tilts further toward compile as the tree grows, because the flat form's evaluation lead grows with N faster than its O(N) rebuild penalty does.

## Exercise 7 - Find the regime in the wild

Three real expression-tree systems, placed on the change-it-versus-compute-it line:

| system | edits | evaluations | regime |
|---|---|---|---|
| spreadsheet column | retyped rarely | recomputed over thousands of rows | compute-many |
| database query | planned once | run over every row | compute-many |
| vectorised feature transform | defined once | applied to a whole dataset | compute-many, extreme |

All three live far past `r* = 0.82`. Compile one expression once and evaluate it over a million-row column and the single O(N) cost of writing the program out - and re-linearising on a shape change - is amortised to nothing: the per-input saving has repaid it thousands of times over. In Python that "evaluate over a column" is the vectorised stack machine, which is why the compute-many regime and the vectorised regime are the same regime - and the same reason the next chapter's full recompute is a single straight sweep rather than a tree walk.
