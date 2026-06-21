# Solutions: 53 - Staleness flows downhill

Numbers below are the Ryzen 9 270 / CPython 3.14.5 / numpy 2.4.4 figures from [`code/scenegraph/scenegraph.py`](https://github.com/root-11/intro-book-python/blob/main/code/scenegraph/scenegraph.py); cross-machine capture is pending, so treat the shape as the claim, not the digits.

## Exercise 1 - Move a joint by hand

With local offsets shoulder 0, elbow +2, hand +1, the world positions are 0, 2, 3. Change the elbow to +5 and recompute: shoulder stays 0 (nothing above it changed), elbow becomes 5, hand becomes 6. The shoulder did not move; the elbow and the hand - everything beneath the changed joint - did. The rule: **a change to a node's local transform makes its own world transform and every descendant's world transform stale, and nothing else.** Staleness flows downhill and stops at the leaves.

## Exercise 2 - Flat, top-down, by level

Store `parent[i]` and `depth[i]` with the nodes in pre-order, so `parent[i] < i`. Group indices by depth, then compose one level at a time:

```python
def levels_of(depth):
    return [np.where(depth == d)[0] for d in range(depth.max() + 1)]

def propagate_full_vec(W, L, parent, levels):
    for k in range(6):
        W[k][levels[0]] = L[k][levels[0]]          # roots: world == local
    for lvl in levels[1:]:
        p = parent[lvl]
        pa, pb, pc, pd, pe, pf = (W[k][p] for k in range(6))   # gather parents (final)
        la, lb, lc, ld, le, lf = (L[k][lvl] for k in range(6))
        W[0][lvl] = pa*la + pb*ld;  W[1][lvl] = pa*lb + pb*le;  W[2][lvl] = pa*lc + pb*lf + pc
        W[3][lvl] = pd*la + pe*ld;  W[4][lvl] = pd*lb + pe*le;  W[5][lvl] = pd*lc + pe*lf + pf
```

Every node at depth `d` has its parent at depth `< d`, which a previous iteration already finalised, so the gather `W[k][p]` reads only valid values - and because no node's parent is in its own level, the scatter into `W[k][lvl]` cannot disturb a parent another node in the same level is reading. The loop runs once per depth, a few dozen times, not once per node.

## Exercise 3 - The vectorised sweep vs the per-node loop

The per-node version is a Python `for i in range(n)` composing one node from its parent. The level-vectorised version is the code above. Same answers, bit for bit (both are IEEE float64 in the same order). The times:

| nodes | scalar per-node | level-vectorised | speedup |
|---:|---:|---:|---:|
| 10,000 | 18.9 ms | 0.37 ms | 51x |
| 100,000 | 199 ms | 7.5 ms | 26x |
| 1,000,000 | (too slow to bother) | 93 ms | - |

The gap is [§52](52_flattening_is_compiling.md)'s lesson exactly: the per-node loop pays the interpreter once per node, and there are a million of them; the vectorised sweep pays it once per *level*, and does the arithmetic for a whole level in one numpy call. The Rust edition's flat-vs-pointer gap is 2.3x-2.8x, all cache; Python's is an order of magnitude, because the interpreter cost dwarfs the cache cost it would otherwise expose.

## Exercise 4 - The subtree is a slice

Record `subtree[i]` (the node count in `i`'s subtree, `i` included) as you build the tree in pre-order. Then "everything beneath and including `i`" is exactly `range(i, i + subtree[i])` - a contiguous slice, found in O(1), no tree-walking. This is the property the whole incremental scheme leans on, and the one a graph destroys.

## Exercise 5 - The dirty crossover

Recompute a contiguous dirty subtree level by level (group the slice's indices by depth, then run the same `compose_into` per level), against the full vectorised sweep (53.7 ms at 1M nodes):

| dirty % | dirty nodes | recompute-dirty | vs all |
|---:|---:|---:|---:|
| 0.1% | 1,001 | 0.10 ms | 510x |
| 1% | 9,975 | 0.29 ms | 183x |
| 10% | 96,698 | 3.0 ms | 17.8x |
| 20% | 243,749 | 10.0 ms | 5.4x |
| 40% | 421,628 | 33.0 ms | 1.6x |
| 60% | 675,530 | 49.4 ms | 1.1x |
| 100% | 1,000,000 | 88.9 ms | 0.6x |

Recompute-dirty wins hugely when little moved and loses once most of the tree is stale - the crossover here is around two-thirds dirty. At 100% it is *slower* than the full sweep despite doing the same arithmetic, because the full sweep reuses its clean per-level index arrays while the dirty path re-derives a worse-ordered grouping and gathers/scatters over it. When everything changed, the bookkeeping is pure overhead: sweep.

## Exercise 6 - Packed versus scattered

Same dirty count (~97k, 1M-node tree), arranged two ways:

| arrangement | recompute |
|---|---:|
| one contiguous subtree | 2.9 ms |
| same count, scattered | 12.7 ms |

About **4.4x apart** at identical work. Both gather each dirty node's parent transform and scatter the result; for the contiguous subtree those accesses stay inside one local range of the arrays and hit cache, while the scattered set's accesses span the whole million-node arrays and miss. Recomputing the stale part is worth doing **only when the stale part is packed** - otherwise the gather/scatter hops so much that you may as well sweep. The Rust edition measures ~10x here; numpy narrows it to ~4.4x because every fancy-index gather carries a fixed overhead that is the same packed or scattered, diluting the locality difference - but locality still decides the winner.

## Exercise 7 - Break the tree (stretch)

Let node `h` be a child of both `b` and `g`. Now `h`'s world transform depends on two parents, and "everything beneath `b`" and "everything beneath `g`" overlap at `h`. There is no single contiguous slice that is `b`'s subtree, because `h` (and anything below it) is also reachable from `g` and need not sit adjacent to `b`'s other descendants. The pre-order trick - subtree equals a range - breaks, and so does the packed recompute: the stale set is now a *cone* through a graph, not a slice you can point at. Marking it requires walking edges, and recomputing it requires a correct order that is no longer "ascending index." That is exactly the next chapter: a spreadsheet, where one cell feeds many.
