# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy"]
# ///
"""
§26 exhibit - subscription tables, keyed by slot vs id, in Python+numpy.

This is the Python counterpart to the Rust edition's
`code/measurement/src/bin/ebp_partition.rs`. It does NOT measure the same
mechanism: the Rust win comes from a scalar `id_to_slot` cache miss inside a
tight loop. Python has no such loop. The Python question is about numpy fancy
indexing: a slot-keyed gather is one fancy index (energy[slots]); an id-keyed
gather is two (energy[id_to_slot[ids]]). The point of this script is to find
out whether the Rust chapter's conclusions survive that change of mechanism,
or whether the interpreter/numpy regime mutes or reverses them (as it did for
EBP sparsity in §19 and false sharing in §33).

Claims under test:
  C1 keying     slot-keyed gather vs id-keyed (double) gather, scattered slots.
  C2 relevance  subscription gather (K) vs vectorised scan-all (N), over sparsity.
  C3 locality   scattered slot gather vs compacted (contiguous) gather; payback.
  C4 reindex    slot reindex (once per interval) vs id redirection (every tick).

All "hot loop" timings gather the touched elements and reduce to a scalar, so
nothing is optimised away and no array is mutated between trials (which would
confound the timing). The reduction stands in for the per-element work a real
system does; the cost under study is the addressing, which is what differs.

    uv run code/measurement/ebp_partition.py
"""

import time
import numpy as np

N = 1_000_000          # population
TRIALS = 9             # odd -> clean median
RNG = np.random.default_rng(0)


def bench(fn, trials=TRIALS):
    """Median wall time of fn() over `trials`, after one warm-up."""
    fn()
    ts = []
    for _ in range(trials):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def make_world():
    energy = RNG.random(N, dtype=np.float64).astype(np.float32)
    return energy


def scattered_slots(k):
    """k distinct slots scattered through [0, N) - the churned, realistic case."""
    return RNG.choice(N, size=k, replace=False).astype(np.uint32)


# --------------------------------------------------------------------------
# C1 - keying: slot-keyed gather vs id-keyed (double) gather
# --------------------------------------------------------------------------
def claim_keying(energy, frac=0.10):
    k = int(N * frac)
    slots = scattered_slots(k)

    # id-keyed: the subscription holds ids; id_to_slot maps id -> slot.
    # Build a permutation so id != slot and the redirection actually moves.
    ids = RNG.permutation(N)[:k].astype(np.uint32)
    id_to_slot = np.empty(N, dtype=np.uint32)
    id_to_slot[ids] = slots  # each subscribed id resolves to one scattered slot

    slot_keyed = lambda: float(energy[slots].sum())
    id_keyed = lambda: float(energy[id_to_slot[ids]].sum())

    t_slot = bench(slot_keyed)
    t_id = bench(id_keyed)
    return t_slot, t_id, t_id / t_slot


# --------------------------------------------------------------------------
# C2 - relevance: subscription gather (K) vs vectorised scan-all (N)
# --------------------------------------------------------------------------
def claim_relevance(energy, frac):
    k = int(N * frac)
    slots = scattered_slots(k)
    mask = np.zeros(N, dtype=bool)
    mask[slots] = True

    subscription = lambda: float(energy[slots].sum())     # gather K
    scan_bool = lambda: float(energy[mask].sum())          # bool-index over N
    scan_mul = lambda: float((energy * mask).sum())        # branchless full-N

    return (bench(subscription), bench(scan_bool), bench(scan_mul))


# --------------------------------------------------------------------------
# C3 - locality: scattered slots vs compacted (contiguous) slots
# --------------------------------------------------------------------------
def claim_locality(energy, frac=0.10):
    k = int(N * frac)
    scattered = scattered_slots(k)
    compacted = np.arange(k, dtype=np.uint32)  # compaction-to-front: 0..k-1

    t_scatter = bench(lambda: float(energy[scattered].sum()))
    t_compact = bench(lambda: float(energy[compacted].sum()))

    # payback: one compaction reorders all live rows (a full-N gather per column);
    # model one column. Saving per tick is (scattered - compacted) on the hot loop.
    order = RNG.permutation(N).astype(np.int64)
    t_compaction = bench(lambda: energy[order])  # the reorder pass cost
    saving = t_scatter - t_compact
    payback_ticks = (t_compaction / saving) if saving > 0 else float("inf")
    return t_scatter, t_compact, t_scatter / t_compact, t_compaction, payback_ticks


# --------------------------------------------------------------------------
# C4 - reindex: slot reindex (once per G) vs id redirection (every tick)
# --------------------------------------------------------------------------
def claim_reindex(frac=0.10, S=2, G=30):
    k = int(N * frac)
    slots = scattered_slots(k)
    remap = RNG.permutation(N).astype(np.uint32)  # old-slot -> new-slot after compaction

    # slot key: rewrite S subscription tables once every G ticks
    reindex_one = lambda: remap[slots]
    t_reindex = bench(reindex_one)
    slot_cost_per_tick = (S * t_reindex) / G

    # id key: pay one extra gather (id_to_slot[ids]) every tick, per subscription
    ids = RNG.permutation(N)[:k].astype(np.uint32)
    id_to_slot = np.empty(N, dtype=np.uint32)
    id_to_slot[ids] = slots
    t_redirect = bench(lambda: id_to_slot[ids])
    id_cost_per_tick = S * t_redirect

    return slot_cost_per_tick, id_cost_per_tick, id_cost_per_tick / slot_cost_per_tick


# --------------------------------------------------------------------------
# C5 - lifecycle: per-tick bulk-filter compaction (A) vs mark-dead + GC-every-G (B)
#
# Both keep the table bounded and run the same hot loops and the same dense
# motion pass over N; those cancel. What differs is the lifecycle overhead:
#   A pays a full compaction + reindex EVERY tick (keeps the table hole-free).
#   B marks D dead per tick (unsubscribe, cheap) and compacts only once per G;
#     between GC passes dead slots sit in the columns, so a scan-all system
#     (motion) wastes work on the holes - with recycling the holes stay ~D
#     (refilled next tick), without it they accumulate to ~D*G/2.
# We measure the three numpy primitives and compute the per-tick overhead.
# --------------------------------------------------------------------------
def claim_lifecycle(deaths=1000, S=2, C=5, frac=0.10):
    INVALID = np.iinfo(np.uint32).max
    cols = [RNG.random(N).astype(np.float32) for _ in range(C)]
    ids = np.arange(N, dtype=np.uint32)
    id_to_slot = np.arange(N, dtype=np.uint32)
    subs = [scattered_slots(int(N * frac)) for _ in range(S)]
    dead = scattered_slots(deaths)
    keep = np.ones(N, dtype=bool); keep[dead] = False
    new_n = int(keep.sum())

    def compact_reindex():               # one full GC pass: compress C cols + reindex map + subs
        old_to_new = np.empty(N, dtype=np.uint32)
        old_to_new[np.flatnonzero(keep)] = np.arange(new_n, dtype=np.uint32)
        for c in cols:
            c[:new_n] = c[keep]
        id_to_slot[ids[keep]] = np.arange(new_n, dtype=np.uint32)
        for sub in subs:
            ns = old_to_new[sub]
            _ = ns[ns != INVALID]

    def mark_dead():                     # B's per-tick op: unsubscribe D from each subscription
        for sub in subs:
            _ = sub[~np.isin(sub, dead)]

    def motion_full():                   # scan-all system over N (common to both)
        cols[0][:] += cols[2] * np.float32(0.033)

    t_compact = bench(compact_reindex)
    t_mark = bench(mark_dead)
    t_motion_elem = bench(motion_full) / N

    return t_compact, t_mark, t_motion_elem, deaths


def main():
    print(f"N = {N:,}   trials = {TRIALS}   numpy {np.__version__}")
    energy = make_world()

    print("\nC1  keying (1M @ 10% subscribed, scattered)")
    t_slot, t_id, ratio = claim_keying(energy)
    print(f"    slot-keyed gather : {t_slot*1e6:8.1f} us")
    print(f"    id-keyed  gather  : {t_id*1e6:8.1f} us")
    print(f"    id / slot         : {ratio:6.2f}x  ({'slots win' if ratio>1.05 else 'comparable' if ratio<1.05 and ratio>0.95 else 'ids win'})")

    print("\nC2  relevance (subscription gather vs scan-all)")
    print(f"    {'frac':>6} {'gather(K)':>12} {'bool-scan(N)':>14} {'mul-scan(N)':>13} {'scan/gather':>12}")
    for frac in (0.01, 0.10, 1.00):
        t_sub, t_bool, t_mul = claim_relevance(energy, frac)
        best_scan = min(t_bool, t_mul)
        print(f"    {frac:6.2f} {t_sub*1e6:10.1f}us {t_bool*1e6:12.1f}us {t_mul*1e6:11.1f}us {best_scan/t_sub:11.2f}x")

    print("\nC3  locality (scattered vs compacted gather, 1M @ 10%)")
    t_sc, t_co, ratio, t_comp, payback = claim_locality(energy)
    print(f"    scattered gather  : {t_sc*1e6:8.1f} us")
    print(f"    compacted gather  : {t_co*1e6:8.1f} us")
    print(f"    scattered/compact : {ratio:6.2f}x")
    print(f"    compaction pass   : {t_comp*1e6:8.1f} us  -> payback {payback:.1f} ticks")

    print("\nC4  reindex amortized (S=2 subscriptions, G=30 tick interval)")
    slot_pt, id_pt, ratio = claim_reindex()
    print(f"    slot cost/tick    : {slot_pt*1e6:8.2f} us  (reindex once per {30} ticks)")
    print(f"    id   cost/tick    : {id_pt*1e6:8.2f} us  (redirection every tick)")
    print(f"    id / slot         : {ratio:6.2f}x  ({'slots win' if ratio>1.05 else 'comparable/ids win'})")

    print("\nC5  lifecycle: per-tick compaction (A) vs mark-dead + GC-every-G (B)")
    t_compact, t_mark, t_motion_elem, deaths = claim_lifecycle()
    print(f"    primitives: compact+reindex {t_compact*1e6:.0f}us  mark-dead {t_mark*1e6:.1f}us  motion {t_motion_elem*1e9:.3f}ns/elem  (deaths={deaths}/tick)")
    print(f"    per-tick lifecycle overhead (lower is better):")
    print(f"      {'G':>5} {'A':>11} {'B recycle':>12} {'B no-recyc':>12}  winner")
    a = t_compact
    for G in (1, 10, 30, 100, 300):
        b_rec = t_mark + t_compact / G + deaths * t_motion_elem
        b_nor = t_mark + t_compact / G + (deaths * G / 2) * t_motion_elem
        best_b = min(b_rec, b_nor)
        winner = "B (mark-dead)" if best_b < a else "A (per-tick)"
        print(f"      {G:>5} {a*1e6:>9.0f}us {b_rec*1e6:>10.0f}us {b_nor*1e6:>10.0f}us  {winner}")


if __name__ == "__main__":
    main()
