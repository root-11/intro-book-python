# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy"]
# ///
"""
Tests for the ecosystem simulator - and a worked example of §43, "tests are systems".

A test here *is* a system: it reads the world's tables and asserts a contract, the same
shape as `inspect` reading the tables to write a log, or a producer reading them to decide.
`check_invariants(world)` has a system's signature - read-set: every column; write-set:
none - and could be dropped straight into `step` as a debug system. That is §43: a test is
not a special kind of program, it is one more pass over the same tables.

These tests check the *machine*, never the *outcome*. The simulator evaluates initial
conditions; extinction, explosion, and limit cycles are all valid results, not failures.
Asserting "the population must stabilise" would be an opinion, not a test.

Run:
    uv run code/sim/tests.py     # run every test, print a summary, exit non-zero on failure
    pytest code/sim/tests.py     # the same tests under pytest (functions are test_*)
"""
import os
import tempfile

import numpy as np

import sim
from sim import Config, World, Log, INVALID, run, step, direct_herd, live_ids_of, replay_live_ids


# --- the invariant system (§43): read the tables, assert the contracts hold -----------
def check_invariants(w: World) -> None:
    """A system over the world. Read-set: every column. Write-set: none. Call it after any
    tick and it must pass, whatever the population is doing."""
    n_active, n_live = w.n_active, w.n_live
    live = w.live[:n_live]

    assert np.array_equal(np.sort(live), np.where(w.alive[:n_active])[0]), "live != alive set"
    assert n_live == int(w.alive[:n_active].sum()), "n_live disagrees with the alive count"

    live_ids = w.cid[live]
    assert np.unique(live_ids).size == n_live, "duplicate live ids"
    assert np.array_equal(w.id_to_slot[live_ids], live), "id_to_slot does not round-trip"
    assert int((w.id_to_slot != INVALID).sum()) == n_live, "stale id_to_slot entries"

    assert n_active <= w.cid.size and w.n_food <= w.fx.size, "capacity exceeded"
    assert np.isfinite(w.energy[live]).all(), "non-finite energy in a live creature"


# --- tests ---------------------------------------------------------------------------
def test_invariants_hold_every_tick():
    """The strongest test: the contracts hold after *every* tick, not only at the end."""
    cfg = Config(ticks=120)
    rng = np.random.default_rng(cfg.seed)
    w, log = World(cfg, rng), Log()
    check_invariants(w)
    for _ in range(cfg.ticks):
        step(w, log, cfg, rng)
        check_invariants(w)


def test_determinism():
    """§16: same seed, same system order, same run."""
    cfg = Config(ticks=120)
    assert run(cfg)[1].population == run(cfg)[1].population


def test_replay_reconstructs_the_world():
    """§37: replaying births minus deaths reconstructs the live population exactly."""
    w, log = run(Config(ticks=200))
    assert np.array_equal(live_ids_of(w), replay_live_ids(log.born, log.dead))


def test_save_load_roundtrip():
    """§36: a saved world reloads bit-for-bit, and a loaded world is still a valid world."""
    w, _ = run(Config(ticks=120))
    path = os.path.join(tempfile.mkdtemp(), "world.npz")
    sim.save_world(w, path)
    w2 = sim.load_world(path)
    assert np.array_equal(live_ids_of(w), live_ids_of(w2))
    assert np.array_equal(w.energy[: w.n_active], w2.energy[: w2.n_active])
    check_invariants(w2)


def test_event_time_is_sub_tick():
    """§12: a death is stamped with its exact sub-tick time. A lone creature with energy E
    and burn rate B starves at t = E / B seconds into the tick, not at the boundary."""
    cfg = Config(ticks=1, n0_creatures=1, food_per_second=0.0, burn_rate=4.0, init_energy=0.1)
    rng = np.random.default_rng(cfg.seed)
    w, log = World(cfg, rng), Log()
    step(w, log, cfg, rng)
    assert len(log.dead) == 1, "the lone creature should have starved"
    death_t, _cid = log.dead[0]
    assert abs(death_t - 0.025) < 1e-6, f"death logged at {death_t}, not the sub-tick 0.025 s"


def test_gc_reclaims():
    """§22/§24: the deferred GC runs and has holes to reclaim."""
    w, _ = run(Config(ticks=200))
    assert w.gc_runs > 0, "the GC never ran"
    assert w.peak_n_active >= w.n_live


def test_run_is_evaluable():
    """The simulator evaluates initial conditions; the machine must hold for whatever
    trajectory results. We assert consistency and completion, NOT a stable population."""
    w, log = run(Config(ticks=400))
    check_invariants(w)
    assert len(log.population) > 0


def test_herd_split_mechanism():
    """§28: a herd larger than max_herd splits. Tested on the mechanism directly, not via a
    population outcome - construct an over-large herd, run direct_herd, see it split."""
    cfg = Config(n0_creatures=450)   # > max_herd (400), all in herd 0
    rng = np.random.default_rng(cfg.seed)
    w = World(cfg, rng)
    assert np.unique(w.herd[w.live[: w.n_live]]).size == 1
    direct_herd(w, cfg, rng)
    assert np.unique(w.herd[w.live[: w.n_live]]).size > 1, "the over-large herd did not split"


def test_no_creature_class():
    """The SoA promise: there is no Creature (or Food) object anywhere in the simulator."""
    assert not hasattr(sim, "Creature"), "a Creature class crept in; the sim must stay SoA"
    assert not hasattr(sim, "Food"), "a Food class crept in; the sim must stay SoA"


# --- standalone runner: no pytest import at module scope, so the file imports cleanly --
def _main() -> None:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failures = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as exc:
            failures += 1
            print(f"  FAIL  {t.__name__}: {exc}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    _main()
