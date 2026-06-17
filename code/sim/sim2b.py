# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy"]
# ///
"""
Through-line ecosystem simulator - UNIFIED ENTITY + PREDATOR (sim2b; diff against sim1b).
================================================================================

This is sim1b with a predator added. `diff sim1b.py sim2b.py` is the whole cost, and the
point is what it does NOT contain: no new patch type, no change to `apply`, no change to
`forage`. A predator is a beast subscribed to the hunter herd and a hunt edge - one more
name in _SUBSCRIPTIONS, a seed, and two lines in `step`. The join maintains the new
subscription by the same loop it already runs, and `forage` eats a grazer with the same
code it eats grass, because both are entities.

Everything that exists is an entity in one table: grass and grazers are the same kind
of row, with the same columns and the same lifecycle (spawn, grow past a threshold and
fission, die when fuel runs out). A *species* is nothing but a set of subscriptions:

    grass   = { regenerate-motion , forage -> (carrion, not modelled yet) }
    grazer  = { herd-motion       , forage -> grass }

The only thing that distinguishes a grass blade from a grazer is which motion system and
which forage edge hold its id. There is no `is_grass` flag and no Grass class. Parameters
live on the *systems*, not the entities: you burn fuel at your motion system's rate and
catch food at your forage edge's radius, so an entity inherits its whole character by
subscribing. `sim2b.py` adds a predator, and the diff against this file is the lesson:
a new trophic level is a subscription and a forage edge, not surgery.

`forage(foragers, targets, radius, gain)` is ONE system. Eating grass and (in sim2b)
eating a grazer are the same write - the target entity dies and the forager gains its
energy - because grass is an entity, not a cheap side table. `apply` consumes a *list*
of forage patches identically and never learns what is eating what.

Run: uv run code/sim/sim1b.py [--check|--save PATH|--log DIR|--plot]
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field

import numpy as np

INVALID = np.iinfo(np.uint32).max


@dataclass
class Config:
    world: float = 100.0
    seed: int = 1
    ticks: int = 600
    dt: float = 1.0 / 30.0
    cap: int = 200_000              # one table for every entity
    gc_interval: int = 30
    repro_threshold: float = 24.0   # uniform: any entity fissions past this energy
    init_energy: float = 10.0

    n0_grass: int = 1500
    n0_grazers: int = 300
    n0_predators: int = 20          # founders subscribed to the hunter herd + the hunt edge

    # hunter motion + hunt edge (predators): herd like grazers, but burn faster and eat them
    hunter_speed: float = 3.5      # slightly slower than the herd (6.0) so grazers can escape
    hunter_burn: float = 8.0
    hunt_radius: float = 3.0
    hunt_gain: float = 20.0

    # regenerate motion (grass): drift slowly, photosynthesise, barely burn
    grass_photosynthesis: float = 6.0   # energy / second from sunlight
    grass_drift: float = 0.8
    grass_burn: float = 0.0

    # herd motion (grazers): cohesion toward the eldest, wander, real burn
    herd_speed: float = 6.0
    herd_burn: float = 4.0
    cohesion: float = 0.15
    wander: float = 0.3
    max_herd: int = 400

    # graze forage edge (grazers eat grass)
    graze_radius: float = 2.0
    graze_gain: float = 8.0


# ------------------------------------------------------------------------------
# World: one entity table + the subscriptions. Committed state only.
# ------------------------------------------------------------------------------
class World:
    _COLS = ("cid", "gen", "px", "py", "vx", "vy", "intent_x", "intent_y",
             "energy", "birth_t", "alive", "herd")
    # the species registry: every subscription apply maintains. Adding a species means
    # adding a name here (and seeding it); apply's maintenance loop knows nothing else.
    _SUBSCRIPTIONS = ("grass", "grazers", "predators")   # +predators vs sim1b

    def __init__(self, cfg: Config, rng: np.random.Generator):
        cap = cfg.cap
        self.cid = np.zeros(cap, np.uint32)
        self.gen = np.zeros(cap, np.uint32)
        self.px = np.zeros(cap, np.float32)
        self.py = np.zeros(cap, np.float32)
        self.vx = np.zeros(cap, np.float32)
        self.vy = np.zeros(cap, np.float32)
        self.intent_x = np.zeros(cap, np.float32)
        self.intent_y = np.zeros(cap, np.float32)
        self.energy = np.zeros(cap, np.float32)
        self.birth_t = np.zeros(cap, np.float64)
        self.alive = np.zeros(cap, bool)
        self.herd = np.zeros(cap, np.uint32)
        self.id_to_slot = np.full(cap, INVALID, np.uint32)
        self.live = np.zeros(cap, np.uint32)
        self.n_live = 0
        self.n_active = 0
        self.next_id = 0
        self.next_herd_id = 1
        self.d_energy = np.zeros(cap, np.float32)   # the tick's energy delta buffer (§15)
        # subscriptions (§17/§19): which system each entity belongs to, as id-sets - not flags
        self.grass = np.zeros(0, np.uint32)
        self.grazers = np.zeros(0, np.uint32)
        self.predators = np.zeros(0, np.uint32)     # +predators subscription vs sim1b
        self.tick = 0
        self.t = 0.0
        self.gc_runs = 0
        self.peak_n_active = 0
        self.late_ticks = 0
        self.peak_tick_s = 0.0

        ng, nz, npr = cfg.n0_grass, cfg.n0_grazers, cfg.n0_predators
        n = ng + nz + npr
        self.px[:n] = rng.uniform(0, cfg.world, n)
        self.py[:n] = rng.uniform(0, cfg.world, n)
        ang = rng.uniform(0, 2 * np.pi, n)
        self.vx[:n] = np.cos(ang) * cfg.herd_speed
        self.vy[:n] = np.sin(ang) * cfg.herd_speed
        self.intent_x[:n] = self.vx[:n]
        self.intent_y[:n] = self.vy[:n]
        self.energy[:n] = cfg.init_energy
        self.cid[:n] = np.arange(n, dtype=np.uint32)
        self.alive[:n] = True
        self.id_to_slot[:n] = np.arange(n, dtype=np.uint32)
        self.live[:n] = np.arange(n, dtype=np.uint32)
        self.n_live = self.n_active = self.next_id = n
        self.grass = self.cid[:ng].copy()              # first ng ids subscribe to grass
        self.grazers = self.cid[ng:ng + nz].copy()     # next nz subscribe to the grazing herd
        self.predators = self.cid[ng + nz:n].copy()    # last npr subscribe to the hunt


@dataclass
class Log:
    born: list = field(default_factory=list)
    dead: list = field(default_factory=list)
    eaten: list = field(default_factory=list)
    population: list = field(default_factory=list)   # (t, n_grass, n_grazers)


@dataclass
class ForagePatch:
    forager_slots: np.ndarray   # who fed
    target_ids: np.ndarray      # the entity each one consumed (dies)
    gains: np.ndarray           # energy gained

@dataclass
class BornPatch:
    parent_ids: np.ndarray
    px: np.ndarray; py: np.ndarray; vx: np.ndarray; vy: np.ndarray
    energy: np.ndarray; herd: np.ndarray

@dataclass
class DeadPatch:
    ids: np.ndarray
    times: np.ndarray


# ==============================================================================
# THE TICK. The DAG. Read it first.
# ==============================================================================

def step(w: World, log: Log, cfg: Config, rng: np.random.Generator) -> None:
    w.tick += 1
    w.d_energy[: w.n_active] = 0.0                # reset the per-tick energy delta (§15)

    # -- motion, dispatched by subscription (each system burns/feeds its own fuel) --
    regenerate(w, cfg, rng)                       # grass: drift + photosynthesise
    herd_move(w, cfg, rng, w.grazers, cfg.herd_speed, cfg.herd_burn)        # grazers: herd + burn
    herd_move(w, cfg, rng, w.predators, cfg.hunter_speed, cfg.hunter_burn)  # +predators: herd + burn

    # -- the trophic edges: a list of forage patches, order irrelevant --
    foraged = [forage(w, cfg, w.grazers, w.grass, cfg.graze_radius, cfg.graze_gain),
               forage(w, cfg, w.predators, w.grazers, cfg.hunt_radius, cfg.hunt_gain)]  # +hunt edge

    # -- lifecycle, uniform for every entity --
    born = reproduce(w, cfg, rng)
    dead = die(w, cfg)

    # -- the join: the single writer of committed state consumes every patch --
    apply(w, log, cfg, foraged, born, dead)

    cleanup(w, cfg)
    inspect(w, log)
    w.t += cfg.dt


# ==============================================================================
# MOTION systems - dispatched by subscription. Each writes pos/vel (sole) for its
# subscribers and adds its own fuel cost/gain to the energy delta buffer.
# ==============================================================================

def regenerate(w: World, cfg: Config, rng: np.random.Generator) -> None:
    """Grass 'motion': a slow random drift plus photosynthesis. Read/write: grass slots."""
    s = _slots(w, w.grass)
    if s.size == 0:
        return
    ang = rng.uniform(0, 2 * np.pi, s.size)
    w.vx[s] = np.cos(ang) * cfg.grass_drift
    w.vy[s] = np.sin(ang) * cfg.grass_drift
    w.px[s] = np.mod(w.px[s] + w.vx[s] * cfg.dt, cfg.world)
    w.py[s] = np.mod(w.py[s] + w.vy[s] * cfg.dt, cfg.world)
    w.d_energy[s] += (cfg.grass_photosynthesis - cfg.grass_burn) * cfg.dt


def herd_move(w: World, cfg: Config, rng: np.random.Generator,
              members_ids: np.ndarray, speed: float, burn: float) -> None:
    """Herd motion for ANY subscription: steer toward the herd's eldest leader, wander,
    integrate, burn fuel at this subscription's rate. Sole writer of pos/vel/intent for its
    members. Parameterised by (subscription, speed, burn) so grazers and predators share it."""
    s = _slots(w, members_ids)
    if s.size == 0:
        return
    h = w.herd[s]
    order = np.lexsort((w.birth_t[s], h))
    sh = h[order]
    first = np.concatenate(([True], sh[1:] != sh[:-1]))
    lead = s[order][first]
    lpx = np.zeros(w.next_herd_id, np.float32); lpx[sh[first]] = w.px[lead]
    lpy = np.zeros(w.next_herd_id, np.float32); lpy[sh[first]] = w.py[lead]
    half = cfg.world * 0.5
    dx = (lpx[h] - w.px[s] + half) % cfg.world - half
    dy = (lpy[h] - w.py[s] + half) % cfg.world - half
    d = np.sqrt(dx * dx + dy * dy) + 1e-6
    ix = w.vx[s] + cfg.cohesion * (dx / d) * speed
    iy = w.vy[s] + cfg.cohesion * (dy / d) * speed
    turn = rng.uniform(-cfg.wander, cfg.wander, s.size).astype(np.float32)
    c, sn = np.cos(turn), np.sin(turn)
    ix, iy = c * ix - sn * iy, sn * ix + c * iy
    sp = np.sqrt(ix * ix + iy * iy) + 1e-6
    w.vx[s] = ix / sp * speed
    w.vy[s] = iy / sp * speed
    w.px[s] = np.mod(w.px[s] + w.vx[s] * cfg.dt, cfg.world)
    w.py[s] = np.mod(w.py[s] + w.vy[s] * cfg.dt, cfg.world)
    w.d_energy[s] -= burn * cfg.dt
    # split a herd that has outgrown max_herd
    sizes = np.bincount(h, minlength=w.next_herd_id)
    for hid in np.nonzero(sizes > cfg.max_herd)[0]:
        members = s[h == hid]
        movers = members[w.px[members] > np.median(w.px[members])]
        w.herd[movers] = w.next_herd_id
        w.next_herd_id += 1


# ==============================================================================
# FORAGE - one general trophic system. foragers eat the nearest target in range;
# the target (an entity) dies, the forager gains its energy. Producer: returns a patch.
# ==============================================================================

def forage(w: World, cfg: Config, foragers_ids: np.ndarray, targets_ids: np.ndarray,
           radius: float, gain: float) -> ForagePatch:
    fo = _slots(w, foragers_ids)
    ta = _slots(w, targets_ids)
    if fo.size == 0 or ta.size == 0:
        return ForagePatch(_i64(), _u32(), _f32())
    # each target goes to its nearest forager in range (the §28 neighbour query)
    dx = w.px[ta][:, None] - w.px[fo][None, :]
    dy = w.py[ta][:, None] - w.py[fo][None, :]
    d2 = dx * dx + dy * dy
    nearest = d2.argmin(axis=1)
    within = d2[np.arange(ta.size), nearest] <= radius ** 2
    target_slots = ta[within]
    forager_slots = fo[nearest[within]]
    return ForagePatch(forager_slots, w.cid[target_slots],
                       np.full(forager_slots.size, gain, np.float32))


# ==============================================================================
# LIFECYCLE - uniform for every entity, whatever it subscribes to.
# ==============================================================================

def reproduce(w: World, cfg: Config, rng: np.random.Generator) -> BornPatch:
    """Any entity at or above the threshold fissions into two; children carry half the
    energy and inherit the parent's subscription (apply wires that). Producer."""
    live = w.live[: w.n_live]
    fat = live[w.energy[live] >= cfg.repro_threshold] if live.size else _i64()
    if fat.size == 0:
        e = _f32()
        return BornPatch(_u32(), e, e, e, e, e, _u32())
    half = (w.energy[fat] * 0.5).astype(np.float32)
    k = fat.size
    ang = rng.uniform(0, 2 * np.pi, 2 * k)
    return BornPatch(
        parent_ids=w.cid[fat],
        px=(np.repeat(w.px[fat], 2) + rng.uniform(-1, 1, 2 * k)).astype(np.float32),
        py=(np.repeat(w.py[fat], 2) + rng.uniform(-1, 1, 2 * k)).astype(np.float32),
        vx=(np.cos(ang) * cfg.herd_speed).astype(np.float32),
        vy=(np.sin(ang) * cfg.herd_speed).astype(np.float32),
        energy=np.repeat(half, 2),
        herd=np.repeat(w.herd[fat], 2))


def die(w: World, cfg: Config) -> DeadPatch:
    """Any entity whose fuel will run out this tick dies, at its sub-tick time (§12).
    Producer. (Energy delta is this tick's net gain/burn from the motion systems.)"""
    live = w.live[: w.n_live]
    if live.size == 0:
        return DeadPatch(_u32(), _f64())
    # net energy change this tick; a creature dies when current + delta crosses zero
    rate = -w.d_energy[live] / cfg.dt                       # burn rate (>0 if losing fuel)
    with np.errstate(divide="ignore", invalid="ignore"):
        t_die = np.where(rate > 0, w.energy[live] / rate, np.inf)
    dying = t_die <= cfg.dt
    return DeadPatch(w.cid[live[dying]], np.clip(t_die[dying], 0.0, cfg.dt).astype(np.float64))


# ==============================================================================
# THE JOIN - the single writer of committed structural state and energy.
# ==============================================================================

def apply(w: World, log: Log, cfg: Config,
          foraged: list, born: BornPatch, dead: DeadPatch) -> None:
    n = w.n_active

    # resolve who reproduces / who dies; the eaten always die regardless
    repro_slots = _slots(w, born.parent_ids)
    is_repro = np.zeros(n, bool); is_repro[repro_slots] = True
    eaten_ids = np.concatenate([p.target_ids for p in foraged]) if foraged else _u32()
    eaten_slots = _slots(w, eaten_ids)
    is_eaten = np.zeros(n, bool); is_eaten[eaten_slots] = True
    # foragers that fed this tick gained energy `die` could not see (it only saw the burn),
    # so feeding saves a forager from the starve check - eat-saves-starve, resolved here
    fed = np.concatenate([p.forager_slots for p in foraged]).astype(np.int64) if foraged else _i64()
    is_fed = np.zeros(n, bool); is_fed[fed] = True

    dead_slots = _slots(w, dead.ids)
    survived = is_repro[dead_slots] | is_eaten[dead_slots] | is_fed[dead_slots]
    dead_ids, dead_times = dead.ids[~survived], dead.times[~survived]

    # 1) energy: commit this tick's net motion delta, then the forage gains. Sole writer.
    w.energy[: n] += w.d_energy[: n]
    for p in foraged:
        np.add.at(w.energy, p.forager_slots, p.gains)

    # 2) remove: dead + fissioned parents + everything eaten (entity death is uniform)
    gone = np.unique(np.concatenate([dead_ids, born.parent_ids, eaten_ids]).astype(np.uint32)) \
        if (dead_ids.size or born.parent_ids.size or eaten_ids.size) else _u32()
    if gone.size:
        slots = w.id_to_slot[gone]; slots = slots[slots != INVALID]
        w.alive[slots] = False
        w.gen[slots] += 1
        w.id_to_slot[gone] = INVALID

    # 3) insert offspring at the tail, minting ids; children inherit the subscription
    ids = _u32(); parent_of_child = _u32()
    nb = born.energy.size
    if nb:
        if w.n_active + nb > cfg.cap:
            _compact(w)
        s = slice(w.n_active, w.n_active + nb)
        ids = np.arange(w.next_id, w.next_id + nb, dtype=np.uint32); w.next_id += nb
        w.cid[s] = ids
        w.px[s], w.py[s], w.vx[s], w.vy[s] = born.px, born.py, born.vx, born.vy
        w.intent_x[s], w.intent_y[s] = born.vx, born.vy
        w.energy[s] = born.energy
        w.herd[s] = born.herd
        w.birth_t[s] = w.t
        w.alive[s] = True
        w.id_to_slot[ids] = np.arange(w.n_active, w.n_active + nb, dtype=np.uint32)
        w.n_active += nb
        parent_of_child = np.repeat(born.parent_ids, 2)
        for cid, pid in zip(ids.tolist(), parent_of_child.tolist()):
            log.born.append((w.t, cid, pid))

    # 4) maintain EVERY subscription the same way (§26): drop the gone, then add the
    #    offspring of this subscription's own members. Nothing here knows what grass or a
    #    grazer is - adding a species is adding a name to World._SUBSCRIPTIONS, not editing
    #    this loop.
    for name in World._SUBSCRIPTIONS:
        sub = getattr(w, name)
        parents = np.intersect1d(born.parent_ids, sub)   # this sub's fissioning members
        if gone.size:
            sub = np.setdiff1d(sub, gone)
        if ids.size:
            new = ids[np.isin(parent_of_child, parents)]
            if new.size:
                sub = np.concatenate([sub, new]).astype(np.uint32)
        setattr(w, name, sub)

    # 5) history: apply commits, so apply logs (§37)
    for cid, ts in zip(dead_ids.tolist(), dead_times.tolist()):
        log.dead.append((w.t + ts, int(cid)))
    for pid in born.parent_ids.tolist():
        log.dead.append((w.t, int(pid)))
    for p in foraged:
        for cid in p.target_ids.tolist():
            log.dead.append((w.t, int(cid)))               # the eaten are removed too
        for cid, val in zip(w.cid[p.forager_slots].tolist(), p.gains.tolist()):
            log.eaten.append((w.t, cid, val))

    _rebuild_live(w)


def cleanup(w: World, cfg: Config) -> None:
    w.peak_n_active = max(w.peak_n_active, w.n_active)
    if w.tick % cfg.gc_interval == 0:
        _compact(w)
        _rebuild_live(w)


def inspect(w: World, log: Log) -> None:
    live = w.live[: w.n_live]
    n_grass = int(np.intersect1d(w.grass, w.cid[live]).size)
    log.population.append((w.t, n_grass, w.n_live - n_grass))


# ------------------------------------------------------------------------------
def _i64(): return np.empty(0, np.int64)
def _u32(): return np.empty(0, np.uint32)
def _f32(): return np.empty(0, np.float32)
def _f64(): return np.empty(0, np.float64)


def _slots(w: World, ids: np.ndarray) -> np.ndarray:
    """Resolve a subscription's ids to current slots (§23), dropping any that are gone."""
    if ids.size == 0:
        return _i64()
    s = w.id_to_slot[ids]
    return s[s != INVALID].astype(np.int64)


def _rebuild_live(w: World) -> None:
    live = np.where(w.alive[: w.n_active])[0]
    w.n_live = int(live.size)
    w.live[: w.n_live] = live


def _compact(w: World) -> None:
    keep = w.alive[: w.n_active].copy()
    m = int(keep.sum())
    if m == w.n_active:
        return
    w.gc_runs += 1
    for name in World._COLS:
        col = getattr(w, name)
        col[:m] = col[: w.n_active][keep]
    w.alive[m: w.n_active] = False
    w.id_to_slot[w.cid[:m]] = np.arange(m, dtype=np.uint32)
    w.n_active = m


# ==============================================================================
# Run loop + plumbing (replay §37, persistence §36, logger, output)
# ==============================================================================

def run(cfg: Config, realtime: bool = False, strict: bool = False) -> tuple[World, Log]:
    rng = np.random.default_rng(cfg.seed)
    w = World(cfg, rng)
    log = Log()
    for cid in range(cfg.n0_grass + cfg.n0_grazers + cfg.n0_predators):
        log.born.append((0.0, cid, -1))
    for _ in range(cfg.ticks):
        start = time.perf_counter()
        step(w, log, cfg, rng)
        elapsed = time.perf_counter() - start
        w.peak_tick_s = max(w.peak_tick_s, elapsed)
        if elapsed > cfg.dt:
            # the budget is exhausted (§4/§11): a part of "living with it". The loop does not
            # silently fall behind - it raises in strict mode (use it in tests/CI), or warns
            # once and keeps counting in `late_ticks`. A benchmark should have caught this
            # first; this is the backstop for the case it did not.
            w.late_ticks += 1
            msg = (f"tick {w.tick}: {elapsed * 1000:.1f} ms over the {cfg.dt * 1000:.1f} ms "
                   f"budget - framerate would drop")
            if strict:
                raise RuntimeError(msg)
            if w.late_ticks == 1:
                print(f"WARNING: {msg} (further overruns counted in late_ticks)")
        if realtime and elapsed < cfg.dt:
            time.sleep(cfg.dt - elapsed)
        if w.n_live == 0:
            break
    return w, log


def live_ids_of(w: World) -> np.ndarray:
    return np.sort(w.cid[: w.n_active][w.alive[: w.n_active]])


def replay_live_ids(born: list, dead: list) -> np.ndarray:
    b = {cid for (_t, cid, _p) in born}
    d = {cid for (_t, cid) in dead}
    return np.array(sorted(b - d), np.uint32)


def save_world(w: World, path: str) -> None:
    arrays, scalars = {}, {}
    for name, v in vars(w).items():
        if isinstance(v, np.ndarray):
            arrays[name] = v
        elif isinstance(v, np.generic):
            scalars[name] = v.item()
        elif isinstance(v, (bool, int, float)):
            scalars[name] = v
    np.savez(path, _meta=np.frombuffer(json.dumps(scalars).encode(), np.uint8), **arrays)


def load_world(path: str) -> World:
    d = np.load(path)
    w = object.__new__(World)
    for k in d.files:
        if k != "_meta":
            setattr(w, k, d[k])
    for k, v in json.loads(d["_meta"].tobytes().decode()).items():
        setattr(w, k, v)
    return w


def summarise(w: World, log: Log, cfg: Config) -> None:
    grass = np.array([p[1] for p in log.population], np.int64)
    graz = np.array([p[2] for p in log.population], np.int64)
    print(f"ticks run        : {len(log.population)}")
    print(f"grass  min/max/fin: {grass.min()} / {grass.max()} / {grass[-1]}")
    print(f"grazer min/max/fin: {graz.min()} / {graz.max()} / {graz[-1]}")
    print(f"peak tick work   : {w.peak_tick_s * 1000:.2f} ms  (budget {cfg.dt * 1000:.1f} ms)")
    print(f"ticks over budget: {w.late_ticks}  (the framerate dropped on these; §4/§11)")
    print(f"GC compactions   : {w.gc_runs}")
    ok = np.array_equal(live_ids_of(w), replay_live_ids(log.born, log.dead))
    print(f"replay (§37)     : {'OK' if ok else 'MISMATCH'} - "
          f"{len(log.born)} births / {len(log.dead)} deaths / {len(log.eaten)} meals")
    tot = grass + graz
    if tot.max() > tot.min():
        bars = " .:-=+*#%@"
        idx = ((tot - tot.min()) / (tot.max() - tot.min()) * (len(bars) - 1)).astype(int)
        print("total population over time:")
        print("".join(bars[i] for i in idx[:: max(1, len(idx) // 70)]))


def maybe_plot(log: Log) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib not installed; skipping --plot)")
        return
    t = [p[0] for p in log.population]
    plt.plot(t, [p[1] for p in log.population], label="grass")
    plt.plot(t, [p[2] for p in log.population], label="grazers")
    plt.xlabel("time (s)"); plt.ylabel("count"); plt.legend(); plt.title("Ecosystem")
    plt.savefig("population.png", dpi=120); print("wrote population.png")


def main() -> None:
    ap = argparse.ArgumentParser(description="Unified-entity ecosystem simulator")
    ap.add_argument("--ticks", type=int, default=Config.ticks)
    ap.add_argument("--seed", type=int, default=Config.seed)
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--save", metavar="PATH")
    ap.add_argument("--strict", action="store_true",
                    help="raise the moment a tick blows the budget (for tests/CI), §4/§11")
    args = ap.parse_args()
    cfg = Config(ticks=args.ticks, seed=args.seed)

    if args.check:
        wa, la = run(cfg)
        _, lb = run(cfg)
        assert la.population == lb.population, "non-deterministic"
        assert np.array_equal(live_ids_of(wa), replay_live_ids(la.born, la.dead)), "replay mismatch"
        print(f"determinism OK : {len(la.population)} ticks identical (§16)")
        print( "replay OK      : the log reconstructs the live population bit-for-bit (§37)")
        return

    w, log = run(cfg, strict=args.strict)
    summarise(w, log, cfg)
    if args.save:
        path = args.save if args.save.endswith(".npz") else args.save + ".npz"
        save_world(w, path); w2 = load_world(path)
        assert np.array_equal(live_ids_of(w), live_ids_of(w2))
        print(f"save/load: OK - {path} reloaded bit-for-bit (§36)")
    if args.plot:
        maybe_plot(log)


if __name__ == "__main__":
    main()
