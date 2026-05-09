"""
Benchmark and correctness test for SimLog.

Tests:
1. Correctness: write N rows, read them back, verify values.
2. Throughput: measure log() overhead per call.
3. CSV export: round-trip through to_csv / DictReader.
"""

import time
import shutil
import csv
import numpy as np
from pathlib import Path
from simlog import SimLog, SimLogCodebook, SimLogSparse, SimLogSparseNP, SimLogText

LOG_DIR = '/tmp/simlog_test'
FIELDS = {
    'time':             'f8',
    'value':            'f8',
    'activity':         'i4',
    'entity_type':      'i4',
    'entity_id':        'i4',
    'mission_id':       'i4',
    'lp':               'i4',
    'task_id':          'i4',
    'priority':         'i4',
    'priority_group':   'i4',
    'derived_priority': 'i4',
}
FIELD_NAMES = list(FIELDS.keys())
N_FIELDS = len(FIELDS)
BUFFER_SIZE = 200_000
N_RECORDS   = 2_000_000

# string values used in the real data for the two text fields
ACTIVITIES   = ['picking', 'putaway', 'replen', 'consolidation', 'count', 'transport', 'staging']
ENTITY_TYPES = ['bot', 'human', 'conveyor', 'station']


def make_row(i) -> dict:
    """Generate one deterministic test row as kwargs dict (all numeric)."""
    return dict(
        time=i * 0.001,
        value=i * 1.23,
        activity=i % 7,
        entity_type=i % 4,
        entity_id=i % 10_000,
        mission_id=i % 500,
        lp=i % 3,
        task_id=i % 200,
        priority=i % 5,
        priority_group=i % 3,
        derived_priority=(i % 5) * (i % 3),
    )


def make_row_str(i) -> dict:
    """Generate one deterministic test row with string activity/entity_type."""
    return dict(
        time=i * 0.001,
        value=i * 1.23,
        activity=ACTIVITIES[i % len(ACTIVITIES)],
        entity_type=ENTITY_TYPES[i % len(ENTITY_TYPES)],
        entity_id=i % 10_000,
        mission_id=i % 500,
        lp=i % 3,
        task_id=i % 200,
        priority=i % 5,
        priority_group=i % 3,
        derived_priority=(i % 5) * (i % 3),
    )


def test_correctness():
    print(f"{'='*60}")
    print(f"Correctness test: {N_RECORDS:,} records, buffer={BUFFER_SIZE:,}")
    print(f"{'='*60}")

    shutil.rmtree(LOG_DIR, ignore_errors=True)

    # write
    with SimLog(LOG_DIR, fields=FIELDS, mode='w', buffer_size=BUFFER_SIZE) as logger:
        for i in range(N_RECORDS):
            logger.log(**make_row(i))

    # read back as arrays
    reader = SimLog(LOG_DIR, mode='r')
    arrays, masks = reader.to_arrays()

    assert len(arrays) == N_FIELDS, f"field count mismatch: {len(arrays)}"
    for name, arr in arrays.items():
        assert len(arr) == N_RECORDS, f"{name}: expected {N_RECORDS}, got {len(arr)}"
        assert masks[name].all(), f"{name}: all masks should be True for full rows"

    # spot-check first and last rows
    for idx in [0, 1, N_RECORDS // 2, N_RECORDS - 1]:
        expected = make_row(idx)
        for name in FIELD_NAMES:
            got = arrays[name][idx]
            exp = expected[name]
            assert np.isclose(got, exp), \
                f"row {idx}, field {name}: expected {exp}, got {got}"

    print("  [PASS] Array round-trip correct.\n")


def test_throughput():
    print(f"{'='*60}")
    print(f"Throughput test: {N_RECORDS:,} records, buffer={BUFFER_SIZE:,}")
    print(f"{'='*60}")

    shutil.rmtree(LOG_DIR, ignore_errors=True)

    # warm-up
    with SimLog(LOG_DIR, fields=FIELDS, mode='w', buffer_size=BUFFER_SIZE) as logger:
        for i in range(BUFFER_SIZE):
            logger.log(**make_row(i))

    shutil.rmtree(LOG_DIR, ignore_errors=True)

    # timed run
    row = make_row(42)  # fixed row to remove make_row cost
    t0 = time.perf_counter()
    with SimLog(LOG_DIR, fields=FIELDS, mode='w', buffer_size=BUFFER_SIZE) as logger:
        _log = logger.log  # local ref
        for _ in range(N_RECORDS):
            _log(**row)
    t1 = time.perf_counter()

    elapsed = t1 - t0
    rate = N_RECORDS / elapsed
    ns_per_call = elapsed / N_RECORDS * 1e9

    print(f"  Records:      {N_RECORDS:>12,}")
    print(f"  Elapsed:      {elapsed:>12.3f} s")
    print(f"  Throughput:   {rate:>12,.0f} records/s")
    print(f"  Per call:     {ns_per_call:>12.0f} ns")

    # baseline: append cost without SimLog
    plain_list = []
    t2 = time.perf_counter()
    for _ in range(N_RECORDS):
        plain_list.append(row)
    t3 = time.perf_counter()
    baseline_ns = (t3 - t2) / N_RECORDS * 1e9

    print(f"\n  Baseline (list.append): {baseline_ns:.0f} ns/call")
    print(f"  Overhead vs baseline:   {ns_per_call - baseline_ns:.0f} ns/call")
    print(f"  Overhead ratio:         {ns_per_call / baseline_ns:.1f}x\n")


def test_read_iter():
    print(f"{'='*60}")
    print(f"DictReader-style iteration test")
    print(f"{'='*60}")

    reader = SimLog(LOG_DIR, mode='r')
    count = 0
    t0 = time.perf_counter()
    for row in reader:
        count += 1
        if count == 1:
            print(f"  First row: { {k: row[k] for k in list(row)[:4]} } ...")
    t1 = time.perf_counter()

    print(f"  Iterated {count:,} rows in {t1 - t0:.3f} s")
    print(f"  [PASS] Iterator works.\n")


def test_csv_export():
    print(f"{'='*60}")
    print(f"CSV export test (first 10,000 rows)")
    print(f"{'='*60}")

    small_dir = LOG_DIR + '_csv'
    shutil.rmtree(small_dir, ignore_errors=True)
    n_csv = 10_000

    with SimLog(small_dir, fields=FIELDS, mode='w', buffer_size=BUFFER_SIZE) as logger:
        for i in range(n_csv):
            logger.log(**make_row(i))

    csv_path = '/tmp/simlog_test.csv'
    reader = SimLog(small_dir, mode='r')
    reader.to_csv(csv_path)

    # verify via csv.DictReader
    with open(csv_path) as f:
        dr = csv.DictReader(f)
        rows = list(dr)

    assert len(rows) == n_csv, f"CSV row count: expected {n_csv}, got {len(rows)}"
    assert rows[0]['time'] == '0.0', f"first time: {rows[0]['time']}"
    print(f"  CSV header: {list(rows[0].keys())}")
    print(f"  CSV rows:   {len(rows):,}")
    print(f"  [PASS] CSV round-trip correct.\n")

    shutil.rmtree(small_dir, ignore_errors=True)


def test_to_arrays_speed():
    print(f"{'='*60}")
    print(f"Bulk read (to_arrays) speed test")
    print(f"{'='*60}")

    reader = SimLog(LOG_DIR, mode='r')
    t0 = time.perf_counter()
    arrays, masks = reader.to_arrays()
    t1 = time.perf_counter()

    total_bytes = sum(arr.nbytes for arr in arrays.values())
    print(f"  Read {N_RECORDS:,} records ({total_bytes / 1e6:.1f} MB) in {t1 - t0:.3f} s")
    print(f"  Throughput: {total_bytes / (t1 - t0) / 1e6:.0f} MB/s")
    print(f"  [PASS]\n")


def test_none_fields():
    print(f"{'='*60}")
    print(f"None/missing field test")
    print(f"{'='*60}")

    none_dir = LOG_DIR + '_none'
    shutil.rmtree(none_dir, ignore_errors=True)

    with SimLog(none_dir, fields=FIELDS, mode='w', buffer_size=BUFFER_SIZE) as logger:
        # row with all fields
        logger.log(time=1.0, value=2.0, activity=1, entity_type=0,
                   entity_id=10, mission_id=5, lp=1, task_id=3,
                   priority=2, priority_group=1, derived_priority=4)
        # row with mission_id=None
        logger.log(time=2.0, value=3.0, activity=2, entity_type=1,
                   entity_id=20, mission_id=None, lp=0, task_id=7,
                   priority=1, priority_group=0, derived_priority=0)
        # row with mission_id missing entirely
        logger.log(time=3.0, value=4.0, activity=3, entity_type=2,
                   entity_id=30, lp=2, task_id=9,
                   priority=3, priority_group=2, derived_priority=6)

    # read back as dicts
    rows = list(SimLog(none_dir, mode='r'))
    assert len(rows) == 3

    # row 0: all present
    assert rows[0]['mission_id'] == 5
    assert rows[0]['time'] == 1.0

    # row 1: mission_id was None
    assert rows[1]['mission_id'] is None
    assert rows[1]['time'] == 2.0

    # row 2: mission_id was missing
    assert rows[2]['mission_id'] is None
    assert rows[2]['entity_id'] == 30

    # check masks via to_arrays
    arrays, masks = SimLog(none_dir, mode='r').to_arrays()
    assert masks['mission_id'][0] == True
    assert masks['mission_id'][1] == False
    assert masks['mission_id'][2] == False
    assert masks['time'].all()  # time was always provided

    print("  [PASS] None/missing fields handled correctly.\n")
    shutil.rmtree(none_dir, ignore_errors=True)


def test_realworld():
    """Ingest sim_logs.csv, encode to numeric, write through SimLog, read back and verify."""
    print(f"{'='*60}")
    print(f"Real-world data test (sim_logs.csv)")
    print(f"{'='*60}")

    csv_path = Path(__file__).parent / 'sim_logs.csv'
    if not csv_path.exists():
        print("  [SKIP] sim_logs.csv not found.\n")
        return

    # build enum maps from the data
    import csv as csvmod
    from datetime import datetime

    with open(csv_path) as f:
        reader = csvmod.DictReader(f)
        raw_rows = list(reader)

    n_rows = len(raw_rows)
    print(f"  CSV rows: {n_rows:,}")

    # collect unique strings for enum encoding
    activities = sorted({r['activity'] for r in raw_rows if r['activity']})
    entity_types = sorted({r['entity_type'] for r in raw_rows if r['entity_type']})
    act_map = {v: i for i, v in enumerate(activities)}
    ent_map = {v: i for i, v in enumerate(entity_types)}

    # parse timestamp to float (seconds since epoch)
    def parse_time(s):
        if not s:
            return None
        return datetime.strptime(s, '%Y-%m-%d %H:%M:%S').timestamp()

    def to_float(s):
        if not s:
            return None
        return float(s)

    def to_int_or_none(s):
        if not s:
            return None
        return int(float(s))

    # encode
    def encode_row(r):
        return dict(
            time=parse_time(r['time']),
            value=to_float(r['value']),
            activity=act_map.get(r['activity']),
            entity_type=ent_map.get(r['entity_type']),
            entity_id=to_int_or_none(r['entity_id']),
            mission_id=to_int_or_none(r['mission_id']),
            lp=to_int_or_none(r['lp']),
            task_id=to_int_or_none(r['task_id']),
            priority=to_int_or_none(r['priority']),
            priority_group=to_int_or_none(r['priority_group']),
            derived_priority=to_int_or_none(r['derived_priority']),
        )

    real_dir = LOG_DIR + '_real'
    shutil.rmtree(real_dir, ignore_errors=True)

    # write
    t0 = time.perf_counter()
    with SimLog(real_dir, fields=FIELDS, mode='w', buffer_size=BUFFER_SIZE) as logger:
        for r in raw_rows:
            logger.log(**encode_row(r))
    t_write = time.perf_counter() - t0

    # read back
    t0 = time.perf_counter()
    arrays, masks = SimLog(real_dir, mode='r').to_arrays()
    t_read = time.perf_counter() - t0

    # verify record count
    for name, arr in arrays.items():
        assert len(arr) == n_rows, f"{name}: expected {n_rows}, got {len(arr)}"

    # verify masks: time and value should always be present
    assert masks['time'].all(), "time should never be None"
    assert masks['value'].all(), "value should never be None"
    assert masks['activity'].all(), "activity should never be None"

    # entity_type has empties in the CSV (e.g. input_picks rows)
    n_missing_etype = (~masks['entity_type']).sum()
    print(f"  entity_type missing: {n_missing_etype:,} / {n_rows:,}")
    assert n_missing_etype > 0, "expected some missing entity_type values"

    # spot check first non-None entity_id
    for i in range(n_rows):
        if masks['entity_id'][i]:
            orig = float(raw_rows[i]['entity_id'])
            assert arrays['entity_id'][i] == int(orig), \
                f"entity_id mismatch at row {i}: {arrays['entity_id'][i]} vs {int(orig)}"
            break

    # round-trip to CSV
    csv_out = '/tmp/simlog_realworld_out.csv'
    SimLog(real_dir, mode='r').to_csv(csv_out)
    with open(csv_out) as f:
        out_rows = list(csvmod.DictReader(f))
    assert len(out_rows) == n_rows, f"CSV round-trip count: {len(out_rows)} vs {n_rows}"

    total_bytes = sum(arr.nbytes for arr in arrays.values())
    print(f"  Write: {n_rows:,} rows in {t_write:.3f} s ({n_rows / t_write:,.0f} rows/s)")
    print(f"  Read:  {total_bytes / 1e6:.1f} MB in {t_read:.3f} s ({total_bytes / t_read / 1e6:.0f} MB/s)")
    print(f"  [PASS] Real-world round-trip correct.\n")

    shutil.rmtree(real_dir, ignore_errors=True)



def test_correctness_codebook():
    """Correctness + None + string activity/entity_type for SimLogCodebook."""
    print(f"{'='*60}")
    print(f"SimLogCodebook correctness test")
    print(f"{'='*60}")

    d = LOG_DIR + '_codebook_correct'
    shutil.rmtree(d, ignore_errors=True)

    with SimLogCodebook(d, fields=FIELDS, mode='w', buffer_size=BUFFER_SIZE) as logger:
        # rows with string activity/entity_type (as in real data)
        for i in range(10_000):
            logger.log(**make_row_str(i))
        # row with mission_id=None
        logger.log(time=99.0, value=1.0, activity='picking', entity_type='bot',
                   entity_id=10, mission_id=None, lp=1, task_id=3,
                   priority=2, priority_group=1, derived_priority=4)
        # row with mission_id missing entirely
        logger.log(time=100.0, value=2.0, activity='putaway', entity_type='human',
                   entity_id=20, lp=0, task_id=7,
                   priority=1, priority_group=0, derived_priority=0)

    reader = SimLogCodebook(d, mode='r')
    arrays, masks = reader.to_arrays()
    cb = reader.codebook
    total = 10_002

    for name, arr in arrays.items():
        assert len(arr) == total, f"{name}: expected {total}, got {len(arr)}"

    # spot-check numeric fields round-trip
    str_fields = {'activity', 'entity_type'}
    for idx in [0, 1, 5000, 9999]:
        expected = make_row_str(idx)
        for name in FIELD_NAMES:
            if name in str_fields:
                continue  # checked via codebook below
            assert np.isclose(arrays[name][idx], expected[name]), \
                f"row {idx}, {name}: expected {expected[name]}, got {arrays[name][idx]}"

    # verify string codebook covers all activity/entity_type values
    assert set(ACTIVITIES).issubset(set(cb)), f"codebook missing activities: {cb}"
    assert set(ENTITY_TYPES).issubset(set(cb)), f"codebook missing entity_types: {cb}"

    # same string -> same codebook code
    code_0 = int(arrays['activity'][0])   # ACTIVITIES[0] = 'picking'
    code_7 = int(arrays['activity'][7])   # ACTIVITIES[7%7] = 'picking'
    assert code_0 == code_7, f"same activity should get same code: {code_0} vs {code_7}"
    assert cb[code_0] == ACTIVITIES[0]

    # different string -> different code
    code_1 = int(arrays['activity'][1])   # ACTIVITIES[1] = 'putaway'
    assert code_0 != code_1, f"different activities should differ: {code_0} vs {code_1}"

    # check None / missing
    assert masks['mission_id'][10_000] == False
    assert masks['mission_id'][10_001] == False
    assert masks['time'][10_000] == True
    assert masks['time'][10_001] == True

    print(f"  [PASS] SimLogCodebook round-trip + string codebook correct.")
    print(f"  Codebook ({len(cb)} entries): {cb}\n")

    shutil.rmtree(d, ignore_errors=True)


def test_correctness_sparse():
    """Correctness + None + string activity/entity_type for SimLogSparse."""
    print(f"{'='*60}")
    print(f"SimLogSparse correctness test")
    print(f"{'='*60}")

    d = LOG_DIR + '_sparse_correct'
    shutil.rmtree(d, ignore_errors=True)

    with SimLogSparse(d, fields=FIELDS, mode='w', buffer_size=BUFFER_SIZE) as logger:
        # rows with string activity/entity_type
        for i in range(10_000):
            logger.log(**make_row_str(i))
        # None and missing
        logger.log(time=99.0, value=1.0, activity='picking', entity_type='bot',
                   entity_id=10, mission_id=None, lp=1, task_id=3,
                   priority=2, priority_group=1, derived_priority=4)
        logger.log(time=100.0, value=2.0, activity='putaway', entity_type='human',
                   entity_id=20, lp=0, task_id=7,
                   priority=1, priority_group=0, derived_priority=0)

    reader = SimLogSparse(d, mode='r')
    arrays, masks = reader.to_arrays()
    cb = reader.codebook
    total = 10_002

    for name, arr in arrays.items():
        assert len(arr) == total, f"{name}: expected {total}, got {len(arr)}"

    # spot-check numeric fields
    str_fields = {'activity', 'entity_type'}
    for idx in [0, 1, 5000, 9999]:
        expected = make_row_str(idx)
        for name in FIELD_NAMES:
            if name in str_fields:
                continue
            assert np.isclose(arrays[name][idx], expected[name]), \
                f"row {idx}, {name}: expected {expected[name]}, got {arrays[name][idx]}"

    # verify string codebook
    assert set(ACTIVITIES).issubset(set(cb)), f"codebook missing activities: {cb}"
    assert set(ENTITY_TYPES).issubset(set(cb)), f"codebook missing entity_types: {cb}"

    code_0 = int(arrays['activity'][0])
    code_7 = int(arrays['activity'][7])
    assert code_0 == code_7, f"same activity should get same code: {code_0} vs {code_7}"
    assert cb[code_0] == ACTIVITIES[0]

    assert masks['mission_id'][10_000] == False
    assert masks['mission_id'][10_001] == False
    assert masks['time'][10_000] == True
    assert masks['time'][10_001] == True

    # verify type coercion + string decoding via iterator
    rows = list(SimLogSparse(d, mode='r'))
    assert isinstance(rows[0]['activity'], str), \
        f"activity should be decoded string, got {type(rows[0]['activity'])}"
    assert rows[0]['activity'] == ACTIVITIES[0]
    assert isinstance(rows[0]['time'], float), \
        f"time should be float, got {type(rows[0]['time'])}"
    assert isinstance(rows[0]['entity_id'], int), \
        f"entity_id should be int, got {type(rows[0]['entity_id'])}"
    assert rows[0]['entity_id'] == 0

    print(f"  [PASS] SimLogSparse string codebook + int coercion correct.")
    print(f"  Codebook ({len(cb)} entries): {cb}\n")
    shutil.rmtree(d, ignore_errors=True)


def test_correctness_sparse_np():
    """Correctness + None + string activity/entity_type for SimLogSparseNP."""
    print(f"{'='*60}")
    print(f"SimLogSparseNP correctness test")
    print(f"{'='*60}")

    d = LOG_DIR + '_sparsenp_correct'
    shutil.rmtree(d, ignore_errors=True)

    with SimLogSparseNP(d, fields=FIELDS, mode='w', buffer_size=BUFFER_SIZE) as logger:
        for i in range(10_000):
            logger.log(**make_row_str(i))
        logger.log(time=99.0, value=1.0, activity='picking', entity_type='bot',
                   entity_id=10, mission_id=None, lp=1, task_id=3,
                   priority=2, priority_group=1, derived_priority=4)
        logger.log(time=100.0, value=2.0, activity='putaway', entity_type='human',
                   entity_id=20, lp=0, task_id=7,
                   priority=1, priority_group=0, derived_priority=0)

    reader = SimLogSparseNP(d, mode='r')
    arrays, masks = reader.to_arrays()
    cb = reader.codebook
    total = 10_002

    for name, arr in arrays.items():
        assert len(arr) == total, f"{name}: expected {total}, got {len(arr)}"

    str_fields = {'activity', 'entity_type'}
    for idx in [0, 1, 5000, 9999]:
        expected = make_row_str(idx)
        for name in FIELD_NAMES:
            if name in str_fields:
                continue
            assert np.isclose(arrays[name][idx], expected[name]), \
                f"row {idx}, {name}: expected {expected[name]}, got {arrays[name][idx]}"

    assert set(ACTIVITIES).issubset(set(cb))
    assert set(ENTITY_TYPES).issubset(set(cb))
    assert masks['mission_id'][10_000] == False
    assert masks['mission_id'][10_001] == False
    assert masks['time'][10_000] == True

    rows = list(SimLogSparseNP(d, mode='r'))
    assert isinstance(rows[0]['activity'], str)
    assert rows[0]['activity'] == ACTIVITIES[0]
    assert isinstance(rows[0]['time'], float)
    assert isinstance(rows[0]['entity_id'], int)

    print(f"  [PASS] SimLogSparseNP string codebook + int coercion correct.")
    print(f"  Codebook ({len(cb)} entries): {cb}\n")
    shutil.rmtree(d, ignore_errors=True)


def run_benchmark(title, fields, buffer_sizes, runs, n_records=N_RECORDS):
    """Parameterised benchmark runner. Results sorted by ns/call.

    runs: list of (cls | None, label, row_dict[, n_override])
        cls=None  →  bare list.append baseline.
        n_override → use fewer records (e.g. slow SimLogText).
    """
    for bs in buffer_sizes:
        print(f"\n{'='*60}")
        print(f"{title}: {len(fields)} fields, buffer_size={bs:,}")
        print(f"{'='*60}")

        results = []
        baseline_ns = None

        for entry in runs:
            cls, label, row = entry[0], entry[1], entry[2]
            n = entry[3] if len(entry) > 3 else n_records

            if cls is None:
                # bare list.append baseline
                plain = []
                t0 = time.perf_counter()
                for _ in range(n):
                    plain.append(row)
                elapsed = time.perf_counter() - t0
                del plain
            else:
                tag = label.replace(' ', '_').replace('(', '').replace(')', '')
                d = LOG_DIR + f'_bench_{tag}_{bs}'
                shutil.rmtree(d, ignore_errors=True)
                kw = dict(fields=fields, buffer_size=bs)
                if cls is not SimLogText:
                    kw['mode'] = 'w'
                t0 = time.perf_counter()
                with cls(d, **kw) as logger:
                    _log = logger.log
                    for _ in range(n):
                        _log(**row)
                elapsed = time.perf_counter() - t0
                shutil.rmtree(d, ignore_errors=True)

            ns = elapsed / n * 1e9
            if baseline_ns is None:
                baseline_ns = ns
            results.append((label, elapsed, ns))

        # sort by ns/call
        results.sort(key=lambda r: r[2])

        print(f"\n  {'Logger':<40} {'Time (s)':>10} {'ns/call':>10} {'vs baseline':>12}")
        print(f"  {'-'*40} {'-'*10} {'-'*10} {'-'*12}")
        for name, t, ns in results:
            ratio = ns / baseline_ns
            print(f"  {name:<40} {t:>10.3f} {ns:>10.0f} {ratio:>11.1f}x")
        print()


def test_benchmark_comparison():
    """Head-to-head: all loggers, 11 fields."""
    # dense loggers get pre-encoded ints (they don't support strings)
    row_full = make_row(42)
    row_sparse = dict(time=42*0.001, value=42*1.23, activity=42%7,
                      entity_type=42%4, entity_id=42%10_000)
    # codebook/sparse loggers get real strings (as in production)
    row_full_str = make_row_str(42)
    row_sparse_str = dict(time=42*0.001, value=42*1.23, activity='picking',
                          entity_type='bot', entity_id=42%10_000)
    n_text = min(N_RECORDS, 500_000)

    run_benchmark(
        title='Benchmark comparison',
        fields=FIELDS,
        buffer_sizes=[BUFFER_SIZE],
        runs=[
            (None,           'list.append (baseline)',  row_sparse),
            (SimLog,         'SimLog (11 kwargs)',      row_full),
            (SimLogCodebook, 'SimLogCodebook (11 kw)',  row_full_str),
            (SimLogCodebook, 'SimLogCodebook (5 kw)',   row_sparse_str),
            (SimLogSparse,   'SimLogSparse (11 kw)',    row_full_str),
            (SimLogSparse,   'SimLogSparse (5 kw)',     row_sparse_str),
            (SimLogSparseNP, 'SimLogSparseNP (11 kw)',  row_full_str),
            (SimLogSparseNP, 'SimLogSparseNP (5 kw)',   row_sparse_str),
            (SimLogText,     'SimLogText (logging)',    row_full, n_text),
        ],
    )


def test_benchmark_wide():
    """Wide-field benchmark: 34 fields, buffer_size 200K and 500K."""
    WIDE_FIELDS = dict(FIELDS)
    for i in range(23):
        WIDE_FIELDS[f'extra_{i:02d}'] = 'f8'

    # dense loggers: pre-encoded ints
    row_sparse = dict(time=42*0.001, value=42*1.23, activity=42%7,
                      entity_type=42%4, entity_id=42%10_000)
    row_full = dict(row_sparse)
    for i in range(23):
        row_full[f'extra_{i:02d}'] = float(i)
    row_full.update(dict(mission_id=500, lp=2, task_id=100, priority=3,
                         priority_group=1, derived_priority=6))

    # codebook/sparse loggers: real strings
    row_sparse_str = dict(time=42*0.001, value=42*1.23, activity='picking',
                          entity_type='bot', entity_id=42%10_000)
    row_full_str = dict(row_sparse_str)
    for i in range(23):
        row_full_str[f'extra_{i:02d}'] = float(i)
    row_full_str.update(dict(mission_id=500, lp=2, task_id=100, priority=3,
                             priority_group=1, derived_priority=6))

    run_benchmark(
        title='Wide-field benchmark',
        fields=WIDE_FIELDS,
        buffer_sizes=[200_000, 500_000],
        runs=[
            (None,           'list.append (baseline)',  row_sparse),
            (SimLogCodebook, 'SimLogCodebook (5 kw)',   row_sparse_str),
            (SimLogCodebook, 'SimLogCodebook (34 kw)',  row_full_str),
            (SimLogSparse,   'SimLogSparse (5 kw)',     row_sparse_str),
            (SimLogSparse,   'SimLogSparse (34 kw)',    row_full_str),
            (SimLogSparseNP, 'SimLogSparseNP (5 kw)',   row_sparse_str),
            (SimLogSparseNP, 'SimLogSparseNP (34 kw)',  row_full_str),
        ],
    )


if __name__ == '__main__':
    test_correctness()
    test_none_fields()
    test_correctness_codebook()
    test_correctness_sparse()
    test_correctness_sparse_np()
    test_throughput()
    test_read_iter()
    test_to_arrays_speed()
    test_csv_export()
    test_realworld()
    test_benchmark_comparison()
    test_benchmark_wide()

    # cleanup
    shutil.rmtree(LOG_DIR, ignore_errors=True)
    print("All tests passed.")

