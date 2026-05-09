"""
SimLog - High-performance columnar logger for simulation data.

Architecture:
    Structure of Arrays (SoA) with double-buffered containers.
    Each field is a pre-allocated numpy array of length `buffer_size`.
    Two containers alternate: while the simulation writes to one,
    a background thread dumps the other to disk.

Write mode (context manager):
    fields = {'time': 'f8', 'value': 'f8', 'activity': 'i4', 'entity_id': 'i4'}
    with SimLog('logs/', fields=fields, mode='w') as log:
        log.log(time=0.01, value=3.14, activity=1, entity_id=42)

Read mode (iterator, like csv.DictReader):
    for row in SimLog('logs/'):
        print(row['time'], row['entity_id'])

Bulk read:
    arrays = SimLog('logs/').to_arrays()
    print(arrays['time'][:10])

Export:
    SimLog('logs/').to_csv('output.csv')
"""

import numpy as np
import threading
import json
import logging
import os
from pathlib import Path


class Container:
    """One SoA data container: a set of pre-allocated numpy arrays + a write pointer.
    Each field has a companion boolean mask array (True = value present, False = None/missing).
    """
    def __init__(self, fields: dict[str, type], buffer_size) -> None:
        self._fields = list(fields.keys())
        for name, dtype in fields.items():
            setattr(self, name, np.empty(buffer_size, dtype=dtype))
            setattr(self, f'_mask_{name}', np.zeros(buffer_size, dtype=np.bool_))  # pre-init False
        self.pointer = 0


class SimLog:
    """High-performance columnar logger for simulation data."""

    def __init__(self, path, fields: dict[str, type]|None=None, mode='r', buffer_size=200_000):
        self._path = Path(path)
        self._mode = mode
        self._buffer_size = buffer_size

        if mode == 'w':
            if fields is None:
                raise ValueError("fields required in write mode")
            self._fields = fields
            self._names = list(fields.keys())
            self._dtypes = [np.dtype(v) for v in fields.values()]
            self._init_writer()
        elif mode == 'r':
            self._init_reader()
        else:
            raise ValueError(f"mode must be 'r' or 'w', got {mode!r}")

    # init

    def _init_writer(self):
        self._path.mkdir(parents=True, exist_ok=True)
        self._seq = 0

        # metadata embedded in every chunk
        self._meta_bytes = np.frombuffer(
            json.dumps({
                'fields': {n: str(d) for n, d in zip(self._names, self._dtypes)},
                'buffer_size': self._buffer_size,
            }).encode(),
            dtype=np.uint8,
        )

        # two pre-allocated SoA containers
        self.buffer_1 = Container(fields=self._fields, buffer_size=self._buffer_size)
        self.buffer_2 = Container(fields=self._fields, buffer_size=self._buffer_size)
        self.active = self.buffer_1
        self.inactive = self.buffer_2

        # synchronisation
        self._flush_ready = threading.Event()
        self._flush_ready.set()       # writer thread starts idle
        self._flush_needed = threading.Event()
        self._flush_ref = None        # container handed to the writer thread
        self._flush_count = 0
        self._done = False

        # background disk-writer thread
        self._writer = threading.Thread(target=self._drain, daemon=True)
        self._writer.start()

    def _init_reader(self):
        if not self._path.exists():
            raise FileNotFoundError(self._path)
        # read metadata from the first chunk
        chunks = sorted(self._path.glob('chunk_*.npz'))
        if not chunks:
            raise FileNotFoundError(f"no chunk files in {self._path}")
        with np.load(chunks[0]) as data:
            meta = json.loads(data['_meta'].tobytes().decode())
        self._fields = {n: np.dtype(d) for n, d in meta['fields'].items()}
        self._names = list(self._fields.keys())
        self._dtypes = list(self._fields.values())
        self._buffer_size = meta['buffer_size']

    # hot path

    def log(self, **kwargs):
        """Append one record. Only provided (non-None) kwargs are written.
        Masks are pre-initialized to False, so missing fields cost zero."""
        active = self.active
        p = active.pointer
        for name, value in kwargs.items():
            if value is not None:
                getattr(active, name)[p] = value
                getattr(active, f'_mask_{name}')[p] = True
        p += 1
        active.pointer = p
        if p == self._buffer_size:
            self._swap()

    # container swap

    def _swap(self):
        """Hand the full container to the writer thread, switch to the other one."""
        self._flush_ready.wait()          # back-pressure: wait if writer is still busy
        self._flush_ready.clear()

        self._flush_ref = self.active
        self._flush_count = self.active.pointer

        # toggle: active becomes inactive, inactive becomes active
        self.active, self.inactive = self.inactive, self.active
        self.active.pointer = 0
        # reset masks back to False for the new active buffer
        for name in self._names:
            getattr(self.active, f'_mask_{name}')[:] = False

        self._flush_needed.set()          # wake the writer thread

    # background writer thread

    def _drain(self):
        while True:
            self._flush_needed.wait()
            self._flush_needed.clear()

            container = self._flush_ref
            count = self._flush_count
            self._flush_ref = None

            if container is not None:
                self._write_chunk(container, count)

            self._flush_ready.set()

            if self._done:
                return

    def _write_chunk(self, container, count):
        """Write one chunk as .npz: metadata + data arrays + packbits masks."""
        path = self._path / f'chunk_{self._seq:06d}.npz'
        self._seq += 1
        arrays = {'_meta': self._meta_bytes}
        for name in self._names:
            arrays[name] = getattr(container, name)[:count]
            arrays[f'_mask_{name}'] = np.packbits(getattr(container, f'_mask_{name}')[:count])
        np.savez(path, **arrays)

    # read path

    def _iter_chunks(self):
        """Yield (count, dict[name, ndarray], dict[name, bool_ndarray]) for each chunk."""
        for chunk_path in sorted(self._path.glob('chunk_*.npz')):
            with np.load(chunk_path) as data:
                count = len(data[self._names[0]])
                arrays = {}
                masks = {}
                for name in self._names:
                    arrays[name] = data[name]
                    masks[name] = np.unpackbits(data[f'_mask_{name}'])[:count].astype(np.bool_)
            yield count, arrays, masks

    def __iter__(self):
        """Iterate rows as dicts, like csv.DictReader. Masked fields yield None."""
        if self._mode != 'r':
            raise RuntimeError("open in 'r' mode to iterate")
        for count, arrays, masks in self._iter_chunks():
            for i in range(count):
                yield {
                    name: (arrays[name][i].item() if masks[name][i] else None)
                    for name in self._names
                }

    def to_arrays(self):
        """Read all chunks into concatenated column arrays.
        Returns dict[str, ndarray] for data and dict[str, ndarray] for masks."""
        if self._mode != 'r':
            raise RuntimeError("open in 'r' mode to read")
        accum = {n: [] for n in self._names}
        mask_accum = {n: [] for n in self._names}
        for _count, arrays, masks in self._iter_chunks():
            for name in self._names:
                accum[name].append(arrays[name])
                mask_accum[name].append(masks[name])
        return (
            {n: np.concatenate(arrs) for n, arrs in accum.items()},
            {n: np.concatenate(arrs) for n, arrs in mask_accum.items()},
        )

    def to_csv(self, output_path):
        """Export all log data to a CSV file."""
        import csv
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self._names)
            writer.writeheader()
            for row in self:
                writer.writerow(row)

    # context manager

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        if self._mode == 'w':
            self.close()

    def close(self):
        """Flush the partially-filled container and join the writer thread."""
        if self.active.pointer > 0:
            self._swap()
        self._flush_ready.wait()
        self._done = True
        self._flush_needed.set()
        self._writer.join()


# SimLogCodebook
# Sparse COO format inspired by zstd's evolving codebook.
# Each field value is stored as a (row_id, key_code, value) triple.
# Float and integer values live in separate streams.
# String values are auto-encoded via an evolving codebook.
# No mask arrays needed — missing fields simply have no entry.

class _SparseContainer:
    __slots__ = ('f_rids', 'f_keys', 'f_vals',
                 'i_rids', 'i_keys', 'i_vals', 'row_count')

    def __init__(self):
        self.f_rids = []
        self.f_keys = []
        self.f_vals = []
        self.i_rids = []
        self.i_keys = []
        self.i_vals = []
        self.row_count = 0

    def reset(self):
        # safe: flush thread has the OTHER container, not this one
        self.f_rids.clear()
        self.f_keys.clear()
        self.f_vals.clear()
        self.i_rids.clear()
        self.i_keys.clear()
        self.i_vals.clear()
        self.row_count = 0


class SimLogCodebook:
    """Sparse codebook logger. Entries stored as (row_id, key_code, value) triples.
    Float and integer values in separate streams. String values auto-encoded
    via an evolving codebook (inspired by zstd dictionary compression)."""

    def __init__(self, path, fields: dict[str, str] | None = None, mode='r', buffer_size=200_000):
        self._path = Path(path)
        self._mode = mode
        self._buffer_size = buffer_size

        if mode == 'w':
            if fields is None:
                raise ValueError("fields required in write mode")
            self._names = list(fields.keys())
            self._dtypes = [np.dtype(v) for v in fields.values()]
            self._n = len(self._names)
            self._init_writer()
        elif mode == 'r':
            self._init_reader()
        else:
            raise ValueError(f"mode must be 'r' or 'w', got {mode!r}")

    def _init_writer(self):
        self._path.mkdir(parents=True, exist_ok=True)
        self._seq = 0

        # key codebook: field name -> integer code (fixed at init)
        self._key_map = {name: i for i, name in enumerate(self._names)}

        # classify fields: float vs int
        self._is_float = [d.kind == 'f' for d in self._dtypes]

        # evolving string-value codebook
        self._str_codes = {}   # string -> int
        self._str_list = []    # int -> string (reverse map)
        self._str_fields = set()  # key_codes that have seen string values

        # base metadata (codebook snapshot added per chunk)
        self._base_meta = {
            'fields': {n: str(d) for n, d in zip(self._names, self._dtypes)},
            'buffer_size': self._buffer_size,
        }

        # double buffer
        self._buf = [_SparseContainer(), _SparseContainer()]
        self._active = 0
        self._build_dispatch()

        # synchronisation
        self._flush_ready = threading.Event()
        self._flush_ready.set()
        self._flush_needed = threading.Event()
        self._flush_ref = None
        self._flush_count = 0
        self._done = False

        self._writer = threading.Thread(target=self._drain, daemon=True)
        self._writer.start()

    def _build_dispatch(self):
        """Map each field name to (key_code, rids_list, keys_list, vals_list).
        Rebuilt after every swap to point to the active container's lists."""
        c = self._buf[self._active]
        self._dispatch = {}
        for name in self._names:
            kc = self._key_map[name]
            if self._is_float[kc]:
                self._dispatch[name] = (kc, c.f_rids, c.f_keys, c.f_vals)
            else:
                self._dispatch[name] = (kc, c.i_rids, c.i_keys, c.i_vals)

    def _init_reader(self):
        if not self._path.exists():
            raise FileNotFoundError(self._path)
        chunks = sorted(self._path.glob('chunk_*.npz'))
        if not chunks:
            raise FileNotFoundError(f"no chunk files in {self._path}")
        # read field schema from the first chunk
        with np.load(chunks[0]) as data:
            meta = json.loads(data['_meta'].tobytes().decode())
        self._names = list(meta['fields'].keys())
        self._dtypes = [np.dtype(d) for d in meta['fields'].values()]
        self._n = len(self._names)
        self._buffer_size = meta['buffer_size']
        self._is_float = [d.kind == 'f' for d in self._dtypes]
        # codebook written once at close()
        cb_path = self._path / '_codebook.json'
        if cb_path.exists():
            with open(cb_path) as f:
                cb = json.load(f)
            self._str_list = cb.get('codebook', [])
            self._str_fields = set(cb.get('str_fields', []))
        else:
            self._str_list = []
            self._str_fields = set()

    @property
    def codebook(self):
        """Return the string codebook: list where index -> original string."""
        return list(self._str_list)

    # hot path

    def log(self, **kwargs):
        """Append one record. String values auto-encoded via codebook."""
        c = self._buf[self._active]
        rid = c.row_count
        disp = self._dispatch
        sc = self._str_codes
        sl = self._str_list
        sf = self._str_fields
        for name, v in kwargs.items():
            if v is not None:
                kc, r, k, vl = disp[name]
                if isinstance(v, str):
                    sv = sc.get(v)
                    if sv is None:
                        sv = len(sl)
                        sc[v] = sv
                        sl.append(v)
                    sf.add(kc)
                    v = sv
                r.append(rid)
                k.append(kc)
                vl.append(v)
        c.row_count = rid + 1
        if c.row_count >= self._buffer_size:
            self._swap()

    # swap + background writer

    def _swap(self):
        self._flush_ready.wait()
        self._flush_ready.clear()
        self._flush_ref = self._buf[self._active]
        self._flush_count = self._flush_ref.row_count
        self._active ^= 1
        self._buf[self._active].reset()
        self._build_dispatch()
        self._flush_needed.set()

    def _drain(self):
        while True:
            self._flush_needed.wait()
            self._flush_needed.clear()
            ref = self._flush_ref
            count = self._flush_count
            self._flush_ref = None
            if ref is not None:
                self._write_chunk(ref, count)
            self._flush_ready.set()
            if self._done:
                return

    def _write_chunk(self, container, count):
        """Convert lists to numpy arrays and write .npz (runs in background thread)."""
        path = self._path / f'chunk_{self._seq:06d}.npz'
        self._seq += 1

        meta = dict(self._base_meta)
        meta['row_count'] = count
        meta_bytes = np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8)

        def to_arr(lst, dtype):
            return np.array(lst, dtype=dtype) if lst else np.array([], dtype=dtype)

        np.savez(path,
            _meta=meta_bytes,
            f_rids=to_arr(container.f_rids, np.int32),
            f_keys=to_arr(container.f_keys, np.int16),
            f_vals=to_arr(container.f_vals, np.float64),
            i_rids=to_arr(container.i_rids, np.int32),
            i_keys=to_arr(container.i_keys, np.int16),
            i_vals=to_arr(container.i_vals, np.int64),
        )

    # read path

    def _iter_chunks(self):
        for chunk_path in sorted(self._path.glob('chunk_*.npz')):
            with np.load(chunk_path) as data:
                meta = json.loads(data['_meta'].tobytes().decode())
                count = meta['row_count']
                yield (count,
                       data['f_rids'], data['f_keys'], data['f_vals'],
                       data['i_rids'], data['i_keys'], data['i_vals'])

    def __iter__(self):
        """Iterate rows as dicts. String-coded values decoded via codebook."""
        if self._mode != 'r':
            raise RuntimeError("open in 'r' mode to iterate")
        sf = self._str_fields
        inv_cb = self._str_list
        for (count,
             f_rids, f_keys, f_vals,
             i_rids, i_keys, i_vals) in self._iter_chunks():
            rows = [{} for _ in range(count)]
            for j in range(len(f_rids)):
                rows[f_rids[j]][self._names[f_keys[j]]] = f_vals[j].item()
            for j in range(len(i_rids)):
                kc = i_keys[j].item()
                v = i_vals[j].item()
                if kc in sf:
                    v = inv_cb[int(v)]
                rows[i_rids[j]][self._names[kc]] = v
            for row in rows:
                for name in self._names:
                    if name not in row:
                        row[name] = None
                yield row

    def to_arrays(self):
        """Reconstruct dense SoA arrays from sparse COO entries."""
        if self._mode != 'r':
            raise RuntimeError("open in 'r' mode to read")
        accum = {n: [] for n in self._names}
        mask_accum = {n: [] for n in self._names}
        for (count,
             f_rids, f_keys, f_vals,
             i_rids, i_keys, i_vals) in self._iter_chunks():
            # allocate dense arrays for this chunk
            arrays = {}
            masks = {}
            for name, dtype in zip(self._names, self._dtypes):
                arrays[name] = np.zeros(count, dtype=dtype)
                masks[name] = np.zeros(count, dtype=np.bool_)
            # vectorised scatter: float entries
            for ki, name in enumerate(self._names):
                if self._is_float[ki] and len(f_rids) > 0:
                    sel = f_keys == ki
                    if sel.any():
                        arrays[name][f_rids[sel]] = f_vals[sel]
                        masks[name][f_rids[sel]] = True
                elif not self._is_float[ki] and len(i_rids) > 0:
                    sel = i_keys == ki
                    if sel.any():
                        arrays[name][i_rids[sel]] = i_vals[sel]
                        masks[name][i_rids[sel]] = True
            for name in self._names:
                accum[name].append(arrays[name])
                mask_accum[name].append(masks[name])
        return (
            {n: np.concatenate(arrs) for n, arrs in accum.items()},
            {n: np.concatenate(arrs) for n, arrs in mask_accum.items()},
        )

    def to_csv(self, out_path):
        """Export to CSV. Codebook-encoded values are decoded back to strings."""
        if self._mode != 'r':
            raise RuntimeError("open in 'r' mode to export")
        with open(out_path, 'w') as f:
            f.write(','.join(self._names) + '\n')
            for row in self:
                parts = []
                for name in self._names:
                    v = row[name]
                    parts.append('' if v is None else str(v))
                f.write(','.join(parts) + '\n')

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        if self._mode == 'w':
            self.close()

    def close(self):
        if self._buf[self._active].row_count > 0:
            self._swap()
        self._flush_ready.wait()
        self._done = True
        self._flush_needed.set()
        self._writer.join()
        # write codebook once at the end
        cb_path = self._path / '_codebook.json'
        with open(cb_path, 'w') as f:
            json.dump({'codebook': self._str_list,
                       'str_fields': list(self._str_fields)}, f)


# SimLogSparse
# Single float64 value stream. All values cast to float.
# Simplifies the hot-path: no dtype dispatch, no tuple unpacking per field.
# Integers within ±2^53 survive the round-trip exactly.

class _SparseF64Container:
    __slots__ = ('rids', 'keys', 'vals', 'row_count')

    def __init__(self):
        self.rids = []
        self.keys = []
        self.vals = []
        self.row_count = 0

    def reset(self):
        self.rids.clear()
        self.keys.clear()
        self.vals.clear()
        self.row_count = 0


class SimLogSparse:
    """Sparse codebook logger with a single float64 value stream.
    All values stored as float64. Integers round-trip exactly for |v| < 2^53."""

    def __init__(self, path, fields: dict[str, str] | None = None, mode='r', buffer_size=200_000):
        self._path = Path(path)
        self._mode = mode
        self._buffer_size = buffer_size

        if mode == 'w':
            if fields is None:
                raise ValueError("fields required in write mode")
            self._names = list(fields.keys())
            self._dtypes = [np.dtype(v) for v in fields.values()]
            self._n = len(self._names)
            self._init_writer()
        elif mode == 'r':
            self._init_reader()
        else:
            raise ValueError(f"mode must be 'r' or 'w', got {mode!r}")

    def _init_writer(self):
        self._path.mkdir(parents=True, exist_ok=True)
        self._seq = 0

        # key codebook: name -> integer code
        self._kc = {name: i for i, name in enumerate(self._names)}

        # type tracking: key_code -> 'i' or 'f' (recorded on first encounter)
        self._key_types = {}

        # set of key codes known to be string-valued (for fast hot-path check)
        self._str_keys = set()

        # evolving string codebook
        self._str_codes = {}
        self._str_list = []

        self._base_meta = {
            'fields': {n: str(d) for n, d in zip(self._names, self._dtypes)},
            'buffer_size': self._buffer_size,
        }

        self._buf = [_SparseF64Container(), _SparseF64Container()]
        self._active = 0

        self._flush_ready = threading.Event()
        self._flush_ready.set()
        self._flush_needed = threading.Event()
        self._flush_ref = None
        self._flush_count = 0
        self._done = False

        self._writer = threading.Thread(target=self._drain, daemon=True)
        self._writer.start()

    def _init_reader(self):
        if not self._path.exists():
            raise FileNotFoundError(self._path)
        chunks = sorted(self._path.glob('chunk_*.npz'))
        if not chunks:
            raise FileNotFoundError(f"no chunk files in {self._path}")
        with np.load(chunks[0]) as data:
            meta = json.loads(data['_meta'].tobytes().decode())
        self._names = list(meta['fields'].keys())
        self._dtypes = [np.dtype(d) for d in meta['fields'].values()]
        self._n = len(self._names)
        self._buffer_size = meta['buffer_size']
        cb_path = self._path / '_codebook.json'
        if cb_path.exists():
            with open(cb_path) as f:
                cb = json.load(f)
            self._str_list = cb.get('codebook', [])
            self._str_keys = set(cb.get('str_keys', []))
            self._key_types = {int(k): v for k, v in cb.get('key_types', {}).items()}
        else:
            self._str_list = []
            self._str_keys = set()
            self._key_types = {}

    @property
    def codebook(self):
        return list(self._str_list)

    # hot path: single stream, no tuple unpack, locals hoisted
    # isinstance only fires once per key (first encounter).
    # After that, _str_keys set lookup replaces isinstance.
    def log(self, **kwargs):
        c = self._buf[self._active]
        rid = c.row_count
        rids = c.rids
        keys = c.keys
        vals = c.vals
        kc = self._kc
        kt = self._key_types
        sk = self._str_keys
        sc = self._str_codes
        sl = self._str_list
        for name, v in kwargs.items():
            if v is not None:
                k = kc[name]
                if k in sk:
                    # known string field: codebook encode, no isinstance needed
                    sv = sc.get(v)
                    if sv is None:
                        sv = len(sl)
                        sc[v] = sv
                        sl.append(v)
                    v = sv
                elif k not in kt:
                    # first value for this key: determine and record type
                    if isinstance(v, str):
                        sk.add(k)
                        kt[k] = 'i'  # codebook codes are ints
                        sv = len(sl)  # first string for this key, always new
                        sc[v] = sv
                        sl.append(v)
                        v = sv
                    else:
                        kt[k] = 'i' if isinstance(v, int) else 'f'
                rids.append(rid)
                keys.append(k)
                vals.append(v)
        c.row_count = rid + 1
        if c.row_count >= self._buffer_size:
            self._swap()

    def _swap(self):
        self._flush_ready.wait()
        self._flush_ready.clear()
        self._flush_ref = self._buf[self._active]
        self._flush_count = self._flush_ref.row_count
        self._active ^= 1
        self._buf[self._active].reset()
        self._flush_needed.set()

    def _drain(self):
        while True:
            self._flush_needed.wait()
            self._flush_needed.clear()
            ref = self._flush_ref
            count = self._flush_count
            self._flush_ref = None
            if ref is not None:
                self._write_chunk(ref, count)
            self._flush_ready.set()
            if self._done:
                return

    def _write_chunk(self, container, count):
        path = self._path / f'chunk_{self._seq:06d}.npz'
        self._seq += 1
        meta = dict(self._base_meta)
        meta['row_count'] = count
        meta_bytes = np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8)

        def to_arr(lst, dtype):
            return np.array(lst, dtype=dtype) if lst else np.array([], dtype=dtype)

        np.savez(path,
            _meta=meta_bytes,
            rids=to_arr(container.rids, np.int32),
            keys=to_arr(container.keys, np.int16),
            vals=to_arr(container.vals, np.float64),
        )

    # read path
    def _iter_chunks(self):
        for chunk_path in sorted(self._path.glob('chunk_*.npz')):
            with np.load(chunk_path) as data:
                meta = json.loads(data['_meta'].tobytes().decode())
                count = meta['row_count']
                yield count, data['rids'], data['keys'], data['vals']

    def __iter__(self):
        if self._mode != 'r':
            raise RuntimeError("open in 'r' mode to iterate")
        kt = self._key_types
        sk = self._str_keys
        inv_cb = self._str_list
        for count, rids, keys, vals in self._iter_chunks():
            rows = [{} for _ in range(count)]
            for j in range(len(rids)):
                kc = keys[j].item()
                v = vals[j].item()
                if kc in sk:
                    v = inv_cb[int(v)]
                elif kt.get(kc) == 'i':
                    v = int(v)
                rows[rids[j]][self._names[kc]] = v
            for row in rows:
                for name in self._names:
                    if name not in row:
                        row[name] = None
                yield row

    def to_arrays(self):
        if self._mode != 'r':
            raise RuntimeError("open in 'r' mode to read")
        accum = {n: [] for n in self._names}
        mask_accum = {n: [] for n in self._names}
        for count, rids, keys, vals in self._iter_chunks():
            arrays = {}
            masks = {}
            for name, dtype in zip(self._names, self._dtypes):
                arrays[name] = np.zeros(count, dtype=dtype)
                masks[name] = np.zeros(count, dtype=np.bool_)
            for ki, name in enumerate(self._names):
                if len(rids) > 0:
                    sel = keys == ki
                    if sel.any():
                        arrays[name][rids[sel]] = vals[sel]
                        masks[name][rids[sel]] = True
            for name in self._names:
                accum[name].append(arrays[name])
                mask_accum[name].append(masks[name])
        return (
            {n: np.concatenate(arrs) for n, arrs in accum.items()},
            {n: np.concatenate(arrs) for n, arrs in mask_accum.items()},
        )

    def to_csv(self, out_path):
        if self._mode != 'r':
            raise RuntimeError("open in 'r' mode to export")
        with open(out_path, 'w') as f:
            f.write(','.join(self._names) + '\n')
            for row in self:
                parts = []
                for name in self._names:
                    v = row[name]
                    parts.append('' if v is None else str(v))
                f.write(','.join(parts) + '\n')

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        if self._mode == 'w':
            self.close()

    def close(self):
        if self._buf[self._active].row_count > 0:
            self._swap()
        self._flush_ready.wait()
        self._done = True
        self._flush_needed.set()
        self._writer.join()
        cb_path = self._path / '_codebook.json'
        with open(cb_path, 'w') as f:
            json.dump({'codebook': self._str_list,
                       'str_keys': list(self._str_keys),
                       'key_types': {str(k): v for k, v in self._key_types.items()}}, f)


# SimLogSparseNP
# Like SimLogSparse but with pre-allocated numpy arrays instead of Python lists.
# Eliminates per-append Python object overhead (~28 bytes/int vs 4-8 bytes numpy scalar).
# Trade-off: must allocate for worst-case entries (buffer_size * n_fields).

class _SparseF64NPContainer:
    __slots__ = ('rids', 'keys', 'vals', 'row_count', 'ptr')

    def __init__(self, max_entries):
        self.rids = np.empty(max_entries, dtype=np.int32)
        self.keys = np.empty(max_entries, dtype=np.int16)
        self.vals = np.empty(max_entries, dtype=np.float64)
        self.row_count = 0
        self.ptr = 0  # write pointer into the sparse arrays

    def reset(self):
        self.row_count = 0
        self.ptr = 0


class SimLogSparseNP:
    """Sparse codebook logger with pre-allocated numpy arrays.
    Same COO layout as SimLogSparse but numpy scalar writes instead of list.append."""

    def __init__(self, path, fields: dict[str, str] | None = None, mode='r', buffer_size=200_000):
        self._path = Path(path)
        self._mode = mode
        self._buffer_size = buffer_size

        if mode == 'w':
            if fields is None:
                raise ValueError("fields required in write mode")
            self._names = list(fields.keys())
            self._dtypes = [np.dtype(v) for v in fields.values()]
            self._n = len(self._names)
            self._init_writer()
        elif mode == 'r':
            self._init_reader()
        else:
            raise ValueError(f"mode must be 'r' or 'w', got {mode!r}")

    def _init_writer(self):
        self._path.mkdir(parents=True, exist_ok=True)
        self._seq = 0
        max_entries = self._buffer_size * self._n  # worst case: every row has every field

        self._kc = {name: i for i, name in enumerate(self._names)}
        self._key_types = {}
        self._str_keys = set()
        self._str_codes = {}
        self._str_list = []

        self._base_meta = {
            'fields': {n: str(d) for n, d in zip(self._names, self._dtypes)},
            'buffer_size': self._buffer_size,
        }

        self._buf = [_SparseF64NPContainer(max_entries), _SparseF64NPContainer(max_entries)]
        self._active = 0

        self._flush_ready = threading.Event()
        self._flush_ready.set()
        self._flush_needed = threading.Event()
        self._flush_ref = None
        self._flush_count = 0
        self._flush_ptr = 0
        self._done = False

        self._writer = threading.Thread(target=self._drain, daemon=True)
        self._writer.start()

    def _init_reader(self):
        if not self._path.exists():
            raise FileNotFoundError(self._path)
        chunks = sorted(self._path.glob('chunk_*.npz'))
        if not chunks:
            raise FileNotFoundError(f"no chunk files in {self._path}")
        with np.load(chunks[0]) as data:
            meta = json.loads(data['_meta'].tobytes().decode())
        self._names = list(meta['fields'].keys())
        self._dtypes = [np.dtype(d) for d in meta['fields'].values()]
        self._n = len(self._names)
        self._buffer_size = meta['buffer_size']
        cb_path = self._path / '_codebook.json'
        if cb_path.exists():
            with open(cb_path) as f:
                cb = json.load(f)
            self._str_list = cb.get('codebook', [])
            self._str_keys = set(cb.get('str_keys', []))
            self._key_types = {int(k): v for k, v in cb.get('key_types', {}).items()}
        else:
            self._str_list = []
            self._str_keys = set()
            self._key_types = {}

    @property
    def codebook(self):
        return list(self._str_list)

    # hot path: numpy scalar writes instead of list.append
    def log(self, **kwargs):
        c = self._buf[self._active]
        rid = c.row_count
        rids = c.rids
        keys = c.keys
        vals = c.vals
        p = c.ptr
        kc = self._kc
        kt = self._key_types
        sk = self._str_keys
        sc = self._str_codes
        sl = self._str_list
        for name, v in kwargs.items():
            if v is not None:
                k = kc[name]
                if k in sk:
                    sv = sc.get(v)
                    if sv is None:
                        sv = len(sl)
                        sc[v] = sv
                        sl.append(v)
                    v = sv
                elif k not in kt:
                    if isinstance(v, str):
                        sk.add(k)
                        kt[k] = 'i'
                        sv = len(sl)
                        sc[v] = sv
                        sl.append(v)
                        v = sv
                    else:
                        kt[k] = 'i' if isinstance(v, int) else 'f'
                rids[p] = rid
                keys[p] = k
                vals[p] = v
                p += 1
        c.ptr = p
        c.row_count = rid + 1
        if c.row_count >= self._buffer_size:
            self._swap()

    def _swap(self):
        self._flush_ready.wait()
        self._flush_ready.clear()
        ref = self._buf[self._active]
        self._flush_ref = ref
        self._flush_count = ref.row_count
        self._flush_ptr = ref.ptr
        self._active ^= 1
        self._buf[self._active].reset()
        self._flush_needed.set()

    def _drain(self):
        while True:
            self._flush_needed.wait()
            self._flush_needed.clear()
            ref = self._flush_ref
            count = self._flush_count
            ptr = self._flush_ptr
            self._flush_ref = None
            if ref is not None:
                self._write_chunk(ref, count, ptr)
            self._flush_ready.set()
            if self._done:
                return

    def _write_chunk(self, container, count, ptr):
        path = self._path / f'chunk_{self._seq:06d}.npz'
        self._seq += 1
        meta = dict(self._base_meta)
        meta['row_count'] = count
        meta_bytes = np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8)
        # slice up to ptr — already numpy, no conversion needed
        np.savez(path,
            _meta=meta_bytes,
            rids=container.rids[:ptr],
            keys=container.keys[:ptr],
            vals=container.vals[:ptr],
        )

    # read path — identical to SimLogSparse
    def _iter_chunks(self):
        for chunk_path in sorted(self._path.glob('chunk_*.npz')):
            with np.load(chunk_path) as data:
                meta = json.loads(data['_meta'].tobytes().decode())
                count = meta['row_count']
                yield count, data['rids'], data['keys'], data['vals']

    def __iter__(self):
        if self._mode != 'r':
            raise RuntimeError("open in 'r' mode to iterate")
        kt = self._key_types
        sk = self._str_keys
        inv_cb = self._str_list
        for count, rids, keys, vals in self._iter_chunks():
            rows = [{} for _ in range(count)]
            for j in range(len(rids)):
                kc = keys[j].item()
                v = vals[j].item()
                if kc in sk:
                    v = inv_cb[int(v)]
                elif kt.get(kc) == 'i':
                    v = int(v)
                rows[rids[j]][self._names[kc]] = v
            for row in rows:
                for name in self._names:
                    if name not in row:
                        row[name] = None
                yield row

    def to_arrays(self):
        if self._mode != 'r':
            raise RuntimeError("open in 'r' mode to read")
        accum = {n: [] for n in self._names}
        mask_accum = {n: [] for n in self._names}
        for count, rids, keys, vals in self._iter_chunks():
            arrays = {}
            masks = {}
            for name, dtype in zip(self._names, self._dtypes):
                arrays[name] = np.zeros(count, dtype=dtype)
                masks[name] = np.zeros(count, dtype=np.bool_)
            for ki, name in enumerate(self._names):
                if len(rids) > 0:
                    sel = keys == ki
                    if sel.any():
                        arrays[name][rids[sel]] = vals[sel]
                        masks[name][rids[sel]] = True
            for name in self._names:
                accum[name].append(arrays[name])
                mask_accum[name].append(masks[name])
        return (
            {n: np.concatenate(arrs) for n, arrs in accum.items()},
            {n: np.concatenate(arrs) for n, arrs in mask_accum.items()},
        )

    def to_csv(self, out_path):
        if self._mode != 'r':
            raise RuntimeError("open in 'r' mode to export")
        with open(out_path, 'w') as f:
            f.write(','.join(self._names) + '\n')
            for row in self:
                parts = []
                for name in self._names:
                    v = row[name]
                    parts.append('' if v is None else str(v))
                f.write(','.join(parts) + '\n')

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        if self._mode == 'w':
            self.close()

    def close(self):
        if self._buf[self._active].row_count > 0:
            self._swap()
        self._flush_ready.wait()
        self._done = True
        self._flush_needed.set()
        self._writer.join()
        cb_path = self._path / '_codebook.json'
        with open(cb_path, 'w') as f:
            json.dump({'codebook': self._str_list,
                       'str_keys': list(self._str_keys),
                       'key_types': {str(k): v for k, v in self._key_types.items()}}, f)


# SimLogText
# Python native logging module. Write-only benchmark baseline.

class SimLogText:
    """Text-based logger using Python's logging module. For benchmark comparison only."""

    def __init__(self, path, fields: dict[str, str] | None = None, buffer_size=200_000):
        self._path = Path(path)
        self._path.mkdir(parents=True, exist_ok=True)
        self._names = list(fields.keys()) if fields else []
        self._sep = ','

        # configure python logger
        self._logger = logging.getLogger(f'simlog_text_{id(self)}')
        self._logger.setLevel(logging.INFO)
        self._logger.propagate = False
        # remove any existing handlers
        self._logger.handlers.clear()

        log_file = self._path / 'sim.log'
        self._handler = logging.FileHandler(log_file, mode='w')
        self._handler.setLevel(logging.INFO)
        self._handler.setFormatter(logging.Formatter('%(message)s'))
        self._logger.addHandler(self._handler)

        # write header
        self._logger.info(self._sep.join(self._names))

    def log(self, **kwargs):
        """Append one record as a comma-separated text line."""
        parts = []
        for name in self._names:
            v = kwargs.get(name)
            parts.append('' if v is None else str(v))
        self._logger.info(self._sep.join(parts))

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        self._handler.flush()
        self._handler.close()
        self._logger.removeHandler(self._handler)
