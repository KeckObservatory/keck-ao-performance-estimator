"""Process-pool parallelism for the engine (parallel project: PLAN section 4,
design block D.2 / D.5, PR-D10).

- ONE pool per process, module-level, created lazily by the first call with
  workers > 1, reused by every later call, recreated only when the worker
  count changes, shut down at interpreter exit (PR-D10: the pool is session
  infrastructure, so a caller making more than one call pays its start once).
- Workers inherit this process's environment unchanged, BLAS threading
  included (parallel PR-D11): OpenBLAS's thread count changes some LAPACK /
  BLAS results in their last bits -- measured, psf_fit_model.py's refused
  OPEN-8 SEED+34 t1 prints a cleaned SR of 6.36 with the default and 6.33
  with OPENBLAS_NUM_THREADS=1 -- so pinning workers to 1 thread would break
  the bit-identity everything here is built on.
- Large arrays reach the workers through ONE shared-memory block per call,
  never through per-task pickles: `Shared` pickles a call's state with every
  ndarray of at least SHARE_MIN_BYTES replaced by a reference into the block,
  and a worker unpickles that state once per call into READ-ONLY views, so an
  in-place write in a worker raises instead of corrupting what every other
  task reads.
- Every result is produced by the SAME serial function on the SAME inputs and
  callers reassemble results in submission order, so `workers > 1` returns
  exactly what `workers = 1` returns.  Asserted, not assumed:
  sr_estimator/regress/parallel_model.py.
"""
import atexit
import copy
import io
import multiprocessing as mp
import os
import pickle
import threading
import time
import uuid
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import shared_memory

import numpy as np

SHARE_MIN_BYTES = 1 << 16

_LOCK = threading.Lock()
_POOL = None
_POOL_WORKERS = 0


# ------------------------------------------------------------------ pool

def _ping(delay):
    time.sleep(delay)
    return os.getpid()


def get_pool(workers):
    """This process's persistent pool of `workers` processes (PR-D10),
    spawned with this process's environment unchanged (PR-D11)."""
    global _POOL, _POOL_WORKERS
    workers = int(workers)
    with _LOCK:
        if _POOL is not None and _POOL_WORKERS == workers:
            return _POOL
        _shutdown_locked()
        pool = ProcessPoolExecutor(max_workers=workers,
                                   mp_context=mp.get_context("spawn"))
        # start every worker now rather than on first use: tasks that each
        # hold a worker briefly force one process per task, so the pool's
        # start cost is paid here, once
        seen = set()
        for _attempt in range(3):
            seen |= set(pool.map(_ping, [0.2] * workers))
            if len(seen) >= workers:
                break
        _POOL, _POOL_WORKERS = pool, workers
        return pool


def _shutdown_locked():
    global _POOL, _POOL_WORKERS
    if _POOL is not None:
        _POOL.shutdown(wait=True, cancel_futures=True)
    _POOL, _POOL_WORKERS = None, 0


def shutdown_pool():
    """Stop the persistent pool (also registered to run at exit)."""
    with _LOCK:
        _shutdown_locked()


atexit.register(shutdown_pool)


# --------------------------------------------------------- shared memory

class _SharingPickler(pickle.Pickler):
    def __init__(self, fh, arrays):
        super().__init__(fh, protocol=pickle.HIGHEST_PROTOCOL)
        self.arrays = arrays
        self.index = {}

    def persistent_id(self, obj):
        if (type(obj) is np.ndarray and obj.nbytes >= SHARE_MIN_BYTES
                and not obj.dtype.hasobject):
            i = self.index.get(id(obj))
            if i is None:
                i = self.index[id(obj)] = len(self.arrays)
                self.arrays.append(obj)
            return ("shm", i)
        return None


class Shared:
    """One call's state, its large arrays in one shared-memory block.  Use as
    a context manager in the parent: the block is released on exit."""

    def __init__(self, obj):
        arrays = []
        fh = io.BytesIO()
        _SharingPickler(fh, arrays).dump(obj)
        self.payload = fh.getvalue()
        specs, off = [], 0
        for a in arrays:
            off = -(-off // 16) * 16
            order = "F" if (a.flags.f_contiguous and not a.flags.c_contiguous) else "C"
            specs.append((a.dtype.str, a.shape, off, order))
            off += a.nbytes
        self.shm = shared_memory.SharedMemory(create=True, size=max(off, 1))
        for a, (_dt, shape, o, order) in zip(arrays, specs):
            np.ndarray(shape, dtype=a.dtype, buffer=self.shm.buf, offset=o,
                       order=order)[...] = a
        self.specs = tuple(specs)
        self.token = uuid.uuid4().hex
        self.nbytes = off

    def ref(self):
        return (self.token, self.shm.name, self.payload, self.specs)

    def close(self):
        self.shm.close()
        try:
            self.shm.unlink()
        except FileNotFoundError:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


_ATTACHED = {}      # worker side: token -> (SharedMemory, state)


def _attach(ref):
    """Worker side: the call's state for `ref`, unpickled once per call."""
    token, name, payload, specs = ref
    hit = _ATTACHED.get(token)
    if hit is not None:
        return hit[1]
    for old in list(_ATTACHED):
        shm_old, _state = _ATTACHED.pop(old)
        del _state
        try:
            shm_old.close()
        except BufferError:
            pass            # a view from that call is still alive; freed at exit
    try:
        shm = shared_memory.SharedMemory(name=name, track=False)
    except TypeError:       # Python < 3.13: no `track`
        shm = shared_memory.SharedMemory(name=name)

    class _Unpickler(pickle.Unpickler):
        def persistent_load(self, pid):
            dt, shape, off, order = specs[pid[1]]
            arr = np.ndarray(shape, dtype=np.dtype(dt), buffer=shm.buf,
                             offset=off, order=order)
            arr.flags.writeable = False
            return arr

    state = _Unpickler(io.BytesIO(payload)).load()
    _ATTACHED[token] = (shm, state)
    return state


# --------------------------------------------------------------- renders

_MODEL_ARRAYS = ("grid", "grad_y", "grad_x")


def _open_shm(name):
    try:
        return shared_memory.SharedMemory(name=name, track=False)
    except TypeError:       # Python < 3.13: no `track`
        return shared_memory.SharedMemory(name=name)


def _render_task(ref, key, bin_px, out_name, slot, shape):
    """Render one model.  Its three arrays are written into slot `slot` of
    the parent's output block and the model comes back without them: 23 x
    7.7 MB of arrays through the result pipe cost more than the renders
    saved.  A model whose arrays are not float64 of `shape` comes back whole."""
    import dataclasses

    from .field_solve import _model_at
    m = _model_at(_attach(ref), key[0] * bin_px, key[1] * bin_px)
    arrays = [getattr(m, a) for a in _MODEL_ARRAYS]
    if any(a.shape != tuple(shape) or a.dtype != np.float64 for a in arrays):
        return m, False
    step = int(np.prod(shape)) * 8
    shm = _open_shm(out_name)
    try:
        for j, a in enumerate(arrays):
            np.ndarray(shape, dtype=np.float64, buffer=shm.buf,
                       offset=(slot * 3 + j) * step)[...] = a
    finally:
        shm.close()
    empty = np.empty((0, 0))
    return dataclasses.replace(m, **{a: empty for a in _MODEL_ARRAYS}), True


def render_models(epsf, keys, workers):
    """{key: `_model_at(epsf, bin centre of key)`} for the distinct `keys`, in
    first-appearance order, rendered in the pool when `workers > 1` (D.2).
    The same function on the same inputs as the serial render; the arrays
    return through shared memory and are copied out bit for bit."""
    import dataclasses

    from .field_solve import _model_at
    keys = list(dict.fromkeys(keys))
    bin_px = 1000.0 / float(epsf.plate_scale_mas)
    if int(workers) <= 1 or len(keys) <= 1:
        return {k: _model_at(epsf, k[0] * bin_px, k[1] * bin_px) for k in keys}
    light = copy.copy(epsf)
    light.__dict__.pop("_model_cache", None)
    grid_n = 2 * int(np.ceil(epsf.r_stamp_px)) * int(epsf.oversample) + 1
    shape = (grid_n, grid_n)
    step = grid_n * grid_n * 8
    out = shared_memory.SharedMemory(create=True, size=len(keys) * 3 * step)
    try:
        with Shared(light) as sh:
            pool = get_pool(workers)
            ref = sh.ref()
            futs = [pool.submit(_render_task, ref, k, bin_px, out.name, i, shape)
                    for i, k in enumerate(keys)]
            models = {}
            for i, (k, f) in enumerate(zip(keys, futs)):
                m, in_block = f.result()
                if in_block:
                    copied = {a: np.array(np.ndarray(shape, dtype=np.float64, buffer=out.buf,
                                                      offset=(i * 3 + j) * step))
                              for j, a in enumerate(_MODEL_ARRAYS)}
                    m = dataclasses.replace(m, **copied)
                models[k] = m
            return models
    finally:
        out.close()
        try:
            out.unlink()
        except FileNotFoundError:
            pass


# -------------------------------------------------------------- measure

def _measure_task(ref, pos):
    from .image_strehl import measure_strehl
    st = _attach(ref)
    return measure_strehl(st["image"], params=st["params"], pos=pos,
                          dl_psf=st["dl_psf"], **st["kw"])


class MeasureBatch:
    """`measure_strehl` at each of `positions`, computed in the pool and read
    back in position order (D.5).  `ahead=None` submits every position at
    once; otherwise asking for position i submits positions i .. i+ahead-1,
    so a caller that stops early wastes at most one batch."""

    def __init__(self, image, params, dl_psf, measure_kw, positions, workers,
                 ahead=None):
        self.positions = [tuple(p) for p in positions]
        self.shared = Shared({"image": np.asarray(image), "params": params,
                              "dl_psf": dl_psf, "kw": dict(measure_kw)})
        self.pool = get_pool(workers)
        self.ahead = ahead
        self.futs = {}
        if ahead is None:
            self._submit(0, len(self.positions))

    def _submit(self, lo, hi):
        ref = self.shared.ref()
        for i in range(lo, min(hi, len(self.positions))):
            if i not in self.futs:
                self.futs[i] = self.pool.submit(_measure_task, ref, self.positions[i])

    def result(self, i):
        if i not in self.futs:
            self._submit(i, i + (self.ahead or 1))
        return self.futs[i].result()

    def close(self):
        for f in self.futs.values():
            f.cancel()
        for f in self.futs.values():
            if not f.cancelled():
                try:
                    f.result()
                except Exception:
                    pass
        self.shared.close()
