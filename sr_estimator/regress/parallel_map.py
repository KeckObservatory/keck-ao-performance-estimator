"""Ordered process-pool map for the regress batteries (parallel design block
D.5; T3).

`pmap(fn, items, workers=1, shared=None)` returns `[fn(item) for item in
items]` -- or `[fn(shared, item) ...]` when `shared` is given -- in item
order.  `workers <= 1` is exactly that list comprehension, in this process.
`workers=None` and the cap follow the engine (`resolve_workers`: D.7, PR-D4).
`workers > 1` runs the same calls in the engine's persistent pool
(`keck_ao_estimator.parallel`, PR-D10); `shared` then reaches the workers
once, through shared memory, not once per item.

`fn` must be importable by a spawned worker: a module-level function of an
importable module, or of a script guarded by `if __name__ == "__main__"`.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(os.path.dirname(HERE)), "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)


def _call_shared(fn, ref, item):
    from keck_ao_estimator.parallel import _attach
    return fn(_attach(ref), item)


def pmap(fn, items, workers=1, shared=None):
    from keck_ao_estimator.parallel import resolve_workers
    items = list(items)
    workers = resolve_workers(workers)
    if workers <= 1:
        if shared is None:
            return [fn(item) for item in items]
        return [fn(shared, item) for item in items]
    from keck_ao_estimator.parallel import Shared, get_pool
    pool = get_pool(workers)
    if shared is None:
        return list(pool.map(fn, items))
    with Shared(shared) as sh:
        ref = sh.ref()
        futs = [pool.submit(_call_shared, fn, ref, item) for item in items]
        return [f.result() for f in futs]
