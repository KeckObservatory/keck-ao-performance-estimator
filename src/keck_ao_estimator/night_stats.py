"""Summary statistics over a selectable period of the night, shared by the
GUI's Data-tab summary panel. Generalizes the exact period semantics
field_snapshot()/field_cn2_profile() (fieldmap.py) already use for the field
map -- 'window' (mean over the observing window, else whole night), 'night'
(whole night), 'time' (nearest single sample to a given HST datetime) -- to
an arbitrary datetime array, so the SAME period selection can be applied to
BOTH of compute_timeline's timebases (res.times: per-DIMM-sample NGS/DIMM/r0;
res.p_times: per-MASS-profile LGS/LTAO/MASS/theta0), not just p_times.

Qt-free, like fieldmap.py/catalogs.py.
"""
import numpy as np


def time_selection_mask(times, when, t_hst, prep):
    """Boolean mask over `times` (a list/array of datetimes, e.g. res.times
    or res.p_times) selecting the period requested by a Field-map-style
    Conditions selector:
      when="window" -> prep.in_any_window(t) for each t (needs prep.windows
                        and prep.show_target; else falls back to the whole
                        night, matching field_snapshot's own convention)
      when="night"  -> everything (whole night)
      when="time"   -> the single sample nearest t_hst (t_hst must be given;
                        falls back to the whole night if t_hst is None)
    Returns an all-False mask (not an error) if `times` is empty, so a caller
    can uniformly test `mask.any()`/index with it without a length check."""
    n = len(times)
    if n == 0:
        return np.zeros(0, dtype=bool)
    if when == "time" and t_hst is not None:
        idx = int(np.argmin([abs((t - t_hst).total_seconds()) for t in times]))
        sel = np.zeros(n, dtype=bool)
        sel[idx] = True
        return sel
    if when == "window" and getattr(prep, "show_target", False) and prep.windows:
        sel = np.array([prep.in_any_window(t) for t in times])
        if sel.any():
            return sel
    return np.ones(n, dtype=bool)


def masked_mean(values, mask):
    """mean of `values[mask]`, or None if the mask selects nothing, `values`
    is empty, or contains no finite values under the mask -- callers show
    '—' rather than crash or silently print NaN."""
    if values is None or len(values) == 0 or mask is None or not mask.any():
        return None
    v = np.asarray(values, dtype=float)[mask]
    v = v[np.isfinite(v)]
    return float(v.mean()) if v.size else None
