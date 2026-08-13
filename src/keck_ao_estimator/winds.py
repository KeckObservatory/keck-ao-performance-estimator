"""GFS upper-level winds over Mauna Kea, collapsed to the two representative
wind speeds the bandwidth term actually takes (budget.bw_wind_scale's
v_ground / v_free) -- so a night can run on the REAL wind profile instead of
the 8/25 m/s defaults. Grew out of the 2026-07-21 FA-event analysis
(FA_event_20260721_notes.md), where the Open-Meteo GFS pressure-level API
was validated against the Hilo radiosonde for this exact site: geopotential
heights agreed to a few meters, directions well, with GFS a little fast at
200 hPa.

Qt-free, like io.py/catalogs.py. Only _fetch_openmeteo() touches the
network -- one HTTP request, a HARD socket timeout (default 10 s), and NO
retry loop: a dead/slow service raises WindsError promptly and the caller
falls back to whatever winds it already had. Responses are cached per UT
date under the same cache dir as the MKWC files, with the same staleness
rule (a still-in-progress UT day is refetched; a finished one is trusted).

The collapse:
  v_free  = [ sum_i w_i v_i^(5/3) / sum_i w_i ]^(3/5)  over the six MASS bin
            altitudes, w_i = per-bin turbulence weights (the night's own
            median masspro J when available, else the measured median MK
            free-atmosphere shape MK_FA_PROFILE_FRAC) -- the same 5/3-moment
            weighting bw_wind_scale itself applies to the ground/FA split.
  v_ground= wind interpolated just above summit level.
Speeds interpolate u/v linearly in GEOPOTENTIAL height (not nominal
pressure altitudes), clamped to the ladder ends.
"""
import json
import os
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone

import numpy as np

from .atmosphere import seeing_to_integrated_cn2
from .constants import MASS_HEIGHTS_M
from .fieldmap import MK_FA_PROFILE_FRAC

# Mauna Kea summit (FA_event handoff values; alt = m ASL)
MK_LAT, MK_LON, MK_ALT_M = 19.8207, -155.4681, 4207.0

# pressure ladder: every level the 2026-07-22 validation exercised, plus
# 600/500/400 hPa to bracket the low MASS bins (0.5-2 km above summit)
_LEVELS_HPA = (600, 500, 400, 350, 300, 250, 200, 150, 100, 70, 50, 40, 30)

# a Keck observing night in UT hours of the MKWC file date (the morning/UT
# date): ~18:00-06:00 HST = 04-16 UT
_NIGHT_HOURS_UT = tuple(range(4, 17))


class WindsError(Exception):
    """GFS winds could not be fetched/parsed. Message is user-showable."""


def _cache_path(cache_dir, ymd):
    return os.path.join(cache_dir, f"gfs_winds_{ymd}.json")


def _ut_day_finished(ymd):
    """Same staleness rule as io.fetch_mkwc_files: trust a cached day only
    once that UT day has fully passed."""
    day = datetime.strptime(ymd, "%Y%m%d")
    now_utc = datetime.now(timezone.utc).replace(tzinfo=None)
    return now_utc >= day + timedelta(days=1)


# tried in order, ONE attempt each (not a retry loop -- they serve different
# date ranges): the live forecast API holds only the recent weeks of GFS;
# the historical-forecast API archives past GFS runs, so an archived night
# (e.g. re-analyzing 20260525 months later) comes from there
_HOSTS = ("https://api.open-meteo.com",
          "https://historical-forecast-api.open-meteo.com")


def _fetch_openmeteo(ymd, timeout, host):
    """One Open-Meteo GFS request for the whole UT day. Returns the decoded
    'hourly' dict. Raises WindsError on ANY failure -- network, HTTP, JSON,
    schema -- after at most `timeout` seconds; never retries."""
    day = f"{ymd[:4]}-{ymd[4:6]}-{ymd[6:]}"
    fields = ",".join(
        f"wind_speed_{p}hPa,wind_direction_{p}hPa,geopotential_height_{p}hPa"
        for p in _LEVELS_HPA)
    url = (f"{host}/v1/forecast"
           f"?latitude={MK_LAT}&longitude={MK_LON}&hourly={fields}"
           f"&wind_speed_unit=ms&start_date={day}&end_date={day}"
           "&models=ncep_gfs_seamless")
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, OSError,
            ValueError) as e:
        raise WindsError(f"GFS winds fetch failed ({type(e).__name__}: "
                         f"{e})") from None
    hourly = data.get("hourly")
    if not hourly or "time" not in hourly:
        raise WindsError("GFS winds fetch failed (response has no hourly "
                         "data — date out of the model's range?)")
    return hourly


def _load_hourly(ymd, cache_dir, timeout, refetch):
    """Cached-or-fetched 'hourly' dict for the UT day, VALIDATED to contain
    usable night-window wind levels before it is accepted/cached. At most
    one attempt per host in _HOSTS (each under the hard timeout), so the
    worst case is bounded at ~2x timeout -- never an unbounded hang."""
    path = _cache_path(cache_dir, ymd)
    if not refetch and os.path.exists(path) and _ut_day_finished(ymd):
        try:
            with open(path, encoding="utf-8") as fh:
                cached = json.load(fh)
            _profile_at_hours(cached, _NIGHT_HOURS_UT)
            return cached
        except (OSError, ValueError, WindsError):
            pass        # unreadable OR unusable (all-null) cache -> refetch
    last_err = None
    for host in _HOSTS:
        try:
            hourly = _fetch_openmeteo(ymd, timeout, host)
            _profile_at_hours(hourly, _NIGHT_HOURS_UT)   # usable? (all-null
        except WindsError as e:                          # = out of range)
            last_err = e
            continue
        try:
            os.makedirs(cache_dir, exist_ok=True)
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(hourly, fh)
        except OSError:
            pass                        # cache write failure is non-fatal
        return hourly
    raise last_err


def _profile_at_hours(hourly, hours_ut):
    """(heights_m[n_lvl], u[n_lvl], v[n_lvl]) per requested hour -> list.
    Levels with any missing value are dropped for that hour."""
    times = hourly["time"]
    idx = [i for i, t in enumerate(times)
           if int(t[11:13]) in hours_ut]
    if not idx:
        raise WindsError("GFS winds: no hourly samples in the night window")
    out = []
    for i in idx:
        hs, us, vs = [], [], []
        for p in _LEVELS_HPA:
            try:
                spd = hourly[f"wind_speed_{p}hPa"][i]
                direc = hourly[f"wind_direction_{p}hPa"][i]
                h = hourly[f"geopotential_height_{p}hPa"][i]
            except (KeyError, IndexError):
                continue
            if None in (spd, direc, h):
                continue
            th = np.radians(float(direc))
            # meteorological dir = FROM; u/v = TOWARD components
            hs.append(float(h))
            us.append(-float(spd) * np.sin(th))
            vs.append(-float(spd) * np.cos(th))
        if len(hs) >= 2:
            order = np.argsort(hs)
            out.append((np.array(hs)[order], np.array(us)[order],
                        np.array(vs)[order]))
    if not out:
        raise WindsError("GFS winds: response held no usable wind levels")
    return out


def _uv_at(profile, alt_m):
    """(u, v) wind components at altitude alt_m ASL: interpolated linearly
    in geopotential height, clamped to the ladder ends."""
    hs, us, vs = profile
    a = float(np.clip(alt_m, hs[0], hs[-1]))
    return float(np.interp(a, hs, us)), float(np.interp(a, hs, vs))


def _speed_at(profile, alt_m):
    return float(np.hypot(*_uv_at(profile, alt_m)))


def night_winds(ymd, cache_dir, fa_weights=None, timeout=10, refetch=False):
    """The night's representative winds from the GFS analysis, as a dict:
      v_ground : median wind just above summit level (m/s)
      v_free   : Cn2-weighted (5/3-moment) free-atmosphere wind (m/s)
      per_bin  : [(height_km_above_summit, median m/s), ...] for the 6 bins
      n_hours  : how many hourly GFS samples the medians cover
      hours    : human-readable window ("04-16 UT")
    fa_weights: 6 per-bin turbulence weights (e.g. the night's median
    masspro J values); default MK_FA_PROFILE_FRAC. Raises WindsError on any
    fetch/parse problem -- bounded by `timeout`, no retries, so a dead
    service can never hang the caller."""
    hourly = _load_hourly(ymd, cache_dir, timeout, refetch)
    profiles = _profile_at_hours(hourly, _NIGHT_HOURS_UT)
    w = np.asarray(fa_weights if fa_weights is not None
                   else MK_FA_PROFILE_FRAC, dtype=float)
    if w.shape != (6,) or not np.isfinite(w).all() or w.sum() <= 0:
        w = np.asarray(MK_FA_PROFILE_FRAC)
    w = w / w.sum()
    bin_alts = MASS_HEIGHTS_M + MK_ALT_M
    v_bins = np.array([[_speed_at(p, a) for a in bin_alts]
                       for p in profiles])            # (n_hours, 6)
    v_bin_med = np.median(v_bins, axis=0)
    v_free = float((w @ v_bin_med ** (5.0 / 3.0)) ** (3.0 / 5.0))
    v_ground = float(np.median([_speed_at(p, MK_ALT_M + 150.0)
                                for p in profiles]))
    # per-bin DIRECTION for the FA-timing advisory (event_lead_lag): the
    # median u/v vector's meteorological from-direction
    uv = np.array([[_uv_at(p, a) for a in bin_alts] for p in profiles])
    u_med, v_med = np.median(uv[:, :, 0], axis=0), np.median(uv[:, :, 1], axis=0)
    dir_from = (np.degrees(np.arctan2(-u_med, -v_med))) % 360.0
    return dict(
        v_ground=round(v_ground, 1), v_free=round(v_free, 1),
        per_bin=[(float(h / 1e3), round(float(v), 1))
                 for h, v in zip(MASS_HEIGHTS_M, v_bin_med)],
        bins_full=[(float(h), round(float(v), 1), round(float(d), 0))
                   for h, v, d in zip(MASS_HEIGHTS_M, v_bin_med, dir_from)],
        n_hours=len(profiles),
        hours=f"{_NIGHT_HOURS_UT[0]:02d}-{_NIGHT_HOURS_UT[-1]:02d} UT")


def tau0_seconds(eps_tot_500, eps_fa_500, v_ground, v_free):
    """Atmospheric coherence time tau0 = 0.314 r0_eff / V_eff (seconds) at
    500 nm, zenith: r0 from the TOTAL seeing, V_eff the same 5/3-moment
    Cn2-weighted wind bw_wind_scale uses, split ground/FA from the
    DIMM-vs-MASS seeing pair. None if the inputs can't support it."""
    if eps_tot_500 is None or eps_tot_500 <= 0 or v_ground <= 0 or v_free <= 0:
        return None
    j_tot = seeing_to_integrated_cn2(eps_tot_500)
    j_fa = (seeing_to_integrated_cn2(eps_fa_500)
            if eps_fa_500 is not None and eps_fa_500 > 0 else 0.0)
    j_fa = min(j_fa, j_tot)
    j_g = j_tot - j_fa
    v_eff = ((j_g * v_ground ** (5.0 / 3.0) + j_fa * v_free ** (5.0 / 3.0))
             / j_tot) ** (3.0 / 5.0)
    lam = 500e-9
    r0 = 0.98 * lam / np.radians(eps_tot_500 / 3600.0)
    return float(0.314 * r0 / v_eff)
