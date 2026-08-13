"""Free-atmosphere timing advisory: how to READ the MASS feed, not a change
to any estimate. From the 2026-07-21 FA-event analysis
(FA_event_20260721_notes): a discrete FA event crossed Keck's beam 22-35
minutes BEFORE the MASS/DIMM recorded it, because the two instruments'
lines of sight pierce a moving layer kilometers apart -- the monitor's
instantaneous FA value is statistically representative of the summit, not
"what the beam sees right now", with a timing uncertainty of order
(pierce-point separation)/(layer wind).

Two pure helpers, both display-only (nothing here feeds the Strehl/FWHM
estimate or the timeline -- deliberately):

  trailing_fa_stats : median/range of the FA seeing over a trailing window,
                      with a "volatile" cue (an event is in progress, so a
                      point sample is least trustworthy exactly now).
  event_lead_lag    : per MASS layer, the SIGNED expected lead of Keck over
                      the monitor for an advecting structure -- the note's
                      §4 projection test run forward: pierce-point offset
                      projected onto the layer wind, with the monitor's own
                      (unknown, within ~h of zenith) pierce point as the
                      +/- band.

Qt-free, no I/O.
"""
import numpy as np


def trailing_fa_stats(p_times, fa_seeing, t_ref, window_min=40.0):
    """Median/range of `fa_seeing` (arcsec, MASS timebase `p_times`) over
    the trailing `window_min` minutes ending at datetime `t_ref`. Returns
    dict(n, med, lo, hi, volatile) or None when fewer than 2 samples land
    in the window (nothing meaningful to summarize).

    volatile: a display cue that an FA event is in progress -- the range is
    both large in absolute terms (>= 0.2") and the peak stands well above
    the window median (>= 1.5x). Checked against the 2026-07-21 event: the
    spike window (0.09-1.18", median ~0.45) trips it; the quiet hour before
    (0.09-0.36") does not. A heuristic for READING the feed, not physics."""
    if t_ref is None or len(p_times) == 0:
        return None
    sel = [i for i, t in enumerate(p_times)
           if 0.0 <= (t_ref - t).total_seconds() <= window_min * 60.0]
    if len(sel) < 2:
        return None
    v = np.asarray(fa_seeing, dtype=float)[sel]
    v = v[np.isfinite(v)]
    if v.size < 2:
        return None
    med, lo, hi = float(np.median(v)), float(v.min()), float(v.max())
    return dict(n=int(v.size), med=med, lo=lo, hi=hi,
                volatile=bool(hi - lo >= 0.2 and hi >= 1.5 * med))


def pierce_points(az_deg, elev_deg, heights_m):
    """Keck's line-of-sight pierce points at each layer height, as
    [(east_m, north_m), ...] horizontal offsets from the summit: the beam
    crosses layer h at d = h/tan(el) toward the target azimuth. Empty when
    the target is at/below the horizon. (The note's §4 geometry table:
    az 148 / el 41.6 puts the 16-km point 18.1 km out at (+9.5 E, -15.4 N).)"""
    if elev_deg is None or elev_deg <= 0.0:
        return []
    tan_el = np.tan(np.radians(elev_deg))
    out = []
    for h_m in heights_m:
        d = h_m / tan_el
        out.append((float(d * np.sin(np.radians(az_deg))),
                    float(d * np.cos(np.radians(az_deg)))))
    return out


def event_lead_lag(az_deg, elev_deg, bins, monitor_azel=None):
    """Per-layer signed lead of KECK over the monitor for an advecting
    structure, for a target at (az_deg, elev_deg) as seen from the summit.

    bins : [(h_above_summit_m, wind_speed_ms, wind_dir_from_deg), ...]
    Returns [(h_km, lead_center_min, half_range_min), ...] -- POSITIVE
    center = Keck's pierce point sits upwind, so Keck sees a layer-h event
    BEFORE the monitor by ~center +/- half_range minutes. Bins with no
    usable wind (< 0.5 m/s) or a target at/below the horizon are skipped.

    monitor_azel : None (default) -> the monitor's pierce point is unknown
    (its star sits anywhere within ~h of zenith), so the center assumes a
    zenith monitor and the +/- band is that ignorance, half_range = h/wind.
    Give an explicit (az, el) to PIN the monitor's pointing: the lead is
    then computed from its actual pierce point and half_range collapses to
    ~0 -- use this to test what monitor pointing reconciles an observed
    lead/lag (the monitor's true pointing is often unlogged, and at high
    layers the ignorance band spans tens of minutes).

    Geometry (the note's §4, forward): a pierce offset p crosses the wind's
    TOWARD vector t_hat at time (p . t_hat)/v; Keck's lead over the monitor
    is ((p_mon - p_keck) . t_hat)/v, with p_mon = 0 for a zenith monitor."""
    out = []
    if elev_deg is None or elev_deg <= 0.0:
        return out
    heights = [b[0] for b in bins]
    keck_pts = pierce_points(az_deg, elev_deg, heights)
    if monitor_azel is not None and monitor_azel[1] > 0.0:
        mon_pts = pierce_points(monitor_azel[0], monitor_azel[1], heights)
    else:
        mon_pts = [(0.0, 0.0)] * len(heights)      # zenith monitor
    for (h_m, v, dir_from), (kp_e, kp_n), (mp_e, mp_n) in zip(
            bins, keck_pts, mon_pts):
        if v is None or v < 0.5 or h_m <= 0:
            continue
        th = np.radians(dir_from)
        t_e, t_n = -np.sin(th), -np.cos(th)     # toward-vector (E, N)
        lead_s = ((mp_e - kp_e) * t_e + (mp_n - kp_n) * t_n) / v
        half = 0.0 if monitor_azel is not None else h_m / v
        out.append((float(h_m / 1e3), float(lead_s / 60.0),
                    float(half / 60.0)))
    return out
