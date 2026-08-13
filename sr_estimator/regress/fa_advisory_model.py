#!/usr/bin/env python3
"""FA timing advisory (fa_advisory.py), headless, hand-checked against the
REAL 2026-07-21 free-atmosphere event (FA_event_20260721_notes): the
forward projection test must reproduce that analysis's verdict -- the
16-km bin on the stratospheric easterlies gives a Keck-first lead whose
window contains the observed 22-35 min, and the 8-km bin (ruled out in the
note) comes out monitor-first. Display-only helpers: nothing here touches
the estimate, so there is no harness/byte-identity surface at all."""
import os, sys
from datetime import datetime, timedelta
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE); sys.path.insert(0, ROOT); os.chdir(ROOT)
sys.path.insert(0, os.path.join(os.path.dirname(ROOT), "src"))
import warnings
warnings.filterwarnings("ignore")
import keck_ao_estimator as engine


def lead_lag_against_the_real_event():
    # M9Y6057 at 21:01 HST: elevation 41.6, azimuth 148 (note §4). Keck
    # 16-km pierce point: d = 16/tan(41.6) = 18.1 km toward SSE ->
    # (+9.59 E, -15.35 N) km, matching the note's (+9.5, -15.4).

    # 40 hPa GFS (14.64 m/s from 78): hand-computed lead center
    # -(p.t_hat)/v = +6.19 km / 14.64 = +7.0 min, band +/- 16/14.64 = 18.2
    (h, c, r), = engine.event_lead_lag(148.0, 41.6, [(16000.0, 14.64, 78.0)])
    assert h == 16.0
    assert abs(c - 7.0) < 0.3 and abs(r - 18.2) < 0.3, (c, r)

    # 50 hPa (12.62 m/s from 102): +16.6 +/- 21.1 min -- the observed
    # 22-35 min Keck-first lag falls inside center+range
    (_, c2, r2), = engine.event_lead_lag(148.0, 41.6, [(16000.0, 12.62, 102.0)])
    assert abs(c2 - 16.6) < 0.3 and abs(r2 - 21.1) < 0.3, (c2, r2)
    assert c2 + r2 >= 22.0, "the observed lag must be reachable"
    assert c2 > 0, "16-km easterlies must put Keck FIRST (the note's verdict)"

    # 8 km bin (200 hPa, 15.3 m/s from 259): the note RULED THIS OUT --
    # WSW flow puts Keck's SSE pierce point DOWNWIND -> monitor first
    (_, c3, _), = engine.event_lead_lag(148.0, 41.6, [(8000.0, 15.3, 259.0)])
    assert c3 < 0, f"8-km bin must be monitor-first (got {c3:+.1f} min)"

    # PINNED monitor pointing: giving an explicit (az, el) replaces the
    # ignorance band with a specific lead from the monitor's own pierce
    # point. A zenith monitor must reproduce the default (band -> 0), and
    # a monitor pointed AT the target must give ~0 lead (co-aligned sight
    # lines pierce the same point).
    (_, cz, rz), = engine.event_lead_lag(
        148.0, 41.6, [(16000.0, 14.64, 78.0)], monitor_azel=(0.0, 90.0))
    assert abs(cz - c) < 1e-6 and rz == 0.0, (cz, rz, c)
    (_, csame, _), = engine.event_lead_lag(
        148.0, 41.6, [(16000.0, 14.64, 78.0)], monitor_azel=(148.0, 41.6))
    assert abs(csame) < 1e-9, f"co-aligned monitor must give ~0 lead ({csame})"
    # the 16-km wind blows toward 258 deg (from 78), so an UPWIND monitor
    # (pointed ENE, ~78 deg) pierces the layer where the event is earlier
    # and sees it sooner than a DOWNWIND (WSW) monitor -> more-negative lead
    (_, c_up, _), = engine.event_lead_lag(
        148.0, 41.6, [(16000.0, 14.64, 78.0)], monitor_azel=(78.0, 55.0))
    (_, c_down, _), = engine.event_lead_lag(
        148.0, 41.6, [(16000.0, 14.64, 78.0)], monitor_azel=(258.0, 55.0))
    assert c_up < cz < c_down, (c_up, cz, c_down)

    # degenerate inputs: below horizon / dead calm -> skipped, never a crash
    assert engine.event_lead_lag(148.0, -5.0, [(16000, 10, 90)]) == []
    assert engine.event_lead_lag(148.0, 0.0, [(16000, 10, 90)]) == []
    assert engine.event_lead_lag(148.0, 41.6, [(16000, 0.1, 90)]) == []
    assert engine.event_lead_lag(148.0, 41.6, [(16000, None, 90)]) == []
    print("  [ok] projection test reproduces the 2026-07-21 event verdict: "
          f"16 km Keck-first ({c2:+.1f}±{r2:.1f} min contains the observed "
          f"22-35), 8 km monitor-first ({c3:+.1f}), degenerates skipped")


def pierce_point_table():
    """pierce_points against the note's §4 geometry table verbatim:
    M9Y6057 at el 41.6 az 148 -> 4 km: 4.5 km out; 8 km: 9.0; 16 km:
    18.1 km out at (dEast, dNorth) = (+9.5, -15.4) km."""
    pts = engine.pierce_points(148.0, 41.6, [4000.0, 8000.0, 16000.0])
    import math
    dist = [math.hypot(e, n) / 1e3 for e, n in pts]
    # exact: h/tan(41.6°) = 4.51/9.01/18.02 km -- the note's Figure 3a says
    # "18.0 km out" (its §4 table's 18.1 was a rounding inconsistency)
    assert abs(dist[0] - 4.5) < 0.05 and abs(dist[1] - 9.0) < 0.05 \
        and abs(dist[2] - 18.0) < 0.05, dist
    e16, n16 = pts[2]
    assert abs(e16 / 1e3 - 9.5) < 0.15 and abs(n16 / 1e3 - -15.4) < 0.15, pts
    assert engine.pierce_points(148.0, 0.0, [4000.0]) == []
    assert engine.pierce_points(148.0, None, [4000.0]) == []
    print("  [ok] pierce_points reproduces the note's geometry table "
          "(4.5 / 9.0 / 18.0 km out; 16 km at +9.5 E, -15.4 N)")


def trailing_stats_on_the_event_series():
    # the note's own §3 table around the spike
    base = datetime(2026, 7, 21, 20, 56)
    times = [base + timedelta(minutes=m) for m in (0, 7, 14, 24, 27, 32, 34, 39)]
    fa = [0.25, 0.09, 0.36, 0.55, 0.63, 0.87, 1.18, 0.85]

    spike = engine.trailing_fa_stats(times, fa, times[-1])
    assert spike["volatile"], "the event window must trip the cue"
    assert abs(spike["hi"] - 1.18) < 1e-9 and spike["n"] == 8

    quiet = engine.trailing_fa_stats(times[:3], fa[:3], times[2])
    assert quiet is not None and not quiet["volatile"], \
        "the quiet pre-event window must NOT trip the cue"

    # window edges: reference before the data / too few samples -> None
    assert engine.trailing_fa_stats(times, fa,
                                    times[0] - timedelta(hours=2)) is None
    assert engine.trailing_fa_stats(times[:1], fa[:1], times[0]) is None
    assert engine.trailing_fa_stats([], [], times[0]) is None
    assert engine.trailing_fa_stats(times, fa, None) is None

    # the window is TRAILING: samples after t_ref are excluded
    mid = engine.trailing_fa_stats(times, fa, times[2])
    assert mid["n"] == 3 and abs(mid["hi"] - 0.36) < 1e-9, mid
    print("  [ok] trailing stats: event window volatile, quiet window not, "
          "trailing-only selection, edge cases -> None")


def main():
    lead_lag_against_the_real_event()
    pierce_point_table()
    trailing_stats_on_the_event_series()


if __name__ == "__main__":
    main()
    print("  [ok] FA-advisory contract holds")
