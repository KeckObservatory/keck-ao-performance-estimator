#!/usr/bin/env python3
"""The three on-axis Strehls -- field map, Data-tab summary stats, and the
Measured-SR tab's Predicted box -- are ONE model with three sets of inputs.

Eduardo, 2026-08-07: "When I match the field map and the summary stats to the
SR tool they give slightly different values for the on-axis SR. Why?"

This pins the answer so it cannot rot:

  (1) Given the SAME instant and the SAME geometry, all three agree EXACTLY.
      That is the load-bearing assertion -- if it ever fails, one of the three
      has grown its own physics and that is a bug, not a configuration.
  (2) Each remaining difference is an INPUT difference, and each is asserted
      here with its sign, so a refactor cannot quietly change which of the
      three is high:
        * period reduction  -- summary stats means the SERIES over the period,
          the field map evaluates ONE median-seeing representative sample
          (fieldmap.field_snapshot). mean(S) != S(median-seeing sample).
        * guide star        -- the SR tool re-evaluates at the FRAME's own
          resolved magnitude / TSS-odometer offset (Eduardo 2026-07-28).
        * wavelength        -- the SR tool works at the frame's EFFWAVE.
        * field centre      -- with an image loaded the map measures offsets
          from the IMAGE pointing, args.tt_offset from the typed target.
  (3) The laser marker is NOT a cause at the origin: the map's on-axis
      lgs_offset is |0 - laser|, which IS the LGS-offset magnitude by
      construction.

Headless and Qt-free: every path below is the exact engine call its widget
makes, cited in the comment above it.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
os.chdir(ROOT)
sys.path.insert(0, os.path.join(os.path.dirname(ROOT), "src"))
import warnings

warnings.filterwarnings("ignore")
import keck_ao_estimator as engine

np = engine.np
DATA = os.path.join(HERE, "data")
TOL = 1e-12                      # "exactly": same call, same floats


def prepared():
    from astropy.utils.iers import conf
    conf.auto_max_age = None     # see _run_tool.py
    args = engine.build_parser().parse_args([
        "--dimm", f"{DATA}/20260525_dimm.dat",
        "--mass", f"{DATA}/20260525_mass.dat",
        "--masspro", f"{DATA}/20260525_masspro.dat",
        "--telescope", "K1", "--target", "--target-name", "HD141569",
        "--ra", "15h49m57.7s", "--dec=-03d55m16s", "--window", "09:00-10:00",
        "--out", os.path.join(HERE, "onaxis_out.png"), "--force"])
    prep = engine.prepare_night(args)
    res = engine.compute_timeline(args, prep)
    return args, prep, res


# --- the three paths, each the engine call its widget makes ----------------

def stats_sr(args, prep, res, when, t):
    """gui/tabs/summary_stats.py::_refresh_summary_stats"""
    sel = engine.time_selection_mask(res.p_times, when, t, prep)
    return engine.masked_mean(res.sr_single, sel)


def fieldmap_sr(args, prep, res, when, t, tt_xy, laser_xy=None):
    """gui/tabs/fieldmap_tab.py::_render_field_map (on-axis = grid centre)"""
    snap = engine.field_snapshot(args, prep, res, when, t)
    if laser_xy is None:                    # fieldmap_tab.py::_laser_xy
        r = engine.DEF_LGS_OFFSET[args.telescope]
        pa = np.radians(225.0)
        laser_xy = (-r * np.sin(pa), r * np.cos(pa))
    val = engine.field_metric_at(args, prep, snap, "single", "strehl",
                                 (0.0, 0.0), tt_xy, laser_xy, (0.0, 0.0))
    return float(val), snap


def srtool_sr(args, res, t, lam_frame_nm, tt_mag, tt_offset):
    """gui/tabs/nirc2_strehl.py::_nirc2_compare (the LGS re-evaluation)"""
    offs = np.array([abs((tt - t).total_seconds()) for tt in res.p_times])
    i = int(offs.argmin())
    eps_tot_los = float(res.p_dimm_in[i] * res.p_zf[i])
    eps_fa_los = float(res.col_mass[i] * res.p_zf[i])
    return float(engine.lgs_strehl(
        eps_tot_los, eps_fa_los, args.telescope, "single", lam_frame_nm,
        tt_mag=tt_mag, tt_offset=tt_offset, lgs_offset=args.lgs_offset,
        legacy=args.legacy_budget,
        bw_factor=engine.ltao_bw_factor(args.ltao_bw_floor_frac),
        v_ground=args.wind_ground, v_free=args.wind_free,
        tt_sensor=getattr(args, "_tt_sensor_base", "strap")))


def one_model_three_inputs():
    args, prep, res = prepared()
    t_hst = res.p_times[len(res.p_times) // 2]
    # OffsetEntry guarantees |offset_xy| == value() -- both are measured from
    # the science target (gui/widgets.py::_compute), so the map's TT marker
    # sits at exactly args.tt_offset unless an image redefines the centre.
    tt_xy = (0.0, float(args.tt_offset))

    a = stats_sr(args, prep, res, "time", t_hst)
    b, snap = fieldmap_sr(args, prep, res, "time", t_hst, tt_xy)
    c = srtool_sr(args, res, t_hst, prep.lam_nm, args.tt_mag, args.tt_offset)
    assert abs(b - a) < TOL and abs(c - a) < TOL, (
        "the three on-axis paths have DIVERGED -- they must be one model:\n"
        f"  summary stats {a!r}\n  field map {b!r}\n  SR tool {c!r}")
    print(f"  [ok] same instant + same geometry -> all three agree: {a:.6f}")

    # (1) period reduction: mean of the series vs the median-seeing sample
    a_n = stats_sr(args, prep, res, "night", None)
    b_n, snap_n = fieldmap_sr(args, prep, res, "night", None, tt_xy)
    sel = engine.time_selection_mask(res.p_times, "night", None, prep)
    eps = np.asarray(res.p_dimm_in) * np.asarray(res.p_zf)
    assert sel.sum() > 1, "need a multi-sample night for this to mean anything"
    assert abs(b_n - a_n) > 1e-4, (
        "whole-night map and stats came out identical -- the mean-vs-"
        "representative-sample distinction this documents has gone away")
    # the map reads HIGH here because this night's seeing distribution is
    # right-skewed: its median is better than its mean (0.600" vs 0.649")
    assert np.median(eps[sel]) < eps[sel].mean(), (
        "night no longer right-skewed; the sign assertion below is only "
        "meaningful while it is")
    assert b_n > a_n, "median-seeing sample should read high on this night"
    print(f"  [ok] period reduction: stats(mean) {a_n:.6f} vs "
          f"map(median sample @ {snap_n['t_hst']:%H:%M}) {b_n:.6f} "
          f"-> {b_n - a_n:+.6f}")

    # (2) the SR tool's resolved guide star -- the biggest term in practice
    c_mag = srtool_sr(args, res, t_hst, prep.lam_nm, 11.0, args.tt_offset)
    assert c_mag > c, "a brighter TT star must not lower the Strehl"
    print(f"  [ok] resolved guide mag 11.0 vs {args.tt_mag:g}: "
          f"{c_mag - c:+.6f}")

    # (3) wavelength: bluer frame than the run -> lower Strehl
    c_lam = srtool_sr(args, res, t_hst, 2124.0, args.tt_mag, args.tt_offset)
    assert c_lam < c, "a shorter wavelength must not raise the Strehl"
    print(f"  [ok] frame EFFWAVE 2124 nm vs {prep.lam_nm:.0f} nm: "
          f"{c_lam - c:+.6f}")

    # (4) field centre: an image pointing 5" off moves the map's TT offset
    b_img, _ = fieldmap_sr(args, prep, res, "time", t_hst,
                           (0.0, float(args.tt_offset) - 5.0))
    assert b_img > b, "a CLOSER TT star must not lower the Strehl"
    print(f"  [ok] image centre 5\" from the typed target: {b_img - b:+.6f}")

    # (5) NOT a cause: at the origin the map's lgs_offset IS the magnitude
    for pa_deg in (0.0, 90.0, 225.0, 254.9):
        r = engine.DEF_LGS_OFFSET[args.telescope]
        pa = np.radians(pa_deg)
        b_pa, _ = fieldmap_sr(args, prep, res, "time", t_hst, tt_xy,
                              (-r * np.sin(pa), r * np.cos(pa)))
        assert abs(b_pa - b) < TOL, (
            f"laser PA {pa_deg} changed the ON-AXIS value -- it cannot: "
            "|0 - laser| is the LGS-offset magnitude at any PA")
    print("  [ok] laser PA does not move the on-axis value at any PA")


def main():
    one_model_three_inputs()


if __name__ == "__main__":
    main()
    print("  [ok] on-axis SR: one model, three input sets (2026-08-07)")
