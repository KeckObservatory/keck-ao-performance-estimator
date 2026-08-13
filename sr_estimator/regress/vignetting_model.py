#!/usr/bin/env python3
"""TSS reachability + vignetting (engine.vignetting, KAON 913) and the
outer-scale open-loop tilt ceiling (engine.psf, KAON 1318 Table 1).

Eduardo 2026-08-07: fold the K1 LBWFS/STRAP vignetting into guide-star
ranking with field-map feedback, and "take only what is useful from 1318".

What this file guards is mostly HONESTY, because the vignetting curve is a
2-parameter fit through 2 points and cannot be validated against anything
the note contains:

  * it must reproduce the three KAON 913 samples it was built from -- if the
    curve is ever re-fitted or the .dat files turn up, this fails loudly
    rather than the numbers quietly drifting;
  * the exact, non-negotiable parts (stage travel, the 60" radius, the
    instrument centres, the plate scale cross-check) must stay exact;
  * ranking must EXCLUDE an unreachable star -- the behaviour the change
    exists for;
  * vignetting must be charged as MAGNITUDES, once, and only to TT modes;
  * the 1318 ceiling must reproduce Table 1 at every tabulated L0 and keep
    the engine's own seeing scaling.
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
from keck_ao_estimator import psf, vignetting as V

np = engine.np


def exact_numbers_stay_exact():
    """The parts of KAON 913 that are measurements, not model."""
    assert V.TSS_TRAVEL_MM == {"x": (-44.0, 59.0), "y": (-69.0, 52.0)}
    assert V.TSS_UNVIGNETTED_RADIUS_MM == 43.6
    # the note's own scale, 43.6 mm <-> 60", must agree with the TSS odometer
    # constant derived independently in ttstar.py
    implied = 60.0 / V.TSS_UNVIGNETTED_RADIUS_MM
    assert abs(implied - engine.TSS_ARCSEC_PER_MM) < 0.002, (
        f"KAON 913 implies {implied:.4f} \"/mm but ttstar says "
        f"{engine.TSS_ARCSEC_PER_MM} -- one of them is wrong")
    print(f"  [ok] KAON 913 scale {implied:.4f} \"/mm confirms "
          f"ttstar.TSS_ARCSEC_PER_MM = {engine.TSS_ARCSEC_PER_MM}")

    # the rotation-invariant reduction: inscribed = 44 mm = 60.5", which
    # lands on the rotator's own 60" unvignetted radius from an independent
    # direction. That coincidence is what makes the radial model defensible;
    # if it ever stops holding, the model needs rethinking, not patching.
    assert abs(V.TSS_INSCRIBED_ARCSEC - V.UNVIGNETTED_RADIUS_ARCSEC) < 1.0, (
        f"inscribed travel {V.TSS_INSCRIBED_ARCSEC:.1f}\" no longer agrees "
        f"with the rotator's {V.UNVIGNETTED_RADIUS_ARCSEC:.1f}\" -- the "
        "radial reduction in vignetting.py assumed they coincide")
    assert V.TSS_CIRCUMSCRIBED_ARCSEC > V.TSS_INSCRIBED_ARCSEC
    print(f"  [ok] guaranteed {V.TSS_INSCRIBED_ARCSEC:.1f}\" vs rotator "
          f"{V.UNVIGNETTED_RADIUS_ARCSEC:.1f}\" (independent, and they agree)")

    # the instruments really do sit at different field centres
    d = V.instrument_centre_offset_arcsec("osiris-imager")
    assert d > 10.0, "the OSIRIS imager centre offset has gone missing"
    print(f"  [ok] instrument centres differ: imager {d:.1f}\" off the "
          "optical axis")


def curve_reproduces_its_anchors():
    """Two parameters, two points, zero degrees of freedom -- so the two
    points must come back exactly."""
    for r, want in V.VIGNETTE_SAMPLES:
        got = float(V.vignetting_fraction(r))
        assert abs(got - want) < 0.002, (
            f"the vignetting curve no longer reproduces KAON 913's own "
            f"{want:.0%} point at {r}\": got {got:.4f}")
    print("  [ok] reproduces all three KAON 913 samples "
          "(0% / 5% / 15% at 2.2 / 45.7 / 62.4\")")

    r = np.linspace(0.0, 150.0, 400)
    v = V.vignetting_fraction(r)
    assert np.all(np.diff(v) >= -1e-12), "vignetting must be monotonic"
    assert v[0] == 0.0 and v.max() <= 1.0
    # never claims total occultation inside the reachable field: a star goes
    # away by leaving the stage box, not by this curve saturating
    assert float(V.vignetting_fraction(V.TSS_CIRCUMSCRIBED_ARCSEC)) < 0.95
    print("  [ok] monotonic, 0 on axis, <1 everywhere the stage can reach")

    # magnitudes, not a second budget term
    for rr in (0.0, 30.0, 60.0, 80.0):
        v1 = float(V.vignetting_fraction(rr))
        dm = float(V.vignetting_mag_penalty(rr))
        assert abs(dm - (-2.5 * np.log10(1.0 - v1))) < 1e-9
    print("  [ok] the flux penalty is exactly -2.5log10(1-v)")


def reachability_gate():
    ok, cert, why = V.tss_reachable(10.0)
    assert ok and cert == "always" and why == ""
    ok, cert, why = V.tss_reachable(0.5 * (V.TSS_INSCRIBED_ARCSEC
                                           + V.TSS_CIRCUMSCRIBED_ARCSEC))
    assert ok and cert == "depends" and "rotator" in why
    ok, cert, why = V.tss_reachable(V.TSS_CIRCUMSCRIBED_ARCSEC + 5.0)
    assert not ok and cert == "never" and "cannot be placed" in why
    print("  [ok] always / depends / never, and 'depends' stays RANKABLE")

    # the exact box, for a caller that does know the bench frame
    assert V.tss_box_reachable(0.0, 0.0)
    assert V.tss_box_reachable(58.0, 51.0)
    assert not V.tss_box_reachable(60.0, 0.0)     # past +59 in x
    assert not V.tss_box_reachable(0.0, -70.0)    # past -69 in y
    print("  [ok] the asymmetric device-frame box is exact and available")


def ranking_uses_it():
    """The behaviour the change exists for: an unreachable star must not be
    ranked, and vignetting must be charged once, to TT modes only."""
    from astropy.utils.iers import conf
    conf.auto_max_age = None
    DATA = os.path.join(HERE, "data")
    args = engine.build_parser().parse_args([
        "--dimm", f"{DATA}/20260525_dimm.dat",
        "--mass", f"{DATA}/20260525_mass.dat",
        "--masspro", f"{DATA}/20260525_masspro.dat", "--telescope", "K1",
        "--out", os.path.join(HERE, "vig_out.png"), "--force"])
    prep = engine.prepare_night(args)
    res = engine.compute_timeline(args, prep)
    snap = engine.field_snapshot(args, prep, res, "night", None)

    def star(name, x, y, r=11.0):
        return {"id": name, "ra": 0.0, "dec": 0.0, "x": x, "y": y,
                "mags": {"R": r, "V": r + 0.3}}

    near = star("near", 10.0, 0.0)
    mid = star("mid", 70.0, 0.0)                       # 'depends' band
    far = star("far", 0.0, V.TSS_CIRCUMSCRIBED_ARCSEC + 20.0)

    got = {e["id"]: e for e in engine.rank_guide_stars(
        args, prep, snap, "single", [near, mid, far], (0.0, 0.0), "R")}
    assert got["far"]["rank"] is None and got["far"]["excluded_reason"], \
        "a star outside the TSS travel must be EXCLUDED, not ranked"
    assert "stage travel" in got["far"]["excluded_reason"]
    assert got["near"]["rank"] == 1, "the reachable near star should win"
    assert got["mid"]["rank"] is not None, \
        "'depends on the rotator' must stay rankable, not be silently dropped"
    assert got["mid"]["tss_certainty"] == "depends"
    print(f"  [ok] LGS ranking: far star excluded "
          f"({got['far']['excluded_reason'][:44]}...), mid star ranked with "
          "'depends'")

    # vignetting charged once, as magnitude, and ONLY in TT modes
    assert got["mid"]["vignette_frac"] > got["near"]["vignette_frac"] > -1e-9
    assert abs(got["mid"]["mag_effective"]
               - (got["mid"]["mag"] + got["mid"]["vignette_mag"])) < 1e-9, \
        "mag_effective must be mag + reddening + vignetting, charged once"
    print(f"  [ok] vignetting charged as magnitude: mid star "
          f"{got['mid']['mag']:.1f} + {got['mid']['vignette_mag']:.2f} = "
          f"{got['mid']['mag_effective']:.2f}")

    ngs = {e["id"]: e for e in engine.rank_guide_stars(
        args, prep, snap, "ngs", [near, mid, far], (0.0, 0.0), "R")}
    assert all(e["vignette_frac"] == 0.0 for e in ngs.values()), \
        "NGS mode must be EXEMPT -- that star is sensed on the HO WFS, not " \
        "through the TSS that KAON 913 measured"
    assert ngs["far"]["rank"] is not None, "NGS mode must not gate on TSS travel"
    print("  [ok] NGS mode exempt: no vignetting, no TSS gate")


def outer_scale_ceiling():
    """KAON 1318 Table 1, and the seeing scaling that must survive it."""
    table = dict(zip(psf._KAON1318_L0_M, psf._KAON1318_TT_TWOAXIS_MAS))
    for L0, want_two_axis in table.items():
        got = psf._open_loop_tilt_oneaxis_mas(engine.REF_TOTAL,
                                              outer_scale_m=L0)
        assert abs(got * np.sqrt(2.0) - want_two_axis) < 0.05, (
            f"L0={L0} m: ceiling {got * np.sqrt(2.0):.2f} mas two-axis but "
            f"KAON 1318 Table 1 says {want_two_axis}")
    print("  [ok] reproduces KAON 1318 Table 1 at all four tabulated L0")

    # infinite L0 must give back the pre-2026-08-07 Kolmogorov value, and it
    # must be HIGHER than every tabulated row -- that was the tell
    kolm = psf._open_loop_tilt_oneaxis_mas(engine.REF_TOTAL,
                                           outer_scale_m=float("inf"))
    assert abs(kolm - psf.OPEN_LOOP_TILT_ONEAXIS_MAS_KOLMOGOROV) < 1e-9
    assert kolm * np.sqrt(2.0) > max(table.values()), (
        "the infinite-L0 Kolmogorov ceiling should exceed every measured "
        "row -- if it does not, the reason for this change is gone")
    assert psf.OPEN_LOOP_TILT_ONEAXIS_MAS < kolm
    print(f"  [ok] default ceiling {psf.OPEN_LOOP_TILT_ONEAXIS_MAS:.1f} mas "
          f"< the old Kolmogorov {kolm:.1f} mas (which exceeded every row)")

    # the engine's seeing scaling must be untouched
    for eps in (0.4, 0.6, 0.9):
        s_tot = (eps / engine.REF_TOTAL) ** (5.0 / 6.0)
        got = psf._open_loop_tilt_oneaxis_mas(eps)
        assert abs(got / psf.OPEN_LOOP_TILT_ONEAXIS_MAS - s_tot) < 1e-9, \
            "the ceiling must still scale as s_tot ~ r0^(-5/6)"
    print("  [ok] still scales exactly as the budget's s_tot")

    # set_outer_scale must be VISIBLE to tiptilt (the stale-import hazard)
    from keck_ao_estimator import tiptilt
    try:
        wide = dict(s_tot=1.0, tt_mag=18.0, tt_offset=400.0)
        psf.set_outer_scale(10.0)
        lo = tiptilt.tt_wfe_nm(**wide)
        psf.set_outer_scale(100.0)
        hi = tiptilt.tt_wfe_nm(**wide)
        assert hi > lo * 1.5, (
            "tiptilt did not see set_outer_scale -- it is holding a stale "
            "`from .psf import OPEN_LOOP_TILT_ONEAXIS_MAS` snapshot; read it "
            "qualified (see psf.set_outer_scale)")
        print(f"  [ok] set_outer_scale reaches tiptilt live "
              f"({lo:.0f} -> {hi:.0f} nm at 400\")")
    finally:
        psf.set_outer_scale(25.0)


def main():
    exact_numbers_stay_exact()
    curve_reproduces_its_anchors()
    reachability_gate()
    ranking_uses_it()
    outer_scale_ceiling()


if __name__ == "__main__":
    main()
    print("  [ok] TSS vignetting model + outer-scale tilt ceiling (2026-08-07)")
