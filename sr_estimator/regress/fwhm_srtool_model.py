#!/usr/bin/env python3
"""psf.fwhm_srtool_mas -- the FOURTH FWHM convention (2026-08-07).

Eduardo: "The FWHM estimation tends to overestimate what the measured SR
tool delivers. We need to add a new 4th FWHM estimation that uses exactly
the same process as the SR tool."

"Exactly the same process" is the contract this file guards, and the first
test is the load-bearing one: the convention must equal what you get by
rendering the model PSF as a frame and handing it to the Measured-SR tab's
OWN routine. If someone ever reimplements the binning "for speed" and it
drifts, this fails.

Also pinned:
  * the two mechanisms that make it differ from the half-max convention
    (annulus sky; pixel/annulus binning), each isolated;
  * the failure paths (bad seeing -> NaN, dead core -> seeing disk, and
    the tool's -1 sentinel never escaping as a negative FWHM);
  * plate-scale sensitivity, since the convention is defined ON a detector.

VALIDATION (not asserted here -- it needs proprietary frames; recorded so
the number is not lost). 2026-08-07, real NIRC2 frames measured with the
tool, each convention then evaluated AT THAT FRAME'S MEASURED STREHL so
only the convention is under test. Median (convention - measured), mas:

    set                       n    srtool  halfmax  gaussfit  gaussfit-sky
    20260727 o Her (isolated) 60    -0.37    -1.41     -3.84     -4.09
    20260730 GC (crowded)     45    -8.86    -9.99    -11.62    -12.58
    20260728 M92 (crowded)    18   -16.62   -17.71    -18.61    -20.26

srtool is the closest on EVERY set, by ~1 mas over half-max. Read the
o Her row as the convention test (isolated standard, well measured); the
crowded rows are dominated by blended-neighbour inflation of the MEASURED
FWHM, not by the convention -- their ORDERING is the informative part.
Note the sign: at a given Strehl the model reads slightly LOW, so the
several-mas overestimates in the predicted-vs-delivered tables are the
STREHL prediction, not the FWHM convention.
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
import numpy as np

from keck_ao_estimator.image_strehl import radial_profile_fwhm
from keck_ao_estimator.nirc2 import (NIRC2_BG_INNER_RADIUS_ARCSEC,
                                     NIRC2_BG_OUTER_RADIUS_ARCSEC,
                                     NIRC2_PLATE_SCALE_MAS)
from keck_ao_estimator.psf import _psf_profile, fwhm_srtool_mas, psf_fwhm_mas

PS = NIRC2_PLATE_SCALE_MAS["narrow"]
MAS_TO_RAD = 1e-3 / 206265.0
KW = dict(fit_nm=90.0, n_act=20.0)


def _reference(strehl, eps, lam, tt, half_px=60, subtract_sky=True):
    """The convention computed the SLOW, OBVIOUS way: build a real frame,
    hand it to the tool. Deliberately NOT sharing code with the
    implementation -- a bigger box, its own sky integral -- so agreement
    means something."""
    I, _aux = _psf_profile(strehl, eps, lam, tt, KW["fit_nm"], KW["n_act"])
    n = 2 * half_px + 1
    yy, xx = np.mgrid[0:n, 0:n]
    r_mas = np.hypot(xx - half_px, yy - half_px) * PS
    sky = 0.0
    if subtract_sky:
        r1 = NIRC2_BG_INNER_RADIUS_ARCSEC * 1000.0
        r2 = NIRC2_BG_OUTER_RADIUS_ARCSEC * 1000.0
        rq = np.linspace(r1, r2, 4096)          # 8x the implementation's grid
        sky = float(np.trapezoid(I(rq * MAS_TO_RAD) * rq, rq)
                    / (0.5 * (r2 ** 2 - r1 ** 2)))
    img = I(r_mas * MAS_TO_RAD) - sky
    return radial_profile_fwhm(img, float(half_px), float(half_px)) * PS


def same_process_as_the_tool():
    """THE contract: identical to rendering + calling the tool's routine."""
    worst = 0.0
    for eps in (0.4, 0.6, 0.9, 1.3):
        for S, tt in ((0.60, 12.0), (0.35, 20.0), (0.20, 35.0),
                      (0.10, 60.0), (0.05, 90.0)):
            got = fwhm_srtool_mas(S, eps, 2200.0, tt, plate_scale_mas=PS,
                                  **KW)
            ref = _reference(S, eps, 2200.0, tt)
            assert np.isfinite(got) and np.isfinite(ref), (S, eps, got, ref)
            worst = max(worst, abs(got - ref))
    assert worst < 1e-9, (
        f"fwhm_srtool_mas has DRIFTED from the tool's own routine by "
        f"{worst:.3g} mas -- it is supposed to BE that routine")
    print(f"  [ok] identical to render+radial_profile_fwhm (max |d| "
          f"{worst:.2e} mas, 20 cases)")


def the_two_mechanisms():
    """The two differences from the half-max convention, isolated, signed
    and SIZED. The sizes are the point: the mechanism everyone expects to
    matter (the annulus sky) does not, and the boring one (pixel binning)
    is the whole effect. Asserting that keeps a future 'optimization' that
    drops the binning from looking harmless."""
    S, eps, tt, lam = 0.35, 0.60, 20.0, 2200.0
    hm = psf_fwhm_mas(S, eps, lam, tt, **KW)
    with_sky = _reference(S, eps, lam, tt, subtract_sky=True)
    no_sky = _reference(S, eps, lam, tt, subtract_sky=False)

    # ANNULUS SKY: negligible. The 1.2-1.4" pedestal is <= ~2e-4 of the
    # peak on this PSF (and can be slightly negative at good Strehl -- the
    # finite-grid Hankel core undershooting), so it cannot move a half-max
    # crossing by a measurable amount. Included because it is what the tool
    # does; asserted here so nobody later "explains" a real discrepancy
    # with it.
    assert abs(with_sky - no_sky) < 0.05, (
        f"the annulus sky moved the FWHM by {with_sky - no_sky:+.3f} mas -- "
        "it is supposed to be negligible (<= 2e-4 of the peak); if this "
        "fires, the PSF's halo normalization has changed")
    print(f"  [ok] annulus sky is negligible: {with_sky - no_sky:+.4f} mas")

    # PIXEL/ANNULUS BINNING: the whole effect, and it BROADENS. The
    # innermost bin (r < 0.707 px) is a MEAN, so the apparent peak sits
    # below the true one, the half-max level drops with it, and the
    # crossing moves OUTWARD.
    assert no_sky > hm, (
        "1-px annulus averaging must BROADEN relative to a continuum "
        f"half-max (got {no_sky:.3f} binned vs {hm:.3f} continuum)")
    print(f"  [ok] pixel/annulus binning broadens it: {no_sky - hm:+.3f} mas")

    # net, on the core-dominated regime the tab actually operates in
    net = fwhm_srtool_mas(S, eps, lam, tt, plate_scale_mas=PS, **KW) - hm
    assert 0.5 < net < 3.0, (
        f"net convention offset {net:+.3f} mas is outside the measured "
        "~1.1 mas -- the validation table in this file's docstring assumed "
        "the old value")
    print(f"  [ok] net vs half-max: {net:+.3f} mas (WIDER, as measured)")


def guards_and_sentinels():
    """No negative FWHM, ever -- and the documented fallbacks."""
    assert np.isnan(fwhm_srtool_mas(0.3, -1.0, 2200.0, 20.0, **KW)), \
        "unusable seeing must give NaN"
    assert np.isnan(fwhm_srtool_mas(0.3, np.nan, 2200.0, 20.0, **KW)), \
        "NaN seeing must give NaN"
    seeing_disk = 0.80 * (2200.0 / 500.0) ** (-0.2) * 1000.0
    for dead in (0.0, -0.1, np.nan):
        got = fwhm_srtool_mas(dead, 0.80, 2200.0, 20.0, **KW)
        assert abs(got - seeing_disk) < 1e-6, (dead, got, seeing_disk)
    print("  [ok] NaN seeing -> NaN; dead core -> the seeing disk")

    # sweep hard and assert the -1 sentinel never leaks out as a FWHM
    for eps in np.linspace(0.2, 2.5, 24):
        for S in (0.9, 0.5, 0.25, 0.08, 0.02, 0.005):
            v = fwhm_srtool_mas(S, eps, 2200.0, 30.0, **KW)
            assert np.isnan(v) or v > 0.0, (
                f"NEGATIVE FWHM {v} leaked at S={S} eps={eps} -- the tool's "
                "-1.0 failure sentinel must be converted to NaN")
    print("  [ok] the tool's -1.0 sentinel never escapes as a FWHM (144)")


def plate_scale_matters():
    """It is defined ON a detector, so a coarser camera must read wider --
    and the default must be the narrow camera (what the tab is used with)."""
    S, eps, tt, lam = 0.35, 0.60, 2200.0, 20.0
    narrow = fwhm_srtool_mas(S, eps, tt, lam,
                             plate_scale_mas=NIRC2_PLATE_SCALE_MAS["narrow"],
                             **KW)
    wide = fwhm_srtool_mas(S, eps, tt, lam,
                           plate_scale_mas=NIRC2_PLATE_SCALE_MAS["wide"],
                           **KW)
    default = fwhm_srtool_mas(S, eps, tt, lam, **KW)
    assert abs(default - narrow) < 1e-12, \
        "the default plate scale must be the NIRC2 narrow camera"
    assert wide > narrow, (
        f"a 4x coarser camera must not read NARROWER ({wide:.2f} wide vs "
        f"{narrow:.2f} narrow)")
    print(f"  [ok] narrow {narrow:.2f} mas < wide {wide:.2f} mas; default "
          "is narrow")


def monotone_in_strehl():
    """Same physics contract the other three conventions hold to: worse
    Strehl must not give a TIGHTER core."""
    prev = None
    for S in (0.70, 0.60, 0.50, 0.40, 0.30, 0.20, 0.12, 0.07):
        v = fwhm_srtool_mas(S, 0.70, 2200.0, 25.0, **KW)
        if prev is not None:
            assert v >= prev - 1e-9, (
                f"FWHM went DOWN as the Strehl fell ({S}): {v} < {prev}")
        prev = v
    print("  [ok] monotone: falling Strehl never tightens the core")


def main():
    same_process_as_the_tool()
    the_two_mechanisms()
    guards_and_sentinels()
    plate_scale_matters()
    monotone_in_strehl()


if __name__ == "__main__":
    main()
    print("  [ok] fwhm_srtool_mas: the SR tool's own process (2026-08-07)")
