"""The command-line interface: argument parsing (build_parser), the run
driver (main), and the zero-argument console entry point (_cli), shared
by the `keck-ao-estimator` console script and the harness-invoked script.
"""
import argparse
import os

import numpy as np

from ._version import __version__, APP_NAME
from .budget import (
    DEF_LGS_OFFSET, DEFAULT_BUDGET_VERSION, LTAO_BW_FLOOR_FRAC,
    LTAO_RATE_SINGLE, LTAO_RATE_TOMO, apply_budget_version, ltao_bw_factor,
)
from .constants import (
    DEF_AIRMASS_PAD, DEF_CACHE_DIR, DEF_DIMM, DEF_ELEV_CUT, DEF_MASS,
    DEF_MASSPRO, DEF_MATCH_TOL, DEF_NGS_BRIGHT, DEF_NGS_FAINT, DEF_TARGET_DEC,
    DEF_TARGET_NAME, DEF_TARGET_RA, DEF_TELESCOPE, DEF_WINDOWS,
    MOFFAT_BETA_KOLM, TEL_DIAMETER_M, V_FREE, V_GROUND,
)
from .export import write_csv_table
from .ngs import NGS_K1_QUADCELL_PENALTY, NGS_PARAMS, NGS_SEEING_LAW, NGS_SK_ANCHOR
from .pipeline import compute_timeline, prepare_night
from .plots import (
    overlay_fwhm_on_main, render_fwhm_figure, render_main_figure,
    render_terms_figure,
)
from .psf import OUTER_SCALE_M, set_outer_scale
from .tiptilt import (DEF_LTAO_TT_THETA0_GAIN, DEF_TT_MAG, DEF_TT_OFFSET,
                      NGS_TILT_SERVO_MAS)


def build_parser():
    """Construct the command-line interface. Run the script with -h to view."""
    # Custom formatter: keep epilog line breaks (Raw) AND show defaults.
    class _Fmt(argparse.RawDescriptionHelpFormatter,
               argparse.ArgumentDefaultsHelpFormatter):
        pass

    p = argparse.ArgumentParser(
        prog="keck-ao-estimator",
        formatter_class=_Fmt,
        description=(
            f"{APP_NAME} (v{__version__}). "
            "Estimate and plot NGS / single-beacon LGS / LTAO Strehl at a "
            "chosen science wavelength (K band by default) across a night "
            "from CFHT MASS/DIMM data, for W. M. Keck Observatory, with the "
            "science target's airmass overlaid. Uses the refined 2026-07 "
            "error budget (wind-weighted bandwidth, LTAO layer-mismatch "
            "penalty, parameterized TT star); --legacy-budget reverts."),
        epilog=(
            "EXAMPLES:\n"
            "  # default night, K2, tomography off (uses local files):\n"
            "  keck-ao-estimator\n\n"
            "  # download a night straight from the MKWC archive by date stamp:\n"
            "  # (the night date is derived from the data automatically)\n"
            "  keck-ao-estimator --fetch-date 20260525 \\\n"
            "      --out night_0524.png\n\n"
            "  # K1 (tomography on by default), custom target and window,\n"
            "  # science target as its own on-axis tip-tilt star (V=9.9):\n"
            "  keck-ao-estimator --telescope K1 \\\n"
            "      --target-name 'M31 nucleus' --ra 00h42m44.3s --dec +41d16m09s \\\n"
            "      --window 23:30-01:45 --tt-mag 9.9 --tt-offset 0\n\n"
            "  # different local data files and output name:\n"
            "  keck-ao-estimator --dimm n_dimm.dat --mass n_mass.dat \\\n"
            "      --masspro n_masspro.dat --out tonight.png\n\n"
            "  # project onto a line of sight 45 deg from zenith (airmass 1.41):\n"
            "  keck-ao-estimator --fetch-date 20260525 --zenith-angle 45\n\n"
            "NOTE ON DATES:\n"
            "  The MKWC files are named by the morning/UT date (e.g. a night that\n"
            "  begins the evening of 24 May 2026 is in the 20260525.* files), so\n"
            "  --fetch-date is that morning stamp. You normally do NOT need\n"
            "  --night: the civil evening date is read straight from the data's\n"
            "  timestamps. Pass --night only to override, and the script warns\n"
            "  if your override disagrees with the data.\n\n"
            "WINDOW FORMAT:\n"
            "  --window HH:MM-HH:MM  (HST clock times). A window whose start hour\n"
            "  is < 12 is treated as after midnight (next calendar day). Repeat\n"
            "  --window for multiple blocks."),
    )

    p.add_argument("--version", action="version",
                   version=f"{APP_NAME} {__version__}")

    g_in = p.add_argument_group("input data files")
    g_in.add_argument("--fetch-date", metavar="YYYYMMDD", default=None,
                      help="download the night's DIMM/MASS/MASSPRO files from "
                           "the MKWC archive for this file date stamp (the "
                           "morning/UT date, e.g. 20260525 for the night of "
                           "24 May 2026). Overrides --dimm/--mass/--masspro.")
    g_in.add_argument("--cache-dir", default=DEF_CACHE_DIR,
                      help="directory to store downloaded MKWC files")
    g_in.add_argument("--refetch", action="store_true",
                      help="always re-download, ignoring any cache (use when "
                           "re-running a night whose data was still growing)")
    g_in.add_argument("--no-refetch", dest="trust_cache", action="store_true",
                      help="always trust the cache if present, even for a night "
                           "that isn't over yet (skips the auto-refetch). By "
                           "default the cache is only trusted once the night's "
                           "UT date has fully passed.")
    g_in.add_argument("--dimm", default=DEF_DIMM,
                      help="DIMM total-seeing .dat file (ignored if --fetch-date)")
    g_in.add_argument("--mass", default=DEF_MASS,
                      help="MASS free-atmosphere-seeing .dat file (ignored if --fetch-date)")
    g_in.add_argument("--masspro", default=DEF_MASSPRO,
                      help="MASS profile .dat file (ignored if --fetch-date)")

    g_wl = p.add_argument_group("science wavelength")
    g_wl.add_argument("--band", default=None,
                      help="science band name (z, Y, J, H, K, Ks, L, M). Sets "
                           "the wavelength for the Strehl panels and theta0. "
                           "Default: K.")
    g_wl.add_argument("--wavelength", type=float, default=None, metavar="NM",
                      help="science wavelength in nm (overrides --band). "
                           "Affects NGS/LGS/LTAO Strehl and theta0; seeing and "
                           "d0 stay at their native definitions.")

    g_tel = p.add_argument_group("telescope / AO mode")
    g_tel.add_argument("--telescope", choices=["K1", "K2"], default=DEF_TELESCOPE,
                       help="Keck telescope. K2: HAKA-class (fitting 60 nm), "
                            "tomography OFF by default. K1: pre-HAKA RTC+OCAM "
                            "class (fitting 141 nm), tomography ON by default.")
    # Tomography default depends on the telescope and is resolved AFTER parsing
    # (see resolve_tomography): K2 -> off, K1 -> on. These two flags let you
    # force it either way explicitly.
    g_tel.add_argument("--tomography", dest="tomography", action="store_true",
                       default=None,
                       help="force the LTAO/tomography panel ON "
                            "(default: off for K2, on for K1)")
    g_tel.add_argument("--no-tomography", dest="tomography", action="store_false",
                       help="force the LTAO/tomography panel OFF")

    g_tgt = p.add_argument_group("science target (airmass overlay) — OFF by default")
    g_tgt.add_argument("--target", dest="show_target", action="store_true",
                       help="draw the target airmass curve AND the observing-"
                            "window boxes. Off by default; without it, no "
                            "airmass, no window boxes, no target.")
    g_tgt.add_argument("--target-name", default=DEF_TARGET_NAME,
                       help="target label for the airmass curve and title")
    g_tgt.add_argument("--ra", default=DEF_TARGET_RA,
                       help="target Right Ascension (any astropy-parseable string)")
    g_tgt.add_argument("--dec", default=DEF_TARGET_DEC,
                       help="target Declination (any astropy-parseable string)")

    g_time = p.add_argument_group("night / observing windows")
    g_time.add_argument("--night", default=None,
                        help="civil (evening) date in HST, YYYY-MM-DD. If "
                             "omitted (recommended), it is derived from the "
                             "data's own timestamps, which avoids any "
                             "file-date/night mismatch.")
    g_time.add_argument("--window", action="append", default=None,
                        metavar="HH:MM-HH:MM",
                        help="observing window in HST; repeat for several. Only "
                             "drawn when --target is given (default: %s)"
                             % DEF_WINDOWS[0])

    g_ngs = p.add_argument_group("NGS guide-star magnitudes")
    g_ngs.add_argument("--ngs-bright", type=float, default=DEF_NGS_BRIGHT,
                       help="bright NGS magnitude (seeing-limited ceiling)")
    g_ngs.add_argument("--ngs-faint", type=float, default=DEF_NGS_FAINT,
                       help="faint NGS magnitude (shows SNR roll-off)")
    g_ngs.add_argument("--ngs-offset", type=float, default=0.0, metavar="ARCSEC",
                       help="NGS offset from the science target, arcsec "
                            "(default 0 = on-axis). When nonzero, both NGS "
                            "curves are multiplied by the angular-"
                            "anisoplanatism Strehl exp(-(theta/theta0)^(5/3)) "
                            "using the per-sample theta0 from the nearest MASS "
                            "profile within --match-tol; NGS points with no "
                            "profile in tolerance become gaps, matching the "
                            "coverage honesty of the LGS/theta0 panels "
                            "(unless --assumed-theta0 provides a fallback).")
    g_ngs.add_argument("--assumed-theta0", type=float, default=15.0,
                       metavar="ARCSEC",
                       help="fallback isoplanatic angle for the off-axis NGS "
                            "correction, given as the K-BAND ZENITH value. "
                            "Default 15\" (the Maunakea median): wherever no "
                            "MASS profile is within --match-tol (including "
                            "whole nights with no masspro data) the assumed "
                            "value is used, scaled internally to the science "
                            "wavelength (lambda^1.2) and to each sample's "
                            "line of sight; real MASS theta0 always wins "
                            "where available. Assumed-theta0 samples are "
                            "drawn as TRIANGLES in the NGS panel (circles = "
                            "MASS theta0), recorded in the CSV "
                            "(ngs_theta0_arcsec), and counted in the plot "
                            "annotation. Set <= 0 to disable the fallback "
                            "and leave honest gaps instead.")
    g_ngs.add_argument("--ngs-seeing-law", choices=["kolmogorov", "gaussian"],
                       default=NGS_SEEING_LAW,
                       help="seeing law inside the NGS Gompertz fit. The fit "
                            "was calibrated on open-loop K-band FWHM of "
                            "~0.19-0.38\" (ZA 0-30); its original 'gaussian' "
                            "exp(-A s^2) term free-falls when the (airmass-"
                            "projected) seeing extrapolates beyond that range, "
                            "under-predicting high-airmass NGS by 0.1-0.2 "
                            "Strehl (2026-07-06 HIP 88553: delivered ~0.50-0.56 "
                            "at airmass 2.34 vs 0.29-0.42 predicted). Default "
                            "'kolmogorov' uses the physical variance ~ s^(5/3) "
                            "scaling, re-anchored at the calibration mid-range "
                            "(sK=%g\"): identical inside the calibrated range "
                            "(<0.005 Strehl), physically-paced roll-off beyond "
                            "it." % NGS_SK_ANCHOR)

    # NGS Gompertz-fit overrides for the ACTIVE telescope (recalibration of the
    # empirical on-sky fit, not a what-if): S = S0 * exp(-A sK^2) *
    # exp(-exp((R-m0)/w)). Default None -> the telescope's fitted value
    # (K2: S0=%g A=%g m0=%g w=%g; K1: S0=%g A=%g m0=%g w=%g).
    _gk, _g1 = NGS_PARAMS["K2"], NGS_PARAMS["K1"]
    g_ngs.add_argument("--ngs-s0", type=float, default=None, metavar="S0",
                       help="override the NGS Gompertz bright-star ceiling S0 "
                            "for the active telescope (K2 %g / K1 %g)."
                            % (_gk["S0"], _g1["S0"]))
    g_ngs.add_argument("--ngs-a", type=float, default=None, metavar="A",
                       help="override the NGS Gompertz seeing exponent A "
                            "(in exp(-A sK^2); K2 %g / K1 %g -- K1 steeper for "
                            "DM-stroke saturation)." % (_gk["A"], _g1["A"]))
    g_ngs.add_argument("--ngs-m0", type=float, default=None, metavar="M0",
                       help="override the NGS Gompertz faint-end midpoint m0 "
                            "(R mag; K2 %g / K1 %g)." % (_gk["m0"], _g1["m0"]))
    g_ngs.add_argument("--ngs-w", type=float, default=None, metavar="W",
                       help="override the NGS Gompertz roll-off width w "
                            "(K2 %g / K1 %g)." % (_gk["w"], _g1["w"]))
    g_ngs.add_argument("--k1-quadcell-penalty", type=float,
                       default=NGS_K1_QUADCELL_PENALTY, metavar="STREHL",
                       help="K1-only: flat Strehl subtracted for the KAPA PRO "
                            "quadcell-saturation effect (default %g, i.e. the "
                            "~5-point historical under-performance vs K2). "
                            "Set 0 to remove it; ignored on K2."
                            % NGS_K1_QUADCELL_PENALTY)

    g_tt = p.add_argument_group("LGS/LTAO tip-tilt star")
    g_tt.add_argument("--tt-sensor", default="strap",
                      choices=["strap", "trick-h", "trick-k"],
                      help="tip-tilt sensor: 'strap' = R-band quadcell with "
                           "the refined measurement row (recalibrated to the "
                           "paired on-sky STRAP/TRICK data; --tt-mag is R); "
                           "'trick-h'/'trick-k' = "
                           "the K1 IR sensor in H/K (--tt-mag is the guide's "
                           "H/K magnitude). On K1 TRICK and OSIRIS split the "
                           "dichroic, so trick-k forces science H and trick-h "
                           "forces science K (default: strap).")
    g_tt.add_argument("--tt-mag", type=float, default=DEF_TT_MAG,
                      help="magnitude of the tip-tilt star in the SENSING band "
                           "(R for strap; H or K for trick) used in the LGS/"
                           "LTAO TT budget (default: %.1f, the budgeted star)"
                           % DEF_TT_MAG)
    g_tt.add_argument("--tt-offset", type=float, default=DEF_TT_OFFSET,
                      help="TT-star offset from the science target in arcsec; "
                           "0 = science target is its own TT reference "
                           "(default: %.1f\", the budgeted star)" % DEF_TT_OFFSET)
    g_tt.add_argument("--lgs-offset", type=float, default=None,
                      metavar="ARCSEC",
                      help="beacon (single-LGS) / asterism-center (LTAO) "
                           "offset from the science direction, arcsec, for "
                           "the angular-anisoplanatism term. Default: the "
                           "telescope's operational offset (K1: %.2f\", "
                           "K2: %.0f\"); the original 44 nm allocation "
                           "assumed 2\"." % (DEF_LGS_OFFSET["K1"],
                                             DEF_LGS_OFFSET["K2"]))
    g_tt.add_argument("--ltao-bw-floor-frac", type=float,
                      default=LTAO_BW_FLOOR_FRAC, metavar="FRAC",
                      help="fraction (0-1) of the single-beacon bandwidth "
                           "VARIANCE that is a fixed loop-latency floor, "
                           "independent of frame rate, when computing the "
                           "slower-rate LTAO bandwidth penalty. The LTAO HO "
                           "loop runs at %g Hz vs %g Hz single-beacon; a pure "
                           "frame-rate law would penalize by (%.2g)^(5/6)=%.2f, "
                           "but a latency floor damps this. Default %.2f "
                           "(NGS-budget-consistent) gives an effective factor "
                           "of %.2f; 0 recovers the pure rate law (%.2f), 1 "
                           "removes the penalty."
                           % (LTAO_RATE_TOMO, LTAO_RATE_SINGLE,
                              LTAO_RATE_SINGLE/LTAO_RATE_TOMO,
                              (LTAO_RATE_SINGLE/LTAO_RATE_TOMO)**(5/6),
                              LTAO_BW_FLOOR_FRAC, ltao_bw_factor(),
                              ltao_bw_factor(0.0)))
    g_tt.add_argument("--wind-ground", type=float, default=V_GROUND,
                      metavar="M_S",
                      help="representative boundary-layer (ground) wind speed "
                           "in m/s for the wind-weighted bandwidth term. "
                           "Default %g. We cannot read the true profile, so "
                           "supply a night estimate (e.g. from a GFS forecast) "
                           "here; ignored under --legacy-budget." % V_GROUND)
    g_tt.add_argument("--wind-free", type=float, default=V_FREE, metavar="M_S",
                      help="representative free-atmosphere (jet) wind speed in "
                           "m/s for the wind-weighted bandwidth term. Default "
                           "%g. Higher free-atm wind raises the bandwidth "
                           "surcharge on free-atm-dominated nights; ignored "
                           "under --legacy-budget." % V_FREE)
    g_tt.add_argument("--legacy-budget", action="store_true",
                      help="disable the 2026-07 budget refinements (wind-"
                           "weighted bandwidth, tomography layer-mismatch "
                           "penalty) and reproduce older runs. The TT star "
                           "still honors --tt-mag/--tt-offset, whose defaults "
                           "match the legacy TT term to <1%%.")
    g_tt.add_argument("--budget-version", choices=["3_1_3", "3_1_1"],
                      default=DEFAULT_BUDGET_VERSION,
                      help="reference-value set for the error budget: 3_1_3 "
                           "(default; AO performance budget v3_1_3, "
                           "zenith-referenced from its ZA=50 column) or "
                           "3_1_1 (the previous baseline values, which "
                           "carried the ZA-50 scaling inside and "
                           "double-counted zenith degradation). Orthogonal "
                           "to --legacy-budget, which switches model "
                           "refinements, not values.")

    g_zen = p.add_argument_group("zenith angle (line-of-sight projection)")
    g_zen.add_argument("--zenith-angle", type=float, default=0.0,
                       metavar="DEG",
                       help="science-target zenith angle in degrees. MASS/DIMM "
                            "seeing is reported corrected to zenith; this "
                            "projects that atmosphere onto the target's actual "
                            "line of sight. Seeing grows as airmass^(3/5), which "
                            "worsens the focal aniso (cone effect) and shrinks theta0. If "
                            "--target is given, zeta is instead computed PER "
                            "SAMPLE from the target's own airmass and this fixed "
                            "value is ignored. (default: 0 = zenith)")

    g_air = p.add_argument_group("airmass curve")
    g_air.add_argument("--elev-cut", type=float, default=DEF_ELEV_CUT,
                       help="(legacy) simple elevation floor. Now SUPERSEDED by "
                            "the per-telescope, azimuth-dependent Keck pointing "
                            "limits, which the airmass plot uses automatically.")
    g_air.add_argument("--airmass-pad", type=float, default=DEF_AIRMASS_PAD,
                       help="half-height (airmass units) of the centered y-range")
    g_air.add_argument("--no-airmass-center", dest="airmass_center",
                       action="store_false",
                       help="do not center the lowest airmass at mid-panel")
    p.set_defaults(airmass_center=True)

    g_misc = p.add_argument_group("misc")
    g_misc.add_argument("--report", choices=["strehl", "fwhm", "both"],
                        default="strehl",
                        help="performance metric to report. 'strehl' (default) "
                             "is the original Strehl-ratio product. 'fwhm' "
                             "replaces the main figure with a FWHM timeline "
                             "(mas), from a per-sample Airy-core + Moffat-halo "
                             "PSF model (core energy = Strehl, halo = seeing "
                             "disk; D=%g m, beta=%g). 'both' keeps the Strehl "
                             "figure and overlays FWHM on a right-hand axis. "
                             "fwhm/both also append FWHM columns to the CSV."
                             % (TEL_DIAMETER_M, MOFFAT_BETA_KOLM))
    g_misc.add_argument("--fwhm-curves",
                        choices=["srtool", "halfmax", "gaussfit",
                                 "gaussfit-sky", "both", "all"],
                        default="srtool",
                        help="which FWHM convention(s) the fwhm/both figures "
                             "plot. 'srtool' (DEFAULT) is what THIS package's "
                             "own Measured-SR tab reads off the same PSF -- "
                             "its find_fwhm.pro port run on a rendered NIRC2 "
                             "frame, sky-subtracted; the only convention "
                             "directly comparable to a MEASURED FWHM, and the "
                             "closest to what the tab delivers (median error "
                             "-0.4 mas vs -1.4 for half-max over 60 "
                             "isolated-standard frames), so it is what a "
                             "predicted-vs-delivered join should use. See "
                             "fwhm_srtool_mas. The others: 'halfmax' (the "
                             "core+halo model, half-max crossing -- no "
                             "confirmed real-tool analog, see psf_fwhm_mas), "
                             "'gaussfit' (no-"
                             "background Gaussian LSQ fit, models the OSIRIS "
                             "quicklook tool's rarely-used Strehl button, "
                             "OSIRISSTREHL_QL2.pro), 'gaussfit-sky' "
                             "(free-background Gaussian LSQ fit, models the "
                             "OSIRIS quicklook tool's hand-drawn-box fit "
                             "feature -- a separate, independent tool from "
                             "the AO Strehl tool), 'both' "
                             "(halfmax+gaussfit, back-compat), or 'all' (all "
                             "four). Plot only; the CSV always carries all "
                             "four sets of columns.")
    g_tt.add_argument("--outer-scale", type=float, default=None,
                      metavar="METRES",
                      help="atmospheric outer scale L0 (m) for the open-loop "
                           "tilt CEILING -- the bound 'a tilt loop cannot do "
                           "worse than no correction at all'. Anchored to "
                           "KAON 1318 Table 1 (uncorrected TT vs L0, "
                           "10-100 m); the default %g m is the Mauna Kea "
                           "MEDIAN both KAON 1318 (Fig. 5) and KAON 1303 "
                           "(sect. 5.5) state. Pass "
                           "'inf' for the pre-2026-08 infinite-outer-scale "
                           "Kolmogorov ceiling (~110 mas one-axis), which "
                           "overstates tilt because tilt is the most "
                           "outer-scale-sensitive mode there is. Only binds "
                           "far off-axis / for faint TT stars."
                           % OUTER_SCALE_M)
    g_misc.add_argument("--fwhm-box-mas", type=float, default=300.0,
                        metavar="MAS",
                        help="fit-domain radius (mas) for BOTH Gaussian-fit "
                             "FWHM conventions ('gaussfit' and 'gaussfit-"
                             "sky'). The OSIRIS quicklook tool's hand-drawn-"
                             "box fit feature has its box drawn by hand with "
                             "the mouse, so there is no single correct value "
                             "-- this is a real, explorable parameter, not a "
                             "calibrated constant. Default 300 preserves the "
                             "already-validated 20260701 numbers; "
                             "OSIRISSTREHL_QL2.pro's own auto-sized box "
                             "works out to ~30.7 mas at K band.")
    g_misc.add_argument("--ngs-tilt-servo", type=float,
                        default=NGS_TILT_SERVO_MAS, metavar="MAS",
                        help="FWHM-path only: one-axis NGS atmospheric "
                             "tilt-servo residual (mas) at the reference "
                             "profile, scaled by s_tot. tt_wfe_nm has no "
                             "seeing-scaling tilt on axis, so without this the "
                             "modelled NGS FWHM is flat. Default %g reproduces "
                             "~52 mas at K-band seeing 0.6\". Never enters the "
                             "Strehl budget." % NGS_TILT_SERVO_MAS)
    g_misc.add_argument("--ltao-tt-theta0-gain", type=float,
                        default=DEF_LTAO_TT_THETA0_GAIN, metavar="G",
                        help="effective tilt-theta0 gain from tomography, "
                             "applied as (1/G)^(5/6) on the TT-star tilt-"
                             "anisoplanatism rows AND the angular-aniso "
                             "charge at the laser/asterism-centre offset, "
                             "in LTAO mode only (never single-beacon, never "
                             "--legacy). Default %g = "
                             "disabled: KAON 1303 section 5.5 shows the "
                             "LGS-tomography null modes (field-varying tilt "
                             "from quadratic modes aloft) are NOT estimable "
                             "with a single TT star, so single-star LTAO "
                             "gets no tilt-aniso reduction. Raise above 1.0 "
                             "only for a multi-star mode or contrary on-sky "
                             "calibration." % DEF_LTAO_TT_THETA0_GAIN)
    g_misc.add_argument("--match-tol", type=int, default=DEF_MATCH_TOL,
                        help="max |dt| (s) when matching DIMM to a MASS sample")
    g_misc.add_argument("--out", default=None,
                        help="output PNG filename. Default: auto-named from the "
                             "data's UT date and telescope, e.g. "
                             "ao_strehl_20260525_K1.png")
    g_misc.add_argument("--no-combined", action="store_true",
                        help="skip the combined single-file figure "
                             "(<UT>_<tel>_all.png) that stacks the main "
                             "timeline and the error-terms figure into one "
                             "image for quick one-file night review")
    g_misc.add_argument("--no-terms-plot", action="store_true",
                        help="skip the companion error-terms figure "
                             "(<out>_terms.png), which plots the non-static "
                             "budget terms (nm RMS) vs time whenever MASS "
                             "profiles are available")
    g_misc.add_argument("--force", action="store_true",
                        help="overwrite the output file if it already exists "
                             "(default: refuse and exit)")

    return p


def main(args):
    # value-set selection first: everything downstream reads the module
    # globals (the GUI never passes through here -- its version picker goes
    # through the slider/override machinery instead)
    apply_budget_version(getattr(args, "budget_version",
                                 DEFAULT_BUDGET_VERSION))
    # outer scale: rebinds psf.OPEN_LOOP_TILT_ONEAXIS_MAS, which tiptilt reads
    # qualified (see psf.set_outer_scale's warning)
    if getattr(args, "outer_scale", None) is not None:
        set_outer_scale(args.outer_scale)
    prep = prepare_night(args)
    res  = compute_timeline(args, prep)
    _ltao_bw_fac = prep._ltao_bw_fac
    baseline_zen_factor = prep.baseline_zen_factor
    dimm_dt = prep.dimm_dt
    dimm_sec = prep.dimm_sec
    dimm_see = prep.dimm_see
    fixed_zen_factor = prep.fixed_zen_factor
    in_any_window = prep.in_any_window
    lam_label = prep.lam_label
    lam_nm = prep.lam_nm
    mass_dt = prep.mass_dt
    mass_sec = prep.mass_sec
    mass_see = prep.mass_see
    night_date = prep.night_date
    out_path = prep.out_path
    profiles = prep.profiles
    show_target = prep.show_target
    tomography_on = prep.tomography_on
    ut_stamp = prep.ut_stamp
    windows = prep.windows
    zen_factor_by_time = prep.zen_factor_by_time
    col_airmass = res.col_airmass
    col_ang1 = res.col_ang1
    col_d0 = res.col_d0
    col_dimm = res.col_dimm
    col_mass = res.col_mass
    col_mm = res.col_mm
    col_ngs_th0 = res.col_ngs_th0
    col_r0_cm = res.col_r0_cm
    col_terms = res.col_terms
    col_theta0 = res.col_theta0
    col_tt10 = res.col_tt10
    col_zf = res.col_zf
    n_fb = res.n_fb
    ngs_bright = res.ngs_bright
    ngs_faint = res.ngs_faint
    ngs_fb = res.ngs_fb
    p_airmass = res.p_airmass
    p_dimm_in = res.p_dimm_in
    p_secs = res.p_secs
    p_times = res.p_times
    p_zf = res.p_zf
    sr_ltao = res.sr_ltao
    sr_single = res.sr_single
    th0_assumed = res.th0_assumed
    times = res.times
    # --- console summary ------------------------------------------------------
    def stats(label, a):
        a = a[~np.isnan(a)]
        if a.size == 0:
            print(f"  {label:24s} (no data)"); return
        print(f"  {label:24s} mean {a.mean():.3f}  min {a.min():.3f}  "
              f"max {a.max():.3f}  median {np.median(a):.3f}")
    n_mass = len(p_times)
    print(f"\nNight {night_date.date()}  telescope {args.telescope}  "
          f"tomography {'ON' if tomography_on else 'off'}  "
          f"({len(times)} DIMM samples, {n_mass} MASS profiles)")
    _lgs_off = DEF_LGS_OFFSET[args.telescope] if args.lgs_offset is None else args.lgs_offset
    print(f"  budget: {'LEGACY' if args.legacy_budget else 'refined 2026-07'}"
          f" values v{getattr(args, 'budget_version', DEFAULT_BUDGET_VERSION)}  "
          f"TT star: R={args.tt_mag:g} at {args.tt_offset:g}\" "
          f"{'(budget default)' if (args.tt_mag == DEF_TT_MAG and args.tt_offset == DEF_TT_OFFSET) else '(custom)'}  "
          f"LGS offset: {_lgs_off:g}\""
          + (f"  LTAO bw factor: {_ltao_bw_fac:.2f}"
             f" (floor-frac {args.ltao_bw_floor_frac:g})" if tomography_on
             and not args.legacy_budget else "")
          + f"  NGS seeing law: {args.ngs_seeing_law}"
          + (f"  NGS offset: {args.ngs_offset:g}\" (aniso-corrected"
             + (f"; assumed theta0={args.assumed_theta0:g}\" on {n_fb} samples" if n_fb else "")
             + ")" if float(args.ngs_offset or 0.0) > 0.0 else ""))
    stats(f"NGS R={args.ngs_bright:g}", ngs_bright)
    stats("single-beacon LGS",   sr_single)
    if tomography_on:
        stats("LTAO",             sr_ltao)
        if n_mass:
            print(f"  LTAO mean gain over single: "
                  f"{np.nanmean(sr_ltao) - np.nanmean(sr_single):+.3f}")

    # --- per-observing-window summary (only when --target is given) ----------
    if show_target:
        for (w0, w1) in windows:
            p_mask = np.array([w0 <= t <= w1 for t in p_times]) if len(p_times) \
                     else np.array([], dtype=bool)
            if p_mask.any():
                print(f"\n  In window {w0:%H:%M}-{w1:%H:%M} HST "
                      f"({p_mask.sum()} MASS profiles):")
                stats("    single-beacon LGS", sr_single[p_mask])
                if tomography_on:
                    stats("    LTAO",          sr_ltao[p_mask])

    # =========================================================================
    #  CSV EXPORT  --  predicted SR vs time, auto-named next to the PNG
    # =========================================================================
    csv_path = os.path.splitext(out_path)[0] + ".csv"
    write_csv_table(args, prep, res, csv_path)

    # --- figures (built by the extracted, GUI-shared render fns) ----------
    #  --report fwhm: the FWHM timeline REPLACES the Strehl figure;
    #  --report both: the Strehl figure gains a right-hand FWHM axis;
    #  --report strehl (default): unchanged, byte-identical to the references.
    if args.report == "fwhm":
        fig = render_fwhm_figure(args, prep, res)
        # per-mode FWHM medians on the console, mirroring the Strehl stats;
        # both conventions, so console and CSV can never be cross-mistaken
        for lbl, a, g in (
                (f"NGS R={args.ngs_bright:g}", res.fwhm_ngs_bright,
                 res.fwhm_gauss_ngs_bright),
                ("single-beacon LGS", res.fwhm_single, res.fwhm_gauss_single),
                ("LTAO", res.fwhm_ltao if tomography_on else None,
                 res.fwhm_gauss_ltao if tomography_on else None)):
            if a is not None and np.isfinite(a).any():
                print(f"  FWHM {lbl:20s} median {np.nanmedian(a):.0f} mas "
                      f"(core+halo)  |  {np.nanmedian(g):.0f} mas "
                      f"(Gaussian-fit sim.)")
    else:
        fig = render_main_figure(args, prep, res)
        if args.report == "both":
            overlay_fwhm_on_main(fig, args, prep, res)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nWrote {out_path}")

    fig2 = render_terms_figure(args, prep, res)
    if fig2 is not None:
        # terms figure follows the UT-date naming convention regardless of
        # how the main output was named (same directory as the main PNG)
        terms_path = os.path.join(
            os.path.dirname(out_path) or ".",
            f"ao_strehl_{ut_stamp}_{args.telescope}_terms.png")
        fig2.savefig(terms_path, dpi=150, bbox_inches="tight")
        print(f"Wrote terms figure {terms_path}")

        if not args.no_combined:
            try:
                from PIL import Image as _PILImage
                im_main  = _PILImage.open(out_path).convert("RGB")
                im_terms = _PILImage.open(terms_path).convert("RGB")
                gap = 30
                # side-by-side: main timeline left, terms right, each centered
                # vertically (better fit for widescreen displays than stacking)
                W = im_main.width + gap + im_terms.width
                H = max(im_main.height, im_terms.height)
                canvas = _PILImage.new("RGB", (W, H), (255, 255, 255))
                canvas.paste(im_main,  (0, (H - im_main.height)  // 2))
                canvas.paste(im_terms, (im_main.width + gap,
                                        (H - im_terms.height) // 2))
                all_path = os.path.join(
                    os.path.dirname(out_path) or ".",
                    f"ao_strehl_{ut_stamp}_{args.telescope}_all.png")
                canvas.save(all_path)
                print(f"Wrote combined figure {all_path}")
            except ImportError:
                print("  NOTE: PIL/Pillow not available — combined figure "
                      "skipped (install pillow, or use the two separate PNGs).")
            except Exception as e:      # never let compositing kill a finished run
                print(f"  WARNING: combined figure skipped ({e}); the main and "
                      f"terms PNGs were written normally.")


def _preparse(argv):
    """argparse rejects '--dec -29d00m28s' (a value with a leading dash
    parses as an option), so the README/KAON example commands would fail
    as written for any southern target. Join the value onto its flag
    ('--dec=-29d00m28s'), which argparse always accepts; done for --ra
    too so the pair behaves identically."""
    out, it = [], iter(argv)
    for a in it:
        if a in ("--ra", "--dec"):
            v = next(it, None)
            out.append(a if v is None else f"{a}={v}")
        else:
            out.append(a)
    return out


def _cli():
    """Zero-argument console entry point (see pyproject [project.scripts])."""
    import sys
    main(build_parser().parse_args(_preparse(sys.argv[1:])))


if __name__ == "__main__":
    _cli()
