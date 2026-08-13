"""Fixed constants: CLI defaults, observatory/instrument facts, and physics
reference values that are never mutated at runtime (contrast budget.py, whose
adjustable parameters ARE mutated live via budget_overrides() for the GUI's
what-if sliders -- see that module's docstring for why the two are kept
separate).
"""
import numpy as np

# =============================================================================
#  DEFAULTS  --  used when a command-line option is not supplied.
#  (Run with -h / --help to see every option and its default.)
# =============================================================================
DEF_DIMM     = "20260525_dimm.dat"
DEF_MASS     = "20260525_mass.dat"
DEF_MASSPRO  = "20260525_masspro.dat"
DEF_TELESCOPE   = "K2"
DEF_TARGET_NAME = "Galactic Center"
DEF_TARGET_RA   = "17h45m40.04s"     # Sgr A*
DEF_TARGET_DEC  = "-29d00m28.1s"     # Sgr A*
DEF_NIGHT_DATE  = "2026-05-24"       # evening (civil) date, HST
DEF_WINDOWS     = ["00:14-03:20"]    # HST clock windows (see --window help)
DEF_NGS_BRIGHT  = 8.0
DEF_NGS_FAINT   = 12.0
DEF_ELEV_CUT    = 36.8
DEF_AIRMASS_PAD = 0.32
DEF_MATCH_TOL   = 600
DEF_OUTPUT      = "ao_strehl_timeline.png"

# MKWC (Mauna Kea Weather Center) seeing-archive URL templates.
# The {ymd} field is the file's YYYYMMDD date stamp (note: the file is named by
# the morning/UT date, e.g. an evening-of-24-May night is in 20260525.*.dat).
MKWC_BASE     = "http://mkwc.ifa.hawaii.edu/current/seeing"
MKWC_DIMM_URL    = MKWC_BASE + "/dimm/{ymd}.dimm.dat"
MKWC_MASS_URL    = MKWC_BASE + "/mass/{ymd}.mass.dat"
MKWC_MASSPRO_URL = MKWC_BASE + "/masspro/{ymd}.masspro.dat"
DEF_CACHE_DIR    = "mkwc_cache"          # where fetched files are stored

# Keck location (used for airmass) -- fixed observatory constants
KECK_LAT_DEG   = 19.8260
KECK_LON_DEG   = -155.4747
KECK_HEIGHT_M  = 4145.0

# Per-telescope pointing limits (Keck horizon / Nasmyth-deck shadow).
#   * In the "wedge" azimuth range the Nasmyth deck blocks low elevations, so
#     the unvignetted floor is higher (wedge_floor).
#   * Outside the wedge the unvignetted floor is 18 deg.
#   * Below 18 deg the target is vignetted (still observable, degraded) down to
#     0 deg -- EXCEPT inside the wedge, where below wedge_floor it is fully
#     blocked by the deck (not merely vignetted).
#   * There is also an upper elevation ceiling, and guiding is not guaranteed
#     above ~85 deg.
POINTING_LIMITS = {
    "K1": dict(wedge=(5.3, 146.2),   wedge_floor=33.3, open_floor=18.0,
               vignet_floor=0.0, ceiling=88.9, guide_warn=85.0),
    "K2": dict(wedge=(185.3, 332.8), wedge_floor=36.8, open_floor=18.0,
               vignet_floor=0.0, ceiling=89.5, guide_warn=85.0),
}

# Physics constants that are not normally tuned per-run
V2K               = 0.744     # 500 nm -> K-band seeing conversion factor
LAMBDA_K_NM       = 2200.0    # K-band wavelength (nm) for the Marechal Strehl
HST_TO_UTC_HOURS  = 10        # HST = UTC - 10 (no daylight saving in Hawaii)

# Standard photometric near-IR band central wavelengths (nm), for --band.
PHOTOMETRIC_BANDS = {
    "z": 900.0, "Y": 1020.0, "J": 1250.0, "H": 1650.0,
    "K": 2200.0, "Ks": 2150.0, "L": 3500.0, "M": 4800.0,
}

# Reference seeing values the budget's seeing-dependent terms are normalized
# to: a term scales as (eps/REF_x)^(5/6) in nm RMS. NOT in ADJUSTABLE_BUDGET_
# PARAMS (budget.py) -- these are the normalization anchor itself, not a
# tunable allocation.
REF_TOTAL   = 0.50    # nominal TOTAL (DIMM) seeing, arcsec
REF_FREEATM = 0.30    # nominal FREE-ATM (MASS) seeing, arcsec

# Representative wind speeds for the wind-weighted bandwidth term
# (budget.bw_wind_scale). Defaults for --wind-ground/--wind-free; NOT in
# ADJUSTABLE_BUDGET_PARAMS (always explicitly threaded through as function
# arguments, never read live via budget_overrides' global mutation).
V_GROUND = 8.0     # m/s -- representative boundary-layer wind
V_FREE   = 25.0    # m/s -- representative free-atmosphere wind

NM_PER_MAS = 164.0 / 13.1   # the TT sheet's own total mas -> nm conversion

# Standard CFHT MASS profile bin heights (6 layers) -- a fixed property of the
# MASS instrument's altitude bins, shared by atmosphere.py (theta0/d0 from the
# profile, Cn2 density) and fieldmap.py (the synthetic-profile predictor).
MASS_HEIGHTS_M = np.array([0.5, 1, 2, 4, 8, 16]) * 1e3   # standard MASS bins

# DM actuator sampling across the pupil (Eduardo, 2026-07-10): sets the AO
# control radius theta_c = (N/2) * lambda/D, inside which the corrected-band
# residuals (servo lag, measurement, aliasing, ...) scatter their light.
# K1: classic 349-actuator Xinetics, 20 across. K2: HAKA, 57 across.
DM_ACTUATORS_ACROSS = {"K1": 20.0, "K2": 57.0}

# Keck aperture / seeing-disk shape constants used throughout the PSF model
# (see psf.py): FWHM ~= 1.029 lambda/D while the AO core dominates, then
# transitions to the Kolmogorov seeing-disk width as Strehl collapses.
TEL_DIAMETER_M   = 9.96    # m; inscribed-circle Keck aperture
MOFFAT_BETA_KOLM = 4.765   # Moffat beta of a Kolmogorov seeing disk

# Science-camera square field of view per telescope (field map default),
# North = +Y, East = -X convention: 20x20" (K1) / 10x10" (K2).
FIELD_FOV_ARCSEC = {"K1": 20.0, "K2": 10.0}
