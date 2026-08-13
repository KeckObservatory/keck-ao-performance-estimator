"""The LGS/LTAO high-order wavefront-error budget: the allocation constants,
the GUI's live what-if slider machinery (ADJUSTABLE_BUDGET_PARAMS,
budget_overrides), the wind-weighted bandwidth and tomography layer-mismatch
refinements, and lgs_budget_terms/lgs_strehl -- the single source of truth
the rest of the engine calls into.

*** HAZARD: live mutation, module-qualified access ***
budget_overrides() mutates values that other code reads. Two mutation shapes
exist here, with different implications for code OUTSIDE this module:

  * FITTING_ERR and STATIC_TEL are dicts; their virtual per-telescope names
    (FITTING_ERR_K1/K2, STATIC_TEL_K1/K2) are overridden via IN-PLACE item
    assignment (`FITTING_ERR["K1"] = v`). Any other module holding a
    reference to the SAME dict object (e.g. via `from budget import
    FITTING_ERR`) sees the mutation immediately -- dict identity is shared,
    only its contents change. Safe to import by plain name.

  * Every other adjustable name (BW_REF, SCINT_REF, FA_REF, ANG_REF, HOMEAS,
    NAFOC, STATIC_CALIB, STATIC_DM, STATIC_INST, STATIC_REG, MARGIN, TOMO_ERR)
    is a plain scalar, overridden by REBINDING this module's global
    (`globals()[k] = v`). A `from budget import ANG_REF` done anywhere else
    captures the value at import time and will NEVER see a later override --
    it is a permanently stale snapshot. Any code outside this module that
    needs the LIVE value of one of these must do `import
    keck_ao_estimator.budget as budget` and read `budget.ANG_REF` (a fresh
    attribute lookup every time), not a bare imported name. Functions defined
    IN this module (lgs_budget_terms, lgs_strehl, static_subtotal, ...) are
    unaffected either way: a function always resolves its own free variables
    against its DEFINING module's globals, regardless of who calls it.
"""
import numpy as np

from .constants import REF_TOTAL, REF_FREEATM, V_GROUND, V_FREE, LAMBDA_K_NM
from .marechal import marechal_strehl
from .tiptilt import (tt_wfe_nm, DEF_TT_MAG, DEF_TT_OFFSET,
                      DEF_LTAO_TT_THETA0_GAIN)

# =============================================================================
#  BUDGET  --  wavefront-error allocations (nm RMS), AT ZENITH for the
#  reference profile (REF_TOTAL / REF_FREEATM).  The pipeline projects each
#  night's measured seeing to the target's line of sight (airmass^(3/5)) and
#  the scaling laws below apply on top -- so these reference numbers must be
#  zenith-referenced or the zenith degradation is counted twice.
#
#  PROVENANCE / VERSIONS (BUDGET_VERSIONS below; adopted 2026-07-24):
#   v3_1_3 (default): the "AO performance error budget" spreadsheet v3_1_3
#     (K1 LGS column, quoted at ZA = 50 deg).  The sheet values CARRY the
#     ZA-50 degradation, so the atmosphere-scaled slots are re-referenced to
#     zenith by (sec 50)^(1/2) = 1.2473 (WFE ~ seeing^(5/6), seeing ~
#     airmass^(3/5)); the tomography residual divides by (sec 50)^(1/5) =
#     1.0924 instead, matching its weaker eps_fa^(1/3) scaling law; the
#     seeing-independent slots are taken as-is.  Validated on the M79
#     2026-01-31 on-axis Strehls: v3_1_1 under-predicted single-LGS 0.238
#     vs 0.327 measured; v3_1_3 predicts 0.315 (within the night's scatter).
#   v3_1_1: the previous baseline.  Its dominant scaled slots equal the
#     v3_1_3 sheet x (0.6/0.5)^(5/6) EXACTLY (fitting 141 = 121 x 1.164,
#     FA 178 = 153 x 1.164): the sheet's ZA-50 conditions had been treated
#     as a 0.6" zenith reference, so the pipeline's per-night projection
#     double-counted the zenith degradation.  Kept selectable
#     (--budget-version 3_1_1 / the WFE tab's version picker) for
#     comparisons against older runs.
#  K2 caveats (no K2 v3_1_3 sheet in hand): FITTING_ERR K2 is the v3_1_1
#  value de-scaled by the same 1.2473 (same ZA-50 provenance); STATIC_TEL K2
#  keeps its v3_1_1 value pending a K2 sheet.
#  NOT revisited (deliberate, 2026-07-24): LTAO_BW_FLOOR_FRAC -- the
#  residual ~0.06 LTAO under-prediction on M79 points at the LTAO bandwidth
#  surcharge, but it stays 0.70 until audited against 600 Hz loop telemetry.
# =============================================================================

DEFAULT_BUDGET_VERSION = "3_1_3"

#  Every adjustable parameter in both reference-value sets, rounded to 0.1 nm
#  (the budget-slider resolution, so the GUI shows defaults exactly).
#  v3_1_3 sheet rows behind each grouped slot (nm, quadrature):
#    scint  = (12 scint, 10 WFS scint, 2 chromatic, 2 dispersion,
#              25 multispectral) = 29.6      -> /1.2473 = 23.7
#    meas   = (49 HO measurement, 40 HO aliasing) = 63.3        (as-is)
#    tel K1 = (66 static, 35 dynamic) = 74.7                    (as-is)
#    calib  = (25 static ZP, 50 dynamic ZP, 15 leaky) = 57.9    (as-is)
#    DM     = (31 finite stroke, 13 hysteresis, 1 digitiz) = 33.6
#    inst   = (30 AO system, 60 instrument) = 67.1
#    reg    = (15 misregistration, 15 pupil scale) = 21.2
#  and the single-row slots: fitting 121 -> 97.0, bw 53 -> 42.5,
#  FA 153 -> 122.7, angular 29 -> 23.3, Na focus 42, margin 130,
#  tomo 93 -> 85.1 (the (sec 50)^(1/5) law).
BUDGET_VERSIONS = {
    "3_1_1": dict(FITTING_ERR_K1=141.0, FITTING_ERR_K2=60.0, BW_REF=60.0,
                  SCINT_REF=46.0, FA_REF=178.0, ANG_REF=44.0, HOMEAS=98.0,
                  NAFOC=57.0, STATIC_TEL_K1=68.2, STATIC_TEL_K2=50.0,
                  STATIC_CALIB=57.9, STATIC_DM=13.0, STATIC_INST=43.1,
                  STATIC_REG=21.2, MARGIN=130.0, TOMO_ERR=93.0),
    "3_1_3": dict(FITTING_ERR_K1=97.0, FITTING_ERR_K2=48.1, BW_REF=42.5,
                  SCINT_REF=23.7, FA_REF=122.7, ANG_REF=23.3, HOMEAS=63.3,
                  NAFOC=42.0, STATIC_TEL_K1=74.7, STATIC_TEL_K2=50.0,
                  STATIC_CALIB=57.9, STATIC_DM=33.6, STATIC_INST=67.1,
                  STATIC_REG=21.2, MARGIN=130.0, TOMO_ERR=85.1),
}
_V = BUDGET_VERSIONS[DEFAULT_BUDGET_VERSION]

# Fitting error depends on the telescope's DM order:
FITTING_ERR = {"K2": _V["FITTING_ERR_K2"], "K1": _V["FITTING_ERR_K1"]}

# High-order terms that scale with the TOTAL seeing (DIMM):
BW_REF      = _V["BW_REF"]      # bandwidth / servo-lag error
SCINT_REF   = _V["SCINT_REF"]   # scintillation (+ chromatic-family rows)
# High-order terms that scale with the FREE-ATM seeing (MASS):
FA_REF      = _V["FA_REF"]      # focal anisoplanatism (single-beacon cone)
ANG_REF     = _V["ANG_REF"]     # angular anisoplanatism (at 2"; refinement 4)
# Fixed high-order terms (seeing-independent):
HOMEAS      = _V["HOMEAS"]      # high-order measurement + aliasing
NAFOC       = _V["NAFOC"]       # sodium-layer focus error
MARGIN      = _V["MARGIN"]      # high-order margin (reserve)

# Static / calibration error in five physical sub-groups (broken out
# 2026-07-15; regrouped to the v3_1_3 sheet 2026-07-24 -- the sheet carries
# DM finite stroke 31 and instrument 60 that the old grouping lacked).  The
# telescope-aberration group is PER-TELESCOPE (K1's primary/segment figure is
# worse); see the K2 caveat in the header.
STATIC_TEL   = {"K2": _V["STATIC_TEL_K2"], "K1": _V["STATIC_TEL_K1"]}
STATIC_CALIB = _V["STATIC_CALIB"]
STATIC_DM    = _V["STATIC_DM"]
STATIC_INST  = _V["STATIC_INST"]
STATIC_REG   = _V["STATIC_REG"]


def static_subtotal(telescope):
    """Quadrature sum of the five static/calibration sub-groups (nm RMS) for
    the given telescope -- the sheet's 'static' subtotal, telescope-dependent
    (v3_1_3: 109.2 nm K2 / 122.5 nm K1). Display/provenance only;
    lgs_strehl() sums the individual groups from lgs_budget_terms()."""
    return float(np.sqrt(STATIC_TEL[telescope]**2 + STATIC_CALIB**2
                         + STATIC_DM**2 + STATIC_INST**2 + STATIC_REG**2))


def apply_budget_version(version):
    """Persistently rebind EVERY adjustable budget parameter to the named
    reference-value set ("3_1_3" default / "3_1_1" legacy values).  Unlike
    budget_overrides this restores nothing: it is the CLI's --budget-version
    switch, applied once per process before computing.  BUDGET_DEFAULTS
    stays the v3_1_3 snapshot, so under "3_1_1" every changed parameter
    shows up in active_budget_overrides() and the run's provenance records
    the non-default budget (it never masquerades as the reference)."""
    vals = BUDGET_VERSIONS[str(version)]
    g = globals()
    for k, v in vals.items():
        if k == "FITTING_ERR_K1":
            FITTING_ERR["K1"] = float(v)
        elif k == "FITTING_ERR_K2":
            FITTING_ERR["K2"] = float(v)
        elif k == "STATIC_TEL_K1":
            STATIC_TEL["K1"] = float(v)
        elif k == "STATIC_TEL_K2":
            STATIC_TEL["K2"] = float(v)
        else:
            g[k] = float(v)


# ---- adjustable budget parameters (GUI slider registry) -----------------------
#  Every entry is a module-global read at CALL time by the budget functions, so
#  temporarily overriding it changes the computed Strehl. The GUI exposes these
#  as sliders; budget_overrides() is the ONLY sanctioned way to modify them
#  (it restores the originals even on exception, so a slider experiment can
#  never contaminate a later run). FITTING_ERR_K1/K2 are virtual names mapping
#  into the FITTING_ERR dict. Values are nm RMS at the reference profile; the
#  seeing/airmass scaling machinery applies on top, so a slider means the same
#  thing on every night. NOTE any output produced under an override MUST carry
#  the overridden values in its provenance (see GUI spec) -- a modified budget
#  must never masquerade as the reference budget.
ADJUSTABLE_BUDGET_PARAMS = {
    #  name              (description,                                lo,   hi)
    "FITTING_ERR_K1": ("DM fitting error, K1",                       50.0, 300.0),
    "FITTING_ERR_K2": ("DM fitting error, K2",                       20.0, 200.0),
    "BW_REF":         ("bandwidth / servo-lag at reference",         10.0, 200.0),
    "SCINT_REF":      ("scintillation at reference",                  5.0, 150.0),
    "FA_REF":         ("focal anisoplanatism (single-beacon cone)",  50.0, 400.0),
    "ANG_REF":        ("angular anisoplanatism at 2\" offset",        5.0, 150.0),
    "HOMEAS":         ("high-order measurement",                     20.0, 250.0),
    "NAFOC":          ("sodium-layer focus",                         10.0, 150.0),
    "STATIC_TEL_K1":  ("telescope aberr (static+dyn), K1",           10.0, 200.0),
    "STATIC_TEL_K2":  ("telescope aberr (static+dyn), K2",           10.0, 200.0),
    "STATIC_CALIB":   ("WFS zero-point calib + leaky integrator",     5.0, 200.0),
    "STATIC_DM":      ("DM hysteresis + drive digitization",          0.0, 100.0),
    "STATIC_INST":    ("uncorrectable AO-system + instrument aberr",  5.0, 200.0),
    "STATIC_REG":     ("DM-to-lenslet misreg + pupil-scale",          0.0, 100.0),
    "MARGIN":         ("high-order margin (reserve)",                 0.0, 250.0),
    "TOMO_ERR":       ("LTAO tomography residual",                   20.0, 250.0),
}


def get_budget_param(name):
    """Current value of an adjustable budget parameter (virtual names OK)."""
    if name == "FITTING_ERR_K1":
        return FITTING_ERR["K1"]
    if name == "FITTING_ERR_K2":
        return FITTING_ERR["K2"]
    if name == "STATIC_TEL_K1":
        return STATIC_TEL["K1"]
    if name == "STATIC_TEL_K2":
        return STATIC_TEL["K2"]
    return globals()[name]


from contextlib import contextmanager


@contextmanager
def budget_overrides(**overrides):
    """Temporarily override adjustable budget parameters, restoring the
    originals on exit (even on exception). Example:
        with budget_overrides(MARGIN=90.0, FA_REF=200.0):
            res = compute_timeline(args, prep)
    Unknown names raise KeyError so a typo can't silently do nothing."""
    for k in overrides:
        if k not in ADJUSTABLE_BUDGET_PARAMS:
            raise KeyError(f"not an adjustable budget parameter: {k!r} "
                           f"(see ADJUSTABLE_BUDGET_PARAMS)")
    saved = {k: get_budget_param(k) for k in overrides}
    g = globals()
    try:
        for k, v in overrides.items():
            if k == "FITTING_ERR_K1":
                FITTING_ERR["K1"] = float(v)
            elif k == "FITTING_ERR_K2":
                FITTING_ERR["K2"] = float(v)
            elif k == "STATIC_TEL_K1":
                STATIC_TEL["K1"] = float(v)
            elif k == "STATIC_TEL_K2":
                STATIC_TEL["K2"] = float(v)
            else:
                g[k] = float(v)
        yield
    finally:
        for k, v in saved.items():
            if k == "FITTING_ERR_K1":
                FITTING_ERR["K1"] = v
            elif k == "FITTING_ERR_K2":
                FITTING_ERR["K2"] = v
            elif k == "STATIC_TEL_K1":
                STATIC_TEL["K1"] = v
            elif k == "STATIC_TEL_K2":
                STATIC_TEL["K2"] = v
            else:
                g[k] = v


# --- LTAO (laser tomography) terms --------------------------------------------
# v3_1_3: the sheet's 93 nm (ZA=50) re-referenced to zenith by (sec 50)^(1/5)
# -- NOT the (sec 50)^(1/2) used for the Kolmogorov slots, because this term's
# own scaling law below is eps_fa^(1/3): a reference night observed at ZA 50
# must reproduce the sheet's 93 under that law, which fixes the factor.
TOMO_ERR          = _V["TOMO_ERR"]   # laser-tomography error replacing FA, nm

# Snapshot of the reference (unmodified) value of every adjustable budget
# parameter, captured once at import (here, AFTER the last such global --
# TOMO_ERR -- is defined) BEFORE any budget_overrides() can run.
# active_budget_overrides() compares live values against it so exported
# products record exactly which parameters were moved off the reference budget
# -- and record NOTHING when none were, so the CLI's outputs stay byte-
# identical to the frozen references (the harness enforces this).
BUDGET_DEFAULTS = {name: get_budget_param(name)
                   for name in ADJUSTABLE_BUDGET_PARAMS}


def active_budget_overrides():
    """Adjustable budget parameters whose current value differs from the
    reference snapshot -> {name: current_value}. Empty unless code is running
    inside a budget_overrides() context (i.e. only the GUI's what-if runs)."""
    return {name: get_budget_param(name)
            for name in ADJUSTABLE_BUDGET_PARAMS
            if get_budget_param(name) != BUDGET_DEFAULTS[name]}

#
#  LTAO BANDWIDTH PENALTY (revised 2026-07 after the K2 NGS budget analysis).
#  The LTAO high-order loop runs SLOWER than single-beacon: on K2, ~600 Hz for
#  LTAO vs ~1500 Hz for single-beacon. The old model multiplied the whole
#  bandwidth term by 2^(5/6) (a pure frame-rate servo-lag law at an assumed
#  half rate). Two corrections from the NGS analysis apply:
#    (1) the true rate ratio is 1500/600 = 2.5, not 2; and
#    (2) servo-lag error is NOT purely frame-rate-limited -- it is dominated by
#        a FIXED loop-latency floor (RTC compute + DM response) that does not
#        change with frame rate; only the residual rate-dependent part scales.
#  So the penalty is applied to the rate-dependent PART only:
#     bw_ltao = sqrt( (floor_frac*BW)^2 + ((1-floor_frac_quad)*BW * ratio^(5/6))^2 )
#  where floor_frac is the fraction of the single-beacon bandwidth VARIANCE that
#  is latency floor (frame-rate-independent). floor_frac=0 recovers the old pure
#  rate law; floor_frac=1 removes the penalty entirely. The NGS budget was
#  strongly floor-dominated (~0.7-0.85); we default to 0.7 and expose it.
LTAO_RATE_SINGLE  = 1500.0   # single-beacon HO loop rate, Hz
LTAO_RATE_TOMO    = 600.0    # LTAO HO loop rate, Hz
LTAO_BW_FLOOR_FRAC = 0.70    # fraction of bandwidth VARIANCE that is latency floor


def ltao_bw_factor(floor_frac=LTAO_BW_FLOOR_FRAC,
                   rate_single=LTAO_RATE_SINGLE, rate_tomo=LTAO_RATE_TOMO):
    """Effective multiplier on the single-beacon bandwidth term for the slower
    LTAO loop, accounting for a fixed latency floor. Returns 1.0 when the loop
    is fully floor-dominated (floor_frac=1) and (rate_single/rate_tomo)^(5/6)
    when there is no floor (floor_frac=0)."""
    ratio = (rate_single / rate_tomo) ** (5.0 / 6.0)   # raw rate penalty
    floor_var = floor_frac                              # variance fraction
    rate_var  = max(1.0 - floor_frac, 0.0)
    # single-beacon bw normalized to 1; LTAO inflates only the rate part
    return float(np.sqrt(floor_var + rate_var * ratio**2))


HALF_RATE_BW_FAC  = ltao_bw_factor()   # backward-compatible name; now floor-aware
#  NOTE: with the OCAM2K EM-gain model the MEASUREMENT error is NOT penalized
#  (EM gain preserves SNR at the lower rate); only the bandwidth term grows.

# =============================================================================
#  BUDGET REFINEMENTS (2026-07) -- validated on the 2025-12-04/06 K1 nights
#  (HD 18770).  Disable both with --legacy-budget to reproduce older runs.
# =============================================================================
#  (1) WIND-WEIGHTED BANDWIDTH.  Servo-lag error scales with the Greenwood
#      frequency: fG^(5/3) ~ integral Cn2(h) v(h)^(5/3) dh.  The legacy budget
#      scaled bw with TOTAL seeing only, i.e. assumed a fixed median wind mix.
#      Refined: split Cn2 into ground (DIMM - MASS) and free-atm (MASS) parts,
#      weight each by a representative wind speed. Ground-dominated nights get
#      bandwidth RELIEF (slow boundary-layer flow); free-atm-dominated nights
#      get a SURCHARGE (fast jet-driven layers). V_GROUND/V_FREE live in
#      keck_ao_estimator.constants.
#
#  (2) TOMOGRAPHY LAYER-MISMATCH PENALTY (LTAO only).  The K1 tomographic
#      reconstructor uses a STATIC layer prior (KAON reconstructor table:
#      altitudes 0/0.5/1/2/4/8/16 km).  When the night's actual aloft Cn2
#      distribution (6 MASS bins, same altitudes as the prior's aloft part)
#      deviates from the prior, tomographic correction degrades.  Model: the
#      mismatch m (total-variation distance between the night's normalized
#      MASS-bin profile and the prior aloft fractions) degrades the correction
#      toward the single-beacon cone term with a QUADRATIC ramp:
#          alt^2 = tomo^2 + m^2 * (FA^2 - tomo^2),      m in [0, 1]
#      m=0 -> stock tomography term; m=1 -> no tomographic benefit.  Both
#      endpoints are existing budget terms; NO fitted parameter.  The ramp is
#      quadratic because an MMSE reconstructor's performance loss is
#      second-order in prior mismatch: the beacons still MEASURE the actual
#      turbulence, and a moderately wrong prior mainly misassigns altitude,
#      which costs little across the asterism's small angles.  (A linear ramp
#      was tried first and erased the LTAO-over-sLGS benefit at the m ~ 0.5
#      typical of ordinary nights, contradicting on-sky experience; the
#      quadratic law preserves it while still closing the extreme-mismatch
#      2025-12-06 validation night.)
RECON_PRIOR_FRAC = np.array([0.4557, 0.1295, 0.0442, 0.0506,
                             0.1167, 0.0926, 0.1107])         # 0..16 km
RECON_PRIOR_ALOFT = RECON_PRIOR_FRAC[1:] / RECON_PRIOR_FRAC[1:].sum()

#  (4) LASER-POINTING OFFSET (--lgs-offset).  The 44 nm angular-anisoplanatism
#      allocation was made for a beacon just 2" from the science direction.
#      Operationally the K1 laser is offset from the science target (and in
#      LTAO the 4-beacon asterism CENTER is offset, which is the relevant
#      separation for the tomographic solution); the K2 laser is not offset
#      at all.  The term now scales with the actual offset:
#          sigma_ang = ANG_REF * s_fa * (theta / ANG_REF_OFFSET)^(5/6)
#      (sigma^2 ~ theta^(5/3)), going to zero on-axis.  Defaults are the
#      per-telescope operational offsets below; override with --lgs-offset.
#
#      K1 MAGNITUDE, 2026-08-07 (Eduardo): the pointing-offset campaign
#      settled on 4.8" WEST and 1.3" SOUTH of the pointing origin in the
#      standard north-up/east-left configuration -> radius 4.97", position
#      angle 254.8 deg (N->E).  This REPLACES the earlier 7" placeholder and
#      is consistent with the rest of the campaign (direct 20260801 boresight
#      measurement 4.7"W 1.0"S; LTAO field-optimum peaks ~5.8-6.1", which sit
#      further out because the asterism-weighted optimum is pushed off the
#      single-beacon boresight).  It is a BENCH STAGE-ALIGNMENT property, so
#      it lives in the DETECTOR frame and is stable night to night -- not a
#      nightly pointing error.  Effect on the budget: the angular term drops
#      from 2.84x to 2.14x ANG_REF, i.e. the K1 LGS estimate improves.
#      The DIRECTION (the PA) is what the field map uses to place the laser;
#      the magnitude alone is what the science-direction budget needs.
ANG_REF_OFFSET = 2.0                       # arcsec; offset the 44 nm assumed
DEF_LGS_OFFSET = {"K1": 4.97, "K2": 0.0}   # operational beacon/asterism-center
                                           # offset from science, arcsec
DEF_LASER_PA_DEG = 254.8                   # N->E; K1 campaign direction
                                           # (4.8" W, 1.3" S). Field map only:
                                           # the budget term is radial.


def bw_wind_scale(eps_total, eps_freeatm, v_ground=V_GROUND, v_free=V_FREE):
    """Wind-weighted bandwidth scale factor replacing s_tot in the bw term
    (refinement (1)). Equals ~s_tot at the reference-night Cn2/wind mix.

    v_ground / v_free : representative boundary-layer / free-atmosphere wind
    speeds (m/s). Default to the module constants; the CLI (--wind-ground/
    --wind-free) and GUI let the user substitute a night's estimate (e.g. from
    a GFS forecast). The REFERENCE mix in the denominator uses the same wind
    speeds, so at the default speeds this is unchanged."""
    J_tot = eps_total ** (5.0 / 3.0)
    J_fa  = min(eps_freeatm, eps_total) ** (5.0 / 3.0)
    J_g   = max(J_tot - J_fa, 0.0)
    W     = J_g * v_ground ** (5.0/3.0) + J_fa * v_free ** (5.0/3.0)
    Jr_t  = REF_TOTAL   ** (5.0 / 3.0)
    Jr_f  = REF_FREEATM ** (5.0 / 3.0)
    W_ref = (Jr_t - Jr_f) * v_ground ** (5.0/3.0) + Jr_f * v_free ** (5.0/3.0)
    return np.sqrt(W / W_ref)


def layer_mismatch(cn2_bins):
    """Total-variation distance between the night's normalized aloft profile
    (6 MASS bins) and the reconstructor's static aloft prior (refinement (2)).
    0 = perfect match, 1 = totally disjoint. Returns 0 if the profile is empty."""
    J = np.asarray(cn2_bins, dtype=float)
    tot = J.sum()
    if not np.isfinite(tot) or tot <= 0:
        return 0.0
    return 0.5 * np.abs(J / tot - RECON_PRIOR_ALOFT).sum()


def lgs_budget_terms(eps_total, eps_freeatm, telescope, mode,
                     cn2_bins=None, tt_mag=DEF_TT_MAG, tt_offset=DEF_TT_OFFSET,
                     lgs_offset=None, legacy=False, bw_factor=HALF_RATE_BW_FAC,
                     v_ground=V_GROUND, v_free=V_FREE, aniso_scale=1.0,
                     tt_sensor="strap", tt_spot_theta=None,
                     strap_law="sheet", ltao_tt_theta0_gain=None):
    """Per-sample LGS/LTAO error-budget terms in nm RMS, as a dict.

    Single source of truth for the budget: lgs_strehl() sums these in
    quadrature and converts to Strehl. Keys: fit, scint, ang, bw, alt, meas,
    nafoc, margin, the five static/calibration sub-groups (stat_tel -- per-
    telescope, stat_calib, stat_dm, stat_inst, stat_reg) and tt (tip-tilt).
    "alt" is the
    focal-anisoplanatism (cone) term for mode="single" and the tomography
    (+ quadratic layer-mismatch) term for mode="ltao". Fixed terms are
    included so the dict is the complete budget for the sample.

    aniso_scale multiplies the angular-anisoplanatism term and the TT-star
    anisoplanatism rows: the seeing scalings assume the REFERENCE profile's
    altitude distribution, and this factor -- (theta0_ref/theta0)^(5/6) --
    expresses a scenario whose theta0 is decoupled from its free-atm seeing
    (GUI prediction tab). Default 1.0 = every night-data path, unchanged.
    """
    fitting = FITTING_ERR[telescope]

    # amplitude scaling factors (on nm RMS), referenced to the nominal night
    s_tot     = (eps_total   / REF_TOTAL)   ** (5.0 / 6.0)   # total-seeing terms
    s_fa      = (eps_freeatm / REF_FREEATM) ** (5.0 / 6.0)   # free-atm terms
    s_fa_weak = (eps_freeatm / REF_FREEATM) ** (1.0 / 3.0)   # weak free-atm dep.

    # terms common to both modes
    fit   = fitting   * s_tot
    scint = SCINT_REF * s_tot
    # angular anisoplanatism at the ACTUAL beacon/asterism-center offset
    # (refinement 4); the legacy budget charged the 2" allocation flat
    if legacy:
        ang = ANG_REF * s_fa * aniso_scale
    else:
        theta = DEF_LGS_OFFSET[telescope] if lgs_offset is None else lgs_offset
        ang   = (ANG_REF * s_fa * aniso_scale
                 * (theta / ANG_REF_OFFSET) ** (5.0 / 6.0))

    # bandwidth scale: wind-weighted Cn2 (refinement 1) unless legacy
    bw_scale = (s_tot if legacy
                else bw_wind_scale(eps_total, eps_freeatm, v_ground, v_free))

    if mode == "single":
        # single-beacon: full focal aniso (cone effect), nominal bandwidth & meas
        alt = FA_REF * s_fa            # focal anisoplanatism (cone effect)
        bw  = BW_REF * bw_scale
    elif mode == "ltao":
        # LTAO: tomography replaces the cone term (weak residual altitude dep.);
        # half-rate bandwidth penalty; measurement preserved (OCAM2K EM gain)
        tomo = TOMO_ERR * s_fa_weak
        if legacy or cn2_bins is None:
            alt = tomo
        else:
            # layer-mismatch penalty (refinement 2): quadratic ramp toward
            # the cone term -- MMSE loss is second-order in prior mismatch
            m  = layer_mismatch(cn2_bins)
            fa = FA_REF * s_fa
            alt = np.sqrt(tomo**2 + m**2 * max(fa**2 - tomo**2, 0.0))
        bw = BW_REF * bw_scale * bw_factor
    else:
        raise ValueError("mode must be 'single' or 'ltao'")

    # tip-tilt budget for the configured TT star (refinement 3)
    # the legacy budget reproduces the frozen 2004 sheet, which includes the
    # original photon-only STRAP measurement row
    _tts = "strap-legacy" if (legacy and tt_sensor == "strap") else tt_sensor
    # tomographic tilt-anisoplanatism reduction (see DEF_LTAO_TT_THETA0_GAIN
    # in tiptilt.py): LTAO's asterism senses the altitude modes that generate
    # field-dependent tilt, so the TT star's aniso rows shrink by
    # (1/gain)^(5/6). LTAO only; legacy stays byte-faithful to the 2004 sheet.
    tt_aniso = aniso_scale
    if mode == "ltao" and not legacy:
        _g = (DEF_LTAO_TT_THETA0_GAIN if ltao_tt_theta0_gain is None
              else float(ltao_tt_theta0_gain))
        _gfac = _g ** (-5.0 / 6.0)
        tt_aniso = aniso_scale * _gfac
        # the same tomographic gain applies at the laser/asterism-centre
        # offset (2026-08-12): the asterism senses the altitude turbulence
        # that decorrelates the science direction from the asterism centre,
        # so the angular-aniso charge shrinks by the same effective-theta0
        # factor as the TT-star rows. Default gain 1.0 -> no change; legacy
        # is excluded above and stays byte-faithful to the 2004 sheet.
        ang *= _gfac
    tt = tt_wfe_nm(s_tot, tt_mag, tt_offset, tt_aniso, sensor=_tts,
                   spot_theta=tt_spot_theta, strap_law=strap_law)

    return dict(fit=fit, scint=scint, ang=ang, bw=bw, alt=alt,
                meas=HOMEAS, nafoc=NAFOC,
                stat_tel=STATIC_TEL[telescope], stat_calib=STATIC_CALIB,
                stat_dm=STATIC_DM, stat_inst=STATIC_INST, stat_reg=STATIC_REG,
                margin=MARGIN, tt=tt)


def lgs_strehl(eps_total, eps_freeatm, telescope, mode, lam_nm=LAMBDA_K_NM,
               cn2_bins=None, tt_mag=DEF_TT_MAG, tt_offset=DEF_TT_OFFSET,
               lgs_offset=None, legacy=False, bw_factor=HALF_RATE_BW_FAC,
               v_ground=V_GROUND, v_free=V_FREE, aniso_scale=1.0,
               tt_sensor="strap", tt_spot_theta=None,
               strap_law="sheet", ltao_tt_theta0_gain=None):
    """Single-beacon LGS or LTAO Strehl for one sample, at wavelength lam_nm.

    Parameters
    ----------
    eps_total    : total (DIMM) seeing, arcsec
    eps_freeatm  : free-atmosphere (MASS) seeing, arcsec
    telescope    : "K1" or "K2"  (selects the fitting-error term)
    mode         : "single" or "ltao"
    lam_nm       : science wavelength (nm); the nm-RMS budget is wavelength-
                   independent, only the Marechal Strehl evaluation uses lam_nm.
    lgs_offset   : beacon (single) / asterism-center (LTAO) offset from the
                   science direction, arcsec. None -> the telescope's
                   operational default (K1: 4.97", K2: 0"). The angular-
                   anisoplanatism term scales as (offset/2")^(5/6) and is
                   zero on-axis; under legacy=True the flat 2"-allocation
                   charge is used regardless.
    cn2_bins     : 6-bin MASS Cn2 profile for the layer-mismatch penalty
                   (LTAO only; None -> penalty skipped)
    tt_mag       : R magnitude of the tip-tilt star   (default: budget star)
    tt_offset    : TT-star offset from science, arcsec (default: budget star)
    legacy       : True -> legacy budget (no wind-weighted bandwidth, no
                   layer-mismatch penalty; TT still honors tt_mag/tt_offset,
                   which at defaults reproduces the legacy TT term to <1%)
    ltao_tt_theta0_gain : effective tilt-theta0 gain from tomography, applied
                   as (1/gain)^(5/6) on the TT-star tilt-anisoplanatism rows
                   AND on the angular-aniso charge at the laser/asterism-
                   centre offset when mode="ltao" (never single-beacon,
                   never legacy).
                   None -> DEF_LTAO_TT_THETA0_GAIN (1.0 = disabled per
                   KAON 1303 section 5.5: single-TT-star LTAO gets no
                   tilt-aniso reduction). See the tiptilt.py block.

    Returns the Strehl at lam_nm (high-order x tip-tilt). The individual nm
    terms are available from lgs_budget_terms() with the same arguments.
    """
    t = lgs_budget_terms(eps_total, eps_freeatm, telescope, mode, cn2_bins,
                         tt_mag, tt_offset, lgs_offset, legacy, bw_factor,
                         v_ground, v_free, aniso_scale, tt_sensor=tt_sensor,
                         tt_spot_theta=tt_spot_theta, strap_law=strap_law,
                         ltao_tt_theta0_gain=ltao_tt_theta0_gain)
    ho = np.sqrt(t["alt"]**2 + t["ang"]**2 + t["fit"]**2 + t["bw"]**2
                 + t["scint"]**2 + t["meas"]**2 + t["nafoc"]**2
                 + t["stat_tel"]**2 + t["stat_calib"]**2 + t["stat_dm"]**2
                 + t["stat_inst"]**2 + t["stat_reg"]**2 + t["margin"]**2)
    return marechal_strehl(ho, lam_nm) * marechal_strehl(t["tt"], lam_nm)
