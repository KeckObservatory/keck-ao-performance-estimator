"""The AO PSF / FWHM model: a Racine-style two- or three-component profile
(tilt-smeared Airy core + corrected-band Moffat shoulder + Kolmogorov seeing
wings), and the THREE FWHM conventions derived from it -- half-max
(psf_fwhm_mas), a no-background least-squares Gaussian fit (fwhm_gaussfit_mas),
and a free-background least-squares Gaussian fit (fwhm_gaussfit_sky_mas). See
the module-level comments below for the physics, and the "REAL MEASUREMENT
TOOLS" block below for what each convention is modeled on and how that was
established (2026-07-19, from the actual IDL source of the tools in question).
This is a straight, behavior-preserving move out of the historical
ao_strehl_timeline.py flat script."""
import sys as _sys

import numpy as np

from .constants import TEL_DIAMETER_M, MOFFAT_BETA_KOLM, REF_TOTAL, NM_PER_MAS
from .marechal import marechal_strehl

# =============================================================================
#  FWHM ESTIMATION  --  Racine-style two-component PSF (added 2026-07-10)
#
#  Strehl alone does not determine the FWHM, so --report fwhm/both builds a
#  standard AO PSF model per sample: a diffraction-limited Airy core carrying
#  energy fraction S (the Strehl) plus a seeing halo carrying (1 - S), and the
#  FWHM is found numerically as the first half-maximum crossing of the summed
#  profile. Model choices (Eduardo, 2026-07-10):
#    * core : true Airy pattern for a circular aperture of TEL_DIAMETER_M
#             (9.96 m = the circle inscribed in the Keck hexagonal aperture;
#             segment gaps / central obscuration neglected);
#    * halo : Moffat profile with beta = 4.765, the value that matches a
#             Kolmogorov seeing disk (Trujillo et al. 2001, MNRAS 328, 977);
#             halo FWHM = line-of-sight 500 nm seeing x (lam/500nm)^(-1/5)
#             (pure Kolmogorov wavelength scaling; no outer-scale correction).
#  Behavior: FWHM ~= 1.029 lam/D while the core dominates, then transitions to
#  the seeing-disk width as S collapses -- the on-sky "core disappears" jump.
# =============================================================================

#  TT CEILING -- "a loop cannot do worse than no correction" (Eduardo,
#  2026-07-13). tt_wfe_nm's anisoplanatism rows grow without bound with the
#  TT-star offset (and with the prediction tab's aniso re-weighting), but
#  physically the worst case is NO tilt correction at all: the science target
#  becomes a seeing-limited spot whose image motion is the full uncorrected
#  atmospheric tilt. One-axis Kolmogorov G-tilt over the aperture:
#      sigma^2 = 0.170 (D/r0)^(5/3) (lambda/D)^2   [rad^2, achromatic:
#  r0 ~ lambda^(6/5) cancels the lambda/D]. Evaluated at the reference seeing
#  this is ~110 mas one-axis; it scales with seeing exactly as the budget's
#  s_tot (both ~ r0^(-5/6)), so the ceiling applied in tt_wfe_nm is
#  OPEN_LOOP_TILT_ONEAXIS_MAS * s_tot. Far below it in all normal budgets --
#  it only binds for extreme offsets / faint stars / aloft-heavy scenarios.
#
#  OUTER SCALE (2026-08-07, from KAON 1318 "KAPA Sky Coverage Assessment").
#  The Kolmogorov form above assumes an INFINITE outer scale, and tilt is the
#  single most outer-scale-sensitive mode there is -- a finite L0 removes
#  exactly the large-scale power that tilt is made of, so infinite-L0
#  Kolmogorov OVERSTATES it. KAON 1318 Table 1 gives the uncorrected
#  (0 Hz) two-axis TT for L0 = 10 / 25 / 50 / 100 m as 68.9 / 85.6 / 105.2 /
#  125.5 mas, i.e. one-axis 48.7 / 60.5 / 74.4 / 88.7 mas. The Kolmogorov
#  expression above gives 110 mas one-axis -- ABOVE even the L0 = 100 m row,
#  which is the tell: the ceiling was set by a law that does not apply.
#
#  So the ceiling is now anchored to Table 1 by interpolating it in log L0,
#  and only the SEEING scaling is kept from the expression above (Table 1 is
#  quoted at one set of conditions; the s_tot ~ r0^(-5/6) scaling applied in
#  tt_wfe_nm is unchanged). Consequences, stated plainly:
#    * the ceiling DROPS ~32% at the default L0, so it binds sooner. That is
#      the point -- it is a physical bound, and it was set too high to ever
#      bite.
#    * it does not move any normal budget: the default TT star (R=15.2 @
#      19.3") totals ~20 mas one-axis, nowhere near either value. What
#      changes is the far-off-axis TRICK regime, where the spot-inflation
#      factor multiplies the measurement row up past 60 mas -- exactly the
#      wide-asterism case over the 102" FOV that KAON 1318 is written about.
#  CAVEAT: Table 1's rows are atmosphere AND telescope jitter merged ("now
#  more realistic and in line with observed on-sky performance"), so this
#  ceiling is a total-image-motion bound, which is the right thing for a
#  "no correction at all" ceiling but is NOT a pure atmospheric tilt.
#  L0 DEFAULT SETTLED AT 50 m (Eduardo, 2026-08-09; was 25 m for two days).
#  The source ambiguity: KAON 1318's Figures 1-4 use L0 = 25 m, but its own
#  Figure 5 caption AND KAON 1303 section 5.5 state 50 m is the Mauna Kea
#  MEDIAN, and KAON 1303's Table 41/42 base cases (the very table this
#  ceiling interpolates) are quoted around it. Keck-specific heritage runs
#  larger still (van Dam et al. 2004 adopted L0 = 75 m fit to Keck data),
#  and for a "cannot do worse than no correction" BOUND the smaller L0 was
#  the optimistic-side choice -- the wrong side for a bound. Ceiling moves
#  60.5 -> 74.4 mas one-axis at reference; no normal budget is anywhere
#  near either value, only ceiling-limited tails (far-off-axis / very
#  faint) shift. L0 is log-normal night-to-night (10-100+ m): use
#  --outer-scale for per-night values.
OUTER_SCALE_M = 50.0            # default L0 (KAON 1318/1303 median)

#  KAON 1318 Table 1, 0 Hz row: uncorrected TT, TWO axes, mas rms.
_KAON1318_L0_M = (10.0, 25.0, 50.0, 100.0)
_KAON1318_TT_TWOAXIS_MAS = (68.9, 85.6, 105.2, 125.5)


def _open_loop_tilt_oneaxis_mas(eps_tot_500, outer_scale_m=None):
    """One-axis uncorrected image motion (mas) at 500 nm seeing eps_tot_500.

    outer_scale_m: None -> OUTER_SCALE_M, interpolating KAON 1318 Table 1 in
    log L0 (and holding its end values beyond the tabulated 10-100 m range,
    rather than extrapolating a fit nothing supports). Pass float('inf') for
    the original infinite-outer-scale Kolmogorov form, which is what this
    returned before 2026-08-07."""
    r0 = 0.98 * 500e-9 / np.radians(eps_tot_500 / 3600.0)
    kolm = float(np.degrees(np.sqrt(0.170) * (500e-9 / TEL_DIAMETER_M)
                            * (TEL_DIAMETER_M / r0) ** (5.0 / 6.0)) * 3.6e6)
    L0 = OUTER_SCALE_M if outer_scale_m is None else float(outer_scale_m)
    if not np.isfinite(L0):
        return kolm
    two_axis_ref = float(np.interp(np.log(np.clip(L0, _KAON1318_L0_M[0],
                                                  _KAON1318_L0_M[-1])),
                                   np.log(_KAON1318_L0_M),
                                   _KAON1318_TT_TWOAXIS_MAS))
    one_axis_ref = two_axis_ref / np.sqrt(2.0)      # Table 1 is two-axis
    # Table 1 is quoted at ONE set of conditions; carry the engine's own
    # seeing scaling by riding the Kolmogorov expression's shape, which has
    # the correct r0^(-5/6) dependence even though its normalization is wrong.
    kolm_ref = float(np.degrees(
        np.sqrt(0.170) * (500e-9 / TEL_DIAMETER_M)
        * (TEL_DIAMETER_M / (0.98 * 500e-9
                             / np.radians(REF_TOTAL / 3600.0))) ** (5.0 / 6.0))
        * 3.6e6)
    return one_axis_ref * (kolm / kolm_ref)


OPEN_LOOP_TILT_ONEAXIS_MAS = _open_loop_tilt_oneaxis_mas(REF_TOTAL)
#  the pre-2026-08-07 value, kept so the change is auditable and so
#  --outer-scale inf reproduces it exactly
OPEN_LOOP_TILT_ONEAXIS_MAS_KOLMOGOROV = _open_loop_tilt_oneaxis_mas(
    REF_TOTAL, outer_scale_m=float("inf"))


def set_outer_scale(outer_scale_m):
    """Set the atmospheric outer scale L0 (m) and recompute the open-loop
    tilt ceiling that depends on it. float('inf') restores the original
    infinite-L0 Kolmogorov ceiling.

    *** module-global rebind -- read it QUALIFIED ***
    This rebinds this module's OPEN_LOOP_TILT_ONEAXIS_MAS, so any other
    module that did `from .psf import OPEN_LOOP_TILT_ONEAXIS_MAS` holds a
    permanently stale snapshot and will never see the change. Consumers must
    do `from . import psf` and read `psf.OPEN_LOOP_TILT_ONEAXIS_MAS` fresh --
    the same hazard, and the same remedy, as budget.py's adjustable scalars
    (see that module's header). tiptilt.py is the only consumer and does
    exactly that."""
    global OUTER_SCALE_M, OPEN_LOOP_TILT_ONEAXIS_MAS
    OUTER_SCALE_M = float(outer_scale_m)
    OPEN_LOOP_TILT_ONEAXIS_MAS = _open_loop_tilt_oneaxis_mas(REF_TOTAL)
    # the package __init__ RE-EXPORTS these two names, so a caller reading
    # `engine.OPEN_LOOP_TILT_ONEAXIS_MAS` would otherwise see a stale value
    # while `engine.tt_wfe_nm` quietly followed the new one -- the worst
    # version of this hazard, since the two would disagree. Keep the
    # re-export in step. Guarded: psf is importable on its own.
    _pkg = _sys.modules.get(__package__)
    if _pkg is not None:
        _pkg.OUTER_SCALE_M = OUTER_SCALE_M
        _pkg.OPEN_LOOP_TILT_ONEAXIS_MAS = OPEN_LOOP_TILT_ONEAXIS_MAS
    return OPEN_LOOP_TILT_ONEAXIS_MAS

#  THREE-COMPONENT PSF (added 2026-07-10, validated on the 20260701 OSIRIS
#  frame set). A two-component core+seeing-disk PSF reproduces the measured
#  half-max FWHM (labelled "AO Strehl tool" below) but NOT what a Gaussian FIT
#  reads on the same frames (labelled "OSIRIS QL": ~1.2x higher in poor
#  conditions): the real PSF carries corrected-band residual light in a
#  SHOULDER at a few lambda/D that a half-max never feels but a fit does. So
#  the model halo is split:
#    * shoulder -- Moffat(beta=4.765) of FWHM = theta_c, carrying the
#      corrected-band residual energy (everything but the fitting error);
#      shape selection note: a uniform disk of radius theta_c was tried first
#      and FALSIFIED by the frame set (too flat -- a fit reads it as sky);
#    * wings    -- the uncorrected seeing disk, carrying the fitting-error
#      share. Energy partition is variance-proportional (first order):
#      E_wings = (1-S_ho) * (sigma_fit/sigma_ho)^2, shoulder the rest.
#  Validation against 32 OSIRIS frames of 20260701 00:35-01:36 (K1, X~1.8):
#      half-max  66.0 mas  vs "AO Strehl tool"  62.8  (+5%)
#      gauss fit 71.8 mas  vs "OSIRIS QL"       76.7  (-6%)
#  with NO tuned constants (theta_c from the DM, split from the budget). NOTE
#  (2026-07-19): the pairing above is empirical (it's what agreed numerically
#  when this was tuned), not a confirmed mechanism match -- see "REAL
#  MEASUREMENT TOOLS" immediately below, established later from the tools'
#  actual IDL source. Kept as the validation anchor since the numbers and the
#  qualitative behavior (fit reads higher than half-max in poor conditions)
#  are real measurements; what to conclude about WHICH synthetic convention
#  should be compared to WHICH number is now open, not settled by this note.

#  REAL MEASUREMENT TOOLS (established 2026-07-19, from IDL source Eduardo
#  provided -- see fwhm_gaussfit_mas / fwhm_gaussfit_sky_mas docstrings for
#  which synthetic convention models which). TWO SEPARATE, INDEPENDENT real
#  programs are in play here, not one -- an earlier draft of this comment
#  conflated them (fixed same day, see the correction note at the bottom):
#    * The "AO Strehl tool" is its OWN STANDALONE IDL program --
#      NIRC2STREHL.pro, calling fwhmastro.pro, calling IDL's gauss2dfit: a
#      2-D ELLIPTICAL fit (independent sigma_x, sigma_y, rotation, FREE
#      constant/sky term) over a SQUARE box tied to the photometric aperture
#      radius (camera-dependent, ~7-21 px -> ~100-140 mas fit-domain
#      HALF-width for NIRC2's narrow/medium/wide cameras). FWHM =
#      (sigma_x+sigma_y)/2 * 2.355 * plate_scale. The box here is AUTOMATIC
#      (set by camera name), NOT hand-drawn -- it has nothing to do with
#      OSIRISSTREHL_QL2.pro or the quicklook tool below; confirmed directly
#      by Eduardo, not inferred.
#    * The OSIRIS quicklook tool ("qlook2") is a SEPARATE, independent
#      interactive tool for viewing/analyzing OSIRIS images generally --
#      also unrelated to the AO Strehl tool. It has (at least) two distinct
#      FWHM/Strehl measurement features of its own:
#        - a HAND-DRAWN-BOX Gaussian fit (per Eduardo: the user drags the
#          box with the mouse) -- almost certainly the cimwin_gauss_*.pro
#          family seen in the qlook2 directory listing (cimwin_gauss_base/
#          button/plot/range_event.pro), which sits alongside mpfitpeak.pro
#          in the same package. NOT YET READ DIRECTLY, so the mechanism
#          below is an inference (mpfitpeak.pro being present nearby, with a
#          free-background Gaussian as its default model), not confirmed
#          the way the other two routines are.
#        - OSIRISSTREHL_QL2.pro, the quicklook tool's own Strehl BUTTON --
#          per Eduardo, "a very rarely used bit ... almost no one uses it."
#          Hand-rolled: MPFITFUN of a 1-D RADIAL Gaussian (CENTGAUSS:
#          amplitude + sigma only, NO background -- sky is pre-subtracted,
#          not fit), isotropic, over an AUTOMATIC (not user-drawn) box that
#          works out to ~30.7 mas radius at K band regardless of camera
#          plate scale (the scale cancels algebraically in the tool's own
#          box-size formula). FWHM = sigma * 2.35 * plate_scale. Real code,
#          confirmed directly -- but being rarely-used, it is likely NOT
#          what actually produced the historical "OSIRIS QL 76.7 mas,
#          user-box, x*y" validation number (see below); the quicklook
#          tool's hand-drawn cimwin_gauss feature is the better candidate
#          for that, matching the original "user-box" description exactly.
#  Net: NEITHER program does a half-max crossing -- psf_fwhm_mas has no
#  confirmed real-tool analog. Its empirical agreement with the 20260701 "AO
#  Strehl tool" number (66.0 vs 62.8, +5%) is plausibly because that tool's
#  box, while an elliptical FIT rather than a threshold crossing, is tight
#  and automatic (~100-140 mas half-width) and so reads close to a half-max
#  crossing on a core-dominated PSF -- not because the mechanisms match.
#  fwhm_gaussfit_mas (no background, isotropic) models OSIRISSTREHL_QL2's
#  mechanism specifically -- real, but that rarely-used Strehl button is
#  probably not what produced the 76.7 mas number historically attributed to
#  "OSIRIS QL". fwhm_gaussfit_sky_mas (free background, isotropic, user
#  box_mas) models the quicklook tool's HAND-DRAWN cimwin_gauss feature (the
#  better candidate for that 76.7 mas number) -- box_mas being a real,
#  adjustable parameter is justified by THAT tool's box being hand-drawn, not
#  by the standalone AO Strehl tool (whose box is automatic). Neither fit
#  mode is a full mechanism match for the standalone AO Strehl tool itself,
#  which is 2-D elliptical; our whole PSF model is azimuthally symmetric, so
#  an elliptical fit has no meaningful analog here regardless of convention.
#  CORRECTION (same day, 2026-07-19): an earlier version of this comment (and
#  the accompanying KAON addendum / GUI-manual text, now fixed) wrongly
#  attributed the hand-drawn box and MPFITPEAK-default mechanism to the
#  standalone "AO Strehl tool" instead of the quicklook tool's cimwin_gauss
#  feature, and implied OSIRISSTREHL_QL2 was the representative quicklook
#  mechanism when it is in fact a rarely-used corner of that tool.
#
#  FOURTH CONVENTION (2026-08-07, Eduardo: "the FWHM estimation tends to
#  overestimate what the measured SR tool delivers ... add a 4th FWHM
#  estimation that uses exactly the same process as the SR tool").
#  fwhm_srtool_mas is that convention, and it is a different KIND of thing
#  from the three above: those three ask "what would a tool of THIS
#  MECHANISM read off this analytic profile", evaluated in the continuum.
#  This one asks "what does THE tool in this package -- the Measured-SR tab,
#  our port of NIRC2STREHL.pro/find_fwhm.pro -- read", and answers it by
#  running that tool's own code, unchanged, on the model PSF rendered as a
#  detector frame. Two things the continuum conventions cannot see are
#  therefore included. MEASURED sizes, 2026-08-07 -- the expected one turned
#  out to be the negligible one:
#    * PIXELS AND ANNULUS AVERAGING -- the whole effect, about +1.1 mas.
#      find_fwhm.pro does not evaluate a profile, it bins detector pixels
#      into 1-px-wide annuli and splines the 21 means. The innermost bin
#      (r < 0.707 px) is a MEAN, so the apparent peak sits below the true
#      one; the half-max LEVEL drops with it and the crossing moves
#      OUTWARD. The binning therefore BROADENS -- the tool reads wider than
#      a continuum half-max of the same PSF, not narrower.
#    * the ANNULUS SKY -- included for fidelity, worth nothing. The tool
#      measures on `image - sky` (mean over the 1.2-1.4" annulus), and the
#      pedestal that removes was expected to pull the crossing inward; on
#      this PSF it is at most 2e-4 of the peak (and at good Strehl it is
#      slightly NEGATIVE, the finite-grid Hankel core undershooting), which
#      moves the FWHM by well under 0.01 mas. Kept because it is what the
#      tool does, not because it changes the answer.
#  DIRECTION, stated plainly because it is the opposite of the guess that
#  prompted this: at a GIVEN Strehl the model does not read high against
#  the tool, it reads slightly LOW -- median -1.4 mas for the half-max
#  convention over 60 isolated-standard NIRC2 frames, -0.4 mas for this
#  one. So this convention removes most of a ~1 mas convention error; the
#  several-mas predicted-vs-delivered gaps are the STREHL prediction, and
#  no FWHM convention will close them. See regress/fwhm_srtool_model.py for
#  the full validation table.
#  It is nevertheless the only one of the four comparable to a measured
#  number WITHOUT a convention caveat -- which is what a
#  predicted-vs-delivered join needs.

#  TIP-TILT IS IMAGE MOTION, NOT A PEAK LOSS (fix, 2026-07-10).
#  The budget's TT term is built from one-axis mas rows in tt_wfe_nm() and only
#  then multiplied by NM_PER_MAS to be *expressed* as a wavefront error. Charging
#  it through Marechal (a peak loss) and stopping there leaves the PSF core
#  pinned at 1.029 lam/D, because an Airy core out-peaks the seeing halo ~100:1
#  at S=0.5 -- so the half-max crossing never leaves the core and the FWHM does
#  not respond to conditions. Physically the residual tilt SMEARS the core: the
#  core is convolved with a 2-D Gaussian of one-axis sigma = the TT jitter
#  (10-34 mas for the budgeted LGS TT star, i.e. comparable to or larger than
#  the 47 mas diffraction core). The convolution is done exactly in the OTF
#  domain, where it is a multiplication:
#      OTF(f) = MTF_circ(f/f_c) * exp(-2 pi^2 sigma^2 f^2),   f_c = D/lambda
#      core(r) = 2 pi \int_0^{f_c} OTF(f) J0(2 pi f r) f df      (energy = 1)
#  To avoid charging the tilt twice, the core carries the HIGH-ORDER-only Strehl
#  S_ho = S_total / Marechal(tt_nm): the convolution then supplies the real TT
#  peak loss. NOTE the resulting PSF peak is therefore not exactly the engine's
#  reported Strehl (a true jitter convolution is not the Marechal approximation);
#  the reported Strehl columns are untouched.

_HANKEL_CACHE = {}


def _hankel_tables(rho_max=40.0, n_rho=1601, n_u=193):
    """Precomputed, DIMENSIONLESS tables for the core convolution.

    With rho = r/(lambda/D) and u = f/f_c, the Hankel kernel J0(2 pi f r) becomes
    J0(2 pi u rho) -- independent of lambda, D and sigma. So the kernel and the
    circular-aperture MTF are built once, and each sample costs one matrix-vector
    product. Returns (rho, u, mtf, K, trapz_weights)."""
    key = (rho_max, n_rho, n_u)
    if key not in _HANKEL_CACHE:
        from scipy.special import j0
        u = np.linspace(0.0, 1.0, n_u)
        mtf = (2.0 / np.pi) * (np.arccos(np.clip(u, -1, 1))
                               - u * np.sqrt(np.clip(1.0 - u * u, 0.0, None)))
        rho = np.linspace(0.0, rho_max, n_rho)
        K = j0(2.0 * np.pi * np.outer(rho, u))
        du = u[1] - u[0]
        tw = np.full(n_u, du); tw[0] *= 0.5; tw[-1] *= 0.5   # trapezoid
        _HANKEL_CACHE[key] = (rho, u, mtf, K, tw)
    return _HANKEL_CACHE[key]


_CORE_Q_STEP = 0.005      # quantization of q = sigma_tt/(lambda/D) for caching
_CORE_CACHE = {}


def _core_profile(q):
    """Energy-normalized radial profile of the tilt-smeared diffraction core,
    on the dimensionless rho = r/(lambda/D) grid.

    The profile depends on the jitter ONLY through q = sigma_tt/(lambda/D), so
    it is cached on a quantized q (step 0.005, i.e. a few hundredths of a mas of
    jitter -> well under 0.1 mas of FWHM). Without this, each of the ~1000
    per-night samples pays its own Hankel matrix-vector product."""
    qb = round(float(q) / _CORE_Q_STEP)
    hit = _CORE_CACHE.get(qb)
    if hit is None:
        rho, u, mtf, K, tw = _hankel_tables()
        qq = qb * _CORE_Q_STEP
        otf = mtf * np.exp(-2.0 * np.pi ** 2 * qq ** 2 * u ** 2)
        hit = (rho, 2.0 * np.pi * (K @ (otf * u * tw)))
        _CORE_CACHE[qb] = hit
    return hit


def _psf_profile(strehl, eps500_los, lam_nm, tt_nm=0.0, fit_nm=None,
                 n_act=None, D_m=TEL_DIAMETER_M):
    """Build the (2- or 3-component) energy-normalized radial PSF profile.

    Components: tilt-smeared Airy core (energy S_ho = S_total/Marechal(tt));
    when fit_nm AND n_act are given, a Moffat shoulder of FWHM = theta_c =
    (n_act/2) lam/D carrying the corrected-band residual energy, and seeing
    wings carrying the fitting-error share (variance-proportional split);
    otherwise all non-core energy sits in the seeing wings (legacy 2-comp).

    Returns (I, aux) with I(r) over r in RADIANS, or None if the seeing is
    unusable. aux carries lam_D, the dimensionless core grid, halo width."""
    if not np.isfinite(eps500_los) or eps500_los <= 0:
        return None
    lam = lam_nm * 1e-9
    lam_D = lam / D_m
    halo_fwhm_rad = (eps500_los * (lam_nm / 500.0) ** (-0.2)) / 206265.0

    S_tot = float(strehl) if np.isfinite(strehl) else 0.0
    S_tot = min(max(S_tot, 0.0), 1.0)
    tt_nm = float(tt_nm) if np.isfinite(tt_nm) else 0.0
    S_tt = marechal_strehl(tt_nm, lam_nm) if tt_nm > 0 else 1.0
    S_ho = min(S_tot / S_tt, 1.0) if S_tt > 0 else S_tot
    sigma_rad = (tt_nm / NM_PER_MAS) * 1e-3 / 206265.0
    rho, core_hat = _core_profile(sigma_rad / lam_D)

    beta = MOFFAT_BETA_KOLM
    alpha_w = halo_fwhm_rad / (2.0 * np.sqrt(2.0 ** (1.0 / beta) - 1.0))
    wing_peak = (beta - 1.0) / (np.pi * alpha_w ** 2)

    # corrected-band shoulder (see the 3-component block comment above)
    E_sh, alpha_sh, sh_peak = 0.0, None, 0.0
    if (fit_nm is not None and n_act is not None and np.isfinite(fit_nm)
            and 0.0 < S_ho < 1.0):
        sig_ho = (lam_nm / (2.0 * np.pi)) * np.sqrt(-np.log(S_ho))
        frac_fit = min((max(float(fit_nm), 0.0) / sig_ho) ** 2, 1.0)
        E_sh = (1.0 - S_ho) * (1.0 - frac_fit)
        theta_c = (float(n_act) / 2.0) * lam_D
        alpha_sh = theta_c / (2.0 * np.sqrt(2.0 ** (1.0 / beta) - 1.0))
        sh_peak = (beta - 1.0) / (np.pi * alpha_sh ** 2)
    E_w = 1.0 - S_ho - E_sh

    def I(r):
        r = np.asarray(r, dtype=float)
        out = E_w * wing_peak * (1.0 + (r / alpha_w) ** 2) ** (-beta)
        if S_ho > 0.0:
            core = np.interp(r / lam_D, rho, core_hat,
                             left=core_hat[0], right=0.0)
            out = out + S_ho / lam_D ** 2 * core
        if E_sh > 0.0:
            out = out + (E_sh * sh_peak
                         * (1.0 + (r / alpha_sh) ** 2) ** (-beta))
        return out

    return I, dict(lam_D=lam_D, rho=rho, halo_fwhm_rad=halo_fwhm_rad,
                   S_ho=S_ho)


_GFIT_TABLES = {}


def _gaussfit_tables(R_mas=300.0, n_r=512):
    """Precomputed design matrices for the simulated quick-look Gaussian fits:
    radial least squares with uniform 2-D pixel weighting (w = r dr), over
    r < R_mas (the fit-domain radius -- see fwhm_gaussfit_mas /
    fwhm_gaussfit_sky_mas for what real tool/feature this approximates and
    why it is a real, callable parameter rather than a fixed constant: the
    OSIRIS quicklook tool's hand-drawn-box Gaussian-fit feature has no single
    correct box size to hardcode -- see the module-level "REAL MEASUREMENT
    TOOLS" comment). Returns (r, s, w, Gw, gg, Sg):
      * no-background fit (free amplitude only): best-fit sigma is
        argmax_s (sum w G_s I)^2 / (sum w G_s^2) = argmax_s (Gw@y)^2/gg;
      * free-background fit (amplitude + constant): needs the additional
        Sg = sum_r w*G_s (and Sw = w.sum(), computed by the caller) to solve
        the 2x2 normal equations -- see fwhm_gaussfit_sky_mas.
    One matvec per sample either way."""
    key = (R_mas, n_r)
    if key not in _GFIT_TABLES:
        r = np.linspace(0.5, R_mas, n_r)                       # mas
        w = r
        s = np.concatenate([np.arange(6.0, 120.0, 0.5),
                            np.arange(120.0, 420.0, 4.0)])     # sigma, mas
        G = np.exp(-r[None, :] ** 2 / (2.0 * s[:, None] ** 2))
        Gw = G * w
        gg = np.einsum("ij,ij->i", Gw, G)
        Sg = Gw.sum(axis=1)
        _GFIT_TABLES[key] = (r, s, w, Gw, gg, Sg)
    return _GFIT_TABLES[key]


def fwhm_gaussfit_mas(strehl, eps500_los, lam_nm, tt_nm=0.0, fit_nm=None,
                      n_act=None, D_m=TEL_DIAMETER_M, box_mas=300.0):
    """FWHM (mas) that a NO-BACKGROUND quick-look GAUSSIAN FIT would report.

    A free-amplitude, no-sky, isotropic least-squares Gaussian over pixels
    within box_mas of the peak, applied to the SAME 3-component model PSF as
    psf_fwhm_mas. This models OSIRISSTREHL_QL2.pro specifically -- the OSIRIS
    quicklook tool's own Strehl BUTTON (a separate, independent tool from the
    "AO Strehl tool"; see the module-level "REAL MEASUREMENT TOOLS" comment).
    It fits a 1-D radial Gaussian (CENTGAUSS) with NO free background (sky is
    pre-subtracted before the fit, same as here) and is isotropic like this
    model, over an AUTOMATIC box that works out to ~30.7 mas radius at K band.
    Per Eduardo this Strehl button is "a very rarely used bit" of the
    quicklook tool -- real code, confirmed from source, but probably NOT what
    actually produced the historical "OSIRIS QL 76.7 mas" validation number
    (the quicklook tool's separate hand-drawn-box feature is the better
    candidate for that; see fwhm_gaussfit_sky_mas). box_mas defaults to 300
    here only to preserve the already-validated 20260701 numbers (71.8
    predicted vs 76.7 measured); pass box_mas=~31 to approximate
    OSIRISSTREHL_QL2's actual fit domain instead, at the cost of no longer
    matching that validation anchor. A fit feels the corrected-band shoulder
    that a half-max never touches (when box_mas is large enough to reach it),
    so in poor conditions this can read systematically higher than
    psf_fwhm_mas. Reported alongside psf_fwhm_mas and fwhm_gaussfit_sky_mas so
    all three conventions are never mistaken for one another."""
    built = _psf_profile(strehl, eps500_los, lam_nm, tt_nm, fit_nm, n_act, D_m)
    if built is None:
        return np.nan
    I, aux = built
    r, s, w, Gw, gg, Sg = _gaussfit_tables(box_mas)
    y = I(r * 1e-3 / 206265.0)                 # mas -> rad
    score = (Gw @ y) ** 2 / gg
    i = int(np.argmax(score))
    if 0 < i < len(s) - 1:                     # parabolic refinement
        a, b, c = score[i - 1], score[i], score[i + 1]
        denom = a - 2 * b + c
        s_best = s[i] + (0.5 * (a - c) / denom * (s[i + 1] - s[i])
                         if denom != 0 else 0.0)
    else:
        s_best = s[i]
    return 2.3548 * float(s_best)


def fwhm_gaussfit_sky_mas(strehl, eps500_los, lam_nm, tt_nm=0.0, fit_nm=None,
                          n_act=None, D_m=TEL_DIAMETER_M, box_mas=300.0):
    """FWHM (mas) that a FREE-BACKGROUND quick-look GAUSSIAN FIT would report.

    A free-amplitude, free-CONSTANT (background/sky), isotropic least-squares
    Gaussian over pixels within box_mas of the peak, applied to the SAME
    3-component model PSF as psf_fwhm_mas / fwhm_gaussfit_mas. This models
    the OSIRIS quicklook tool's HAND-DRAWN-box Gaussian-fit feature (the user
    drags the box with the mouse) -- likely the cimwin_gauss_*.pro family
    seen alongside mpfitpeak.pro in the quicklook tool's own package
    ("qlook2"); NOT read directly, so the mechanism below is an informed
    inference, not confirmed from source the way OSIRISSTREHL_QL2.pro and
    fwhmastro.pro are. mpfitpeak.pro's default Gaussian model fits a
    Gaussian PLUS a free constant term (NTERMS=4 unless overridden):
    y = A*exp(-0.5*((x-x0)/sigma)^2) + c. This quicklook feature -- NOT the
    separate, standalone "AO Strehl tool" (that one is gauss2dfit-based, see
    the module-level "REAL MEASUREMENT TOOLS" comment) -- is also the better
    candidate for the historical "OSIRIS QL 76.7 mas, user-box" validation
    number, since OSIRISSTREHL_QL2.pro (what fwhm_gaussfit_mas models) is a
    rarely-used corner of the quicklook tool. Unlike fwhm_gaussfit_mas, a
    free background can absorb flat halo/sky flux inside the box instead of
    forcing it into a wider fitted sigma. box_mas has no historically-
    validated default here (the real feature's box is hand-drawn, not fixed
    or automatic), so it defaults to 300 only for consistency with
    fwhm_gaussfit_mas; treat it as a real knob to explore, not a calibrated
    constant.

    Solves the 2-parameter (amplitude, background) weighted linear
    least-squares problem in closed form for each candidate sigma on the
    precomputed grid, then parabolically refines the argmax the same way as
    fwhm_gaussfit_mas. Candidate sigmas for which the amplitude/background
    system is near-singular (G_s nearly constant over the box, at large sigma
    relative to a small box_mas) are excluded via the non-finite score guard."""
    built = _psf_profile(strehl, eps500_los, lam_nm, tt_nm, fit_nm, n_act, D_m)
    if built is None:
        return np.nan
    I, aux = built
    r, s, w, Gw, gg, Sg = _gaussfit_tables(box_mas)
    y = I(r * 1e-3 / 206265.0)                 # mas -> rad
    Sw = float(w.sum())
    Sy = float((w * y).sum())
    Sgy = Gw @ y
    det = gg * Sw - Sg ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        A = (Sgy * Sw - Sg * Sy) / det
        b = (gg * Sy - Sg * Sgy) / det
        score = A * Sgy + b * Sy
    score = np.where(np.isfinite(score), score, -np.inf)
    i = int(np.argmax(score))
    if 0 < i < len(s) - 1:                     # parabolic refinement
        a2, b2, c2 = score[i - 1], score[i], score[i + 1]
        denom = a2 - 2 * b2 + c2
        s_best = s[i] + (0.5 * (a2 - c2) / denom * (s[i + 1] - s[i])
                         if denom != 0 else 0.0)
    else:
        s_best = s[i]
    return 2.3548 * float(s_best)


def psf_fwhm_mas(strehl, eps500_los, lam_nm, tt_nm=0.0, D_m=TEL_DIAMETER_M,
                 fit_nm=None, n_act=None):
    """Half-max FWHM (mas) of the AO PSF for one sample: a tilt-smeared
    diffraction core carrying the high-order Strehl, plus (when fit_nm/n_act
    are given) the corrected-band Moffat shoulder at theta_c and the seeing
    wings -- see _psf_profile and the 3-component block comment.

    strehl      : TOTAL Strehl at lam_nm as reported by the engine (HO x TT)
    eps500_los  : line-of-sight total seeing at 500 nm, arcsec (halo width)
    lam_nm      : science wavelength, nm
    tt_nm       : the budget's tip-tilt term (nm RMS). Converted back to the
                  one-axis image-motion jitter it really is via NM_PER_MAS,
                  used to smear the core, and divided out of the Strehl so the
                  tilt is not charged twice. tt_nm=0 -> pure Airy core.
    fit_nm      : the per-sample fitting-error term (nm RMS) -- sets the
                  shoulder/wings energy split. None -> legacy 2-component.
    n_act       : DM actuators across the pupil -> theta_c = (n_act/2) lam/D.
    Returns NaN if the seeing is unusable; the seeing-disk width if strehl<=0.
    This is the HALF-MAX convention: the first radius where the profile falls
    to half its peak. NONE of the real measurement tools/features traced so
    far (see the module-level "REAL MEASUREMENT TOOLS" comment -- the
    standalone "AO Strehl tool", and the OSIRIS quicklook tool's two separate
    fit features) actually compute a half-max crossing -- they all fit a
    Gaussian, of one flavor or another. This convention tracked the 20260701
    "AO Strehl tool" number closely (+5%), which is why it is paired with
    that number in the validation block, but that agreement is now
    understood to be empirical (a tight elliptical fit reads close to a
    half-max crossing on a core-dominated PSF) rather than a mechanism
    match -- the AO Strehl tool doesn't do a half-max crossing either. See
    fwhm_gaussfit_mas (no background -- models OSIRISSTREHL_QL2.pro) and
    fwhm_gaussfit_sky_mas (free background -- models the quicklook tool's
    hand-drawn-box feature) for the two fit-based conventions.
    """
    if not np.isfinite(eps500_los) or eps500_los <= 0:
        return np.nan
    halo_fwhm_mas = (eps500_los * (lam_nm / 500.0) ** (-0.2)) * 1000.0
    if not (np.isfinite(strehl) and strehl > 0.0):
        return halo_fwhm_mas                       # no core: seeing disk

    from scipy.optimize import brentq              # lazy: scipy only for FWHM

    I, aux = _psf_profile(strehl, eps500_los, lam_nm, tt_nm, fit_nm, n_act,
                          D_m)
    rho, lam_D = aux["rho"], aux["lam_D"]
    r_grid = rho * lam_D

    half = float(I(0.0)) / 2.0
    # Bracket the FIRST half-max crossing. It almost always lies inside the
    # core (rho < 6), so scan that dense slice first and only fall back to the
    # full grid when the core has collapsed and the crossing is out in the halo.
    n_near = int(np.searchsorted(rho, 6.0))
    for r_scan in (r_grid[:n_near], r_grid):
        below = np.nonzero(I(r_scan) < half)[0]
        if len(below) and below[0] > 0:
            i = int(below[0])
            r_half = brentq(lambda r: float(I(r)) - half,
                            r_scan[i - 1], r_scan[i])
            return 2.0 * r_half * 206265.0e3
    return halo_fwhm_mas                           # core gone: seeing disk


# --------------------------------------------------------------------------
#  FOURTH CONVENTION: what OUR OWN Measured-SR tab reads
# --------------------------------------------------------------------------
#  The frame geometry (which pixel falls in which 1-px annulus) does not
#  depend on the PSF at all -- only on the plate scale. Cache it, so a
#  per-sample call costs one vectorized I(r) over ~2000 radii instead of
#  rebuilding a coordinate grid every time (the field map evaluates this
#  thousands of times per redraw).
_SRTOOL_GRIDS = {}


def _srtool_grid(plate_scale_mas, half_px=21):
    """(radius-in-mas image, its shape, the centre pixel) for the synthetic
    detector frame fwhm_srtool_mas measures. The star sits at the CENTRE of
    a pixel: a deterministic, best-case sub-pixel phase, stated rather than
    randomized (a real star lands anywhere in its pixel, which adds scatter
    of a fraction of a mas but no bias worth modelling here)."""
    key = (float(plate_scale_mas), int(half_px))
    if key not in _SRTOOL_GRIDS:
        n = 2 * int(half_px) + 1
        c = float(half_px)
        yy, xx = np.mgrid[0:n, 0:n]
        r_mas = np.hypot(xx - c, yy - c) * float(plate_scale_mas)
        _SRTOOL_GRIDS[key] = (r_mas, c)
    return _SRTOOL_GRIDS[key]


def fwhm_srtool_mas(strehl, eps500_los, lam_nm, tt_nm=0.0, fit_nm=None,
                    n_act=None, D_m=TEL_DIAMETER_M, plate_scale_mas=None,
                    bg_inner_arcsec=None, bg_outer_arcsec=None):
    """FWHM (mas) that THIS PACKAGE'S OWN Measured-SR tab would report.

    Not a fourth idea about what a Gaussian fit does -- the SAME CODE the
    Measured-SR tab runs (`image_strehl.radial_profile_fwhm`, our port of
    find_fwhm.pro), applied to the same 3-component model PSF as
    psf_fwhm_mas, rendered onto a NIRC2 detector grid and sky-subtracted
    the way the tool does it. Added 2026-08-07 to remove the convention
    mismatch between the Measured-SR tab's PREDICTED and MEASURED FWHM
    boxes. Reads ~1.1 mas WIDER than psf_fwhm_mas, essentially all of it
    the 1-px annulus binning (the annulus sky is <= 2e-4 of the peak here
    and moves nothing) -- see the "FOURTH CONVENTION" block in the module
    header for the measured decomposition and the real-frame validation.

    Process, in the tool's order:
      1. render I(r) on a pixel grid at `plate_scale_mas`
         (default NIRC2 narrow, the camera the tab is used with);
      2. subtract the ANNULUS SKY -- the mean of the PSF over
         [bg_inner, bg_outer] (defaults = the tab's own 1.2"/1.4"). The
         annulus holds ~10^5 pixels of a radially symmetric function, so
         its pixel mean is evaluated as the exact area-weighted radial
         integral rather than by rendering a 283x283 frame for it;
      3. hand the result to radial_profile_fwhm and scale by the plate
         scale -- exactly line `fwhm_mas = radial_profile_fwhm(_meas -
         sky, x, y) * ps` in image_strehl.measure_strehl.

    Returns NaN if the seeing is unusable, the seeing-disk width if the
    core is gone (same guards as psf_fwhm_mas), and NaN if the tool itself
    refuses the profile (its -1.0 sentinel: peak off-centre or no half-max
    crossing) -- a negative FWHM must never leave this function, since
    every consumer treats FWHM as a positive quantity.
    """
    # deferred: nirc2/image_strehl are the MEASUREMENT layer and the engine
    # does not otherwise depend on them (docs/development.md). This one
    # convention deliberately does -- being the tool's own code is the whole
    # point -- but it stays a call-time dependency, not an import-time one.
    from .image_strehl import radial_profile_fwhm
    from .nirc2 import (NIRC2_BG_INNER_RADIUS_ARCSEC,
                        NIRC2_BG_OUTER_RADIUS_ARCSEC, NIRC2_PLATE_SCALE_MAS)

    if plate_scale_mas is None:
        plate_scale_mas = NIRC2_PLATE_SCALE_MAS["narrow"]
    if bg_inner_arcsec is None:
        bg_inner_arcsec = NIRC2_BG_INNER_RADIUS_ARCSEC
    if bg_outer_arcsec is None:
        bg_outer_arcsec = NIRC2_BG_OUTER_RADIUS_ARCSEC

    if not np.isfinite(eps500_los) or eps500_los <= 0:
        return np.nan
    halo_fwhm_mas = (eps500_los * (lam_nm / 500.0) ** (-0.2)) * 1000.0
    if not (np.isfinite(strehl) and strehl > 0.0):
        return halo_fwhm_mas                       # no core: seeing disk
    built = _psf_profile(strehl, eps500_los, lam_nm, tt_nm, fit_nm, n_act,
                         D_m)
    if built is None:
        return np.nan
    I, _aux = built

    MAS_TO_RAD = 1e-3 / 206265.0
    # (2) annulus sky, area-weighted: mean over the annulus of a radially
    # symmetric I is int I(r) 2 pi r dr / (pi (r2^2 - r1^2)).
    r1 = float(bg_inner_arcsec) * 1000.0
    r2 = float(bg_outer_arcsec) * 1000.0
    # 64 nodes: the integrand is a smooth Moffat tail over a 200 mas span and
    # the whole term is <= 2e-4 of the peak (see the header), so the
    # quadrature error is orders of magnitude below anything that could move
    # a half-max crossing. 512 was the first guess and cost ~6% of the call.
    rq = np.linspace(r1, r2, 64)
    sky = float(np.trapezoid(I(rq * MAS_TO_RAD) * rq, rq)
                / (0.5 * (r2 ** 2 - r1 ** 2)))

    # (1) render, (3) hand to the tool's own routine
    r_mas, c = _srtool_grid(plate_scale_mas)
    img = I(r_mas * MAS_TO_RAD) - sky
    fwhm_px = radial_profile_fwhm(img, c, c)
    if not (np.isfinite(fwhm_px) and fwhm_px > 0.0):
        return np.nan                              # the tool's -1 sentinel
    return float(fwhm_px) * float(plate_scale_mas)
