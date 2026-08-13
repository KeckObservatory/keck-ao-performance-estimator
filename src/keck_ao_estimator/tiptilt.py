"""The tip-tilt error budget: tt_wfe_nm (the STRAP/STRAP-legacy/TRICK
measurement rows, built from one-axis mas terms for any TT-star brightness
and offset) and ngs_tt_nm (the NGS-guide-star tilt residual used only by the
FWHM path, never the Strehl budget)."""
import numpy as np

from . import psf as _psf
from .constants import NM_PER_MAS

#  READ THE CEILING QUALIFIED, NEVER AS A BARE IMPORT. psf.set_outer_scale()
#  REBINDS psf.OPEN_LOOP_TILT_ONEAXIS_MAS, and `from .psf import
#  OPEN_LOOP_TILT_ONEAXIS_MAS` would freeze this module at the import-time
#  value -- the exact hazard budget.py's header documents for its own
#  adjustable scalars. Every use below goes through _psf. (Kept exported
#  under the old name for callers that only want the default.)
OPEN_LOOP_TILT_ONEAXIS_MAS = _psf.OPEN_LOOP_TILT_ONEAXIS_MAS

# Tip-tilt budget: TT_ATM_REF/TT_FIXED are unused elsewhere in the codebase
# (superseded by tt_wfe_nm's detailed one-axis-mas-row breakdown below) --
# kept as found; a pure code-move refactor doesn't remove dead code (that's a
# separate cleanup, flagged to the maintainer rather than done silently here).
TT_ATM_REF  = 151.0   # atmospheric tip-tilt (scales with TOTAL seeing)
TT_FIXED    = 63.0    # fixed tip-tilt error (seeing-independent)

# (3) PARAMETERIZED TIP-TILT STAR (--tt-mag / --tt-offset).  The legacy TT
#     budget (163.6 nm at reference seeing) was built for an R=15.2 TT star
#     19.3" off axis.  The TT error is now recomputed from the underlying
#     one-axis mas rows for any TT-star brightness and offset:
#       * measurement scales with photon noise: x 10^(-0.2 (15.2 - R))
#       * tilt + centroid anisoplanatism scale linearly with offset (0 on-axis)
#       * measurement / bandwidth / tilt-aniso rows scale with total seeing
#       * dispersion / NCP / wind shake / TT margin are fixed
#     Defaults (R=15.2 @ 19.3") reproduce the legacy TT budget to <1%.
DEF_TT_MAG    = 15.2   # R mag of the budgeted TT star
DEF_TT_OFFSET = 19.3   # arcsec off axis

#  REFINED STRAP MEASUREMENT ROW (recalibrated 2026-07-15).  STRAP centroids a
#  SEEING-LIMITED R-band spot, and the original sheet's photon-only row
#  (7.25 mas at R=15.2, slope 0.2) is far too optimistic against on-sky data:
#  the TRICK commissioning paper (Rampy et al., "Near-infrared tip-tilt
#  sensing at Keck", Table 1) gives three PAIRED same-star on-axis STRAP/TRICK
#  H-Strehl measurements; dividing out the common high-order Strehl implies
#  STRAP total TT of ~8.4 / 12.0 / 17.8 mas one-axis at R = 12.0 / 13.0 / 15.5
#  -- versus the sheet's ~6-7 mas.  The refined measurement row is the
#  least-squares fit through those points (floor rows subtracted):
#      sigma_meas = STRAP_MEAS_REF * 10**(STRAP_BETA*(R - STRAP_MAG0)) * s_tot
#  The sub-photon slope (0.116 < 0.2) reflects the closed loop's gain
#  optimization at the faint end.  At R = 9.9 this row is ~3.9 mas, so the
#  KAON 1542 SS6 on-sky validation (HD 18770, R=9.9 on-axis) shifts by
#  <~0.01 in predicted K Strehl (verified against the fetched 2025-12-04/06
#  nights).  The OLD row is NO LONGER user-selectable (the strap-legacy
#  CLI/GUI option was retired 2026-08-07 once the recalibration settled) but
#  survives internally: --legacy always uses it (it IS part of the frozen
#  2004 sheet), and ngs_tt_nm() builds the NGS FWHM path on it.
STRAP_MEAS_REF = 6.9    # one-axis mas at STRAP_MAG0, reference seeing
STRAP_MAG0     = 12.0   # R magnitude of the calibration anchor
STRAP_BETA     = 0.116  # faint-end slope (fit through the three Table-1 pts)

#  ON-SKY 2026 STRAP ROW (KAPA characterization campaign, finding #8).
#  The 2026 campaign's predicted-vs-delivered ladder (pvd_meta_v4, six
#  nights, ON-AXIS metric) shows the residual measured-minus-predicted
#  Strehl tracking TT magnitude: about -0.115 at R = 10, +0.084 at
#  R = 15.6, crossing zero at R = 13.2 +- 0.3.  Reading: the magnitude
#  DEPENDENCE is too steep.  At the reference profile the current row
#  makes SR(10)/SR(15.6) = 1.485 where the campaign measures 1.234;
#  matching that ratio, with the row re-anchored so the crossing
#  magnitude R = 13.2 is left unchanged, gives:
STRAP_MEAS_REF_ONSKY = 7.83   # one-axis mas at STRAP_MAG0
STRAP_BETA_ONSKY     = 0.070  # magnitude slope
STRAP_PIVOT_ONSKY    = 13.2   # R where the correction is zero by construction
#  TWO THINGS THIS DELIBERATELY DOES **NOT** DO, both because the data do
#  not support them:
#   1. It corrects the SHAPE only.  The magnitude-INDEPENDENT part of the
#      residual is not absorbed, because it carries a known measurement
#      bias (the on-axis plane fit's faint-star population bias, ~-0.05 on
#      0131-class fields) and the 0501-class elevated-turbulence offset.
#      Folding those into a sensor row would be fitting the model to our
#      own measurement systematics.
#   2. It does not attribute the WHOLE residual to this row.  That is
#      arithmetically impossible: matching the full residual would require
#      removing ~63600 nm^2 of TT variance at R = 15.6 from a term that
#      totals ~69500 nm^2 and has an R-INDEPENDENT floor of ~18500 nm^2
#      (bandwidth, wind shake, margin).  The fit would demand a NEGATIVE
#      magnitude slope -- TT error improving as the star gets fainter.
#  CAVEAT FOR THE REVIEWER: beta 0.070 is flatter still than the already
#  sub-photon 0.116 (photon-limited quad-cell is 0.2).  A measurement row
#  flatter than photon statistics is physically uncomfortable, and is the
#  reason this row is SELECTABLE rather than the default: the campaign
#  establishes the residual TREND robustly, but not that the STRAP
#  measurement row is where it belongs.  Faint-end predictions move a lot
#  (R = 19: SR 0.04 -> 0.24 at the reference profile), well beyond the
#  R <= 15.6 the campaign actually measured, so treat R > 16 as
#  extrapolation.  Select with --strap-law onsky2026.

#  STRAP FAINT-END STEEPENING (2026-08-09, Eduardo's performance-sheet
#  values for a 10" off-axis STRAP star: R=10/12/14/16/18 ->
#  8.0/9.4/15.7/36.2/102.6 mas one-axis TOTAL).  The sheet agrees with the
#  refined on-sky-calibrated row within +/-13% for R <= 14, but its
#  inverted MEASUREMENT row steepens continuously (0.145 -> 0.230 dex/mag)
#  -- the photon->background quadcell transition a constant slope lacks.
#  The sheet knee itself sits ~1-1.5 mag too bright vs the sky (it implies
#  ~28.5 mas total at R=15.5 where the MEASURED paired on-sky anchor is
#  17.8), so this is a HYBRID: the calibrated law holds through its last
#  measured anchor (R = 15.5) and the sheet's slopes are grafted beyond,
#  anchor-continuous.  Hybrid row: R=16: 22.2 / 17: 35.6 / 18: 58.6 /
#  19: 99.5 mas.  On the Besancon-matched + Hardy-convention sky-coverage
#  comparison this tracks the KAON 1318 requirement curve from ~28%
#  coverage outward (see keck_ao_experiments/skycoverage/
#  KAON1303_TRICK_FINDINGS.md section 4f).  Applied to BOTH selectable
#  strap laws (the steepening is sensor physics, independent of the
#  bright-end anchor law); strap-legacy (frozen 2004 sheet) is untouched.
#  Unconfirmed on sky beyond R=15.5 -- an R>16 ladder point would pin the
#  grafted knee.  Byte-identity note: the default budget star is R=15.2,
#  BELOW the knee, so all harness goldens are unchanged.
STRAP_FAINT_KNEE   = 15.5   # last measured anchor of the refined law
STRAP_BETA_FAINT1  = 0.204  # dex/mag, 15.5 < R <= 17.5 (sheet 14-16 slope)
STRAP_FAINT_KNEE2  = 17.5
STRAP_BETA_FAINT2  = 0.230  # dex/mag beyond 17.5 (sheet 16-18 slope)

#  TRICK IR TIP-TILT SENSOR (K1), vs the STRAP R-band quadcell.  TRICK
#  centroids the AO-corrected (laser-HO-loop), near-diffraction-limited
#  guide-star image in H or K, so it holds tip-tilt to a much fainter guide
#  magnitude than the seeing-limited R quadcell.  Only the MEASUREMENT row
#  differs; the tilt-anisoplanatism, bandwidth, wind-shake and margin rows are
#  telescope systematics shared with STRAP (tip-tilt is achromatic image
#  motion).  The measurement row is calibrated to the 2026-05-28 (M3) on-sky
#  LTAO K-Strehl-vs-guide-magnitude data (DIMM 0.69", theta0 22.5"): the loop
#  is floor-limited (flat Strehl) to a sensing-band magnitude ~14, then rolls
#  off steeply as the WFS runs out of signal.  One-axis mas:
#     sigma_meas = TRICK_MEAS_REF * 10**(TRICK_BETA*(m_wfs - TRICK_MAG0)) * s_tot
#  with m_wfs the guide magnitude IN THE SENSING BAND (H or K), and s_tot the
#  usual seeing scaling (worse seeing -> less flux in the corrected core).
TRICK_MEAS_REF = 5.34   # one-axis mas at TRICK_MAG0, reference seeing
TRICK_MAG0     = 14.0   # sensing-band magnitude where meas ~ the floor (knee)
TRICK_BETA     = 0.60   # faint-end roll-off exponent (steeper than the 0.2 of
                        #   pure photon noise: the loop nears its SNR limit)
#  TRICK spot degradation off-axis: the sensor centroids the AO-corrected core
#  AT THE STAR, whose quality falls with the star's angular anisoplanatism
#  from the laser -- so the TRICK measurement row inflates off-axis (the
#  commissioning paper: "beyond a certain off-axis distance STRAP will
#  outperform TRICK, since performance of the former is independent of
#  high-order correction").  Modeled as a spot-size multiplier
#  sqrt(1 + (theta/TRICK_SPOT_THETA)^2); at the paper's brightest cases the
#  row is so small the inflation is invisible out to 40-50" (as observed),
#  while a faint (m ~ 14) TRICK star loses to STRAP tens of arcsec out.
TRICK_SPOT_THETA = 25.0   # arcsec; e-folding-ish scale of the spot inflation
#  SPOT DEGRADATION IS SET BY THE EFFECTIVE ISOPLANATIC ANGLE (2026-07-26,
#  Eduardo): "the exact rate of the falloff determining when TRICK crosses
#  STRAP and when TRICK hits the seeing limited floor is based on the
#  effective isoplanatic angle of the observation."  TRICK_SPOT_THETA above
#  is the reference-profile value; pass spot_theta scaled by the
#  observation's theta0 to move the crossing.  A larger theta0 keeps the
#  core intact further out and pushes the crossing outward.
#
#  TRICK DOES CROSS STRAP.  The 0606 M15 same-star ladder has them EQUAL by
#  ~20" and STRAP AHEAD by 40": once the core is gone TRICK is centroiding a
#  seeing-limited spot with a sensor built for a diffraction-limited one, so
#  it can be WORSE than a quadcell designed for seeing-limited spots.
#  An earlier version of this block imposed TRICK_SPOT_MAX = 1.40 and a cap
#  at the STRAP row, on the mistaken reading that TRICK could never fall
#  below STRAP.  Both are REMOVED: they forbade the observed crossing.
#  OPEN CALIBRATION: with TRICK_SPOT_THETA = 25" the model still has TRICK
#  ahead of STRAP at 51" (ratio 0.94 at H = R = 13), so it does NOT yet
#  reproduce the M15 crossing at ~20".  Fitting theta0-scaled spot
#  degradation to the 0606/0530 same-star ladders is the next step and needs
#  those ladders' theta0 values; not guessed here.
TRICK_SPOT_MAX = float("inf")   # no artificial ceiling (see above)

#  SEEING-LIMITED KNEE ON THE MAGNITUDE ROW.  Separate from the off-axis
#  question above, and still required: TRICK_BETA = 0.6 is a core-loss
#  roll-off calibrated over H ~ 9.5-14.5.  Extrapolated it runs away (337
#  mas at H = 17), hits the open-loop tilt ceiling by H ~ 16.5, and Marechal
#  maps that to EXACTLY ZERO Strehl -- at a magnitude where TRICK is stated
#  to close the loop.  The core can only be lost once; beyond that the spot
#  is seeing-limited and cannot grow further, so only signal-to-noise still
#  degrades and the row continues on the PHOTON-LIMITED slope from the knee.
#  The knee position inherits the theta0 dependence above.
TRICK_KNEE_MAG   = 14.7   # sensing mag where the core is gone at theta0_ref
TRICK_BETA_FAINT = 0.20   # photon-limited slope beyond the knee
TRICK_SEEING_LIMITED_CAP = True

#  TOMOGRAPHIC TILT-ANISOPLANATISM REDUCTION (LTAO ONLY; Eduardo 2026-08-07).
#  The tilt-anisoplanatism rows below (aniso 9.17 mas + cent 1.28 mas, both
#  linear in TT-star offset) come from the 2004 single-conjugate budget sheet
#  and are charged IDENTICALLY to single-beacon and LTAO -- but the
#  differential tilt between the TT star's direction and the science
#  direction is generated by ALTITUDE turbulence (focus/astigmatism modes
#  aloft produce field-varying tilt), and those modes are exactly what the
#  LGS asterism senses and tomography reconstructs.  The lasers cannot sense
#  GLOBAL tilt (that still needs the star), but a tomographic reconstructor
#  can correct the field DEPENDENCE of tilt, so under LTAO the star's tilt
#  decorrelates over a larger effective theta0 than the sheet assumes.
#  Applied in lgs_budget_terms as (1/gain)^(5/6) on the TT aniso rows only
#  (mode == "ltao" and not legacy; the HO ang term and every single-beacon
#  path are untouched, and --legacy stays byte-faithful to the frozen sheet).
#  At 30" offset those rows are ~85% of the TT variance (~93% at the 51"
#  patrol edge), so this is decisive exactly in the sky-coverage regime.
#  DEFAULT 1.0 (OFF) SINCE 2026-08-09, PER KAON 1303 SECTION 5.5: for a
#  SINGLE TT star -- the descoped KAPA reality and everything this
#  estimator models -- the LGS-tomography null modes ("a field-varying tilt
#  that concentrates mostly in the quadratic modes (focus and astigmatism)")
#  are NOT recoverable: "for three TT stars, the low-order tomographic
#  algorithm will estimate those quadratic terms, whereas for two and one
#  TT star those errors manifest in the final budget", and "the great leap
#  in performance occurs when going from one to two TT stars" (Fig. 530).
#  So the mechanism argued above requires >= 2 TT stars; with one star the
#  tilt aniso is charged in full, same as single-conjugate.  The on-sky
#  check agreed by omission: the M79 sLGS-vs-LTAO contrast was 1.1-sigma,
#  inconclusive.  The knob stays (--ltao-tt-theta0-gain, GUI LGS tab) in
#  case a multi-star mode or contrary on-sky ladder ever revives it.
DEF_LTAO_TT_THETA0_GAIN = 1.0


def tt_wfe_nm(s_tot, tt_mag=DEF_TT_MAG, tt_offset=DEF_TT_OFFSET,
              aniso_scale=1.0, sensor="strap", spot_theta=None,
              strap_law="sheet"):
    """Tip-tilt WFE (nm RMS) for a TT star at tt_offset arcsec, total-seeing
    scale factor s_tot. Built from the budget's one-axis mas rows (see
    refinement (3) above).

    sensor: 'strap' -> the R-band quadcell with the REFINED measurement row
    (recalibrated to the paired on-sky STRAP/TRICK data; see the STRAP block
    above); tt_mag is the R magnitude. 'strap-legacy' -> the original sheet's
    photon-only row (7.25 mas at R=15.2, slope 0.2), INTERNAL-ONLY since
    2026-08-07: used by the --legacy frozen-2004-sheet budget and by
    ngs_tt_nm(), no longer a CLI/GUI choice. 'trick' -> the K1 IR sensor;
    tt_mag is the guide magnitude in
    the SENSING band (H or K), the measurement row uses the diffraction-
    limited-centroiding law, and it inflates off-axis as the star's corrected
    core degrades (spot_theta: the star-laser separation driving that
    degradation; None -> use tt_offset as a proxy).

    aniso_scale multiplies the ANISOPLANATISM rows only. The budget's seeing
    scalings assume the REFERENCE profile's altitude distribution; when a
    scenario decouples theta0 from the seeing (the GUI's prediction tab), the
    altitude re-weighting (theta0_ref/theta0)^(5/6) enters here. 1.0 (the
    default, used by every night-data path) is the reference shape."""
    off_fac  = tt_offset / DEF_TT_OFFSET
    if sensor == "trick":
        meas = TRICK_MEAS_REF * 10.0 ** (TRICK_BETA * (tt_mag - TRICK_MAG0)) * s_tot
        th_spot = tt_offset if spot_theta is None else spot_theta
        meas *= min(np.sqrt(1.0 + (th_spot / TRICK_SPOT_THETA) ** 2),
                    TRICK_SPOT_MAX)     # saturates: see TRICK_SPOT_MAX
        if TRICK_SEEING_LIMITED_CAP and tt_mag > TRICK_KNEE_MAG:
            # Beyond the knee the corrected core is gone and the spot is
            # SEEING-LIMITED: it cannot grow further, so the steep
            # core-loss exponent stops applying and only signal-to-noise
            # still degrades. Continue from the knee on the photon-limited
            # slope instead. This does NOT cap TRICK at STRAP's level --
            # the M15 ladder shows TRICK crossing STRAP off-axis, so TRICK
            # centroiding a seeing-limited spot can be WORSE than a
            # quadcell built for one.
            knee = (TRICK_MEAS_REF
                    * 10.0 ** (TRICK_BETA * (TRICK_KNEE_MAG - TRICK_MAG0))
                    * s_tot)
            knee *= min(np.sqrt(1.0 + (th_spot / TRICK_SPOT_THETA) ** 2),
                        TRICK_SPOT_MAX)
            meas = knee * 10.0 ** (TRICK_BETA_FAINT
                                   * (tt_mag - TRICK_KNEE_MAG))
    elif sensor == "strap-legacy":
        flux_fac = 10.0 ** (-0.2 * (DEF_TT_MAG - tt_mag))  # sigma ~ 1/sqrt(flux)
        meas = 7.25 * flux_fac * s_tot     # the frozen 2004 sheet's row
    else:
        _ref, _beta = ((STRAP_MEAS_REF_ONSKY, STRAP_BETA_ONSKY)
                       if strap_law == "onsky2026"
                       else (STRAP_MEAS_REF, STRAP_BETA))
        # bright of the faint knee: the calibrated law as before; beyond it
        # the sheet-derived steepening (see STRAP FAINT-END block above)
        _m1 = min(tt_mag, STRAP_FAINT_KNEE)
        meas = _ref * 10.0 ** (_beta * (_m1 - STRAP_MAG0))
        if tt_mag > STRAP_FAINT_KNEE:
            _m2 = min(tt_mag, STRAP_FAINT_KNEE2)
            meas *= 10.0 ** (STRAP_BETA_FAINT1 * (_m2 - STRAP_FAINT_KNEE))
            if tt_mag > STRAP_FAINT_KNEE2:
                meas *= 10.0 ** (STRAP_BETA_FAINT2
                                 * (tt_mag - STRAP_FAINT_KNEE2))
        meas *= s_tot
    bw     = 0.47 * s_tot                 # tilt bandwidth
    aniso  = 9.17 * off_fac * s_tot * aniso_scale   # tilt anisoplanatism
    cent   = 1.28 * off_fac * aniso_scale # residual centroid anisoplanatism
    disp, ncp, shake, margin = 0.53, 0.03, 2.60, 5.00
    mas = np.sqrt(meas**2 + bw**2 + aniso**2 + cent**2
                  + disp**2 + ncp**2 + shake**2 + margin**2)
    # ceiling: a loop cannot do worse than no correction -- at extreme
    # offsets the star is doing TT on a seeing-limited spot, whose image
    # motion is the full uncorrected atmospheric tilt (~110 mas one-axis at
    # the reference seeing, scaling with s_tot). See OPEN_LOOP_TILT block.
    mas = min(mas, _psf.OPEN_LOOP_TILT_ONEAXIS_MAS * s_tot)
    return mas * NM_PER_MAS


#  NGS TILT-SERVO RESIDUAL (FWHM path only).
#  tt_wfe_nm() cannot make an ON-AXIS guide star's tilt grow with seeing: its
#  only seeing-scaling rows for a bright on-axis star are meas (0.26 mas x
#  s_tot) and bw (0.47 mas x s_tot), while a FIXED 5.66 mas quadrature floor
#  (wind shake 2.60 + TT margin 5.00) dominates. The 9.17 mas tilt-anisoplanatism
#  row -- which is what makes the off-axis LGS TT star respond to conditions --
#  vanishes on axis. So the modelled NGS FWHM sits at a flat ~48.6 mas, whereas
#  on-sky it degrades slowly with seeing (Eduardo: ~52 mas at K-band seeing
#  0.6"). The missing physics is servo-lag residual on ATMOSPHERIC tilt, whose
#  amplitude scales as eps^(5/6) -- exactly the engine's s_tot. This row supplies
#  it, added in quadrature to tt_wfe_nm's one-axis jitter:
#      sigma_ngs^2 = (tt_wfe_nm(...)/NM_PER_MAS)^2 + (NGS_TILT_SERVO_MAS*s_tot)^2
#  CALIBRATION: the coefficient is fitted to that SINGLE anchor point (it
#  reproduces 52.0 mas at eps_K=0.6" and gives 49.5 -> 55.6 mas across
#  eps_K = 0.15" -> 0.90"). Tune with --ngs-tilt-servo as better data arrives.
#  IMPORTANT: this is used ONLY by the FWHM path. It is deliberately NOT added
#  to tt_wfe_nm(), which feeds lgs_budget_terms() and hence the reported Strehl;
#  putting it there would change every Strehl output and the frozen references.
#  Self-consistency is preserved because psf_fwhm_mas divides the tilt back out
#  (S_ho = S_total / Marechal(tt)) before smearing the core.
NGS_TILT_SERVO_MAS = 6.189   # one-axis mas at the reference profile (s_tot=1)


def ngs_tt_nm(s_tot, mag, offset, tilt_servo_mas=NGS_TILT_SERVO_MAS):
    """Residual tip-tilt for the NGS case, as an equivalent nm RMS (so it can
    be handed straight to psf_fwhm_mas).

    The NGS guide star IS the tilt reference (sensed on the NGS WFS itself,
    not STRAP -- so the 2026-07 STRAP-quadcell recalibration deliberately does
    NOT apply here): start from the original parameterized TT model at the NGS
    magnitude/offset, then add the atmospheric tilt-servo residual that
    tt_wfe_nm lacks on axis (see NGS_TILT_SERVO_MAS). FWHM path only -- never
    feeds the Strehl budget."""
    base_mas = tt_wfe_nm(s_tot, mag, offset,
                         sensor="strap-legacy") / NM_PER_MAS   # one-axis mas
    servo_mas = tilt_servo_mas * s_tot
    # same physical ceiling as tt_wfe_nm: the servo residual is atmospheric
    # tilt too, so the total still cannot exceed the uncorrected tilt
    tot_mas = min(float(np.hypot(base_mas, servo_mas)),
                  _psf.OPEN_LOOP_TILT_ONEAXIS_MAS * s_tot)
    return tot_mas * NM_PER_MAS
