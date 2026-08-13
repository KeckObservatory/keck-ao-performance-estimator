"""The NGS (natural guide star) Strehl model: a per-telescope empirical
Gompertz fit (K-band, on-sky calibrated) with a Kolmogorov or Gaussian seeing
law and an implied-WFE extrapolation to other science wavelengths."""
import numpy as np

from .constants import V2K, LAMBDA_K_NM
from .marechal import marechal_strehl

# --- NGS Gompertz model coefficients (K-band) --------------------------------
#  The NGS K-band Strehl is modeled as
#      S = S0 * exp(-A * sK^2) * exp(-exp((m - m0)/w))
#  i.e. a Gaussian seeing-degradation term times a Gompertz faint-end roll-off.
#
#  Two distinct fits are kept, one per telescope, because K1 and K2 use
#  different AO systems with different histories:
#
#   * K2 (HAKA-class): the on-sky HAKA fit. Bright-star ceiling 0.755, seeing
#     exponent 0.738, Gompertz m0 = 13.76, w = 1.71. (Report Figure 5, maroon.)
#     REFIT 2026-08-07 (Eduardo) after new data was added to the HAKA set;
#     the previous fit was 0.751 / 0.702 / 13.43 / 1.53. The refit is a mild
#     move in every term: a hair more ceiling, slightly more seeing
#     sensitivity, and a faint end that both starts ~0.33 mag later and rolls
#     off more gently (w 1.53 -> 1.71), so the biggest changes land in the
#     m ~ 13-16 range rather than at the bright end.
#
#   * K1 (pre-HAKA RTC + OCAM2K class): the historical RTC+OCAM reference curve
#     (Report Figure 5, purple). Reconstructed from that curve's constraints
#     (crossover with HAKA at R = 11.4 at 0.26", anchored to 6% Strehl at m=17,
#     shared width w = 1.53): zero-seeing bright-star ceiling 0.61, m0 = 15.7.
#     LEFT AT THE OLD WIDTH on purpose (Eduardo, 2026-08-07): its w = 1.53 was
#     "shared" with the PRE-refit HAKA fit, and K1's own curve has not been
#     re-derived against the new one -- so K1's width no longer tracks K2's,
#     and the R = 11.4 crossover it was reconstructed from is now approximate.
#     Re-deriving K1 needs its two anchors re-confirmed, not just w copied over.
#     TWO K1-specific physical adjustments are then applied:
#       (a) a STEEPER seeing exponent (A = 1.0 vs 0.702) because the lower-stroke
#           K1 deformable mirror saturates in poor seeing -> more seeing-sensitive
#           (see DM-stroke note in lgs_strehl / the plot caption);
#       (b) a flat -0.05 Strehl penalty for the "KAPA PRO induced Quadcell
#           saturation effect" (NGS_K1_QUADCELL_PENALTY) -- an empirical fudge
#           factor that brings the modeled K1 NGS into line with the ~5-point
#           historical under-performance vs K2.
NGS_PARAMS = {
    "K2": dict(S0=0.755, A=0.738, m0=13.76, w=1.71),   # HAKA refit 2026-08-07
    "K1": dict(S0=0.61,  A=1.00,  m0=15.73, w=1.53),
}
NGS_K1_QUADCELL_PENALTY = 0.05    # flat Strehl subtracted on K1 (quadcell sat.)

#  NGS SEEING LAW (revised 2026-07 after the HIP 88553 high-airmass validation).
#  The Gompertz fits were calibrated on open-loop K-band FWHM of ~0.19-0.38"
#  (all at ZA 0-30). Their Gaussian seeing term exp(-A sK^2) is empirically
#  fine inside that range, but a Gaussian FREE-FALLS when extrapolated: at
#  high airmass the projected line-of-sight seeing lands 2-3x beyond the
#  calibration maximum, and the quadratic exponent turns airmass into an
#  X^(6/5) hit on -ln(S) where the physical closed-loop residual variance
#  grows only linearly with airmass (turbulence path ~ integral Cn2 ~ X).
#  Validated on 2026-07-06 HIP 88553 (airmass 2.34): delivered S ~ 0.50-0.56
#  vs 0.29-0.42 predicted by the quadratic form.
#
#  Fix: Kolmogorov scaling. Residual variance ~ seeing^(5/3), so the seeing
#  term becomes exp(-A' sK^(5/3)) with A' = A * anchor^(1/3) re-anchored at
#  the calibration mid-range (sK = 0.30"). Inside the calibrated range the
#  two laws agree to <0.005 Strehl (so nothing that was validated changes);
#  beyond it the 5/3 law rolls off at the physical rate instead of the
#  Gaussian's. --ngs-seeing-law gaussian reverts to the original form.
NGS_SEEING_LAW = "kolmogorov"     # "kolmogorov" (default) or "gaussian"
NGS_SK_ANCHOR  = 0.30             # K-band anchor: mid calibration range 0.19-0.38"


def ngs_strehl(eps_total_500nm, mag, telescope="K2", lam_nm=LAMBDA_K_NM,
               seeing_law=None, ngs_s0=None, ngs_a=None, ngs_m0=None,
               ngs_w=None, k1_quadcell=None):
    """NGS Strehl for total seeing + guide-star magnitude, at wavelength lam_nm.

    seeing_law: "kolmogorov" (default, module constant NGS_SEEING_LAW) uses the
    physical variance ~ seeing^(5/3) scaling re-anchored inside the fit's
    calibration range; "gaussian" reproduces the original exp(-A sK^2) fit
    form, which free-falls when the (possibly airmass-projected) seeing is
    extrapolated beyond the ~0.19-0.38" K-band calibration range.

    Uses a separate Gompertz fit per telescope (see NGS_PARAMS):
      * K2  -> on-sky HAKA fit (ceiling 0.755, seeing exponent 0.738).
      * K1  -> historical RTC+OCAM reference (ceiling 0.61), with a steeper
               seeing exponent (1.0) for DM-stroke saturation, and a flat
               -0.05 Strehl penalty for the KAPA PRO quadcell saturation effect.

    The Gompertz fits are EMPIRICAL K-band fits. To report a different science
    wavelength we back the K-band Strehl out to an implied RMS wavefront error
    (sigma = (lam_K / 2pi) * sqrt(-ln S_K)) and re-evaluate the Marechal Strehl
    at lam_nm. This is a model extrapolation of a K-band fit (flagged as such in
    the plot), reasonable for near-IR bands but increasingly rough far from K.
    """
    # Copy so per-call Gompertz overrides never mutate the module-level fit.
    # Any of the four terms (ceiling S0, seeing exponent A, faint-end midpoint
    # m0, roll-off width w) may be overridden for the ACTIVE telescope; None
    # keeps that telescope's fitted value. The K1 quadcell penalty is a
    # separate post-fit K1-only term.
    par = dict(NGS_PARAMS[telescope])
    if ngs_s0 is not None: par["S0"] = float(ngs_s0)
    if ngs_a  is not None: par["A"]  = float(ngs_a)
    if ngs_m0 is not None: par["m0"] = float(ngs_m0)
    if ngs_w  is not None: par["w"]  = float(ngs_w)
    if k1_quadcell is None:
        k1_quadcell = NGS_K1_QUADCELL_PENALTY
    if seeing_law is None:
        seeing_law = NGS_SEEING_LAW
    sK = eps_total_500nm * V2K            # convert 500 nm seeing to K-band
    if seeing_law == "gaussian":
        atm = np.exp(-par["A"] * sK ** 2)              # original fit form
    else:
        # Kolmogorov: residual variance ~ seeing^(5/3), re-anchored so the
        # two laws coincide at the calibration mid-range (see NGS_SEEING_LAW)
        a_eff = par["A"] * NGS_SK_ANCHOR ** (2.0 - 5.0 / 3.0)
        atm = np.exp(-a_eff * sK ** (5.0 / 3.0))
    s_K = (par["S0"] * atm
           * np.exp(-np.exp((mag - par["m0"]) / par["w"])))
    if telescope == "K1":
        s_K = max(0.0, s_K - k1_quadcell)               # quadcell-saturation fudge

    if abs(lam_nm - LAMBDA_K_NM) < 1.0:
        return s_K                         # already K-band, no conversion
    # back out implied WFE from the K-band Strehl, then re-evaluate at lam_nm
    s_K = min(max(s_K, 1e-6), 0.999999)    # clamp for the log
    sigma_nm = (LAMBDA_K_NM / (2.0 * np.pi)) * np.sqrt(-np.log(s_K))
    return marechal_strehl(sigma_nm, lam_nm)
