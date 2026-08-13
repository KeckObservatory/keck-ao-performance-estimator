"""Turbulence-profile physics: the Fried isoplanatic angle theta0 and the
focal-anisoplanatism cone diameter d0 from the 6-bin MASS Cn2 profile, the
Cn2 density conversion for display, the zenith-angle seeing projection, and
the Kolmogorov seeing<->integrated-Cn2 inversion used by the prediction tab's
synthetic profiles."""
import numpy as np

from .constants import MASS_HEIGHTS_M

_H_NA          = 90e3                                    # sodium beacon altitude
#  d0 calibration constant, fixed so the report's A3 reference profile
#  (0.30" free-atm seeing at ~7 km) reproduces d0 = 1.36 m.
_D0_CONST      = 1.3216e-4


def theta0_d0_from_profile(cn2_bins, zenith_angle_deg=0.0, lam_nm=500.0):
    """Compute (theta0 [arcsec], d0 [m]) from the 6 MASS Cn2 bins.

    theta0 is returned at the requested wavelength lam_nm (it scales as
    lambda^(6/5) from the 500 nm value). d0 is wavelength-INDEPENDENT (a physical
    aperture diameter set by the cone geometry) and is returned in metres.

    theta0(500) = [2.914 k^2 (sec z)^(8/3) * sum_i J_i h_i^(5/3)]^(-3/5)
    theta0(lam) = theta0(500) * (lam/500)^(6/5)
    d0          = C_d * [ (sec z) * sum_i J_i h_i^(5/3) (1 - h_i/H)^(5/3) ]^(-3/5)
    """
    J = np.asarray(cn2_bins, dtype=float)
    if J.size != MASS_HEIGHTS_M.size or not np.any(J > 0):
        return (np.nan, np.nan)
    zsec = 1.0 / np.cos(np.radians(min(abs(zenith_angle_deg), 85.0)))
    k500 = 2 * np.pi / 500e-9

    # theta0 at 500 nm, then scale to lam_nm
    M_theta = np.sum(J * MASS_HEIGHTS_M ** (5.0 / 3.0))
    theta0_500 = (2.914 * k500 ** 2 * zsec ** (8.0 / 3.0) * M_theta) ** (-3.0 / 5.0)
    theta0_arcsec = theta0_500 * 206265.0 * (lam_nm / 500.0) ** (6.0 / 5.0)

    # d0 (cone-weighted), wavelength-independent
    frac = np.clip(MASS_HEIGHTS_M / _H_NA, 0, 1)
    M_d0 = np.sum(J * MASS_HEIGHTS_M ** (5.0 / 3.0) * (1 - frac) ** (5.0 / 3.0)) * zsec
    d0 = _D0_CONST * M_d0 ** (-3.0 / 5.0)

    return (theta0_arcsec, d0)


def cn2_density_profile(cn2_bins):
    """Convert the 6 layer-INTEGRATED MASS bins J_i [m^1/3] to a Cn^2 DENSITY
    [m^-2/3] at each bin altitude, for display only.

    The MASS bins are log-spaced (heights double: 0.5,1,2,4,8,16 km), so each
    bin's effective thickness is Δh_i = h_i·ln2 (the spacing d(ln h)=ln2
    between neighbours); Cn2_i = J_i / Δh_i. Returns (heights_km, cn2_density)
    as arrays. No budget term uses the density -- the model works from the
    integrated J_i directly; this is purely for the profile plot."""
    J = np.asarray(cn2_bins, dtype=float)
    dh = MASS_HEIGHTS_M * np.log(2.0)
    return MASS_HEIGHTS_M / 1e3, J / dh


def zenith_seeing_factor(zenith_angle_deg):
    """Multiplicative factor that projects zenith-corrected seeing onto a line
    of sight at the given zenith angle.

    MASS/DIMM report seeing already corrected to zenith. Along a line of sight
    at zenith angle zeta the airmass is X = sec(zeta), the Fried parameter
    shrinks as r0 ∝ cos(zeta)^(3/5), and since seeing ∝ 1/r0 the seeing grows as

        eps(zeta) = eps_zenith * X^(3/5)  =  eps_zenith * sec(zeta)^(3/5).

    The same factor is applied to BOTH total (DIMM) and free-atmosphere (MASS)
    seeing, so it propagates correctly into every seeing-dependent budget term:
    the total-seeing terms grow, and the free-atmosphere terms (focal
    anisoplanatism, angular anisoplanatism -- i.e. the theta0 / cone-effect
    penalty) grow with them, which is the physical signature of theta0 shrinking
    toward the horizon. Clamped to zeta < 85 deg to avoid the sec divergence at
    the horizon, where this simple plane-parallel scaling breaks down anyway.
    """
    z = np.radians(min(abs(zenith_angle_deg), 85.0))
    airmass = 1.0 / np.cos(z)
    return airmass ** (3.0 / 5.0)


def seeing_to_integrated_cn2(eps_arcsec_500):
    """Invert the Kolmogorov seeing relation to the integrated turbulence
    sum(J) [m^1/3] at 500 nm, zenith:  eps = 0.98 lam/r0  with
    r0 = [0.423 k^2 sum(J)]^(-3/5). Validated against MKWC masspro files:
    applied to a file's own six J bins this reproduces its free-atm seeing
    column to ~1% (median over the 20260525 night, 244 profiles)."""
    lam = 500e-9
    k2 = (2.0 * np.pi / lam) ** 2
    r0 = 0.98 * lam / np.radians(eps_arcsec_500 / 3600.0)
    return r0 ** (-5.0 / 3.0) / (0.423 * k2)
