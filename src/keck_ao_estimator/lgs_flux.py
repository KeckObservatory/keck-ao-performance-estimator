"""LGS sodium-return flux vs pointing (azimuth, elevation) at Maunakea, and
the measurement-error scaling it implies.

The budget's high-order measurement term (budget.HOMEAS) is a zenith
value held fixed at every pointing. But the sodium return reaching the
telescope falls with airmass and depends on azimuth through the
geomagnetic field: optical pumping of the sodium D2 line is most efficient
when the beam runs along the field lines (Holzlöhner et al. 2010, A&A 510,
A20). This module gives the return relative to zenith,

    F(az, el) = (1/X) * T^(2(X-1)) * g(theta_B) / g(theta_B at zenith),
    g(theta) = 1 - C sin^2(theta),

with X = 1/sin(el) the airmass. The column the beam crosses grows as X but
the range^2 dilution as X^2, so the photon return per m^2 at the telescope
goes as 1/X. T^(2(X-1)) is the extra up-and-down extinction beyond zenith
(T = one-way zenith transmission at 589 nm). theta_B is the angle between
the beam and the geomagnetic field line. For photon-noise-limited
centroiding the measurement error scales as F^(-1/2).

Constants and provenance
------------------------
* T = 0.84: the Maunakea atmospheric transmission used in Holzlöhner et
  al.'s Bloch-model return map for a TOPTICA laser at Maunakea (the
  "T_a = 0.84" of that figure).
* Field direction: dip (inclination) 36 deg, declination 9.5 deg E.
  Approximate IGRF values for Maunakea (+-2 deg). The field points north
  and down, so the field line runs up toward magnetic south at elevation =
  dip. The return map's maximum sits south of zenith for this reason.
* C = 0.45: the geomagnetic contrast, calibrated on-sky 2026-09-21 UT (K2,
  HAKA engineering). The LGS WFS counts (header AOAOAMED) on TYC 5858-779-1
  (EL 31.4, AZ 129) vs UCAC4 748-00066 (EL 49.5, AZ 9) gave a ratio of 0.75;
  this model gives 0.74 (0.55 with C = 0, i.e. 1/X and extinction only).
  It also matches the 2016 NGL return-vs-pointing data (sodium return in
  equivalent R mag vs zenith distance, by azimuth) to ~0.1 mag at
  ZD 50-57: N branch 0.95 vs ~0.85 mag, SE branch 0.87 vs ~0.85 mag
  fainter than zenith.
  Keck experiments repo: haka_20260921/ (meas_error_scaling_20260921.py).

What this does NOT model: nightly and minute-scale sodium abundance
changes (tens of percent), laser power, the growth of the LGS spot size
along the line of sight (seeing blur ~X^0.6, partly offset by shorter
elongation at longer range), and the aliasing part of HOMEAS, which does
not scale with flux.
"""
import numpy as np

LGS_FLUX_TRANSMISSION = 0.84      # one-way zenith transmission at 589 nm
GEOMAG_DIP_DEG = 36.0             # inclination at Maunakea (approx. IGRF)
GEOMAG_DECL_DEG = 9.5             # declination, deg E of true north
GEOMAG_CONTRAST = 0.45            # C in g(theta) = 1 - C sin^2(theta)


def _unit(az_deg, el_deg):
    az, el = np.radians(az_deg), np.radians(el_deg)
    return np.array([np.sin(az) * np.cos(el), np.cos(az) * np.cos(el), np.sin(el)])


def _field_line():
    """Unit vector along the field line, pointing up toward magnetic south."""
    return _unit(180.0 + GEOMAG_DECL_DEG, GEOMAG_DIP_DEG)


def lgs_return_rel(az_deg, el_deg, contrast=None, transmission=None):
    """Sodium return per m^2 at the telescope relative to zenith (zenith = 1).

    az_deg: azimuth, N through E, in degrees.
    el_deg: elevation in degrees (clipped to >= 5 deg).
    """
    c = GEOMAG_CONTRAST if contrast is None else float(contrast)
    t = LGS_FLUX_TRANSMISSION if transmission is None else float(transmission)
    el = max(float(el_deg), 5.0)
    x = 1.0 / np.sin(np.radians(el))
    b = _field_line()

    def g(v):
        return 1.0 - c * (1.0 - float(np.dot(v, b)) ** 2)

    return (1.0 / x) * t ** (2.0 * (x - 1.0)) * g(_unit(az_deg, el)) / g(np.array([0.0, 0.0, 1.0]))


def lgs_return_rel_azavg(el_deg, contrast=None, transmission=None, n=72):
    """Azimuth-averaged return at elevation el_deg, for scenarios that know
    only a zenith angle (no azimuth)."""
    azs = np.arange(n) * 360.0 / n
    return float(np.mean([lgs_return_rel(a, el_deg, contrast, transmission) for a in azs]))


def meas_scale(az_deg, el_deg, contrast=None, transmission=None):
    """Factor on the zenith measurement term at (az, el): F^(-1/2).
    az_deg None -> the azimuth-averaged return at that elevation."""
    if az_deg is None:
        f = lgs_return_rel_azavg(el_deg, contrast, transmission)
    else:
        f = lgs_return_rel(az_deg, el_deg, contrast, transmission)
    return f ** -0.5
