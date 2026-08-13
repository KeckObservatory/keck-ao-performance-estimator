"""Statistics of a MEASURED Strehl field (the Measured SR tab's field
map): peak location, error-weighted mean, best-fit performance gradient,
and -- when the field supports it -- an isoplanatic angle theta0 fitted
from the anisoplanatic falloff itself.

Physics.  Anisoplanatism adds a wavefront variance sigma^2 =
(theta/theta0)^(5/3) (Fried 1982) at angular distance theta from the AO
reference, so by the Marechal relation the measured field obeys

    S(theta) = S0 * exp(-(theta/theta0)^(5/3)).

ln S is therefore LINEAR in u = theta^(5/3) with slope -theta0^(-5/3):
a weighted least-squares line through the measured (u_i, ln S_i) gives
theta0 = (-slope)^(-3/5) directly from real data, no model atmosphere
involved.  The AO reference position is not in the frame headers, so the
best-measured star (peak SR) stands in for it -- the correction peaks at
the reference.  Caveats, stated where the number is shown: with an LGS
the low-order (tip-tilt) anisoplanatism relative to the TT star also
shapes the field, so the fitted value is an EFFECTIVE theta0 for the
delivered correction, not a pure atmospheric theta0; and theta0 scales
as lambda^(6/5), so the 500 nm value reported for comparison with the
seeing monitors uses that standard scaling.

The gradient is the error-weighted least-squares plane S(x, y) =
a + gx*x + gy*y over the field (arcsec offsets, detector orientation):
|g| is the performance slope in SR/arcmin and its direction points
DOWNHILL (toward degrading performance -- in practice, away from the AO
reference).  A smooth plane is the right first-order model precisely
because the anisoplanatic field varies smoothly (the same physics behind
field_consistent's outlier rule).
"""
from dataclasses import dataclass

import numpy as np

# theta0 ~ lambda^(6/5): pure Kolmogorov wavelength scaling
THETA0_WAVELENGTH_EXP = 6.0 / 5.0
# below this many stars a plane fit / falloff fit is not meaningful
GRADIENT_MIN_STARS = 3
THETA0_MIN_STARS = 4


@dataclass
class FieldStats:
    n: int
    # peak-performance star (arcsec offsets from frame centre, detector)
    peak_sr: float
    peak_dx_arcsec: float
    peak_dy_arcsec: float
    # error-weighted field mean +/- weighted scatter
    mean_sr: float
    scatter_sr: float
    # least-squares plane: magnitude (SR per arcmin), downhill direction
    # (deg CCW from +x, detector orientation); None below GRADIENT_MIN_STARS
    grad_sr_per_arcmin: float | None = None
    grad_pa_deg: float | None = None
    # effective isoplanatic angle from the measured falloff; None when the
    # field is too small, too flat, or not decreasing outward
    theta0_arcsec: float | None = None
    theta0_err_arcsec: float | None = None
    theta0_500nm_arcsec: float | None = None
    s0_fit: float | None = None
    theta0_note: str = ""


def _weights(strehls, sr_errs):
    """Inverse-variance weights; equal weights when errors are missing
    or degenerate (all zero -- e.g. noiseless synthetic fields)."""
    if sr_errs is None:
        return np.ones_like(strehls)
    e = np.asarray(sr_errs, dtype=float)
    if not np.all(e > 0):
        return np.ones_like(strehls)
    return 1.0 / e ** 2


def theta0_from_ratios(thetas_arcsec, ratios, ratio_errs=None):
    """theta0 from Strehl RATIOS S(theta)/S(0): a through-origin
    weighted fit of -ln r against theta^(5/3).

    This is the drift-immune form of the anisoplanatic fit: when S(0)
    is measured close in time to each off-axis S(theta) (e.g. repeated
    on-axis anchor frames), the ratio cancels whatever the atmosphere
    was doing overall and leaves only the angular falloff, whose model
    ln r = -(theta/theta0)^(5/3) passes through the origin exactly --
    no S0 free parameter.  Returns (theta0_arcsec, err_arcsec, note);
    theta0 is None with the reason in note when the fit is impossible
    (no off-axis points, or ratios not decreasing).
    """
    th = np.asarray(thetas_arcsec, dtype=float)
    r = np.asarray(ratios, dtype=float)
    off = (th > 0) & (r > 0)
    if off.sum() < 1:
        return None, None, "no off-axis ratios"
    u = th[off] ** (5.0 / 3.0)
    y = -np.log(r[off])
    if ratio_errs is not None and np.all(np.asarray(ratio_errs) > 0):
        w = (r[off] / np.asarray(ratio_errs, dtype=float)[off]) ** 2
    else:
        w = np.ones(off.sum())
    denom = float(np.sum(w * u ** 2))
    k = float(np.sum(w * u * y) / denom)
    if k <= 0:
        return None, None, "ratios do not decrease with angle"
    theta0 = k ** (-3.0 / 5.0)
    resid = y - k * u
    dof = max(int(off.sum()) - 1, 1)
    k_err = float(np.sqrt(np.sum(w * resid ** 2) / dof / denom))
    return (float(theta0),
            float(theta0 * (3.0 / 5.0) * (k_err / k)), "")


def field_statistics(xs_arcsec, ys_arcsec, strehls, sr_errs=None,
                     wavelength_um=None):
    """Peak / weighted mean / gradient / theta0 of a measured SR field.

    xs, ys: star offsets from the frame centre in arcsec (detector
    orientation, the field map's own axes).  strehls: measured SR per
    star.  sr_errs: propagated 1-sigma SR uncertainties (equal weights
    when absent).  wavelength_um: the frame's effective wavelength, used
    only to also express theta0 at the 500 nm seeing-monitor standard.
    """
    x = np.asarray(xs_arcsec, dtype=float)
    y = np.asarray(ys_arcsec, dtype=float)
    s = np.asarray(strehls, dtype=float)
    n = len(s)
    if n == 0:
        raise ValueError("empty field")
    w = _weights(s, sr_errs)

    ipk = int(np.argmax(s))
    mean = float(np.sum(w * s) / np.sum(w))
    scatter = float(np.sqrt(np.sum(w * (s - mean) ** 2) / np.sum(w)))
    stats = FieldStats(n=n, peak_sr=float(s[ipk]),
                       peak_dx_arcsec=float(x[ipk]),
                       peak_dy_arcsec=float(y[ipk]),
                       mean_sr=mean, scatter_sr=scatter)

    # ---- gradient plane: weighted LSQ  S = a + gx*x + gy*y ------------
    if n >= GRADIENT_MIN_STARS:
        A = np.column_stack([np.ones(n), x, y])
        Aw = A * np.sqrt(w)[:, None]
        sol, *_ = np.linalg.lstsq(Aw, s * np.sqrt(w), rcond=None)
        # collinear positions leave the plane degenerate -> rank check
        if np.linalg.matrix_rank(Aw) == 3:
            gx, gy = float(sol[1]), float(sol[2])
            g = float(np.hypot(gx, gy))
            stats.grad_sr_per_arcmin = g * 60.0
            # downhill: the direction performance DEGRADES toward
            stats.grad_pa_deg = float(np.degrees(np.arctan2(-gy, -gx)))

    # ---- theta0 from the anisoplanatic falloff ------------------------
    if n < THETA0_MIN_STARS:
        stats.theta0_note = (f"needs >= {THETA0_MIN_STARS} stars")
        return stats
    good = s > 0
    theta = np.hypot(x - x[ipk], y - y[ipk])
    off = good & (theta > 0)
    if off.sum() < THETA0_MIN_STARS - 1:
        stats.theta0_note = "too few stars away from the peak"
        return stats
    u = theta[off] ** (5.0 / 3.0)
    lns = np.log(s[off])
    # sigma_lnS = sigma_S / S; equal weights when errors are absent
    if sr_errs is not None and np.all(np.asarray(sr_errs) > 0):
        wl = (s[off] / np.asarray(sr_errs, dtype=float)[off]) ** 2
    else:
        wl = np.ones(off.sum())
    W = np.sum(wl)
    ub, lb = np.sum(wl * u) / W, np.sum(wl * lns) / W
    du = u - ub
    denom = np.sum(wl * du ** 2)
    if denom <= 0:
        stats.theta0_note = "degenerate geometry (equal radii)"
        return stats
    slope = float(np.sum(wl * du * (lns - lb)) / denom)
    if slope >= 0:
        stats.theta0_note = ("no falloff with radius -- field flat or "
                             "noise-dominated")
        return stats
    k = -slope                              # = theta0^(-5/3)
    theta0 = k ** (-3.0 / 5.0)
    # slope uncertainty from the weighted residual variance
    resid = lns - (lb + slope * du)
    dof = max(int(off.sum()) - 2, 1)
    var_slope = float(np.sum(wl * resid ** 2) / dof / denom)
    k_err = np.sqrt(var_slope)
    theta0_err = theta0 * (3.0 / 5.0) * (k_err / k) if k > 0 else None
    stats.theta0_arcsec = float(theta0)
    stats.theta0_err_arcsec = (float(theta0_err)
                               if theta0_err is not None else None)
    stats.s0_fit = float(np.exp(lb + slope * (0.0 - ub)))
    if wavelength_um and wavelength_um > 0:
        stats.theta0_500nm_arcsec = float(
            theta0 * (0.5 / wavelength_um) ** THETA0_WAVELENGTH_EXP)
    return stats
