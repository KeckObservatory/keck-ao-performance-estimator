"""The field map: performance (Strehl or FWHM) vs field position.

A single-snapshot 2-D map of how the mode's anisoplanatism degrades AO
performance across the science field. At each field point P the budget is
re-evaluated with the offset(s) measured FROM P to the correction reference:
  * NGS   -> S(P) = S_onaxis * exp(-(|P - NGS_star|/theta0)^(5/3))
  * single-> lgs_strehl(..., lgs_offset=|P - laser|, tt_offset=|P - TT_star|)
  * LTAO  -> as single but with tomography + the layer-mismatch penalty
so at the science target (P=0) it reproduces exactly the science-direction
value. Convention: North = +Y, East = -X. Field: 20x20" (K1) / 10x10" (K2).
"""
import numpy as np

from .constants import (
    REF_TOTAL, LAMBDA_K_NM, FIELD_FOV_ARCSEC, MASS_HEIGHTS_M,
    DM_ACTUATORS_ACROSS,
)
from .marechal import marechal_strehl
from .atmosphere import (
    seeing_to_integrated_cn2, theta0_d0_from_profile, zenith_seeing_factor,
)
from .budget import FITTING_ERR, lgs_strehl, layer_mismatch
from .ngs import ngs_strehl
from .tiptilt import tt_wfe_nm, ngs_tt_nm, DEF_LTAO_TT_THETA0_GAIN
from .psf import (psf_fwhm_mas, fwhm_gaussfit_mas, fwhm_gaussfit_sky_mas,
                  fwhm_srtool_mas)

# PHYSICAL Mauna Kea free-atmosphere Cn2 SHAPE at the 6 MASS altitudes
# (0.5/1/2/4/8/16 km ABOVE THE SUMMIT). This is the measured median
# free-atmosphere distribution, distinct from budget.RECON_PRIOR_ALOFT (which
# is the tomographic reconstructor's ASSUMED prior -- a fixed RTC design
# choice, not the atmosphere, and left untouched so the frozen budget /
# layer-mismatch outputs are unchanged). The measured MK free atmosphere
# peaks at the TROPOPAUSE/jet, captured by the 8 km MASS bin (its triangular
# response integrates ~6-12 km above the summit, i.e. ~10-16 km ASL); the
# 4 km layer is comparatively WEAK. Aloft values 2/4/8/16 km are the RAVEN
# Maunakea median (Ono et al. 2016, mean Cn2dh 1.12/1.52/3.14/1.45e-14),
# which agrees with the MASS/SCIDAR campaign of Tokovinin et al. 2005 (8 &
# 16 km layers dominate the free atmosphere); the 0.5/1 km values are the
# boundary-layer top MASS reports just above the ground layer. Used only by
# synthetic_field_snapshot() (the GUI prediction tab), never by the CLI.
MK_FA_PROFILE_FRAC = np.array([1.6, 0.9, 1.12, 1.52, 3.14, 1.45])
MK_FA_PROFILE_FRAC = MK_FA_PROFILE_FRAC / MK_FA_PROFILE_FRAC.sum()


def field_snapshot(args, prep, res, when="window", time_hst=None):
    """Pick a single representative atmospheric sample for the field map and
    return its state. 'window' -> the observing window (whole night if none);
    'night' -> the whole night; 'time' -> the sample nearest time_hst (a
    datetime). The representative sample is the one whose LOS total seeing is
    closest to the median over the selection (self-consistent eps/theta0/Cn2).
    Returns a dict, or None if the needed data is absent."""
    p_times = res.p_times
    if len(p_times) == 0:
        return None                       # NGS-only night: handled by the GUI
    eps_tot_los = res.p_dimm_in * res.p_zf
    eps_fa_los  = res.col_mass * res.p_zf
    if when == "time" and time_hst is not None:
        idx = int(np.argmin([abs((t - time_hst).total_seconds()) for t in p_times]))
        desc = f"{p_times[idx]:%H:%M} HST"
    else:
        if (when == "window" and getattr(prep, "show_target", False)
                and prep.windows):
            sel = np.array([prep.in_any_window(t) for t in p_times])
            desc = "observing-window median"
        else:
            sel = np.ones(len(p_times), dtype=bool)
            desc = "whole-night median"
        if not sel.any():
            sel = np.ones(len(p_times), dtype=bool)
            desc = "whole-night median"
        med = np.median(eps_tot_los[sel])
        cand = np.nonzero(sel)[0]
        idx = int(cand[np.argmin(np.abs(eps_tot_los[cand] - med))])
    return dict(
        eps_tot_los=float(eps_tot_los[idx]), eps_fa_los=float(eps_fa_los[idx]),
        theta0_los=float(res.col_theta0[idx]),
        cn2_bins=(res.col_cn2[idx] if len(res.col_cn2) else None),
        airmass=float(res.p_airmass[idx]) if len(res.p_airmass) else np.nan,
        t_hst=p_times[idx], when_desc=desc)


def field_cn2_profile(args, prep, res, when="window", time_hst=None):
    """The night's REAL (as-measured, zenith) Cn2 profile for the same
    conditions the field map uses: 'window' -> MEAN over the observing window
    (whole night if none), 'night' -> MEAN over the whole night, 'time' ->
    the exact profile nearest time_hst. Returns a dict with the 6-bin free-atm
    profile plus the mean total/free-atm seeing (for the ground layer), or None
    if there are no MASS profiles. Distinct from field_snapshot(), which picks
    ONE representative sample; here window/night are genuine means."""
    p_times = res.p_times
    if len(p_times) == 0 or not len(res.col_cn2):
        return None
    J = np.asarray(res.col_cn2, dtype=float)          # (n, 6) zenith MASS bins
    eps_tot = np.asarray(res.p_dimm_in, dtype=float)  # total seeing, zenith
    eps_fa = np.asarray(res.col_mass, dtype=float)    # free-atm seeing, zenith
    if when == "time" and time_hst is not None:
        idx = int(np.argmin([abs((t - time_hst).total_seconds())
                             for t in p_times]))
        sel = np.zeros(len(p_times), dtype=bool); sel[idx] = True
        desc = f"{p_times[idx]:%H:%M} HST"
    else:
        if (when == "window" and getattr(prep, "show_target", False)
                and prep.windows):
            sel = np.array([prep.in_any_window(t) for t in p_times])
            desc = "observing-window mean"
        else:
            sel = np.ones(len(p_times), dtype=bool)
            desc = "whole-night mean"
        if not sel.any():
            sel = np.ones(len(p_times), dtype=bool)
            desc = "whole-night mean"
    return dict(
        cn2_mean=J[sel].mean(axis=0),
        eps_tot_zenith=float(eps_tot[sel].mean()),
        eps_fa_zenith=float(eps_fa[sel].mean()),
        n=int(sel.sum()), when_desc=desc)


def synthetic_field_snapshot(eps_tot_zenith, eps_fa_zenith,
                             zenith_angle_deg=0.0, lam_nm=LAMBDA_K_NM,
                             theta0_k_zenith=None):
    """Build a field_snapshot()-shaped dict for a HYPOTHETICAL scenario, so
    field_map_grid() can predict performance for conditions the user dials in
    rather than a night's data. Inputs are ZENITH values (the convention MKWC
    reports); the line-of-sight projection is applied here.

    eps_tot_zenith / eps_fa_zenith : total (DIMM) and free-atm (MASS) seeing
        at 500 nm, zenith, arcsec. free-atm is clamped to <= total.
    theta0_k_zenith : isoplanatic angle at K-band, zenith, arcsec. None ->
        derived from the synthesized profile itself (reconstructor-prior
        shape), which is also how the preset values are generated.

    The 6-bin Cn2 profile (needed for the LTAO layer-mismatch penalty) is the
    reconstructor's aloft prior tilted in altitude, J_i ~ prior_i * h_i^alpha,
    with sum(J) pinned by the free-atm seeing and alpha solved (bisection) so
    the profile's theta0 matches the requested one -- i.e. a small requested
    theta0 pushes the turbulence aloft, a large one pulls it down, and the
    layer mismatch m follows from that same profile. No new fitted constants.
    """
    eps_fa_zenith = min(float(eps_fa_zenith), float(eps_tot_zenith))
    zf = zenith_seeing_factor(zenith_angle_deg)
    cosz = np.cos(np.radians(min(abs(zenith_angle_deg), 85.0)))

    J_tot = seeing_to_integrated_cn2(eps_fa_zenith)

    def profile(alpha):
        # base = the measured MK free-atmosphere shape (tropopause-dominated),
        # tilted in altitude by alpha to reach a requested theta0
        w = MK_FA_PROFILE_FRAC * (MASS_HEIGHTS_M / MASS_HEIGHTS_M[0]) ** alpha
        return J_tot * w / w.sum()

    def th0_of(alpha):        # profile theta0 at K-band, zenith
        return theta0_d0_from_profile(profile(alpha), 0.0, LAMBDA_K_NM)[0]

    if theta0_k_zenith is None:
        alpha = 0.0
        theta0_k_zenith = th0_of(0.0)
    else:
        theta0_k_zenith = float(theta0_k_zenith)
        # theta0 decreases monotonically with alpha (weight moves aloft);
        # clamp to the reachable range of the tilt, then bisect.
        lo, hi = -4.0, 4.0
        if theta0_k_zenith >= th0_of(lo):
            alpha = lo
        elif theta0_k_zenith <= th0_of(hi):
            alpha = hi
        else:
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                if th0_of(mid) > theta0_k_zenith:
                    lo = mid
                else:
                    hi = mid
            alpha = 0.5 * (lo + hi)
    J = profile(alpha)

    theta0_los = (theta0_k_zenith * (lam_nm / LAMBDA_K_NM) ** (6.0 / 5.0)
                  * cosz ** (8.0 / 5.0))
    # The budget's aniso terms scale with free-atm seeing ASSUMING the median
    # MK profile shape (equivalent to theta0 ~ 1/eps_fa). Requesting a theta0
    # different from that shape's value at this seeing re-weights the altitude
    # moment by g^(5/3), so the aniso WFE terms carry g^(5/6):
    aniso_scale = (th0_of(0.0) / theta0_k_zenith) ** (5.0 / 6.0)
    return dict(
        eps_tot_los=eps_tot_zenith * zf, eps_fa_los=eps_fa_zenith * zf,
        theta0_los=float(theta0_los), cn2_bins=J,
        airmass=float(1.0 / cosz), t_hst=None,
        when_desc="predicted scenario", synthetic=True,
        eps_tot_zenith=float(eps_tot_zenith),
        eps_fa_zenith=float(eps_fa_zenith),
        theta0_k_zenith=float(theta0_k_zenith),
        zenith_angle_deg=float(zenith_angle_deg),
        aniso_scale=float(aniso_scale),
        alpha=float(alpha), m=float(layer_mismatch(J)))


def field_map_grid(args, prep, snap, mode, metric, ngs_xy, tt_xy, laser_xy,
                   n_grid=41, fov=None, ngs_delta_var=0.0):
    """Compute the field grid. Positions are (x, y) arcsec in the plot frame
    (x = West+, y = North+; i.e. x = -East). Returns (extent, Z, meta) where
    extent = [xmin,xmax,ymin,ymax] for imshow and Z[i,j] is the metric at
    (x_j, y_i). metric is 'strehl' or 'fwhm'; mode is 'ngs'/'single'/'ltao'.

    fov : (width, height) arcsec of the field to map (e.g. a rectangular
    OSIRIS-spectrograph FOV). None -> the telescope's square science FOV
    (FIELD_FOV_ARCSEC), preserving the original square behaviour.

    ngs_delta_var : signed wavefront-VARIANCE change (nm^2 RMS) to swap into
    the NGS model (ignored for LGS/LTAO, whose budget terms apply directly).
    The empirical NGS fit is delivered performance -- budget rows are already
    inside it and cannot be turned individually -- so a budget what-if is
    expressed as the same Marechal variance swap the timeline's projected-NGS
    overlay uses: sigma'^2 = sigma^2 + (b^2 - a^2) per changed term. Positive
    = added error (lower Strehl), negative = an upgrade (e.g. a denser DM
    lowering FITTING_ERR). Default 0.0 -> the NGS map is the pure empirical
    model, unchanged."""
    tel = args.telescope
    if fov is None:
        fov_x = fov_y = FIELD_FOV_ARCSEC[tel]
    else:
        fov_x, fov_y = float(fov[0]), float(fov[1])
    half_x, half_y = fov_x / 2.0, fov_y / 2.0
    xs = np.linspace(-half_x, half_x, n_grid)
    ys = np.linspace(-half_y, half_y, n_grid)
    ctx = _field_context(args, prep, snap, mode, metric, ngs_xy, tt_xy,
                         laser_xy, ngs_delta_var)
    Z = np.full((n_grid, n_grid), np.nan)
    for i, py in enumerate(ys):
        for j, px in enumerate(xs):
            Z[i, j] = _field_point(ctx, px, py)
    # value at the science target (grid centre) -- reported on the plot
    c = n_grid // 2
    meta = dict(fov=fov_x, fov_x=fov_x, fov_y=fov_y, s_tot=ctx["s_tot"],
                lam_label=prep.lam_label, target=float(Z[c, c]))
    return [-half_x, half_x, -half_y, half_y], Z, meta


def _field_context(args, prep, snap, mode, metric, ngs_xy, tt_xy, laser_xy,
                   ngs_delta_var, ngs_bright_override=None, tt_mag_override=None):
    """Precompute everything a per-point field evaluation needs that does NOT
    depend on where in the field the point is. Shared by field_map_grid (looped
    over the grid) and field_metric_at (one point), so both use identical
    physics.

    ngs_bright_override / tt_mag_override: substitute a DIFFERENT guide-star
    magnitude than args.ngs_bright/args.tt_mag, for evaluating "what if THIS
    star (at ngs_xy/tt_xy) were the guide star" without touching the GUI's
    own controls -- see gs_ranking.rank_guide_stars, the only caller that
    passes these. None (default) reproduces the exact prior behavior, so
    every existing caller (field_map_grid, field_metric_at's other callers)
    is byte-identical."""
    tel = args.telescope
    lam_nm = prep.lam_nm
    et, ef, th0 = snap["eps_tot_los"], snap["eps_fa_los"], snap["theta0_los"]
    s_tot = (et / REF_TOTAL) ** (5.0 / 6.0)
    fit_nm = FITTING_ERR[tel] * s_tot
    n_act = DM_ACTUATORS_ACROSS[tel]
    fitkw = dict(seeing_law=args.ngs_seeing_law, ngs_s0=args.ngs_s0,
                 ngs_a=args.ngs_a, ngs_m0=args.ngs_m0, ngs_w=args.ngs_w,
                 k1_quadcell=args.k1_quadcell_penalty)
    # aniso re-weighting: 1.0 for night snapshots (reference-shape budget,
    # keeps the on-axis value identical to the timeline's science estimate);
    # synthetic snapshots carry the theta0-decoupling factor
    f_an = snap.get("aniso_scale", 1.0)
    tt_sensor = getattr(args, "_tt_sensor_base", "strap")
    # TRICK's spot degradation follows the STAR-LASER separation (fixed for
    # the whole map), not the per-pixel star-science distance
    star_laser = float(np.hypot(tt_xy[0] - laser_xy[0], tt_xy[1] - laser_xy[1]))
    ltao_tt_gain = getattr(args, "ltao_tt_theta0_gain", None)
    bkw = dict(legacy=args.legacy_budget, bw_factor=prep._ltao_bw_fac,
               v_ground=args.wind_ground, v_free=args.wind_free,
               aniso_scale=f_an, tt_sensor=tt_sensor,
               tt_spot_theta=star_laser,
               ltao_tt_theta0_gain=ltao_tt_gain)
    # the FWHM metrics call tt_wfe_nm directly (below), outside
    # lgs_budget_terms -- fold the same LTAO tilt-aniso reduction into the
    # aniso_scale that call uses, so Strehl and FWHM maps stay consistent
    tt_f_an = f_an
    if mode == "ltao" and not args.legacy_budget:
        _g = (DEF_LTAO_TT_THETA0_GAIN if ltao_tt_gain is None
              else float(ltao_tt_gain))
        tt_f_an = f_an * _g ** (-5.0 / 6.0)
    cn2 = snap["cn2_bins"] if mode == "ltao" else None
    ngs_bright = args.ngs_bright if ngs_bright_override is None else ngs_bright_override
    tt_mag = args.tt_mag if tt_mag_override is None else tt_mag_override
    return dict(mode=mode, metric=metric, tel=tel, lam_nm=lam_nm, et=et, ef=ef,
                th0=th0, s_tot=s_tot, fit_nm=fit_nm, n_act=n_act,
                ngs_bright=ngs_bright, tt_mag=tt_mag,
                ngs_delta_var=ngs_delta_var, f_an=f_an, tt_f_an=tt_f_an,
                tt_sensor=tt_sensor,
                star_laser=star_laser, cn2=cn2, fitkw=fitkw, bkw=bkw,
                ngs_xy=ngs_xy, tt_xy=tt_xy, laser_xy=laser_xy)


def _field_point(ctx, px, py):
    """The field metric at one point (px, py) arcsec given a _field_context --
    byte-for-byte the body of field_map_grid's original inner loop."""
    mode, metric = ctx["mode"], ctx["metric"]
    lam_nm, et = ctx["lam_nm"], ctx["et"]
    if mode == "ngs":
        th = float(np.hypot(px - ctx["ngs_xy"][0], py - ctx["ngs_xy"][1]))
        S = ngs_strehl(et, ctx["ngs_bright"], ctx["tel"], lam_nm, **ctx["fitkw"])
        if ctx["th0"] > 0:
            S *= np.exp(-(th / ctx["th0"]) ** (5.0 / 3.0))
        if ctx["ngs_delta_var"]:
            # budget what-if: Marechal variance swap (see docstring)
            sig2 = (lam_nm / (2.0 * np.pi)) ** 2 * (-np.log(
                min(max(float(S), 1e-6), 1.0)))
            S = marechal_strehl(
                np.sqrt(max(sig2 + ctx["ngs_delta_var"], 0.0)), lam_nm)
        tt_nm = ngs_tt_nm(ctx["s_tot"], ctx["ngs_bright"], th)
    else:
        th_l = float(np.hypot(px - ctx["laser_xy"][0], py - ctx["laser_xy"][1]))
        th_t = float(np.hypot(px - ctx["tt_xy"][0], py - ctx["tt_xy"][1]))
        S = lgs_strehl(et, ctx["ef"], ctx["tel"], mode, lam_nm, cn2_bins=ctx["cn2"],
                       tt_mag=ctx["tt_mag"], tt_offset=th_t,
                       lgs_offset=th_l, **ctx["bkw"])
        tt_nm = tt_wfe_nm(ctx["s_tot"], ctx["tt_mag"], th_t, ctx["tt_f_an"],
                          sensor=ctx["tt_sensor"], spot_theta=ctx["star_laser"])
    if metric == "fwhm":
        return psf_fwhm_mas(S, et, lam_nm, tt_nm=tt_nm,
                            fit_nm=ctx["fit_nm"], n_act=ctx["n_act"])
    if metric == "fwhm_gaussfit":
        return fwhm_gaussfit_mas(S, et, lam_nm, tt_nm=tt_nm,
                                 fit_nm=ctx["fit_nm"], n_act=ctx["n_act"])
    if metric == "fwhm_srtool":
        # the 4th convention: what THIS package's Measured-SR tab would read
        # off this PSF (psf.fwhm_srtool_mas) -- the one directly comparable
        # to a measured number
        return fwhm_srtool_mas(S, et, lam_nm, tt_nm=tt_nm,
                               fit_nm=ctx["fit_nm"], n_act=ctx["n_act"])
    if metric == "fwhm_gaussfit_sky":
        return fwhm_gaussfit_sky_mas(S, et, lam_nm, tt_nm=tt_nm,
                                     fit_nm=ctx["fit_nm"], n_act=ctx["n_act"])
    return S


def field_metric_at(args, prep, snap, mode, metric, ngs_xy, tt_xy, laser_xy,
                    pt_xy, ngs_delta_var=0.0, ngs_bright_override=None,
                    tt_mag_override=None):
    """Metric (strehl / fwhm / fwhm_gaussfit / fwhm_gaussfit_sky / fwhm_srtool)
    at a SINGLE field point
    pt_xy = (x, y) arcsec in the plot frame (x = West+, y = North+), using
    exactly field_map_grid's per-point model -- so a target dropped anywhere on
    the field map reads the same value the map shows at that point.

    ngs_bright_override / tt_mag_override: see _field_context -- None
    (default) is the exact prior behavior."""
    ctx = _field_context(args, prep, snap, mode, metric, ngs_xy, tt_xy,
                         laser_xy, ngs_delta_var,
                         ngs_bright_override=ngs_bright_override,
                         tt_mag_override=tt_mag_override)
    return float(_field_point(ctx, float(pt_xy[0]), float(pt_xy[1])))
