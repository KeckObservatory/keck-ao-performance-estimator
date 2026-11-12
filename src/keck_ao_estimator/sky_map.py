"""All-sky coverage: at each of 821 sight lines (dec >= -40 deg), the chance
that the best guide star -- or a laser tip-tilt source -- delivers at least
a given Strehl; averaged over the sight lines, the whole-sky curve.

The bundled field sky_coverage.SKY_FIELDS samples four sight lines with a
Monte Carlo over real stars. This samples the whole sky instead, from Gaia
DR3 counts per sight line (data/sky_allsky_gaia_dr3.npz, built by
sr_estimator/build_sky_map.py), with field-star (Poisson) statistics:

    A(m, s)   = pi * min(theta_s(m), patrol)^2
                the area around a pointing in which a star of magnitude m
                delivers Strehl >= s; theta_s from the same (magnitude,
                offset) Strehl surface the curve tab uses
                (sky_coverage.strehl_surface), so the model is the field
                map's on-axis model
    lambda(s) = sum over the sight line's stars of A(m, s) / A_cone
    P(s)      = 1 - exp(-lambda(s))     the chance the best star reaches s

Checked against a direct Monte Carlo on the bundled real fields (Gaia DR2,
best star within 60"): the two agree to within 0.008 in sky fraction
(keck_ao_experiments/plgs_sky_coverage_20261102/validate_poisson.py).

MAGNITUDES. R per star from Gaia G and BP-RP (Riello et al. 2021, Gaia
EDR3 documentation Table 5.9), so the all-sky STRAP / NGS curves carry a
colour term the bundled fields (G as R) do not. H and K from G and BP-RP
through cubics fitted to a matched Gaia DR3 - 2MASS sample in 36 cones,
with each |b| band's median residual and scatter (5-point Gauss-Hermite
spread). 2MASS stars with no Gaia counterpart are added back per |b| band
as stars only TRICK can use: a quarter of the H <= 15.5 stars at |b| < 10
(reddened), 1-7 % elsewhere (mostly unresolved galaxies, so this favours
TRICK slightly).

SENSORS. 'R' (STRAP, or the NGS WFS), 'H' / 'K' (TRICK), 'R+H' / 'R+K'
(per pointing the better of STRAP and TRICK: a star counts when either
sensor reaches s with it), 'laser' (a tip-tilt source of fixed residual,
LASER_TT_MAS by default, everywhere on the sky).

TRUTH STAR. With truth=True (single-LGS / LTAO), every option also needs
an R <= TRUTH_MAG star anywhere in the patrol field for focus / truth
sensing, its offset unpenalized. A STRAP star that reaches s is such a star
already; a TRICK star or the laser need one:
    P(X and truth) = 1 - e^-lX - e^-lT + e^-l(X u T)
with lT the R <= 19 stars at the full patrol area. NGS ignores it.

Qt-free.
"""
import os
from functools import lru_cache

import numpy as np

from .constants import NM_PER_MAS
from .sky_coverage import (_DATA_DIR, COVERAGE_LEVELS, PATROL_RADIUS_ARCSEC,
                           SENSING, STREHL_LEVELS, strehl_surface)

ALLSKY_FIELD = "allsky"
ALLSKY_LABEL = "All sky, dec ≥ −40° (Gaia DR3)"
ALLSKY_FILE = os.path.join(_DATA_DIR, "sky_allsky_gaia_dr3.npz")
B_EDGES = (0.0, 10.0, 20.0, 30.0, 45.0, 60.0, 90.01)   # |b| bands, deg
C_CLIP = (-0.2, 4.5)       # BP-RP range of the G - H / G - K fits
MISS_H_MAX = 15.5          # 2MASS stars missing from Gaia: counted to this H
TRUTH_MAG = 19.0           # R; = SENSING["R"] limit, see _weights
LASER_TT_MAS = 20.0        # one-axis, the polychromatic-laser goal
ALLSKY_SENSORS = ("R", "H", "K", "R+H", "R+K", "laser")
SKY_LEVELS = np.round(np.arange(0.0, 1.0 + 1e-9, 0.0025), 4)
# Gaia G, BP-RP -> Cousins R (Gaia EDR3 doc Table 5.9), valid 0 < c < 4
_G_MINUS_R = (-0.02275, 0.3961, -0.1243, -0.01396, 0.003775)
_GH_NODES = np.array([-2.856970, -1.355626, 0.0, 1.355626, 2.856970])
_GH_WEIGHTS = np.array([0.011257, 0.222076, 0.533333, 0.222076, 0.011257])
_MAP_RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec",
             "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab",
             "#184f95", "#104281", "#0d366b"]


def _band(absb):
    return np.clip(np.searchsorted(B_EDGES, absb, side="right") - 1, 0,
                   len(B_EDGES) - 2)


@lru_cache(maxsize=1)
def load_allsky():
    """The bundled all-sky counts as per-star rows: dict of ra, dec, l, b,
    area (arcsec^2, per sight line) and cone, R, H, K, w (one row per
    colour-magnitude bin and scatter node; w the expected star count)."""
    with np.load(ALLSKY_FILE) as z:
        d = {k: z[k] for k in z.files}
    g = (d["gb"] + 0.5) * float(d["g_step"])
    c = np.where(d["cb"] > -999, (d["cb"] + 0.5) * float(d["c_step"]), np.nan)
    rows = dict(cone=[], g=[], c=[], w=[])
    for k in range(len(d["ra"])):
        s = d["cone"] == k
        gk, ck, nk = g[s], c[s], d["n"][s].astype(float)
        ok = np.isfinite(ck)
        cone, gg, cc, ww = [], [gk[ok]], [ck[ok]], [nk[ok]]
        # a count with no colour takes the colour mix of its own G bin
        # (the sight line's if the bin has none; 0.9 if that has none)
        for gi, ni in zip(gk[~ok], nk[~ok]):
            pool = ok & (gk == gi)
            pool = pool if pool.any() else ok
            if pool.any():
                gg.append(np.full(pool.sum(), gi)); cc.append(ck[pool])
                ww.append(ni * nk[pool] / nk[pool].sum())
            else:
                gg.append([gi]); cc.append([0.9]); ww.append([ni])
        gg, cc, ww = map(np.concatenate, (gg, cc, ww))
        rows["cone"].append(np.full(len(gg), k))
        rows["g"].append(gg); rows["c"].append(cc); rows["w"].append(ww)
    cone, g, c, w = (np.concatenate(rows[x]) for x in ("cone", "g", "c", "w"))
    R = g - np.polynomial.polynomial.polyval(np.clip(c, 0.0, 4.0), _G_MINUS_R)
    bi = _band(np.abs(d["b"]))[cone]
    cl = np.clip(c, *C_CLIP)
    H0 = g - np.polynomial.polynomial.polyval(cl, d["gh_coeffs"]) - \
        d["h_offset"][bi]
    K0 = g - np.polynomial.polynomial.polyval(cl, d["gk_coeffs"]) - \
        d["k_offset"][bi]
    nn = len(_GH_NODES)
    H = (H0[:, None] + d["h_sigma"][bi][:, None] * _GH_NODES).ravel()
    K = (K0[:, None] + d["k_sigma"][bi][:, None] * _GH_NODES).ravel()
    w = (w[:, None] * _GH_WEIGHTS).ravel()
    R, cone, bi = (np.repeat(x, nn) for x in (R, cone, bi))
    # 2MASS stars Gaia lacks: copies of the bright-H rows, TRICK-only
    f = d["miss_frac"][bi]
    m = H <= MISS_H_MAX
    cone = np.r_[cone, cone[m]]
    H, K = np.r_[H, H[m]], np.r_[K, K[m]]
    R = np.r_[R, np.full(m.sum(), np.inf)]
    w = np.r_[w, w[m] * f[m] / (1.0 - f[m])]
    r = np.radians(d["radius"].astype(float))
    area = 2.0 * np.pi * (1.0 - np.cos(r)) * (180.0 / np.pi * 3600.0) ** 2
    return dict(ra=d["ra"].astype(float), dec=d["dec"].astype(float),
                l=d["l"].astype(float), b=d["b"].astype(float), area=area,
                cone=cone, R=R, H=H, K=K, w=w, n_cones=len(d["ra"]))


def _split(m, mags):
    """Linear weights of magnitudes m on the surface grid mags: (i0, i1, f)
    with weight 1-f at i0 and f at i1; brighter than the grid clips to its
    first node; fainter than its last returns i0 = i1 = len(mags) (the
    unusable slot)."""
    step = mags[1] - mags[0]
    u = (np.clip(m, mags[0], None) - mags[0]) / step
    out = m > mags[-1] + 1e-9
    u = np.where(out, 0.0, np.minimum(u, len(mags) - 1))
    i0 = np.floor(u).astype(int)
    f = u - i0
    i1 = np.minimum(i0 + 1, len(mags) - 1)
    n = len(mags)
    return np.where(out, n, i0), np.where(out, n, i1), np.where(out, 0.0, f)


@lru_cache(maxsize=4)
def _weights(band):
    """W[cone, iR, iX]: expected stars per sight line on the R x band
    magnitude grids of the Strehl surfaces (sky_coverage.strehl_surface
    nodes; last index of each = beyond the faint limit). R <= 19 on the R
    grid is also what makes a star a truth star (TRUTH_MAG)."""
    sky = load_allsky()
    mr = _grid("R")
    mx = _grid(band)
    r0, r1, fr = _split(sky["R"], mr)
    x0, x1, fx = _split(sky[band], mx)
    W = np.zeros((sky["n_cones"], len(mr) + 1, len(mx) + 1))
    for ir, wr in ((r0, 1 - fr), (r1, fr)):
        for ix, wx in ((x0, 1 - fx), (x1, fx)):
            np.add.at(W, (sky["cone"], ir, ix), sky["w"] * wr * wx)
    return W


def _grid(band):
    from .sky_coverage import _MAG_BRIGHT, _MAG_STEP
    return np.arange(_MAG_BRIGHT, SENSING[band][1] + 1e-9, _MAG_STEP)


def theta_levels(offs, Z, levels=SKY_LEVELS):
    """th[i, k]: the largest offset at which magnitude row i of the Strehl
    surface Z still delivers levels[k] (linear between offset nodes; the
    surface is made non-increasing in offset first); -1 where not even on
    axis."""
    Z = np.minimum.accumulate(Z, axis=1)
    ge = Z[:, :, None] >= levels[None, None, :]
    k = ge.sum(axis=1)
    th = np.full(k.shape, -1.0)
    th[k == len(offs)] = offs[-1]
    part = (k > 0) & (k < len(offs))
    i, L = np.nonzero(part)
    j = k[part] - 1
    z1, z2 = Z[i, j], Z[i, j + 1]
    frac = np.where(z1 > z2, (z1 - levels[L]) / np.where(z1 > z2, z1 - z2, 1),
                    0.0)
    th[part] = offs[j] + frac * (offs[j + 1] - offs[j])
    return th


def _areas(args, prep, snap, mode, band, tilt_law, patrol, ngs_delta_var):
    """pi min(theta, patrol)^2 per magnitude node and level, with a zero row
    appended for the beyond-the-limit slot."""
    mags, offs, Z = strehl_surface(args, prep, snap, mode, band, tilt_law,
                                   patrol, ngs_delta_var)
    th = theta_levels(offs, Z)
    a = np.where(th >= 0, np.pi * np.minimum(th, patrol) ** 2, 0.0)
    return np.vstack([a, np.zeros((1, a.shape[1]))])


def laser_strehl(args, prep, snap, mode, tilt_law="marechal",
                 laser_mas=LASER_TT_MAS):
    """Strehl at the science target with a laser tip-tilt source of fixed
    one-axis residual laser_mas (budget.lgs_budget_terms tt_fixed_nm)."""
    from .config import resolve_lgs_offset
    from .fieldmap import _field_context, _field_point
    lgs_off = resolve_lgs_offset(args)
    ctx = _field_context(args, prep, snap, mode, "strehl", (0.0, 0.0),
                         (0.0, 0.0), (lgs_off, 0.0), 0.0)
    ctx["bkw"] = dict(ctx["bkw"], tilt_law=tilt_law,
                      tt_fixed_nm=float(laser_mas) * NM_PER_MAS)
    return float(_field_point(ctx, 0.0, 0.0))


def allsky_coverage(args, prep, snap, mode, sensor="R", tilt_law="marechal",
                    patrol=PATROL_RADIUS_ARCSEC, truth=True,
                    laser_mas=LASER_TT_MAS, ngs_delta_var=0.0):
    """Per sight line and Strehl level, the chance of reaching it, and the
    whole-sky curve (module docstring).

    args, prep, snap, mode, tilt_law, patrol, ngs_delta_var : as for
        sky_coverage.sky_coverage_curve.
    sensor : one of ALLSKY_SENSORS; NGS is always 'R'.
    truth  : require an R <= TRUTH_MAG truth star (single-LGS / LTAO only).
    laser_mas : the laser's one-axis TT residual (sensor 'laser').

    Returns sky_coverage_curve's dict (frac, sr, reach, sr_at, frac_at and
    what was run, field = ALLSKY_FIELD) plus levels, P[sight line, level],
    ra, dec, l, b, truth, laser_mas and laser_sr (None unless 'laser').
    """
    if sensor not in ALLSKY_SENSORS:
        raise ValueError(f"sensor must be one of {ALLSKY_SENSORS}, "
                         f"not {sensor!r}")
    if mode == "ngs":
        sensor, tilt_law, truth = "R", "marechal", False
    sky = load_allsky()
    A = sky["area"][:, None]
    band = "K" if sensor in ("K", "R+K") else "H"
    W = _weights(band)
    nr = W.shape[1] - 1
    lam_t = W[:, :nr, :].sum(axis=(1, 2)) * np.pi * patrol ** 2 / A[:, 0]
    laser_sr = None
    if sensor == "laser":
        laser_sr = laser_strehl(args, prep, snap, mode, tilt_law, laser_mas)
        base = (1.0 - np.exp(-lam_t)) if truth else np.ones(len(lam_t))
        P = base[:, None] * (SKY_LEVELS[None, :] <= laser_sr)
    else:
        aR = (_areas(args, prep, snap, mode, "R", tilt_law, patrol,
                     ngs_delta_var) if "R" in sensor else None)
        aX = (_areas(args, prep, snap, mode, band, tilt_law, patrol,
                     ngs_delta_var) if sensor != "R" else None)
        if sensor == "R":
            lam = W.sum(axis=2) @ aR / A
            lam_u = None                 # every STRAP star is a truth star
        elif sensor in ("H", "K"):
            lam = W.sum(axis=1) @ aX / A
            lam_u = W[:, nr, :] @ aX / A
        else:
            best = np.maximum(aR[:, None, :], aX[None, :, :])
            lam = W.reshape(len(W), -1) @ best.reshape(-1, best.shape[2]) / A
            lam_u = W[:, nr, :] @ aX / A
        if truth and lam_u is not None:
            lt = lam_t[:, None]
            P = (1.0 - np.exp(-lam) - np.exp(-lt)
                 + np.exp(-(lt + lam_u)))
        else:
            P = 1.0 - np.exp(-lam)
    P = np.clip(np.minimum.accumulate(P, axis=1), 0.0, 1.0)
    F = P.mean(axis=0)                   # equal-area grid: sky fraction
    if sensor == "laser":                # a step: the same Strehl everywhere
        res = _curve(F)
        res["sr"] = np.full(len(res["frac"]), laser_sr)
        res["sr_at"] = {f: (laser_sr if f <= res["reach"] else float("nan"))
                        for f in COVERAGE_LEVELS}
        res["frac_at"] = {t: (res["reach"] if t <= laser_sr else 0.0)
                          for t in STREHL_LEVELS}
        res["avail"] = base
    else:
        res = _curve(F)
    res.update(levels=SKY_LEVELS, P=P, ra=sky["ra"], dec=sky["dec"],
               l=sky["l"], b=sky["b"], truth=bool(truth),
               laser_mas=float(laser_mas), laser_sr=laser_sr,
               mode=mode, sensor=sensor, tilt_law=tilt_law,
               field=ALLSKY_FIELD, synthetic=False, patrol=float(patrol),
               mag_limit=(None if sensor == "laser" else
                          SENSING[band if sensor in ("H", "K") else "R"][1]),
               lam_nm=float(prep.lam_nm))
    return res


def _curve(F, n=1000):
    """Strehl vs sky fraction from F(level) = sky fraction reaching it."""
    reach = float(F[0])
    frac = np.arange(1, n + 1) / n
    frac = frac[frac <= reach + 1e-12]
    k = (F[None, :] >= frac[:, None] - 1e-12).sum(axis=1) - 1
    k1 = np.minimum(k + 1, len(F) - 1)
    f0, f1 = F[k], F[k1]
    step = SKY_LEVELS[k1] - SKY_LEVELS[k]
    sr = SKY_LEVELS[k] + np.where(f0 > f1, (f0 - frac) / np.where(
        f0 > f1, f0 - f1, 1.0), 0.0) * step
    sr = np.minimum.accumulate(sr)
    return dict(frac=frac, sr=sr, reach=reach,
                sr_at={f: (float(np.interp(f, frac, sr)) if f <= reach
                           and len(frac) else float("nan"))
                       for f in COVERAGE_LEVELS},
                frac_at={t: float(np.interp(t, SKY_LEVELS, F))
                         for t in STREHL_LEVELS})


def sky_probability(result, level):
    """P(Strehl >= level) per sight line, from an allsky_coverage result."""
    if result.get("laser_sr") is not None:
        return result["avail"] * float(level <= result["laser_sr"])
    P, lv = result["P"], result["levels"]
    k = int(np.clip(np.searchsorted(lv, level) - 1, 0, len(lv) - 2))
    t = (level - lv[k]) / (lv[k + 1] - lv[k])
    return np.clip((1 - t) * P[:, k] + t * P[:, k + 1], 0.0, 1.0)


# ---- the map -------------------------------------------------------------------
def _unit(ra, dec):
    ra, dec = np.radians(ra), np.radians(dec)
    return np.c_[np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra),
                 np.sin(dec)]


def smooth_map(ra, dec, values, dec_min, step=1.0, sigma_deg=3.5, k=12):
    """Gaussian-weighted average of per-sight-line values on a regular
    RA / Dec grid (sight lines ~6.4 deg apart); NaN below dec_min."""
    from scipy.spatial import cKDTree
    ra_e = np.arange(0.0, 360.0 + 1e-9, step)
    de_e = np.arange(-90.0, 90.0 + 1e-9, step)
    RA, DE = np.meshgrid(0.5 * (ra_e[1:] + ra_e[:-1]),
                         0.5 * (de_e[1:] + de_e[:-1]))
    d, i = cKDTree(_unit(ra, dec)).query(_unit(RA.ravel(), DE.ravel()), k=k)
    ang = 2.0 * np.degrees(np.arcsin(np.clip(d / 2.0, 0, 1)))
    wgt = np.exp(-0.5 * (ang / sigma_deg) ** 2)
    v = ((wgt * values[i]).sum(1) / wgt.sum(1)).reshape(RA.shape)
    v[DE < dec_min - 2.0] = np.nan
    return ra_e, de_e, v


@lru_cache(maxsize=1)
def _galactic_lines():
    """(b, [(x, y) segments]) for b = 0 and +-30 in Mollweide radians."""
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    out = []
    lon = np.linspace(0, 360, 721)
    for b in (0.0, 30.0, -30.0):
        c = SkyCoord(l=lon * u.deg, b=np.full_like(lon, b) * u.deg,
                     frame="galactic").icrs
        x, y = np.radians(180.0 - c.ra.deg), np.radians(c.dec.deg)
        cut = np.where(np.abs(np.diff(x)) > np.pi)[0] + 1
        out.append((b, list(zip(np.split(x, cut), np.split(y, cut)))))
    return tuple(out)


def draw_sky_map(fig, result, level, title="", what=""):
    """Equatorial Mollweide map of P(Strehl >= level) per sight line from an
    allsky_coverage result, Gaussian-smoothed. `what` names the guide
    option in the colour-bar label."""
    from matplotlib.colors import LinearSegmentedColormap
    fig.clear()
    vals = sky_probability(result, level)
    dec_min = float(np.min(result["dec"]))
    ra_e, de_e, v = smooth_map(result["ra"], result["dec"], vals, dec_min)
    ax = fig.add_axes([0.03, 0.17, 0.94, 0.70], projection="mollweide")
    RA, DE = np.meshgrid(ra_e, de_e)
    cmap = LinearSegmentedColormap.from_list("sky_blue", _MAP_RAMP)
    m = ax.pcolormesh(np.radians(180.0 - RA), np.radians(DE), v, cmap=cmap,
                      vmin=0.0, vmax=1.0, shading="flat", rasterized=True)
    for b, segs in _galactic_lines():
        for x, y in segs:
            ax.plot(x, y, color="0.1", lw=1.1 if b == 0 else 0.8,
                    ls="-" if b == 0 else (0, (2, 2)), alpha=0.75)
    ax.axhspan(-np.pi / 2, np.radians(dec_min), color="#e9e8e3", lw=0,
               zorder=0.5)
    ax.grid(True, color="white", lw=0.5, alpha=0.6)
    ax.set_xticks(np.radians([-120, -60, 0, 60, 120]))
    ax.set_xticklabels(["20h", "16h", "12h", "8h", "4h"], color="white",
                       fontweight="bold", fontsize=9)
    ax.tick_params(axis="y", labelsize=8.5)
    sky = float(vals.mean())
    ax.set_title((f"{title}\n" if title else "")
                 + f"Strehl ≥ {level:.2f} on {100 * sky:.0f} % of the sky "
                 "(dec ≥ −40°)", fontsize=10)
    cax = fig.add_axes([0.30, 0.08, 0.40, 0.03])
    cb = fig.colorbar(m, cax=cax, orientation="horizontal")
    cb.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cb.set_ticklabels(["0 %", "25 %", "50 %", "75 %", "100 %"])
    cb.outline.set_visible(False)
    cax.set_title(f"pointings where {what or 'the best star'} reaches "
                  f"Strehl ≥ {level:.2f}", fontsize=9)
    fig.text(0.01, 0.01, "Equatorial Mollweide, RA increasing left. Solid: "
             "Galactic plane; dashed: b = ±30°. Grey: dec < −40°, not "
             "computed. 821 sight lines ~6.4° apart, smoothed σ 3.5°.",
             fontsize=7.5, color="0.4")
    return ax
