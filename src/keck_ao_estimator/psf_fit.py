"""Neighbour subtraction: PSF fitting used to REMOVE contaminants, not to
measure the star (Decision D2).

The measured-SR tool's crowded-field mitigations to date -- the crowding
metric, robust sky, pick sky, the EE aperture correction -- all attack the
SKY estimate or the aperture CONVENTION.  None of them touches the actual
failure of aperture photometry in a crowded field: neighbours sitting
INSIDE the photometry aperture.  This module does exactly that and nothing
else.  For each star:

  1. select the catalogued neighbours whose modelled light lands in the
     target's aperture or sky annulus;
  2. fit the ePSF to those neighbours AND to the target simultaneously,
     over a shared local background;
  3. subtract the fitted NEIGHBOURS only -- never the target;
  4. hand the cleaned array back so the existing, unchanged
     `measure_strehl` photometry runs on it.

The target is in the fit but not in the subtraction because leaving it out
would let the neighbours' amplitudes absorb the target's own light, which
is the classic way a "cleaning" step eats the signal it was protecting.

Why not measure by PSF fitting instead: with one fitted ePSF every star has
the same shape, so peak/flux is a FIELD CONSTANT.  That is not a per-star
Strehl and must never be presented as one.  Keeping the aperture
measurement also keeps the convention -- field map, consistency clip, EE
correction and the KAON's two-convention reporting all compose with it
unchanged instead of forking the metric.

Qt-free by rule (numpy/scipy/astropy only).
"""
from dataclasses import dataclass, field

import numpy as np

from .image_strehl import aperture_flux
from .nirc2 import (
    NIRC2_BG_INNER_RADIUS_ARCSEC, NIRC2_BG_OUTER_RADIUS_ARCSEC,
    NIRC2_PHOTOMETRY_RADIUS_ARCSEC,
)

__all__ = [
    "PSF_FIT_MAX_NEIGHBOURS", "PSF_FIT_NEIGHBOUR_FLOOR_FRAC",
    "PSF_FIT_POS_TOL_FWHM", "PSF_FIT_SIGMA_REJECT", "PSF_FIT_FOOTPRINT_FWHM",
    "PSF_FIT_MAX_SUBTRACTED_FRAC", "PSF_FIT_SR_VALIDATED_MAX",
    "PSF_FIT_SR_ENVELOPE_NOTE", "PSF_FIT_BIAS_SAFE_NOTE",
    "PSF_FIT_BIAS_UNSAFE_NOTE",
    "Neighbour", "CleanReport", "clean_star", "select_neighbours",
    "group_fit", "component_footprint",
]

# keep the N largest predicted contributors.  Not a quality judgement --
# a runtime bound: the fit is O(n_components) in both the linear solve and
# the Jacobian, and a GC-density field puts ~75 catalogued stars inside a
# target's neighbour-search radius.  Everything dropped is COUNTED and its
# predicted aperture contribution is summed into the report, so the user
# sees exactly what was left behind (never silent -- the EE feature's
# quiet empty-field path cost a round trip once).
PSF_FIT_MAX_NEIGHBOURS = 16

# a candidate is only worth fitting if its predicted flux inside the
# target's aperture exceeds this fraction of the target's own aperture
# flux.  0.1% is an order of magnitude below the 1-2% SR effects this
# feature exists to fix, so the floor cannot hide a relevant contaminant.
PSF_FIT_NEIGHBOUR_FLOOR_FRAC = 0.001

# how far a component's position may move from its catalogued position,
# in units of the ePSF FWHM.  The catalogue comes from `cntrd`, already
# sub-pixel; letting a component wander further lets it lock onto a
# different star, or onto the target's own core.
PSF_FIT_POS_TOL_FWHM = 0.35

# fit footprint radius per component, in ePSF FWHM.  6 FWHM (~27 px on
# NIRC2 narrow at Kp) captures the core and the first several diffraction
# rings, where essentially all of the fit information lives; including the
# whole annulus disc for every component would multiply the pixel count
# several-fold for residual sensitivity that the sky noise has already
# swamped.
PSF_FIT_FOOTPRINT_FWHM = 6.0

# pixel rejection.  A hard 5-sigma cut with ONE refit, instead of a robust
# loss (soft_l1 etc.): the amplitudes are solved LINEARLY inside the fit
# (variable projection), and an exact linear solve on a clean pixel set is
# both faster and far easier to reason about than a robust nonlinear solve
# whose amplitude step is only approximately optimal.  What the cut is
# for: cosmic rays, bad pixels the mask missed, and saturated cores.
PSF_FIT_SIGMA_REJECT = 5.0

# Refuse the cleaning when this much of the target's aperture flux turned
# out to be neighbour light (D19, adopted by Eduardo 2026-07-29).  Above
# ~0.9 the "target" was a MINORITY of the light in its own aperture, so
# what survives subtraction is a small difference of large numbers and the
# measurement is mostly model.  Measured on a GC-density synthetic field,
# gating here makes the cleaned set better than the uncleaned one on BOTH
# metrics -- median SR bias -0.1428 -> +0.0526 and MAD 0.0883 -> 0.0769 --
# where the ungated set improved the median but nearly doubled the scatter.
# Stable across 0.70-0.95.  Note the flag that does NOT work for this:
# `residual_frac`, which is anti-discriminating (see STATUS, Diagnostic A).
PSF_FIT_MAX_SUBTRACTED_FRAC = 0.95

# Strehl above which this feature's accuracy is NOT established (D25).
# Measured envelope, 3 noise realizations, ePSF strict/converged at every
# SR: the |bias| <= 0.02 target region is met 24/24 at SR 0.15 and 0.30,
# 18/24 at 0.60, and 13/20 at 0.80, worst case +0.07 on equal-brightness
# pairs inside 0.3".  The mechanism is localized -- neighbour flux is
# over-subtracted from the aperture while the peak is untouched, and
# SR = peak/flux, so over-subtracting flux inflates SR -- but the CAUSE is
# unexplained after sixteen falsified hypotheses.  Full record, including
# every refutation and where to resume, in the psf_fit development notes
# (repo history: plans/psf_fit/HIGH_SR_ENVELOPE.md).
#
# Not enforced here: `clean_star` works in flux and never computes a
# Strehl.  The check belongs where SR is known -- `measure_strehl` (O4)
# and the GUI (WP-3) -- and the constant lives here so both use one number
# and neither invents its own.
PSF_FIT_SR_VALIDATED_MAX = 0.30

# Which way the residual error runs, and why the user must be told (D27,
# Eduardo 2026-07-31: "an underestimation is always preferred to an
# overestimation ... as long as it is clear to the user that the likely
# result is an underestimation").
#
# SR = peak / flux, so the sign follows directly:
#   UNDER-subtracting leaves neighbour light in the aperture -> flux too
#   high -> SR too LOW. An underestimate. SAFE.
#   OVER-subtracting removes real flux -> flux too low -> SR too HIGH. An
#   overestimate, and the direction that misleads.
#
# Measured, and NOT uniform:
#   * At SR <= 0.30 the model under-predicts aperture flux by ~1.5 %, so it
#     under-subtracts and the residual error is an UNDERESTIMATE. Safe.
#   * Above SR 0.30 the measured S2 bias is POSITIVE, +0.03 at sr=0.60 and
#     up to +0.07 at sr=0.80 -- an OVERESTIMATE, the unsafe direction. That
#     is why the envelope warning has to name the direction and not merely
#     say "less accurate".
#   * The theoretical PSF (D26) carries no static speckle or instrument
#     structure, so it models less of the star than reality and
#     under-subtracts. Underestimate. Safe.
PSF_FIT_BIAS_SAFE_NOTE = (
    "Expected bias: UNDERESTIMATE (any un-subtracted neighbour light "
    "inflates the flux, which lowers peak/flux). Erring low is the "
    "intended direction.")
PSF_FIT_BIAS_UNSAFE_NOTE = (
    "WARNING -- expected bias: OVERESTIMATE. Above Strehl "
    f"{PSF_FIT_SR_VALIDATED_MAX:.2f} the measured bias is POSITIVE "
    "(+0.03 at SR 0.60, up to +0.07 at SR 0.80), i.e. this SR is likely "
    "too HIGH, not too low. Treat it as an upper bound.")
PSF_FIT_SR_ENVELOPE_NOTE = (
    "PSF-fit cleaning is validated to |SR bias| <= 0.02 for Strehl <= "
    "0.30; above that its accuracy degrades (worst measured +0.07 at "
    "Strehl 0.80 for equal-brightness pairs inside 0.3\").")


@dataclass(frozen=True)
class Neighbour:
    """One fitted contaminant.  `subtracted` is False for a candidate that
    was fitted but deliberately not removed (saturated core), and for one
    dropped by the max-neighbour cap; `note` says which."""
    x: float                    # catalogued position
    y: float
    fit_x: float                # fitted position
    fit_y: float
    amp: float                  # stamp flux in the measurement sky convention
    flux_in_aperture: float     # modelled flux inside the TARGET's aperture
    sep_arcsec: float
    saturated: bool
    subtracted: bool
    note: str = ""


@dataclass(frozen=True)
class CleanReport:
    """Everything the cleaning did, or refused to do, for one star.

    Every field here is meant to be reportable.  A path that produced no
    change still returns a report with `cleaned=False` and a `note` that
    names the reason -- there is no outcome this feature does not log.
    """
    cleaned: bool
    note: str                   # human sentence, GUI-log ready
    epsf_tag: str               # "strict" | "loose" | "uncalibrated"
    n_candidates: int
    n_subtracted: int
    n_dropped: int
    n_saturated: int
    subtracted_flux: float      # model flux removed from inside the aperture
    subtracted_frac: float      # ... as a fraction of the UNCLEANED flux
    dropped_frac: float         # predicted aperture contribution of the
                                # candidates the cap dropped, same units
    residual_frac: float        # see `clean_star` for the exact definition
    crowding_before: float
    crowding_after: float
    n_fit_pixels: int
    n_rejected_pixels: int
    fit_status: int             # scipy least_squares status (<=0 = refused)
    # D20: this star should be left OFF the field map by default, but
    # remain reinsertable by the user -- exactly how a `field_consistent`
    # outlier is treated today (GUI: `_n2_field_dropped`, x on the map,
    # `_on_nirc2_reject_star` puts it back with "reinserted into the fit
    # by user").  Set ONLY for the over-contamination refusal: a star
    # refused because it had no neighbours worth subtracting is a
    # perfectly good measurement and must stay on the map.  Encoded as a
    # field rather than inferred from `note`, so no consumer has to
    # pattern-match a human sentence to decide what to plot.
    exclude_from_field: bool = False
    neighbours: tuple = field(default=(), repr=False)


def select_neighbours(catalog, pos, params, epsf_model, target_flux,
                      photometry_radius_arcsec=NIRC2_PHOTOMETRY_RADIUS_ARCSEC,
                      bg_outer_arcsec=NIRC2_BG_OUTER_RADIUS_ARCSEC,
                      max_neighbours=PSF_FIT_MAX_NEIGHBOURS,
                      floor_frac=PSF_FIT_NEIGHBOUR_FLOOR_FRAC):
    """Pick which catalogued stars get fitted for the target at `pos`.

    Search radius is `bg_outer + r_stamp`: any star farther than that has
    no modelled light inside the target's annulus at all.  Candidates are
    ranked by PREDICTED contamination -- their catalogued flux times the
    ePSF integrated over the target's aperture -- rather than by brightness
    or by separation, because a bright star just outside the annulus can
    matter less than a faint one sitting on the aperture edge.

    Returns (kept, dropped) as two lists of candidate dicts.  `kept` is
    everything above `floor_frac * target_flux`, capped at
    `max_neighbours`; `dropped` is the remainder that cleared the floor but
    lost the cap, and its summed predicted contribution goes into the
    report so the shortfall is visible.
    """
    ps = float(params.plate_scale_mas)
    photrad = photometry_radius_arcsec * 1000.0 / ps
    reach = bg_outer_arcsec * 1000.0 / ps + epsf_model.r_stamp_px
    tx, ty = float(pos[0]), float(pos[1])
    fwhm = float(epsf_model.fwhm_px)

    # the target's own aperture pixels, subsampled: this is a RANKING
    # integral, and a stride-4 sample of a 31 800-pixel disc estimates it
    # to well under the 0.1 % floor's precision while costing 1/16 as many
    # interpolations across ~75 GC-density candidates
    stride = 4
    r_i = int(np.ceil(photrad))
    ay, ax = np.mgrid[-r_i:r_i + 1:stride, -r_i:r_i + 1:stride]
    inside = (ay * ay + ax * ax) <= photrad * photrad
    ap_y = (ay[inside] + int(round(ty))).astype(float)
    ap_x = (ax[inside] + int(round(tx))).astype(float)
    cell_area = float(stride * stride)

    # Amplitude estimate from the CORE HEIGHT, not from the catalogue's
    # small-aperture flux.  psi is normalized to unit flux over the whole
    # stamp, so an amplitude means "total stamp flux"; the catalogue flux is
    # measured in a 2 x FWHM aperture and captures only ~26 % of that for a
    # seeing-halo PSF (measured on S2 frames: cat/true = 0.264 even at 1.2"
    # separation, where no blending can be blamed).  Multiplying one by the
    # other mixes two conventions and under-predicts contamination ~4x,
    # which silently stiffens the floor.  peak / psi_peak is convention-
    # consistent and, being a local maximum, is also far more robust than an
    # annulus-based flux for a star embedded in a brighter star's halo.
    pk = float(getattr(epsf_model, "peak_value", 0.0))
    rows = []
    for c in catalog:
        cxx, cyy = float(c["x"]), float(c["y"])
        d = np.hypot(cxx - tx, cyy - ty)
        if d > reach or d <= 0.5 * fwhm:
            continue                    # out of reach, or the target itself
        amp = (float(c.get("peak", 0.0)) / pk if pk > 0
               else float(c.get("flux", 0.0)))
        # predicted flux this star deposits INSIDE the target's aperture
        shape = epsf_model.evaluate_at(ap_y, ap_x, cxx, cyy)
        pred = amp * float(shape.sum()) * cell_area
        rows.append({"x": cxx, "y": cyy, "flux": amp,
                     "peak": float(c.get("peak", 0.0)),
                     "sep_arcsec": float(d * ps / 1000.0),
                     "predicted": pred})

    floor = abs(float(target_flux)) * float(floor_frac)
    above = [r for r in rows if r["predicted"] > floor]
    # largest predicted contribution first, then the closer one -- a tie
    # in predicted flux is broken toward the star whose mis-subtraction
    # would land nearer the target's core
    above.sort(key=lambda r: (-r["predicted"], r["sep_arcsec"]))
    return above[:int(max_neighbours)], above[int(max_neighbours):]


def component_footprint(shape, components, r_comp, clip_center=None,
                        clip_radius=None):
    """Pixels worth fitting: the union of radius-`r_comp` discs around the
    components, optionally clipped to a disc around `clip_center`.

    Shared by `clean_star` and by `epsf.build_epsf`'s donor cleaning so the
    two cannot drift apart -- and because handing the fit a whole stamp box
    instead of the components' neighbourhoods is a 16x pixel-count penalty
    for information the sky noise has already swamped (measured: 80 089
    pixels versus ~4 900 for a two-component fit on NIRC2 narrow).

    Returns (yy, xx) flat index arrays, empty when nothing qualifies.
    """
    comps = [(float(cx), float(cy)) for cx, cy in components]
    if not comps:
        return np.array([], dtype=int), np.array([], dtype=int)
    lo_y = max(int(np.floor(min(c[1] for c in comps) - r_comp)), 0)
    hi_y = min(int(np.ceil(max(c[1] for c in comps) + r_comp)), shape[0] - 1)
    lo_x = max(int(np.floor(min(c[0] for c in comps) - r_comp)), 0)
    hi_x = min(int(np.ceil(max(c[0] for c in comps) + r_comp)), shape[1] - 1)
    if hi_y < lo_y or hi_x < lo_x:
        return np.array([], dtype=int), np.array([], dtype=int)
    gy, gx = np.mgrid[lo_y:hi_y + 1, lo_x:hi_x + 1]
    near = np.zeros(gy.shape, dtype=bool)
    for cxx, cyy in comps:
        near |= ((gy - cyy) ** 2 + (gx - cxx) ** 2) <= r_comp ** 2
    if clip_center is not None and clip_radius is not None:
        near &= (((gy - float(clip_center[1])) ** 2
                  + (gx - float(clip_center[0])) ** 2)
                 <= float(clip_radius) ** 2)
    return gy[near].ravel(), gx[near].ravel()


def group_fit(image, components, epsf_model, footprint, sky_sigma,
              background="constant", pos_tol_px=None,
              sigma_reject=PSF_FIT_SIGMA_REJECT, badmask=None,
              saturation=None):
    """Least-squares fit of `components` to `image` over `footprint`.

    Parameterization -- VARIABLE PROJECTION.  Amplitudes and the background
    enter the model LINEARLY, so they are not given to the nonlinear
    solver: at every trial position the amplitudes and background are
    solved exactly by `numpy.linalg.lstsq` on the design matrix, and only
    the 2 positional parameters per component go to
    `scipy.optimize.least_squares`.  This halves the nonlinear dimension,
    removes the amplitude/background degeneracy from the trust region
    entirely, and makes the fit converge in ~10-15 iterations instead of
    stalling on a valley.

    Negative amplitudes are not physical.  Rather than a constrained
    solver, a component whose solved amplitude goes negative is dropped
    (amplitude pinned to 0) and the linear system is re-solved, at most
    once per component; every drop is recorded.

    Jacobian: analytic.  d(model)/d(component x) is the interpolated
    gradient of psi, which `EpsfModel.gradient_at` returns from grids
    pre-computed at build time -- 3 interpolations per component instead of
    a numerical Jacobian's 2 full model evaluations per component per
    iteration.

    Background model: CONSTANT, and `background="plane"` exists but is not
    the default.  The footprint spans at most the sky annulus (1.4" on
    NIRC2 narrow); over that scale a tilted plane is strongly degenerate
    with a bright neighbour's halo gradient, which is precisely how a fit
    starts eating real flux.  The sky level itself is re-estimated by
    `aperture_flux` on the cleaned array afterwards, so the constant here
    only has to absorb the local pedestal.

    Loss is LINEAR, with outliers removed rather than down-weighted:
    pixels flagged in `badmask`, pixels at or above the saturation level,
    and (after one pass) pixels deviating more than `sigma_reject` x
    sky_sigma from the model are dropped and the fit is repeated once.

    Returns (amps, positions, background, residual, info) where `info`
    carries the solver status, the pixel counts, and the dropped-component
    list.  A non-converged solve returns status <= 0 and the caller
    REFUSES the cleaning rather than subtracting a bad model.

    (Implemented in O2 rather than O3: `epsf.build_epsf` needs exactly
    this fitter to subtract the donors' own neighbours between build
    cycles, and duplicating it inside `epsf.py` would have meant two
    fitters to keep honest.  `epsf` imports from `psf_fit`; the
    dependency runs one way only.)
    """
    yy, xx = footprint
    yy = np.asarray(yy, dtype=int).ravel()
    xx = np.asarray(xx, dtype=int).ravel()
    comps = [(float(cx), float(cy)) for cx, cy in components]
    n_comp = len(comps)
    if n_comp == 0:
        raise ValueError("group_fit needs at least one component")

    n_bg = 3 if background == "plane" else 1
    data_full = np.asarray(image, dtype=float)[yy, xx]
    keep = np.isfinite(data_full)
    if badmask is not None:
        keep &= ~np.asarray(badmask, dtype=bool)[yy, xx]
    if saturation is not None and np.isfinite(saturation):
        # Saturated pixels carry no amplitude information -- their value is
        # the detector ceiling, not the star.  Leaving them in lets a
        # clipped core drag the whole joint solve, which corrupts the
        # amplitudes of the star's NEIGHBOURS as well as its own.  The
        # docstring promised this exclusion from the start; it was not
        # implemented until a CP2 saturated-neighbour demonstration failed.
        keep &= data_full < float(saturation)
    if keep.sum() < 2 * n_comp + n_bg:
        return (np.zeros(n_comp), np.array(comps, dtype=float), 0.0,
                np.zeros(yy.size),
                {"status": -1, "message": "too few usable pixels",
                 "n_fit_pixels": int(keep.sum()), "n_rejected": 0,
                 "dropped": list(range(n_comp))})

    sky_sigma = float(sky_sigma) if sky_sigma and sky_sigma > 0 else 1.0
    if pos_tol_px is None:
        pos_tol_px = PSF_FIT_POS_TOL_FWHM * float(epsf_model.fwhm_px)
    pos_tol_px = max(float(pos_tol_px), 1e-3)

    # background columns are constant across the fit; build them once
    def _bg_columns(sel):
        cols = [np.ones(int(sel.sum()))]
        if n_bg == 3:
            cols.append(xx[sel] - xx[sel].mean())
            cols.append(yy[sel] - yy[sel].mean())
        return cols

    state = {"amps": np.ones(n_comp), "dropped": set()}

    def _design(p, sel):
        cols = []
        for k, (cx, cy) in enumerate(comps):
            cols.append(epsf_model.evaluate_at(
                yy[sel], xx[sel], cx + p[2 * k], cy + p[2 * k + 1]))
        cols.extend(_bg_columns(sel))
        return np.column_stack(cols)

    def _solve(p, sel):
        """Exact linear solve for amplitudes + background at positions p,
        dropping negative amplitudes one at a time (at most once per
        component) rather than reaching for a constrained solver."""
        A = _design(p, sel)
        d = data_full[sel]
        live = list(range(n_comp))
        amps = np.zeros(n_comp)
        bg = 0.0
        for _ in range(n_comp + 1):
            idx = live + list(range(n_comp, n_comp + n_bg))
            c, *_ = np.linalg.lstsq(A[:, idx], d, rcond=None)
            amps = np.zeros(n_comp)
            for j, k in enumerate(live):
                amps[k] = c[j]
            bg = float(c[len(live)])        # first background column
            neg = [k for k in live if amps[k] < 0.0]
            if not neg or not live:
                break
            live.remove(min(neg, key=lambda k: amps[k]))
        state["amps"] = amps
        state["dropped"] = set(range(n_comp)) - set(live)
        return amps, bg, A

    def _model(p, sel):
        amps, bg, A = _solve(p, sel)
        c = np.concatenate([amps, [bg] + [0.0] * (n_bg - 1)])
        return A @ c

    def _resid(p, sel):
        return (data_full[sel] - _model(p, sel)) / sky_sigma

    def _jac(p, sel):
        """Kaufman approximation: hold the linearly-solved amplitudes fixed
        while differentiating the positional parameters.  Standard for
        variable projection, and cheap here because the gradient comes
        from pre-computed grids rather than finite differences."""
        amps = state["amps"]
        J = np.zeros((int(sel.sum()), 2 * n_comp))
        for k, (cx, cy) in enumerate(comps):
            gx, gy = epsf_model.gradient_at(
                yy[sel], xx[sel], cx + p[2 * k], cy + p[2 * k + 1])
            J[:, 2 * k] = -amps[k] * gx / sky_sigma
            J[:, 2 * k + 1] = -amps[k] * gy / sky_sigma
        return J

    from scipy.optimize import least_squares

    sel = keep.copy()
    p0 = np.zeros(2 * n_comp)
    lo = np.full(2 * n_comp, -pos_tol_px)
    hi = np.full(2 * n_comp, pos_tol_px)
    result = None
    n_rejected = 0
    for it in range(2):          # one refit after sigma rejection
        result = least_squares(_resid, p0, jac=_jac, args=(sel,),
                               bounds=(lo, hi), method="trf",
                               xtol=1e-4, ftol=1e-4, gtol=1e-8,
                               max_nfev=200 + 40 * n_comp)
        if it == 1 or result.status <= 0:
            break
        res = data_full[sel] - _model(result.x, sel)
        bad = np.abs(res) > sigma_reject * sky_sigma
        if not bad.any():
            break
        idx = np.flatnonzero(sel)[bad]
        sel[idx] = False
        n_rejected = int(bad.sum())
        if sel.sum() < 2 * n_comp + n_bg:
            sel[idx] = True      # rejection would starve the fit; undo it
            n_rejected = 0
            break
        p0 = result.x

    amps, bg, _ = _solve(result.x, sel)
    positions = np.array(
        [[comps[k][0] + result.x[2 * k], comps[k][1] + result.x[2 * k + 1]]
         for k in range(n_comp)], dtype=float)
    resid_full = data_full - _model_full(image, yy, xx, comps, result.x,
                                         amps, bg, epsf_model)
    info = {"status": int(result.status), "message": str(result.message),
            "n_fit_pixels": int(sel.sum()), "n_rejected": n_rejected,
            "dropped": sorted(state["dropped"]), "nfev": int(result.nfev),
            "cost": float(result.cost)}
    return amps, positions, float(bg), resid_full, info


def _model_full(image, yy, xx, comps, p, amps, bg, epsf_model):
    """Model over the WHOLE footprint (including sigma-rejected pixels),
    so the returned residual covers every pixel the caller passed in."""
    out = np.full(yy.size, float(bg))
    for k, (cx, cy) in enumerate(comps):
        if amps[k] == 0.0:
            continue
        out += epsf_model.evaluate_at(yy, xx, cx + p[2 * k],
                                      cy + p[2 * k + 1], amp=amps[k])
    return out


def clean_star(image, pos, params, epsf, *, catalog=None,
               photometry_radius_arcsec=NIRC2_PHOTOMETRY_RADIUS_ARCSEC,
               bg_inner_arcsec=NIRC2_BG_INNER_RADIUS_ARCSEC,
               bg_outer_arcsec=NIRC2_BG_OUTER_RADIUS_ARCSEC,
               robust_sky=False, sky_override=None,
               max_neighbours=PSF_FIT_MAX_NEIGHBOURS,
               floor_frac=PSF_FIT_NEIGHBOUR_FLOOR_FRAC,
               background="constant", subtract_saturated=False,
               max_subtracted_frac=PSF_FIT_MAX_SUBTRACTED_FRAC,
               sigma_reject=PSF_FIT_SIGMA_REJECT, badmask=None):
    """Subtract the target's neighbours.  Returns (cleaned_image, report).

    `image` is the SIGMA-FILTERED work array, full frame, and the return is
    a full-frame copy with only the fit footprint modified.  Full frame,
    not a cutout, because `measure_strehl` centroids, extracts its
    `find_peak` box, runs `radial_profile_fwhm` and computes
    `aperture_edge_clip_frac` in absolute detector coordinates; handing it
    a cutout would mean re-mapping every one of those and would silently
    break the edge geometry.

    Refusal conditions -- any one of these keeps the UNCLEANED array and
    returns `cleaned=False` with the reason in `note`:

      * the ePSF is unusable (tag "uncalibrated");
      * no candidate clears the contamination floor (a genuinely isolated
        star -- this is the S1 no-op path and it must be bit-exact);
      * the solver did not converge (status <= 0);
      * every component was dropped for negative amplitude;
      * the crowding metric measured on the CLEANED array is higher than
        on the uncleaned one.  Subtraction that makes the sky-estimator
        disagreement worse has removed the wrong thing; PLAN Section 5
        makes refusing mandatory, and the note says so out loud.

    Saturated neighbours are FLAGGED, NOT FAKED.  A neighbour whose core
    exceeds the frame saturation level has no measurable core, so its
    amplitude would be constrained only by its wings -- wing fitting, which
    PLAN Section 3 puts out of scope until Eduardo rules on it at CP2.
    Default `subtract_saturated=False`: such a neighbour is fitted (so it
    does not corrupt its neighbours' amplitudes) but not subtracted, and
    both `n_saturated` and its `Neighbour.note` record it.

    RESIDUAL METRIC (the exact definition; this is a reported contract):

        residual_frac = max(sum_{p in A} |res(p)| - F_noise, 0)
                        ------------------------------------------
                              sum_{p in A} sum_i M_i(p)

        F_noise = N_ap * sigma_sky * sqrt(2/pi)

    where A is the target's photometry aperture, res is the group fit's
    residual (data minus the FULL model, target included), and M_i are the
    fitted neighbour models.  Denominator = `subtracted_flux`, the model
    flux actually removed from the aperture; numerator = everything the fit
    failed to explain there.

    `F_noise` is the residual that noise alone explains -- for zero-mean
    Gaussian noise E|res| = sigma*sqrt(2/pi) per pixel (D14).  Without it
    the ratio is not conservative, it is meaningless: summed over ~31 800
    aperture pixels at ~10 ADU the noise term alone is ~3e5 ADU against a
    subtracted flux of ~1e4, and the metric reads several hundred per cent
    on subtractions that measurably work.

    What survives the subtraction still includes the target's own model
    error, so the metric continues to OVER-state the leftover neighbour
    light.  That part is deliberate: the conservative direction is the one
    that never claims a cleaner measurement than it delivered.
    `residual_frac` is 0.0 exactly when nothing was subtracted.
    """
    from .epsf import _box, _robust_sky, deep_star_catalog

    work = np.asarray(image, dtype=float)
    ps = float(params.plate_scale_mas)
    tx, ty = float(pos[0]), float(pos[1])
    photrad = photometry_radius_arcsec * 1000.0 / ps
    r_in = bg_inner_arcsec * 1000.0 / ps
    r_out = bg_outer_arcsec * 1000.0 / ps

    def _refuse(note, **kw):
        base = dict(
            cleaned=False, note=note,
            epsf_tag=getattr(epsf, "tag", ""), n_candidates=0,
            n_subtracted=0, n_dropped=0, n_saturated=0,
            subtracted_flux=0.0, subtracted_frac=0.0, dropped_frac=0.0,
            residual_frac=0.0, crowding_before=0.0, crowding_after=0.0,
            n_fit_pixels=0, n_rejected_pixels=0, fit_status=0,
            neighbours=())
        base.update(kw)
        return work, CleanReport(**base)

    if epsf is None or not getattr(epsf, "usable", False):
        return _refuse(
            "cleaning skipped: no usable ePSF ("
            + (getattr(epsf, "note", "") or "no ePSF supplied") + ")")

    model = epsf.at(tx, ty)
    fwhm_cat = float(model.fwhm_px)
    if catalog is None:
        catalog = deep_star_catalog(work, params)

    # uncleaned baseline: the sky mode must match what the measurement
    # will use, or crowding_before/after are not the same statistic
    if sky_override is not None:
        skyval = float(sky_override)
    else:
        skyval = None
    flux0, sky0, crowd0, sky_sigma0, _n_ap, _n_ann = aperture_flux(
        work, photrad, tx, ty, insky_px=r_in, outsky_px=r_out,
        skyval=skyval, robust=robust_sky)
    if not np.isfinite(flux0) or flux0 == 0.0:
        return _refuse("cleaning skipped: target aperture flux is zero or "
                       "non-finite, so there is nothing to clean against")
    _sky_g, sky_sigma = _robust_sky(work)
    if sky_sigma0 > 0.0:
        sky_sigma = sky_sigma0

    kept, dropped = select_neighbours(
        catalog, (tx, ty), params, model, flux0,
        photometry_radius_arcsec=photometry_radius_arcsec,
        bg_outer_arcsec=bg_outer_arcsec, max_neighbours=max_neighbours,
        floor_frac=floor_frac)
    n_cand = len(kept) + len(dropped)
    dropped_frac = (sum(r["predicted"] for r in dropped) / abs(flux0)
                    if dropped else 0.0)
    if not kept:
        return _refuse(
            f"0 neighbours above the {100 * floor_frac:.1f}% contamination "
            f"floor within {bg_outer_arcsec:.1f}\" -- nothing to subtract "
            f"(the star is effectively isolated)",
            crowding_before=float(crowd0), crowding_after=float(crowd0),
            n_candidates=n_cand, epsf_tag=epsf.tag)

    # Saturation is an ABSOLUTE detector level, so it must be tested
    # against the RAW core maximum -- the catalogue's `peak` is
    # sky-subtracted, and comparing that to the ceiling under-reads by the
    # sky and misses genuinely saturated stars (on the S4b frame, clipped
    # at 8000 ADU with ~100 ADU of sky, every saturated neighbour tested
    # as 7900 < 8000 and was silently subtracted with an amplitude its
    # clipped core could not constrain).  This mirrors `measure_strehl`'s
    # own saturation check, which also uses the raw sub-image maximum.
    sat_level = float(params.max_counts) * float(params.coadds)
    _r_core = max(1.0, 0.75 * fwhm_cat)
    for r in kept:
        _sl, _, _ = _box(work.shape, r["x"], r["y"], _r_core)
        _sub = work[_sl]
        r["raw_peak"] = float(_sub.max()) if _sub.size else 0.0
        r["saturated"] = bool(r["raw_peak"] >= sat_level)

    # --- fit footprint: components' cores, clipped to the target's own
    # aperture+annulus disc.  Everything outside is sky-noise-swamped.
    fwhm = float(model.fwhm_px)
    r_comp = PSF_FIT_FOOTPRINT_FWHM * fwhm
    comps = [(tx, ty)] + [(r["x"], r["y"]) for r in kept]
    fy, fx = component_footprint(work.shape, comps, r_comp,
                                 clip_center=(tx, ty), clip_radius=r_out)
    if fy.size < 2 * len(comps) + 4:
        return _refuse(
            "cleaning skipped: fit footprint too small (star too close to "
            "the array edge for a constrained fit)",
            crowding_before=float(crowd0), crowding_after=float(crowd0),
            n_candidates=n_cand, epsf_tag=epsf.tag)

    amps, positions, bg, resid, info = group_fit(
        work, comps, model, (fy, fx), sky_sigma, background=background,
        sigma_reject=sigma_reject, badmask=badmask, saturation=sat_level)
    if info["status"] <= 0:
        return _refuse(
            f"cleaning refused: the group fit did not converge "
            f"({info.get('message', 'status <= 0')})",
            crowding_before=float(crowd0), crowding_after=float(crowd0),
            n_candidates=n_cand, epsf_tag=epsf.tag,
            fit_status=int(info["status"]),
            n_fit_pixels=int(info["n_fit_pixels"]))

    # --- which neighbours actually get removed -------------------------
    subtract_idx = []
    for k, r in enumerate(kept, start=1):
        if amps[k] <= 0.0:
            r["note"] = "dropped by the fit (non-positive amplitude)"
        elif r["saturated"] and not subtract_saturated:
            r["note"] = ("saturated core -- FLAGGED, not subtracted; its "
                         "amplitude would be set by wings alone")
        else:
            r["note"] = ""
            subtract_idx.append(k)
    n_sat = sum(1 for r in kept if r["saturated"])

    def _neighbour_tuple(fluxes=None):
        """Neighbour detail, available to the REFUSAL paths too -- a user
        told only "refused" cannot see that the culprit was a saturated
        star 0.4" away.  Never silent (RULES section 5)."""
        return tuple(
            Neighbour(x=r["x"], y=r["y"],
                      fit_x=float(positions[k][0]),
                      fit_y=float(positions[k][1]),
                      amp=float(amps[k]),
                      flux_in_aperture=float((fluxes or {}).get(k, 0.0)),
                      sep_arcsec=r["sep_arcsec"], saturated=r["saturated"],
                      subtracted=(k in subtract_idx) if fluxes else False,
                      note=r.get("note", ""))
            for k, r in enumerate(kept, start=1))

    if not subtract_idx:
        return _refuse(
            "cleaning refused: every candidate was either dropped by the "
            f"fit or saturated ({n_sat} saturated of {len(kept)})",
            crowding_before=float(crowd0), crowding_after=float(crowd0),
            n_candidates=n_cand, n_saturated=n_sat, epsf_tag=epsf.tag,
            fit_status=int(info["status"]),
            n_fit_pixels=int(info["n_fit_pixels"]),
            neighbours=_neighbour_tuple())

    # --- subtract, over the neighbours' own stamps (not just the fit
    # footprint): the aperture reaches pixels the fit did not need
    cleaned = work.copy()
    for k in subtract_idx:
        nxx, nyy = positions[k]
        _sl, syy, sxx = _box(work.shape, nxx, nyy, model.r_stamp_px)
        m = model.evaluate_at(syy.ravel(), sxx.ravel(), nxx, nyy,
                              amp=amps[k]).reshape(syy.shape)
        cleaned[_sl] -= m

    # --- gate on the tool's own crowding metric ------------------------
    flux1, sky1, crowd1, _s1, _n1, _n2 = aperture_flux(
        cleaned, photrad, tx, ty, insky_px=r_in, outsky_px=r_out,
        skyval=skyval, robust=robust_sky)
    # Gate on the ABSOLUTE annulus contamination, not on `crowding` itself
    # (D16).  crowding = |mean_sky - clipped_sky| * n_ap / |flux|, and
    # successful cleaning REMOVES flux, so the denominator shrinks: removing
    # 13 % of the aperture flux multiplies crowding by 1.15 and removing
    # 50 % doubles it, with the annulus statistic untouched.  Gating on the
    # ratio therefore refuses precisely the cases that worked -- measured:
    # every contrast 0-1 mag cell of the S2 grid was refused this way while
    # its uncleaned bias ran -0.08 to -0.15.  The guard's INTENT (PLAN
    # section 5: "a subtraction that increases the crowding metric ->
    # refuse") is preserved exactly by comparing the contamination in ADU,
    # which is what "removed the wrong thing" actually means.
    # ...and require the worsening to exceed the statistic's own NOISE.
    # For a neighbour inside the aperture -- the case this feature exists
    # for -- cleaning does not touch the sky annulus at all, so contam0 and
    # contam1 differ only by sampling noise: sigma/sqrt(N_ann) * n_ap, which
    # is 2794 ADU here, 2.8 % of a 1e5 ADU target. A strict `>` is then a
    # coin flip that refuses about half the SUCCESSFUL cases for no physical
    # reason. 2 sigma keeps the guard sensitive to a real worsening (a
    # mis-placed subtraction dumps flux into the annulus and moves this
    # statistic by far more) without letting noise veto good work.
    contam0 = abs(crowd0) * abs(flux0)
    contam1 = abs(crowd1) * abs(flux1)
    contam_noise = sky_sigma / max(np.sqrt(max(_n_ann, 1)), 1e-9) * _n_ap
    if contam1 > contam0 + 2.0 * contam_noise:
        return _refuse(
            f"cleaning refused: annulus contamination got WORSE after "
            f"subtraction ({contam0:.1f} -> {contam1:.1f} ADU, beyond the "
            f"{2.0 * contam_noise:.1f} ADU noise on that statistic; "
            f"crowding {crowd0:.4f} -> {crowd1:.4f}); the fit removed the "
            f"wrong thing, so the uncleaned measurement is kept",
            crowding_before=float(crowd0), crowding_after=float(crowd1),
            n_candidates=n_cand, n_saturated=n_sat, epsf_tag=epsf.tag,
            fit_status=int(info["status"]),
            n_fit_pixels=int(info["n_fit_pixels"]),
            n_rejected_pixels=int(info["n_rejected"]),
            neighbours=_neighbour_tuple())

    # --- reporting metrics, on the aperture, per the docstring ---------
    r_i = int(np.ceil(photrad))
    ay, ax = np.mgrid[-r_i:r_i + 1, -r_i:r_i + 1]
    sel = (ay * ay + ax * ax) <= photrad * photrad
    apy = (ay[sel] + int(round(ty))).astype(float)
    apx = (ax[sel] + int(round(tx))).astype(float)
    ok = ((apy >= 0) & (apy < work.shape[0])
          & (apx >= 0) & (apx < work.shape[1]))
    apy, apx = apy[ok], apx[ok]
    sub_flux = 0.0
    for k in subtract_idx:
        nxx, nyy = positions[k]
        sub_flux += float(model.evaluate_at(apy, apx, nxx, nyy,
                                            amp=amps[k]).sum())
        kept[k - 1]["flux_in_aperture"] = float(
            model.evaluate_at(apy, apx, nxx, nyy, amp=amps[k]).sum())

    sub_frac = sub_flux / abs(flux0)
    if sub_frac > max_subtracted_frac:
        return _refuse(
            f"cleaning refused: {100 * sub_frac:.1f}% of the aperture flux "
            f"was neighbour light, above the {100 * max_subtracted_frac:.0f}%"
            f" limit -- the star is a minority of the light in its own "
            f"aperture, so a cleaned measurement would be mostly model",
            crowding_before=float(crowd0), crowding_after=float(crowd1),
            n_candidates=n_cand, n_saturated=n_sat, epsf_tag=epsf.tag,
            subtracted_flux=float(sub_flux), subtracted_frac=float(sub_frac),
            dropped_frac=float(dropped_frac),
            fit_status=int(info["status"]),
            n_fit_pixels=int(info["n_fit_pixels"]),
            n_rejected_pixels=int(info["n_rejected"]),
            exclude_from_field=True)

    # numerator: the fit residual inside the aperture, MINUS the residual
    # that photon/read noise alone explains (D14).  Without that
    # subtraction the metric is meaningless rather than merely
    # conservative: sum|res| over ~31 800 aperture pixels at ~10 ADU is
    # ~3e5 ADU against a subtracted flux of ~1e4, so it reads 190-680 %
    # on subtractions that are demonstrably working.  For zero-mean
    # Gaussian noise E|res| = sigma*sqrt(2/pi) per pixel, so the floor is
    # N_ap * sigma * sqrt(2/pi); a coherent mis-subtraction of amplitude
    # dA instead contributes ~dA in flux, which is what we want to see.
    # It still includes the TARGET's own model error, so it still
    # over-states the leftover neighbour light -- the conservative
    # direction, exactly as designed.
    in_ap = ((fy - ty) ** 2 + (fx - tx) ** 2) <= photrad * photrad
    n_in_ap = int(in_ap.sum())
    resid_abs = float(np.abs(resid[in_ap]).sum()) if n_in_ap else 0.0
    noise_floor = n_in_ap * sky_sigma * np.sqrt(2.0 / np.pi)
    resid_excess = max(resid_abs - noise_floor, 0.0)
    residual_frac = (resid_excess / abs(sub_flux)) if sub_flux != 0.0 else 0.0

    neighbours = tuple(
        Neighbour(x=r["x"], y=r["y"],
                  fit_x=float(positions[k][0]), fit_y=float(positions[k][1]),
                  amp=float(amps[k]),
                  flux_in_aperture=float(r.get("flux_in_aperture", 0.0)),
                  sep_arcsec=r["sep_arcsec"], saturated=r["saturated"],
                  subtracted=k in subtract_idx, note=r.get("note", ""))
        for k, r in enumerate(kept, start=1))

    cap_note = getattr(catalog, "cap_note", "")
    parts = [f"{len(subtract_idx)} neighbour(s) subtracted",
             f"{100 * sub_flux / abs(flux0):.1f}% of the aperture flux",
             f"residual {100 * residual_frac:.1f}% "
             f"(|res| {resid_abs:.0f} - noise floor {noise_floor:.0f} ADU)",
             f"crowding {crowd0:.3f} -> {crowd1:.3f}"]
    if n_sat:
        parts.append(f"{n_sat} saturated neighbour(s) FLAGGED, not "
                     f"subtracted" if not subtract_saturated
                     else f"{n_sat} saturated neighbour(s) subtracted")
    if dropped:
        parts.append(f"{len(dropped)} candidate(s) beyond the "
                     f"{max_neighbours}-neighbour cap left "
                     f"{100 * dropped_frac:.2f}% behind")
    if info["n_rejected"]:
        parts.append(f"{info['n_rejected']} pixel(s) sigma-rejected")

    # D27 direction statement: NOT emitted here. `clean_star` does not
    # know the resulting Strehl, so it cannot tell whether the residual
    # error is the UNDERESTIMATE that holds inside the validated envelope
    # or the OVERESTIMATE that holds above SR 0.30. Embedding the SAFE
    # note unconditionally put a flat "expected bias: UNDERESTIMATE" into
    # `CleanReport.note` while `measure_strehl._bias_note` correctly
    # emitted the OVERESTIMATE warning for the same star -- and the GUI
    # logs both fields back to back, so above the envelope the observer
    # read two adjacent lines claiming OPPOSITE signs. That destroys the
    # one thing D27 exists to provide. The direction now comes solely
    # from `psf_clean_bias`, which is computed where the Strehl is known.
    bias_note = (" Model is THEORETICAL (D26): it carries no static "
                 "speckle or instrument structure, so it subtracts less "
                 "than reality -- which pushes the residual in the same "
                 "direction as the model-free case."
                 if getattr(epsf, "tag", "") == "theoretical" else "")
    return cleaned, CleanReport(
        cleaned=True,
        note="[psf-clean] " + "; ".join(parts) + "." + cap_note + bias_note,
        epsf_tag=epsf.tag, n_candidates=n_cand,
        n_subtracted=len(subtract_idx), n_dropped=len(dropped),
        n_saturated=n_sat, subtracted_flux=float(sub_flux),
        subtracted_frac=float(sub_flux / abs(flux0)),
        dropped_frac=float(dropped_frac),
        residual_frac=float(residual_frac), crowding_before=float(crowd0),
        crowding_after=float(crowd1),
        n_fit_pixels=int(info["n_fit_pixels"]),
        n_rejected_pixels=int(info["n_rejected"]),
        fit_status=int(info["status"]), neighbours=neighbours)
