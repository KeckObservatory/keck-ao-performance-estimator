"""Field-level neighbour subtraction: one simultaneous solution of every
catalogued star in the frame, then subtract everything but the target.

`psf_fit.clean_star` solves each target's neighbourhood on its own: the
target plus at most 16 neighbours, fitted over a footprint clipped to the
target's sky annulus, with every other star in the frame left in the data.
Stars just outside that footprint still shine into it, and each target's
fit is blind to how the same neighbours were solved for the target next
door.  This module removes both limitations and keeps everything else:

  1. group the catalogue into CONNECTED COMPONENTS whose fit discs overlap
     (a component too large for one fit is split, and counted);
  2. sweep the groups brightest-first, fitting each with the UNCHANGED
     `psf_fit.group_fit` on the data with every OTHER star's current model
     subtracted, and repeat until no amplitude moves by more than `tol`
     (block Gauss-Seidel over the frame);
  3. for a target, subtract every solved star except the target -- the
     whole frame (`scope="frame"`) or only the stars `clean_star` would
     have picked (`scope="footprint"`) -- and apply `clean_star`'s own
     refusal gates, in the same order, with the same wording.

The target is never subtracted and the solved amplitudes are never
presented as a Strehl (psf_fit D2): this is still cleaning, followed by the
unchanged aperture measurement in `measure_strehl`.

A frame whose solution does not converge is a REFUSAL for every target on
it, never a silent fallback to the per-target engine.

Qt-free by rule (numpy/scipy only).
"""
import time
from dataclasses import dataclass

import numpy as np

from .image_strehl import aperture_flux
from .nirc2 import (
    NIRC2_BG_INNER_RADIUS_ARCSEC, NIRC2_BG_OUTER_RADIUS_ARCSEC,
    NIRC2_PHOTOMETRY_RADIUS_ARCSEC,
)
from .psf_fit import (
    PSF_FIT_FOOTPRINT_FWHM, PSF_FIT_MAX_NEIGHBOURS,
    PSF_FIT_MAX_SUBTRACTED_FRAC, PSF_FIT_NEIGHBOUR_FLOOR_FRAC,
    PSF_FIT_SIGMA_REJECT, CleanReport, Neighbour, component_footprint,
    group_fit, select_neighbours,
)

__all__ = [
    "FIELD_SOLVE_MAX_GROUP", "FIELD_SOLVE_AMP_FLOOR_SIGMA",
    "SolvedStar", "FieldSolution", "solve_field", "field_clean",
]

# The largest group handed to one `group_fit` call: `clean_star`'s own
# largest fit, the target plus PSF_FIT_MAX_NEIGHBOURS.  A connected
# component bigger than this is split into sub-groups; that is sound
# because every sub-group is fitted with every other star's CURRENT model
# subtracted, the neighbouring sub-groups included (block Gauss-Seidel).
FIELD_SOLVE_MAX_GROUP = PSF_FIT_MAX_NEIGHBOURS + 1

# Convergence is a relative amplitude change, and a star at the noise floor
# flickers by more than 1 % every sweep forever -- a real field would never
# converge because of stars that cannot move the target's number.  Changes
# are therefore measured against at least the amplitude whose core is this
# many sky sigmas, which is `deep_star_catalog`'s own detection floor
# (FS-D10).
FIELD_SOLVE_AMP_FLOOR_SIGMA = 5.0


@dataclass(frozen=True)
class SolvedStar:
    """One catalogued star in a field solution."""
    x_cat: float                # catalogue position: the fit anchor
    y_cat: float
    x: float                    # solved position
    y: float
    amp: float                  # stamp flux in the psi convention
    model_key: tuple            # epsf.at() cache key of the group's anchor
    group: int
    saturated: bool
    converged: bool             # its own last relative change <= tol
    dropped: bool               # removed from the solution (counted)
    note: str = ""


@dataclass(frozen=True)
class FieldSolution:
    """Every catalogued star of one frame, solved together.

    Frozen: `measure_field` shares one solution across all the stars it
    measures, so nothing downstream may mutate it."""
    stars: tuple                # SolvedStar, catalogue order
    converged: bool
    n_sweeps: int
    max_rel_change: tuple       # one float per sweep run
    n_groups: int
    n_split_groups: int
    n_dropped: int
    n_failed_fits: tuple        # group fits with status <= 0, per sweep
    tol: float
    runtime_s: float
    epsf_tag: str
    shape: tuple
    note: str

    @property
    def n_live(self):
        return sum(1 for s in self.stars if not s.dropped)


def _model_key(epsf, x, y):
    bin_px = 1000.0 / float(epsf.plate_scale_mas)
    return (round(float(x) / bin_px), round(float(y) / bin_px))


def _model_for_key(epsf, key):
    """The EpsfModel `epsf.at` cached under `key` (the anchor's 1-arcsec
    bin).  Asking for the bin's own centre lands on the same key, so this
    is a cache hit, never a second recombination."""
    bin_px = 1000.0 / float(epsf.plate_scale_mas)
    return epsf.at(key[0] * bin_px, key[1] * bin_px)


def _stamp(shape, model, x, y, amp):
    from .epsf import _box
    sl, yy, xx = _box(shape, x, y, model.r_stamp_px)
    m = model.evaluate_at(yy.ravel(), xx.ravel(), x, y, amp=amp)
    return sl, m.reshape(yy.shape)


def _groups(xs, ys, peaks, link_px, max_group):
    """Connected components of the overlap graph, each split into
    sub-groups of at most `max_group` when needed.  Returns
    (groups as lists of star indices, number of components split),
    brightest anchor first."""
    n = len(xs)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        d = np.hypot(xs[i + 1:] - xs[i], ys[i + 1:] - ys[i])
        for j in np.flatnonzero(d <= link_px) + i + 1:
            ri, rj = find(i), find(int(j))
            if ri != rj:
                parent[rj] = ri
    comps = {}
    for i in range(n):
        comps.setdefault(find(i), []).append(i)

    out, n_split = [], 0
    for members in comps.values():
        if len(members) <= max_group:
            out.append(sorted(members, key=lambda k: -peaks[k]))
            continue
        n_split += 1
        todo = sorted(members, key=lambda k: -peaks[k])
        while todo:
            seed = todo[0]
            rest = todo[1:]
            rest.sort(key=lambda k: np.hypot(xs[k] - xs[seed], ys[k] - ys[seed]))
            grp = [seed] + rest[:max_group - 1]
            out.append(grp)
            taken = set(grp)
            todo = [k for k in todo if k not in taken]
    out.sort(key=lambda g: -peaks[g[0]])
    return out, n_split


def solve_field(work, params, epsf, catalog=None, *, max_sweeps=5,
                tol=0.01, badmask=None, saturation=None,
                sigma_reject=PSF_FIT_SIGMA_REJECT):
    """Solve every catalogued star of `work` together.  -> FieldSolution.

    `work` is the SIGMA-FILTERED array `measure_strehl` measures on, and
    `catalog` the `deep_star_catalog` of that same array (built here when
    omitted).  Never raises for a data condition: an unusable ePSF or an
    empty catalogue comes back as `converged=False` with the reason, so
    every consumer takes the same refusal path.
    """
    from .epsf import _box, _robust_sky, deep_star_catalog

    t0 = time.time()
    work = np.asarray(work, dtype=float)
    shape = work.shape
    tag = str(getattr(epsf, "tag", ""))

    def _not_built(note):
        return FieldSolution(
            stars=(), converged=False, n_sweeps=0, max_rel_change=(),
            n_groups=0, n_split_groups=0, n_dropped=0, n_failed_fits=(),
            tol=float(tol), runtime_s=time.time() - t0, epsf_tag=tag,
            shape=tuple(shape), note=note)

    if epsf is None or not getattr(epsf, "usable", False):
        return _not_built(
            "field solution not built: no usable ePSF ("
            + (getattr(epsf, "note", "") or "no ePSF supplied") + ")")
    if catalog is None:
        catalog = deep_star_catalog(work, params)
    if len(catalog) == 0:
        return _not_built("field solution not built: the star catalogue "
                          "is empty")
    if saturation is None:
        saturation = float(params.max_counts) * float(params.coadds)

    fwhm = float(epsf.at().fwhm_px)
    r_comp = PSF_FIT_FOOTPRINT_FWHM * fwhm
    _sky, sky_sigma = _robust_sky(work)
    sky_sigma = float(sky_sigma) if sky_sigma and sky_sigma > 0 else 1.0

    n = len(catalog)
    xc = np.array([float(c["x"]) for c in catalog])
    yc = np.array([float(c["y"]) for c in catalog])
    pk = np.array([float(c.get("peak", 0.0)) for c in catalog])

    groups, n_split = _groups(xc, yc, pk, 2.0 * r_comp, FIELD_SOLVE_MAX_GROUP)
    keys, models = [], []
    group_of = np.zeros(n, dtype=int)
    for g, members in enumerate(groups):
        key = _model_key(epsf, xc[members[0]], yc[members[0]])
        keys.append(key)
        models.append(_model_for_key(epsf, key))
        for k in members:
            group_of[k] = g

    # saturation is an absolute detector level: judged on the RAW work
    # array, exactly as clean_star judges its neighbours, and handed to the
    # fit as a mask because the data a group sees has other stars removed
    bad = work >= float(saturation)
    if badmask is not None:
        bad |= np.asarray(badmask, dtype=bool)
    r_core = max(1.0, 0.75 * fwhm)
    saturated = np.zeros(n, dtype=bool)
    for k in range(n):
        sl, _, _ = _box(shape, xc[k], yc[k], r_core)
        sub = work[sl]
        saturated[k] = bool(sub.size and sub.max() >= float(saturation))

    amp = np.empty(n)
    for k in range(n):
        pv = float(models[group_of[k]].peak_value)
        amp[k] = pk[k] / pv if pv > 0 else float(catalog[k].get("flux", 0.0))
    px, py = xc.copy(), yc.copy()
    live = np.ones(n, dtype=bool)
    notes = [""] * n
    last_change = np.full(n, np.inf)

    # S = work - (sum of every live star's current model)
    S = work.copy()

    def _put(k):
        sl, m = _stamp(shape, models[group_of[k]], px[k], py[k], amp[k])
        S[sl] -= m

    def _take(k):
        sl, m = _stamp(shape, models[group_of[k]], px[k], py[k], amp[k])
        S[sl] += m

    for k in range(n):
        if amp[k] > 0 and np.isfinite(amp[k]):
            _put(k)
        else:
            live[k] = False
            notes[k] = "dropped: no positive starting amplitude"

    history, fails = [], []
    converged = False
    for sweep in range(1, int(max_sweeps) + 1):
        worst, n_fail = 0.0, 0
        for g, members in enumerate(groups):
            mem = [k for k in members if live[k]]
            if not mem:
                continue
            model = models[g]
            comps = [(xc[k], yc[k]) for k in mem]
            fy, fx = component_footprint(shape, comps, r_comp)
            if fy.size < 2 * len(mem) + 4:
                for k in mem:
                    _take(k)
                    live[k] = False
                    notes[k] = "dropped: fit footprint too small (array edge)"
                continue
            for k in mem:
                _take(k)
            amps, positions, _bg, _res, info = group_fit(
                S, comps, model, (fy, fx), sky_sigma, background="constant",
                sigma_reject=sigma_reject, badmask=bad)
            if info["status"] <= 0:
                n_fail += 1
                for k in mem:
                    _put(k)
                continue
            floor = (FIELD_SOLVE_AMP_FLOOR_SIGMA * sky_sigma
                     / float(model.peak_value)) if model.peak_value > 0 else 0.0
            for j, k in enumerate(mem):
                a_new = float(amps[j])
                xn, yn = float(positions[j][0]), float(positions[j][1])
                if not (np.isfinite(a_new) and a_new > 0.0
                        and np.isfinite(xn) and np.isfinite(yn)
                        and 0.0 <= xn <= shape[1] - 1
                        and 0.0 <= yn <= shape[0] - 1):
                    live[k] = False
                    notes[k] = (f"dropped at sweep {sweep}: non-positive or "
                                "non-finite amplitude, or off-frame position")
                    continue
                change = abs(a_new - amp[k]) / max(abs(amp[k]), floor, 1e-300)
                last_change[k] = change
                worst = max(worst, change)
                amp[k], px[k], py[k] = a_new, xn, yn
                _put(k)
        history.append(float(worst))
        fails.append(int(n_fail))
        if worst <= tol and n_fail == 0:
            converged = True
            break

    stars = tuple(
        SolvedStar(x_cat=float(xc[k]), y_cat=float(yc[k]), x=float(px[k]),
                   y=float(py[k]), amp=float(amp[k]) if live[k] else 0.0,
                   model_key=keys[group_of[k]], group=int(group_of[k]),
                   saturated=bool(saturated[k]),
                   converged=bool(live[k] and last_change[k] <= tol),
                   dropped=not bool(live[k]), note=notes[k])
        for k in range(n))
    n_drop = int((~live).sum())
    n_sweeps = len(history)
    head = (f"{int(live.sum())} star(s) in {len(groups)} group(s)"
            + (f" ({n_split} oversized component(s) split)" if n_split else "")
            + (f", {n_drop} dropped" if n_drop else ""))
    if converged:
        note = (f"field solution: {head}; converged in {n_sweeps} sweep(s) "
                f"(max relative amplitude change {history[-1]:.4f} <= tol "
                f"{tol:g})")
    else:
        note = (f"field solution did not converge: {head}; max relative "
                f"amplitude change {history[-1]:.4f} at sweep {n_sweeps} "
                f"(tol {tol:g}), {fails[-1]} group fit(s) failed in that "
                "sweep")
    return FieldSolution(
        stars=stars, converged=converged, n_sweeps=n_sweeps,
        max_rel_change=tuple(history), n_groups=len(groups),
        n_split_groups=int(n_split), n_dropped=n_drop,
        n_failed_fits=tuple(fails), tol=float(tol),
        runtime_s=time.time() - t0, epsf_tag=tag, shape=tuple(shape),
        note=note)


def field_clean(work, solution, pos, params, epsf, *, scope="frame",
                catalog=None,
                photometry_radius_arcsec=NIRC2_PHOTOMETRY_RADIUS_ARCSEC,
                bg_inner_arcsec=NIRC2_BG_INNER_RADIUS_ARCSEC,
                bg_outer_arcsec=NIRC2_BG_OUTER_RADIUS_ARCSEC,
                robust_sky=False, sky_override=None,
                subtract_saturated=False,
                floor_frac=PSF_FIT_NEIGHBOUR_FLOOR_FRAC,
                max_subtracted_frac=PSF_FIT_MAX_SUBTRACTED_FRAC):
    """Clean one target against a FieldSolution.  -> (cleaned, CleanReport)

    Same contract as `psf_fit.clean_star`: `work` is the sigma-filtered
    full frame, the return is a full-frame copy (or `work` itself on every
    refusal), and every outcome carries a note.  Gates, in `clean_star`'s
    order: usable ePSF; non-zero target flux; a CONVERGED solution; a
    non-empty subtraction set; annulus contamination not worse (D16);
    subtracted fraction under `max_subtracted_frac` (D19).

    `scope="frame"` subtracts every solved star except the target whose
    stamp reaches the aperture or annulus (FS-D12: farther stars
    contribute exactly nothing to the measurement).
    `scope="footprint"` subtracts exactly the stars `select_neighbours`
    would pick for this target (same floor, no cap), as solved by the
    field solution (FS-D9) -- the per-target choice of WHICH stars, with
    the global answer for HOW MUCH.
    """
    from .epsf import _robust_sky, deep_star_catalog

    if scope not in ("frame", "footprint"):
        raise ValueError(f"field_clean: unknown scope {scope!r} "
                         "(expected 'frame' or 'footprint')")
    work = np.asarray(work, dtype=float)
    ps = float(params.plate_scale_mas)
    tx, ty = float(pos[0]), float(pos[1])
    photrad = photometry_radius_arcsec * 1000.0 / ps
    r_in = bg_inner_arcsec * 1000.0 / ps
    r_out = bg_outer_arcsec * 1000.0 / ps
    n_sol = solution.n_live if solution is not None else 0
    n_sw = solution.n_sweeps if solution is not None else 0

    def _refuse(note, **kw):
        base = dict(
            cleaned=False, note=note,
            epsf_tag=getattr(epsf, "tag", ""), n_candidates=0,
            n_subtracted=0, n_dropped=0, n_saturated=0,
            subtracted_flux=0.0, subtracted_frac=0.0, dropped_frac=0.0,
            residual_frac=0.0, crowding_before=0.0, crowding_after=0.0,
            n_fit_pixels=0, n_rejected_pixels=0, fit_status=0,
            neighbours=(), engine="field", n_solution_stars=n_sol,
            n_sweeps=n_sw)
        base.update(kw)
        return work, CleanReport(**base)

    if epsf is None or not getattr(epsf, "usable", False):
        return _refuse(
            "cleaning skipped: no usable ePSF ("
            + (getattr(epsf, "note", "") or "no ePSF supplied") + ")")

    skyval = float(sky_override) if sky_override is not None else None
    flux0, sky0, crowd0, sky_sigma0, _n_ap, _n_ann = aperture_flux(
        work, photrad, tx, ty, insky_px=r_in, outsky_px=r_out,
        skyval=skyval, robust=robust_sky)
    if not np.isfinite(flux0) or flux0 == 0.0:
        return _refuse("cleaning skipped: target aperture flux is zero or "
                       "non-finite, so there is nothing to clean against")
    if solution is None or not solution.converged:
        return _refuse(
            "cleaning refused: field solution did not converge ("
            + (solution.note if solution is not None
               else "no field solution supplied") + ")",
            crowding_before=float(crowd0), crowding_after=float(crowd0))
    _sky_g, sky_sigma = _robust_sky(work)
    if sky_sigma0 > 0.0:
        sky_sigma = sky_sigma0

    model_t = epsf.at(tx, ty)
    fwhm = float(model_t.fwhm_px)
    live = [k for k, s in enumerate(solution.stars) if not s.dropped]

    def _nearest(x, y, among):
        best, best_d = None, 0.5 * fwhm
        for k in among:
            s = solution.stars[k]
            d = np.hypot(s.x_cat - x, s.y_cat - y)
            if d <= best_d:
                best, best_d = k, d
        return best

    target = _nearest(tx, ty, live)
    others = [k for k in live if k != target]
    unmatched = 0
    if scope == "frame":
        # every star whose stamp can reach the measured disc (aperture +
        # annulus), wings included and with no contamination floor (S2);
        # a star farther than r_out + r_stamp has a model of exactly 0.0
        # over every pixel the measurement reads, so leaving it out changes
        # nothing -- and keeps an isolated star a true no-op (FS-D12)
        reach = r_out + float(epsf.r_stamp_px)
        chosen = [k for k in others
                  if np.hypot(solution.stars[k].x - tx,
                              solution.stars[k].y - ty) <= reach]
    else:
        if catalog is None:
            catalog = deep_star_catalog(work, params)
        kept, _none = select_neighbours(
            catalog, (tx, ty), params, model_t, flux0,
            photometry_radius_arcsec=photometry_radius_arcsec,
            bg_outer_arcsec=bg_outer_arcsec, max_neighbours=10 ** 9,
            floor_frac=floor_frac)
        chosen = []
        for r in kept:
            k = _nearest(r["x"], r["y"], others)
            if k is None:
                unmatched += 1
            elif k not in chosen:
                chosen.append(k)
    n_cand = len(chosen)
    n_sat = sum(1 for k in chosen if solution.stars[k].saturated)
    subtract = [k for k in chosen
                if subtract_saturated or not solution.stars[k].saturated]
    if not subtract:
        why = (f"{n_sat} saturated, FLAGGED not subtracted" if n_sat
               else "the star is effectively isolated")
        return _refuse(
            f"0 components to subtract for this target (scope={scope}; "
            f"{why})",
            crowding_before=float(crowd0), crowding_after=float(crowd0),
            n_candidates=n_cand, n_saturated=n_sat, epsf_tag=epsf.tag)

    cleaned = work.copy()
    for k in subtract:
        s = solution.stars[k]
        sl, m = _stamp(work.shape, _model_for_key(epsf, s.model_key),
                       s.x, s.y, s.amp)
        cleaned[sl] -= m

    # --- annulus contamination gate: clean_star's D16 formula, verbatim
    flux1, sky1, crowd1, _s1, _n1, _n2 = aperture_flux(
        cleaned, photrad, tx, ty, insky_px=r_in, outsky_px=r_out,
        skyval=skyval, robust=robust_sky)
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
            fit_status=1)

    # --- reporting metrics over the target aperture, as in clean_star
    r_i = int(np.ceil(photrad))
    ay, ax = np.mgrid[-r_i:r_i + 1, -r_i:r_i + 1]
    sel = (ay * ay + ax * ax) <= photrad * photrad
    apy = (ay[sel] + int(round(ty))).astype(float)
    apx = (ax[sel] + int(round(tx))).astype(float)
    ok = ((apy >= 0) & (apy < work.shape[0])
          & (apx >= 0) & (apx < work.shape[1]))
    apy, apx = apy[ok], apx[ok]
    sub_flux = 0.0
    in_ap = {}
    for k in subtract:
        s = solution.stars[k]
        f = float(_model_for_key(epsf, s.model_key).evaluate_at(
            apy, apx, s.x, s.y, amp=s.amp).sum())
        in_ap[k] = f
        sub_flux += f
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
            fit_status=1, exclude_from_field=True)

    # residual: D14's definition over the target aperture.  With the target
    # in the solution, its own model is removed too; without it, the
    # target's light counts as residual -- over-stated, the conservative
    # direction D14 asks for, and named in the note.
    iy, ix = apy.astype(int), apx.astype(int)
    res = cleaned[iy, ix] - sky0
    if target is not None:
        st = solution.stars[target]
        res = res - _model_for_key(epsf, st.model_key).evaluate_at(
            apy, apx, st.x, st.y, amp=st.amp)
    resid_abs = float(np.abs(res).sum())
    noise_floor = iy.size * sky_sigma * np.sqrt(2.0 / np.pi)
    residual_frac = (max(resid_abs - noise_floor, 0.0) / abs(sub_flux)
                     if sub_flux != 0.0 else 0.0)

    neighbours = tuple(
        Neighbour(x=solution.stars[k].x_cat, y=solution.stars[k].y_cat,
                  fit_x=solution.stars[k].x, fit_y=solution.stars[k].y,
                  amp=solution.stars[k].amp, flux_in_aperture=in_ap[k],
                  sep_arcsec=float(np.hypot(solution.stars[k].x_cat - tx,
                                            solution.stars[k].y_cat - ty)
                                   * ps / 1000.0),
                  saturated=solution.stars[k].saturated, subtracted=True)
        for k in subtract)

    parts = [f"{len(subtract)} component(s) subtracted (scope={scope})",
             f"{100 * sub_frac:.1f}% of the aperture flux",
             f"residual {100 * residual_frac:.1f}% "
             f"(|res| {resid_abs:.0f} - noise floor {noise_floor:.0f} ADU)",
             f"crowding {crowd0:.3f} -> {crowd1:.3f}",
             f"solution of {solution.n_live} star(s), {solution.n_sweeps} "
             f"sweep(s)"]
    if n_sat:
        parts.append(f"{n_sat} saturated star(s) FLAGGED, not subtracted"
                     if not subtract_saturated
                     else f"{n_sat} saturated star(s) subtracted")
    if target is None:
        parts.append("target not in the catalogue, so its light counts as "
                     "residual")
    if unmatched:
        parts.append(f"{unmatched} footprint pick(s) not in the solution")
    return cleaned, CleanReport(
        cleaned=True,
        note="[psf-clean:field] " + "; ".join(parts) + ".",
        epsf_tag=epsf.tag, n_candidates=n_cand,
        n_subtracted=len(subtract), n_dropped=0, n_saturated=n_sat,
        subtracted_flux=float(sub_flux), subtracted_frac=float(sub_frac),
        dropped_frac=0.0, residual_frac=float(residual_frac),
        crowding_before=float(crowd0), crowding_after=float(crowd1),
        n_fit_pixels=0, n_rejected_pixels=0, fit_status=1,
        neighbours=neighbours, engine="field", n_solution_stars=n_sol,
        n_sweeps=n_sw)
