"""Guide-star auto-ranking: given a loaded catalogue of candidate stars (see
catalogs.stars_field_xy), rank them by the delivered Strehl/FWHM AT THE
SCIENCE TARGET if each, in turn, were used as the guide/TT-tilt reference --
the decision an observer is actually making when picking a guide star, rather
than inspecting candidates one at a time on the field map.

Reuses field_metric_at's per-point physics unchanged (via the
ngs_bright_override / tt_mag_override kwargs added to fieldmap.py for this):
a star's own field position substitutes for the current ngs/tt offset, and
its estimated sensing-band magnitude substitutes for the current ngs_bright/
tt_mag, then the SAME metric is evaluated at the science target (the field
centre, (0, 0) in the plot frame). So "closer star at equal magnitude ranks
higher" and "brighter star at equal offset ranks higher" both fall out of the
existing anisoplanatism/tip-tilt physics with no special-casing here.

Qt-free; the field-map tab's Rank button (FieldMapOverlaysMixin) is the only
caller.
"""
import numpy as np

from .fieldmap import field_metric_at
from .photometry import (
    SENSOR_FAINT_LIMIT, estimate_sensing_mag, optical_extinction_lower_bound,
)
from .vignetting import (
    VIGNETTE_UNUSABLE_FRAC, tss_reachable, vignetting_fraction,
    vignetting_mag_penalty, vignetting_note,
)

# FWHM conventions: smaller is better. Strehl: larger is better.
_LOWER_IS_BETTER = {"fwhm", "fwhm_gaussfit", "fwhm_gaussfit_sky",
                    "fwhm_srtool"}


def rank_guide_stars(args, prep, snap, mode, stars, laser_xy, sensor,
                     metric="strehl", sci_xy=(0.0, 0.0), ngs_delta_var=0.0):
    """Rank candidate guide stars by the delivered `metric` at sci_xy if EACH
    star, in turn, were the guide/TT reference for `mode`.

    args, prep, snap, metric, ngs_delta_var : as for field_metric_at.
    mode   : 'ngs' ranks each star AS the NGS/science reference itself
             (offset from sci_xy drives the exp(-(th/theta0)^(5/3)) aniso
             term). 'single'/'ltao' ranks each star as the TT reference
             (offset from sci_xy drives tt_wfe_nm; laser_xy is FIXED, from
             the caller -- ranking never moves the laser).
    stars  : catalogue stars with field-xy offsets already attached, i.e.
             catalogs.stars_field_xy(...) output -- each a dict with at
             least id/ra/dec/mags/x/y.
    laser_xy : the CURRENT laser position (plot-frame arcsec) -- only used
             for single/ltao (TRICK spot-separation degradation included).
    sensor : 'R' (STRAP) or 'H'/'K' (TRICK) -- selects the sensing-band
             magnitude estimate (photometry.estimate_sensing_mag) and the
             practical faint limit (photometry.SENSOR_FAINT_LIMIT) used to
             flag/exclude a star.
    sci_xy : the science-target field position (plot-frame arcsec) to
             evaluate delivered performance AT. Defaults to the field
             centre (0, 0) -- the normal science pointing.

    Returns a list of dicts (each `dict(star, **ranking fields)`, so the
    original id/ra/dec/mags/x/y survive), sorted best-first:
      rank            : 1-based int, or None if excluded (excluded entries
                        are appended AFTER all ranked ones, sorted brightest
                        magnitude first).
      mag, mag_kind, mag_label : from estimate_sensing_mag (None/None/None
                        if no magnitude is derivable at all).
      mag_effective   : the magnitude the ranking/limit actually USED --
                        equal to `mag` except when the optical-reddening
                        safety (below) adds an extinction lower bound.
      reddening_note  : None, or a short string (e.g. 'J-K=3.0, A_R>=12 mag')
                        when the OPTICAL (R) sensing magnitude was guessed
                        from a reddened star's near-IR photometry -- a
                        'verify, may be optically invisible' flag.
      offset_arcsec   : distance from the star's field position to sci_xy.
      delivered_value : the predicted metric at sci_xy with this star as the
                        guide reference (evaluated at mag_effective), or None
                        if excluded.
      excluded_reason : None, or why the star was excluded (no derivable
                        magnitude / too faint for the sensor / reddened past
                        the optical limit / prediction failed).
    OPTICAL-REDDENING SAFETY (R sensing only): an R magnitude guessed from a
    star's near-IR photometry (estimate_sensing_mag's 'near' fallback) is
    drastically too bright for a reddened star -- dust dims the optical ~7x
    more than K, so it may be invisible to a STRAP-class WFS while bright in
    K (the dusty-Galactic-Centre trap). We bound the optical extinction from
    the IR colour excess (photometry.optical_extinction_lower_bound) and rank
    CONSERVATIVELY on mag+A_R (a lower bound -> a best case), excluding once
    that crosses the sensor limit, and flag it either way. IR sensors (TRICK
    H/K) are unaffected -- their photometry is direct.
    TSS REACHABILITY + VIGNETTING (TT modes only; added 2026-08-07 from
    KAON 913 -- see vignetting.py, and note it is a MODEL, not the measured
    map). Ranking previously had no concept of whether the tip-tilt sensor
    could physically be put on a star: a candidate 120" out ranked on its
    Strehl alone, when the TSS stage cannot travel there at all. Now:
      * a star outside the stage travel at ANY rotator angle is EXCLUDED;
      * one outside the guaranteed circle but inside the box is ranked, with
        a `tss_certainty` of 'depends' and a note, because refusing it would
        hide usable guide stars -- whether it is reachable depends on the
        bench angle, which this engine does not carry;
      * vignetting is charged as MAGNITUDES OF LOST FLUX added to
        mag_effective (dm = -2.5log10(1-v)), so it flows through the sensor's
        own noise row and the faint limit exactly like any other flux
        deficit, instead of being a second hand-tuned budget term;
      * past VIGNETTE_UNUSABLE_FRAC the star is excluded outright.
    NGS mode is deliberately EXEMPT: there the star is the science reference
    sensed on the high-order WFS, not through the TSS that KAON 913 measured.
    The vignetting radius is measured from the FIELD CENTRE (the sensor's
    optical axis), which is not the same point as sci_xy when the science
    target is itself offset -- the stage cares where the star is on the
    bench, not where the science is.

    Cost: one field_metric_at evaluation per star with a derivable
    magnitude -- the same per-point work as one map pixel, trivial for a
    catalogue-sized list."""
    limit = SENSOR_FAINT_LIMIT[sensor]
    lower_better = metric in _LOWER_IS_BETTER
    ranked, excluded = [], []
    for star in stars:
        mag, kind, label = estimate_sensing_mag(star["mags"], sensor)
        offset = float(np.hypot(star["x"] - sci_xy[0], star["y"] - sci_xy[1]))
        # optical-reddening safety: only for R sensing of an IR-derived
        # ('near') magnitude; a_r=0 / note=None otherwise (incl. all TRICK H/K)
        a_r, red_note = (
            optical_extinction_lower_bound(star["mags"])
            if (sensor == "R" and kind == "near") else (0.0, None))
        # TSS geometry is measured from the FIELD CENTRE (the sensor's
        # optical axis), not from the science target -- the stage cares where
        # the star sits on the bench. TT modes only (see the docstring).
        tt_mode = mode != "ngs"
        r_tss = float(np.hypot(star["x"], star["y"]))
        v_frac = float(vignetting_fraction(r_tss)) if tt_mode else 0.0
        v_mag = float(vignetting_mag_penalty(r_tss)) if tt_mode else 0.0
        reach, certainty, reach_why = (tss_reachable(r_tss) if tt_mode
                                       else (True, "always", ""))
        entry = dict(star, mag=mag, mag_kind=kind, mag_label=label,
                    offset_arcsec=offset, rank=None, delivered_value=None,
                    excluded_reason=None, reddening_note=red_note,
                    vignette_frac=v_frac, vignette_mag=v_mag,
                    tss_certainty=certainty,
                    tss_note=(vignetting_note(r_tss) if tt_mode else ""),
                    mag_effective=(mag + a_r + v_mag if mag is not None
                                   else None))
        if mag is None:
            entry["excluded_reason"] = f"no derivable {sensor}-band magnitude"
            excluded.append(entry)
            continue
        if not reach:
            entry["excluded_reason"] = reach_why
            excluded.append(entry)
            continue
        if v_frac >= VIGNETTE_UNUSABLE_FRAC:
            entry["excluded_reason"] = (
                f"~{100 * v_frac:.0f}% vignetted at {r_tss:.0f}\" off the "
                "field centre -- the tip-tilt sensor sees too little of the "
                "pupil (modelled, KAON 913)")
            excluded.append(entry)
            continue
        # conservative: >= mag, raised only by reddening and by vignetting
        mag_eff = mag + a_r + v_mag
        try:
            if mode == "ngs":
                val = field_metric_at(
                    args, prep, snap, mode, metric, (star["x"], star["y"]),
                    (0.0, 0.0), laser_xy, sci_xy, ngs_delta_var=ngs_delta_var,
                    ngs_bright_override=mag_eff)
            else:
                val = field_metric_at(
                    args, prep, snap, mode, metric, (0.0, 0.0),
                    (star["x"], star["y"]), laser_xy, sci_xy,
                    ngs_delta_var=ngs_delta_var, tt_mag_override=mag_eff)
        except Exception:
            val = np.nan
        if not np.isfinite(val):
            entry["excluded_reason"] = "no valid prediction (seeing unusable)"
            excluded.append(entry)
            continue
        entry["delivered_value"] = float(val)
        if mag_eff > limit:
            if red_note:
                entry["excluded_reason"] = (
                    f"IR-red ({red_note}): optical mag past the {sensor} limit "
                    f"{limit:g} -- likely invisible to the optical WFS, verify")
            elif v_mag > 0.005 and mag <= limit:
                # the star itself is inside the limit; VIGNETTING is what
                # pushed it out. Say so, rather than calling it "too faint"
                # -- the observer's fix is a different guide star or a
                # different field rotation, not a brighter magnitude.
                entry["excluded_reason"] = (
                    f"vignetting puts it past the {sensor} limit: mag "
                    f"{mag:.1f} + {v_mag:.2f} lost to ~{100 * v_frac:.0f}% "
                    f"vignetting at {r_tss:.0f}\" = {mag_eff:.1f} > {limit:g}")
            else:
                entry["excluded_reason"] = (
                    f"too faint for {sensor}-band sensing (mag {mag:.1f} > "
                    f"practical limit {limit:g})")
            excluded.append(entry)
            continue
        ranked.append(entry)
    ranked.sort(key=lambda e: e["delivered_value"], reverse=not lower_better)
    for i, e in enumerate(ranked, 1):
        e["rank"] = i
    excluded.sort(key=lambda e: e["mag"] if e["mag"] is not None else np.inf)
    return ranked + excluded
