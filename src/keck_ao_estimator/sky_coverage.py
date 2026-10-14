"""Sky coverage: delivered Strehl versus the fraction of sky that reaches it.

For a telescope, AO mode and guide sensor, at ONE atmospheric snapshot (the
same snapshot dict the field map takes: a night's sample or the Prediction
tab's scenario):

  1. Strehl at the science target is evaluated on a (guide magnitude, guide
     offset) grid with the field map's own per-point model (fieldmap.
     _field_point), so a star at the current guide magnitude and offset reads
     exactly what the field map shows on axis.
       NGS            the star is the wavefront reference:
                      ngs_strehl(R) x exp(-(r/theta0)^(5/3))
       single / LTAO  the star is the tip-tilt reference:
                      lgs_strehl(tt_mag, tt_offset = r)
  2. Monte-Carlo pointings are dropped on a real star field; each pointing
     keeps the best star inside the patrol radius, trading brightness against
     distance, scored on that grid.
  3. The sorted best-Strehl values are the curve: the Strehl delivered over
     at least a fraction f of the sky.

The faint limit only stops a curve (the loop is not asked to close on
fainter stars); the Strehl at every magnitude is the model's own. A curve
ends at the fraction of sky that has any such star in the patrol field.

Star fields (data/sky_field_<tag>.npz, built by
sr_estimator/build_sky_fields.py), two kinds:

  REAL       Gaia DR2 G < 20.5 and the 2MASS point source catalogue in
             0.5 deg cones on four sight lines. Two limits of that supply: G
             stands in for the R magnitude STRAP and the NGS WFS sense in (no
             colour term), and 2MASS is complete only to H ~ 15.1 / K ~ 14.3,
             so TRICK curves are conservative where they lean on fainter
             stars.
  SYNTHETIC  "b60_synth" (synthetic_sky_field): a Poisson field with
             power-law counts, matched to the statistics KAON 1303 sect. 5.11
             publishes for its Besancon-model fields at b = 60 deg. KAON 1303
             / KAON 1318 computed their sky-coverage curves on Besancon
             fields, so this is the supply to read a curve against theirs.
             It is NOT a Besancon model run: it reproduces their published
             availability and a one-slope count law, nothing more. Normalized
             at the faint cut, it carries about 2-3x fewer stars brighter
             than magnitude 15 than the real b = 60 deg field (H <= 14: 164
             vs 339 per deg^2; G <= 14: 42 vs 117), which is what makes the
             KAON curves fall so steeply over the first 10 % of sky.

TILT LAW. law="hardy" swaps the tip-tilt Strehl factor to 1/(1+x^2)
(marechal.tilt_strehl), the form KAON 1303 / KAON 1318 use, for reading a
curve against theirs. The tip-tilt sensor rows were fitted through the
Marechal composition, so the Hardy curve is a convention what-if, not a
second calibrated prediction: at the 2026 campaign seeing it overpredicted
the measured on-axis curves by 0.05-0.12 Strehl where Marechal tracked them
within 0.03 (keck_ao_experiments/skycoverage/KAON1303_TRICK_FINDINGS.md).
NGS has no separate tilt term (one empirical fit), so the law does not apply.

Qt-free.
"""
import os
from functools import lru_cache

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import cKDTree

from .config import resolve_lgs_offset
from .fieldmap import _field_context, _field_point
from .marechal import TILT_LAWS

_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# tag -> (galactic l, b in degrees, label); l is None for a synthetic field
SKY_FIELDS = {
    "b90_l00": (0.0, 90.0, "b = 90° pole (Gaia + 2MASS)"),
    "b60_l90": (90.0, 60.0, "b = 60°, l = 90° (Gaia + 2MASS)"),
    "b60_synth": (None, 60.0, "b = 60° synthetic (KAON 1303 counts)"),
    "b30_l90": (90.0, 30.0, "b = 30°, l = 90° (Gaia + 2MASS)"),
    "b20_l45": (45.0, 20.0, "b = 20°, l = 45° (Gaia + 2MASS)"),
}
DEF_SKY_FIELD = "b60_l90"          # the KAON 1318 Figure 12 sight line
SKY_FIELD_RADIUS_ARCSEC = 1800.0   # cone radius of each bundled field

# Synthetic field, from KAON 1303 sect. 5.11's statistics for its
# Besancon-model fields at b = 60 deg: about 60 % of pointings find a usable
# star in the 102" field, i.e. 1 - exp(-rho pi 51"^2) = 0.60, rho = 0.404
# usable stars per arcmin^2, applied per sensing band; counts a power law in
# magnitude (0.30 dex/mag) up to the usable cut. Same recipe and seed as
# keck_ao_experiments/skycoverage/synthetic_starfield_test_20260809.py. K
# reuses the H stars (H - K = 0): KAON 1303 gives no separate K statistics.
SYNTH_STARS_PER_ARCMIN2 = 0.404
SYNTH_COUNT_SLOPE = 0.30           # dex / mag
SYNTH_MAG_BRIGHT = 8.0
SYNTH_SEED = 20260809

# Guide-star patrol radius: KAON 913's unvignetted, rotator-limited radius
# (vignetting.UNVIGNETTED_RADIUS_ARCSEC) and the field map's field of regard.
# KAON 1318's own sky-coverage runs used 51" (a 102" field).
PATROL_RADIUS_ARCSEC = 60.0
PATROL_RADII_ARCSEC = (60.0, 51.0)

# sensing band -> (catalogue band, faint limit). The limits are the KAON 1318
# Figure 12 cuts (m_R <= 19, m_H <= 17); K takes the H cut.
SENSING = {"R": ("G", 19.0), "H": ("H", 17.0), "K": ("K", 17.0)}

N_POINTINGS = 20000
_MAG_BRIGHT = 6.0     # every model is flat brighter than this; stars clip to it
_MAG_STEP = 0.25
_OFFSET_STEP = 1.0    # arcsec

# KAON 1318 Figure 12 (digitized from the figure): K-band Strehl vs sky
# fraction for KAPA LTAO with ONE tip-tilt star at b = 60 deg, l = 90 deg,
# ZA 30 deg, in a 51" patrol radius, on SYNTHETIC Besancon-model star fields
# with the Hardy tilt law (KAON 1303 sect. 5.11). A target for the
# four-beacon system, drawn for reference only. The matching supply here is
# the "b60_synth" field, not the real catalogues.
KAON1318_FIG12_LABEL = ("KAON 1318 Fig. 12 target (KAPA LTAO, 1 TT star, "
                        "b = 60°, ZA 30°,\nsynthetic Besançon star field, "
                        "51″ patrol, Hardy law)")
KAON1318_FIG12 = {
    "R": ([0.001, 0.005, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60],
          [0.57, 0.48, 0.31, 0.22, 0.17, 0.13, 0.10, 0.07, 0.04]),
    "H": ([0.001, 0.005, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60],
          [0.57, 0.57, 0.41, 0.36, 0.30, 0.26, 0.21, 0.18, 0.15]),
}

_HELD_COLOURS = ("#E24A00", "#12A150", "#7A3FA0", "#D9822B", "#0E7C7B",
                 "#8C564B", "#7F7F7F")

COVERAGE_LEVELS = (0.01, 0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.90)
STREHL_LEVELS = (0.1, 0.2, 0.3, 0.4, 0.5)


def is_synthetic_field(tag):
    """True for a synthetic star field, False for a real catalogue one."""
    return SKY_FIELDS[tag][0] is None


def synthetic_sky_field():
    """Generate the synthetic star field (see SYNTH_* above), in the
    load_sky_field layout. The bundled copy is what load_sky_field reads;
    this is its recipe."""
    rng = np.random.default_rng(SYNTH_SEED)
    area = np.pi * SKY_FIELD_RADIUS_ARCSEC ** 2
    out = {}
    for key, band, mlim in (("ir", "H", SENSING["H"][1]),
                            ("g", "G", SENSING["R"][1])):
        n = rng.poisson(SYNTH_STARS_PER_ARCMIN2 / 3600.0 * area)
        lo = 10.0 ** (SYNTH_COUNT_SLOPE * SYNTH_MAG_BRIGHT)
        hi = 10.0 ** (SYNTH_COUNT_SLOPE * mlim)
        m = np.log10(lo + rng.random(n) * (hi - lo)) / SYNTH_COUNT_SLOPE
        rr = SKY_FIELD_RADIUS_ARCSEC * np.sqrt(rng.random(n))
        th = 2.0 * np.pi * rng.random(n)
        out[f"{key}_x"] = (rr * np.cos(th)).astype("f4")
        out[f"{key}_y"] = (rr * np.sin(th)).astype("f4")
        out[band] = m.astype("f4")
    out["K"] = out["H"].copy()
    return out


@lru_cache(maxsize=None)
def load_sky_field(tag):
    """The bundled star field `tag` (a SKY_FIELDS key): dict of g_x, g_y, G
    (Gaia, or the synthetic R stars) and ir_x, ir_y, H, K (2MASS, or the
    synthetic H / K stars); positions are arcsec from the field centre."""
    if tag not in SKY_FIELDS:
        raise ValueError(f"sky field must be one of {tuple(SKY_FIELDS)}, "
                         f"not {tag!r}")
    with np.load(os.path.join(_DATA_DIR, f"sky_field_{tag}.npz")) as z:
        return {k: z[k] for k in z.files}


@lru_cache(maxsize=16)
def _pairs(tag, band, patrol, n_point, seed):
    """Every (pointing, star) pair closer than the patrol radius, for
    pointings uniform over the field (kept a patrol radius inside its edge).
    Depends only on the star field, so it is cached across condition and
    model changes. Returns (pointing index, star magnitude, separation)."""
    cat = load_sky_field(tag)
    cat_band, mlim = SENSING[band]
    key = "g" if cat_band == "G" else "ir"
    m = cat[cat_band].astype(float)
    ok = np.isfinite(m) & (m <= mlim)
    stars = np.c_[cat[f"{key}_x"][ok], cat[f"{key}_y"][ok]].astype(float)
    m = m[ok]
    rng = np.random.default_rng(seed)
    rmax = SKY_FIELD_RADIUS_ARCSEC - patrol - 5.0
    rr = rmax * np.sqrt(rng.random(n_point))
    th = 2.0 * np.pi * rng.random(n_point)
    pts = np.c_[rr * np.cos(th), rr * np.sin(th)]
    if not len(stars):
        return np.zeros(0, int), np.zeros(0), np.zeros(0)
    hit = cKDTree(pts).sparse_distance_matrix(
        cKDTree(stars), patrol, output_type="ndarray")
    return hit["i"], m[hit["j"]], hit["v"]


def strehl_surface(args, prep, snap, mode, sensor="R", tilt_law="marechal",
                   patrol=PATROL_RADIUS_ARCSEC, ngs_delta_var=0.0):
    """Strehl at the science target on a (guide magnitude, guide offset)
    grid. Returns (mags, offsets, Z) with Z[i, j] at (mags[i], offsets[j]).

    args, prep, snap, ngs_delta_var : as for fieldmap.field_metric_at. args
        carries the telescope and every model setting; the tip-tilt sensor
        comes from `sensor` here, not from args.
    mode   : 'ngs', 'single' or 'ltao'.
    sensor : sensing band of the guide star: 'R' (NGS WFS, or STRAP) or
             'H' / 'K' (TRICK). NGS is always 'R'.
    tilt_law : 'marechal' or 'hardy' (module docstring); ignored for NGS.

    The laser sits at the LGS offset in force (config.resolve_lgs_offset). A
    TRICK star's spot degradation follows its separation from the laser,
    taken as the RMS over the star's position angle, sqrt(r^2 + L^2).
    """
    if tilt_law not in TILT_LAWS:
        raise ValueError(f"tilt law must be one of {TILT_LAWS}, "
                         f"not {tilt_law!r}")
    if mode == "ngs":
        sensor = "R"
    mlim = SENSING[sensor][1]
    mags = np.arange(_MAG_BRIGHT, mlim + 1e-9, _MAG_STEP)
    offs = np.arange(0.0, patrol + 1e-9, _OFFSET_STEP)
    lgs_off = resolve_lgs_offset(args)
    ctx = _field_context(args, prep, snap, mode, "strehl", (0.0, 0.0),
                         (0.0, 0.0), (lgs_off, 0.0), ngs_delta_var)
    if mode != "ngs":
        ctx["tt_sensor"] = "trick" if sensor in ("H", "K") else "strap"
        ctx["bkw"] = dict(ctx["bkw"], tt_sensor=ctx["tt_sensor"],
                          tilt_law=tilt_law)
    Z = np.empty((len(mags), len(offs)))
    for j, r in enumerate(offs):
        ctx["ngs_xy"] = ctx["tt_xy"] = (float(r), 0.0)
        if mode != "ngs":
            ctx["bkw"]["tt_spot_theta"] = float(np.hypot(r, lgs_off))
        for i, m in enumerate(mags):
            ctx["ngs_bright"] = ctx["tt_mag"] = float(m)
            Z[i, j] = _field_point(ctx, 0.0, 0.0)
    return mags, offs, Z


def sky_coverage_curve(args, prep, snap, mode, sensor="R", tilt_law="marechal",
                 field=DEF_SKY_FIELD, patrol=PATROL_RADIUS_ARCSEC,
                 ngs_delta_var=0.0, n_point=N_POINTINGS, seed=11):
    """Strehl versus sky fraction for one configuration (module docstring).

    Arguments as for strehl_surface, plus `field` (a SKY_FIELDS key).
    Returns a dict:
      frac, sr   : the curve -- sr[k] is the Strehl delivered over at least
                   a fraction frac[k] of the sky (sr descending). It ends at
                   `reach`.
      reach      : fraction of pointings with any star inside the faint
                   limit and the patrol radius.
      sr_at      : {f: Strehl} at COVERAGE_LEVELS; NaN beyond `reach`.
      frac_at    : {Strehl: sky fraction reaching at least it} at
                   STREHL_LEVELS.
      mode, sensor, tilt_law, field, synthetic, patrol, mag_limit, lam_nm :
                   what was run (tilt_law reads 'marechal' for NGS, whatever
                   was asked; synthetic is is_synthetic_field(field)).
    """
    if mode == "ngs":
        sensor, tilt_law = "R", "marechal"
    mags, offs, Z = strehl_surface(args, prep, snap, mode, sensor, tilt_law,
                                   patrol, ngs_delta_var)
    i, m, d = _pairs(field, sensor, float(patrol), int(n_point), int(seed))
    best = np.full(n_point, -1.0)          # -1: no star in reach
    if len(i):
        interp = RegularGridInterpolator((mags, offs), Z)
        s = interp(np.c_[np.clip(m, mags[0], mags[-1]),
                         np.clip(d, offs[0], offs[-1])])
        np.maximum.at(best, i, s)
    sr = np.sort(best[best >= 0.0])[::-1]
    frac = np.arange(1, len(sr) + 1) / n_point
    reach = len(sr) / n_point
    return dict(frac=frac, sr=sr, reach=reach,
                sr_at={f: (float(np.interp(f, frac, sr)) if f <= reach
                           else float("nan")) for f in COVERAGE_LEVELS},
                frac_at={t: float(np.sum(sr >= t)) / n_point
                         for t in STREHL_LEVELS},
                mode=mode, sensor=sensor, tilt_law=tilt_law, field=field,
                synthetic=is_synthetic_field(field),
                patrol=float(patrol), mag_limit=SENSING[sensor][1],
                lam_nm=float(prep.lam_nm))


def draw_sky_coverage(fig, result, title="", label=None, held=(),
                      requirement=None, log_x=False, box_title=None):
    """Draw a sky_coverage_curve() result on a matplotlib Figure.

    held        : earlier curves to keep for comparison, as (label, frac, sr).
                  One identical to `result` is not drawn again, so the same
                  curve never takes two legend entries.
    requirement : None, or (label, frac, sr) drawn dashed (KAON1318_FIG12).
    log_x       : logarithmic sky-fraction axis from 0.1 % (NGS curves live
                  in the first few percent).
    box_title   : heading for the readout box, naming the curve it reads.
    The box lists, for `result`, the Strehl at COVERAGE_LEVELS and the sky
    fraction reaching each of STREHL_LEVELS; its border is the curve's
    colour.
    """
    fig.clear()
    ax = fig.add_subplot(111)
    ymax = float(result["sr"][0]) if len(result["sr"]) else 0.0
    for k, (lab, f, s) in enumerate(held):
        if np.array_equal(s, result["sr"]) and np.array_equal(
                f, result["frac"]):
            continue                    # it is the current curve
        ax.plot(100.0 * np.asarray(f), s, lw=1.4, label=lab,
                color=_HELD_COLOURS[k % len(_HELD_COLOURS)])
        ymax = max(ymax, float(np.max(s)))
    if requirement is not None:
        lab, f, s = requirement
        ax.plot(100.0 * np.asarray(f), s, "--", color="0.35", lw=1.8,
                label=lab)
        ymax = max(ymax, float(np.max(s)))
    ax.plot(100.0 * result["frac"], result["sr"], color="#1F6FE0", lw=2.6,
            label=label or "current selection", zorder=5)
    for f, s in result["sr_at"].items():
        ax.plot(100.0 * f, s, "o", color="#1F6FE0", ms=4.5, zorder=6)
    if len(result["sr"]) and result["reach"] < 1.0:
        # the curve stops here: no star in reach over the rest of the sky
        ax.plot(100.0 * result["reach"], result["sr"][-1], "|",
                color="#1F6FE0", ms=14, mew=2.2, zorder=6)
    if log_x:
        ax.set_xscale("log")
        ax.set_xlim(0.1, 100)
        ax.set_xticks([0.1, 1, 10, 100], labels=["0.1", "1", "10", "100"])
    else:
        ax.set_xlim(0, 100)
    ax.set_ylim(0, max(0.1, 1.08 * ymax))
    ax.set_xlabel("sky fraction (%)")
    ax.set_ylabel("Strehl delivered over at least that fraction of sky")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8.5, loc="upper right", framealpha=0.95)
    if title:
        ax.set_title(title, fontsize=10)
    rows = "\n".join(
        f"{100 * f:3.0f} %    " + (f"{s:.3f}" if np.isfinite(s) else "  —  ")
        for f, s in result["sr_at"].items())
    inv = "\n".join(f"≥ {t:.1f}    {100 * f:5.1f} %"
                    for t, f in result["frac_at"].items())
    head = f"{box_title}\n\n" if box_title else ""
    ax.text(0.985, 0.76,
            f"{head}sky      Strehl\n{rows}\n\nStrehl      sky\n{inv}\n\n"
            f"star in reach {100 * result['reach']:3.0f} %",
            transform=ax.transAxes, ha="right", va="top", fontsize=8.5,
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#1F6FE0",
                      lw=1.4, alpha=0.92))
    return ax
