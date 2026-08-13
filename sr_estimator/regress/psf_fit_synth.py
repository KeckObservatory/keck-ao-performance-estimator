#!/usr/bin/env python3
"""Synthetic-frame builders for the psf_fit validation battery (S1-S4).

Qt-free, engine-adjacent regress module -- WP-1 of the psf_fit development
plan (repo history).
Every builder returns raw (unreduced, ADU) frames plus a truth dict that
validates against the schema in STATUS.md's WP-1 handoff, so the harness
(psf_fit_model.py) and Opus's O2/O3 engine work can both consume it without
re-deriving anything here.

Physics (STATUS.md WP-1 handoff): each star is rendered as
`flux * P`, `P = sr*DL + (1-sr)*H`, DL = the real NIRC2 diffraction-limited
PSF (`nirc2_dl_psf`), H = a Moffat seeing disk (FWHM 0.55", beta =
MOFFAT_BETA_KOLM = 4.765 by default -- reusing the repo's own SR*DL +
(1-SR)*seeing decomposition rather than inventing a new one). Both are
normalized to unit sum on the same pixel grid, so a fitted amplitude is a
flux in ADU. Frame construction order (RULES section 4, the 2026-07-23
flat-consistency lesson) is mandatory and exact:

    scene  = sum_stars flux_i * P(x_i, y_i)        ADU, sky-free
    raw    = (scene + sky) * flat                  flat AFTER summing
    raw    = Poisson(raw * gain) / gain             photon noise
    raw   += Normal(0, read_noise)                  read noise, post-flat
    raw    = minimum(raw, saturation_adu)           hard clip = saturation

Truth SR is CALIBRATED, never assumed: `sr` is a construction parameter,
not what `measure_strehl` returns on the mixture (peak/flux of a mixture is
not the mixture of peak/flux). Every builder call renders one isolated,
noiseless star with the same P and measures it with the EXISTING
`measure_strehl`, emitting `sr_truth_isolated` -- that number, not `sr`, is
what S2/S3 bias is defined against.

FAKE-1 rule: truth values are unmistakably synthetic (fluxes are
1.0e5 * 2^-k, seed=20260729 everywhere unless a case says otherwise) so a
synthetic result can never be confused with a real measurement.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(os.path.dirname(ROOT), "src"))
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

import keck_ao_estimator as engine
from keck_ao_estimator.nirc2 import NIRC2_SATURATION_ADU

__all__ = [
    "SEED", "S2_SEPARATIONS_ARCSEC", "S2_CONTRASTS_MAG", "S2_SRS",
    "synth_params", "synth_frame", "build_s1", "build_s2",
    "build_s2_donor_frame", "build_s3", "build_s4",
    "build_s5_sparse", "build_s5_moderate", "build_s5_extreme",
]

SEED = 20260729

S2_SEPARATIONS_ARCSEC = (0.10, 0.15, 0.20, 0.30, 0.45, 0.60, 0.80, 1.00, 1.20)
S2_CONTRASTS_MAG = (0, 1, 2, 3, 4, 5)
S2_SRS = (0.15, 0.30, 0.60)

# GC 20260728 luminosity function (measured off-repo; aggregate numbers
# baked in below), the S3 spec's source of every number.
S3_N_STARS = 259
S3_LF_SLOPE_DLOGN_DMAG = 0.26
S3_DMAG_MAX = 5.6
S3_BRIGHTEST_FLUX_ADU = 3.0e5
S3_MEDIAN_NN_RANGE_ARCSEC = (0.25, 0.35)
S3_N_TARGETS = 30


# ------------------------------------------------------------- parameters

def synth_params(camname="narrow", effwave_um=2.1245, coadds=1, detgain=4.0,
                  pmsname="largehex", pmrangl_deg=0.0):
    """A synthetic `Nirc2FrameParams` -- direct construction (not via a
    fabricated FITS header) so `pmrangl_deg` is a plain, controllable input
    rather than something reverse-engineered through the ROTPPOSN/EL
    formula in `nirc2_frame_params`."""
    return engine.Nirc2FrameParams(
        camname=camname,
        pmsname=pmsname,
        effwave_um=effwave_um,
        pmrangl_deg=pmrangl_deg,
        coadds=coadds,
        max_counts=NIRC2_SATURATION_ADU / detgain,
        daytime=False,
        plate_scale_mas=engine.NIRC2_PLATE_SCALE_MAS[camname],
        utc=None,
        object_name="psf_fit_synth",
    )


def _derive_params(params, **overrides):
    """A new `Nirc2FrameParams` sharing `params`' values except for the
    given overrides (e.g. camname="wide" for S4c) -- recovers `detgain`
    from `max_counts` since the dataclass doesn't carry it directly."""
    detgain = NIRC2_SATURATION_ADU / params.max_counts
    kw = dict(camname=params.camname, pmsname=params.pmsname,
              effwave_um=params.effwave_um, coadds=params.coadds,
              detgain=detgain, pmrangl_deg=params.pmrangl_deg)
    kw.update(overrides)
    return synth_params(**kw)


def _params_dict(params):
    detgain = NIRC2_SATURATION_ADU / params.max_counts
    return {
        "camname": params.camname,
        "pmsname": params.pmsname,
        "effwave_um": float(params.effwave_um),
        "pmrangl_deg": float(params.pmrangl_deg),
        "coadds": int(params.coadds),
        "detgain": float(detgain),
        "plate_scale_mas": float(params.plate_scale_mas),
        "max_counts": float(params.max_counts),
    }


# -------------------------------------------------------------- PSF model

def _render_psf_stamp(params, halo_fwhm_arcsec, halo_beta, npix):
    """(DL, Moffat), each unit-sum over an npix x npix grid on the same
    pixel-corner centring convention `nirc2_dl_psf` uses (centre at
    npix/2 - 0.5), so mixing and later shifting them together is exact."""
    dl = engine.nirc2_dl_psf(params.camname, params.pmsname,
                             params.effwave_um, params.pmrangl_deg,
                             npix=npix, daytime=params.daytime)
    ctr = npix / 2.0 - 0.5
    yy, xx = np.mgrid[0:npix, 0:npix]
    r_px = np.hypot(xx - ctr, yy - ctr)
    fwhm_px = halo_fwhm_arcsec * 1000.0 / params.plate_scale_mas
    alpha_px = fwhm_px / (2.0 * np.sqrt(2.0 ** (1.0 / halo_beta) - 1.0))
    moffat = (1.0 + (r_px / alpha_px) ** 2) ** (-halo_beta)
    moffat /= moffat.sum()
    return dl, moffat


def _place_star(scene, x, y, flux, npix, prefiltered):
    """Add `flux * P` to `scene`, P shifted to the sub-pixel position
    (x, y) by cubic-spline interpolation (`prefiltered` is P run through
    `scipy.ndimage.spline_filter` once by the caller -- shared across every
    star in a frame, since re-prefiltering the same stamp per star would
    dominate S3's 259-star runtime). Returns the placed peak ADU (for the
    truth record), 0.0 if the star fell entirely off-array."""
    ctr = npix / 2.0 - 0.5
    ix = int(np.floor(x - ctr))
    iy = int(np.floor(y - ctr))
    frac_x = (x - ctr) - ix
    frac_y = (y - ctr) - iy
    shifted = ndimage.shift(prefiltered, (frac_y, frac_x), order=3,
                            mode="constant", cval=0.0, prefilter=False)
    sy0, sy1 = max(iy, 0), min(iy + npix, scene.shape[0])
    sx0, sx1 = max(ix, 0), min(ix + npix, scene.shape[1])
    if sy0 >= sy1 or sx0 >= sx1:
        return 0.0
    py0, py1 = sy0 - iy, sy1 - iy
    px0, px1 = sx0 - ix, sx1 - ix
    contribution = flux * shifted[py0:py1, px0:px1]
    scene[sy0:sy1, sx0:sx1] += contribution
    return float(contribution.max())


def _donor_flux_for_window(params, sr, *, halo_beta=None, ref_flux=1.0e5,
                           sky=100.0, read_noise=10.0, gain=4.0,
                           npix=512, shape=(1024, 1024)):
    """WP-1d: the donor flux that lands a rendered star's PEAK at the
    geometric-mean midpoint of the engine's donor acceptance window --
    `[EPSF_DONOR_MIN_SNR * sky_sigma, EPSF_PEAK_CEILING_FRAC * max_counts *
    coadds]` -- for THIS (camera, sr, halo_beta). CP2 assessment item 3: a
    bare flux=1.0e5 donor's peak falls outside this window at sr=0.15
    (too faint -- more flux sits in the broad seeing halo, so peak/flux is
    lower) and on the wide camera (too bright -- a wide pixel covers ~16x
    the sky area of a narrow one, so the same total flux gives ~16x the
    peak). The window's WIDTH is a physics statement (CP2 item 3) and is
    never touched here -- only which flux lands where inside it.

    `sky_sigma` is the exact analytic noise sigma for the given
    sky/read_noise/gain (Poisson-on-sky plus Gaussian read noise is a
    fully known, exactly-specified model here -- not an approximation).
    The PEAK, by contrast, genuinely cannot be predicted this way: it
    depends on the SR*DL + (1-SR)*seeing mixture at whatever sub-pixel
    phase a star lands on, so it is MEASURED from an actual rendered
    isolated star at `ref_flux` (per the WP-1d instruction) and scaled
    linearly -- peak is exactly proportional to flux for a fixed PSF
    shape and phase. The measurement phase (+0.37, +0.61 px off-centre)
    is arbitrary but fixed, only meant to get the overall scale right:
    real donors on the built frame each land at their own jittered phase
    regardless, and their true `peak_adu` is recorded per-star by
    `synth_frame` from the ACTUAL placement, not from this estimate.

    Returns (scaled_flux, floor_adu, ceiling_adu).
    """
    if halo_beta is None:
        halo_beta = engine.MOFFAT_BETA_KOLM
    dl, moffat = _render_psf_stamp(params, 0.55, halo_beta, npix)
    pref = ndimage.spline_filter(sr * dl + (1.0 - sr) * moffat, order=3)
    scene = np.zeros(shape, dtype=float)
    cy, cx = shape[0] / 2.0 + 0.37, shape[1] / 2.0 + 0.61
    measured_peak_at_ref = _place_star(scene, cx, cy, ref_flux, npix, pref)

    sky_sigma = float(np.sqrt(sky / gain + read_noise ** 2))
    floor = engine.EPSF_DONOR_MIN_SNR * sky_sigma
    ceiling = engine.EPSF_PEAK_CEILING_FRAC * params.max_counts * params.coadds
    target_peak = float(np.sqrt(floor * ceiling))

    scaled_flux = ref_flux * (target_peak / measured_peak_at_ref)
    return scaled_flux, floor, ceiling


def _isolated_truth_sr(params, sr, halo_fwhm_arcsec, halo_beta,
                       flux=1.0e5, npix=512, shape=(1024, 1024)):
    """Measure ONE isolated, noiseless star built from the same P -- this
    IS the definition of sr_truth_isolated (STATUS.md WP-1: truth SR is
    calibrated, never assumed, since peak/flux of the SR*DL + (1-SR)*seeing
    mixture is not the mixture of peak/flux)."""
    dl, moffat = _render_psf_stamp(params, halo_fwhm_arcsec, halo_beta, npix)
    p = sr * dl + (1.0 - sr) * moffat
    pref = ndimage.spline_filter(p, order=3)
    scene = np.zeros(shape, dtype=float)
    cy, cx = shape[0] / 2.0, shape[1] / 2.0
    _place_star(scene, cx, cy, flux, npix, pref)
    r = engine.measure_strehl(scene, params=params)
    if not r.ok:
        raise RuntimeError(f"isolated-truth measurement failed: {r.error!r}")
    return float(r.strehl)


# --------------------------------------------------------- truth-JSON schema

_REQUIRED_TOP = {
    "case": str, "seed": int, "params": dict, "sky_adu": float,
    "read_noise_adu": float, "gain_e_per_adu": float, "saturation_adu": float,
    "flat_applied": bool, "flat_source": str, "sr_construction": float,
    "sr_truth_isolated": float, "halo": dict, "stars": list, "pairs": list,
}
_REQUIRED_PARAMS = {
    "camname": str, "pmsname": str, "effwave_um": float, "pmrangl_deg": float,
    "coadds": int, "detgain": float, "plate_scale_mas": float,
    "max_counts": float,
}
_REQUIRED_HALO = {"model": str, "fwhm_arcsec": float, "beta": float}
_REQUIRED_STAR = {
    "id": int, "x": float, "y": float, "flux_adu": float, "peak_adu": float,
    "role": str,
}
_REQUIRED_PAIR = {
    "target_id": int, "neighbour_id": int, "sep_arcsec": float,
    "dmag": float, "pa_deg": float,
}


def _check_keys(d, required, where):
    for key, typ in required.items():
        if key not in d:
            raise ValueError(f"{where}: missing key {key!r}")
        val = d[key]
        if typ is float:
            ok = isinstance(val, (int, float)) and not isinstance(val, bool)
        elif typ is int:
            ok = isinstance(val, int) and not isinstance(val, bool)
        else:
            ok = isinstance(val, typ)
        if not ok:
            raise TypeError(
                f"{where}.{key}: expected {typ.__name__}, got {type(val).__name__}")


def _validate_truth(truth):
    """Validate a truth dict against the WP-1 schema (STATUS.md). Checks
    the REQUIRED keys/types are present; extra builder-specific keys (e.g.
    sr_contrast_group, median_nn_sep_arcsec, expected_epsf_tag) are allowed
    and not part of the contract."""
    _check_keys(truth, _REQUIRED_TOP, "truth")
    _check_keys(truth["params"], _REQUIRED_PARAMS, "truth.params")
    _check_keys(truth["halo"], _REQUIRED_HALO, "truth.halo")
    for i, star in enumerate(truth["stars"]):
        _check_keys(star, _REQUIRED_STAR, f"truth.stars[{i}]")
    for i, pair in enumerate(truth["pairs"]):
        _check_keys(pair, _REQUIRED_PAIR, f"truth.pairs[{i}]")
    return True


# ------------------------------------------------------------- frame builder

def synth_frame(stars, params, *, sr=0.30, shape=(1024, 1024), sky=100.0,
                read_noise=10.0, gain=4.0, halo_fwhm_arcsec=0.55,
                halo_beta=None, flat=None, saturation_adu=None,
                seed=SEED, npix=512, case="S?"):
    """Build one raw (unreduced) synthetic NIRC2 frame plus its truth dict.

    `stars`: [(x, y, flux_adu), ...] or [(x, y, flux_adu, role), ...] --
    role defaults to "target" for stars[0] and "neighbour" for the rest.

    `flat`: None (default) loads the packaged superflat (RULES section 4:
    every synthetic frame must be flat-consistent); an ndarray uses it
    directly; False skips the flat step entirely (raw = scene + sky, no
    multiply) -- the S4a no-flat variant.

    `halo_beta`: None (default) uses MOFFAT_BETA_KOLM (4.765), the repo's
    own Kolmogorov-seeing Moffat index -- the standard SR*DL + (1-SR)*seeing
    decomposition. DEVIATION FROM THE LITERAL WP-1 API BLOCK, flagged in
    the WP-1 report: that block's `halo_beta=2.5` default contradicts its
    own physics section ("H = a Moffat ... beta = MOFFAT_BETA_KOLM = 4.765")
    and its own truth-JSON example (`"beta": 4.765`); 2.5 is used ONLY for
    the explicit S2 broad-wing sensitivity slice, exactly as the physics
    section specifies.

    `read_noise`: default 10.0 ADU, NOT the API block's 60.0 -- a SECOND
    flagged deviation. 60 ADU (~240 e- at gain=4) is unrealistically high
    for NIRC2 and, empirically, makes even a genuinely isolated S1 star
    read CROWDED purely from sky-annulus sampling noise (mean vs
    sigma-clipped-median of ~16500 annulus pixels differing by chance):
    measured crowding 0.05-0.15 across random seeds at 60 ADU, comfortably
    under 0.03 at 10 ADU, with the mean SR bias unaffected either way
    (dSR ~ +0.004 at both noise levels; only the scatter shrinks). Since
    the S1 case exists specifically to be the uncontaminated no-op
    baseline, a default that spuriously flags it CROWDED defeats its own
    purpose.
    """
    if halo_beta is None:
        halo_beta = engine.MOFFAT_BETA_KOLM

    apply_flat = flat is not False
    if flat is None:
        flat_arr = engine.load_nirc2_calibration()[0]
        flat_source = "load_nirc2_calibration()[0]"
    elif flat is False:
        flat_arr = np.ones(shape, dtype=float)
        flat_source = "none (S4a no-flat variant)"
    else:
        flat_arr = np.asarray(flat, dtype=float)
        flat_source = "caller-supplied"
    if flat_arr.shape != shape:
        raise ValueError(
            f"flat shape {flat_arr.shape} != frame shape {shape}")

    if saturation_adu is None:
        saturation_adu = params.max_counts * params.coadds

    dl, moffat = _render_psf_stamp(params, halo_fwhm_arcsec, halo_beta, npix)
    p = sr * dl + (1.0 - sr) * moffat
    pref = ndimage.spline_filter(p, order=3)

    scene = np.zeros(shape, dtype=float)
    star_records = []
    for i, s in enumerate(stars):
        x, y, flux = float(s[0]), float(s[1]), float(s[2])
        role = s[3] if len(s) > 3 else ("target" if i == 0 else "neighbour")
        peak = _place_star(scene, x, y, flux, npix, pref)
        star_records.append({
            "id": i, "x": x, "y": y, "flux_adu": flux,
            "peak_adu": peak, "role": role,
        })

    rng = np.random.default_rng(seed)
    raw = (scene + sky) * flat_arr
    raw = rng.poisson(np.clip(raw * gain, 0.0, None)).astype(float) / gain
    raw = raw + rng.normal(0.0, read_noise, shape)
    raw = np.minimum(raw, saturation_adu)

    sr_truth_isolated = _isolated_truth_sr(params, sr, halo_fwhm_arcsec,
                                           halo_beta, npix=npix, shape=shape)

    truth = {
        "case": case,
        "seed": int(seed),
        "params": _params_dict(params),
        "sky_adu": float(sky),
        "read_noise_adu": float(read_noise),
        "gain_e_per_adu": float(gain),
        "saturation_adu": float(saturation_adu),
        "flat_applied": bool(apply_flat),
        "flat_source": flat_source,
        "sr_construction": float(sr),
        "sr_truth_isolated": sr_truth_isolated,
        "halo": {"model": "moffat", "fwhm_arcsec": float(halo_fwhm_arcsec),
                 "beta": float(halo_beta)},
        "stars": star_records,
        "pairs": [],
    }
    _validate_truth(truth)
    return raw, truth


# -------------------------------------------------------------------- S1

def build_s1(params, **kw):
    """No-op guard: a genuinely isolated TARGET, plus distant (>=3")
    donor stars elsewhere on the frame so a real ePSF can be built --
    WP-1c (STATUS.md). O3 found the original bare-single-star S1 frame
    has no donors at all, so `build_epsf` correctly returns
    `uncalibrated` and cleaning is skipped through the "no usable ePSF"
    branch -- the no-op guard never reached the path it exists to
    protect (an isolated star measured with a WORKING ePSF). The target
    keeps its original position/flux; 8 donors sit on the default
    jittered 3x3 lattice, minimum spacing ~342 px = 3.4" on narrow --
    clear of both the >=3" requirement and of the target, EXCEPT the
    lattice's own centre point, which lands almost exactly on the target
    (both near detector centre) and must be dropped rather than blindly
    taking the lattice's first 8 points. The target has no catalogued
    neighbour within `bg_outer` (1.4") once that point is excluded, so
    its own cleaning still finds zero candidates -- that remains the
    no-op being guarded. This is the CI-wired case -- must stay fast."""
    target = (512.37, 511.61, 1.0e5, "target")
    tx, ty = target[0], target[1]
    lattice9 = _lattice_positions(9)
    donor_pts = sorted(lattice9,
                       key=lambda p: (p[0] - tx) ** 2 + (p[1] - ty) ** 2)[1:]
    donors = [(x, y, 1.0e5, "donor") for x, y in donor_pts]
    stars = [target] + donors
    return [synth_frame(stars, params, sr=kw.pop("sr", 0.30), case="S1",
                        seed=kw.pop("seed", SEED), **kw)]


# -------------------------------------------------------------------- S2

def _lattice_positions(n, lo=170.0, hi=854.0):
    """A roughly-square grid of >= n points spanning [lo, hi]^2, first n
    taken row-major, each nudged by a deterministic low-discrepancy
    sub-pixel offset (golden-ratio increments in x and y).

    Without the jitter every point sits at an exact integer pixel, i.e.
    the SAME (zero) sub-pixel phase -- and D11 (STATUS.md) is explicit
    that donor phase diversity, not donor count, is what an empirical PSF
    is built from. A donor ladder built on the bare linspace grid measured
    exactly EPSF_DEFAULT_OVERSAMPLE^-2 = 25% phase coverage regardless of
    how many donors passed the isolation cut (every one of them landed in
    the same phase class), tripping the engine's phase-coverage floor and
    downgrading strict -> loose. The jitter is <1 px, far below any
    isolation threshold used here (>= 0.25"), so it does not affect
    separation-based cuts. For n=9 the un-jittered base grid is exactly
    the {170, 512, 854} 3x3 grid named in STATUS.md's S2 spec."""
    side = int(np.ceil(np.sqrt(n)))
    coords = np.linspace(lo, hi, side) if side > 1 else np.array([(lo + hi) / 2.0])
    pts = [(x, y) for y in coords for x in coords][:n]
    return [(x + (i * 0.6180339887498949) % 1.0,
             y + (i * 0.7548776662466927) % 1.0)
            for i, (x, y) in enumerate(pts)]


def _s2_pair_stars(params, sep_arcsec_list, contrast_mag,
                   target_flux=1.0e5, lo=170.0, hi=854.0):
    """9 (or len(sep_arcsec_list)) target+neighbour pairs on a lattice, PA
    rotated 40 deg/pair so the grid samples orientations, not one axis.
    Returns (stars, pairs) for `synth_frame`."""
    ps = params.plate_scale_mas
    lattice = _lattice_positions(len(sep_arcsec_list), lo=lo, hi=hi)
    neighbour_flux = target_flux * 10.0 ** (-contrast_mag / 2.5)
    stars, pairs = [], []
    for i, (sep, (cx, cy)) in enumerate(zip(sep_arcsec_list, lattice)):
        pa_deg = 40.0 * i
        pa = np.radians(pa_deg)
        sep_px = sep * 1000.0 / ps
        tid = len(stars)
        stars.append((cx, cy, target_flux, "target"))
        nx, ny = cx + sep_px * np.cos(pa), cy + sep_px * np.sin(pa)
        nid = len(stars)
        stars.append((nx, ny, neighbour_flux, "neighbour"))
        pairs.append({
            "target_id": tid, "neighbour_id": nid, "sep_arcsec": float(sep),
            "dmag": float(contrast_mag), "pa_deg": float(pa_deg),
        })
    return stars, pairs


def _s2_lattice_frame(params, sr, contrast_mag, separations, seed, *,
                      halo_beta=None, case="S2", target_flux=1.0e5, **kw):
    stars, pairs = _s2_pair_stars(params, separations, contrast_mag,
                                  target_flux=target_flux)
    raw, truth = synth_frame(stars, params, sr=sr, seed=seed, case=case,
                             halo_beta=halo_beta, **kw)
    truth["pairs"] = pairs
    truth["sr_contrast_group"] = {"sr": float(sr),
                                  "contrast_mag": float(contrast_mag)}
    return raw, truth


def build_s2_donor_frame(params, sr, *, halo_beta=None, seed=SEED, **kw):
    """WP-2b: the clean donor frame the harness builds ONE ePSF from per
    (sr, halo_beta) -- 9 isolated singles (no companions) on the
    jittered lattice, at the SAME sr/halo_beta as the pair frames that
    ePSF will clean.

    O3 finding: every star on an S2 PAIR frame is itself part of a
    blend, so an ePSF built directly from a pair frame never converges
    (measured: delta 0.019, PROVISIONAL, vs 0.0021 from a clean donor
    frame with the identical PSF). This separates "how good is the
    neighbour subtraction given a sound ePSF" (what the S2 bias surface
    measures) from "how good is an ePSF built where there are no clean
    donors" (a real question, but S3's -- GC density makes that the
    honest case there, so S3 keeps building its ePSF from its own
    field).

    WP-1d: donor flux is scaled per `_donor_flux_for_window` rather than
    a bare 1.0e5 -- at sr=0.15 that flux's peak falls under the engine's
    donor SNR floor (CP2 assessment item 3). `truth["donor_peak_window_adu"]`
    records the acceptance window used; each donor's actual flux/peak are
    already in `truth["stars"][i]` (synth_frame's own schema, unchanged)."""
    sky = kw.get("sky", 100.0)
    read_noise = kw.get("read_noise", 10.0)
    gain = kw.get("gain", 4.0)
    flux, floor, ceiling = _donor_flux_for_window(
        params, sr, halo_beta=halo_beta, sky=sky, read_noise=read_noise,
        gain=gain)
    donor_stars = [(x, y, flux) for x, y in _lattice_positions(9)]
    raw, truth = synth_frame(donor_stars, params, sr=sr, halo_beta=halo_beta,
                             case="S2_donor", seed=seed, **kw)
    truth["donor_peak_window_adu"] = [float(floor), float(ceiling)]
    return raw, truth


def build_s2(params, *, separations=S2_SEPARATIONS_ARCSEC,
            contrasts=S2_CONTRASTS_MAG, srs=S2_SRS, n_noise=3, **kw):
    """Blended pairs: one lattice frame per (sr, contrast), n_noise noise
    realizations each (same geometry, different noise draw), plus one
    isolated control frame per sr and a 2-frame broad-wing (beta=2.5)
    sensitivity slice at sr=0.30, contrasts {0, 3}."""
    frames = []
    for sr in srs:
        for contrast in contrasts:
            for k in range(n_noise):
                frames.append(_s2_lattice_frame(
                    params, sr, contrast, separations, seed=SEED + k, **kw))
        iso_stars = [(512.0, 512.0, 1.0e5, "target")]
        frames.append(synth_frame(iso_stars, params, sr=sr,
                                  case="S2_isolated", seed=SEED, **kw))
    for contrast in (0, 3):
        frames.append(_s2_lattice_frame(
            params, 0.30, contrast, separations, seed=SEED, halo_beta=2.5,
            case="S2_broadwing", **kw))
    return frames


# -------------------------------------------------------------------- S3

def _sample_gc_dmag(rng, n, slope=S3_LF_SLOPE_DLOGN_DMAG, dmag_max=S3_DMAG_MAX):
    """Inverse-CDF sample from dN/dDmag ~ 10^(slope*Dmag) over
    [0, dmag_max] -- the GC 20260728 luminosity-function slope
    (gc_lf_20260728.json)."""
    u = rng.random(n)
    top = 10.0 ** (slope * dmag_max) - 1.0
    return np.log10(1.0 + u * top) / slope


def build_s3(params, *, n_stars=S3_N_STARS, n_noise=1, seed=SEED, **kw):
    """GC-like crowded field: density and brightness distribution drawn
    from the real 20260728 GC luminosity function (aggregate numbers baked
    into the S3_* constants above). Raises if
    the realized median nearest-neighbour separation misses [0.25, 0.35]"
    -- the builder self-assertion that the field is actually GC-like."""
    ps = params.plate_scale_mas
    dl, moffat = _render_psf_stamp(params, 0.55, engine.MOFFAT_BETA_KOLM, 512)
    peak_frac = float((0.30 * dl + 0.70 * moffat).max())
    donor_ceiling = engine.EPSF_PEAK_CEILING_FRAC * params.max_counts * params.coadds

    frames = []
    for k in range(n_noise):
        rng = np.random.default_rng(seed + k)
        x = rng.uniform(0.0, 1024.0, n_stars)
        y = rng.uniform(0.0, 1024.0, n_stars)
        dmag = _sample_gc_dmag(rng, n_stars)
        flux = S3_BRIGHTEST_FLUX_ADU * 10.0 ** (-dmag / 2.5)
        order = np.argsort(flux)[::-1]
        x, y, flux = x[order], y[order], flux[order]

        pts_arcsec = np.column_stack([x * ps / 1000.0, y * ps / 1000.0])
        dnn, _ = cKDTree(pts_arcsec).query(pts_arcsec, k=2)
        med_nn = float(np.median(dnn[:, 1]))
        if not (S3_MEDIAN_NN_RANGE_ARCSEC[0] <= med_nn
                <= S3_MEDIAN_NN_RANGE_ARCSEC[1]):
            raise AssertionError(
                f"S3 median NN separation {med_nn:.3f}\" outside "
                f"{S3_MEDIAN_NN_RANGE_ARCSEC} (n={n_stars}, seed={seed + k}) "
                "-- field is not GC-like, case is invalid")

        target_mask = (flux * peak_frac) < donor_ceiling
        target_ids = [int(i) for i in np.nonzero(target_mask)[0][:S3_N_TARGETS]]
        stars = [(float(xi), float(yi), float(fi),
                  "target" if i in target_ids else "field")
                 for i, (xi, yi, fi) in enumerate(zip(x, y, flux))]

        raw, truth = synth_frame(stars, params, sr=0.30, case="S3",
                                 seed=seed + k, **kw)
        truth["median_nn_sep_arcsec"] = med_nn
        truth["target_ids"] = target_ids
        frames.append((raw, truth))
    return frames


# -------------------------------------------------------------------- S5

# PLAN section 10.3 / WP-5 handoff (STATUS.md, Lane A -> Lane C,
# 2026-07-31): three density classes drawn from committable numbers, NO
# supplied ePSF anywhere -- build_epsf runs on the field itself (S3's
# own pattern, generalized below via _build_density_field). D34 found
# M92's single-frame density is GC-class, not moderate, so "moderate"
# is defined by density (the donor-count curve's peak, rho=0.15) with
# M92's measured LF SHAPE, not by reusing M92 wholesale.
S5_BRIGHTEST_FLUX_ADU = 1.0e5   # FAKE-1 base, matches S1/S2's convention

S5_SPARSE_N = 20                # rho ~0.02/arcsec^2; "use n=20 fixed" (handoff)
S5_SPARSE_LF_SLOPE = 0.26
S5_SPARSE_DMAG_MAX = 5.0

# D36 (Lane A, 2026-07-31): the original moderate spec (n=16, dmag_max=5.0,
# bare flux 1.0e5) was UNBUILDABLE -- the donor acceptance window
# [EPSF_DONOR_MIN_SNR*sky_sigma, EPSF_PEAK_CEILING_FRAC*max_counts*coadds]
# is only 2.16 mag wide, so an LF spread over 5 mag from one bright anchor
# leaves ~1 star in-window at any density. Corrected: n=24, dmag_max=2.0,
# and the brightest flux is ANCHORED with `_donor_flux_for_window` (the
# WP-1d fix) instead of a bare 1.0e5, so the population lands inside the
# window rather than merely starting from a value that happens to.
# Measured with these: 6 donors, tag=strict, converged=True, delta=0.0017
# -- moderate no longer needs D31 (see D36's "For Lane C" addendum).
S5_MODERATE_N = 24               # rho=0.232/arcsec^2 -- D36's corrected count
S5_MODERATE_LF_SLOPE = 0.33      # measured M92 LF shape (density from this table)
S5_MODERATE_DMAG_MAX = 2.0       # D36: the donor window is only 2.16 mag wide

S5_EXTREME_N = 295               # rho=2.8496/arcsec^2, gc_lf_20260728.json
S5_EXTREME_LF_SLOPE = 0.2605
S5_EXTREME_DMAG_MAX = 5.617
S5_EXTREME_BRIGHTEST_FLUX_ADU = 3.0e5   # matches S3's own GC-class flux
S5_EXTREME_NN_RANGE_ARCSEC = (0.25, 0.35)   # same field as S3 -- reuse its tolerance
S5_EXTREME_N_TARGETS = 30

# D36 item 2: two moderate targets landed at x=1012.7 and x=5.4 -- against
# the frame edge, where `measure_strehl` must fail outright (builder
# failures, not engine ones, that pollute any bias statistic). The margin
# mirrors `select_neighbours`' own reach (`bg_outer_arcsec + r_stamp_px`,
# and `build_epsf`'s `r_stamp_arcsec` defaults to `bg_outer_arcsec` too) --
# a target closer to the border than this has neighbours/annulus pixels
# guaranteed off-array regardless of what the field around it looks like.
S5_EDGE_MARGIN_ARCSEC = 2.0 * engine.NIRC2_BG_OUTER_RADIUS_ARCSEC


def _build_density_field(params, *, n_stars, brightest_flux_adu, lf_slope,
                         dmag_max, sr=0.30, n_targets=None, seed=SEED,
                         n_noise=1, case="S5", nn_expect_range=None,
                         shape=(1024, 1024), **kw):
    """Shared machinery behind the three WP-5 density classes -- S3's own
    pattern (`build_s3`), generalized so density / LF slope / brightest
    flux / target count are parameters instead of S3's hardcoded module
    constants. `n_targets=None` measures every star (sparse/moderate, few
    enough that all of them are useful); an int caps it to the N
    brightest below the donor peak ceiling (S3's own convention,
    appropriate for extreme's 295 stars). `nn_expect_range`, if given,
    is a hard builder self-assertion like S3's; otherwise the realized
    median NN separation is recorded in truth but not enforced -- the
    handoff gives only an approximate expectation for sparse/moderate,
    not a validated tolerance.

    D36 item 2: candidate TARGETS (not field/donor stars -- those may
    legitimately sit anywhere, same as any real field) within
    `S5_EDGE_MARGIN_ARCSEC` of any border are excluded before the
    brightest-N selection, so a target can never land where
    `measure_strehl`'s own aperture/annulus reach must go off-array."""
    ps = params.plate_scale_mas
    ny, nx = shape
    margin_px = S5_EDGE_MARGIN_ARCSEC * 1000.0 / ps
    dl, moffat = _render_psf_stamp(params, 0.55, engine.MOFFAT_BETA_KOLM, 512)
    peak_frac = float((sr * dl + (1.0 - sr) * moffat).max())
    donor_ceiling = engine.EPSF_PEAK_CEILING_FRAC * params.max_counts * params.coadds

    frames = []
    for k in range(n_noise):
        rng = np.random.default_rng(seed + k)
        x = rng.uniform(0.0, float(nx), n_stars)
        y = rng.uniform(0.0, float(ny), n_stars)
        dmag = _sample_gc_dmag(rng, n_stars, slope=lf_slope, dmag_max=dmag_max)
        flux = brightest_flux_adu * 10.0 ** (-dmag / 2.5)
        order = np.argsort(flux)[::-1]
        x, y, flux = x[order], y[order], flux[order]

        pts_arcsec = np.column_stack([x * ps / 1000.0, y * ps / 1000.0])
        dnn, _ = cKDTree(pts_arcsec).query(pts_arcsec, k=2)
        med_nn = float(np.median(dnn[:, 1]))
        if nn_expect_range is not None:
            lo, hi = nn_expect_range
            if not (lo <= med_nn <= hi):
                raise AssertionError(
                    f"{case} median NN separation {med_nn:.3f}\" outside "
                    f"{nn_expect_range} (n={n_stars}, seed={seed + k}) -- "
                    "field density does not match the WP-5 spec, case "
                    "is invalid")

        edge_clear = ((x >= margin_px) & (x <= nx - margin_px)
                      & (y >= margin_px) & (y <= ny - margin_px))
        target_mask = ((flux * peak_frac) < donor_ceiling) & edge_clear
        cand_ids = np.nonzero(target_mask)[0]
        if n_targets is not None:
            cand_ids = cand_ids[:n_targets]
        target_ids = [int(i) for i in cand_ids]
        stars = [(float(xi), float(yi), float(fi),
                  "target" if i in target_ids else "field")
                 for i, (xi, yi, fi) in enumerate(zip(x, y, flux))]

        raw, truth = synth_frame(stars, params, sr=sr, case=case,
                                 seed=seed + k, shape=shape, **kw)
        truth["median_nn_sep_arcsec"] = med_nn
        truth["target_ids"] = target_ids
        frames.append((raw, truth))
    return frames


def build_s5_sparse(params, *, seed=SEED, n_noise=1, **kw):
    """WP-5 sparse class: ~20 stars over the NIRC2 narrow FOV
    (rho ~0.02/arcsec^2 by construction). D34's donor-count table says
    plainly there are not enough stars here for `build_epsf` to find
    donors either -- this is the real-field analogue of the S1 no-op
    guard, and per the handoff it is the one WP-5 class assertable
    right now (must hold before AND after D31)."""
    return _build_density_field(
        params, n_stars=S5_SPARSE_N, brightest_flux_adu=S5_BRIGHTEST_FLUX_ADU,
        lf_slope=S5_SPARSE_LF_SLOPE, dmag_max=S5_SPARSE_DMAG_MAX,
        n_targets=None, seed=seed, n_noise=n_noise, case="S5_sparse", **kw)


def build_s5_moderate(params, *, seed=SEED, n_noise=1, sr=0.30, **kw):
    """WP-5 moderate class: rho=0.232/arcsec^2 (24 stars), D36's CORRECTED
    spec -- LF slope 0.33 is the measured M92 shape (unchanged); n and
    dmag_max were revised because the original handoff values (16, 5.0)
    were unbuildable: the donor acceptance window is only 2.16 mag wide,
    so a 5-mag-deep LF from one bright anchor leaves ~1 star in-window at
    any density. The brightest flux is ANCHORED with
    `_donor_flux_for_window` (the WP-1d fix) rather than a bare constant,
    so the population lands INSIDE the window instead of merely starting
    from a value that happens to. Measured: 6 donors, tag=strict,
    converged=True, delta=0.0017 -- moderate BUILDS TODAY and does not
    need D31 (D36's "For Lane C" addendum, item 3)."""
    brightest_flux_adu, _floor, _ceiling = _donor_flux_for_window(params, sr)
    return _build_density_field(
        params, n_stars=S5_MODERATE_N, brightest_flux_adu=brightest_flux_adu,
        lf_slope=S5_MODERATE_LF_SLOPE, dmag_max=S5_MODERATE_DMAG_MAX,
        n_targets=None, seed=seed, n_noise=n_noise, sr=sr,
        case="S5_moderate", **kw)


def build_s5_extreme(params, *, seed=SEED, n_noise=1, **kw):
    """WP-5 extreme class: GC-class, the real 20260728 luminosity
    function (`gc_lf_20260728.json`) -- the same field S3 already
    builds, restated under its own S5 name/case rather than reusing
    `build_s3` directly, since S5 is its own no-supplied-ePSF acceptance
    battery (S3 stays exactly as Phase 1 left it). Per D32 the pass
    criterion is a legible, MODEL-level refusal (one line for the whole
    field), not a numeric accuracy target -- either outcome (refusal, or
    a build that succeeds after D31) is a documented pass."""
    return _build_density_field(
        params, n_stars=S5_EXTREME_N,
        brightest_flux_adu=S5_EXTREME_BRIGHTEST_FLUX_ADU,
        lf_slope=S5_EXTREME_LF_SLOPE, dmag_max=S5_EXTREME_DMAG_MAX,
        n_targets=S5_EXTREME_N_TARGETS, seed=seed, n_noise=n_noise,
        case="S5_extreme", nn_expect_range=S5_EXTREME_NN_RANGE_ARCSEC, **kw)


# -------------------------------------------------------------------- S4

def _s4a_pair(params):
    """S4a flat-consistency: the SAME S1-like frame built with and without
    the flat step (the 2026-07-23 lesson guard -- a missing flat step
    masquerades as a ~2% systematic)."""
    stars = [(512.37, 511.61, 1.0e5, "target")]
    with_flat = synth_frame(stars, params, sr=0.30, case="S4a_flat", seed=SEED)
    without_flat = synth_frame(stars, params, sr=0.30, case="S4a_noflat",
                               seed=SEED, flat=False)
    return [with_flat, without_flat]


def build_s4_saturation(params, **kw):
    """S4b: an S2-like lattice frame plus 3 extra stars whose peaks exceed
    saturation_adu (hard-clipped by the builder's minimum() step), one of
    them 0.4" from the first lattice target -- a saturated NEIGHBOUR."""
    sr, contrast = 0.30, 2
    stars, pairs = _s2_pair_stars(params, S2_SEPARATIONS_ARCSEC, contrast)
    dl, moffat = _render_psf_stamp(params, 0.55, engine.MOFFAT_BETA_KOLM, 512)
    peak_frac = float((sr * dl + (1.0 - sr) * moffat).max())
    saturation_adu = params.max_counts * params.coadds
    sat_flux = 5.0 * saturation_adu / max(peak_frac, 1e-6)

    ps = params.plate_scale_mas
    tx, ty = stars[0][0], stars[0][1]
    sep_px = 0.4 * 1000.0 / ps
    extra = [
        (tx + sep_px, ty, sat_flux, "saturated_neighbour"),
        (300.0, 900.0, sat_flux, "saturated_isolated_1"),
        (900.0, 300.0, sat_flux, "saturated_isolated_2"),
    ]
    raw, truth = synth_frame(stars + extra, params, sr=sr, case="S4b",
                             seed=SEED, **kw)
    truth["pairs"] = pairs
    return [(raw, truth)]


def build_s4_wide(params, *, separations=S2_SEPARATIONS_ARCSEC,
                  contrasts=(0, 2), sr=0.30, **kw):
    """S4c: wide-camera (1.10 px/lambda/D, genuinely sub-Nyquist) frames.
    The builder only makes the frames; the oversample in {2, 4} sweep is
    the harness's job (STATUS.md).

    WP-1d: target flux is scaled per `_donor_flux_for_window` -- a bare
    1.0e5 ADU target SATURATES the donor peak-ceiling cut on wide (CP2
    assessment item 3: a wide pixel covers ~16x the sky area of a narrow
    one, so the same total flux gives ~16x the peak). Neighbour flux
    still follows from the target via the contrast ratio
    (`_s2_pair_stars`), so relative contrast is unchanged.

    SECOND, FLAGGED deviation from the original WP-1 spec (3 separations
    {0.3, 0.6, 1.0}"): default widened to all 9 of `S2_SEPARATIONS_ARCSEC`.
    With flux fixed, 3 separations means only 6 catalogued stars total, and
    even a couple lost to mutual pair-proximity (unavoidable at wide's
    coarse plate scale: 0.3" is ~7.6 px there) leaves fewer than
    `EPSF_MIN_DONORS=5` -- a DIFFERENT failure mode than the peak-window
    one WP-1d targeted, discovered only after fixing that one. 9
    separations gives 18 candidates, comfortable margin. This is a
    builder-side geometry choice, not a change to the engine's
    acceptance window."""
    wide_params = _derive_params(params, camname="wide")
    sky = kw.get("sky", 100.0)
    read_noise = kw.get("read_noise", 10.0)
    gain = kw.get("gain", 4.0)
    flux, floor, ceiling = _donor_flux_for_window(
        wide_params, sr, sky=sky, read_noise=read_noise, gain=gain)
    frames = []
    for contrast in contrasts:
        raw, truth = _s2_lattice_frame(wide_params, sr, contrast,
                                       separations, seed=SEED, case="S4c",
                                       target_flux=flux, **kw)
        truth["donor_peak_window_adu"] = [float(floor), float(ceiling)]
        frames.append((raw, truth))
    return frames


def build_s4_ladder(params, **kw):
    """S4d: the donor-isolation ladder (strict / loose / uncalibrated) --
    three sparse frames exercising `build_epsf`'s degraded-mode rungs.

    Lattice margin matches `_s2_pair_stars`' own default (170-854 px):
    `donor_candidates`' first cut requires the WHOLE stamp + sky annulus
    on-array, and r_stamp defaults to bg_outer_arcsec (1.4" = 140.8 px on
    narrow), so anything inside ~141 px of an edge is rejected outright.

    `read_noise` defaults lower than `synth_frame`'s own default (2.0 ADU,
    not 10.0): `donor_candidates`' annulus-crowding cut (<=
    CROWDING_WARN_FRAC) is the same mean-vs-clipped-median statistic that
    made a genuinely isolated S1 star read spuriously CROWDED at high
    noise (see synth_frame's docstring) -- at 10 ADU it intermittently
    tipped 3-4 of these 8 bright, isolated ladder donors over the 0.05
    crowding floor by pure chance, which is a noise-realization artifact
    of the TEST, not a donor-selection question worth exercising here."""
    kw.setdefault("read_noise", 2.0)
    flux = 1.0e5
    ps = params.plate_scale_mas

    strict_pts = _lattice_positions(8)
    stars_strict = [(x, y, flux, "donor") for x, y in strict_pts]
    raw_s, truth_s = synth_frame(stars_strict, params, sr=0.30,
                                 case="S4d_strict", seed=SEED, **kw)
    truth_s["expected_epsf_tag"] = "strict"

    loose_pts = _lattice_positions(6)
    sep_px = 0.35 * 1000.0 / ps
    stars_loose = []
    for x, y in loose_pts:
        stars_loose.append((x, y, flux, "donor"))
        stars_loose.append((x + sep_px, y, flux * 0.5, "companion"))
    raw_l, truth_l = synth_frame(stars_loose, params, sr=0.30,
                                 case="S4d_loose", seed=SEED, **kw)
    truth_l["expected_epsf_tag"] = "loose"

    # D40 (STATUS.md, OPEN for Lane C item 1): 4 stars gave exactly 4
    # usable donors, which BUILDS under EPSF_MIN_DONORS=4 (correct engine
    # behaviour -- D39 lowered it from 5). A count picked to sit one below
    # whatever the current threshold happens to be is fragile by
    # construction; a single candidate star caps usable donors at <= 1
    # no matter what EPSF_MIN_DONORS is set to next, short of 1 itself.
    unc_pts = _lattice_positions(1)
    stars_unc = [(x, y, flux, "donor") for x, y in unc_pts]
    raw_u, truth_u = synth_frame(stars_unc, params, sr=0.30,
                                 case="S4d_uncalibrated", seed=SEED, **kw)
    truth_u["expected_epsf_tag"] = "uncalibrated"

    return [(raw_s, truth_s), (raw_l, truth_l), (raw_u, truth_u)]


def build_s4(params, **kw):
    return {
        "flat": _s4a_pair(params),
        "saturation": build_s4_saturation(params, **kw),
        "wide": build_s4_wide(params, **kw),
        "ladder": build_s4_ladder(params, **kw),
    }


# ------------------------------------------------------------- self-check

def _selfcheck():
    """WP-1 acceptance criteria 2-4 (STATUS.md): a runnable snippet
    recovering sr_truth_isolated on S1, S4a agreement <= 0.002, and S3's
    median-NN builder assertion. Criterion 1 (ruff) and 5 (schema
    validation, exercised automatically by every synth_frame call above)
    are not repeated here."""
    import time
    t0 = time.time()
    print("psf_fit_synth self-check:")
    params = synth_params()
    flat = engine.load_nirc2_calibration()[0]

    raw, truth = build_s1(params)[0]
    star = truth["stars"][0]
    r = engine.measure_strehl(engine.reduce_frame(raw, flat=flat),
                              params=params, pos=(star["x"], star["y"]))
    ok = r.ok and abs(r.strehl - truth["sr_truth_isolated"]) < 0.02
    print(f"  [{'ok' if ok else 'FAIL'}] S1 measured SR {r.strehl:.4f} vs "
          f"sr_truth_isolated {truth['sr_truth_isolated']:.4f}")

    (raw_f, truth_f), (raw_nf, truth_nf) = _s4a_pair(params)
    r_f = engine.measure_strehl(
        engine.reduce_frame(raw_f, flat=flat), params=params,
        pos=(truth_f["stars"][0]["x"], truth_f["stars"][0]["y"]))
    r_nf = engine.measure_strehl(
        engine.reduce_frame(raw_nf, flat=None), params=params,
        pos=(truth_nf["stars"][0]["x"], truth_nf["stars"][0]["y"]))
    d = abs(r_f.strehl - r_nf.strehl)
    print(f"  [{'ok' if d <= 0.002 else 'FAIL'}] S4a flat-consistency "
          f"|dS|={d:.4f} <= 0.002  (with-flat {r_f.strehl:.4f}, "
          f"without-flat {r_nf.strehl:.4f})")

    s3_frames = build_s3(params, n_noise=1)
    med_nn = s3_frames[0][1]["median_nn_sep_arcsec"]
    print(f"  [ok] S3 median NN separation {med_nn:.3f}\" in "
          f"{S3_MEDIAN_NN_RANGE_ARCSEC} (builder self-assertion enforced "
          "at construction time)")

    print(f"done in {time.time() - t0:.1f}s")
    return ok and d <= 0.002


if __name__ == "__main__":
    passed = _selfcheck()
    sys.exit(0 if passed else 1)
