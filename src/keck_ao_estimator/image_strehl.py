"""Measured Strehl from NIRC2 images (port of the K2 IDL Strehl tool).

Pipeline (find_strehl.pro + calc_and_display.pro, widget defaults):
reduce (background, flat floored at 0.2, bad-pixel repair) -> 3x3 sigma
filter -> star at brightest pixel or supplied position -> DAOPHOT-style
derivative centroid (cntrd.pro) -> aperture photometry with a mean-sky
annulus (mvdaper.pro) -> pixelation-corrected peak via sinc-deconvolved
8x Fourier upsampling (find_peak.pro) -> Strehl against the identical
photometry on the diffraction-limited PSF -> spline radial-profile FWHM
(find_fwhm.pro) and Marechal WFE.

Index convention: numpy arrays are [row, col] = [y, x]; all (x, y)
arguments and results use IDL/detector convention (x = column).

Known deviations from the IDL original, all documented against the golden
outputs in the regress model: the pupil rasterizer matches POLYFILLV to
15/262144 pixels (see nirc2_psf._fill_polygon), leaving measured Strehl
within 0.001 of the IDL tool; the radial-profile spline is a natural
cubic rather than IDL SPLINE's tension-1.0 spline; the saturation check
uses the raw sub-image maximum rather than a cubic-congrid upsample; auto
bad-pixel detection (fix_image.pro) is not ported -- a bad-pixel mask is
required for repair.
"""
from dataclasses import dataclass

import numpy as np

from .nirc2 import (
    NIRC2_BG_INNER_RADIUS_ARCSEC, NIRC2_BG_OUTER_RADIUS_ARCSEC,
    NIRC2_PEAK_RADIUS_ARCSEC, NIRC2_PHOTOMETRY_RADIUS_ARCSEC,
    Nirc2FrameParams, nirc2_frame_params,
)
from .nirc2_psf import nirc2_dl_psf
# NOTE: psf_fit imports aperture_flux from here, so importing it at module
# level is circular. The D27 bias constants are fetched lazily below, in
# the one place they are used.

# annulus contamination (mean-vs-clipped-median sky disagreement integrated
# over the aperture) above this fraction of the measured flux flags the
# measurement as CROWDED -- the plain-mean sky is no longer trustworthy
CROWDING_WARN_FRAC = 0.05
# fraction of the sky annulus's OUTER disc allowed off the array before a
# measurement is flagged EDGE and the curve-of-growth auto radius is
# disabled: a clipped growth curve flattens early (the halo area is simply
# missing), settles at a too-small aperture, and inflates peak/flux --
# Eduardo's i260226_a017005 edge star read SR 0.50 auto vs 0.24 fixed
EDGE_CLIP_WARN_FRAC = 0.05

# auto star-count quality gate: a field-map point whose sky-noise SR
# uncertainty exceeds this is too poor to constrain the field
SR_ERR_MAX = 0.05


# ---------------------------------------------------------------- utilities

def _idl_median(values):
    """IDL MEDIAN without /EVEN: upper-middle element of the sorted data."""
    v = np.sort(np.asarray(values).ravel())
    return v[v.size // 2]


def _smooth3(a):
    """IDL SMOOTH(a, 3): 3x3 boxcar mean, edge pixels left unchanged."""
    out = a.copy()
    c = np.zeros_like(a[1:-1, 1:-1])
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            c += a[1 + dy:a.shape[0] - 1 + dy, 1 + dx:a.shape[1] - 1 + dx]
    out[1:-1, 1:-1] = c / 9.0
    return out


def sigma_filter3(image, n_sigma=3.0):
    """Astrolib sigma_filter with the tool's defaults (3x3 box).

    Pixels deviating from their 8-neighbor mean by more than n_sigma times
    the local scatter are replaced by that mean.
    """
    im = np.asarray(image, dtype=float)
    mean = (9.0 * _smooth3(im) - im) / 8.0
    dev = (im - mean) ** 2
    fact = float(n_sigma) ** 2 / 7.0            # N_sigma^2 / (box^2 - 2)
    var = fact * (9.0 * _smooth3(dev) - dev)    # astrolib: no /(bw^2-1) here
    out = im.copy()
    bad = dev >= var
    out[bad] = mean[bad]
    return out


def radius_map(shape, x, y):
    """Distance of every pixel from (x, y) in pixels (radmap.pro)."""
    yy = np.arange(shape[0], dtype=float) - y
    xx = np.arange(shape[1], dtype=float) - x
    return np.hypot(xx[None, :], yy[:, None])


def deadpix_fill(image, badmask, neighbors=3):
    """Iterative bad-pixel repair (bpixfix.pro): each pass replaces bad
    pixels having >= `neighbors` good 3x3 neighbors with the IDL-median of
    those neighbors, until none remain."""
    im = np.asarray(image, dtype=float).copy()
    bp = np.asarray(badmask, dtype=bool).copy()
    ny, nx = im.shape
    while bp.any():
        fixed_any = False
        ys, xs = np.nonzero(bp)
        good = ~bp
        for yb, xb in zip(ys, xs):
            y0, y1 = max(yb - 1, 0), min(yb + 1, ny - 1) + 1
            x0, x1 = max(xb - 1, 0), min(xb + 1, nx - 1) + 1
            g = good[y0:y1, x0:x1]
            if g.sum() >= neighbors:
                im[yb, xb] = _idl_median(im[y0:y1, x0:x1][g])
                bp[yb, xb] = False
                fixed_any = True
        if not fixed_any:       # isolated cluster smaller than `neighbors`
            break
    return im


def _binary_erode3(m):
    """3x3 binary erosion, zero-padded edges (IDL ERODE semantics)."""
    p = np.zeros((m.shape[0] + 2, m.shape[1] + 2), dtype=bool)
    p[1:-1, 1:-1] = m
    out = np.ones_like(m, dtype=bool)
    for dy in (0, 1, 2):
        for dx in (0, 1, 2):
            out &= p[dy:dy + m.shape[0], dx:dx + m.shape[1]]
    return out


def _binary_dilate3(m):
    """3x3 binary dilation, zero-padded edges (IDL DILATE semantics)."""
    p = np.zeros((m.shape[0] + 2, m.shape[1] + 2), dtype=bool)
    p[1:-1, 1:-1] = m
    out = np.zeros_like(m, dtype=bool)
    for dy in (0, 1, 2):
        for dx in (0, 1, 2):
            out |= p[dy:dy + m.shape[0], dx:dx + m.shape[1]]
    return out


def fix_image(image):
    """Auto bad-pixel repair (fix_image.pro, O. Lai / D.S. Acton / MvD):
    the OSIRIS tool's whole reduction beyond background subtraction.

    Kill the four (wrapped-apodization) corner pixels, flag |pixel| >
    3 sigma of the whole image, morphologically OPEN the flag map with a
    3x3 kernel and keep only what opening removes -- isolated spikes;
    star cores are extended and survive opening, so they are spared --
    then iteratively replace each flagged pixel with the mean of its
    (wrapping, per IDL SHIFT) 4-neighbors."""
    focn = np.asarray(image, dtype=float).copy()
    for yi in (0, -1):
        for xi in (0, -1):
            focn[yi, xi] = 0.0
    flag = np.abs(focn) > 3.0 * focn.std()
    dead = flag & ~_binary_dilate3(_binary_erode3(flag))

    bmap = dead.astype(float)
    gmap = 1.0 - bmap
    im = focn * gmap

    def _n4(a):
        return (np.roll(a, 1, 0) + np.roll(a, -1, 0)
                + np.roll(a, 1, 1) + np.roll(a, -1, 1))

    while bmap.sum():
        t1 = _n4(gmap)
        t2 = _n4(im)
        im = im + (t1 != 0) * bmap * t2 / (t1 + 1e-15)
        bmap = (t1 == 0).astype(float)
        gmap = 1.0 - bmap
    return im


# ---------------------------------------------------------------- centroid

def cntrd(img, x, y, fwhm, silent=True):
    """DAOPHOT FIND derivative centroid (cntrd.pro). Returns (xcen, ycen),
    (-1., -1.) when the centroid cannot be computed."""
    img = np.asarray(img, dtype=float)
    ysize, xsize = img.shape
    nhalf = max(int(0.637 * fwhm), 2)
    nbox = 2 * nhalf + 1
    nhalfbig = nhalf + 3
    nbig = nbox + 6
    ix = int(x + 0.5)
    iy = int(y + 0.5)

    if (ix < nhalfbig or ix + nhalfbig > xsize - 1 or
            iy < nhalfbig or iy + nhalfbig > ysize - 1):
        return -1.0, -1.0

    bigbox = img[iy - nhalfbig:iy + nhalfbig + 1,
                 ix - nhalfbig:ix + nhalfbig + 1]
    flat = np.flatnonzero(bigbox == bigbox.max())
    idx = flat % nbig
    idy = flat // nbig
    if flat.size > 1:
        idx = int(round(idx.sum() / flat.size))
        idy = int(round(idy.sum() / flat.size))
    else:
        idx = int(idx[0])
        idy = int(idy[0])
    xmax = ix - nhalfbig + idx
    ymax = iy - nhalfbig + idy

    if (xmax < nhalf or xmax + nhalf > xsize - 1 or
            ymax < nhalf or ymax + nhalf > ysize - 1):
        return -1.0, -1.0

    strbox = img[ymax - nhalf:ymax + nhalf + 1,
                 xmax - nhalf:xmax + nhalf + 1]
    ir = max(nhalf - 1, 1)
    dd = np.arange(nbox - 1) + 0.5 - nhalf
    w = 1.0 - 0.5 * (np.abs(dd) - 0.5) / (nhalf - 0.5)
    sumc = w.sum()

    # X: difference along x, sum over the central y rows
    deriv = np.roll(strbox, -1, axis=1) - strbox
    deriv = deriv[nhalf - ir:nhalf + ir + 1, 0:nbox - 1].sum(axis=0)
    sumd = (w * deriv).sum()
    sumxd = (w * dd * deriv).sum()
    sumxsq = (w * dd ** 2).sum()
    if sumxd >= 0:
        return -1.0, -1.0
    dx = sumxsq * sumd / (sumc * sumxd)
    if abs(dx) > nhalf:
        return -1.0, -1.0
    xcen = xmax - dx

    # Y: difference along y, sum over the central x columns
    deriv = np.roll(strbox, -1, axis=0) - strbox
    deriv = deriv[0:nbox - 1, nhalf - ir:nhalf + ir + 1].sum(axis=1)
    sumd = (w * deriv).sum()
    sumxd = (w * dd * deriv).sum()
    if sumxd >= 0:
        return -1.0, -1.0
    dy = sumxsq * sumd / (sumc * sumxd)
    if abs(dy) > nhalf:
        return -1.0, -1.0
    ycen = ymax - dy

    return float(xcen), float(ycen)


# ----------------------------------------------------------- peak and flux

def find_peak(image, x, y, boxsize, oversamp=8):
    """Pixelation-corrected peak via sinc-deconvolved Fourier upsampling
    (find_peak.pro): FFT a box around the star, deconvolve the pixel
    transfer function, zero-pad by `oversamp`, inverse FFT, take the max."""
    image = np.asarray(image, dtype=float)
    boxsize = 2 * int(np.ceil(boxsize / 2.0))
    boxhalf = boxsize // 2
    ext = boxsize * oversamp

    fftsinc = np.zeros(ext)
    fftsinc[:oversamp] = 1.0
    karr = np.roll(np.arange(ext, dtype=float) - ext / 2.0, ext // 2)
    sinc = (boxsize * np.fft.fft(fftsinc) / ext *
            np.exp(1j * karr / ext * np.pi * (oversamp - 1)))
    sinc = np.roll(np.real(sinc), ext // 2)
    sinc = sinc[ext // 2 - boxhalf:ext // 2 + boxhalf]
    sinc2d = np.outer(sinc, sinc)

    blx = int(np.floor(x - boxhalf))
    bly = int(np.floor(y - boxhalf))
    blx = min(max(blx, 0), image.shape[1] - boxsize)
    bly = min(max(bly, 0), image.shape[0] - boxsize)
    subim = image[bly:bly + boxsize, blx:blx + boxsize]

    fftim = np.fft.fft2(subim) / (boxsize * boxsize)
    sh = np.roll(fftim, (-boxhalf, -boxhalf), axis=(0, 1)) / sinc2d
    zp = np.zeros((ext, ext), dtype=complex)
    zp[:boxsize, :boxsize] = sh
    zp = np.roll(zp, (-boxhalf, -boxhalf), axis=(0, 1))
    upsampled = np.real(np.fft.ifft2(zp)) * ext * ext
    return float(upsampled.max())


def sigma_clipped_median(values, n_sigma=3.0, iters=5):
    """Iterated sigma-clipped median: reject > n_sigma outliers about the
    median until stable.  In a crowded field the sky annulus is mostly
    empty sky plus point-source spikes, so this recovers the true sky
    where a plain mean is dragged high by the neighbors."""
    v = np.asarray(values, dtype=float).ravel()
    for _ in range(iters):
        med = np.median(v)
        sig = v.std()
        if sig == 0.0:
            break
        keep = np.abs(v - med) <= n_sigma * sig
        if keep.all():
            break
        v = v[keep]
    return float(np.median(v))


def aperture_flux(image, aper_px, x, y, insky_px=None, outsky_px=None,
                  skyval=None, robust=False):
    """Single-star aperture photometry (mvdaper.pro): flux inside aper_px
    after subtracting the [insky, outsky] annulus sky (or a given skyval).
    The annulus estimator is mvdaper's plain mean by default; robust=True
    uses the sigma-clipped median instead (a deliberate deviation for
    crowded fields -- neighbors in the annulus drag the mean high).
    Returns (flux, sky, crowding) where crowding is the sky-estimator
    disagreement (mean - clipped median) integrated over the aperture and
    normalized by |flux| -- the fraction of the measured flux attributable
    to annulus contamination (0.0 when skyval was supplied)."""
    image = np.asarray(image, dtype=float)
    # Work on the bounding box of the largest radius we need, not the whole
    # frame.  Building a 1024x1024 radius map to select a ~100 px aperture
    # costs ~13x more pixels than the job requires, and this is the single
    # hottest call in the crowded-field path (donor vetting and every group
    # fit go through it thousands of times).
    #
    # BIT-IDENTICAL, not merely close: the box is clipped to the array, so
    # it selects exactly the pixels the full-frame map would; and a
    # boolean-indexed gather walks row-major, so restricting to a box
    # preserves both the SET and the ORDER of the selected values.  Same
    # values in the same order means `.sum()` accumulates identically,
    # down to the last ulp.  `radius_map` itself is unchanged -- it is
    # public API and the byte-identity harness covers callers of it.
    reach = float(aper_px)
    if skyval is None and outsky_px is not None:
        reach = max(reach, float(outsky_px))
    y0 = max(int(np.floor(y - reach)), 0)
    y1 = min(int(np.ceil(y + reach)) + 1, image.shape[0])
    x0 = max(int(np.floor(x - reach)), 0)
    x1 = min(int(np.ceil(x + reach)) + 1, image.shape[1])
    if y1 <= y0 or x1 <= x0:                 # entirely off-array
        return 0.0, float(skyval or 0.0), 0.0, 0.0, 0, 0
    sub = image[y0:y1, x0:x1]
    rmap = radius_map(sub.shape, x - x0, y - y0)
    ap_sel = rmap <= aper_px
    crowding = 0.0
    sky_sigma = 0.0
    n_ann = 0
    if skyval is None:
        ann = sub[(rmap >= insky_px) & (rmap <= outsky_px)]
        mean_sky = float(ann.mean())
        clipped_sky = sigma_clipped_median(ann)
        sky_sigma = 1.4826 * float(np.median(np.abs(ann - clipped_sky)))
        n_ann = int(ann.size)
        skyval = clipped_sky if robust else mean_sky
        flux = float((sub[ap_sel] - skyval).sum())
        if flux != 0.0:
            crowding = abs(mean_sky - clipped_sky) * ap_sel.sum() / abs(flux)
    else:
        flux = float((sub[ap_sel] - skyval).sum())
    return flux, float(skyval), crowding, sky_sigma, int(ap_sel.sum()), n_ann


# ------------------------------------------------------------ radial FWHM

def _natural_cubic_spline(xk, yk, xt):
    """Natural cubic spline interpolation (stand-in for IDL SPLINE)."""
    xk = np.asarray(xk, dtype=float)
    yk = np.asarray(yk, dtype=float)
    n = xk.size
    h = np.diff(xk)
    # tridiagonal solve for second derivatives, natural boundaries
    a = np.zeros(n)
    b = np.ones(n)
    c = np.zeros(n)
    d = np.zeros(n)
    b[1:-1] = 2.0 * (h[:-1] + h[1:])
    a[1:-1] = h[:-1]
    c[1:-1] = h[1:]
    d[1:-1] = 6.0 * ((yk[2:] - yk[1:-1]) / h[1:] - (yk[1:-1] - yk[:-2]) / h[:-1])
    for i in range(1, n):           # Thomas algorithm
        m = a[i] / b[i - 1]
        b[i] -= m * c[i - 1]
        d[i] -= m * d[i - 1]
    m2 = np.zeros(n)
    m2[-1] = d[-1] / b[-1]
    for i in range(n - 2, -1, -1):
        m2[i] = (d[i] - c[i] * m2[i + 1]) / b[i]

    j = np.clip(np.searchsorted(xk, xt) - 1, 0, n - 2)
    t = xt - xk[j]
    hj = h[j]
    return (yk[j] + t * ((yk[j + 1] - yk[j]) / hj - hj * (2 * m2[j] + m2[j + 1]) / 6.0)
            + t ** 2 * m2[j] / 2.0 + t ** 3 * (m2[j + 1] - m2[j]) / (6.0 * hj))


def radial_profile_fwhm(image, x, y):
    """FWHM in pixels from a spline of the radial profile (find_fwhm.pro).
    Returns -1.0 when the profile peak is off-center or no half-max
    crossing is found.  The input should already be sky-subtracted."""
    image = np.asarray(image, dtype=float)
    inrad = 0.5 * np.sqrt(2.0)
    outrad = 20.0
    drad = 1.0
    outsky = outrad + 2.0 * drad + 20.0

    sz_y, sz_x = image.shape
    x0 = max(int(np.floor(x - outsky)), 0)
    x1 = min(int(np.ceil(x + outsky)), sz_x - 1)
    y0 = max(int(np.floor(y - outsky)), 0)
    y1 = min(int(np.ceil(y + outsky)), sz_y - 1)
    img = image[y0:y1 + 1, x0:x1 + 1]
    distsq = radius_map(img.shape, x - x0, y - y0) ** 2

    nrad = int(np.ceil((outrad - inrad) / drad)) + 1
    rad = np.zeros(nrad)
    prof = np.zeros(nrad)
    for i in range(nrad):
        if i == 0:
            rin, rout, rin2 = 0.0, inrad, -0.01
        else:
            rin = inrad + drad * (i - 1)
            rout = min(rin + drad, outrad)
            rin2 = rin * rin
        sel = (distsq > rin2) & (distsq <= rout * rout)
        np_count = int(sel.sum())
        if np_count > 0:
            rad[i] = (rout + rin) / 2.0
            prof[i] = img[sel].sum() / np_count
        else:
            rad[i] = rout
            prof[i] = prof[i - 1] if i else 0.0

    if prof.argmax() != 0:
        return -1.0     # profile peak off-center
    splrad = rad.min() + np.arange(nrad * 50 + 1) * (rad.max() - rad.min()) / (nrad * 50)
    splprof = _natural_cubic_spline(rad, prof, splrad)
    half = 0.5 * splprof.max()
    below = np.nonzero(splprof < half)[0]
    if below.size == 0 or below[0] < 2:
        return -1.0
    i = below[0]
    return float(splrad[i] + splrad[i - 1])


# ------------------------------------------------------------- reduction

def reduce_frame(raw, background=None, flat=None, badmask=None):
    """(raw - background) / flat centred into a square frame, then
    bad-pixel repair when a mask is supplied (find_strehl.pro).

    The embed target is 1024x1024 (the IDL original's hardcoded NIRC2
    detector) or the frame's own size when larger, so full-frame data
    from bigger detectors (e.g. the 2048x2048 OSIRIS imager) reduces
    without slicing errors.  Calibration arrays must match the embedded
    shape -- a mismatch raises ValueError rather than mis-applying a
    detector's calibration to a different instrument."""
    raw = np.asarray(raw, dtype=float)
    n = max(1024, *raw.shape)

    def _embed(a):
        ys, xs = a.shape
        out = np.zeros((n, n))
        bly, blx = n // 2 - ys // 2, n // 2 - xs // 2
        out[bly:bly + ys, blx:blx + xs] = a
        return out

    im = _embed(raw)
    bg = _embed(np.asarray(background, dtype=float)) \
        if background is not None else 0.0
    red = im - bg
    if flat is not None:
        flat = np.asarray(flat, dtype=float)
        if flat.shape != red.shape:
            raise ValueError(
                f"flat shape {flat.shape} != frame shape {red.shape} -- "
                "wrong instrument's calibration?")
        red = red / flat
    if badmask is not None:
        badmask = np.asarray(badmask)
        if badmask.shape != red.shape:
            raise ValueError(
                f"bad-pixel mask shape {badmask.shape} != frame shape "
                f"{red.shape} -- wrong instrument's calibration?")
        if badmask.any():
            red = deadpix_fill(red, badmask)
    return red


def _packaged_cal_path(name):
    from importlib.resources import files
    return str(files("keck_ao_estimator") / "data" / name)


_CAL_CACHE = {}


def load_nirc2_calibration(flat_path="default", mask_path="default"):
    """Load a superflat/supermask pair the way the widget does: flat
    floored at 0.2, the known hot pixel (x=1023, y=411) forced bad.

    The K2 summit calibration pair ships with the package and is used by
    default; pass explicit paths to override, or None to skip a piece.
    Returns (flat, mask); either may be None."""
    from astropy.io import fits
    if flat_path == "default":
        flat_path = _packaged_cal_path("superflat.fits.gz")
    if mask_path == "default":
        mask_path = _packaged_cal_path("supermask.fits.gz")
    flat = mask = None
    if flat_path is not None:
        flat = np.maximum(np.asarray(fits.getdata(flat_path), dtype=float), 0.2)
    if mask_path is not None:
        mask = np.asarray(fits.getdata(mask_path)).astype(bool)
        mask[411, 1023] = True      # [y, x]: strehl_widget.pro's hot pixel
    return flat, mask


# ----------------------------------------------------------- orchestration

@dataclass(frozen=True)
class Nirc2StrehlResult:
    strehl: float
    fwhm_mas: float
    wfe_nm: float
    x: float                # detector x (column), IDL convention
    y: float
    peak: float
    flux: float
    sky: float
    saturated: bool
    params: Nirc2FrameParams
    error: str = ""
    crowding: float = 0.0       # annulus contamination as a fraction of flux
    sky_mode: str = "annulus-mean"
    sr_err: float = 0.0         # sky-noise SR uncertainty (lower bound)
    photrad_used_arcsec: float = 0.0    # aperture actually used
    edge_clip: float = 0.0      # fraction of the outer-annulus disc off-array

    # --- PSF-fit neighbour subtraction (opt-in; see psf_fit.py) --------
    # Additive only: every field defaults to the value the untouched
    # default path produces, so a result built without psf_clean is
    # indistinguishable from one built before the feature existed.
    cleaned: bool = False       # neighbours were actually subtracted
    n_subtracted: int = 0
    subtracted_frac: float = 0.0    # neighbour light removed / uncleaned flux
    residual_frac: float = 0.0      # fit residual left in the aperture,
                                    # over what was subtracted (psf_fit
                                    # defines it exactly; conservative)
    epsf_tag: str = ""          # "strict" | "loose" | "uncalibrated"
                                # | "theoretical"
    # D27: which way this measurement is likely to be wrong. Empty when no
    # cleaning ran. Erring LOW is the intended direction; above the
    # validated envelope the sign flips and this says so.
    psf_clean_bias: str = ""
    psf_clean_note: str = ""    # human sentence, including null outcomes
    # matched-aperture control: the same measurement WITHOUT the
    # subtraction, at the SAME photometry radius, so the pair is directly
    # comparable.  It is not necessarily what the default path would have
    # produced, because with auto_radius the radius is chosen on the
    # cleaned frame (a contaminated growth curve is the very thing the
    # cleaning removes).  0.0 when no cleaning was attempted.
    strehl_uncleaned: float = 0.0
    crowding_uncleaned: float = 0.0
    # D20: this star should sit OFF the field map by default and stay
    # REINSERTABLE, exactly like a `field_consistent` outlier.  Set only
    # by the over-contamination refusal -- a star refused for having no
    # neighbours worth subtracting is a perfectly good measurement.
    # `measure_field` keeps it in the returned list, flagged, so the GUI
    # can move it into `_n2_field_dropped` and draw the x; hiding it here
    # would deny the GUI the thing it needs to make the star recoverable.
    psf_clean_excluded: bool = False

    @property
    def crowded(self):
        return self.crowding > CROWDING_WARN_FRAC

    @property
    def edge(self):
        """Photometry footprint clipped by the detector edge: flux and sky
        come from a truncated aperture/annulus, so the SR is suspect (and
        the curve-of-growth auto radius was skipped)."""
        return self.edge_clip > EDGE_CLIP_WARN_FRAC

    @property
    def unphysical(self):
        """Measured SR outside (0, 1] is physically impossible -- the
        photometry is compromised (sky estimate, crowding, saturation,
        or the wrong object)."""
        return self.ok and not 0.0 < self.strehl <= 1.0

    @property
    def ok(self):
        return self.error == ""


def _bias_note(strehl, rep):
    """D27: which way this measurement is likely to be wrong.

    Imported lazily -- `psf_fit` imports `aperture_flux` from this module,
    so a module-level import here is circular."""
    if rep is None or not getattr(rep, "cleaned", False):
        return ""
    from .psf_fit import (
        PSF_FIT_BIAS_SAFE_NOTE, PSF_FIT_BIAS_UNSAFE_NOTE,
        PSF_FIT_SR_VALIDATED_MAX,
    )
    if strehl > PSF_FIT_SR_VALIDATED_MAX:
        return PSF_FIT_BIAS_UNSAFE_NOTE
    note = PSF_FIT_BIAS_SAFE_NOTE
    if getattr(rep, "epsf_tag", "") == "theoretical":
        note += (" Model is THEORETICAL (D26): it carries no static speckle "
                 "or instrument structure, so it subtracts less than "
                 "reality, reinforcing the underestimate.")
    return note


def _failed(params, message):
    return Nirc2StrehlResult(0.0, 0.0, 0.0, -1.0, -1.0, 0.0, 0.0, 0.0,
                             False, params, error=message)


def measure_strehl(image, params=None, header=None, pos=None,
                   background_subtracted=False,
                   photometry_radius_arcsec=NIRC2_PHOTOMETRY_RADIUS_ARCSEC,
                   bg_inner_arcsec=NIRC2_BG_INNER_RADIUS_ARCSEC,
                   bg_outer_arcsec=NIRC2_BG_OUTER_RADIUS_ARCSEC,
                   peak_radius_arcsec=NIRC2_PEAK_RADIUS_ARCSEC,
                   dl_psf=None, robust_sky=False, sky_override=None,
                   auto_radius=False, psf_clean=False, epsf=None,
                   star_catalog=None):
    """Measure the Strehl of a reduced NIRC2 image (calc_and_display.pro).

    `image` should come from reduce_frame (or be otherwise reduced).  Give
    `pos=(x, y)` to measure a chosen star instead of the brightest pixel.
    `background_subtracted=True` reproduces the widget's nbg>0 branch
    (photometry sky forced to 0).  Pass `dl_psf` to reuse a PSF across
    frames sharing camera/stop/wavelength/rotation.

    `psf_clean=True` opts into PSF-fit neighbour subtraction (`psf_fit.py`):
    the star's neighbours are fitted with the field's empirical PSF and
    removed before this same, unchanged photometry runs.  `epsf` and
    `star_catalog` let a caller build both once and share them across a
    field (`measure_field` does).  With `psf_clean=False` -- the default --
    not one arithmetic operation below changes: the tool stays a
    byte-faithful port of the IDL widget (RULES Section 1).
    """
    if params is None:
        if header is None:
            raise ValueError("need params or header")
        params = nirc2_frame_params(header)

    ps = params.plate_scale_mas
    _bgout_px = bg_outer_arcsec * 1000.0 / ps
    # D9: remember the CALLER's radius before auto_radius overwrites it.
    # The post-clean re-optimization below must be bounded by the
    # caller's radius, NOT by the uncleaned optimum -- the optimizer
    # never returns more than its r_max, so bounding it by the shrunken
    # uncleaned value means the cleaned curve can only shrink further,
    # which is the exact opposite of what that block exists to do. The
    # pair stays aperture-matched either way, so SR is not wrong; the
    # cleaning BENEFIT is just systematically understated.
    _caller_photrad_arcsec = photometry_radius_arcsec
    if auto_radius:
        if pos is not None:
            _ax, _ay = float(pos[0]), float(pos[1])
        else:
            # brightest-pixel autofind: locate first, then optimize
            _work0 = sigma_filter3(np.asarray(image, dtype=float))
            _iy, _ix = np.unravel_index(int(_work0.argmax()), _work0.shape)
            _ax, _ay = float(_ix), float(_iy)
        # near the detector edge the growth curve is truncated -- it
        # flattens early, settles at a too-small aperture, and inflates
        # peak/flux.  Keep the caller's fixed radius there (EDGE-flagged
        # below) instead of optimizing on a clipped curve.
        if aperture_edge_clip_frac(np.asarray(image).shape, _ax, _ay,
                                   _bgout_px) <= EDGE_CLIP_WARN_FRAC:
            photometry_radius_arcsec, _conv = optimize_photometry_radius(
                image, _ax, _ay, ps,
                r_max_arcsec=photometry_radius_arcsec,
                bg_inner_arcsec=bg_inner_arcsec,
                bg_outer_arcsec=bg_outer_arcsec)
    photrad = photometry_radius_arcsec * 1000.0 / ps
    radius = int(np.ceil(peak_radius_arcsec * 1000.0 / ps))
    box = 2 * radius + 1

    if dl_psf is None:
        dl_psf = nirc2_dl_psf(params.camname, params.pmsname,
                              params.effwave_um, params.pmrangl_deg,
                              npix=512, daytime=params.daytime,
                              sfp=getattr(params, "sfp", False))
    # matched apertures: strehlone below is computed with the SAME
    # photrad as the star, so an optimized radius stays self-consistent
    ctr = dl_psf.shape[0] // 2
    dlpeak = find_peak(dl_psf, ctr, ctr, box)
    # reference photometry mirrors bmacaper.pro: centroid first, sky = 0
    crad = max(photrad / 2.0, 6.0)
    cx, cy = cntrd(dl_psf, ctr, ctr, crad / 2.0)
    if cx < 0:
        cx, cy = float(ctr), float(ctr)
    refflux = aperture_flux(dl_psf, photrad, cx, cy, skyval=0.0)[0]
    strehlone = dlpeak / refflux

    work = sigma_filter3(np.asarray(image, dtype=float))

    if pos is None:
        iy, ix = np.unravel_index(int(work.argmax()), work.shape)
    else:
        ix, iy = pos
    x, y = cntrd(work, ix, iy, radius)
    if x < 0 or y < 0:
        return _failed(params, "centroid failed; try a bigger aperture")

    # --- PSF-fit neighbour subtraction (opt-in).  Order of operations is
    # D9: clean on the CALLER's radius, then let auto_radius optimize on
    # the CLEANED frame (a contaminated growth curve is the very thing the
    # cleaning removes), then report the uncleaned control at the SAME
    # radius so the pair is comparable.
    _clean = None
    _rep = None
    if psf_clean:
        from .psf_fit import clean_star
        if epsf is None:
            from .epsf import build_epsf
            epsf = build_epsf(
                work, params, catalog=star_catalog,
                photometry_radius_arcsec=photometry_radius_arcsec,
                bg_inner_arcsec=bg_inner_arcsec,
                bg_outer_arcsec=bg_outer_arcsec)
        _clean, _rep = clean_star(
            work, (x, y), params, epsf, catalog=star_catalog,
            photometry_radius_arcsec=photometry_radius_arcsec,
            bg_inner_arcsec=bg_inner_arcsec,
            bg_outer_arcsec=bg_outer_arcsec, robust_sky=robust_sky,
            sky_override=sky_override)
        if not _rep.cleaned:
            _clean = None
        elif auto_radius:
            # D9: the growth curve is optimized on the CLEANED frame.
            # optimize_photometry_radius exists to stop before a
            # neighbour's wing enters the aperture; once the neighbour is
            # gone the curve legitimately settles later, and refusing to
            # use that throws away the benefit.  The default path above is
            # untouched -- this only runs when cleaning actually happened.
            if aperture_edge_clip_frac(work.shape, x, y,
                                       _bgout_px) <= EDGE_CLIP_WARN_FRAC:
                photometry_radius_arcsec, _conv = \
                    optimize_photometry_radius(
                        _clean, x, y, ps,
                        r_max_arcsec=_caller_photrad_arcsec,
                        bg_inner_arcsec=bg_inner_arcsec,
                        bg_outer_arcsec=bg_outer_arcsec)
                photrad = photometry_radius_arcsec * 1000.0 / ps
                # strehlone is aperture-matched, so it must follow
                crad = max(photrad / 2.0, 6.0)
                _cx, _cy = cntrd(dl_psf, ctr, ctr, crad / 2.0)
                if _cx < 0:
                    _cx, _cy = float(ctr), float(ctr)
                strehlone = dlpeak / aperture_flux(
                    dl_psf, photrad, _cx, _cy, skyval=0.0)[0]

    # D44: robust sky and PSF-fit cleaning DOUBLE-CORRECT.  Both exist to
    # remove the same neighbour light -- cleaning takes the neighbours out
    # of the image, and the clipped-median sky then clips an annulus that
    # no longer needs clipping.  Measured on a moderate random field
    # (n=23): psf_clean alone gives a signed median bias of +0.1038, and
    # psf_clean + robust_sky gives +0.2007 -- 93 % worse, and the worst of
    # the four arms tested.  Once cleaning has SUCCEEDED the annulus is
    # already clean, so the plain mean is the right estimator and
    # `robust_sky` is ignored (Eduardo, 2026-08-01).  It is honoured
    # normally whenever cleaning did not run, which is the whole point of
    # the option on an uncleaned frame.
    _robust = robust_sky
    _robust_ignored = False
    if _clean is not None and robust_sky:
        _robust, _robust_ignored = False, True
    if background_subtracted:
        skyval, sky_mode = 0.0, "background-frames"
    elif sky_override is not None:
        skyval, sky_mode = float(sky_override), "picked"
    else:
        skyval = None
        sky_mode = "annulus-clipped-median" if _robust else "annulus-mean"
    _meas = work if _clean is None else _clean
    flux, sky, crowding, sky_sigma, n_ap, n_ann = aperture_flux(
        _meas, photrad, x, y,
        insky_px=bg_inner_arcsec * 1000.0 / ps,
        outsky_px=bg_outer_arcsec * 1000.0 / ps,
        skyval=skyval, robust=_robust)

    peak = find_peak(_meas, x, y, box) - sky
    strehl = (peak / flux) / strehlone

    # sky-noise error propagation (lower bound: source photon noise needs
    # the detector gain, which the headers don't reliably carry):
    #   sigma_F^2 = N_ap*sigma_sky^2 + (N_ap*sigma_sky)^2/N_ann  (flux +
    #   sky-level uncertainty), sigma_peak ~ sigma_sky, and
    #   sigma_S/S = sqrt((sigma_peak/peak)^2 + (sigma_F/F)^2)
    sr_err = 0.0
    if sky_sigma > 0.0 and peak != 0.0 and flux != 0.0:
        var_f = (n_ap * sky_sigma ** 2
                 + (n_ap * sky_sigma) ** 2 / max(n_ann, 1))
        sr_err = abs(strehl) * float(
            np.sqrt((sky_sigma / peak) ** 2 + var_f / flux ** 2))

    xi, yi = int(x), int(y)
    if (xi - 2 * radius + 1 < 0 or yi - 2 * radius + 1 < 0 or
            xi + 2 * radius >= work.shape[1] or yi + 2 * radius >= work.shape[0]):
        return _failed(params, "star too close to the edge of the image")
    subpop = work[yi - 2 * radius + 1:yi + 2 * radius + 1,
                  xi - 2 * radius + 1:xi + 2 * radius + 1]
    saturated = subpop.max() / params.coadds > params.max_counts

    fwhm_mas = radial_profile_fwhm(_meas - sky, x, y) * ps
    wfe_nm = (float(np.sqrt(-np.log(strehl)) * params.effwave_um * 1e3 / (2 * np.pi))
              if 0.0 < strehl < 1.0 else 0.0)

    # matched-aperture control (D9): the same measurement WITHOUT the
    # subtraction, at the SAME radius, so the pair is directly comparable
    sr_unclean = 0.0
    crowd_unclean = 0.0
    if _clean is not None:
        f_u, s_u, c_u, *_ = aperture_flux(
            work, photrad, x, y,
            insky_px=bg_inner_arcsec * 1000.0 / ps,
            outsky_px=bg_outer_arcsec * 1000.0 / ps,
            skyval=skyval, robust=robust_sky)
        p_u = find_peak(work, x, y, box) - s_u
        if f_u != 0.0 and strehlone != 0.0:
            sr_unclean = float((p_u / f_u) / strehlone)
        crowd_unclean = float(c_u)

    return Nirc2StrehlResult(
        strehl=float(strehl), fwhm_mas=float(fwhm_mas), wfe_nm=wfe_nm,
        x=x, y=y, peak=float(peak), flux=flux, sky=sky,
        saturated=bool(saturated), params=params,
        crowding=float(crowding), sky_mode=sky_mode, sr_err=float(sr_err),
        photrad_used_arcsec=float(photometry_radius_arcsec),
        edge_clip=float(aperture_edge_clip_frac(work.shape, x, y,
                                                _bgout_px)),
        cleaned=bool(_rep.cleaned) if _rep is not None else False,
        # D27: name the expected DIRECTION of the residual error, not just
        # its size. Under-subtraction leaves neighbour light in, inflating
        # flux and lowering peak/flux -- an underestimate, which is the
        # intended direction. Above the validated envelope the measured
        # bias flips POSITIVE, so the number is likely too HIGH and the
        # user has to be told that explicitly rather than merely warned
        # that accuracy "degrades".
        psf_clean_bias=_bias_note(strehl, _rep),
        n_subtracted=int(_rep.n_subtracted) if _rep is not None else 0,
        subtracted_frac=float(_rep.subtracted_frac) if _rep else 0.0,
        residual_frac=float(_rep.residual_frac) if _rep else 0.0,
        epsf_tag=str(_rep.epsf_tag) if _rep is not None else "",
        psf_clean_note=(
            (str(_rep.note) + (
                " Robust sky (sigma-clip) was IGNORED for this star: "
                "cleaning already removed the neighbours, and applying "
                "both double-corrects the same light (measured: signed "
                "bias +0.10 cleaned alone vs +0.20 with robust sky). "
                "The plain annulus mean is used on the cleaned frame."
                if _robust_ignored else ""))
            if _rep is not None else ""),
        strehl_uncleaned=sr_unclean, crowding_uncleaned=crowd_unclean,
        psf_clean_excluded=bool(getattr(_rep, "exclude_from_field", False))
        if _rep is not None else False)


def osiris_reduce(raw, background=None, crop=True):
    """OSIRIS reduction (find_strehl.pro, K1 fork): full 2048x2048 frames
    are cropped to the central 1024x1024, smaller reads are centred into
    a 2048 canvas (the upstream corner-anchored embed looks buggy; we
    centre -- a documented deviation that cannot affect full frames);
    then (image - background) and fix_image.  No flat, no mask.

    crop=False keeps full 2048 frames whole (the GUI shows the entire
    detector; the measurement itself is crop-independent) -- the IDL
    tool's crop remains the default for golden fidelity."""
    def _prep(a):
        a = np.asarray(a, dtype=float)
        if a.shape[0] == 2048 and a.shape[1] == 2048:
            return a[512:1536, 512:1536] if crop else a
        out = np.zeros((2048, 2048))
        ys, xs = a.shape
        bly, blx = 1024 - ys // 2, 1024 - xs // 2
        out[bly:bly + ys, blx:blx + xs] = a
        return out

    red = _prep(raw)
    if background is not None:
        red = red - _prep(background)
    return fix_image(red)


def aperture_edge_clip_frac(shape, x, y, r_px):
    """Fraction of the radius-r_px disc around (x, y) that falls OUTSIDE
    the array bounds -- 0.0 for an interior star, approaching 0.5 for a
    star on an edge and 0.75 in a corner.  This is the geometry that
    silently truncates aperture photometry: rmap-based selections just
    stop at the boundary, so flux, sky, and the curve of growth all come
    from a partial footprint."""
    h, w = shape
    r = max(float(r_px), 1.0)
    y0, y1 = int(np.floor(y - r)), int(np.ceil(y + r)) + 1
    x0, x1 = int(np.floor(x - r)), int(np.ceil(x + r)) + 1
    yy, xx = np.mgrid[y0:y1, x0:x1]
    disc = (yy - y) ** 2 + (xx - x) ** 2 <= r * r
    inside = disc & (yy >= 0) & (yy < h) & (xx >= 0) & (xx < w)
    n_disc = int(disc.sum())
    if n_disc == 0:
        return 1.0
    return 1.0 - float(inside.sum()) / n_disc


def optimize_photometry_radius(image, x, y, plate_scale_mas,
                               r_max_arcsec=NIRC2_PHOTOMETRY_RADIUS_ARCSEC,
                               r_min_arcsec=0.25, step_arcsec=0.05,
                               bg_inner_arcsec=NIRC2_BG_INNER_RADIUS_ARCSEC,
                               bg_outer_arcsec=NIRC2_BG_OUTER_RADIUS_ARCSEC,
                               n_settle=3, k_sigma=2.0):
    """Curve-of-growth photometry-radius selection.

    Physics: the aperture flux F(r) of a real point source grows with r
    until the remaining halo flux is below the sky noise; past that
    radius, additional area only adds noise -- and in a crowded field,
    eventually a neighbor's flux.  This picks the smallest radius where
    the next n_settle growth increments are each consistent with zero
    (|dF| < k_sigma x the increment's sky-noise sigma, computed from the
    robust annulus scatter), i.e. where further flux is unmeasurable in
    THIS frame.  Used with matched apertures (the DL reference is
    measured with the same radius), the resulting Strehl is exact for
    the coherent core and can only be biased HIGH by halo flux that is
    itself below the frame's noise floor.  Returns (radius_arcsec,
    converged).  If the growth never settles within r_max (bright halo,
    heavy crowding), r_max is returned with converged=False."""
    image = np.asarray(image, dtype=float)
    ps = plate_scale_mas
    rmap = radius_map(image.shape, x, y)
    ann = image[(rmap >= bg_inner_arcsec * 1000.0 / ps)
                & (rmap <= bg_outer_arcsec * 1000.0 / ps)]
    sky = sigma_clipped_median(ann)
    sky_sigma = 1.4826 * float(np.median(np.abs(ann - sky)))
    radii = np.arange(r_min_arcsec, r_max_arcsec + step_arcsec / 2,
                      step_arcsec)
    flux = []
    npix = []
    for r_as in radii:
        sel = rmap <= r_as * 1000.0 / ps
        flux.append(float((image[sel] - sky).sum()))
        npix.append(int(sel.sum()))

    # criterion 1 -- noise settlement: smallest r whose next n_settle
    # increments are each consistent with zero sky-noise-wise
    for i in range(len(radii) - n_settle):
        settled = True
        for j in range(i, i + n_settle):
            d_n = max(npix[j + 1] - npix[j], 1)
            if (abs(flux[j + 1] - flux[j])
                    > k_sigma * sky_sigma * float(np.sqrt(d_n))):
                settled = False
                break
        if settled:
            return float(radii[i]), True

    # criterion 2 -- contamination upturn: a point source's AZIMUTHALLY
    # AVERAGED growth rate (flux per new pixel) falls with r apart from
    # diffraction rings, which are one radial step wide at these plate
    # scales (ring spacing ~ lambda/D ~ 0.05").  A rise that is both
    # significant AND SUSTAINED over n_settle consecutive steps is a
    # neighbor's wing entering the aperture, not a ring.  Stop at the
    # rate minimum before the first sustained upturn.
    rates = []
    rate_sig = []
    for j in range(len(radii) - 1):
        d_n = max(npix[j + 1] - npix[j], 1)
        rates.append((flux[j + 1] - flux[j]) / d_n)
        rate_sig.append(sky_sigma / float(np.sqrt(d_n)))
    i_min = 0
    run = 0
    for j in range(1, len(rates)):
        comb = k_sigma * float(np.hypot(rate_sig[j], rate_sig[i_min]))
        if rates[j] > rates[i_min] + comb and rates[j] > k_sigma * rate_sig[j]:
            run += 1
            # rings are one radial step wide; two consecutive significant
            # rises is already not a ring
            if run >= max(2, n_settle - 1):
                return float(radii[i_min]), True
        else:
            run = 0
            if rates[j] < rates[i_min]:
                i_min = j
    return float(r_max_arcsec), False


def find_stars(image, n_stars=5, exclude_px=40.0, min_snr=10.0,
               rel_floor=0.001, star_fwhm_px=11):
    """Iterative brightest-star finder: calc_cmdstrehl.pro's NSTARS loop
    (find the peak, record it, blank a disc, repeat) with two floors so
    auto-find neither descends into noise nor harvests the diffraction
    halo of the bright stars: an SNR floor (min_snr x the robust scatter)
    and a relative floor (rel_floor x the brightest star's peak -- halo
    knots scale with the star, so a pure SNR floor cannot reject them;
    knots sit at ~0.03% of their star's peak, so the 0.1% default
    rejects them while spanning 3 dex of real star brightness --
    fainter neighbors can still be added by hand).  Refined positions closer than exclude_px to an
    already-kept star are treated as the same star.  Returns
    [(x, y), ...], brightest first."""
    work = sigma_filter3(image)
    sky = sigma_clipped_median(work)
    sig = 1.4826 * float(np.median(np.abs(work - sky)))
    if sig <= 0.0:
        sig = float(work.std()) or 1.0
    masked = work.copy()
    pts = []
    first_peak = None
    while len(pts) < int(n_stars):
        iy, ix = np.unravel_index(int(masked.argmax()), masked.shape)
        peak = float(masked[iy, ix])
        if peak < sky + min_snr * sig:
            break
        if first_peak is None:
            first_peak = peak
        elif peak - sky < rel_floor * (first_peak - sky):
            break
        masked[radius_map(masked.shape, ix, iy) <= exclude_px] = sky
        x, y = cntrd(work, ix, iy, star_fwhm_px)
        if x < 0 or y < 0:
            continue
        if any(np.hypot(x - px, y - py) <= exclude_px for px, py in pts):
            continue        # halo knot / re-detection of a kept star
        pts.append((float(x), float(y)))
    return pts


def field_consistent(results, k=2.5, sr_floor=0.05, fwhm_floor_frac=0.10):
    """Split field results into (kept, outliers) by robust self-consistency.

    The clip is GRADIENT-AWARE (Eduardo 2026-07-25): on strong-FA
    nights with the LGS asterism off the field centre, the true SR
    field varies a lot across one FOV, and clipping raw values about
    the field MEDIAN systematically discards the best-corrected stars
    -- they are the minority on the asterism side, so the bad majority
    defines the median and the good tail reads as "outliers".  Each
    metric (SR, FWHM) is therefore detrended by a robust plane fit
    v ~ a*x + b*y + c over the star positions, and the k x 1.4826*MAD
    clip runs on the RESIDUALS about that plane: a point far off the
    LOCAL trend in either metric is a photometry artifact (a blend, a
    contaminated aperture, a noise-dominated faint star); a point
    merely riding the anisoplanatic gradient is physics and stays.
    Thresholds keep the old floors (sr_floor absolute SR,
    fwhm_floor_frac x median FWHM) so tight fields don't reject their
    own scatter.  Fields smaller than 5 points pass through untouched;
    fields too small (< 8) or too collinear to constrain a plane fall
    back to the pre-2026-07-25 median clip."""
    results = list(results)
    if len(results) < 5:
        return results, []
    srs = np.array([r.strehl for r in results])
    fws = np.array([r.fwhm_mas for r in results])
    xs = np.array([float(r.x) for r in results])
    ys = np.array([float(r.y) for r in results])

    def _plane_resid(vals):
        """Residuals about a robust plane over (xs, ys), or None when
        the geometry can't constrain one (small n / near-collinear)."""
        if len(vals) < 8:
            return None
        xc = xs - xs.mean()
        yc = ys - ys.mean()
        sv = np.linalg.svd(np.c_[xc, yc], compute_uv=False)
        if sv[-1] < 1e-3 * max(sv[0], 1e-9):     # collinear positions
            return None
        A = np.c_[xc, yc, np.ones_like(xc)]
        resid = vals - A @ np.linalg.lstsq(A, vals, rcond=None)[0]
        for _ in range(2):                       # shed gross outliers,
            mad = np.median(np.abs(resid - np.median(resid)))
            good = np.abs(resid - np.median(resid)) <= max(
                3.0 * 1.4826 * mad, 1e-12)       # then refit the plane
            if good.sum() < 6 or good.all():
                break
            coef = np.linalg.lstsq(A[good], vals[good], rcond=None)[0]
            resid = vals - A @ coef
        return resid

    def _limits(vals, floor):
        resid = _plane_resid(vals)
        if resid is None:                        # median fallback
            resid = vals - float(np.median(vals))
        med = float(np.median(resid))
        sig = 1.4826 * float(np.median(np.abs(resid - med)))
        return resid, med, max(k * sig, floor)

    s_res, s_med, s_thr = _limits(srs, sr_floor)
    f_res, f_med, f_thr = _limits(fws, fwhm_floor_frac * float(np.median(fws)))
    kept, out = [], []
    for r, sr_d, fw_d in zip(results, s_res, f_res):
        if abs(sr_d - s_med) > s_thr or abs(fw_d - f_med) > f_thr:
            out.append(r)
        else:
            kept.append(r)
    return kept, out


def measure_field(image, params, positions=None, n_stars=5,
                  exclude_px=None, dl_psf=None, psf_clean=False, epsf=None,
                  star_catalog=None, **measure_kw):
    """Measure several stars across one frame -> a measured field map.

    positions given -> measure each; else find_stars supplies candidates
    (exclusion radius defaults to the photometry radius).  n_stars=None
    means AUTO: star quality decides the count -- candidates are accepted
    while their propagated sky-noise SR uncertainty stays below
    SR_ERR_MAX, and the run stops after two consecutive quality failures
    (candidates come brightest-first, so fainter ones only get worse).
    One DL PSF is shared across the stars.  Returns the list of OK
    results, in the input/brightness order.  Full-aperture photometry in
    a multi-star field inherits the usual contamination caveats -- the
    crowding metric flags affected stars, and robust-sky / pick-sky /
    auto_radius kwargs pass straight through.

    `psf_clean=True` builds the field's empirical PSF and the deep
    neighbour catalogue ONCE and shares both across the stars, exactly as
    `dl_psf` is shared today: the ePSF is a property of the field, and
    rebuilding it per star would be both slow and inconsistent."""
    photrad_as = measure_kw.get("photometry_radius_arcsec",
                                NIRC2_PHOTOMETRY_RADIUS_ARCSEC)
    if exclude_px is None:
        exclude_px = photrad_as * 1000.0 / params.plate_scale_mas
    if dl_psf is None:
        dl_psf = nirc2_dl_psf(params.camname, params.pmsname,
                              params.effwave_um, params.pmrangl_deg,
                              npix=512, daytime=params.daytime,
                              sfp=getattr(params, "sfp", False))
    # The ePSF and the deep neighbour catalogue are properties of the
    # FIELD, so they are built once here and shared across every star --
    # exactly as dl_psf already is.  Rebuilding per star would be slow and,
    # worse, inconsistent: each star would be cleaned against a slightly
    # different model.
    if psf_clean and (epsf is None or star_catalog is None):
        from .epsf import build_epsf, deep_star_catalog
        _work = sigma_filter3(np.asarray(image, dtype=float))
        if star_catalog is None:
            star_catalog = deep_star_catalog(_work, params)
        if epsf is None:
            epsf = build_epsf(_work, params, catalog=star_catalog,
                              photometry_radius_arcsec=photrad_as)
    if psf_clean:
        measure_kw = dict(measure_kw, psf_clean=True, epsf=epsf,
                          star_catalog=star_catalog)
    # matched apertures: strehlone below is computed with the SAME
    # photrad as the star, so an optimized radius stays self-consistent
    cap = None
    auto = n_stars is None
    if positions is None:
        # rejected stars must not consume map slots: detect a deeper
        # candidate list (the SNR/relative floors still bound it) and
        # keep measuring until n_stars are ACCEPTED, quality runs out
        # (auto), or the candidates do -- the floors are the natural
        # "no more good stars" stop.  Hand-supplied positions are all
        # measured, no cap.
        cap = 30 if auto else int(n_stars)
        positions = find_stars(image,
                               n_stars=max(3 * cap, cap + 20),
                               exclude_px=exclude_px)
    out = []
    queue = list(positions)
    poor_run = 0
    while queue:
        if cap is not None and len(out) >= cap:
            break
        x, y = queue.pop(0)
        r = measure_strehl(image, params=params, pos=(x, y),
                           dl_psf=dl_psf, **measure_kw)
        # a field map is only as honest as its points: failed centroids,
        # SR outside (0, 1], saturated stars, and broken radial profiles
        # (fwhm <= 0) are all rejected
        good = (r.ok and not r.unphysical and not r.saturated
                and r.fwhm_mas > 0.0)
        if good and auto and r.sr_err > SR_ERR_MAX:
            poor_run += 1
            if poor_run >= 2:
                break               # fainter candidates only get worse
            good = False
        elif good:
            poor_run = 0
        if good:
            out.append(r)
        if not queue or (cap is not None and len(out) >= cap):
            # field self-consistency: drop statistical outliers, and let
            # the loop backfill from remaining candidates if any
            out, dropped = field_consistent(out)
            if dropped and queue and cap is not None and len(out) < cap:
                continue
    return out


def measure_osiris_frame(path, background=None, pos=None, sfp=False,
                         **kwargs):
    """Load a raw OSIRIS imager frame, reduce it the K1 tool's way
    (osiris_reduce), and measure.  No saturation ceiling exists in the
    OSIRIS tool, so the result never flags saturated."""
    from astropy.io import fits

    from .osiris import osiris_frame_params

    with fits.open(path) as hdul:
        raw = np.asarray(hdul[0].data, dtype=float)
        header = hdul[0].header
    params = osiris_frame_params(header, sfp=sfp)
    red = osiris_reduce(raw, background=background)
    return measure_strehl(red, params=params, pos=pos,
                          background_subtracted=background is not None,
                          **kwargs)


def measure_nirc2_frame(path, flat="default", mask="default",
                        background=None, pos=None, **kwargs):
    """Convenience: load a raw NIRC2 FITS frame, reduce, measure.

    By default the frame is reduced with the packaged K2 summit
    superflat/supermask (cached after first load); pass arrays to
    override or None to skip flat-fielding / bad-pixel repair."""
    from astropy.io import fits
    if isinstance(flat, str) or isinstance(mask, str):
        if "default" not in _CAL_CACHE:
            _CAL_CACHE["default"] = load_nirc2_calibration()
        dflat, dmask = _CAL_CACHE["default"]
        flat = dflat if isinstance(flat, str) else flat
        mask = dmask if isinstance(mask, str) else mask
    with fits.open(path) as hdul:
        raw = np.asarray(hdul[0].data, dtype=float)
        header = hdul[0].header
    reduced = reduce_frame(raw, background=background, flat=flat, badmask=mask)
    return measure_strehl(reduced, header=header, pos=pos,
                          background_subtracted=background is not None,
                          **kwargs)
