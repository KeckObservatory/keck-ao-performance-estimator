"""NIRC2 diffraction-limited PSF model (port of nirc2pupil.pro/nirc2psf.pro).

The pupil is rasterized from the actual pupil-stop engineering dimensions
(KAON 253) -- six rotated sextant polygons plus their mirror images for the
hex stops, an annulus with spider cuts for INCIRCLE -- then FFT'd into a
monochromatic diffraction-limited PSF at the camera plate scale, following
the original's resampling rules (pupil sampling finer than 0.10 m/pix,
pupil-plane span wider than 12 m, both enforced with power-of-two factors).

Index convention: arrays are numpy [row, col]; the IDL image x axis (first
IDL subscript) maps to numpy axis 1 and y to axis 0.  The sextant rotation
matrix [[-sin, cos], [cos, sin]] is symmetric, which makes the port immune
to the IDL '#' operand-order trap.
"""
import numpy as np

from .nirc2 import (
    NIRC2_DAYTIME_PUPIL_M, NIRC2_OPEN_SECONDARY_M, NIRC2_PMR_ZERO_DEG,
    NIRC2_PMS_M_PER_INCH, NIRC2_PUPIL_STOPS,
)

OSIRIS_WL_PUPIL_M = 11.14   # K1 white-light source pupil diameter (sfp)

# camera plate scales in radians/pixel (nirc2psf.pro). osimg (OSIRIS
# imager, value from Jim Lyke) and sharc are carried over from the IDL
# original: the PSF model is shared across the Keck AO imagers -- only
# the plate scale and pupil stop differ -- so an OSIRIS port reuses this
# machinery unchanged.
_CAMERA_RAD_PER_PIX = {
    "narrow": 0.009942 / 206265.0,
    "medium": 0.019829 / 206265.0,
    "wide": 0.039686 / 206265.0,
    "osimg": 0.02034 / 206265.0,
    "sharc": 0.020 / 206265.0,
    "osiris": 0.00995 / 206265.0,   # post-2016 OSIRIS imager (K1 fork)
}

_COS30 = np.cos(np.radians(30.0))
_SIN30 = 0.5


def _fill_polygon(mask, rows, cols, value):
    """Emulation of IDL's POLYFILLV, derived against an IDL-written oracle
    pupil array (and cross-checked against GDL's polyfillv.pro):

    - vertex coordinates are TRUNCATED to integers first (IDL long());
    - scan lines run through pixel centers (row + 0.5), even-odd parity;
    - a pixel is filled when its center lies in the closed interval
      [x_left, x_right] of a crossing pair: left = ceil(xl - 0.5),
      right = floor(xr - 0.5), i.e. half-integer ties are included on
      both ends.

    On the 512x512 largehex reference pupil this reproduces IDL's pixel
    set except for 15 of 262144 pixels (0.006%, all on the sub-pixel web
    seams between sextants, a tie-handling subtlety IDL's C rasterizer
    resolves differently); the effect on the DL reference ratio is
    +0.17%, i.e. below 0.001 in measured Strehl.
    """
    ny, nx = mask.shape
    fy = np.trunc(np.asarray(rows, dtype=float)).astype(int)
    fx = np.trunc(np.asarray(cols, dtype=float)).astype(int)
    y1s, y2s = fy[:-1].astype(float), fy[1:].astype(float)
    x1s, x2s = fx[:-1].astype(float), fx[1:].astype(float)
    keep = y1s != y2s
    y1s, y2s, x1s, x2s = y1s[keep], y2s[keep], x1s[keep], x2s[keep]
    if y1s.size == 0:
        return
    r0 = max(int(min(y1s.min(), y2s.min())), 0)
    r1 = min(int(max(y1s.max(), y2s.max())), ny - 1)
    lo = np.minimum(y1s, y2s)
    hi = np.maximum(y1s, y2s)
    for y in range(r0, r1 + 1):
        fpy = y + 0.5
        hit = (lo <= fpy) & (fpy < hi)
        if not hit.any():
            continue
        t = (fpy - y1s[hit]) / (y2s[hit] - y1s[hit])
        xs = np.sort(x1s[hit] + t * (x2s[hit] - x1s[hit]))
        for k in range(0, xs.size - 1, 2):
            c0 = max(int(np.ceil(xs[k] - 0.5)), 0)
            c1 = min(int(np.floor(xs[k + 1] - 0.5)), nx - 1)
            if c0 <= c1:
                mask[y, c0:c1 + 1] = value


def nirc2_pupil(npix=256, du=None, pmsname="largehex", pmrangl=0.0,
                daytime=False, sfp=False):
    """Binary NIRC2 pupil image, npix x npix, du meters/pixel at the primary."""
    if du is None:
        du = 2.124e-6 / (npix * 0.00994 / 206265.0)
    pmsname = pmsname.strip().lower()
    d = NIRC2_PUPIL_STOPS.get(pmsname)
    if d is None:
        d = NIRC2_PUPIL_STOPS["largehex"]
        pmsname = "largehex"

    ctr = npix / 2.0 - 0.5
    ax = np.arange(npix) - ctr
    r = np.hypot(ax[None, :], ax[:, None])   # distance in pixels

    pupil = np.zeros((npix, npix), dtype=np.uint8)
    if sfp:
        # K1 white-light source (OSIRIS fork's nirc2pupil): plain circle
        pupil[r * du < OSIRIS_WL_PUPIL_M / 2.0] = 1
        return pupil
    if daytime:
        pupil[r * du <= NIRC2_DAYTIME_PUPIL_M / 2.0] = 1
        return pupil

    scale = du * NIRC2_PMS_M_PER_INCH       # inches per pupil-image pixel
    angles = np.radians(60.0 * np.arange(6) - pmrangl + NIRC2_PMR_ZERO_DEG)

    if pmsname == "incircle":
        r_inch = r * du * NIRC2_PMS_M_PER_INCH
        pupil[(r_inch < d[0]) & (r_inch > d[1])] = 1
        # spider vanes: thin rectangles cut out of the annulus
        v = np.array([[-d[2], 0.0], [d[2], 0.0], [d[2], d[0] * 1.1],
                      [-d[2], d[0] * 1.1], [-d[2], 0.0]]).T   # (2, 5)
        for a in angles:
            m = np.array([[-np.sin(a), np.cos(a)], [np.cos(a), np.sin(a)]])
            rv = npix / 2.0 + (m @ v) / scale
            _fill_polygon(pupil, rv[0], rv[1], 0)
        return pupil

    # hex stops: one sextant's vertices in inches (nirc2pupil.pro), mirrored
    s = (d[0] - d[1]) / _COS30              # segment edge length
    v0 = np.array([
        [d[5], d[4] / _COS30 - d[5] * _SIN30],
        [d[5], d[2] / _COS30 + d[5] * _SIN30],
        [s * _COS30, d[2] / _COS30 + s * _SIN30],
        [2 * s * _COS30, d[2] / _COS30],
        [3 * s * _COS30, d[2] / _COS30 + s * _SIN30],
        [d[0] * _SIN30, d[0] * _COS30],
        [d[4] * _SIN30, d[4] * _COS30],
        [d[5], d[4] / _COS30 - d[5] * _SIN30],
    ]).T                                    # (2, 8)
    v1 = v0 * np.array([[-1.0], [1.0]])     # mirror across the y axis

    for a in angles:
        m = np.array([[-np.sin(a), np.cos(a)], [np.cos(a), np.sin(a)]])
        for v in (v0, v1):
            rv = npix / 2.0 + (m @ v) / scale
            _fill_polygon(pupil, rv[0], rv[1], 1)

    if pmsname == "open":                   # cut out circular secondary
        pupil[r * du < NIRC2_OPEN_SECONDARY_M] = 0
    if pmsname == "fixedhex":               # 12 m / 3 m annular vignette
        pupil[(r * du >= 6.0) | (r * du < 1.5)] = 0
    return pupil


def _block_mean(a, fac):
    """IDL REBIN downsample: mean over fac x fac blocks."""
    if fac == 1:
        return a
    n = a.shape[0] // fac
    return a.reshape(n, fac, n, fac).mean(axis=(1, 3))


def nirc2_dl_psf(camname="narrow", pmsname="largehex", effwave_um=2.1245,
                 pmrangl=0.0, npix=512, pos=(0.0, 0.0), daytime=False,
                 sfp=False, return_pupil=False):
    """Diffraction-limited monochromatic NIRC2 PSF, normalized to unit sum.

    Port of nirc2psf.pro.  The half-pixel phase-ramp offset of the original
    is kept, so the PSF core sits on pixel corners for pos=(0, 0) exactly
    as the IDL reference images do.
    """
    camname = camname.strip().lower()
    pscl = _CAMERA_RAD_PER_PIX.get(camname, _CAMERA_RAD_PER_PIX["narrow"])
    lam_m = effwave_um * 1e-6

    tmp = pscl * 12.0 / lam_m
    rpfac = max(int(2.0 ** np.ceil(np.log2(tmp))), 1)   # span > 12 m
    npix1 = npix * rpfac
    pscl1 = pscl / rpfac
    du = lam_m / (npix1 * pscl1)
    rdfac = max(int(2.0 ** np.ceil(np.log2(du / 0.10))), 1)  # du < 0.10 m
    npix2 = npix1 * rdfac
    du = lam_m / (npix2 * pscl1)

    pupil = nirc2_pupil(npix=npix2, du=du, pmsname=pmsname, pmrangl=pmrangl,
                        daytime=daytime, sfp=sfp)

    xx = np.arange(npix2, dtype=float)[None, :]   # IDL x -> numpy axis 1
    yy = np.arange(npix2, dtype=float)[:, None]
    rpos = np.asarray(pos, dtype=float) * rpfac - 0.5

    def _one_psf(rp):
        phase = 2.0 * np.pi * (xx * rp[0] + yy * rp[1]) / npix2
        wf = pupil * np.exp(1j * phase)
        rpsf = np.abs(np.fft.fftshift(np.fft.fft2(wf))) ** 2
        lo, hi = npix2 // 2 - npix1 // 2, npix2 // 2 + npix1 // 2
        return _block_mean(rpsf[lo:hi, lo:hi], rpfac)

    if not daytime:
        psf = _one_psf(rpos)
    else:
        # PCU single-mode-fiber source: Gaussian mode field convolved in
        # via a weighted 7 x 7 grid of sub-pixel offsets (nirc2psf.pro)
        mfd_um = 10.4 if effwave_um <= 2.0 else 12.6
        mfd_mas = mfd_um / 0.727
        w0_pix = (mfd_mas / (pscl * 206265.0 * 1e3)) / 2.0
        psf = np.zeros((npix, npix))
        for k1 in range(7):
            for k2 in range(7):
                dist = np.array([(k1 - 3.0) / 1.5, (k2 - 3.0) / 1.5])
                weight = np.exp(-2.0 * (dist ** 2).sum() / w0_pix ** 2)
                psf += weight * _one_psf(rpos + dist)

    psf /= psf.sum()
    return (psf, pupil) if return_pupil else psf
