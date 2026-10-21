"""Diffraction-limited PSF for a telescope described by numbers instead of
code: a circular primary with an optional central obscuration and spider
vanes. Instruments whose parameter file sets ``pupil: {model: circular}``
use it (instruments.py); Keck's segmented pupil stays in nirc2_psf.

The sampling follows nirc2_dl_psf: the image is computed on a pupil grid
whose span is at least twice the diameter (no wrap-around at the field
edge) and whose step is at most diameter/100, then block-averaged back to
the detector pixels; the half-pixel phase offset puts the core on pixel
corners for pos=(0, 0), as there.
"""
import numpy as np

__all__ = ["circular_pupil", "circular_dl_psf"]


def circular_pupil(npix, du, diameter_m, obscuration_m=0.0, spiders=0,
                   spider_width_m=0.0, spider_angle_deg=0.0, rotation_deg=0.0):
    """Binary pupil, npix x npix at du metres/pixel: a disc of
    `diameter_m` minus a central disc of `obscuration_m`, cut by `spiders`
    vanes of `spider_width_m` through the centre (evenly spaced, the first
    at spider_angle_deg + rotation_deg from +x)."""
    c = (npix - 1) / 2.0
    y, x = np.mgrid[0:npix, 0:npix]
    x = (x - c) * du
    y = (y - c) * du
    r = np.hypot(x, y)
    pupil = (r <= diameter_m / 2.0) & (r >= obscuration_m / 2.0)
    if spiders and spider_width_m > 0.0:
        for k in range(int(spiders)):
            a = np.radians(spider_angle_deg + rotation_deg + 180.0 * 2 * k
                           / int(spiders))
            along = x * np.cos(a) + y * np.sin(a)
            across = -x * np.sin(a) + y * np.cos(a)
            pupil &= ~((np.abs(across) < spider_width_m / 2.0) & (along >= 0))
    return pupil.astype(np.uint8)


def circular_dl_psf(diameter_m, plate_scale_mas, effwave_um, npix=512,
                    obscuration_m=0.0, spiders=0, spider_width_m=0.0,
                    spider_angle_deg=0.0, rotation_deg=0.0, pos=(0.0, 0.0),
                    return_pupil=False):
    """Monochromatic diffraction-limited PSF of a circular pupil on the
    detector grid (plate_scale_mas per pixel), normalised to unit sum."""
    if diameter_m <= 0 or plate_scale_mas <= 0 or effwave_um <= 0:
        raise ValueError("diameter, plate scale and wavelength must be > 0")
    pscl = plate_scale_mas / 1000.0 / 206265.0          # rad / pixel
    lam_m = effwave_um * 1e-6
    tmp = pscl * 2.0 * diameter_m / lam_m
    rpfac = max(int(2.0 ** np.ceil(np.log2(tmp))), 1)   # span > 2 D
    npix1 = npix * rpfac
    pscl1 = pscl / rpfac
    du = lam_m / (npix1 * pscl1)
    rdfac = max(int(2.0 ** np.ceil(np.log2(du / (diameter_m / 100.0)))), 1)
    npix2 = npix1 * rdfac
    du = lam_m / (npix2 * pscl1)
    pupil = circular_pupil(npix2, du, diameter_m, obscuration_m, spiders,
                           spider_width_m, spider_angle_deg, rotation_deg)
    xx = np.arange(npix2, dtype=float)[None, :]
    yy = np.arange(npix2, dtype=float)[:, None]
    rp = np.asarray(pos, dtype=float) * rpfac - 0.5
    wf = pupil * np.exp(1j * 2.0 * np.pi * (xx * rp[0] + yy * rp[1]) / npix2)
    rpsf = np.abs(np.fft.fftshift(np.fft.fft2(wf))) ** 2
    lo, hi = npix2 // 2 - npix1 // 2, npix2 // 2 + npix1 // 2
    psf = rpsf[lo:hi, lo:hi]
    if rpfac > 1:
        n = psf.shape[0] // rpfac
        psf = psf.reshape(n, rpfac, n, rpfac).mean(axis=(1, 3))
    psf = psf / psf.sum()
    return (psf, pupil) if return_pupil else psf
