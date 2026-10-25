"""Forward-model FWHM for coarse pixels (the OSIRIS spectrograph's spaxels).

The tools' FWHM (find_peak: area above half maximum in an 8x Fourier-
upsampled box) is exact enough at NIRC2's 10 mas, where the K-band core
spans ~4.5 pixels. On 20-100 mas spaxels it is not: the upsampling of a
small even box depends on where the star sits in its spaxel, and the
diffraction-limited PSF alone reads 43.1 mas with its core on a spaxel
corner and 59.4 mas on a spaxel centre at 20 mas (56.9 vs 81.6 at 35 mas,
Kn3), so a ratio to it jumps from cube to cube.

Here the core is fitted instead: the diffraction-limited PSF of the Keck
pupil (nirc2_psf, hexagonal) blurred by a circular Gaussian, shifted to the
star and integrated over each pixel, plus a constant, fitted (scipy
least_squares) to the valid pixels within ``max(2 lambda/D, 2.5 pixels)``
of the star. The FWHM reported is that of the fitted OPTICAL PSF -- the
blurred diffraction-limited PSF itself, no pixel -- as the diameter of the
circle whose area is the area above half maximum (the 2026-10 IDL
definition, applied to the continuous PSF), and the reference is the same
for no blur: the hexagonal-pupil diffraction limit at that wavelength
(Kn3 2.1756 um: 44.7 mas). Neither depends on the pixel phase.

The model has one width parameter, so it measures the core: a halo is
absorbed partly by the constant and partly by the blur. The Gaussian is a
blur kernel on the hex PSF, not a Gaussian PSF fit, and the
diffraction-limited reference is never a Gaussian.
"""
from dataclasses import dataclass

import numpy as np

__all__ = ["ModelFwhm", "fit_model_fwhm", "optical_dl_fwhm_mas",
           "FINE_PIXEL_MAS", "KECK_D_M"]

FINE_PIXEL_MAS = 2.5            # model grid: at most this per fine pixel
KECK_D_M = 10.0                 # lambda/D for the fit radius only
FIT_RADIUS_LD = 2.0             # fit pixels within max(2 lambda/D, ...)
FIT_RADIUS_MIN_PX = 2.5         # ... 2.5 pixels) of the star
MAX_BLUR_LD = 5.0               # Gaussian sigma bound, lambda/D
CLIP_NOTE_FRAC = 0.10           # say so when more of the core is off the field


@dataclass(frozen=True)
class ModelFwhm:
    """One model fit: FWHM of the fitted optical PSF and of the
    diffraction limit (mas), the blur sigma (mas), the star's position
    (pixels), amplitude (total flux of the model), background, the rms
    residual over the peak, the number of pixels fitted, and a note."""
    ok: bool
    fwhm_mas: float = 0.0
    fwhm_dl_mas: float = 0.0
    sigma_mas: float = 0.0
    x: float = 0.0
    y: float = 0.0
    amplitude: float = 0.0
    background: float = 0.0
    resid_frac: float = 0.0
    n_pix: int = 0
    note: str = ""


class _Grid:
    """The diffraction-limited OTF on a fine grid for one pixel scale,
    wavelength and pupil, and the transfer functions of the model."""

    def __init__(self, plate_scale_mas, effwave_um, pmsname, pmrangl_deg,
                 n_coarse):
        from .nirc2_psf import nirc2_dl_psf
        self.ps = float(plate_scale_mas)
        self.q = max(1, int(np.ceil(self.ps / FINE_PIXEL_MAS)))
        self.fine_mas = self.ps / self.q
        n = 2 * int(n_coarse) * self.q          # twice the fit window
        n += n % 2
        self.n = n
        psf = nirc2_dl_psf("narrow", pmsname, effwave_um, pmrangl_deg,
                           npix=n, pos=(0.5, 0.5),
                           plate_scale_mas=self.fine_mas)
        self.otf = np.fft.fft2(np.fft.ifftshift(psf))   # core at the origin
        f = np.fft.fftfreq(n)                   # cycles per fine pixel
        self.fx = f[None, :]
        self.fy = f[:, None]
        self.f2 = self.fx ** 2 + self.fy ** 2
        self.box = np.sinc(self.q * self.fx) * np.sinc(self.q * self.fy)
        # the region optical_fwhm samples: ~1.03 lambda/D for no blur
        self.optical_guess_mas = 1.03 * effwave_um * 1e-6 / KECK_D_M * 206265e3

    def _blur(self, sigma_mas):
        s = float(sigma_mas) / self.fine_mas
        return np.exp(-2.0 * np.pi ** 2 * s * s * self.f2)

    def _evaluate(self, spec, xf, yf):
        """The inverse transform of `spec` at fine-pixel offsets xf (columns)
        and yf (rows) from the origin only: img[j, i] = sum spec[ky, kx]
        exp(2 pi i (fy yf_j + fx xf_i)) / n^2."""
        ex = np.exp(2j * np.pi * self.fx.T * xf[None, :])
        ey = np.exp(2j * np.pi * self.fy * yf[None, :])
        return np.real(ey.T @ spec @ ex) / (self.n * self.n)

    def sample(self, sigma_mas, dx, dy, u, v):
        """Pixel-integrated model (unit total flux) at the pixel centres
        (u, v) (pixels from the window centre), for a star at (dx, dy)
        pixels from it."""
        spec = self.otf * self._blur(sigma_mas) * self.box
        return self._evaluate(spec, self.q * (u - dx),
                              self.q * (v - dy)) * self.q * self.q

    def optical_fwhm(self, sigma_mas):
        """Area-above-half-max FWHM (mas) of the blurred optical PSF, the
        area counted on a grid of 1/200 of the region around the core (a
        coarser grid quantizes it: 2.5 mas reads 43.79 mas at both 2.124
        and 2.176 um)."""
        spec = self.otf * self._blur(sigma_mas)
        guess = np.hypot(self.optical_guess_mas, 2.3548 * float(sigma_mas))
        half = 0.75 * guess / self.fine_mas + 1.0       # fine pixels
        t = np.linspace(-half, half, 201)
        img = self._evaluate(spec, t, t)
        step_mas = (t[1] - t[0]) * self.fine_mas
        n_half = int(np.count_nonzero(img >= img.max() / 2.0))
        return float(np.sqrt(4.0 / np.pi * n_half) * step_mas)


_GRIDS = {}


def _grid(plate_scale_mas, effwave_um, pmsname, pmrangl_deg, n_coarse):
    key = (round(plate_scale_mas, 3), round(effwave_um, 5), pmsname,
           round(pmrangl_deg, 2), int(n_coarse))
    if key not in _GRIDS:
        if len(_GRIDS) > 32:
            _GRIDS.clear()
        _GRIDS[key] = _Grid(plate_scale_mas, effwave_um, pmsname,
                            pmrangl_deg, n_coarse)
    return _GRIDS[key]


def _fit_window(plate_scale_mas, effwave_um):
    lam_d_mas = effwave_um * 1e-6 / KECK_D_M * 206265e3
    r_px = max(FIT_RADIUS_LD * lam_d_mas / plate_scale_mas, FIT_RADIUS_MIN_PX)
    return r_px, 2 * int(np.ceil(r_px)) + 1, lam_d_mas


def optical_dl_fwhm_mas(plate_scale_mas, effwave_um, pmsname="open",
                        pmrangl_deg=0.0):
    """The diffraction-limited optical FWHM (mas) the fit compares with."""
    _r, n, _ld = _fit_window(plate_scale_mas, effwave_um)
    return _grid(plate_scale_mas, effwave_um, pmsname, pmrangl_deg,
                 n).optical_fwhm(0.0)


def fit_model_fwhm(image, x, y, plate_scale_mas, effwave_um, valid=None,
                   pmsname="open", pmrangl_deg=0.0):
    """Fit the blurred, pixel-integrated diffraction-limited PSF to the core
    of the star near (x, y) in `image` (pixels; `valid` marks usable ones)
    and return a ModelFwhm (module docstring). The part of the fit window
    off the image or on invalid pixels is left out (an OSIRIS broadband
    field is 16 spaxels wide); the note says how much of the core was used
    when more than 10 % was not. ok is False, with the reason in note, when
    the star is off the image, fewer than 8 core pixels are usable or the
    fit fails."""
    from scipy.optimize import least_squares
    image = np.asarray(image, dtype=float)
    valid = np.ones(image.shape, bool) if valid is None else np.asarray(valid, bool)
    r_px, n, lam_d = _fit_window(plate_scale_mas, effwave_um)
    h = n // 2
    cx, cy = int(round(x)), int(round(y))
    if not (0 <= cx < image.shape[1] and 0 <= cy < image.shape[0]):
        return ModelFwhm(False, note="the star is off the image")
    # the window may run off the image (the OSIRIS broadband fields are 16
    # spaxels wide): pixels off it are left out like invalid ones, so the
    # fit uses the part of the core on the field
    win = np.zeros((n, n))
    ok = np.zeros((n, n), dtype=bool)
    y0, y1 = max(cy - h, 0), min(cy + h + 1, image.shape[0])
    x0, x1 = max(cx - h, 0), min(cx + h + 1, image.shape[1])
    win[y0 - (cy - h):y1 - (cy - h), x0 - (cx - h):x1 - (cx - h)] = image[y0:y1, x0:x1]
    ok[y0 - (cy - h):y1 - (cy - h), x0 - (cx - h):x1 - (cx - h)] = valid[y0:y1, x0:x1]
    u = np.arange(n, dtype=float) - h
    rr = np.hypot(u[None, :] - (x - cx), u[:, None] - (y - cy))
    core = rr <= r_px
    use = ok & core
    if use.sum() < 8:
        return ModelFwhm(False, note=f"only {int(use.sum())} valid pixels in the core")
    clipped = 1.0 - float(use.sum()) / float(core.sum())
    g = _grid(plate_scale_mas, effwave_um, pmsname, pmrangl_deg, n)
    data = win[use]
    edge = win[ok & (rr > r_px - 1.0)] if (ok & (rr > r_px - 1.0)).any() else data
    b0 = float(np.median(edge))
    a0 = max(float((win[ok] - b0).sum()), 1e-12)
    sig_max = MAX_BLUR_LD * lam_d

    def resid(p):
        amp, dx, dy, sig, b = p
        return amp * g.sample(abs(sig), dx, dy, u, u)[use] + b - data

    p0 = [a0, x - cx, y - cy, 0.3 * lam_d, b0]
    lo = [0.0, -h, -h, 0.0, -np.inf]
    hi = [np.inf, h, h, sig_max, np.inf]
    try:
        sol = least_squares(resid, p0, bounds=(lo, hi), x_scale="jac")
    except Exception as e:
        return ModelFwhm(False, note=f"model fit failed ({type(e).__name__}: {e})")
    if not sol.success:
        return ModelFwhm(False, note=f"model fit did not converge ({sol.message})")
    amp, dx, dy, sig, b = sol.x
    peak = max(float(data.max() - b), 1e-12)
    notes = []
    if clipped > CLIP_NOTE_FRAC:
        notes.append(f"model fit on {100 * (1 - clipped):.0f} % of the core (the rest off the field)")
    if sig >= 0.98 * sig_max:
        notes.append(f"blur at its bound ({MAX_BLUR_LD:g} lambda/D): the core is not resolved")
    note = "; ".join(notes)
    return ModelFwhm(
        True, fwhm_mas=g.optical_fwhm(abs(sig)), fwhm_dl_mas=g.optical_fwhm(0.0),
        sigma_mas=float(abs(sig)), x=float(cx + dx), y=float(cy + dy),
        amplitude=float(amp), background=float(b),
        resid_frac=float(np.sqrt(np.mean(sol.fun ** 2)) / peak),
        n_pix=int(use.sum()), note=note)
