"""ORKID visible camera front end for the Strehl tool.

ORKID (the K2 AO bench visible camera, Andor Marana sCMOS) shares the measurement core with NIRC2 and OSIRIS
(``measure_strehl``); only the front end differs, as for OSIRIS:

- **Plate scale 6.60 mas/pixel**, measured on the bench 2026-10-14 by moving the PCU source 0.5 mm in x and in y
  (104.2 px both ways at 0.727 mm per arcsec; the pixels are square). The 2022 lab value was 6.65 (ORKID
  performance slides, slide 12, 2022-06-20; design 6.694 mas = Nyquist at 650 nm for 10 m). The camera server's
  PLATESCALE (0.0182 arcsec) and the FITS WCS built from it are wrong and never read.
- **Wavelength from the two filter wheels** (FWPOS, PWPOS header cards) through the ORKID filter table:
  narrowband centres, wideband band midpoints. The FW element wins when both carry one. Open/open has no
  wavelength and is refused.
- **Reduction:** cube planes are averaged, an optional background frame subtracted, and bad pixels
  repaired with ``fix_image``. There is no flat and no bad-pixel map.
- **Reference PSF built here**, not by ``nirc2_dl_psf``, which looks the plate scale up by NIRC2 camera name.
  The same algorithm is used: the pupil from ``nirc2_pupil``, sampled finely enough, and for bench (daytime)
  frames the 11.0 m circular PCU pupil convolved with a Gaussian source blur, given as an equivalent fiber mode
  field diameter ``mfd_um`` at the PCU focus (0.727 mm per arcsec; Gaussian sigma = MFD / 4).
- **The bench blur is measured, not the fiber's spec.** The PCU fiber is SMF-28 (single-mode only above 1260 nm,
  so few-mode at 630-940 nm, LP01 MFD about 7 um there; NIRC2's 10.4 um is its 1550 nm value). A non-redundant
  mask image (FW_H_A + PW_NRM, 2026-10-14) gave fringe visibility against baseline with a total Gaussian blur of
  sigma 5.8 mas, the same with the AO loops open and closed; 1 ms frames show only 1-1.5 mas of image motion. The
  fiber accounts for about 3 mas; the rest is detector charge diffusion or a magnified fiber image (open). The
  default ``mfd_um`` (16.9 um) is the equivalent of that total blur, so the bench reference contains everything
  the AO cannot correct. On-sky frames get no blur; if part of it is the detector, on-sky Strehls are optimistic
  by up to the same factor (about 1.3 at 900 nm).
- **Smaller default apertures** (photometry 0.35 arcsec, sky annulus 0.40-0.48 arcsec, peak box 0.1 arcsec):
  the image sharpening frames are 150 x 150 pixels (1.0 arcsec). The star and the reference use the same
  aperture, so the Strehl stays consistent at any radius that contains the core.
- **Saturation ceiling 60000 counts** in HDR mode (16-bit, 65535 maximum; the linearity limit is not
  documented).

The pupil angle for on-sky frames is not known for ORKID and defaults to 0; the diffraction-limited peak does
not depend on the rotation, so the Strehl is insensitive to it.

Engine module: edited here and copied verbatim into PyAO (kaotools/strehl_tool/_vendored_estimator) by
sr_estimator/sync_pyao_strehl_tool.py. The instrument parameter file data/instruments/orkid.yaml selects it
(params, reduce and pupil model "orkid"; instruments.py).
"""
import re
from datetime import datetime

import numpy as np

from .image_strehl import fix_image, measure_strehl
from .nirc2 import Nirc2FrameParams, opt_header_float
from .nirc2_psf import nirc2_pupil

ORKID_PLATE_SCALE_MAS = 6.60                 # bench PCU moves 2026-10-14 (2022 lab: 6.65)
ORKID_MAX_COUNTS = 60000.0
ORKID_PHOTOMETRY_RADIUS_ARCSEC = 0.35
ORKID_BG_INNER_RADIUS_ARCSEC = 0.40
ORKID_BG_OUTER_RADIUS_ARCSEC = 0.48
ORKID_PEAK_RADIUS_ARCSEC = 0.1
#: Total bench source blur measured with the NRM (Gaussian sigma, mas), fiber plus detector or optics (2026-10-14).
ORKID_BENCH_BLUR_SIGMA_MAS = 5.8
#: Fiber mode field diameter (um at the PCU focus) equivalent to that blur: MFD = 4 sigma x 0.727 um/mas.
ORKID_DEFAULT_MFD_UM = round(4 * ORKID_BENCH_BLUR_SIGMA_MAS * 0.727, 1)
ORKID_PMS_DAYTIME = "open"

# Filter wheel position -> wavelength in microns (ORKID filter table, 2026-10-09). Narrowband: [OI] 630/10,
# H-alpha continuum 640/6, H-alpha 656/5, Jovian methane 889/18, CT940/20 940/20 nm. Wideband: midpoint of
# R 554-696, i 698-843 and z 830-1000 nm.
ORKID_FILTER_WAVELENGTH_UM = {
    "FW_BESSELL_R": 0.625, "FW_SLOAN_I": 0.7705, "FW_O_I": 0.630, "FW_H_A": 0.656, "FW_H_A_CONT": 0.640,
    "PW_SLOAN_Z": 0.915, "PW_JOVIAN": 0.889, "PW_CT940_20BP": 0.940,
}

# ORKID camera server file names: <PREFIX><SETNR:4>_<FILENR:4>.fits, e.g. test_0014_0000.fits
ORKID_FRAME_PATTERN = re.compile(r"^(.*?)(\d{4})_(\d{4})\.fits$")


def is_orkid_header(header):
    """True for a frame written by the ORKID camera server (CAMNAME = orkid; there is no INSTRUME card)."""
    return str(header.get("CAMNAME", "")).strip().lower() == "orkid"


def orkid_wavelength_um(fwpos, pwpos):
    """Wavelength in microns of a filter wheel combination, or None when neither element carries one."""
    for name in (str(fwpos).strip().upper(), str(pwpos).strip().upper()):
        if name in ORKID_FILTER_WAVELENGTH_UM:
            return ORKID_FILTER_WAVELENGTH_UM[name]
    return None


def orkid_header_on_sky(header):
    """True when the header says the telescope was tracking (AXESTAT = tracking): an on-sky frame.

    The camera server writes no PCU card. Daytime bench frames have the telescope parked (2026-10-12: AXESTAT
    'not controlling', GUIDING FALSE, DOMESTAT standby); sky frames have AXESTAT and DOMESTAT 'tracking'
    (2026-10-23, 10 Lac). A non-sidereal target (DTRACK enabled) also tracks.
    """
    return str(header.get("AXESTAT", "")).strip().lower() == "tracking"


def orkid_frame_params(header, daytime=True, pmrangl_deg=0.0):
    """Measurement parameters from an ORKID FITS header.

    :param header: ORKID FITS header.
    :param bool daytime: True for the bench (PCU fiber) source, False on sky. ORKID headers do not say which.
    :param float pmrangl_deg: Pupil rotation for on-sky frames (unknown for ORKID, 0 by default).
    :return: Parameters for ``measure_strehl``.
    :rtype: Nirc2FrameParams
    """
    fwpos = str(header.get("FWPOS", "")).strip()
    pwpos = str(header.get("PWPOS", "")).strip()
    effwave = orkid_wavelength_um(fwpos, pwpos)
    if effwave is None:
        raise ValueError(f"ORKID filters {fwpos}/{pwpos} carry no wavelength; a Strehl needs a filter.")
    utc = None
    date_obs, tval = header.get("DATE-OBS"), header.get("UTC", header.get("UT"))
    if date_obs and tval:
        try:
            utc = datetime.fromisoformat(f"{date_obs}T{str(tval).split('.')[0]}")
        except ValueError:
            utc = None
    return Nirc2FrameParams(
        camname="orkid",
        pmsname=ORKID_PMS_DAYTIME,
        effwave_um=effwave,
        pmrangl_deg=float(pmrangl_deg),
        coadds=1,
        max_counts=ORKID_MAX_COUNTS,
        daytime=bool(daytime),
        plate_scale_mas=ORKID_PLATE_SCALE_MAS,
        utc=utc,
        object_name=str(header.get("TARGNAME", "")).strip(),
        az_deg=opt_header_float(header, "AZ"),
        el_deg=opt_header_float(header, "EL"),
        airmass=opt_header_float(header, "AIRMASS"),
        filter_name=f"{fwpos}+{pwpos}",
    )


def orkid_reduce(raw, background=None):
    """Reduce an ORKID frame: average the planes of a cube, subtract an optional background, repair pixels.

    The camera server writes single frames as one-plane cubes and multi-frame exposures as cubes; the planes are
    averaged like coadds.

    :param raw: Frame or cube as read from the FITS file (BZERO applied).
    :param background: Optional background frame or cube of the same image size.
    :return: Reduced 2D image.
    """
    red = _mean_plane(raw)
    if background is not None:
        red = red - _mean_plane(background)
    return fix_image(red)


def _mean_plane(data):
    """Return a 2D image: the frame itself, or the mean over the planes of a cube."""
    data = np.squeeze(np.asarray(data, dtype=float))
    return data.mean(axis=0) if data.ndim == 3 else data


def _block_mean(a, f):
    """Mean over f x f blocks."""
    if f == 1:
        return a
    n = a.shape[0] // f
    return a[:n * f, :n * f].reshape(n, f, n, f).mean(axis=(1, 3))


def orkid_dl_psf(effwave_um, pixel_scale_mas=ORKID_PLATE_SCALE_MAS, pmsname=ORKID_PMS_DAYTIME, pmrangl=0.0,
                 npix=512, daytime=True, mfd_um=ORKID_DEFAULT_MFD_UM):
    """Diffraction-limited ORKID PSF normalized to unit sum.

    Same algorithm as the NIRC2 model (``nirc2_dl_psf``): an oversampled pupil FFT binned back to detector
    pixels with the half-pixel offset of the IDL reference, and for bench frames the fiber mode applied as a
    weighted 7 x 7 grid of sub-pixel offsets. The plate scale and the fiber mode field diameter are parameters.

    :param float effwave_um: Wavelength in microns.
    :param float pixel_scale_mas: Detector pixel scale in mas.
    :param str pmsname: Pupil stop name for on-sky frames (``nirc2_pupil``).
    :param float pmrangl: Pupil rotation in degrees for on-sky frames.
    :param int npix: Output size in pixels.
    :param bool daytime: Bench PCU source (11.0 m circular pupil and fiber mode) instead of the sky.
    :param float mfd_um: Source blur as an equivalent fiber mode field diameter in microns at the PCU focus
        (0.727 mm per arcsec); the default is the measured bench blur.
    :return: PSF, npix x npix.
    """
    pscl = pixel_scale_mas / 1000.0 / 206265.0
    lam_m = effwave_um * 1e-6
    rpfac = max(int(2.0 ** np.ceil(np.log2(pscl * 12.0 / lam_m))), 1)
    npix1 = npix * rpfac
    pscl1 = pscl / rpfac
    du = lam_m / (npix1 * pscl1)
    rdfac = max(int(2.0 ** np.ceil(np.log2(du / 0.10))), 1)
    npix2 = npix1 * rdfac
    du = lam_m / (npix2 * pscl1)
    pupil = nirc2_pupil(npix=npix2, du=du, pmsname=pmsname, pmrangl=pmrangl, daytime=daytime)

    xx = np.arange(npix2, dtype=float)[None, :]
    yy = np.arange(npix2, dtype=float)[:, None]

    def one_psf(rp):
        """PSF for a sub-pixel offset rp (pixels)."""
        phase = 2.0 * np.pi * (xx * rp[0] + yy * rp[1]) / npix2
        rpsf = np.abs(np.fft.fftshift(np.fft.fft2(pupil * np.exp(1j * phase)))) ** 2
        lo, hi = npix2 // 2 - npix1 // 2, npix2 // 2 + npix1 // 2
        return _block_mean(rpsf[lo:hi, lo:hi], rpfac)

    rpos = np.array([0.0, 0.0]) * rpfac - 0.5
    if not daytime or mfd_um <= 0:
        psf = one_psf(rpos)
    else:
        w0_pix = (mfd_um / 0.727) / pixel_scale_mas / 2.0
        psf = np.zeros((npix, npix))
        for k1 in range(7):
            for k2 in range(7):
                dist = np.array([(k1 - 3.0) / 1.5, (k2 - 3.0) / 1.5])
                psf += np.exp(-2.0 * (dist ** 2).sum() / w0_pix ** 2) * one_psf(rpos + dist)
    return psf / psf.sum()


def measure_orkid_frame(path, background=None, pos=None, daytime=True, mfd_um=ORKID_DEFAULT_MFD_UM,
                        photometry_radius_arcsec=ORKID_PHOTOMETRY_RADIUS_ARCSEC,
                        bg_inner_arcsec=ORKID_BG_INNER_RADIUS_ARCSEC, bg_outer_arcsec=ORKID_BG_OUTER_RADIUS_ARCSEC,
                        peak_radius_arcsec=ORKID_PEAK_RADIUS_ARCSEC, dl_psf=None, **kwargs):
    """Load an ORKID frame, reduce it and measure its Strehl ratio.

    :param str path: ORKID FITS file.
    :param background: Optional background frame (array) to subtract.
    :param pos: Optional (x, y) of the star; the brightest pixel otherwise.
    :param bool daytime: Bench PCU source (default) or sky.
    :param float mfd_um: Source blur (equivalent mode field diameter) for bench frames; default measured.
    :param dl_psf: Precomputed reference PSF, to reuse across frames with the same filter.
    :return: The measurement result of ``measure_strehl``.
    """
    from astropy.io import fits

    with fits.open(path) as hdul:
        raw = np.asarray(hdul[0].data, dtype=float)
        header = hdul[0].header
    if not is_orkid_header(header):
        raise ValueError(f"{path} is not an ORKID frame (CAMNAME={header.get('CAMNAME')!r}).")
    params = orkid_frame_params(header, daytime=daytime)
    if dl_psf is None:
        dl_psf = orkid_dl_psf(params.effwave_um, params.plate_scale_mas, params.pmsname, params.pmrangl_deg,
                              daytime=params.daytime, mfd_um=mfd_um)
    return measure_strehl(orkid_reduce(raw, background), params=params, pos=pos,
                          background_subtracted=background is not None,
                          photometry_radius_arcsec=photometry_radius_arcsec, bg_inner_arcsec=bg_inner_arcsec,
                          bg_outer_arcsec=bg_outer_arcsec, peak_radius_arcsec=peak_radius_arcsec,
                          dl_psf=dl_psf, **kwargs)
