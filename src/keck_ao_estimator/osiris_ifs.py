"""OSIRIS spectrograph (IFU) datacubes for the Strehl tools: FWHM first,
SR relative.

The summit's online reduction pipeline (ORP) builds a cube for nearly every
spectrograph frame, about 5 s after the frame, in
``/s/sdata1100/<account>/<YYMMDD>/SPEC/ORP/s<date>_a<set><frame>_<filter>_<scale>.fits``:
the primary HDU is the flux cube, FITS axes (wavelength, x, y) -- numpy
``(y, x, lambda)`` -- with CTYPE1 = WAVE in nm, then a noise extension and
the data-quality extension (9 = good). The raw detector frames in
``SPEC/raw`` are spectra, not images, and are refused (the tool does not
rebuild cubes: that needs the pipeline's rectification matrices).

The field is (lenslets across x scale) x (64 x scale): 16 lenslets in the
broadband filters, 48 in the narrowband ones, at 20 / 35 / 50 / 100 mas
(KAPE's OSIRIS_SPEC_LENSLETS). That is too small for the tools' 1"
photometry and 1.2-1.4" sky annulus, and the growth curve has not converged
where the field ends (HD146606, Kn3 35 mas, 2026-07-14: SR 0.40 at 0.25",
0.29 at 0.8", no plateau). So the SR here is a RELATIVE number -- one
aperture (default 0.25", sky 0.30-0.40") kept fixed to compare cubes, or
wavelengths, with each other -- and not the 1" convention of NIRC2 or the
OSIRIS imager. The FWHM needs only the core and is the measurement (Eduardo
2026-10-25: "getting a good FWHM is still a useful tool"). On spaxels the
tools' FWHM (find_peak) depends on where the star sits in its spaxel (the
diffraction limit alone reads 43.1-59.4 mas at 20 mas), so the FWHM here is
the model fit of model_fwhm.py: the blurred hexagonal-pupil PSF integrated
over the spaxels, its optical FWHM compared with the diffraction limit's at
the same wavelength (Kn3: 44.4 mas), independent of the spaxel phase. Should
the fit fail, the tools' FWHM is kept and the note says so.

Wavelength: a cube is measured whole (the mean over the good channels) or
in bins of channels; the reference PSF is monochromatic at the bin's
central wavelength from the cube's own axis (not a filter table). Spaxels
with no good channel (the field's trapezoid corners) are left out: the sky
is the sigma-clipped median of the valid spaxels in the annulus, and the
photometry footprint's share on invalid spaxels is the EDGE fraction.
Pupil: the OSIRIS imager tool's -- the open Keck pupil at 38 deg
(osiris.py) -- so the two OSIRIS modes share one reference.

An A-B cube (the ORP subtracted the dither partner: osiris_orp) also holds
the star NEGATIVE where it was in the partner, 0.3-1" away. A negative image
IFS_MIN_PEAK_SIGMA below the field and within reach of the sky annulus is
left out (spaxels within IFS_NEG_MASK_ARCSEC of it are treated as invalid
for the sky, the photometry and the model fit), and the note says so; when
the mask reaches into the photometry aperture the note gives the share
masked: the SR is biased high, and not usable above 10 %.

A cube without its star (dithered off the field) is refused: the star's
spaxel must stand IFS_MIN_PEAK_SIGMA (8) robust sigma above the median of
the valid spaxels (20 ORP cubes of 2026-07-14 / 10-20 / 10-21: no star
1.6-4.7 sigma, stars 10.7-766 sigma). The model fit uses the part of its
window on the field, so a star a spaxel or two from the edge of a 16-spaxel
broadband field still gets its FWHM (the note gives the share of the core
used).
"""
from dataclasses import dataclass, replace

import numpy as np

__all__ = ["IFS_GOOD_DQ", "IFS_MIN_PEAK_SIGMA", "IFS_NEG_MASK_ARCSEC", "IFS_PHOTOMETRY_ARCSEC", "IFS_SKY_INNER_ARCSEC",
           "IFS_SKY_OUTER_ARCSEC", "IFS_PEAK_ARCSEC", "IFS_DEFAULT_BIN",
           "IfsCube", "is_ifs_cube", "is_ifs_raw", "spaxel_scale_mas",
           "wavelengths_um", "channel_bins", "bin_label", "read_cube", "collapse",
           "display_image", "ifs_frame_params", "ifs_dl_psf", "dl_fwhm_mas",
           "measure_ifs", "ifs_sky_mode"]

IFS_GOOD_DQ = 9                 # ORP data-quality value of a good voxel
IFS_PHOTOMETRY_ARCSEC = 0.25    # default relative-SR aperture (fits 20 mas)
IFS_SKY_INNER_ARCSEC = 0.30
IFS_SKY_OUTER_ARCSEC = 0.40
IFS_PEAK_ARCSEC = 0.10
IFS_DEFAULT_BIN = 25            # channels per wavelength plane
IFS_MIN_SKY_SPAXELS = 8         # fewer valid annulus spaxels: sky = 0
IFS_NEG_MASK_ARCSEC = 0.15      # an A-B cube's negative star: spaxels this
                                # close to it are left out of the SR and fit
IFS_NEG_APERTURE_MAX = 0.10     # more of the SR aperture masked: SR not usable
IFS_MIN_PEAK_SIGMA = 8.0        # the star's spaxel above the median of the
                                # valid ones, in robust sigma (MAD): else
                                # there is no star in the field
_PAD_MARGIN_PX = 4


@dataclass(frozen=True)
class IfsCube:
    """One spectrograph cube: flux (y, x, lambda), good voxels, header and
    the wavelength of every channel (microns)."""
    data: np.ndarray
    good: np.ndarray
    header: object
    wavelengths_um: np.ndarray

    @property
    def nchan(self):
        """Number of wavelength channels."""
        return self.data.shape[2]


def is_ifs_cube(header):
    """True for an OSIRIS spectrograph datacube (ORP / DRP output)."""
    return (int(header.get("NAXIS", 0) or 0) == 3
            and str(header.get("CTYPE1", "") or "").strip().upper().startswith("WAVE"))


def is_ifs_raw(header):
    """True for a raw spectrograph detector frame (spectra, not an image)."""
    return (str(header.get("INSTR", "") or "").strip().lower() == "spec"
            and not is_ifs_cube(header))


def spaxel_scale_mas(header):
    """The spaxel scale (SSCALE, arcsec) in mas."""
    try:
        s = float(str(header.get("SSCALE", "")).strip())
    except ValueError:
        s = 0.0
    if s <= 0:
        raise ValueError("no spaxel scale (SSCALE) in the cube header")
    return s * 1000.0


def wavelengths_um(header):
    """Wavelength of every channel (microns) from the cube's WAVE axis."""
    n = int(header["NAXIS1"])
    crval = float(header["CRVAL1"])
    cdelt = float(header.get("CDELT1", header.get("CD1_1", 0.0)))
    crpix = float(header.get("CRPIX1", 1.0))
    unit = str(header.get("CUNIT1", "nm") or "nm").strip().lower()
    scale = {"nm": 1e-3, "um": 1.0, "micron": 1.0, "microns": 1.0,
             "angstrom": 1e-4, "m": 1e6}.get(unit, 1e-3)
    return (crval + cdelt * (np.arange(n) + 1.0 - crpix)) * scale


def channel_bins(nchan, width):
    """[(lo, hi), ...] channel ranges of `width` channels (the last may be
    shorter; one under half a bin is merged into the one before)."""
    nchan, width = int(nchan), max(int(width), 1)
    edges = list(range(0, nchan, width)) + [nchan]
    bins = list(zip(edges[:-1], edges[1:]))
    if len(bins) > 1 and bins[-1][1] - bins[-1][0] < width / 2:
        bins[-2] = (bins[-2][0], nchan)
        bins.pop()
    return bins


def bin_label(wavelengths_um, lo, hi):
    """'λc 2.1612 µm (25 ch)': the central wavelength of channels lo:hi (the
    mean of theirs, the wavelength the bin is measured at) and how many."""
    wl = float(np.mean(np.asarray(wavelengths_um)[lo:hi]))
    return f"λc {wl:.4f} µm ({hi - lo} ch)"


def read_cube(path):
    """Read an ORP/DRP cube: (y, x, lambda) flux, good voxels (data quality
    9, else finite), header, wavelengths."""
    from astropy.io import fits
    with fits.open(path) as hdul:
        header = hdul[0].header
        if not is_ifs_cube(header):
            raise ValueError("not an OSIRIS spectrograph cube (CTYPE1 is not WAVE)")
        data = np.asarray(hdul[0].data, dtype=np.float32)
        good = np.isfinite(data)
        if len(hdul) > 2 and hdul[2].data is not None \
                and hdul[2].data.shape == data.shape:
            good &= np.asarray(hdul[2].data) == IFS_GOOD_DQ
    return IfsCube(data=data, good=good, header=header,
                   wavelengths_um=wavelengths_um(header))


def collapse(cube, lo=0, hi=None):
    """(image, valid): the mean of the good channels lo:hi of every spaxel,
    and which spaxels have any good channel there."""
    hi = cube.nchan if hi is None else hi
    d = cube.data[..., lo:hi]
    g = cube.good[..., lo:hi]
    n = g.sum(axis=2)
    s = np.where(g, d, 0.0).sum(axis=2, dtype=float)
    valid = n > 0
    image = np.where(valid, s / np.maximum(n, 1), 0.0)
    return image, valid


def display_image(image, valid):
    """The image with invalid spaxels set to the median of the valid ones
    (a neutral grey on display)."""
    fill = float(np.median(image[valid])) if valid.any() else 0.0
    return np.where(valid, image, fill)


def ifs_frame_params(header, wl_um):
    """Measurement parameters for a cube (or a wavelength bin of one) at
    wavelength `wl_um`: the OSIRIS header values (time, target, telescope)
    with the spaxel scale, the spectrograph filter and the open Keck pupil
    at 38 deg of the OSIRIS imager tool."""
    from .osiris import osiris_frame_params
    p = osiris_frame_params(header)
    return replace(p, camname="osiris_ifs", effwave_um=float(wl_um),
                   plate_scale_mas=spaxel_scale_mas(header),
                   filter_name=str(header.get("SFILTER", "") or "").strip(),
                   max_counts=float("inf"))


def ifs_dl_psf(params, npix=512):
    """The diffraction-limited PSF at the spaxel scale (unit sum)."""
    from .nirc2_psf import nirc2_dl_psf
    return nirc2_dl_psf("narrow", params.pmsname, params.effwave_um,
                        params.pmrangl_deg, npix=npix,
                        daytime=params.daytime,
                        plate_scale_mas=params.plate_scale_mas)


_DL_FWHM = {}


def dl_fwhm_mas(params, dl_psf, peak_radius_arcsec=IFS_PEAK_ARCSEC,
                idl_version=None):
    """FWHM of the diffraction-limited PSF `dl_psf` (core on a spaxel
    corner), measured as a star is with the tools' find_peak: the fallback
    reference when the model fit fails. Cached per scale, wavelength,
    pupil, box and convention."""
    from .image_strehl import SR_TOOL_IDL_DEFAULT, measure_strehl
    idl = idl_version or SR_TOOL_IDL_DEFAULT
    key = (round(params.plate_scale_mas, 3), round(params.effwave_um, 5),
           params.pmsname, round(params.pmrangl_deg, 3),
           round(peak_radius_arcsec, 4), idl, dl_psf.shape)
    if key not in _DL_FWHM:
        c = dl_psf.shape[0] // 2
        r = measure_strehl(np.asarray(dl_psf, dtype=float) * 1e6,
                           params=params, pos=(c, c), sky_override=0.0,
                           photometry_radius_arcsec=IFS_PHOTOMETRY_ARCSEC,
                           bg_inner_arcsec=IFS_SKY_INNER_ARCSEC,
                           bg_outer_arcsec=IFS_SKY_OUTER_ARCSEC,
                           peak_radius_arcsec=peak_radius_arcsec,
                           dl_psf=dl_psf, idl_version=idl)
        if len(_DL_FWHM) > 64:
            _DL_FWHM.clear()
        _DL_FWHM[key] = float(r.fwhm_mas) if r.ok else 0.0
    return _DL_FWHM[key]


def ifs_sky_mode(n_sky):
    """The sky_mode text of an IFU measurement with `n_sky` annulus spaxels."""
    if n_sky >= IFS_MIN_SKY_SPAXELS:
        return f"ifs: annulus median of {n_sky} valid spaxels"
    return "ifs: sky 0 (annulus off the IFU field)"


def measure_ifs(image, valid, params, pos=None,
                photometry_radius_arcsec=IFS_PHOTOMETRY_ARCSEC,
                bg_inner_arcsec=IFS_SKY_INNER_ARCSEC,
                bg_outer_arcsec=IFS_SKY_OUTER_ARCSEC,
                peak_radius_arcsec=IFS_PEAK_ARCSEC, dl_psf=None,
                idl_version=None):
    """Measure a collapsed cube (or one wavelength bin): the model-fit
    FWHM and the diffraction limit's (model_fwhm.py), and the relative SR
    in the given aperture.

    `image`, `valid`: from collapse(); `pos` = (x, y) in the cube's spaxels
    (None: the brightest valid spaxel). The image is padded so the
    measurement's boxes stay inside it; invalid and padding spaxels hold the
    sky, so they add no flux. Returns a Nirc2StrehlResult with mode "ifs",
    x / y in the cube's spaxels, edge_clip = the share of the sky annulus's
    outer disc on invalid spaxels or off the field."""
    from .image_strehl import (SR_TOOL_IDL_DEFAULT, _failed, measure_strehl,
                               sigma_clipped_median, sigma_filter3)
    image = np.asarray(image, dtype=float)
    valid = np.asarray(valid, dtype=bool)
    if not valid.any():
        return _failed(params, "the cube has no valid spaxel")
    idl = idl_version or SR_TOOL_IDL_DEFAULT
    ps = params.plate_scale_mas
    smooth = sigma_filter3(display_image(image, valid))
    if pos is None:
        work = np.where(valid, smooth, -np.inf)
        iy, ix = np.unravel_index(int(np.argmax(work)), work.shape)
        x0, y0 = float(ix), float(iy)
    else:
        x0, y0 = float(pos[0]), float(pos[1])
    # a star, not the brightest noise spaxel: the centroid alone can lock on
    # noise in a cube without its star
    vals = image[valid]
    med = float(np.median(vals))
    mad = 1.4826 * float(np.median(np.abs(vals - med)))
    iyc = min(max(int(round(y0)), 0), image.shape[0] - 1)
    ixc = min(max(int(round(x0)), 0), image.shape[1] - 1)
    signif = (float(smooth[iyc, ixc]) - med) / mad if mad > 0 else np.inf
    if signif < IFS_MIN_PEAK_SIGMA:
        return _failed(params, f"no star: the spaxel at ({x0:.0f}, {y0:.0f}) is {signif:.1f} sigma above the "
                               f"field (< {IFS_MIN_PEAK_SIGMA:g}) -- is the star in the IFU field?")
    # an A-B cube's negative star near enough to touch the sky annulus: masked
    notes = []
    field_valid = valid
    if mad > 0:
        neg = np.where(valid, smooth, np.inf)
        jy, jx = np.unravel_index(int(np.argmin(neg)), neg.shape)
        depth = (med - float(smooth[jy, jx])) / mad
        d_neg = float(np.hypot(jx - x0, jy - y0)) * ps / 1000.0
        if depth >= IFS_MIN_PEAK_SIGMA and d_neg <= bg_outer_arcsec + IFS_NEG_MASK_ARCSEC:
            rr_neg = np.hypot(np.arange(image.shape[1])[None, :] - jx,
                              np.arange(image.shape[0])[:, None] - jy) * ps / 1000.0
            valid = valid & (rr_neg > IFS_NEG_MASK_ARCSEC)
            note = (f"negative image (A-B) {d_neg:.2f}\" away left out (r {IFS_NEG_MASK_ARCSEC:.2f}\")")
            aperture = field_valid & (np.hypot(np.arange(image.shape[1])[None, :] - x0,
                                               np.arange(image.shape[0])[:, None] - y0) * ps / 1000.0
                                      <= photometry_radius_arcsec)
            share = float((aperture & ~valid).sum()) / max(int(aperture.sum()), 1)
            if share > 0:
                note += (f": {100 * share:.0f} % of the SR aperture masked, the SR is "
                         + ("NOT usable" if share > IFS_NEG_APERTURE_MAX else "biased high"))
            notes.append(note)
    yy, xx = np.indices(image.shape)
    rr = np.hypot(xx - x0, yy - y0) * ps / 1000.0
    ring = valid & (rr >= bg_inner_arcsec) & (rr <= bg_outer_arcsec)
    n_sky = int(ring.sum())
    sky = (float(sigma_clipped_median(image[ring]))
           if n_sky >= IFS_MIN_SKY_SPAXELS else 0.0)
    pad = int(np.ceil(max(bg_outer_arcsec, 4 * peak_radius_arcsec)
                      * 1000.0 / ps)) + _PAD_MARGIN_PX
    canvas = np.full((image.shape[0] + 2 * pad, image.shape[1] + 2 * pad), sky)
    canvas[pad:-pad, pad:-pad] = np.where(valid, image, sky)
    if dl_psf is None:
        dl_psf = ifs_dl_psf(params)
    r = measure_strehl(canvas, params=params, pos=(x0 + pad, y0 + pad),
                       sky_override=sky,
                       photometry_radius_arcsec=photometry_radius_arcsec,
                       bg_inner_arcsec=bg_inner_arcsec,
                       bg_outer_arcsec=bg_outer_arcsec,
                       peak_radius_arcsec=peak_radius_arcsec, dl_psf=dl_psf,
                       idl_version=idl)
    if not r.ok:
        return replace(r, error=f"{r.error} -- is the star in the IFU field? "
                                "(brightest valid spaxel at "
                                f"{x0:.0f}, {y0:.0f})")
    x, y = r.x - pad, r.y - pad
    disc = np.hypot(np.arange(canvas.shape[1])[None, :] - r.x,
                    np.arange(canvas.shape[0])[:, None] - r.y) \
        <= bg_outer_arcsec * 1000.0 / ps
    vcanvas = np.zeros(canvas.shape, dtype=bool)
    vcanvas[pad:-pad, pad:-pad] = field_valid      # EDGE: the field's edge, not a masked negative image
    n_disc = int(disc.sum())
    edge = 1.0 - float((disc & vcanvas).sum()) / n_disc if n_disc else 1.0
    from .model_fwhm import fit_model_fwhm
    fit = fit_model_fwhm(image, x, y, ps, params.effwave_um, valid=valid,
                         pmsname=params.pmsname, pmrangl_deg=params.pmrangl_deg,
                         daytime=params.daytime)
    if fit.ok:
        fwhm, fwhm_dl = fit.fwhm_mas, fit.fwhm_dl_mas
        if fit.note:
            notes.append(fit.note)
    else:
        fwhm, fwhm_dl = r.fwhm_mas, dl_fwhm_mas(params, dl_psf,
                                                 peak_radius_arcsec, idl)
        notes.append(f"model FWHM unavailable ({fit.note}): FWHM is the tools' "
                     "find_peak value, which depends on the spaxel phase here")
    sky_mode = ifs_sky_mode(n_sky) + (" (negative image masked)" if field_valid is not valid else "")
    return replace(r, x=float(x), y=float(y), sky_mode=sky_mode,
                   mode="ifs", edge_clip=float(edge), fwhm_mas=float(fwhm),
                   fwhm_dl_mas=float(fwhm_dl), note="; ".join(notes))
