"""NIRC2 coadds, one at a time, from the unprocessed ``_unp`` file.

NIRC2 writes every frame twice: ``nNNNN.fits``, the SUM of its coadds, and
``n_unp_NNNN.fits`` beside it, the raw detector reads of each coadd (KOA
exports the same file as ``<name>_unp.fits``). The ``_unp`` file is a
primary header with no data followed by one int16 image extension per
coadd, holding that coadd's reads:

* CDS (SAMPMODE 2): two reads, the coadd is ``read[1] - read[0]``;
* MCDS (SAMPMODE 3, MULTISAM N): 2N reads, the coadd is the mean of the
  last N minus the mean of the first N.

The coadds of 2026-07-27 ``n0001`` (o Her, 20 x 54 ms CDS) sum to the
processed frame exactly; an MCDS frame (2026-08-07 ``n0001``, 16 reads)
matches to the processed frame's rounding (0.5 ADU).

Short-exposure SR. In a single coadd the star's peak is steady (+-1.3 % on
o Her) but the flux in the 1" aperture is not: NIRC2's pattern ("chevron")
noise changes from coadd to coadd and does not average out over the
aperture and the sky annulus, so the full-flux SR of a coadd scatters over
0.55-1.16 for a frame of SR 0.69 (Eduardo 2026-10-25: "classic chevron
noise"; blank sky 2-3" away scatters the same, 0.56 vs 0.49 ADU/px, so it is
the detector, not the halo). The star's flux, however, is the same in every
coadd, so a coadd is measured against the FRAME's flux:

    SR_k = (peak_k / (F_frame / N)) / S1 = SR_frame * N * peak_k / peak_frame

with peak_k found at the frame's star position (the free search can lose
the star in a faint coadd). On o Her this gives 0.696 +- 0.009 for a frame
of 0.692: the coadd's peak, i.e. how sharp that coadd was, against the
frame's flux. FWHM needs no flux and is the coadd's own.
"""
import os
import re
from dataclasses import replace

import numpy as np

__all__ = ["unp_path", "coadd_count", "read_coadd", "coadd_window",
           "unp_embed_offset", "measure_coadd", "COADD_SKY_MODE"]

COADD_SKY_MODE = "coadd: frame flux / N"


def unp_path(path):
    """The ``_unp`` file holding the coadds of NIRC2 frame `path`, or None.

    ``<dir>/nNNNN.fits`` -> ``<dir>/n_unp_NNNN.fits`` (the summit name);
    a KOA ``<stem>.fits`` -> ``<stem>_unp.fits``. Only an existing file is
    returned; `path` itself being an ``_unp`` file returns None."""
    d, name = os.path.split(str(path))
    if re.search(r"_unp[_.]", name, re.IGNORECASE):
        return None
    m = re.match(r"^n(\d+)\.fits$", name)
    cands = []
    if m:
        cands.append(f"n_unp_{m.group(1)}.fits")
    stem, ext = os.path.splitext(name)
    cands.append(f"{stem}_unp{ext}")
    for c in cands:
        p = os.path.join(d, c)
        if os.path.isfile(p):
            return p
    return None


def _image_extensions(hdul):
    """The extensions of an open ``_unp`` file that hold a coadd's reads,
    in order, up to the first one whose data are not all in the file (a
    file still being copied, or cut short)."""
    name = hdul.filename()
    size = os.path.getsize(name) if name and os.path.isfile(name) else None
    out = []
    for k in range(1, len(hdul)):
        try:
            hdu = hdul[k]
            info = hdul.fileinfo(k)
        except Exception:
            break
        if size is not None and (info is None
                                 or info["datLoc"] + info["datSpan"] > size):
            break
        if int(hdu.header.get("NAXIS", 0)) == 3:
            out.append(hdu)
    return out


def coadd_count(unp, expected=None):
    """Number of complete coadds in the ``_unp`` file `unp` (a path), at
    most `expected` (default: its COADDS card)."""
    from astropy.io import fits
    with fits.open(unp, memmap=True) as hdul:
        n = len(_image_extensions(hdul))
        if expected is None:
            expected = int(hdul[0].header.get("COADDS", n) or n)
    return min(n, int(expected)) if expected else n


def _reads_split(header, nreads):
    """(first, last) read slices of one coadd for the frame's sampling."""
    mode = int(header.get("SAMPMODE", 2) or 2)
    if mode == 3:
        n = int(header.get("MULTISAM", nreads // 2) or nreads // 2)
        n = max(1, min(n, nreads // 2))
        return slice(0, n), slice(nreads - n, nreads)
    return slice(0, 1), slice(nreads - 1, nreads)


def coadd_window(hdul, k, y0=None, y1=None, x0=None, x1=None):
    """Coadd `k` (1-based) of an open ``_unp`` file (float, raw detector
    orientation, the processed frame's units), or only its rows y0:y1 and
    columns x0:x1 -- a memory-mapped file then reads only those."""
    exts = _image_extensions(hdul)
    if not 1 <= k <= len(exts):
        raise ValueError(f"no coadd {k}: the _unp file holds {len(exts)}")
    data = exts[k - 1].data                     # (reads, y, x)
    first, last = _reads_split(hdul[0].header, data.shape[0])
    ys, xs = slice(y0, y1), slice(x0, x1)
    a = np.asarray(data[first, ys, xs], dtype=float).mean(axis=0)
    b = np.asarray(data[last, ys, xs], dtype=float).mean(axis=0)
    return b - a


def read_coadd(unp, k):
    """Coadd `k` (1-based) of the ``_unp`` file at path `unp`."""
    from astropy.io import fits
    with fits.open(unp, memmap=True) as hdul:
        return coadd_window(hdul, k)


def unp_embed_offset(shape, n=1024):
    """(row, column) where reduce_frame puts a raw frame of `shape` in its
    square canvas, so a coadd window lines up with the reduced frame."""
    ys, xs = shape
    n = max(n, ys, xs)
    return n // 2 - ys // 2, n // 2 - xs // 2


def measure_coadd(coadd_image, frame_result, n_coadds, **kwargs):
    """Short-exposure SR of one reduced coadd against its frame.

    `frame_result` is the processed frame's measurement (same settings);
    `kwargs` go to measure_strehl (radii, dl_psf, robust_sky, idl_version).
    The coadd is measured at the frame's star with coadds = 1; its SR is
    then the coadd's peak over the frame's flux / N (module docstring),
    its FWHM its own. The sky-noise error adds the coadd's peak noise to
    the frame's relative error."""
    from .image_strehl import _failed, aperture_flux, measure_strehl
    fp = frame_result.params
    if not frame_result.ok:
        return _failed(fp, f"the frame is not measurable "
                           f"({frame_result.error}): no coadd SR")
    n = max(int(n_coadds), 1)
    params = replace(fp, coadds=1)
    kwargs = dict(kwargs)
    kwargs.pop("auto_radius", None)             # the frame's radius is used
    kwargs.pop("pos", None)
    kwargs["photometry_radius_arcsec"] = frame_result.photrad_used_arcsec
    r = measure_strehl(coadd_image, params=params,
                       pos=(frame_result.x, frame_result.y), **kwargs)
    if not r.ok:
        return r
    if frame_result.peak <= 0 or frame_result.strehl == 0:
        return _failed(params, "the frame's peak is not positive: no coadd SR")
    sr = float(frame_result.strehl * n * r.peak / frame_result.peak)
    ps = params.plate_scale_mas
    bgin = kwargs.get("bg_inner_arcsec")
    bgout = kwargs.get("bg_outer_arcsec")
    sigma = 0.0
    if bgin is not None and bgout is not None:
        *_, sky_sigma, _n_ap, _n_ann = aperture_flux(
            np.asarray(coadd_image, dtype=float),
            frame_result.photrad_used_arcsec * 1000.0 / ps, r.x, r.y,
            insky_px=bgin * 1000.0 / ps, outsky_px=bgout * 1000.0 / ps)
        sigma = float(sky_sigma)
    rel_frame = (frame_result.sr_err / frame_result.strehl
                 if frame_result.strehl else 0.0)
    rel_peak = sigma / r.peak if r.peak > 0 else 0.0
    sr_err = abs(sr) * float(np.hypot(rel_frame, rel_peak))
    wfe = (float(np.sqrt(-np.log(sr)) * params.effwave_um * 1e3 / (2 * np.pi))
           if 0.0 < sr < 1.0 else 0.0)
    return replace(r, strehl=sr, sr_err=sr_err, wfe_nm=wfe,
                   flux=float(frame_result.flux / n),
                   sky_mode=COADD_SKY_MODE, mode="coadd",
                   crowding=frame_result.crowding,
                   edge_clip=frame_result.edge_clip)
