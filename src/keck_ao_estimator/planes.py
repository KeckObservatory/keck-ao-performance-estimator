"""Display-only plane sources for the Strehl tools' cube playback (KAPE's Measured SR tab and PyAO's strehl_tool): what CUBE PLANE steps through.

* CubePlanes -- the planes of a data cube in the primary HDU (ORKID 100-frame files), memory-mapped and unscaled;
* CoaddPlanes -- the coadds of a NIRC2 frame from its ``_unp`` file (nirc2_coadds), placed where reduce_frame puts
  the frame, so zoom and clicks line up with the measured frame;
* IfsPlanes -- the wavelength bins of an OSIRIS spectrograph cube (osiris_ifs), each the mean of its good channels.

Each has ``n`` (planes), ``shape`` (the frame shape on display), ``window(k, y0, y1, x0, x1)`` (plane k, 1-based,
rows y0:y1 and columns x0:x1 of that frame, float), ``full(k)`` (the whole plane as the measured frame is drawn,
for the display limits; None when the window alone is used), ``label(k)`` (how the playback title names plane k)
and ``close()``.

Display limits (Eduardo 2026-10-25: a NIRC2 coadd changed colour and stretch when played, an ORKID plane did not):
a played plane must be drawn as the same plane is when measured. NIRC2's default Limits, IDL ±5σ, is the mean
± 5 std of the whole frame; computed on a raw, block-averaged window of the view instead, hot pixels, the flat, the
averaging and a zoom onto the star all moved it (o Her: span 92 vs 106 ADU for the same coadds). So coadds and IFU
bins give ``full``: a coadd reduced as its measurement is (reduce_frame's canvas, flat, bad pixels repaired by
display_fill: deadpix_fill vectorized, the same pixels, 15x faster), an IFU bin whole. Cube
planes keep the fast window read (ORKID: 31 fps on 2048^2; its min/max limits barely move).
"""
import numpy as np

from . import nirc2_coadds, osiris_ifs

PLANE_KINDS = ("cube", "coadd", "ifs")
# what CUBE PLANE 0 means for each kind (the spin box shows it)
PLANE_ZERO_TEXT = {"cube": "mean", "coadd": "sum", "ifs": "all λ"}


class CubePlanes:
    """The planes of the cube in the primary HDU of `path` (memory-mapped and unscaled: a window reads only its
    part; BSCALE applied)."""

    def __init__(self, path):
        """Open the cube; ValueError if the primary HDU is not one."""
        from astropy.io import fits
        # the raw memory map, not hdul[0].section: a section slicing columns reads row by row (8 ms for a 300 x 300
        # region of a 2048^2 ORKID plane, against 0.04 ms here)
        self._hdul = fits.open(path, do_not_scale_image_data=True)
        header = self._hdul[0].header
        planes = int(header.get("NAXIS3", 1)) if int(header.get("NAXIS", 0)) == 3 else 1
        if planes < 2:
            self.close()
            raise ValueError("not a cube")
        self.n = planes
        self.shape = (int(header["NAXIS2"]), int(header["NAXIS1"]))
        self._raw = self._hdul[0].data
        self._bscale = float(header.get("BSCALE", 1.0))
        self._bzero = float(header.get("BZERO", 0.0))

    def window(self, k, y0, y1, x0, x1):
        """Plane `k` (1-based), rows y0:y1, columns x0:x1 (BZERO left in: the display subtracts a background)."""
        img = np.asarray(self._raw[k - 1, y0:y1, x0:x1], dtype=float)
        if self._bscale != 1.0:
            img *= self._bscale
        return img

    def full(self, k):
        """None: a cube plane is drawn from its window alone (the fast path: ORKID 2048^2 planes at 31 fps)."""
        return None

    def value(self, k, y, x):
        """Pixel (x, y) of plane `k` in physical units (BSCALE and BZERO applied), for the cursor readout."""
        return float(self._raw[k - 1, y, x]) * self._bscale + self._bzero

    def label(self, k):
        """'plane k/n'."""
        return f"plane {k}/{self.n}"

    def close(self):
        """Close the file."""
        if self._hdul is not None:
            try:
                self._hdul.close()
            except Exception:
                pass
            self._hdul = self._raw = None


def display_fill(img, bad, neighbors=3):
    """`img` with the `bad` pixels repaired exactly as the measurement does (image_strehl.deadpix_fill, bpixfix.pro:
    passes replace each bad pixel having >= `neighbors` good 3x3 neighbours by the IDL median, the upper middle, of
    them), vectorized: a pass only reads pixels good at its start, so all its fixes are independent (0.3 s -> ~20 ms
    on a NIRC2 frame; the same pixels)."""
    out = np.array(img, dtype=float)
    bp = np.asarray(bad, dtype=bool).copy()
    ny, nx = out.shape
    offsets = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)]
    while bp.any():
        ys, xs = np.nonzero(bp)
        good = ~bp
        vals = np.full((ys.size, 9), np.inf)
        ok = np.zeros((ys.size, 9), dtype=bool)
        for j, (dy, dx) in enumerate(offsets):
            yy, xx = ys + dy, xs + dx
            inside = (yy >= 0) & (yy < ny) & (xx >= 0) & (xx < nx)
            g = np.zeros(ys.size, dtype=bool)
            g[inside] = good[yy[inside], xx[inside]]
            ok[:, j] = g
            vals[g, j] = out[yy[g], xx[g]]
        count = ok.sum(axis=1)
        fix = count >= neighbors
        if not fix.any():           # isolated cluster smaller than `neighbors`
            break
        srt = np.sort(vals[fix], axis=1)                    # the good values first, inf after
        out[ys[fix], xs[fix]] = srt[np.arange(srt.shape[0]), count[fix] // 2]
        bp[ys[fix], xs[fix]] = False
    return out


class CoaddPlanes:
    """The coadds of the NIRC2 frame at `path` from its ``_unp`` file, on the canvas of the reduced frame (shape
    `shape`): a raw frame of h x w sits where reduce_frame embeds it; outside it the plane is 0. `calibration` =
    (flat, mask) of the reduced canvas (either None) flat-fields the coadd and fills its bad pixels for display."""

    def __init__(self, path, shape, n=None, calibration=None):
        """Open the _unp file beside `path`; ValueError if there is none or it holds fewer than two coadds."""
        from astropy.io import fits
        unp = nirc2_coadds.unp_path(path)
        if unp is None:
            raise ValueError("no _unp file beside it: its coadds are not on this disk")
        count = nirc2_coadds.coadd_count(unp, expected=n)
        if count < 2:
            raise ValueError(f"its _unp file holds {count} complete coadd(s)")
        self._hdul = fits.open(unp, memmap=True)
        self.n = count
        self.shape = tuple(shape)
        h1 = next(h.header for h in self._hdul[1:] if int(h.header.get("NAXIS", 0)) == 3)
        self._raw_shape = (int(h1["NAXIS2"]), int(h1["NAXIS1"]))
        self._oy, self._ox = nirc2_coadds.unp_embed_offset(self._raw_shape, n=max(self.shape))
        flat, mask = calibration if calibration is not None else (None, None)
        self._flat = np.asarray(flat, dtype=float) if flat is not None and np.shape(flat) == self.shape else None
        self._mask = np.asarray(mask, dtype=bool) if mask is not None and np.shape(mask) == self.shape else None
        self._full_k, self._full = None, None

    def full(self, k):
        """Coadd `k` (1-based) as its measurement is drawn: reduce_frame's canvas, flat-fielded, bad pixels repaired
        (display_fill, the same pixels as the measurement's). The last one is kept (window() of it reads it)."""
        if self._full_k != k:
            canvas = np.zeros(self.shape)
            h, w = self._raw_shape
            canvas[self._oy:self._oy + h, self._ox:self._ox + w] = nirc2_coadds.coadd_window(self._hdul, k)
            if self._flat is not None:
                canvas = canvas / self._flat
            if self._mask is not None and self._mask.any():
                canvas = display_fill(canvas, self._mask)
            self._full_k, self._full = k, canvas
        return self._full

    def window(self, k, y0, y1, x0, x1):
        """Coadd `k` (1-based) on the reduced frame's grid, rows y0:y1 and columns x0:x1 (0 off the detector)."""
        return self.full(k)[y0:y1, x0:x1]

    def value(self, k, y, x):
        """Pixel (x, y) of coadd `k` as drawn, for the cursor readout."""
        return float(self.full(k)[y, x])

    def label(self, k):
        """'coadd k/n'."""
        return f"coadd {k}/{self.n}"

    def close(self):
        """Close the _unp file."""
        if self._hdul is not None:
            try:
                self._hdul.close()
            except Exception:
                pass
            self._hdul = None


class IfsPlanes:
    """The wavelength bins (`width` channels each) of the OSIRIS spectrograph cube at `path`; a bin is collapsed when
    first shown and kept (cubes are 13-18 MB)."""

    def __init__(self, path, width):
        """Read the cube; ValueError if it is not one or has a single bin."""
        self._cube = osiris_ifs.read_cube(path)
        self._bins = osiris_ifs.channel_bins(self._cube.nchan, width)
        if len(self._bins) < 2:
            raise ValueError(f"{self._cube.nchan} channels make one bin of {width}")
        self.n = len(self._bins)
        self.shape = self._cube.data.shape[:2]
        self._planes = {}

    def window(self, k, y0, y1, x0, x1):
        """Bin `k` (1-based), rows y0:y1 and columns x0:x1, invalid spaxels at the valid ones' median."""
        if k not in self._planes:
            lo, hi = self._bins[k - 1]
            img, valid = osiris_ifs.collapse(self._cube, lo, hi)
            self._planes[k] = osiris_ifs.display_image(img, valid)
        return self._planes[k][y0:y1, x0:x1]

    def full(self, k):
        """Bin `k` (1-based) whole, as its measurement is drawn."""
        return self.window(k, 0, self.shape[0], 0, self.shape[1])

    def value(self, k, y, x):
        """Spaxel (x, y) of bin `k` as drawn (invalid spaxels at the valid ones' median), for the cursor readout."""
        return float(self.full(k)[y, x])

    def label(self, k):
        """'λc 2.1612 µm (25 ch) k/n': the bin's central wavelength."""
        lo, hi = self._bins[k - 1]
        return f"{osiris_ifs.bin_label(self._cube.wavelengths_um, lo, hi)} {k}/{self.n}"

    def close(self):
        """Drop the cube."""
        self._cube = None
        self._planes = {}


def open_planes(kind, path, shape=None, width=osiris_ifs.IFS_DEFAULT_BIN, n=None, calibration=None):
    """The plane source of `kind` ("cube", "coadd", "ifs") for the frame at `path` (`shape`: the reduced frame's,
    for coadds; `width`: channels per IFU bin; `n`: the coadds expected; `calibration`: (flat, mask) for coadds)."""
    if kind == "coadd":
        return CoaddPlanes(path, shape, n=n, calibration=calibration)
    if kind == "ifs":
        return IfsPlanes(path, width)
    return CubePlanes(path)


def plane_info(path, header, inst, ifs_bin=osiris_ifs.IFS_DEFAULT_BIN, remote=False):
    """(planes, kind) of the frame at `path` for CUBE PLANE: an OSIRIS IFU cube's wavelength bins of `ifs_bin`
    channels ("ifs"), a NIRC2 frame's coadds in its _unp file ("coadd"; for a `remote` PATH, whose _unp file is
    copied only when a coadd is measured, the COADDS card), a data cube's planes ("cube"); planes 0 when there is
    nothing to step through."""
    naxis = int(header.get("NAXIS", 0) or 0)
    if inst is not None and inst.is_ifs and osiris_ifs.is_ifs_cube(header):
        return len(osiris_ifs.channel_bins(int(header["NAXIS1"]), ifs_bin)), "ifs"
    if inst is not None and inst.coadds_file == "unp" and naxis == 2:
        n = int(header.get("COADDS", 1) or 1) if remote else 0
        unp = nirc2_coadds.unp_path(path) if path else None
        if unp is not None:
            try:
                n = nirc2_coadds.coadd_count(unp, expected=int(header.get("COADDS", 1) or 1))
            except Exception:
                n = 0
        return (n if n > 1 else 0), "coadd"
    if naxis == 3:
        return int(header.get("NAXIS3", 1)), "cube"
    return 0, "cube"
