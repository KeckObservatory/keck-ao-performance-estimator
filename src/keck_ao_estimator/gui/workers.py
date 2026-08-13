"""Worker threads that run engine calls off the GUI thread: PrepareWorker
(the expensive prepare_night fetch/parse step), SkyFetchWorker (the DSS/2MASS
cutout download for the field-map backdrop), CatalogFetchWorker (the Vizier
guide-star lookup), and ResolveWorker (the SIMBAD target-name lookup).
"""
import contextlib
import io

from qtcompat import QtCore, QThread, Signal

from ..catalogs import query_guide_stars
from ..imaging import _fetch_sky_jpeg
from ..pipeline import prepare_night
from ..target_resolve import resolve_target_name
from ..winds import night_winds


class PrepareWorker(QThread):
    """Runs the expensive engine.prepare_night() off the GUI thread (§4): it
    does the MKWC fetch / file parse and astropy target geometry. The fast
    compute_timeline() + render happen back on the GUI thread once this
    finishes, so budget-slider recomputes never touch this worker."""
    prepared = Signal(object, str)   # (prep SimpleNamespace, captured stdout)
    failed = Signal(str, str)        # (short message, full log)

    def __init__(self, args, parent=None):
        super().__init__(parent)
        self.args = args

    def run(self):
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                prep = prepare_night(self.args)
        except SystemExit as e:
            self.failed.emit(str(e) or "engine exited", buf.getvalue())
            return
        except Exception as e:
            self.failed.emit(f"{type(e).__name__}: {e}", buf.getvalue())
            return
        self.prepared.emit(prep, buf.getvalue())


class SkyFetchWorker(QThread):
    """Fetches a DSS/2MASS cutout off the GUI thread (the CDS request takes a
    few seconds). Emits (grayscale array, half_arcsec, "") on success or
    (None, 0, error) on failure."""
    done = Signal(object, float, str)

    def __init__(self, url, half_arcsec, parent=None):
        super().__init__(parent)
        self.url = url
        self.half = half_arcsec

    def run(self):
        try:
            arr = _fetch_sky_jpeg(self.url)
            self.done.emit(arr, self.half, "")
        except Exception as e:
            self.done.emit(None, 0.0, f"{type(e).__name__}: {e}")


class CatalogFetchWorker(QThread):
    """Runs a Vizier guide-star query off the GUI thread (the CDS round-trip
    takes a few seconds). Emits (catalog_name, stars, "") on success or
    (catalog_name, [], error) on failure. stars is the list of dicts from
    catalogs.query_guide_stars (id, ra, dec, mags)."""
    done = Signal(str, object, str)

    def __init__(self, catalog, ra_deg, dec_deg, radius_arcsec, parent=None):
        super().__init__(parent)
        self.catalog = catalog
        self.ra_deg = ra_deg
        self.dec_deg = dec_deg
        self.radius_arcsec = radius_arcsec

    def run(self):
        try:
            stars = query_guide_stars(self.catalog, self.ra_deg, self.dec_deg,
                                      self.radius_arcsec)
            self.done.emit(self.catalog, stars, "")
        except Exception as e:
            self.done.emit(self.catalog, [], f"{type(e).__name__}: {e}")


class ResolveWorker(QThread):
    """Resolves a target name via SIMBAD off the GUI thread (the query takes
    a second or two). Emits (result_dict, "") on success -- see
    target_resolve.resolve_target_name for the dict's keys -- or
    (None, error) on failure (name didn't resolve, or a network error)."""
    done = Signal(object, str)

    def __init__(self, name, parent=None):
        super().__init__(parent)
        self.name = name

    def run(self):
        try:
            result = resolve_target_name(self.name)
            self.done.emit(result, "")
        except Exception as e:
            self.done.emit(None, f"{type(e).__name__}: {e}")


class GfsWindsWorker(QThread):
    """Fetches the night's GFS winds (winds.night_winds) off the GUI thread.
    The engine call carries its own hard timeout and never retries, so this
    thread is bounded too -- but the fetch (~1-2 s normally, <= ~10 s worst
    case) must still not block the UI. Emits (result_dict, "") on success or
    (None, error) on any failure."""
    done = Signal(object, str)

    def __init__(self, ymd, cache_dir, fa_weights=None, parent=None):
        super().__init__(parent)
        self.ymd, self.cache_dir, self.fa_weights = ymd, cache_dir, fa_weights

    def run(self):
        try:
            result = night_winds(self.ymd, self.cache_dir,
                                 fa_weights=self.fa_weights)
            self.done.emit(result, "")
        except Exception as e:
            self.done.emit(None, str(e))


class NativeFilenameWorker(QThread):
    """Reads each FITS file's own DATAFILE/FRAMENO header cards off the GUI
    thread, for directories of KOA-renamed frames (OI.<utdate>.<sec>.<hun>.fits
    etc.) where the archive's own filename has nothing to do with the
    observer's night log. DATAFILE is the original observatory filename
    (e.g. i260112_a000061.fits) that KOA preserves in every downloaded
    frame's header -- no external metadata table needed. Emits done(dict)
    mapping disk filename -> native display name, omitting any file where
    DATAFILE is absent or unreadable (its disk name is used as-is)."""
    done = Signal(object)

    def __init__(self, dirpath, filenames, parent=None):
        super().__init__(parent)
        self.dirpath = dirpath
        self.filenames = filenames

    def run(self):
        import os

        from astropy.io import fits

        out = {}
        for name in self.filenames:
            try:
                header = fits.getheader(os.path.join(self.dirpath, name))
                native = header.get("DATAFILE")
                if native:
                    out[name] = str(native)
            except Exception:
                continue
        self.done.emit(out)


class Nirc2MeasureWorker(QThread):
    """Measures the Strehl of a sequence of NIRC2 frames off the GUI thread
    (the IDL Strehl tool's GO loop: image_strehl.measure_strehl per frame,
    each needing its own ~1 s diffraction-limited PSF). Per frame emits
    frame_done(imno, result_or_None, params, reduced_image, dl_psf,
    header) --
    result is None when autofind is off (the frame is loaded and displayed
    but measurement waits for the user's click); a missing or broken frame
    emits frame_failed(imno, message) and the loop continues. Emits
    finished_all() when the sequence is exhausted.

    pause()/resume() genuinely SUSPEND the measurement loop (mutex +
    wait condition, checked once per frame before any work is done), so
    the GUI can stop a run mid-sequence to ask the user something and
    have the run actually wait for the answer rather than racing ahead
    and queueing more frames behind the dialog (Eduardo 2026-07-28)."""
    frame_done = Signal(object, object, object, object, object, object)
    frame_failed = Signal(object, str)
    finished_all = Signal()

    def __init__(self, path, prefix, im1, nim, bg1, nbg, radii,
                 autofind=True, files=None, robust_sky=False,
                 sky_override=None, auto_radius=False, psf_clean=False,
                 parent=None):
        super().__init__(parent)
        self.path, self.prefix = path, prefix
        self.im1, self.nim, self.bg1, self.nbg = im1, nim, bg1, nbg
        self.radii = radii          # (photrad, bgin, bgout, peakrad) arcsec
        self.autofind = autofind
        self.files = files          # [(label, path)] overrides the numbering
        self.robust_sky = robust_sky
        self.sky_override = sky_override
        self.auto_radius = auto_radius
        self.psf_clean = psf_clean
        self._pause_mutex = QtCore.QMutex()
        self._pause_cond = QtCore.QWaitCondition()
        self._paused = False
        self._aborted = False

    def pause(self):
        """Ask the loop to stop before its next frame (returns immediately;
        the worker may still be finishing the frame already in flight)."""
        self._pause_mutex.lock()
        self._paused = True
        self._pause_mutex.unlock()

    def resume(self):
        self._pause_mutex.lock()
        self._paused = False
        self._pause_cond.wakeAll()
        self._pause_mutex.unlock()

    def abort(self):
        """Stop the sequence entirely; also releases a paused loop."""
        self._pause_mutex.lock()
        self._aborted = True
        self._paused = False
        self._pause_cond.wakeAll()
        self._pause_mutex.unlock()

    def _wait_if_paused(self):
        """Block here while paused. Returns False if the run was aborted."""
        self._pause_mutex.lock()
        while self._paused and not self._aborted:
            self._pause_cond.wait(self._pause_mutex)
        stop = self._aborted
        self._pause_mutex.unlock()
        return not stop

    def _fname(self, no):
        import os
        return os.path.join(self.path, f"{self.prefix}{no:04d}.fits")

    def run(self):
        import numpy as np
        from astropy.io import fits

        from ..image_strehl import (find_stars, load_nirc2_calibration,
                                    measure_strehl, osiris_reduce,
                                    reduce_frame)
        from ..nirc2 import nirc2_frame_params
        from ..nirc2_psf import nirc2_dl_psf
        from ..osiris import detect_instrument, osiris_frame_params

        try:
            flat, mask = load_nirc2_calibration()
        except Exception as e:
            self.frame_failed.emit(self.im1, f"calibration: {e}")
            self.finished_all.emit()
            return

        background = None
        if self.nbg > 0:
            bgs = []
            for no in range(self.bg1, self.bg1 + self.nbg):
                try:
                    bgs.append(np.asarray(fits.getdata(self._fname(no)),
                                          dtype=float))
                except Exception:
                    # find_strehl.pro: a bad background disables backgrounds
                    bgs = []
                    break
            if bgs:
                background = np.mean(bgs, axis=0)

        photrad, bgin, bgout, peakrad = self.radii
        dl_cache = {}
        seq = (self.files if self.files is not None else
               [(str(no), self._fname(no))
                for no in range(self.im1, self.im1 + self.nim)])
        for no, fpath in seq:
            # honour a pause requested by the GUI (e.g. a duplicate-frame
            # question) BEFORE touching the next frame, so the run really
            # waits for the answer instead of queueing more results behind
            # the dialog
            if not self._wait_if_paused():
                break
            try:
                with fits.open(fpath) as hdul:
                    header = hdul[0].header
                    # NIRC2 and OSIRIS route to their own pipelines; an
                    # explicit other INSTRUME (GSAOI, ...) is refused by
                    # name; a header with neither identity is accepted as
                    # NIRC2 only when the CAMNAME card is present
                    inst = detect_instrument(header)
                    instrume = str(header.get("INSTRUME", "")).upper()
                    if (inst == ""
                            and not (instrume == ""
                                     and header.get("CAMNAME") is not None)):
                        self.frame_failed.emit(
                            no, f"{instrume or 'unknown instrument'}: only "
                                "NIRC2 and OSIRIS frames are supported")
                        continue
                    raw = np.asarray(hdul[0].data, dtype=float)
                if inst == "osiris":
                    params = osiris_frame_params(header)
                    if background is not None:
                        # numbered n#### backgrounds are a NIRC2 concept
                        self.frame_failed.emit(
                            no, "OSIRIS backgrounds are not supported in "
                                "the GUI yet -- measured without")
                    # full frame for the GUI (Eduardo: show the whole
                    # detector, not the tool's central-1024 crop)
                    reduced = osiris_reduce(raw, crop=False)
                else:
                    params = nirc2_frame_params(header)
                    reduced = reduce_frame(raw, background=background,
                                           flat=flat, badmask=mask)
                key = (params.camname, params.pmsname, params.effwave_um,
                       round(params.pmrangl_deg, 3), params.daytime,
                       params.sfp)
                if key not in dl_cache:
                    dl_cache.clear()    # keep at most one 512^2 PSF around
                    dl_cache[key] = nirc2_dl_psf(
                        params.camname, params.pmsname, params.effwave_um,
                        params.pmrangl_deg, npix=512, daytime=params.daytime,
                        sfp=params.sfp)
                result = None
                if self.autofind:
                    kwargs = dict(
                        params=params,
                        background_subtracted=background is not None,
                        photometry_radius_arcsec=photrad,
                        bg_inner_arcsec=bgin, bg_outer_arcsec=bgout,
                        peak_radius_arcsec=peakrad, dl_psf=dl_cache[key],
                        robust_sky=self.robust_sky,
                        sky_override=self.sky_override,
                        auto_radius=self.auto_radius,
                        psf_clean=self.psf_clean)
                    result = measure_strehl(reduced, **kwargs)
                    if not result.ok:
                        # AUTOFIND's brightest pixel is not always a
                        # measurable star: on the 20260528 M13 frames it
                        # landed on the TT star's saturated/bled plateau
                        # and the centroid degenerated ("centroid
                        # failed"). Fall back to the detection-floor
                        # star finder (find_stars rejects hot pixels and
                        # flat-topped plateaus) and take the first
                        # candidate that measures.
                        why = result.error or "not measurable"
                        photrad_px = (photrad * 1000.0
                                      / params.plate_scale_mas)
                        for pos in find_stars(reduced, n_stars=3,
                                              exclude_px=photrad_px):
                            retry = measure_strehl(reduced, pos=pos,
                                                   **kwargs)
                            if retry.ok:
                                result = retry
                                self.frame_failed.emit(
                                    no, f"autofind brightest pixel "
                                        f"unmeasurable ({why}) -- fell "
                                        f"back to a detected star at "
                                        f"({pos[0]:.0f}, {pos[1]:.0f})")
                                break
                self.frame_done.emit(no, result, params, reduced,
                                     dl_cache[key], header)
            except Exception as e:
                self.frame_failed.emit(no, f"{type(e).__name__}: {e}")
        self.finished_all.emit()
