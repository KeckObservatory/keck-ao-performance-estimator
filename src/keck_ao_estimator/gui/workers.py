"""Worker threads that run engine calls off the GUI thread: PrepareWorker
(the expensive prepare_night fetch/parse step), SkyFetchWorker (the DSS/2MASS
cutout download for the field-map backdrop), CatalogFetchWorker (the Vizier
guide-star lookup), ResolveWorker (the SIMBAD target-name lookup),
FieldPrologueWorker (parallel Phase 2: the WHOLE "Measure field" prologue --
find_stars, the deep neighbour catalogue, the field ePSF and, for the field
engine, solve_field -- off the GUI thread, superseding fieldsolve P3-2's
narrower FieldSolveWorker, which covered only the solve step and left
find_stars/catalogue/ePSF blocking the GUI thread behind processEvents()
calls), and FieldMeasureWorker (the per-target measurement loop, dispatched
through keck_ao_estimator.parallel.MeasureBatch so workers > 1 genuinely
parallelizes it, exactly the machinery measure_field() itself uses
internally). Together these two remove every processEvents() call from the
Measure-field flow (parallel PR-CP3); Nirc2MeasureWorker (the Run sequence)
already had none. FrameWatchWorker polls the night's frame directory for the
Measured SR tab's auto-measure (the IDL tool's autoimage timer; rsync for a
remote PATH), and LatestNightWorker finds tonight's NIRC2 (local /s disks)
or OSIRIS (AO server, rsync) directory; KeckNetworkCheck gates the polling
to machines on the Keck network.
"""
import contextlib
import io
import threading

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


def plane_label(label, plane):
    """The frame label for plane `plane` (1-based) of a cube, a NIRC2
    frame's coadd or an IFU cube's wavelength bin: ``<label> p<plane>``;
    plane 0 (the whole frame) keeps the label. In sync with PyAO."""
    return f"{label} p{int(plane)}" if plane else str(label)


def fits_complete(path):
    """True when `path` holds its whole primary HDU (a copy taken while the
    camera server was still writing the file does not)."""
    import os

    from astropy.io import fits
    try:
        with fits.open(path) as hdul:
            info = hdul.fileinfo(0)
        return os.path.getsize(path) >= info["datLoc"] + info["datSpan"]
    except Exception:
        return False


class Nirc2MeasureWorker(QThread):
    """Measures the Strehl of a sequence of frames off the GUI thread (the
    IDL Strehl tool's GO loop: one measurement per frame). Per frame emits
    frame_done(imno, result_or_None, params, reduced_image, dl_psf,
    header) -- result is None when autofind is off (the frame is loaded and
    displayed but measurement waits for the user's click); a missing or
    broken frame emits frame_failed(imno, message) and the loop continues.
    Emits finished_all() when the sequence is exhausted; status(text) while
    copying a remote frame.

    `plane` (1-based) measures that plane of each frame instead of the whole
    frame (0): a cube's plane, a NIRC2 frame's coadd from its _unp file
    (short-exposure SR, nirc2_coadds) or an OSIRIS IFU cube's wavelength bin
    of `ifs_bin` channels (osiris_ifs); labels carry `` p<plane>``.
    `remote` = (host, dir) when PATH is remote: `path` is its local cache
    and a frame missing there (or copied while still being written) is
    copied with rsync first. `paths` maps each label to the file read and
    `extras` to what the GUI needs beyond the signal (the plane count and
    kind, an IFU cube's valid spaxels, a coadd's frame, the title note, the
    ORP subtraction). In sync with PyAO's strehl_tool MeasureWorker.

    pause()/resume() genuinely SUSPEND the measurement loop (mutex +
    wait condition, checked once per frame before any work is done), so
    the GUI can stop a run mid-sequence to ask the user something and
    have the run actually wait for the answer rather than racing ahead
    and queueing more frames behind the dialog (Eduardo 2026-07-28)."""
    frame_done = Signal(object, object, object, object, object, object)
    frame_failed = Signal(object, str)
    finished_all = Signal()
    status = Signal(str)

    def __init__(self, path, prefix, im1, nim, bg1, nbg, radii,
                 autofind=True, files=None, robust_sky=False,
                 sky_override=None, auto_radius=False, psf_clean=False,
                 idl_version=None, digits=4, parent=None, suffix="",
                 kind=None, plane=0, remote=None, ifs_bin=None):
        super().__init__(parent)
        from ..osiris_ifs import IFS_DEFAULT_BIN
        self.path, self.prefix = path, prefix
        self.suffix = suffix    # text after the number: ORKID _0000, IFU _*
        self.im1, self.nim, self.bg1, self.nbg = im1, nim, bg1, nbg
        self.radii = radii          # (photrad, bgin, bgout, peakrad) arcsec
        self.autofind = autofind
        self.files = files          # [(label, path)] overrides the numbering
        self.robust_sky = robust_sky
        self.sky_override = sky_override
        self.auto_radius = auto_radius
        self.psf_clean = psf_clean
        self.idl_version = idl_version   # None = the engine default
        self.digits = digits    # frame-number width: NIRC2 4, OSIRIS 6
        self.kind = kind        # the selected instrument (file-name fallback)
        self.plane = int(plane or 0)
        self.ifs_bin = int(ifs_bin or IFS_DEFAULT_BIN)
        self.remote = remote
        self.paths = {}
        self.extras = {}
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
        """The numbered frame <path>/<prefix><number><suffix>.fits; a `*` in
        the suffix (OSIRIS IFU cubes' _<filter>_<scale>) is matched in a
        local path (the first match)."""
        import glob
        import os
        full = os.path.join(self.path, f"{self.prefix}{no:0{self.digits}d}"
                                       f"{self.suffix}.fits")
        if "*" in full:
            found = sorted(glob.glob(full))
            if found:
                return found[0]
        return full

    def sequence(self):
        """The (label, path) list this run measures (labels with the plane)."""
        seq = (list(self.files) if self.files is not None else
               [(str(no), self._fname(no))
                for no in range(self.im1, self.im1 + self.nim)])
        return [(plane_label(label, self.plane), fpath)
                for label, fpath in seq]

    def _local(self, fpath):
        """`fpath`, first copied from the remote PATH when it is missing from
        the cache or incomplete there."""
        import os
        if self.remote is None or fits_complete(fpath):
            return fpath
        from ..frame_watch import remote_fetch
        host, rdir = self.remote
        name = os.path.basename(fpath)
        self.status.emit(f"copying {name} from {host}:{rdir} (rsync)")
        return remote_fetch(host, f"{rdir}/{name}", os.path.dirname(fpath))

    def _read_raw(self, hdul, name):
        """The frame's pixels: the whole frame or cube, or only the selected
        plane of a cube (read alone)."""
        import numpy as np
        if not self.plane:
            return np.asarray(hdul[0].data, dtype=float)
        naxis = int(hdul[0].header.get("NAXIS", 0))
        planes = int(hdul[0].header.get("NAXIS3", 1)) if naxis == 3 else 1
        if self.plane > planes:
            raise ValueError(f"{name} has {planes} plane"
                             f"{'s' if planes != 1 else ''}: no plane "
                             f"{self.plane}")
        if naxis == 3:
            return np.asarray(hdul[0].section[self.plane - 1], dtype=float)
        return np.asarray(hdul[0].data, dtype=float)

    def _kwargs(self, dl, params=None, background_subtracted=False):
        """measure_strehl's keywords for this run."""
        photrad, bgin, bgout, peakrad = self.radii
        kwargs = dict(
            background_subtracted=background_subtracted,
            photometry_radius_arcsec=photrad, bg_inner_arcsec=bgin,
            bg_outer_arcsec=bgout, peak_radius_arcsec=peakrad, dl_psf=dl,
            robust_sky=self.robust_sky, sky_override=self.sky_override,
            auto_radius=self.auto_radius, psf_clean=self.psf_clean)
        if params is not None:
            kwargs["params"] = params
        if self.idl_version is not None:
            kwargs["idl_version"] = self.idl_version
        return kwargs

    def run(self):
        import os

        import numpy as np
        from astropy.io import fits

        from ..frame_watch import (HEADER_WAIT_S, header_identifies,
                                   wait_for_header)
        from ..image_strehl import _failed, find_stars, measure_strehl
        from ..instruments import (FramePrep, identify_frame, identity_text,
                                   names)
        from ..planes import plane_info

        # per-instrument calibration, loaded once; a missing one no longer
        # stops the run: frames are then shown but not measured (Eduardo
        # 2026-10-06; in sync with PyAO's strehl_tool)
        prep = FramePrep()

        background = None
        if self.nbg > 0:
            bgs = []
            for no in range(self.bg1, self.bg1 + self.nbg):
                try:
                    bgs.append(np.asarray(fits.getdata(
                        self._local(self._fname(no))), dtype=float))
                except Exception:
                    # find_strehl.pro: a bad background disables backgrounds
                    bgs = []
                    break
            if bgs:
                background = np.mean(bgs, axis=0)

        photrad, bgin, bgout, peakrad = self.radii
        for no, fpath in self.sequence():
            # honour a pause requested by the GUI (e.g. a duplicate-frame
            # question) BEFORE touching the next frame, so the run really
            # waits for the answer instead of queueing more results behind
            # the dialog
            if not self._wait_if_paused():
                break
            reduced = params = header = dl = None
            try:
                fpath = self._local(fpath)
                self.paths[no] = fpath
                header = fits.getheader(fpath)
                if not header_identifies(header) and \
                        str(header.get("INSTRUME", "") or "") == "":
                    # no identity at all: the summit writer fills the
                    # header in after the file has its size (frame_watch),
                    # so give it HEADER_WAIT_S to name the instrument
                    self.frame_failed.emit(
                        no, "header does not name the instrument yet -- "
                            f"waiting up to {HEADER_WAIT_S:.0f} s for it")
                    waited = wait_for_header(
                        fpath, should_stop=lambda: self._aborted)
                    if waited is not None:
                        header = waited
                # by the header's identity cards, else by the frame's file
                # name (OSIRIS daytime frames carry Keck I's CURRINST =
                # MOSFIRE and no INSTRUME; in sync with PyAO's strehl_tool)
                inst, how = identify_frame(header, os.path.basename(fpath),
                                           selected=self.kind)
                if inst is None:
                    self.frame_failed.emit(
                        no, f"{identity_text(header)}: not a known "
                            f"instrument (known: {', '.join(names())}; add "
                            "one with a parameter file)")
                    continue
                if how == "file name":
                    self.frame_failed.emit(
                        no, f"its header names no known instrument "
                            f"({identity_text(header)}) -- measured as "
                            f"{inst.name} from its file name")
                if inst.is_ifs:
                    self._measure_ifs(no, fpath, header, inst, prep)
                    continue
                if self.plane and inst.coadds_file == "unp" and \
                        int(header.get("NAXIS", 0) or 0) == 2:
                    self._measure_coadd(no, fpath, header, inst, prep,
                                        background)
                    continue
                with fits.open(fpath) as hdul:
                    raw = self._read_raw(hdul, os.path.basename(fpath))
                params = inst.frame_params(header)
                bg = background
                if bg is not None and not inst.backgrounds_supported:
                    self.frame_failed.emit(
                        no, f"{inst.name} backgrounds are not supported in "
                            "the GUI yet -- measured without")
                    bg = None
                flat, mask, cal_error, first = prep.calibration(inst)
                if cal_error is not None and first:
                    self.frame_failed.emit(no, cal_error + " -- frames are "
                                           "shown but not measured")
                # OSIRIS: the full frame for the GUI (Eduardo: show the
                # whole detector, not the tool's central-1024 crop)
                reduced = inst.reduce_frame(raw, background=bg, flat=flat,
                                            mask=mask)
                if cal_error is not None:
                    self.frame_done.emit(no, _failed(params, cal_error),
                                         params, reduced, None, header)
                    continue
                dl = prep.dl_psf(inst, params)
                result = None
                if self.autofind:
                    kwargs = self._kwargs(dl, params, bg is not None)
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
                planes, kind = plane_info(fpath, header, inst, self.ifs_bin,
                                          remote=self.remote is not None)
                self.extras[no] = {"planes": planes, "kind": kind}
                self.frame_done.emit(no, result, params, reduced, dl,
                                     header)
            except Exception as e:
                msg = f"{type(e).__name__}: {e}"
                if reduced is not None and params is not None:
                    # the frame was read: show it with the failure
                    # (Eduardo 2026-10-06) instead of nothing
                    self.frame_done.emit(no, _failed(params, msg), params,
                                         reduced, dl, header)
                else:
                    self.frame_failed.emit(no, msg)
        self.finished_all.emit()

    def _measure_ifs(self, no, fpath, header, inst, prep):
        """An OSIRIS spectrograph cube: the whole cube, or wavelength bin
        `plane` of `ifs_bin` channels (osiris_ifs: model FWHM vs the
        diffraction limit, SR relative); a raw spectrograph frame is
        refused. Records the bin's central wavelength and what the ORP
        subtracted (osiris_orp) for the GUI."""
        import os

        from .. import osiris_ifs, osiris_orp
        name = os.path.basename(fpath)
        if not osiris_ifs.is_ifs_cube(header):
            self.frame_failed.emit(
                no, f"{name}: a raw spectrograph frame (spectra, not an "
                    "image) -- open its cube in the night's SPEC/ORP")
            return
        cube = osiris_ifs.read_cube(fpath)
        bins = osiris_ifs.channel_bins(cube.nchan, self.ifs_bin)
        if self.plane > len(bins):
            raise ValueError(f"{name} has {len(bins)} wavelength bins of "
                             f"{self.ifs_bin} channels: no bin {self.plane}")
        lo, hi = (0, cube.nchan) if not self.plane else bins[self.plane - 1]
        image, valid = osiris_ifs.collapse(cube, lo, hi)
        params = osiris_ifs.ifs_frame_params(
            header, float(cube.wavelengths_um[lo:hi].mean()))
        dl = prep.dl_psf(inst, params)
        result = None
        if self.autofind:
            photrad, bgin, bgout, peakrad = self.radii
            result = osiris_ifs.measure_ifs(
                image, valid, params, photometry_radius_arcsec=photrad,
                bg_inner_arcsec=bgin, bg_outer_arcsec=bgout,
                peak_radius_arcsec=peakrad, dl_psf=dl,
                idl_version=self.idl_version)
        stem = "_".join(name.split("_")[:2])
        subtracted = osiris_orp.subtracted_frame(header)
        self.extras[no] = {
            "planes": len(bins), "kind": "ifs", "valid": valid,
            "note": osiris_ifs.bin_label(cube.wavelengths_um, lo, hi),
            "orp": osiris_orp.describe_subtraction(stem, subtracted),
            "orp_kind": osiris_orp.subtraction_kind(stem, subtracted)}
        self.frame_done.emit(no, result, params,
                             osiris_ifs.display_image(image, valid), dl,
                             header)

    def _local_unp(self, fpath):
        """The _unp file of the NIRC2 frame `fpath`, copied from the remote
        PATH first when it is not here."""
        import os
        import re

        from .. import nirc2_coadds
        unp = nirc2_coadds.unp_path(fpath)
        if unp is not None or self.remote is None:
            return unp
        from ..frame_watch import remote_fetch
        host, rdir = self.remote
        m = re.match(r"^n(\d+)\.fits$", os.path.basename(fpath))
        name = (f"n_unp_{m.group(1)}.fits" if m else
                os.path.splitext(os.path.basename(fpath))[0] + "_unp.fits")
        self.status.emit(f"copying {name} (every coadd: large) from "
                         f"{host}:{rdir} (rsync)")
        remote_fetch(host, f"{rdir}/{name}", os.path.dirname(fpath))
        return nirc2_coadds.unp_path(fpath)

    def _measure_coadd(self, no, fpath, header, inst, prep, background):
        """Coadd `plane` of a NIRC2 frame from its _unp file: the processed
        frame first (the star and its flux), then the coadd's short-exposure
        SR against it (nirc2_coadds.measure_coadd)."""
        import os
        from dataclasses import replace

        import numpy as np
        from astropy.io import fits

        from .. import nirc2_coadds
        from ..image_strehl import _failed, measure_strehl
        name = os.path.basename(fpath)
        unp = self._local_unp(fpath)
        if unp is None:
            self.frame_failed.emit(
                no, f"{name}: no _unp file beside it (n_unp_NNNN.fits): its "
                    "coadds are not on this disk")
            return
        params = inst.frame_params(header)
        n = nirc2_coadds.coadd_count(unp, expected=params.coadds)
        if self.plane > n:
            raise ValueError(f"{name}: its _unp file holds {n} complete "
                             f"coadd{'s' if n != 1 else ''}: no coadd "
                             f"{self.plane}")
        flat, mask, cal_error, first = prep.calibration(inst)
        if cal_error is not None and first:
            self.frame_failed.emit(no, cal_error + " -- frames are shown but "
                                   "not measured")
        bg = background
        frame = inst.reduce_frame(np.asarray(fits.getdata(fpath), dtype=float),
                                  background=bg, flat=flat, mask=mask)
        coadd = inst.reduce_frame(
            nirc2_coadds.read_coadd(unp, self.plane),
            background=None if bg is None else bg / max(params.coadds, 1),
            flat=flat, mask=mask)
        self.extras[no] = {"planes": n, "kind": "coadd", "frame": frame,
                           "coadds": params.coadds,
                           "note": f"coadd {self.plane}/{n}"}
        p1 = replace(params, coadds=1)
        if cal_error is not None:
            self.frame_done.emit(no, _failed(p1, cal_error), p1, coadd, None,
                                 header)
            return
        dl = prep.dl_psf(inst, params)
        result = None
        if self.autofind:
            kwargs = self._kwargs(dl, background_subtracted=bg is not None)
            kwargs.pop("psf_clean", None)
            fr = measure_strehl(frame, params=params, **kwargs)
            result = nirc2_coadds.measure_coadd(coadd, fr, params.coadds,
                                                **kwargs)
        self.frame_done.emit(no, result, p1, coadd, dl, header)


class RemoteListWorker(QThread):
    """One rsync listing of a remote ``host:/dir`` PATH for the frame list:
    emits done(path, [(name, mtime POSIX or None)], error) -- the files
    (directories left out) with their modification times, or None and the
    error. Frames are copied only when measured. In sync with PyAO."""
    done = Signal(object, object, object)

    def __init__(self, path, host, rdir, parent=None):
        super().__init__(parent)
        self.path, self.host, self.rdir = path, host, rdir

    def run(self):
        from .. import frame_watch
        try:
            rows = frame_watch._rsync_list(f"{self.host}:{self.rdir}/")
        except Exception as e:
            self.done.emit(self.path, None, f"{type(e).__name__}: {e}")
            return
        out = []
        for is_dir, name, _size, mtime in rows:
            if not is_dir:
                out.append((name, mtime.timestamp() if mtime is not None
                            else None))
        self.done.emit(self.path, out, None)


class OrpGapWorker(QThread):
    """One check of an OSIRIS IFU night for raw spectrograph frames without
    a cube (osiris_orp.check_night): emits done([(frame, reason)]) without
    the frames in `reported`, or failed(message). In sync with PyAO."""
    done = Signal(object)
    failed = Signal(str)

    def __init__(self, path, reported=(), drf_cache=None, parent=None):
        super().__init__(parent)
        self.path = path
        self.reported = set(reported)
        self.drf_cache = {} if drf_cache is None else drf_cache

    def run(self):
        from .. import frame_watch, osiris_orp
        try:
            remote = frame_watch.split_remote(self.path)
            if remote is None:
                out = osiris_orp.check_night(self.path,
                                             drf_cache=self.drf_cache,
                                             reported=self.reported)
            else:
                host, rdir = remote
                out = osiris_orp.check_night(
                    rdir, remote=host,
                    cache_dir=frame_watch.remote_cache_dir(host, rdir) + "_orp",
                    drf_cache=self.drf_cache, reported=self.reported)
        except Exception as e:
            self.failed.emit(f"{type(e).__name__}: {e}")
            return
        self.done.emit(out)


class FrameWatchWorker(QThread):
    """The IDL Strehl tool's autoimage timer: polls a frame directory and
    emits new_frame(local_path) once the last frame of instrument `kind`
    ("nirc2": n####.fits, "osiris": i<YYMMDD>_a######.fits) by name is
    complete: frame_watch.frame_ready (whole FITS blocks, size unchanged
    since the previous poll) AND frame_watch.frame_header_ready (the
    header parses and names the instrument -- the summit NIRC2 writer
    fills it in after the file has its size). A frame whose header has
    not named the instrument after frame_watch.HEADER_WAIT_S is emitted
    anyway, with a status message, so the refusal shows in the log. The
    first complete frame seen is emitted too -- the IDL tool also
    measures the current last image as soon as autoimage is switched on.

    A local PATH is polled every `interval_s`. A REMOTE one
    (``host:/abs/dir``, frame_watch.split_remote) is listed with rsync
    every frame_watch.REMOTE_POLL_S and each new frame is copied into
    frame_watch.remote_cache_dir before being emitted; listing/copy
    failures and recoveries go out as status(message), once per change.

    Runs off the GUI thread: the /s data disks are NFS hard mounts and
    rsync goes over ssh, so a stall blocks this thread, never the GUI.
    set_path() re-targets a running watcher (and forgets what it had
    emitted); request_stop() ends it within ~0.1 s of its current poll
    (or of the rsync call in flight, which carries its own timeout)."""
    new_frame = Signal(str)
    status = Signal(str)

    def __init__(self, path, kind=None, interval_s=1.0, parent=None,
                 prefix=""):
        super().__init__(parent)
        self._lock = threading.Lock()
        self._path = path
        self.kind = kind
        # ORKID: only this PREFIX's frames (frames.watch_by_prefix)
        self.prefix = prefix
        self._interval_ms = int(interval_s * 1000)
        self._stop = False

    def set_path(self, path):
        with self._lock:
            self._path = path

    def request_stop(self):
        self._stop = True

    def run(self):
        import time

        from ..frame_watch import (HEADER_WAIT_S, REMOTE_POLL_S,
                                   frame_header_ready, frame_ready,
                                   newest_frame, prune_cache,
                                   remote_cache_dir, remote_fetch,
                                   remote_newest_frame, split_remote)

        watched, emitted, pending, error = None, None, None, None
        header_wait = None      # (name, first time seen sized but headerless)
        while not self._stop:
            with self._lock:
                path = self._path
            if path != watched:
                watched, emitted, pending, error = path, None, None, None
                header_wait = None
            remote = split_remote(path)
            nf = None
            try:
                if remote is not None:
                    nf = remote_newest_frame(*remote, kind=self.kind,
                                             prefix=self.prefix)
                elif path:
                    nf = newest_frame(path, self.kind, self.prefix)
                if error is not None:
                    self.status.emit("rsync listing OK again")
                    error = None
            except OSError as e:
                msg = f"rsync listing failed: {e}"
                if msg != error:
                    self.status.emit(msg)
                    error = msg
            if nf is not None and (emitted is None or nf[0] > emitted):
                name, full, size = nf
                prev = pending[1] if pending and pending[0] == name else None
                pending = (name, size)
                if frame_ready(size, prev):
                    try:
                        if remote is not None:
                            cache = remote_cache_dir(*remote)
                            full = remote_fetch(remote[0], full, cache)
                            prune_cache(cache)
                        ready = frame_header_ready(full, self.kind)
                        if not ready:
                            # sized but not yet identifiable: keep it
                            # pending (re-checked next poll) until the
                            # header names the instrument or the wait
                            # runs out
                            if header_wait is None or header_wait[0] != name:
                                header_wait = (name, time.monotonic())
                            if time.monotonic() - header_wait[1] >= HEADER_WAIT_S:
                                self.status.emit(
                                    f"{name}: header still does not name "
                                    f"the instrument after "
                                    f"{HEADER_WAIT_S:.0f} s -- measuring "
                                    "anyway")
                                ready = True
                        if ready:
                            emitted, pending, header_wait = name, None, None
                            if not self._stop:
                                self.new_frame.emit(full)
                    except OSError as e:     # retried on the next poll
                        self.status.emit(
                            f"rsync copy of {name} failed: {e} -- retrying")
            interval_ms = (int(REMOTE_POLL_S * 1000) if remote is not None
                           else self._interval_ms)
            for _ in range(max(1, interval_ms // 100)):
                if self._stop:
                    break
                self.msleep(100)


class KeckNetworkCheck(QtCore.QObject):
    """keck_network.keck_network_check on a DAEMON Python thread -- not a
    QThread: off the Keck network the probes take seconds to time out,
    and a window closed meanwhile must neither wait for them nor abort on
    a still-running QThread. Emits done(on_keck_network, reason), queued
    onto the GUI thread."""
    done = Signal(bool, str)

    def start(self):
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        from ..keck_network import keck_network_check
        try:
            ok, reason = keck_network_check()
        except Exception as e:
            ok, reason = False, f"check failed: {type(e).__name__}: {e}"
        try:
            self.done.emit(ok, reason)
        except RuntimeError:        # the window was destroyed meanwhile
            pass


class LatestNightWorker(QThread):
    """Finds tonight's frame directory off the GUI thread, the way the
    instrument's parameter file says (instruments.find_latest_night):
    NIRC2 on the local /s disks (~1 s), OSIRIS on /s/sdata1100 directly
    or over rsync (2-20 s), an added instrument from its `night` section.
    Emits done(path_or_empty, is_tonight, error_or_empty)."""
    done = Signal(str, bool, str)

    def __init__(self, kind, parent=None):
        super().__init__(parent)
        self.kind = kind

    def run(self):
        from .. import instruments
        try:
            path, tonight = instruments.find_latest_night(
                instruments.get(self.kind))
        except Exception as e:
            self.done.emit("", False, f"{type(e).__name__}: {e}")
            return
        self.done.emit(path or "", tonight, "")


class FieldPrologueWorker(QThread):
    """parallel Phase 2 (PR-CP3): the WHOLE "Measure field" prologue --
    find_stars, the deep neighbour catalogue, the field ePSF, and (engine
    "field") solve_field -- off the GUI thread, emitting a `stage` signal
    before each step (the same wording `_nirc2_stage()` used to print)
    instead of a blocking `processEvents()` call.

    Supersedes fieldsolve P3-2's `FieldSolveWorker`: that class only
    covered the solve step (~3.6s/frame, FS-CP3b) and left find_stars/the
    catalogue/the ePSF build blocking the GUI thread behind
    `_nirc2_stage()`'s `processEvents()` calls. This is the FS-OPEN-7
    lesson applied to the whole flow (parallel PR-P0-2/P0-3), not just
    the one stage that happened to be measured first.

    `psf_clean=False` still finds stars here (off the GUI thread) but
    skips the catalogue/ePSF/solve entirely, matching the byte-identical
    default path. `solve_field` never raises for a DATA condition (an
    unusable ePSF, an empty catalogue, a non-converging solve all come
    back as a normal `FieldSolution` with `converged=False` and the
    reason in `.note`) -- `failed` is therefore only a genuine
    programming error.

    PR-CP3 acceptance (Opus, B2): `workers` reaches `solve_field`'s own
    `workers=` too, so the field solution's group-ePSF renders use the
    same persistent pool as the per-target loop instead of running
    serially on this thread -- those renders are the single largest
    parallel saving in the engine (API `solve_field` 5.31s -> 2.79s
    parallel, `results/cp2/d12/d12_mf_timeline.log`), and leaving them
    serial meant a warm-pool "Measure field" run only tied serial
    instead of beating it. Identity for `solve_field(workers=N)` is
    already asserted by `parallel_model.py` (c) (same stars, same
    models as `workers=1`) -- no new identity work needed here."""
    stage = Signal(str)
    positions_found = Signal(list)
    catalog_built = Signal(object)
    epsf_built = Signal(object)
    solution_built = Signal(object)
    failed = Signal(str)
    finished_all = Signal()

    def __init__(self, image, params, n_candidates, exclude_px, psf_clean,
                engine_choice, workers=1, parent=None):
        super().__init__(parent)
        self.image, self.params = image, params
        self.n_candidates, self.exclude_px = n_candidates, exclude_px
        self.psf_clean, self.engine_choice = psf_clean, engine_choice
        self.workers = workers

    def run(self):
        from ..epsf import build_epsf, deep_star_catalog
        from ..field_solve import solve_field
        from ..image_strehl import find_stars, sigma_filter3
        try:
            self.stage.emit("finding stars…")
            positions = find_stars(self.image, n_stars=self.n_candidates,
                                   exclude_px=self.exclude_px)
            self.positions_found.emit(positions)
            if positions and self.psf_clean:
                work = sigma_filter3(self.image)
                self.stage.emit("building the deep neighbour catalogue…")
                catalog = deep_star_catalog(work, self.params)
                self.catalog_built.emit(catalog)
                self.stage.emit(
                    f"building the field ePSF from {len(catalog)} "
                    "catalogued stars…")
                epsf = build_epsf(work, self.params, catalog=catalog)
                self.epsf_built.emit(epsf)
                if epsf.usable and self.engine_choice == "field":
                    self.stage.emit(
                        "building the field solution (one simultaneous "
                        f"solve of all {len(catalog)} stars)…")
                    solution = solve_field(work, self.params, epsf, catalog,
                                          workers=self.workers)
                    self.solution_built.emit(solution)
        except Exception as e:
            self.failed.emit(f"{type(e).__name__}: {e}")
            return
        self.finished_all.emit()


class FieldMeasureWorker(QThread):
    """parallel Phase 2 (PR-CP3): the "Measure field" per-target loop, off
    the GUI thread, using the SAME parallel machinery `measure_field`
    itself uses internally (`keck_ao_estimator.parallel.MeasureBatch`) --
    this generalizes the `FieldSolveWorker` pattern to the per-target
    measurement loop rather than replacing it with something new.

    Deliberately NOT a call to the top-level `measure_field()`: that
    function is a single opaque call with no per-result hook, and the GUI
    needs each result AS IT ARRIVES (to flash the star panel, apply the
    accept/reject/quality-gate decision, and pop it onto the map one at a
    time) plus the ability to stop after any one of them (Cancel). Both
    need a signal per target, which only the lower-level `MeasureBatch`
    (the same class `measure_field` calls) can give from outside. The
    accept/reject/quality-gate/backfill DECISION logic itself is
    unchanged and stays on the GUI thread (`_nirc2_field_on_result`) --
    this class only dispatches `measure_strehl` calls and reads them
    back in order, exactly what `measure_field`'s own loop does.

    `workers=1` runs the same calls serially, in-process, no pool --
    bit-identical to before this class existed."""
    target_result = Signal(object, int)   # (Nirc2StrehlResult, k, 1-indexed)
    finished_all = Signal()
    failed = Signal(str)

    def __init__(self, image, params, dl_psf, measure_kw, positions,
                workers, parent=None):
        super().__init__(parent)
        self.image, self.params, self.dl_psf = image, params, dl_psf
        self.measure_kw = dict(measure_kw)
        self.positions = [tuple(p) for p in positions]
        self.workers = workers
        self._stop = threading.Event()

    def request_stop(self):
        """Cancel (PLAN section 4): checked between targets, i.e. the
        loop below finishes whichever target it is currently reading
        back before honouring this -- "Cancel stops within one target"."""
        self._stop.set()

    def run(self):
        from ..image_strehl import measure_strehl
        from ..parallel import MeasureBatch
        batch = None
        try:
            if self.workers and self.workers > 1 and self.positions:
                batch = MeasureBatch(self.image, self.params, self.dl_psf,
                                     self.measure_kw, self.positions,
                                     self.workers)
            for k, pos in enumerate(self.positions, start=1):
                if self._stop.is_set():
                    break
                if batch is None:
                    r = measure_strehl(self.image, params=self.params,
                                       pos=pos, dl_psf=self.dl_psf,
                                       **self.measure_kw)
                else:
                    r = batch.result(k - 1)
                self.target_result.emit(r, k)
        except Exception as e:
            self.failed.emit(f"{type(e).__name__}: {e}")
            return
        finally:
            if batch is not None:
                # MeasureBatch.close(): cancels every not-yet-read future
                # and drains the rest -- the executor.shutdown(cancel_
                # futures=True) PLAN section 4 asks for, at the one
                # granularity the engine actually exposes (a batch, not
                # the whole persistent pool, which outlives this call).
                batch.close()
        self.finished_all.emit()
