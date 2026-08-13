"""Field-map tab: the 2-D Strehl/FWHM-vs-field-position map, its FOV/survey
controls, the DSS/2MASS or local-FITS sky backdrop, and the drawing code
for the map itself (asterism, field-of-regard, science-camera FOV, marker
overlays).

Interactive overlays (guide-star catalogue, dropped science targets) live in
FieldMapOverlaysMixin, and the cosmetic view transforms (Field-PA rotation,
backdrop/frame flip) live in FieldMapViewMixin -- both kept out of this
already-large rendering module, composed alongside it on MainWindow.
"""
import os

import numpy as np
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from qtcompat import QtCore, QtWidgets

import keck_ao_estimator as engine

from ..constants import (
    FIELD_OF_REGARD_RADIUS_ARCSEC, FM_C_FOR, FM_C_LASER,
    FM_C_STAR, FM_C_TARGET, HIPS_SURVEYS, LGS_ASTERISM_PA_DEG,
    LGS_ASTERISM_RADIUS_ARCSEC, LOCAL_BACKDROP, NIGHTTIME_FM_COND,
    NIRC2_FOVS_ARCSEC, OSIRIS_IMAGER_FOV_ARCSEC, OSIRIS_SPEC_LENSLETS,
    OSIRIS_SPEC_SCALES,
)
from ..widgets import TimeEdit, _shrinkable_label
from ..workers import SkyFetchWorker
from ...imaging import _hips2fits_url, sky_image_from_file
from ..theme import set_cue


class FieldMapMixin:
    def _build_field_map_tab(self):
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)
        top = QtWidgets.QHBoxLayout()
        self.fm_mode = QtWidgets.QComboBox()
        self.fm_mode.addItems(["NGS", "single-LGS", "LTAO"])
        self.fm_mode.setCurrentText("LTAO" if self.defaults.telescope == "K1"
                                    else "single-LGS")
        self.fm_metric = QtWidgets.QComboBox()
        # the SR-tool convention leads the FWHM entries (Eduardo
        # 2026-08-07): it is the one comparable to a measured number.
        # Strehl stays the first item overall -- it is the map's default
        # metric and this change is about ordering the FWHM conventions,
        # not about defaulting the map to FWHM.
        self.fm_metric.addItems(["Strehl", "FWHM (as the SR tool reads it)",
                                 "FWHM (half-max)",
                                 "FWHM (Gaussian fit)",
                                 "FWHM (Gaussian fit +background)"])
        self.fm_cond = QtWidgets.QComboBox()
        self.fm_cond.addItems(["observing window", "whole night", "specific time",
                               NIGHTTIME_FM_COND])
        self.fm_time = TimeEdit()             # free-typing (see widgets.py)
        self.fm_time.setEnabled(False)
        for lbl, wdg in (("Mode:", self.fm_mode), ("Metric:", self.fm_metric),
                         ("Conditions:", self.fm_cond), ("Time:", self.fm_time)):
            top.addWidget(QtWidgets.QLabel(lbl)); top.addWidget(wdg)
        self.fm_match_btn = QtWidgets.QPushButton("Match SR tool")
        self.fm_match_btn.setToolTip(
            "Set Conditions to 'specific time' at the last frame measured "
            "in the Measured SR tab AND the science wavelength to that "
            "frame's EFFWAVE, so the map shows the field at that exact "
            "image's moment and wavelength")
        self.fm_match_btn.clicked.connect(
            lambda: self._nirc2_match_tool("fm"))
        top.addWidget(self.fm_match_btn)
        self._init_fm_pa_control(top)          # FieldMapViewMixin
        top.addStretch(1)
        v.addLayout(top)

        # second row: science-camera FOV (telescope-dependent) + field of regard
        row2 = QtWidgets.QHBoxLayout()
        self._fm_fov_widgets = []          # (widget, is_label) for show/hide

        def _add(lbl_text, wdg):
            lab = QtWidgets.QLabel(lbl_text)
            row2.addWidget(lab); row2.addWidget(wdg)
            self._fm_fov_widgets += [(lab, wdg)]
            return wdg

        # K1 OSIRIS: imager vs spectrograph, and for spectrograph scale+lenslet
        self.fm_osiris_mode = QtWidgets.QComboBox()
        self.fm_osiris_mode.addItems(["imager (20×20″)", "spectrograph"])
        self.fm_osiris_scale = QtWidgets.QComboBox()
        self.fm_osiris_scale.addItems([f"{s:g}" for s in OSIRIS_SPEC_SCALES])
        self.fm_osiris_scale.setCurrentText("0.05")
        self.fm_osiris_lenslet = QtWidgets.QComboBox()
        self.fm_osiris_lenslet.addItems([str(n) for n in OSIRIS_SPEC_LENSLETS])
        self.fm_osiris_lenslet.setCurrentText("32")
        # K2 NIRC2: three square FOVs
        self.fm_nirc2_fov = QtWidgets.QComboBox()
        self.fm_nirc2_fov.addItems([f"{v:g}×{v:g}″" for v in NIRC2_FOVS_ARCSEC])

        self._fm_lab_osiris = _add("OSIRIS:", self.fm_osiris_mode)
        self._fm_lab_scale = _add("scale (″):", self.fm_osiris_scale)
        self._fm_lab_lenslet = _add("lenslet:", self.fm_osiris_lenslet)
        self._fm_lab_nirc2 = _add("NIRC2 FOV:", self.fm_nirc2_fov)

        self.fm_for = QtWidgets.QCheckBox(
            f"Field of regard ({FIELD_OF_REGARD_RADIUS_ARCSEC:g}″ radius)")
        self.fm_for.setToolTip(
            "Show the LGS tip-tilt guide-star patrol field (60″ radius). The "
            "view zooms out to the patrol circle; the science FOV stays the "
            "heat-mapped region in the centre.")
        row2.addWidget(self.fm_for)
        # TSS reachability / vignetting overlay (KAON 913). Off by default:
        # it is a MODEL, and it should not silently redraw everyone's map.
        self.fm_tss = QtWidgets.QCheckBox("TSS vignetting")
        self.fm_tss.setToolTip(
            "Overlay where the K1 tip-tilt sensor can actually be placed, and "
            "how much light it loses there (KAON 913 — a MODEL, not the "
            "measured map; the real illsubaps tables are not in the note we "
            "have). Solid ring = reachable at every rotator angle and "
            "unvignetted; dashed = the stage's longest reach, so the band "
            "between them depends on the bench angle this app does not carry; "
            "dotted = modelled vignetting contours, at the same levels the "
            "guide-star ranking charges as lost flux. LGS/LTAO only — an NGS "
            "star is sensed on the high-order WFS, not through the TSS.")
        self.fm_tss.toggled.connect(self._on_fieldmap_input_changed)
        row2.addWidget(self.fm_tss)
        self.fm_fov_readout = QtWidgets.QLabel()
        set_cue(self.fm_fov_readout, "secondary")
        _shrinkable_label(self.fm_fov_readout)
        row2.addWidget(self.fm_fov_readout, 1)
        v.addLayout(row2)

        # third row: two independent sky layers -- a survey backdrop (fills the
        # field of regard) and a local FITS frame inscribed at its native size
        row3 = QtWidgets.QHBoxLayout()
        row3.addWidget(QtWidgets.QLabel("Backdrop:"))
        self.fm_sky = QtWidgets.QComboBox()
        self.fm_sky.addItems(["off"] + list(HIPS_SURVEYS) + [LOCAL_BACKDROP])
        self.fm_sky.setToolTip(
            "Wide backdrop over the field, centred on the science field (a "
            "loaded frame's pointing, else the target). A DSS/2MASS survey "
            "fetched from CDS, or 'Local FITS…' to use your own wide image "
            "(e.g. a GSAOI mosaic) like a survey. Fills the field of regard; a "
            "loaded frame is inscribed on top at its own extent.")
        row3.addWidget(self.fm_sky)
        self.fm_sky_reload = QtWidgets.QPushButton("↻")
        self.fm_sky_reload.setFixedWidth(28)
        self.fm_sky_reload.setToolTip("re-fetch / re-pick the backdrop")
        row3.addWidget(self.fm_sky_reload)
        row3.addWidget(QtWidgets.QLabel("  Frame:"))
        self.fm_frame_btn = QtWidgets.QPushButton("Load FITS…")
        self.fm_frame_btn.setToolTip(
            "Load a local science frame (OSIRIS/NIRC2). It defines the field "
            "centre and is inscribed over the survey at its own angular size, "
            "so the surrounding field of regard stays visible in the survey.")
        row3.addWidget(self.fm_frame_btn)
        self.fm_frame_clear = QtWidgets.QPushButton("×")
        self.fm_frame_clear.setFixedWidth(24)
        self.fm_frame_clear.setToolTip("clear the loaded frame")
        row3.addWidget(self.fm_frame_clear)
        self.fm_sky_status = QtWidgets.QLabel()
        set_cue(self.fm_sky_status, "secondary")
        _shrinkable_label(self.fm_sky_status)
        row3.addWidget(self.fm_sky_status, 1)
        v.addLayout(row3)

        # fourth row: per-image WCS corrections (FieldMapViewMixin) -- flip
        # the two sky layers in X/Y (untrustworthy parity) + a manual
        # image-only PA override (wrong WCS orientation). Both are display
        # overrides for a mislabeled file, not routine controls.
        row4 = QtWidgets.QHBoxLayout()
        self._init_fm_flip_controls(row4)      # FieldMapViewMixin
        self._init_fm_img_pa_control(row4)     # FieldMapViewMixin
        row4.addStretch(1)
        v.addLayout(row4)
        # two layers: survey background (fills the FoR) + local frame inset.
        self._sky_bg_img = None; self._sky_bg_half = 0.0    # survey backdrop
        self._sky_bg_note = ""
        self._sky_fg_img = None; self._sky_fg_half = 0.0    # inscribed frame
        self._sky_fg_note = ""
        self._sky_key = None                 # survey-fetch cache key
        self._sky_worker = None
        self._sky_local_path = None          # inscribed-frame FITS path
        self._sky_bg_local_path = None       # local-FITS backdrop path (if any)
        self._sky_center = None              # frame pointing = field centre

        self._fm_holder = self._make_canvas_tab(
            "Run first — the field map needs a prepared night.")
        v.addWidget(self._fm_holder["navbar"])
        v.addWidget(self._fm_holder["widget"], 1)

        # interactive overlays (guide-star catalogue + dropped science targets
        # + the right-click drop/select menu) live in FieldMapOverlaysMixin to
        # keep this rendering module focused.
        self._init_fm_overlays(v)

        self.fm_sky.currentTextChanged.connect(self._on_survey_changed)
        self.fm_sky_reload.clicked.connect(self._on_backdrop_reload)
        self.fm_frame_btn.clicked.connect(lambda: self._load_local_sky())
        self.fm_frame_clear.clicked.connect(self._clear_frame)

        self.fm_cond.currentTextChanged.connect(
            lambda t: self.fm_time.setEnabled(t == "specific time"))
        for wdg in (self.fm_mode, self.fm_metric, self.fm_cond):
            wdg.currentTextChanged.connect(self._on_fieldmap_input_changed)
        # a Mode/Metric switch makes an existing ranking's numbers state a
        # different physical quantity entirely -- invalidate rather than show
        # stale badges/table under the new selection (see gs_ranking.py).
        # Offset/laser drags do NOT invalidate here: they're a much higher-
        # frequency, continuous interaction where requiring a fresh Rank
        # click is the expected/acceptable tradeoff (like the sensor-switch
        # invalidation in lgs.py, but that one is a discrete click too).
        self.fm_mode.currentTextChanged.connect(
            lambda *_: self._invalidate_gs_ranking())
        self.fm_metric.currentTextChanged.connect(
            lambda *_: self._invalidate_gs_ranking())
        self.fm_time.timeChanged.connect(self._on_fieldmap_input_changed)
        # the LGS-tab real-Cn² profile follows the same Conditions selector
        self.fm_cond.currentTextChanged.connect(self._update_lgs_profile_plot)
        self.fm_time.timeChanged.connect(self._update_lgs_profile_plot)
        for wdg in (self.fm_osiris_scale, self.fm_osiris_lenslet,
                    self.fm_nirc2_fov):
            wdg.currentTextChanged.connect(self._on_fieldmap_input_changed)
        self.fm_osiris_mode.currentTextChanged.connect(self._on_fm_fov_mode)
        self.fm_for.toggled.connect(self._on_fieldmap_input_changed)
        self.tel_k1.toggled.connect(self._sync_fm_fov_controls)
        self._sync_fm_fov_controls()
        return w

    def _sync_fm_fov_controls(self, *_):
        """Show only the FOV controls relevant to the active telescope /
        instrument mode (K1 OSIRIS imager|spectrograph vs K2 NIRC2), then
        refresh the resolved-FOV readout."""
        k1 = self.tel_k1.isChecked()
        spec = k1 and self.fm_osiris_mode.currentText().startswith("spectro")
        vis = {
            self._fm_lab_osiris: k1, self.fm_osiris_mode: k1,
            self._fm_lab_scale: spec, self.fm_osiris_scale: spec,
            self._fm_lab_lenslet: spec, self.fm_osiris_lenslet: spec,
            self._fm_lab_nirc2: not k1, self.fm_nirc2_fov: not k1,
        }
        for lab, wdg in self._fm_fov_widgets:
            wdg.setVisible(vis.get(wdg, True))
            lab.setVisible(vis.get(wdg, True))
        self._update_fm_fov_readout()

    def _on_fm_fov_mode(self, *_):
        self._sync_fm_fov_controls()
        self._on_fieldmap_input_changed()

    def _current_fov(self):
        """(width, height) arcsec of the selected science FOV."""
        if self.tel_k1.isChecked():
            if self.fm_osiris_mode.currentText().startswith("imager"):
                return (OSIRIS_IMAGER_FOV_ARCSEC, OSIRIS_IMAGER_FOV_ARCSEC)
            scale = float(self.fm_osiris_scale.currentText())
            nx = int(self.fm_osiris_lenslet.currentText())
            return (nx * scale, 64 * scale)
        v = float(self.fm_nirc2_fov.currentText().split("×")[0])
        return (v, v)

    def _update_fm_fov_readout(self):
        wdt, hgt = self._current_fov()
        instr = ("OSIRIS" if self.tel_k1.isChecked() else "NIRC2")
        self.fm_fov_readout.setText(f"→ {instr} FOV {wdt:g}×{hgt:g}″")

    # ---- sky-image overlay --------------------------------------------------
    def _target_radec_deg(self):
        """(ra_deg, dec_deg) parsed from the Target-tab RA/Dec fields, or None
        if unparseable. Works regardless of the target-enable toggle.

        Deliberately the BASE (pre-target-offset) position: this is the
        fallback field-map/backdrop centre, and a backdrop must stay put when
        the target offset changes (only the target marker should move within
        it) -- see _backdrop_shift_arcsec, which is what actually reflects
        the offset in the field-map drawing."""
        ra, dec = self._science_coords()
        ra, dec = ra.strip(), dec.strip()
        if not ra or not dec:
            return None
        try:
            c = engine.parse_radec(ra, dec)
            return (float(c.ra.deg), float(c.dec.deg))
        except Exception:
            return None

    def _sky_view_half(self):
        """Half-extent (arcsec) an ONLINE cutout covers. Always the field of
        regard, so it spans the FoR view and the science-FOV view (a centre
        crop) alike -- toggling the view never triggers a re-fetch."""
        return FIELD_OF_REGARD_RADIUS_ARCSEC

    def _field_center_deg(self):
        """(ra_deg, dec_deg) of the field centre: the loaded frame's pointing if
        any, else the typed (base) target. None if neither is available."""
        if self._sky_center is not None:
            return float(self._sky_center.ra.deg), float(self._sky_center.dec.deg)
        return self._target_radec_deg()

    def _backdrop_shift_arcsec(self):
        """(x, y) position of the backdrop's own reference point (a loaded
        frame's pointing, else the base typed target -- i.e. _field_center_deg)
        relative to the EFFECTIVE (offset-applied) target, in the field-map's
        plot frame (x=West+, y=North+). Drawing the backdrop shifted by this
        keeps it visually FIXED when the target offset changes -- only the
        target marker (fixed at the plot's (0, 0), and everything measured
        relative to it) moves within it, matching what a fixed real image
        must do. (0.0, 0.0) -- no shift -- if either endpoint is unavailable,
        which reproduces the pre-target-offset behaviour exactly."""
        fc = self._field_center_deg()
        eff = self._effective_target_coords()
        if fc is None or eff is None:
            return (0.0, 0.0)
        try:
            target = engine.parse_radec(*eff)
            from astropy.coordinates import SkyCoord
            import astropy.units as u
            center = SkyCoord(fc[0] * u.deg, fc[1] * u.deg)
            dlon, dlat = target.spherical_offsets_to(center)
            return (-float(dlon.arcsec), float(dlat.arcsec))
        except Exception:
            return (0.0, 0.0)

    def _update_sky_status(self):
        parts = [p for p in (self._sky_bg_note if self._sky_bg_img is not None
                             else "",
                             self._sky_fg_note if self._sky_fg_img is not None
                             else "") if p]
        self.fm_sky_status.setText("  ·  ".join(parts))

    def _on_survey_changed(self, text):
        if text == "off":
            self._sky_bg_img = None; self._sky_key = None; self._sky_bg_note = ""
            self._sky_bg_local_path = None
            self._update_sky_status(); self._on_fieldmap_input_changed()
            return
        if text == LOCAL_BACKDROP:
            self._load_bg_local()
            return
        self._sky_bg_local_path = None       # a survey supersedes a local backdrop
        self._load_sky()

    def _on_backdrop_reload(self):
        """↻: re-fetch a survey backdrop, or re-pick the local-FITS backdrop."""
        if self.fm_sky.currentText() == LOCAL_BACKDROP:
            self._load_bg_local()            # re-open the file dialog
        else:
            self._load_sky(force=True)

    def _refresh_backdrop(self, force=False):
        """Re-apply the active backdrop after the field centre or view size
        changes: re-resample a local-FITS backdrop, or re-fetch a survey."""
        if self._sky_bg_local_path is not None:
            self._load_bg_local(self._sky_bg_local_path)
        elif self.fm_sky.currentText() in HIPS_SURVEYS:
            self._load_sky(force=force)

    def _load_bg_local(self, path=None):
        """Load a local wide FITS as the BACKDROP -- resampled by its own WCS
        onto the field grid centred on the science field (a loaded frame's
        pointing, else the typed target, else the image's own centre) at field-
        of-regard size. Lets a wide image (e.g. a GSAOI mosaic) be used like a
        DSS/2MASS survey. Independent of whether a frame is inscribed."""
        if path is None:
            path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, "Load backdrop image", "",
                "Sky image (*.fits *.fit *.fits.gz *.png);;FITS (*.fits *.fit "
                "*.fits.gz);;PNG with WCS (*.png)")
        if not path:
            # cancelled with no active local backdrop -> revert the combo to off
            if self._sky_bg_local_path is None:
                self.fm_sky.blockSignals(True)
                self.fm_sky.setCurrentText("off")
                self.fm_sky.blockSignals(False)
            return
        try:
            from astropy.coordinates import SkyCoord
            import astropy.units as u
            rd = self._field_center_deg()
            ctr = SkyCoord(rd[0] * u.deg, rd[1] * u.deg) if rd is not None else None
            half = self._sky_view_half()             # field of regard
            arr, _note, _used, _obs, _nm, bhalf = sky_image_from_file(
                path, center=ctr, half=half)
            if not np.any(np.isfinite(arr)):
                raise ValueError("backdrop has no data over the field centre")
            self._sky_bg_img = arr
            self._sky_bg_half = bhalf
            self._sky_bg_local_path = path
            self._sky_key = None                     # not a survey-cache entry
            where = ("frame" if self._sky_center is not None
                     else "target" if rd is not None else "image centre")
            self._sky_bg_note = (f"backdrop {os.path.basename(path)} "
                                 f"(centred on {where})")
            if self.fm_sky.currentText() != LOCAL_BACKDROP:
                self.fm_sky.blockSignals(True)
                self.fm_sky.setCurrentText(LOCAL_BACKDROP)
                self.fm_sky.blockSignals(False)
        except Exception as e:
            self._sky_bg_img = None; self._sky_bg_local_path = None
            self._sky_bg_note = f"backdrop load failed: {e}"
        self._update_sky_status(); self._on_fieldmap_input_changed()

    def _sky_field_center(self):
        """(ra, dec) strings of the field centre when a loaded frame defines it
        (its pointing), else None -> the typed science target is used."""
        if self._sky_center is None:
            return None
        # hms/dms so SkyCoord() infers the units, matching the Target-tab format
        return (self._sky_center.ra.to_string(unit="hourangle", sep="hms"),
                self._sky_center.dec.to_string(unit="deg", sep="dms"))

    def _load_sky(self, force=False):
        """Fetch the selected survey backdrop for the field centre at the field-
        of-regard size, off-thread. Skips the fetch if the cache matches."""
        survey = self.fm_sky.currentText()
        if survey not in HIPS_SURVEYS:
            return
        rd = self._field_center_deg()
        if rd is None:
            self._sky_bg_note = "survey needs a target/frame centre"
            self._update_sky_status()
            return
        half = self._sky_view_half()             # field of regard
        key = (round(rd[0], 5), round(rd[1], 5), survey, round(half, 1))
        if not force and key == self._sky_key and self._sky_bg_img is not None:
            return
        if self._sky_worker is not None:
            return
        url = _hips2fits_url(HIPS_SURVEYS[survey], rd[0], rd[1],
                             (2 * half) / 3600.0)
        self._sky_bg_note = f"fetching {survey}…"; self._update_sky_status()
        self._sky_pending_key = key
        self._sky_worker = SkyFetchWorker(url, half, self)
        self._sky_worker.done.connect(self._on_sky_fetched)
        self._sky_worker.finished.connect(self._on_sky_worker_cleanup)
        self._sky_worker.start()

    def _on_sky_worker_cleanup(self):
        self._sky_worker = None

    def _on_sky_fetched(self, arr, half, err):
        if arr is None:
            self._sky_bg_note = f"survey fetch failed: {err[:40]}"
            self._update_sky_status()
            return
        self._sky_bg_img = arr
        self._sky_bg_half = float(half)
        self._sky_key = getattr(self, "_sky_pending_key", None)
        self._sky_bg_note = (f"survey {self.fm_sky.currentText()} "
                             f"({arr.shape[1]}px)")
        self._update_sky_status(); self._on_fieldmap_input_changed()

    def _clear_frame(self):
        """Remove the inscribed local frame. Any backdrop stays, re-centred on
        the target now that no frame defines the field centre."""
        self._sky_fg_img = None; self._sky_local_path = None
        self._sky_center = None; self._sky_fg_note = ""
        self._refresh_backdrop(force=True)
        self._update_sky_status(); self._on_fieldmap_input_changed()

    def _load_local_sky(self, path=None):
        """Load a local FITS as the inscribed foreground frame. Its OWN pointing
        defines the field centre and it is shown at its native angular size
        (survey, if any, fills the surround). Also makes the frame's target the
        active target and drives the snapshot from the frame timestamp."""
        if path is None:
            path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, "Load image frame", "",
                "Sky image (*.fits *.fit *.fits.gz *.png)")
        if not path:
            return
        try:
            arr, note, center, obs_hst, tname, half = sky_image_from_file(path)
            if not np.any(np.isfinite(arr)):
                raise ValueError("frame has no data over its own centre")
            self._sky_fg_img = arr
            self._sky_fg_half = half
            self._sky_local_path = path
            self._sky_center = center
            # the frame's target joins tonight's target list AND becomes the
            # active target (its coordinates fill the Target-tab fields)
            fc = self._sky_field_center()
            self._add_target(tname or os.path.splitext(os.path.basename(path))[0],
                             fc[0], fc[1], select=True)
            tnote = ""
            if obs_hst is not None:
                # drive the field-map snapshot from the frame's own timestamp
                for w in (self.fm_cond, self.fm_time):
                    w.blockSignals(True)
                self.fm_cond.setCurrentText("specific time")
                self.fm_time.setEnabled(True)
                # the field enters wall time in the DISPLAY zone: UT clock
                # when UTC mode is on, HST otherwise
                h_disp = (obs_hst.hour + 10) % 24 if self._utc() \
                    else obs_hst.hour
                self.fm_time.setTime(QtCore.QTime(h_disp, obs_hst.minute))
                for w in (self.fm_cond, self.fm_time):
                    w.blockSignals(False)
                tnote = f", {self._fmt_hm(obs_hst)}"
            self._sky_fg_note = (
                f"frame {os.path.basename(path)} ({2*half:.0f}″), centre "
                f"{center.to_string('hmsdms', sep=':', precision=0)}{tnote}")
            # re-centre the backdrop (survey or local FITS) on the frame
            # pointing so the two layers align
            self._refresh_backdrop(force=True)
        except Exception as e:
            self._sky_fg_img = None; self._sky_local_path = None
            self._sky_center = None
            self._sky_fg_note = f"FITS load failed: {str(e)[:50]}"
        self._update_sky_status(); self._on_fieldmap_input_changed()

    def _field_map_visible(self):
        return (hasattr(self, "plot_tabs")
                and self.plot_tabs.currentIndex() == 1)

    def _on_fieldmap_input_changed(self, *_):
        """A field-map control or a reference position changed: mark stale and,
        if the tab is showing, throttle a redraw. Reference positions (offset
        entries, laser PA) don't change the science estimate, only the map.
        Throttled rather than debounced: a redraw re-evaluates the whole grid
        (tens-to-~130 ms), so rendering on every tick while a spinbox button
        is held would synchronously block the UI thread each time and feel
        choppy -- but resetting a timer on every tick (a plain debounce) never
        fires while ticks keep arriving faster than the interval, so holding
        the button would show nothing move until release. Only arming the
        timer when it isn't already running gives one redraw roughly every
        150 ms for the whole duration of the hold -- smooth AND live."""
        self._update_fm_fov_readout()
        self._fieldmap_dirty = True
        if self._field_map_visible():
            if not self._fm_debounce.isActive():
                self._fm_debounce.start()
            # (re)arm the LOD settle timer: while it's running the redraws
            # above render a coarse grid; it fires one full-res redraw once
            # input goes quiet (see _render_field_map / _fm_settle).
            self._fm_settle.start()

    def _render_field_map_if_visible(self):
        if self._field_map_visible() and getattr(self, "_fieldmap_dirty", True):
            self._render_field_map()

    def _render_field_map_full(self):
        """Trailing full-resolution redraw after a scrub settles. _fm_settle
        has just fired (so it's no longer active), so _render_field_map picks
        the full grid. Re-dirty first: the last coarse redraw during the hold
        cleared the flag."""
        self._fieldmap_dirty = True
        self._render_field_map_if_visible()

    @staticmethod
    def _metric_style(metric):
        """(cmap, colorbar label, display name) for a field-map metric."""
        if metric == "fwhm":
            return "viridis_r", "FWHM (mas)", "FWHM (half-max)"
        if metric == "fwhm_gaussfit":
            return "viridis_r", "FWHM (mas)", "FWHM (Gaussian fit)"
        if metric == "fwhm_gaussfit_sky":
            return "viridis_r", "FWHM (mas)", "FWHM (Gaussian fit +background)"
        if metric == "fwhm_srtool":
            return "viridis_r", "FWHM (mas)", "FWHM (as the SR tool reads it)"
        return "viridis", "Strehl", "Strehl"

    def _fm_title_text(self, snap, metric_name, meta, for_on, sci_fov):
        """The two-line field-map axes title (telescope/mode/metric + the
        conditions the snapshot was taken under). Shared by the full and the
        interactive renderers so the title never differs between them."""
        tel = self._gui_telescope()
        R_for = FIELD_OF_REGARD_RADIUS_ARCSEC
        if for_on:
            head = (f"{tel} {self.fm_mode.currentText()} {metric_name} over the "
                    f"{R_for:g}″ field of regard (FOV {sci_fov[0]:g}×"
                    f"{sci_fov[1]:g}″)\n")
        else:
            head = (f"{tel} {self.fm_mode.currentText()} {metric_name} across "
                    f"the {sci_fov[0]:g}×{sci_fov[1]:g}″ field\n")
        if snap.get("synthetic"):
            return (head + f"{meta['lam_label']} · PREDICTED SCENARIO — "
                    f"DIMM {snap['eps_tot_zenith']:.2f}″ / "
                    f"free-atm {snap['eps_fa_zenith']:.2f}″ at zenith, "
                    f"θ₀ᴷ {snap['theta0_k_zenith']:.1f}″, "
                    f"ZA {snap['zenith_angle_deg']:g}° (X={snap['airmass']:.2f})")
        return (head + f"{meta['lam_label']} · "
                f"{self._tz_text(snap['when_desc'])} "
                f"({self._fmt_hm(snap['t_hst'])}, X={snap['airmass']:.2f}, "
                f"seeing {snap['eps_tot_los']:.2f}″, θ₀ {snap['theta0_los']:.1f}″)")

    def _fm_budget_note(self, mode, ngs_dvar):
        """The 'MODIFIED BUDGET' flag text for a what-if map (or None if the
        budget is untouched). Matches _draw_field_map's wording exactly."""
        if not self.last_offsets:
            return None
        if mode == "ngs":
            if ngs_dvar:
                eff = float(np.sqrt(abs(ngs_dvar)))
                sign = "⊕" if ngs_dvar > 0 else "⊖"
                return f"MODIFIED BUDGET — projected NGS ({sign}{eff:.0f} nm quad)"
            return ("MODIFIED BUDGET — no NGS-projecting change "
                    "(moved terms are LGS-only)")
        return "MODIFIED BUDGET"

    def _build_fm_live_scaffold(self, key, extent, Z, cmap, clabel, for_on,
                                sci_fov, mode):
        """Build the persistent parts of the interactive field map ONCE: the
        figure, axes, heatmap image, colorbar, static decorations (axis
        labels, grid, field-of-regard circle/box, N/E badge) and a stable
        proxy legend. Per-frame updates (heatmap data, moving markers,
        contours, title) happen in _draw_field_map_live. Returns the handle
        dict cached as self._fm_live and rebuilt whenever `key` changes."""
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        from matplotlib.patches import Circle, Rectangle
        fig = Figure(figsize=(6.5, 6), layout="constrained")
        ax = fig.add_subplot(111)
        ax.set_anchor("C")
        trans = self._fm_rotation_transform(ax)
        im = ax.imshow(Z, extent=extent, origin="lower", cmap=cmap,
                       aspect="equal", zorder=2, transform=trans)
        cax = make_axes_locatable(ax).append_axes("right", size="4%", pad=0.12)
        fig.colorbar(im, cax=cax, label=clabel)
        R_for = FIELD_OF_REGARD_RADIUS_ARCSEC
        half_sx, half_sy = sci_fov[0] / 2.0, sci_fov[1] / 2.0
        if for_on:
            view = R_for * 1.04
            ax.add_patch(Circle((0, 0), R_for, fill=False, ls="-", lw=1.3,
                                 ec=FM_C_FOR, alpha=0.9, zorder=4, transform=trans))
            ax.add_patch(Rectangle((-half_sx, -half_sy), 2 * half_sx,
                                   2 * half_sy, fill=False, ls="-", lw=1.2,
                                   ec=FM_C_FOR, alpha=0.9, zorder=4,
                                   transform=trans))
            ax.set_xlim(-view, view); ax.set_ylim(-view, view)
        else:
            half_gx, half_gy = (extent[1] - extent[0]) / 2.0, (extent[3] - extent[2]) / 2.0
            if self._fm_pa_deg() != 0.0 and (half_gx != half_gy):
                r = float(np.hypot(half_gx, half_gy))
                ax.set_xlim(-r, r); ax.set_ylim(-r, r)
            else:
                ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
        ax.set_xlabel("← East      offset (arcsec)      West →", fontsize=9)
        ax.set_ylabel("← South     offset (arcsec)     North →", fontsize=9)
        self._draw_compass(ax, trans)
        if self._sky_center is not None:
            ax.annotate("field defined by loaded image", xy=(0.5, 0.015),
                        xycoords="axes fraction", ha="center", fontsize=7,
                        color="w", bbox=dict(boxstyle="round,pad=0.2",
                                             fc="black", alpha=0.4, ec="none"))
        ax.grid(alpha=0.15)
        # a stable proxy legend (the real, value-bearing one is drawn on the
        # trailing full render; here it must not churn as markers move)
        star_lbl = "NGS star" if mode == "ngs" else "TT star"
        proxies = [Line2D([], [], marker="*", ls="", ms=12, mfc=FM_C_TARGET,
                          mec="k", label="field centre"),
                   Line2D([], [], marker="o", ls="", mfc=FM_C_STAR, mec="k",
                          label=star_lbl)]
        if mode == "single":
            proxies.append(Line2D([], [], marker="D", ls="", mfc=FM_C_LASER,
                                  mec="k", label="laser (589 nm)"))
        elif mode == "ltao":
            proxies.append(Line2D([], [], marker="D", ls="", mfc=FM_C_LASER,
                                  mec="k", label="4-LGS asterism"))
        if for_on:
            proxies.append(Line2D([], [], ls="-", color=FM_C_FOR,
                                  label=f"field of regard ({R_for:g}″) + FOV"))
        ax.legend(handles=proxies, loc="lower right", fontsize=7, framealpha=0.9)
        return {"key": key, "fig": fig, "ax": ax, "im": im, "dyn": []}

    def _draw_field_map_live(self, extent, Z, meta, mode, metric, snap,
                             ngs_xy, tt_xy, laser_xy, sci_fov, ngs_dvar):
        """Fast in-place redraw used WHILE a control is scrubbed. Reuses the
        persistent scaffold and only updates the heatmap (im.set_data is ~free)
        plus the moving markers and unlabeled contours -- deliberately lighter
        than _draw_field_map (no contour labels, no clamped-marker arrows),
        since the trailing settle frame re-renders the full, polished map."""
        for_on = self.fm_for.isChecked()
        fov_x, fov_y = meta["fov_x"], meta["fov_y"]
        cmap, clabel, metric_name = self._metric_style(metric)
        is_fwhm = metric != "strehl"
        # rebuild the scaffold whenever anything STATIC in it would change
        # (axes limits, colorbar, FOV box, image-defined badge); per-frame
        # dynamic bits below don't enter the key.
        key = (mode, metric, for_on, round(fov_x, 3), round(fov_y, 3), cmap,
               round(sci_fov[0], 3), round(sci_fov[1], 3),
               self._sky_center is not None, self._gui_telescope(),
               round(self._fm_pa_deg(), 1))
        live = getattr(self, "_fm_live", None)
        if live is None or live["key"] != key:
            live = self._build_fm_live_scaffold(key, extent, Z, cmap, clabel,
                                                for_on, sci_fov, mode)
            self._fm_live = live
        ax, im = live["ax"], live["im"]
        trans = self._fm_rotation_transform(ax)
        # the whole point: update the heatmap in place instead of rebuilding
        im.set_data(Z); im.set_extent(extent)
        if np.isfinite(Z).any():
            vmin, vmax = float(np.nanmin(Z)), float(np.nanmax(Z))
            if vmax > vmin:
                im.set_clim(vmin, vmax)
        # drop last frame's dynamic artists, redraw the light ones
        for a in live["dyn"]:
            try:
                a.remove()
            except Exception:
                pass
        dyn = []
        xg = np.linspace(extent[0], extent[1], Z.shape[1])
        yg = np.linspace(extent[2], extent[3], Z.shape[0])
        dyn.append(ax.contour(xg, yg, Z, colors="white", linewidths=0.5,
                              alpha=0.6, zorder=3, transform=trans))
        tval = meta["target"]
        tstr = f"{tval:.0f} mas" if is_fwhm else f"{tval:.3f}"
        dyn += ax.plot(0, 0, "*", ms=16, mfc=FM_C_TARGET, mec="k", mew=0.8,
                       zorder=7, transform=trans)
        dyn.append(ax.annotate(f"centre: {tstr}", xy=(0, 0), xycoords=trans,
                               textcoords="offset points", xytext=(8, -12),
                               fontsize=8, fontweight="bold", color=FM_C_TARGET,
                               zorder=8, bbox=dict(boxstyle="round,pad=0.2",
                               fc="white", ec=FM_C_TARGET, lw=0.8, alpha=0.9)))
        view = (FIELD_OF_REGARD_RADIUS_ARCSEC * 1.04) if for_on else None
        clip_x = view if for_on else fov_x / 2.0
        clip_y = view if for_on else fov_y / 2.0
        star_xy = ngs_xy if mode == "ngs" else tt_xy
        star_lbl = "NGS star" if mode == "ngs" else "TT star"
        dyn += self._mark_ref(ax, star_xy, clip_x, clip_y, star_lbl, FM_C_STAR,
                              transform=trans)
        if mode == "single":
            dyn += self._mark_ref(ax, (laser_xy[0], laser_xy[1], True),
                                  clip_x, clip_y, "laser (589 nm)", FM_C_LASER,
                                  marker="D", transform=trans)
        elif mode == "ltao":
            dyn += self._draw_asterism(ax, laser_xy, transform=trans)
        dyn += self._draw_tss_vignetting(ax, mode, transform=trans)
        dyn += self._draw_catalog_stars(ax, transform=trans)
        dyn += self._draw_gs_ranking_badges(ax, transform=trans)
        dyn += self._draw_fm_targets(ax, is_fwhm, transform=trans)
        note = self._fm_budget_note(mode, ngs_dvar)
        if note:
            dyn.append(ax.annotate(note, xy=(0.03, 0.03),
                       xycoords="axes fraction", fontsize=8, fontweight="bold",
                       color="#C0392B", va="bottom",
                       bbox=dict(boxstyle="round,pad=0.25", fc="white",
                                 alpha=0.85, ec="#C0392B")))
        ax.set_title(self._fm_title_text(snap, metric_name, meta, for_on,
                                         sci_fov), fontsize=9, fontweight="bold",
                     color="#B34700" if snap.get("synthetic") else "black")
        self._fm_refresh_target_list(is_fwhm)
        live["dyn"] = dyn
        holder = self._fm_holder
        if holder["canvas"].figure is not live["fig"]:
            self._show_figure(holder, live["fig"])   # swap the live fig in
        else:
            holder["canvas"].draw_idle()

    def _laser_xy(self):
        """Laser position (x=West+, y=North+) = radial at the LGS-offset
        magnitude, position angle from laser_pa (N->E; default
        engine.DEF_LASER_PA_DEG = the K1 campaign direction)."""
        tel = "K1" if self.tel_k1.isChecked() else "K2"
        r = (self.lgs_offset.value() if self.lgs_offset_enable.isChecked()
             else engine.DEF_LGS_OFFSET[tel])
        pa = np.radians(self.laser_pa.value())
        return (-r * np.sin(pa), r * np.cos(pa))     # x=-East(=West+), y=North

    def _render_field_map(self):
        self._fieldmap_dirty = False
        holder = self._fm_holder
        pred_on = self.pred_enable.isChecked()
        if (self.prep is None or self.res is None) and not pred_on:
            self._show_placeholder(
                holder, "Run first — or enable the Prediction tab's scenario "
                        "to map hypothetical conditions with no night loaded.")
            return
        if pred_on:
            # hypothetical scenario from the Prediction tab: the synthetic
            # snapshot supplies seeing/theta0/Cn2, so no MASS data — and,
            # since 2026-08-12, no Run — is needed: args/prep fall back to
            # widget-collected values and a minimal surrogate.
            fm_args, fm_prep = self._fm_args(), self._fm_prep()
            if fm_args is None:
                self._show_placeholder(
                    holder, "Prediction scenario: fix the control inputs "
                            "(a field could not be parsed).")
                return
            snap = self._pred_snapshot()
        else:
            fm_args, fm_prep = self.args_cached, self.prep
            when, t_hst = self._fm_when_time()
            snap = engine.field_snapshot(self.args_cached, self.prep, self.res,
                                         when, t_hst)
            if snap is None:
                self._show_placeholder(
                    holder, "Field map needs MASS profiles (none this night).")
                return
        mode = {"NGS": "ngs", "single-LGS": "single",
                "LTAO": "ltao"}[self.fm_mode.currentText()]
        metric = {"Strehl": "strehl", "FWHM (half-max)": "fwhm",
                  "FWHM (Gaussian fit)": "fwhm_gaussfit",
                  "FWHM (Gaussian fit +background)": "fwhm_gaussfit_sky",
                  "FWHM (as the SR tool reads it)": "fwhm_srtool"}[
                      self.fm_metric.currentText()]
        # when a real image defines the field, star-coord offsets are measured
        # from the image pointing, not the typed target
        fc = self._sky_field_center()
        ngs_xy = self.ngs_offset.offset_xy(fc)
        tt_xy = self.tt_offset.offset_xy(fc)
        laser_xy = self._laser_xy()
        # the field of regard maps performance over the WHOLE patrol field
        # (a denser grid keeps the wider map smooth); otherwise just the
        # science FOV. The science FOV is drawn as a box either way.
        # While a control is being scrubbed (_fm_settle running) drop to a
        # coarse grid so each frame is cheap enough to feel live; the trailing
        # full-res redraw fires once input settles (odd counts keep an exact
        # centre pixel, so the reported on-axis value never jumps coarse<->full).
        interactive = self._fm_settle.isActive()
        sci_fov = self._current_fov()
        if self.fm_for.isChecked():
            grid_fov = (2 * FIELD_OF_REGARD_RADIUS_ARCSEC,) * 2
            n_grid = 35 if interactive else 71
        else:
            grid_fov = sci_fov
            n_grid = 21 if interactive else 41
        # NGS budget what-if: the empirical NGS model contains the budget
        # implicitly, so slider changes reach the NGS map as the same Marechal
        # variance swap the timeline's projected-NGS overlay uses (LGS/LTAO
        # maps take the overridden budget directly instead).
        fm_dvar = (self._ngs_delta_var(self.last_offsets, fm_args)
                   if (mode == "ngs" and self.last_offsets) else 0.0)
        try:
            # field_map_grid recomputes Strehl from the module-level budget, so
            # re-apply the WFE overrides the visible result was computed under --
            # otherwise the map ignores the sliders (they've been restored by the
            # time recompute_and_draw's budget_overrides block exits).
            with engine.budget_overrides(**self.last_offsets):
                extent, Z, meta = engine.field_map_grid(
                    fm_args, fm_prep, snap, mode, metric,
                    ngs_xy[:2], tt_xy[:2], laser_xy, n_grid=n_grid,
                    fov=grid_fov, ngs_delta_var=fm_dvar)
                # per-target values (exact single points, same budget context)
                self._fm_eval_markers(snap, mode, metric, ngs_xy, tt_xy,
                                      laser_xy, fm_dvar,
                                      args=fm_args, prep=fm_prep)
        except Exception as e:
            self._on_failed(f"field map failed: {e}", "")
            return
        # while scrubbing (interactive) and drawing the plain filled heatmap
        # (no sky overlay), take the fast in-place path: reuse a persistent
        # figure and only update the heatmap + moving markers (~free) instead
        # of rebuilding every artist (~40 ms, dominated by colorbar creation
        # and contour labels). Any failure falls back to the full renderer.
        has_sky = self._sky_bg_img is not None or self._sky_fg_img is not None
        if interactive and not has_sky:
            try:
                self._draw_field_map_live(extent, Z, meta, mode, metric, snap,
                                          ngs_xy, tt_xy, laser_xy, sci_fov,
                                          fm_dvar)
                return
            except Exception:
                self._fm_live = None      # drop a possibly-broken scaffold
        # constrained layout re-solves on every draw, so the axis labels stay
        # inside the canvas whatever size/aspect the dock gives it (a one-shot
        # tight_layout() at creation size leaves the xlabel clipped)
        fig = Figure(figsize=(6.5, 6), layout="constrained")
        bg_shift = self._backdrop_shift_arcsec()
        self._draw_field_map(fig, extent, Z, meta, mode, metric, snap,
                             ngs_xy, tt_xy, laser_xy, sci_fov,
                             ngs_dvar=fm_dvar, bg_shift=bg_shift)
        self._show_figure(holder, fig)

    def _draw_field_map(self, fig, extent, Z, meta, mode, metric, snap,
                        ngs_xy, tt_xy, laser_xy, sci_fov, ngs_dvar=0.0,
                        bg_shift=(0.0, 0.0)):
        ax = fig.add_subplot(111)
        trans = self._fm_rotation_transform(ax)
        for_on = self.fm_for.isChecked()
        R_for = FIELD_OF_REGARD_RADIUS_ARCSEC
        half_gx, half_gy = meta["fov_x"] / 2.0, meta["fov_y"] / 2.0  # grid extent
        half_sx, half_sy = sci_fov[0] / 2.0, sci_fov[1] / 2.0        # science FOV
        # markers clip to the plotted view (field of regard, or the FOV)
        view = (R_for * 1.04) if for_on else None
        clip_x = view if for_on else half_gx
        clip_y = view if for_on else half_gy
        is_fwhm = metric != "strehl"
        cmap, clabel, metric_name = self._metric_style(metric)
        # sky layers: a survey backdrop (fills the field of regard) with a local
        # science frame inscribed on top at its native size (NaN outside the
        # frame lets the survey show through). Over an image a filled heatmap is
        # unreadable, so the performance is CONTOUR LINES only; with neither
        # layer it is the usual filled heatmap.
        has_sky = self._sky_bg_img is not None or self._sky_fg_img is not None
        xg = np.linspace(extent[0], extent[1], Z.shape[1])
        yg = np.linspace(extent[2], extent[3], Z.shape[0])
        if has_sky:
            bx, by = bg_shift    # backdrop's own centre, relative to the
                                 # target-at-(0,0) plot frame -- keeps a
                                 # fixed backdrop fixed when the target
                                 # offset moves the marker instead (see
                                 # _backdrop_shift_arcsec)
            # imagery (backdrop/frame + the frame outline) is drawn under the
            # IMAGE transform: the Field-PA rotation PLUS the manual image-only
            # PA override (about the image centre bx,by), so a wrong-WCS image
            # can be turned to match the catalogue. The contours/markers below
            # stay on `trans` (Field PA only), so the image moves relative to
            # them. At img-PA 0 img_trans == trans (no change).
            img_trans = self._fm_image_transform(ax, (bx, by))
            def _draw_layer(img, R, z, flip_x=False, flip_y=False):
                # user-requested mirror for an untrustworthy WCS parity --
                # see FieldMapViewMixin._fm_flip_image
                img = self._fm_flip_image(img, flip_x, flip_y)
                finite = img[np.isfinite(img)]
                if finite.size:
                    vmin, vmax = np.percentile(finite, [1.0, 99.5])
                    if vmax <= vmin:
                        vmin, vmax = float(finite.min()), float(finite.max()) or 1.0
                else:
                    vmin, vmax = 0.0, 1.0
                ax.imshow(img, extent=[bx - R, bx + R, by - R, by + R],
                          origin="upper", cmap="gray", aspect="equal",
                          zorder=z, vmin=vmin, vmax=vmax, transform=img_trans)
            if self._sky_bg_img is not None:              # survey backdrop
                _draw_layer(self._sky_bg_img, self._sky_bg_half, 0,
                           self.fm_bg_flip_x.isChecked(),
                           self.fm_bg_flip_y.isChecked())
            if self._sky_fg_img is not None:              # inscribed frame
                _draw_layer(self._sky_fg_img, self._sky_fg_half, 1,
                           self.fm_fg_flip_x.isChecked(),
                           self.fm_fg_flip_y.isChecked())
                # outline the inscribed frame so its edge reads clearly
                fh = self._sky_fg_half
                from matplotlib.patches import Rectangle as _Rect
                ax.add_patch(_Rect((bx - fh, by - fh), 2 * fh, 2 * fh, fill=False,
                                   ec="#39C", lw=0.8, ls=":", alpha=0.7,
                                   zorder=2, transform=img_trans))
            cs = ax.contour(xg, yg, Z, levels=10, cmap=cmap, linewidths=1.2,
                            zorder=3, transform=trans)
            ax.clabel(cs, inline=True, fontsize=6,
                      fmt="%.0f" if is_fwhm else "%.2f")
            mappable = cs
        else:
            mappable = ax.imshow(Z, extent=extent, origin="lower", cmap=cmap,
                                 aspect="equal", zorder=2, transform=trans)
            cs = ax.contour(xg, yg, Z, colors="white", linewidths=0.5,
                            alpha=0.6, zorder=3, transform=trans)
            ax.clabel(cs, inline=True, fontsize=6,
                      fmt="%.0f" if is_fwhm else "%.2f")
        # equal aspect makes the axes box square; anchor it centre and hang the
        # colorbar off the axes (a divider axes that tracks the shrunk box) so
        # the plot+bar stay centred in the (wide, short) shared canvas instead
        # of being shoved to the right edge
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        ax.set_anchor("C")
        cax = make_axes_locatable(ax).append_axes("right", size="4%", pad=0.12)
        fig.colorbar(mappable, cax=cax, label=clabel)

        # field of regard: the 60" guide-star patrol circle, and the science
        # FOV drawn as a box so it is visible at the zoomed-out scale
        if for_on:
            from matplotlib.patches import Circle, Rectangle
            ax.add_patch(Circle((0, 0), R_for, fill=False, ls="-", lw=1.3,
                                 ec=FM_C_FOR, alpha=0.9, zorder=4,
                                 transform=trans))
            ax.add_patch(Rectangle((-half_sx, -half_sy), 2 * half_sx, 2 * half_sy,
                                    fill=False, ls="-", lw=1.2, ec=FM_C_FOR,
                                    alpha=0.9, zorder=4, transform=trans))
            ax.plot([], [], "-", color=FM_C_FOR,
                    label=f"field of regard ({R_for:g}″) + FOV")

        # reference markers: field centre (blue), offset star (orange), laser
        # (yellow). Report the on-axis metric AT the field centre on its marker;
        # user-dropped science targets (green) each report their own value.
        tval = meta["target"]
        tstr = f"{tval:.0f} mas" if is_fwhm else f"{tval:.3f}"
        ax.plot(0, 0, "*", ms=16, mfc=FM_C_TARGET, mec="k", mew=0.8, zorder=7,
                label=f"field centre — {tstr}", transform=trans)
        ax.annotate(f"centre: {tstr}", xy=(0, 0), xycoords=trans,
                    textcoords="offset points",
                    xytext=(8, -12), fontsize=8, fontweight="bold",
                    color=FM_C_TARGET, zorder=8,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white",
                              ec=FM_C_TARGET, lw=0.8, alpha=0.9))
        star_xy = ngs_xy if mode == "ngs" else tt_xy
        star_lbl = "NGS star" if mode == "ngs" else "TT star"
        self._mark_ref(ax, star_xy, clip_x, clip_y, star_lbl, FM_C_STAR,
                       transform=trans)
        if mode == "single":
            self._mark_ref(ax, (laser_xy[0], laser_xy[1], True), clip_x, clip_y,
                           "laser (589 nm)", FM_C_LASER, marker="D",
                           transform=trans)
        elif mode == "ltao":
            self._draw_asterism(ax, laser_xy, transform=trans)
        self._draw_tss_vignetting(ax, mode, transform=trans)
        self._draw_catalog_stars(ax, transform=trans)  # guide-star catalogue
        self._draw_gs_ranking_badges(ax, transform=trans)  # rank badges (top 3)
        self._draw_fm_targets(ax, is_fwhm, transform=trans)  # dropped targets
        self._fm_refresh_target_list(is_fwhm)

        if for_on:
            ax.set_xlim(-view, view); ax.set_ylim(-view, view)
        elif self._fm_pa_deg() != 0.0 and (half_gx != half_gy):
            # a rotated non-square FOV can swing outside its own tight
            # unrotated bounding box -- widen the view to the diagonal so
            # nothing gets clipped by the axes at any PA
            r = float(np.hypot(half_gx, half_gy))
            ax.set_xlim(-r, r); ax.set_ylim(-r, r)
        else:
            ax.set_xlim(-half_gx, half_gx); ax.set_ylim(-half_gy, half_gy)
        ax.set_xlabel("← East      offset (arcsec)      West →", fontsize=9)
        ax.set_ylabel("← South     offset (arcsec)     North →", fontsize=9)
        tel = self._gui_telescope()
        if for_on:
            head = (f"{tel} {self.fm_mode.currentText()} {metric_name} over the "
                    f"{R_for:g}″ field of regard (FOV {sci_fov[0]:g}×"
                    f"{sci_fov[1]:g}″)\n")
        else:
            head = (f"{tel} {self.fm_mode.currentText()} {metric_name} across "
                    f"the {sci_fov[0]:g}×{sci_fov[1]:g}″ field\n")
        if snap.get("synthetic"):
            ax.set_title(
                head + f"{meta['lam_label']} · PREDICTED SCENARIO — "
                f"DIMM {snap['eps_tot_zenith']:.2f}″ / "
                f"free-atm {snap['eps_fa_zenith']:.2f}″ at zenith, "
                f"θ₀ᴷ {snap['theta0_k_zenith']:.1f}″, "
                f"ZA {snap['zenith_angle_deg']:g}° (X={snap['airmass']:.2f})",
                fontsize=9, fontweight="bold", color="#B34700")
        else:
            ax.set_title(
                head + f"{meta['lam_label']} · "
                f"{self._tz_text(snap['when_desc'])} "
                f"({self._fmt_hm(snap['t_hst'])}, X={snap['airmass']:.2f}, "
                f"seeing {snap['eps_tot_los']:.2f}″, θ₀ {snap['theta0_los']:.1f}″)",
                fontsize=9, fontweight="bold")
        self._draw_compass(ax, trans)
        self._draw_img_pa_warning(ax)          # flag a non-zero manual image PA
        if self._sky_center is not None:
            ax.annotate(
                "field defined by loaded image",
                xy=(0.5, 0.015), xycoords="axes fraction", ha="center",
                fontsize=7, color="w",
                bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.4,
                          ec="none"))
        # a what-if map must never pass for the reference budget (§5.3): flag
        # any off-default sliders. LGS/LTAO take the overridden budget
        # directly; the NGS map takes it as the Marechal variance swap, so say
        # which (and say so when nothing projects, e.g. only LGS-only terms).
        if self.last_offsets:
            if mode == "ngs":
                if ngs_dvar:
                    eff = float(np.sqrt(abs(ngs_dvar)))
                    sign = "⊕" if ngs_dvar > 0 else "⊖"
                    note = (f"MODIFIED BUDGET — projected NGS "
                            f"({sign}{eff:.0f} nm quad)")
                else:
                    note = ("MODIFIED BUDGET — no NGS-projecting change "
                            "(moved terms are LGS-only)")
            else:
                note = "MODIFIED BUDGET"
            ax.annotate(note, xy=(0.03, 0.03), xycoords="axes fraction",
                        fontsize=8, fontweight="bold", color="#C0392B",
                        va="bottom",
                        bbox=dict(boxstyle="round,pad=0.25", fc="white",
                                  alpha=0.85, ec="#C0392B"))
        ax.legend(loc="lower right", fontsize=7, framealpha=0.9)
        ax.grid(alpha=0.15)

    @staticmethod
    def _draw_asterism(ax, center, transform=None):
        """Draw the LTAO 4-LGS sodium asterism: a dashed circle of radius
        LGS_ASTERISM_RADIUS_ARCSEC about the laser (asterism) centre, with the
        four beacons on it. Markers/circle are clipped to the field; the ring
        conveys the asterism even where beacons fall outside. `transform`
        (default ax.transData) is the field-PA rotation, if any -- see
        _fm_rotation_transform. Returns the list of artists created (so the
        interactive renderer can remove them)."""
        from matplotlib.patches import Circle
        trans = transform if transform is not None else ax.transData
        cx, cy = center
        R = LGS_ASTERISM_RADIUS_ARCSEC
        arts = []
        ring = Circle((cx, cy), R, fill=False, ls="--", lw=1.1,
                      ec=FM_C_LASER, alpha=0.9, zorder=5, transform=trans)
        ax.add_patch(ring); arts.append(ring)
        arts += ax.plot(cx, cy, "+", ms=8, color=FM_C_LASER, mew=1.5, zorder=6,
                        transform=trans)
        for pa in LGS_ASTERISM_PA_DEG:
            r = np.radians(pa)
            arts += ax.plot(cx - R * np.sin(r), cy + R * np.cos(r), "D", ms=9,
                            mfc=FM_C_LASER, mec="k", mew=0.8, zorder=6,
                            transform=trans)
        arts += ax.plot([], [], "D", mfc=FM_C_LASER, mec="k",
                        label=f"4-LGS asterism, 589 nm (r={R:g}″)")
        return arts

    @staticmethod
    def _mark_ref(ax, xy, half_x, half_y, label, color, marker="o", transform=None):
        """Plot a reference marker; if it lies outside the view, clamp it to
        the edge with an arrow and annotate the true offset. `transform`
        (default ax.transData) is the field-PA rotation, if any -- see
        _fm_rotation_transform. Returns the list of artists created (so the
        interactive renderer can remove them)."""
        trans = transform if transform is not None else ax.transData
        x, y, known = xy
        r = float(np.hypot(x, y))
        inside = abs(x) <= half_x and abs(y) <= half_y
        tag = label + ("" if known else " (dir. assumed)")
        arts = []
        if inside:
            arts += ax.plot(x, y, marker, ms=10, mfc=color, mec="k", mew=0.8,
                            zorder=6, label=tag, transform=trans)
            arts.append(ax.annotate(f"{r:.1f}″", (x, y), xycoords=trans,
                                    textcoords="offset points",
                                    xytext=(6, 6), fontsize=7, color="w"))
        else:
            s = 0.92 * min(half_x / max(abs(x), 1e-9),
                           half_y / max(abs(y), 1e-9))
            ex, ey = x * s, y * s
            # white box keeps the label legible on the map, and gives the
            # arrow a patch to start FROM (no shaft through the text); both
            # ends are in the (possibly rotated) data frame
            arts.append(ax.annotate(f"{tag}\n{r:.1f}″ →", xy=(ex, ey),
                        xycoords=trans, xytext=(ex * 0.6, ey * 0.6),
                        textcoords=trans, fontsize=7, color=color,
                        ha="center", va="center", zorder=8,
                        bbox=dict(boxstyle="round,pad=0.25", fc="white",
                                  ec=color, lw=0.8, alpha=0.9),
                        arrowprops=dict(arrowstyle="->", color=color, lw=1.5,
                                        shrinkA=8)))
            arts += ax.plot([], [], marker, mfc=color, mec="k", label=tag)
        return arts

