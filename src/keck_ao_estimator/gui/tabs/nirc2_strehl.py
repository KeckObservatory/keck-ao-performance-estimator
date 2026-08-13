"""Measured-Strehl tab (NIRC2 + OSIRIS): the summit IDL Strehl tools,
re-hosted on the ported engine (image_strehl.py) with a modernized
interface -- the initial IDL-clone layout served to validate the port
against the originals and has since been reorganized (Frames /
Photometry / Results group boxes, Measure button, display-stretch
controls).

Semantics preserved from the IDL tools: the numbered-sequence Measure
flow, autofind vs click-the-star, the four photometry radii and their
ordering guard, the aperture circles, the MODEL PSF / MEASURED STAR
cutout pair, and the aligned results log.  Instrument routing is by
header identity (NIRC2 INSTRUME / OSIRIS CURRINST); OSIRIS displays the
whole detector.  Additions beyond the originals: measured-vs-predicted
readouts against the loaded estimate, OBJECT/RA/Dec -> target wiring,
crowded-field mitigations (robust sky, pick-sky, CROWDED/UNPHYSICAL
warnings), the live pick-zoom magnifier, and instrument-agnostic display
stretches (display-only, never the measurement).  Measurement runs on a
worker thread (Nirc2MeasureWorker), one result line per frame.  AUTO
UPDATE IMAGE (summit file watching) remains out of scope until an
inside-firewall deployment.
"""
import numpy as np
import matplotlib.patheffects as patheffects
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from qtcompat import Qt, QtCore, QtWidgets

import keck_ao_estimator as engine

from ..theme import set_cue
from ..widgets import SortableItem, _dspin, _shrinkable_label
from ..workers import CatalogFetchWorker, NativeFilenameWorker, Nirc2MeasureWorker
from .starlist_picker import _starlist_entry_mags

# field-map annotation text over the new image background needs a halo to
# stay legible regardless of the underlying pixel brightness
_TEXT_HALO = [patheffects.withStroke(linewidth=2.2, foreground="white")]

# Measured-SR structured CSV log (item 4): (row-dict key, column header).
# Order per Eduardo (2026-07-28): identity/time, RA/Dec, then SR/FWHM
# numbers, filter, airmass, pixel position, az/el, then seeing. ao_mode
# and guide_star added same day (2026-07-28, second pass): the AO ops
# mode (AOOPSMOD, decoded) and the guide star's resolved IDENTITY, not
# just its magnitude -- see _nirc2_resolve_guide_star.
NIRC2_CSV_COLUMNS = [
    ("time_utc", "Time (UTC)"),
    ("frame_number", "Frame #"),
    ("target_name", "Target"),
    ("ao_mode", "AO mode"),
    ("guide_star", "Guide star"),
    ("guide_mag", "Guide mag"),
    ("guide_mag_src", "Guide mag source"),
    ("ra", "RA"),
    ("dec", "Dec"),
    ("measured_sr", "Measured SR"),
    ("predicted_sr", "Predicted SR"),
    ("delta_sr", "ΔSR"),
    ("measured_fwhm", "Measured FWHM (mas)"),
    ("predicted_fwhm", "Predicted FWHM (mas)"),
    ("delta_fwhm", "ΔFWHM (mas)"),
    ("filter", "Filter"),
    ("airmass", "Airmass"),
    ("pixel_x", "Pixel X"),
    ("pixel_y", "Pixel Y"),
    ("az_deg", "Az (deg)"),
    ("el_deg", "El (deg)"),
    ("lbwfs_fwhm", "LBWFS FWHM"),
    ("dimm_seeing", "DIMM seeing (arcsec)"),
    ("mass_seeing", "MASS seeing (arcsec)"),
]

# duplicate-measurement guard: same frame AND the star centroid within this
# many pixels of an already-logged row (re-clicking a point on the field
# map to inspect it re-triggers the full display/log pipeline -- see
# _on_nirc2_map_pick -- so this is a real, common case, not a hypothetical)
_DUP_POS_TOL_PX = 3.0


class Nirc2StrehlTabMixin:
    def _build_nirc2_tab(self):
        w = QtWidgets.QWidget()
        outer = QtWidgets.QHBoxLayout(w)

        # ---- left column: controls in themed group boxes -------------------
        left = QtWidgets.QVBoxLayout()

        gb_frames = QtWidgets.QGroupBox("Frames")
        form = QtWidgets.QFormLayout(gb_frames)

        path_row = QtWidgets.QHBoxLayout()
        self.n2_path = QtWidgets.QLineEdit()
        self.n2_path.setPlaceholderText("directory of NIRC2 / OSIRIS frames")
        browse = QtWidgets.QPushButton("…")
        browse.setFixedWidth(28)
        browse.clicked.connect(self._on_nirc2_browse)
        path_row.addWidget(self.n2_path, 1)
        path_row.addWidget(browse)
        form.addRow("Path:", self._wrap(path_row))

        self.n2_im1 = QtWidgets.QSpinBox()
        self.n2_im1.setRange(0, 9999)
        self.n2_im1.setValue(1)
        form.addRow("First image:", self.n2_im1)
        self.n2_nim = QtWidgets.QSpinBox()
        self.n2_nim.setRange(1, 999)
        self.n2_nim.setValue(1)
        form.addRow("Images:", self.n2_nim)
        self.n2_bg1 = QtWidgets.QSpinBox()
        self.n2_bg1.setRange(0, 9999)
        self.n2_bg1.setValue(2)
        form.addRow("First background:", self.n2_bg1)
        self.n2_nbg = QtWidgets.QSpinBox()
        self.n2_nbg.setRange(0, 999)
        self.n2_nbg.setValue(0)
        self.n2_nbg.setToolTip("0 = sky from the annulus (no background frames)")
        form.addRow("Backgrounds:", self.n2_nbg)

        self.n2_nstars = QtWidgets.QSpinBox()
        self.n2_nstars.setRange(0, 50)
        self.n2_nstars.setValue(5)
        self.n2_nstars.setSpecialValueText("Auto")
        self.n2_nstars.setToolTip(
            "Stars for 'Measure field' auto-find (brightest first, halo "
            "knots and noise rejected). 'Auto' lets star QUALITY decide "
            "the count: candidates are kept while their propagated "
            "sky-noise SR uncertainty stays below ±0.05. Single-frame "
            "Measure always uses one star.")
        form.addRow("Stars:", self.n2_nstars)

        self.n2_autofind = QtWidgets.QCheckBox("Autofind (brightest pixel)")
        self.n2_autofind.setChecked(True)
        self.n2_autofind.setToolTip(
            "ON: measure the brightest star automatically.\n"
            "OFF: after Measure, click the star in the image — a live "
            "magnifier follows the cursor.")
        form.addRow("", self.n2_autofind)
        left.addWidget(gb_frames)

        gb_phot = QtWidgets.QGroupBox("Photometry")
        form = QtWidgets.QFormLayout(gb_phot)

        self.n2_robust_sky = QtWidgets.QCheckBox("Robust sky (σ-clip)")
        self.n2_robust_sky.setToolTip(
            "Sky from the annulus' σ-clipped MEDIAN instead of mvdaper's "
            "plain mean — use in crowded fields, where neighbors in the "
            "annulus drag the mean high (the CROWDED warning). Off = "
            "byte-faithful to the summit IDL tool.")
        self.n2_pick_sky = QtWidgets.QPushButton("Pick sky")
        self.n2_pick_sky.setCheckable(True)
        self.n2_pick_sky.setToolTip(
            "Arm, then click an EMPTY patch in the image: sky becomes the "
            "σ-clipped median of a 41x41 px box there, bypassing the "
            "annulus entirely (best in very crowded fields). The value "
            "sticks for later measurements; click again to clear it.")
        self.n2_pick_sky.clicked.connect(self._on_nirc2_pick_sky)
        self.n2_photrad = _dspin(0.05, 5.0, 0.05, engine.NIRC2_PHOTOMETRY_RADIUS_ARCSEC, 3, " arcsec")
        form.addRow("Photometry radius:", self.n2_photrad)
        self.n2_bgin = _dspin(0.05, 8.0, 0.05, engine.NIRC2_BG_INNER_RADIUS_ARCSEC, 3, " arcsec")
        form.addRow("Inner sky radius:", self.n2_bgin)
        self.n2_bgout = _dspin(0.05, 9.0, 0.05, engine.NIRC2_BG_OUTER_RADIUS_ARCSEC, 3, " arcsec")
        form.addRow("Outer sky radius:", self.n2_bgout)
        self.n2_peakrad = _dspin(0.02, 1.0, 0.01, engine.NIRC2_PEAK_RADIUS_ARCSEC, 3, " arcsec")
        form.addRow("Peak-finding radius:", self.n2_peakrad)
        self.n2_auto_rad = QtWidgets.QCheckBox("Auto aperture")
        self.n2_auto_rad.setToolTip(
            "Curve-of-growth photometry radius per star: stop where flux "
            "growth is sky-noise dominated, or before a neighbor's wing "
            "enters (sustained growth-rate upturn; diffraction rings are "
            "one step wide and ignored). The DL reference uses the same "
            "radius (matched apertures). Off = the IDL tool's fixed "
            "radius.")
        # one toggle per spanning row: all three in one field-column row
        # clip at real font metrics inside the 330 px left column ("Robust
        # sky (σ-clip)" truncated to "Robust") — the same overpacked-row
        # class as the winds-status and nighttime-status labels
        self.n2_ee_corr = QtWidgets.QCheckBox(
            "EE aperture correction (field only)")
        self.n2_ee_corr.setToolTip(
            "Crowded-field absolute SR (Eduardo 2026-07-25): each field "
            "star measured with a small auto aperture is ALSO measured "
            "at the full radius; clean pairs calibrate the field's own "
            "halo ratio h, and crowded stars get the growth-curve "
            "correction SR·h/(1−SR·(1−h)) — making small-aperture SRs "
            "absolute (full-radius convention). Field map only; "
            "requires Auto aperture; roughly doubles measuring time. "
            "Calibrated per field — h is never reused across fields.")
        self.n2_ee_corr.setEnabled(False)
        self.n2_auto_rad.toggled.connect(
            lambda on: (self.n2_ee_corr.setEnabled(on),
                        None if on else self.n2_ee_corr.setChecked(False)))
        self.n2_psf_clean = QtWidgets.QCheckBox(
            "PSF-fit neighbour subtraction (Measure field only)")
        self.n2_psf_clean.setToolTip(
            "MEASURE FIELD ONLY (Eduardo 2026-08-07). The empirical PSF is "
            "a property of the FIELD and costs seconds to build; building "
            "it to look at ONE star delayed every single-frame Measure for "
            "a correction that, on a sparse NIRC2 field, almost always "
            "declines to do anything. So this checkbox now only bites on "
            "'Measure field' — and on clicks/picks made afterwards, which "
            "reuse that same field model for free. A plain Measure skips "
            "it and says so in the log. "
            "Fit the field's empirical PSF to every catalogued neighbour "
            "landing inside a star's photometry aperture or sky annulus, "
            "and subtract the fitted neighbours before measuring — the "
            "crowded-field failure robust sky / pick sky don't address "
            "(a neighbour INSIDE the aperture, not just in the sky "
            "annulus). Off = byte-faithful to the summit IDL tool. "
            "Validated to |SR bias| <= 0.02 for Strehl <= 0.30, where "
            "the residual error is an UNDERESTIMATE; above 0.30 it "
            "turns into an overestimate and the log says so explicitly "
            "— cleaning still runs, just with lower confidence. Every "
            "cleaned measurement logs which way it is likely wrong. "
            "NOTE: on any star cleaning SUCCEEDS on, Robust sky "
            "(sigma-clip) is IGNORED — both remove the same neighbour "
            "light, and using them together doubles the bias (measured: "
            "signed +0.10 cleaned alone, +0.20 with both). The log says "
            "so per star, and Robust sky still applies normally to any "
            "star cleaning declined. "
            "If the frame cannot supply enough isolated donor "
            "stars, cleaning is skipped and the default number stands. "
            "A star whose "
            "cleaning would remove almost all its own aperture flux is "
            "left off the field map rather than reported (reinsertable, "
            "same as a field-consistency outlier).")
        form.addRow(self.n2_robust_sky)
        form.addRow(self.n2_auto_rad)
        form.addRow(self.n2_ee_corr)
        form.addRow(self.n2_psf_clean)
        pick_row = QtWidgets.QHBoxLayout()
        pick_row.addWidget(self.n2_pick_sky)
        pick_row.addStretch(1)
        form.addRow(self._wrap(pick_row))
        sky_note = QtWidgets.QLabel(
            "Sky comes from the annulus mean (the IDL tool's way). In "
            "crowded fields neighbor stars inflate it — Robust sky uses "
            "the annulus' σ-clipped median instead; Pick sky uses a "
            "clicked empty patch.")
        sky_note.setWordWrap(True)
        set_cue(sky_note, "secondary")
        sky_note.setStyleSheet("QLabel { font-size:11px; }")
        form.addRow(sky_note)
        left.addWidget(gb_phot)

        self.n2_go = QtWidgets.QPushButton("Measure")
        self.n2_go.setToolTip(
            "The IDL tool's GO!. Re-measures the frame currently loaded "
            "(so a changed aperture / sky / stretch can be re-run on it "
            "directly); measures the numbered FIRST IMAGE / N IMAGES "
            "sequence when no frame is loaded, or as soon as you change "
            "either spin.")
        self.n2_go.clicked.connect(self._on_nirc2_go)
        # Measure shares its row with the log opt-out, so a run can be
        # made deliberately throwaway (re-measuring to eyeball a frame
        # without touching the image log at all)
        self.n2_go.setSizePolicy(QtWidgets.QSizePolicy.Policy.Maximum,
                                 QtWidgets.QSizePolicy.Policy.Fixed)
        self.n2_append_log = QtWidgets.QCheckBox("Append to log")
        self.n2_append_log.setChecked(True)
        self.n2_append_log.setToolTip(
            "Checked (default): every measurement is added to the image "
            "log. Unchecked: measurements are shown but NOT logged — "
            "nothing is added and no duplicate questions are asked.")
        go_row = QtWidgets.QHBoxLayout()
        go_row.addWidget(self.n2_go)
        go_row.addWidget(self.n2_append_log)
        go_row.addStretch(1)
        left.addLayout(go_row)
        cal_note = QtWidgets.QLabel(
            "Frames are reduced with the bundled K2 superflat/supermask "
            "(strehl_widget.pro's calibration) automatically.")
        cal_note.setWordWrap(True)
        set_cue(cal_note, "secondary")
        cal_note.setStyleSheet("QLabel { font-size:11px; }")
        left.addWidget(cal_note)

        files_lbl = QtWidgets.QLabel("Frames in path — double-click to measure:")
        set_cue(files_lbl, "secondary")
        left.addWidget(files_lbl)
        self.n2_native_names = QtWidgets.QCheckBox("Show native filenames")
        self.n2_native_names.setToolTip(
            "For directories of KOA-renamed frames (OI.<utdate>.<sec>."
            "<hundredths>.fits etc.): reads each file's own DATAFILE "
            "header card -- the original observatory filename KOA "
            "preserves on ingest (e.g. i260112_a000061.fits) -- and "
            "shows that instead, so the list reads the same as your "
            "night log. Files without a DATAFILE card keep showing "
            "their on-disk name.")
        self.n2_native_names.toggled.connect(self._nirc2_refresh_files)
        left.addWidget(self.n2_native_names)
        self.n2_files = QtWidgets.QListWidget()
        self.n2_files.setToolTip(
            "n####.fits frames found in PATH — double-click one to set "
            "FIRST IMAGE to it and measure it immediately")
        self.n2_files.itemDoubleClicked.connect(self._on_nirc2_file_dclick)
        self.n2_path.textChanged.connect(self._nirc2_refresh_files)
        left.addWidget(self.n2_files, 1)
        left_w = QtWidgets.QWidget()
        left_w.setLayout(left)
        left_w.setMaximumWidth(330)
        outer.addWidget(left_w)

        # ---- right column: image displays + readouts + log -----------------
        right = QtWidgets.QVBoxLayout()

        stretch_row = QtWidgets.QHBoxLayout()
        self.n2_stretch = QtWidgets.QComboBox()
        self.n2_stretch.addItems(["IDL ±5σ", "linear %", "sqrt", "asinh",
                                  "log"])
        self.n2_stretch.setToolTip(
            "Display stretch (display only — never touches the "
            "measurement). 'IDL ±5σ' is the summit tool's window; the "
            "others clip black at the 0.5th percentile with the white "
            "point set here — asinh/log lift faint objects.")
        self.n2_white = _dspin(90.0, 100.0, 0.1, 99.5, 2, " %")
        self.n2_white.setToolTip("White-point percentile for the "
                                 "non-IDL stretches")
        self.n2_white.setEnabled(False)
        self.n2_stretch.currentTextChanged.connect(self._on_nirc2_stretch)
        self.n2_white.valueChanged.connect(self._on_nirc2_stretch)
        stretch_row.addWidget(QtWidgets.QLabel("Stretch:"))
        stretch_row.addWidget(self.n2_stretch)
        stretch_row.addWidget(QtWidgets.QLabel("White:"))
        stretch_row.addWidget(self.n2_white)
        stretch_row.addStretch(1)
        right.addLayout(stretch_row)

        self.n2_fig = Figure(figsize=(4.6, 4.2), layout="constrained")
        self.n2_canvas = FigureCanvasQTAgg(self.n2_fig)
        self.n2_canvas.setMinimumHeight(260)
        self.n2_canvas.mpl_connect("button_press_event", self._on_nirc2_click)
        self.n2_canvas.mpl_connect("motion_notify_event", self._on_nirc2_motion)

        img_tab = QtWidgets.QWidget()
        img_v = QtWidgets.QVBoxLayout(img_tab)
        img_v.setContentsMargins(0, 0, 0, 0)
        img_v.addWidget(self.n2_canvas)

        map_tab = QtWidgets.QWidget()
        map_v = QtWidgets.QVBoxLayout(map_tab)
        map_v.setContentsMargins(0, 0, 0, 0)
        fm_row = QtWidgets.QHBoxLayout()
        self.n2_field_btn = QtWidgets.QPushButton("Measure field")
        self.n2_field_btn.setToolTip(
            "Auto-find the N brightest stars (the Stars spin) in the "
            "current frame, measure each, and map the results")
        self.n2_field_btn.clicked.connect(self._on_nirc2_measure_field)
        self.n2_add_star = QtWidgets.QPushButton("Add star by click")
        self.n2_add_star.setCheckable(True)
        self.n2_add_star.setToolTip(
            "While armed, clicks on the Image view measure that star and "
            "add it to the map (works with the magnifier). Also works "
            "directly in the popped-out map when it's showing the image "
            "backdrop (\"Show image\" checkbox) -- no need to switch to "
            "the Image tab.")
        self.n2_field_clear = QtWidgets.QPushButton("Clear")
        self.n2_field_clear.clicked.connect(self._on_nirc2_field_clear)
        self.n2_reject_star = QtWidgets.QPushButton("Reject star")
        self.n2_reject_star.setEnabled(False)
        self.n2_reject_star.setToolTip(
            "Remove the selected star from the measured field (click a "
            "point on the map to select it and inspect its measurement); "
            "every field statistic regenerates without it.  Selecting a "
            "rejected star (the × markers — auto-clipped or previously "
            "rejected) turns this into 'Reinsert star': the opposite "
            "move, back into the fit (Eduardo 2026-07-25)")
        self.n2_reject_star.clicked.connect(self._on_nirc2_reject_star)
        fm_row.addWidget(self.n2_field_btn)
        fm_row.addWidget(self.n2_add_star)
        fm_row.addWidget(self.n2_field_clear)
        fm_row.addWidget(self.n2_reject_star)
        fm_row.addSpacing(12)
        fm_row.addWidget(QtWidgets.QLabel("Metric:"))
        self.n2_map_metric = QtWidgets.QComboBox()
        self.n2_map_metric.addItems(["SR", "FWHM (mas)"])
        self.n2_map_metric.setToolTip(
            "Color the measured map by Strehl or FWHM. Both scales are "
            "oriented brighter = better (FWHM reversed).")
        self.n2_map_metric.currentTextChanged.connect(
            lambda *_: self._nirc2_draw_map())
        fm_row.addWidget(self.n2_map_metric)
        self.n2_map_popout = QtWidgets.QPushButton("Pop out")
        self.n2_map_popout.setToolTip(
            "Open the field map in its own resizable window — drag the "
            "divider under the image to grow the embedded views too")
        self.n2_map_popout.clicked.connect(self._on_nirc2_map_popout)
        fm_row.addWidget(self.n2_map_popout)
        fm_row.addStretch(1)
        map_v.addLayout(fm_row)
        self.n2_map_fig = Figure(figsize=(4.6, 4.0), layout="constrained")
        self.n2_map_canvas = FigureCanvasQTAgg(self.n2_map_fig)
        self.n2_map_canvas.mpl_connect("pick_event", self._on_nirc2_map_pick)
        self._n2_sel_star = None
        map_v.addWidget(self.n2_map_canvas, 1)
        self.n2_field_stats = QtWidgets.QLabel("")
        self.n2_field_stats.setWordWrap(True)
        self.n2_field_stats.setStyleSheet("font-size: 11px;")
        self.n2_field_stats.setToolTip(
            "Statistics of the measured field: peak star, error-weighted "
            "mean ± scatter, best-fit performance gradient (arrow on the "
            "map points downhill), and an EFFECTIVE isoplanatic angle "
            "fitted from the anisoplanatic falloff "
            "S(θ) = S₀·exp(−(θ/θ₀)^(5/3)) about the best-measured star. "
            "With an LGS the tip-tilt star also shapes the field, so "
            "θ₀ is the delivered correction's falloff scale, not a pure "
            "atmospheric θ₀. The 500 nm value uses the standard λ^(6/5) "
            "scaling for comparison with the seeing monitors.")
        map_v.addWidget(self.n2_field_stats)
        self.n2_ee_out = QtWidgets.QLabel("")
        self.n2_ee_out.setWordWrap(True)
        self.n2_ee_out.setStyleSheet(
            "QLabel { font-weight:bold; font-size:11px; padding:3px 6px; "
            "border-radius:3px; background:#fff3cd; color:#664d03; }")
        self.n2_ee_out.setToolTip(
            "The EE aperture correction's per-field calibration result "
            "(h) — shown here instead of buried in the log because it "
            "changes how every small-aperture SR on this map should be "
            "read")
        self.n2_ee_out.setVisible(False)
        map_v.addWidget(self.n2_ee_out)

        self.n2_view_tabs = QtWidgets.QTabWidget()
        self.n2_view_tabs.addTab(img_tab, "Image")
        self.n2_view_tabs.addTab(map_tab, "Field map")

        top_w = QtWidgets.QWidget()
        top_v = QtWidgets.QVBoxLayout(top_w)
        top_v.setContentsMargins(0, 0, 0, 0)
        top_v.addWidget(self.n2_view_tabs)
        bottom_w = QtWidgets.QWidget()
        bottom_v = QtWidgets.QVBoxLayout(bottom_w)
        bottom_v.setContentsMargins(0, 0, 0, 0)
        self.n2_split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        self.n2_split.addWidget(top_w)
        self.n2_split.addWidget(bottom_w)
        self.n2_split.setStretchFactor(0, 2)
        self.n2_split.setStretchFactor(1, 1)
        self.n2_split.setSizes([560, 420])
        right.addWidget(self.n2_split, 1)

        zoom_row = QtWidgets.QHBoxLayout()
        self.n2_fig_dl = Figure(figsize=(1.7, 1.7), layout="constrained")
        self.n2_canvas_dl = FigureCanvasQTAgg(self.n2_fig_dl)
        self.n2_fig_star = Figure(figsize=(1.7, 1.7), layout="constrained")
        self.n2_canvas_star = FigureCanvasQTAgg(self.n2_fig_star)
        self.n2_cap_dl = QtWidgets.QLabel("MODEL PSF")
        self.n2_cap_star = QtWidgets.QLabel("MEASURED STAR")
        zoom_row.addStretch(1)
        for cap, canvas in ((self.n2_cap_dl, self.n2_canvas_dl),
                            (self.n2_cap_star, self.n2_canvas_star)):
            canvas.setFixedSize(150, 150)
            set_cue(cap, "secondary")
            cap.setAlignment(QtCore.Qt.AlignmentFlag.AlignHCenter)
            col = QtWidgets.QVBoxLayout()
            col.addWidget(cap)
            col.addWidget(canvas)
            zoom_row.addLayout(col)
        zoom_row.addStretch(1)
        bottom_v.addLayout(zoom_row)

        self.n2_warn = QtWidgets.QLabel("")
        _shrinkable_label(self.n2_warn)
        self.n2_warn.setWordWrap(True)
        self.n2_warn.setAlignment(QtCore.Qt.AlignmentFlag.AlignHCenter)
        bottom_v.addWidget(self.n2_warn)

        def _readout(width=76):
            e = QtWidgets.QLineEdit()
            e.setReadOnly(True)
            e.setMaximumWidth(width)
            return e

        gb_res = QtWidgets.QGroupBox("Results")
        res_v = QtWidgets.QVBoxLayout(gb_res)

        # OBJECT + header RA/Dec on top (sexagesimal, straight from the
        # frame) so the star can be found in a target list by eye
        obj_row = QtWidgets.QHBoxLayout()
        self.n2_object = _readout(110)
        self.n2_ra = _readout(105)
        self.n2_dec = _readout(105)
        self.n2_set_target = QtWidgets.QPushButton("Set as target")
        self.n2_set_target.setEnabled(False)
        self.n2_set_target.setToolTip(
            "Use the frame's OBJECT as the estimator target: selects it in "
            "the loaded target list when a name matches, otherwise fills the "
            "Target tab's name AND the header RA/Dec for you")
        self.n2_set_target.clicked.connect(self._on_nirc2_set_target)
        self.n2_tt_star = QtWidgets.QPushButton("TT star")
        self.n2_tt_star.setEnabled(False)
        self.n2_tt_star.setToolTip(
            "Estimate the tip-tilt star's position from the frame's AO "
            "headers: the TT-sensor stage (AOTSX/AOTSY) sits on the star "
            "and the active pointing origin (POXPOS/POYPOS) is the field "
            "centre — same focal-plane frame, 1.375″/mm, so their "
            "difference is the delivered star-to-field-centre separation "
            "(rotation-free). Its identity/magnitude then comes from a "
            "catalogue RING match at that separation (the bench→sky "
            "direction is not derivable from one frame).")
        self.n2_tt_star.clicked.connect(self._on_nirc2_tt_star)
        obj_row.addWidget(QtWidgets.QLabel("OBJECT:"))
        obj_row.addWidget(self.n2_object)
        obj_row.addWidget(QtWidgets.QLabel("RA:"))
        obj_row.addWidget(self.n2_ra)
        obj_row.addWidget(QtWidgets.QLabel("Dec:"))
        obj_row.addWidget(self.n2_dec)
        obj_row.addWidget(self.n2_set_target)
        obj_row.addWidget(self.n2_tt_star)
        obj_row.addStretch(1)
        res_v.addLayout(obj_row)
        self.n2_ttstar_out = QtWidgets.QLabel("")
        self.n2_ttstar_out.setWordWrap(True)
        # the navy this used to hard-code is near-invisible on the dark
        # theme (Eduardo 2026-08-07). Declare WHAT it is and let the active
        # theme pick the color -- the size-only widget stylesheet still
        # merges with the app-level cue color (see gui/theme.py).
        set_cue(self.n2_ttstar_out, "info")
        self.n2_ttstar_out.setStyleSheet("QLabel { font-size:11px; }")
        res_v.addWidget(self.n2_ttstar_out)

        # readout table: SR and FWHM columns, with each delta DIRECTLY
        # under its measured/predicted pair (Eduardo 2026-07-23)
        grid = QtWidgets.QGridLayout()
        grid.setHorizontalSpacing(6)
        self.n2_strehl_out = _readout()
        self.n2_fwhm_out = _readout()
        self.n2_wfe_out = _readout()
        self.n2_pred_sr = _readout()
        self.n2_pred_fwhm = _readout()
        self.n2_dsr = _readout()
        self.n2_dfwhm = _readout()
        grid.addWidget(QtWidgets.QLabel("SR"), 0, 1)
        grid.addWidget(QtWidgets.QLabel("FWHM (mas)"), 0, 2)
        grid.addWidget(QtWidgets.QLabel("MEASURED:"), 1, 0)
        grid.addWidget(self.n2_strehl_out, 1, 1)
        grid.addWidget(self.n2_fwhm_out, 1, 2)
        grid.addWidget(QtWidgets.QLabel("WFE:"), 1, 3)
        grid.addWidget(self.n2_wfe_out, 1, 4)
        grid.addWidget(QtWidgets.QLabel("nm"), 1, 5)
        grid.addWidget(QtWidgets.QLabel("PREDICTED:"), 2, 0)
        grid.addWidget(self.n2_pred_sr, 2, 1)
        grid.addWidget(self.n2_pred_fwhm, 2, 2)
        grid.addWidget(QtWidgets.QLabel("Δ (meas−pred):"), 3, 0)
        grid.addWidget(self.n2_dsr, 3, 1)
        grid.addWidget(self.n2_dfwhm, 3, 2)
        grid.setColumnStretch(6, 1)
        res_v.addLayout(grid)
        bottom_v.addWidget(gb_res)

        log_row = QtWidgets.QHBoxLayout()
        log_row.addWidget(QtWidgets.QLabel("Log:"))
        log_row.addStretch(1)
        self.n2_log_popout = QtWidgets.QPushButton("Pop out")
        self.n2_log_popout.setToolTip(
            "Open the log in its own resizable window — it's the same "
            "text, live-updating, just easier to read at size")
        self.n2_log_popout.clicked.connect(self._on_nirc2_log_popout)
        log_row.addWidget(self.n2_log_popout)
        bottom_v.addLayout(log_row)

        self.n2_log = QtWidgets.QPlainTextEdit()
        self.n2_log.setReadOnly(True)
        self.n2_log.setMaximumBlockCount(500)
        self.n2_log.setLineWrapMode(
            QtWidgets.QPlainTextEdit.LineWrapMode.NoWrap)
        font = self.n2_log.font()
        font.setFamily("monospace")
        self.n2_log.setFont(font)
        self.n2_log.setMinimumHeight(140)
        bottom_v.addWidget(self.n2_log, 1)
        # structured counterpart of the free-text log above (item 4): one
        # row per measurement, the Image-log pop-out tab's/CSV export's
        # source of truth
        self._n2_csv_rows = []
        self._n2_csv_table = None
        self._n2_dup_batch_policy = None   # "all remaining" choice, per run
        self._n2_dup_pending = []          # arrived while a dialog was open
        self._n2_dup_dialog_open = False   # re-entrancy guard
        self._n2_draining = False          # drain-loop re-entrancy guard
        # last TT-star catalogue ring match, tagged with the frame it was
        # run for (see _on_nirc2_tt_star_catalog / _nirc2_resolve_guide_star)
        self._n2_tt_star_resolved = None
        self._n2_guide_mag_warned = None    # frame already warned about
        # per-target catalogue self-magnitude cache for whole-night batch
        # runs (see _nirc2_prefetch_guide_stars): normalized target name ->
        # star-dict list, populated once per DISTINCT target and reused
        # across every frame of it, never per-frame
        self._n2_auto_gs_cache = {}
        self._n2_prefetch_worker = None

        outer.addLayout(right, 1)

        self._n2_worker = None
        self._n2_names_worker = None   # single-flight native-filename lookup
        self._n2_field = []            # measured-field results for the map
        self._n2_last_draw = None      # (title, result-or-None) for redraws
        self._n2_pick_locked = False   # zoom stops following after a click
        self._n2_sky_override = None   # picked sky value (ADU), None = annulus
        self._n2_frame_hst = None   # last measured frame's HST timestamp
        self._n2_image = None       # last reduced frame (for click re-measure)
        self._n2_params = None
        self._n2_dl = None
        self._n2_imno = None
        self._n2_bg_used = False
        return w

    # ---- handlers ----------------------------------------------------------
    def _on_nirc2_browse(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(
            self, "NIRC2 frame directory", self.n2_path.text() or "")
        if d:
            self.n2_path.setText(d)      # textChanged refreshes the file list

    def _nirc2_refresh_files(self):
        import os
        self.n2_files.clear()
        d = self.n2_path.text().strip()
        if not os.path.isdir(d):
            return
        try:
            names = sorted(n for n in os.listdir(d)
                           if n.lower().endswith((".fits", ".fits.gz")))
        except OSError:
            return
        for name in names:
            item = QtWidgets.QListWidgetItem(name)
            item.setData(QtCore.Qt.ItemDataRole.UserRole, name)
            self.n2_files.addItem(item)
        if self.n2_native_names.isChecked() and names:
            self._nirc2_start_native_lookup(d, names)

    def _nirc2_start_native_lookup(self, dirpath, names):
        """Background DATAFILE-header lookup for the native-filenames
        toggle. Doesn't wait for or cancel a still-running prior scan (a
        QThread can't be safely force-stopped mid-read) -- an in-flight
        worker from a since-abandoned directory just finishes on its own
        and gets discarded by the stale-directory check below."""
        self._n2_names_lookup_dir = dirpath
        self._n2_names_worker = NativeFilenameWorker(dirpath, names, parent=self)
        self._n2_names_worker.done.connect(self._on_nirc2_native_names)
        self._n2_names_worker.start()

    def _on_nirc2_native_names(self, mapping):
        # the path field may have moved on to a different directory while
        # this scan was running -- only apply results that still match
        if self.n2_path.text().strip() != getattr(
                self, "_n2_names_lookup_dir", None):
            return
        for row in range(self.n2_files.count()):
            item = self.n2_files.item(row)
            disk_name = item.data(QtCore.Qt.ItemDataRole.UserRole)
            native = mapping.get(disk_name)
            if native:
                item.setText(native)
                item.setToolTip(disk_name)

    def _on_nirc2_file_dclick(self, item):
        import os
        import re
        disk_name = item.data(QtCore.Qt.ItemDataRole.UserRole) or item.text()
        m = re.match(r"^n(\d{4})\.fits$", disk_name)
        if m is not None:       # summit-numbered frame: drive FIRST IMAGE
            self.n2_im1.setValue(int(m.group(1)))
            self.n2_nim.setValue(1)
            # _nirc2_start, NOT _on_nirc2_go: a double-click names the frame
            # explicitly, so it must never be re-routed to a previously
            # remembered file. It could be -- _on_nirc2_go reuses that file
            # while the spins are unchanged, and the spins here can land on
            # exactly the values a previous non-numbered load left behind
            # (gui_phase29 does precisely that: an OSIRIS file loaded at
            # im1=7/nim=1, then a double-click on n0007.fits). The clear
            # inside _nirc2_start(files=None) also drops the stale memory.
            self._nirc2_start()
            return
        # any other FITS (KOA exports etc.): measure this one file directly;
        # the worker refuses non-NIRC2 instruments by header with a log line
        path = self.n2_path.text().strip()
        label = os.path.splitext(item.text())[0]   # native name if shown
        self._nirc2_start(files=[(label, os.path.join(path, disk_name))])

    def _nirc2_radii(self):
        return (self.n2_photrad.value(), self.n2_bgin.value(),
                self.n2_bgout.value(), self.n2_peakrad.value())

    def _on_nirc2_go(self):
        """Measure. RE-MEASURES THE LOADED FRAME when one is loaded
        (Eduardo 2026-08-07) instead of always re-running the numbered
        sequence: a frame opened by double-clicking a non-summit-numbered
        file (KOA exports and the like) is not described by the FIRST
        IMAGE / N IMAGES spins at all, so Measure used to walk off and
        measure some unrelated numbered frame -- there was no way to
        simply re-measure what was on screen after changing an aperture
        or a sky setting.

        The numbered path is untouched, and an edit to either spin is
        taken as "I mean the numbered sequence now" and wins: the
        remembered file is only reused while the spins still read exactly
        what they read when it was loaded."""
        files = getattr(self, "_n2_loaded_files", None)
        if files is not None and self._nirc2_seq_state() == getattr(
                self, "_n2_loaded_seq", None):
            self.n2_log.appendPlainText(
                f"re-measuring the loaded frame: {files[0][0]}")
            self._nirc2_start(files=files)
            return
        self._nirc2_start()

    def _nirc2_seq_state(self):
        """The frame-selection spins, for deciding whether the user has
        moved off the loaded frame -- see _on_nirc2_go."""
        return (self.n2_im1.value(), self.n2_nim.value())

    def _nirc2_start(self, files=None):
        """Shared worker launch for GO! (numbered sequence) and the file
        list's double-click (explicit file), with the strehl_widget guards."""
        if self._n2_worker is not None and self._n2_worker.isRunning():
            self.n2_log.appendPlainText("! measurement already running")
            return
        path = self.n2_path.text().strip()
        if not path:
            self.n2_log.appendPlainText(
                "! set the PATH to a directory of NIRC2 frames")
            return
        if (self.n2_photrad.value() > self.n2_bgin.value()
                or self.n2_bgin.value() >= self.n2_bgout.value()):
            self.n2_log.appendPlainText(
                "! must have photometry radius < inner background radius "
                "< outer background radius")     # strehl_widget.pro's guard
            return
        # remember WHAT is being loaded so Measure can re-measure it
        # (_on_nirc2_go). A numbered run clears it: from then on the spins
        # describe the frame on screen and the ordinary path is right.
        self._n2_loaded_files = files
        self._n2_loaded_seq = self._nirc2_seq_state()
        self.n2_go.setEnabled(False)
        # a fresh run starts with no remembered duplicate-handling choice
        # -- see _nirc2_ask_duplicate_paused
        self._n2_dup_batch_policy = None
        self._n2_dup_pending = []
        # resolve every distinct target's guide star BEFORE any frame is
        # measured (see _nirc2_prefetch_guide_stars) -- so a whole-night
        # BATCH across many targets never falls back to the bare spinbox
        # default just because a target wasn't manually loaded first. Only
        # for an actual multi-file run: a single-frame GO!/double-click is
        # ordinary interactive clicking-through, not the unattended-batch
        # case this exists for, and gating it out keeps a single click from
        # unexpectedly touching the network (or paying its latency) just to
        # look at one already-open frame -- the per-frame resolver's other
        # tiers (starlist, TT-star ring match, target-list entry) still
        # apply exactly as before for that case.
        seq = self._nirc2_pending_seq(path, files)
        if len(seq) > 1:
            self._nirc2_prefetch_guide_stars(
                seq, lambda: self._nirc2_start_worker(path, files))
        else:
            self._nirc2_start_worker(path, files)

    def _nirc2_start_worker(self, path, files):
        # PSF-fit cleaning is a MEASURE-FIELD feature only (Eduardo
        # 2026-08-07). Left on this path it built a fresh empirical PSF for
        # EVERY frame -- seconds of deep-catalogue + ePSF work before the
        # first number appears -- purely to look at one star, and on a
        # sparse NIRC2 field it then reports `uncalibrated` and changes
        # nothing. So the checkbox does not reach the worker; it is honoured
        # by "Measure field" (which builds the model once for the whole
        # field) and by clicks made afterwards, which reuse that model.
        if self.n2_psf_clean.isChecked():
            self.n2_log.appendPlainText(
                "  [psf-clean] skipped — PSF-fit cleaning applies to "
                "'Measure field' (and to clicks after it), not to a "
                "single-frame Measure")
        self._n2_worker = Nirc2MeasureWorker(
            path, "n", self.n2_im1.value(), self.n2_nim.value(),
            self.n2_bg1.value(), self.n2_nbg.value(), self._nirc2_radii(),
            autofind=self.n2_autofind.isChecked(), files=files,
            robust_sky=self.n2_robust_sky.isChecked(),
            sky_override=self._n2_sky_override,
            auto_radius=self.n2_auto_rad.isChecked(),
            psf_clean=False, parent=self)
        self._n2_worker.frame_done.connect(self._on_nirc2_frame_done)
        self._n2_worker.frame_failed.connect(self._on_nirc2_frame_failed)
        self._n2_worker.finished_all.connect(self._on_nirc2_finished)
        self._n2_worker.start()

    def _on_nirc2_frame_done(self, imno, result, params, reduced, dl,
                             header=None):
        import datetime as dt
        self._n2_header = header       # AO keywords for the TT-star odometer
        if reduced is not self._n2_image:
            self._n2_field = []        # the map belongs to ONE frame
            self._n2_field_dropped = []     # else stale × markers persist
            self._nirc2_clear_selection()   # else a stale ring can persist
            self._nirc2_set_ee_readout("")  # else a stale h can persist
            # the ePSF and neighbour catalogue belong to ONE frame too, and
            # single-star picks now reuse them (_nirc2_measure_at) -- so they
            # must die with the frame or a click on the NEW frame would be
            # cleaned against the OLD frame's model
            self._n2_field_epsf = None
            self._n2_field_catalog = None
            self._nirc2_draw_map()
        self._n2_image = reduced
        self._n2_params = params
        self._n2_dl = dl
        self._n2_imno = imno
        self._n2_bg_used = self.n2_nbg.value() > 0
        if params.utc is not None:
            self._n2_frame_hst = params.utc - dt.timedelta(
                hours=engine.HST_TO_UTC_HOURS)
        if result is None:      # AUTOFIND off: display and wait for a click
            self._nirc2_show_frame_only(params)
        else:
            # no [psf-clean] line here: this path never cleans any more
            # (_nirc2_start_worker passes psf_clean=False)
            self._nirc2_display(result)

    def _nirc2_show_frame_only(self, params):
        """AUTOFIND-off GO!: show the reduced frame (no measurement, no
        circles) and prompt for the star click; the identity readouts fill
        from the header so the target can be looked up meanwhile."""
        self._n2_pick_locked = False
        self._n2_last_draw = (
            f"Image {self._n2_imno} — {self._nirc2_mode_text(params)}"
            " — CLICK ON THE STAR", None)
        self._nirc2_draw_main()
        for box in (self.n2_strehl_out, self.n2_fwhm_out, self.n2_wfe_out,
                    self.n2_pred_sr, self.n2_pred_fwhm, self.n2_dsr,
                    self.n2_dfwhm):
            box.setText("")
        self._nirc2_show_identity(params)
        self.n2_warn.setText("")

    @staticmethod
    def _nirc2_mode_text(params):
        return {True: "LGS", False: "NGS"}.get(params.lgs, "AO mode unknown")

    def _nirc2_show_identity(self, params):
        name = params.object_name
        self.n2_object.setText(name)
        self.n2_ra.setText(params.ra)
        self.n2_dec.setText(params.dec)
        self.n2_set_target.setEnabled(bool(name or params.ra))
        self.n2_tt_star.setEnabled(
            engine.tt_star_offset(getattr(self, "_n2_header", None) or {})
            is not None)
        self.n2_ttstar_out.setText("")

    def _on_nirc2_frame_failed(self, imno, message):
        self.n2_log.appendPlainText(f"Image {imno}: {message}")

    def _on_nirc2_finished(self):
        self.n2_go.setEnabled(True)
        # anything still queued behind a dialog when the run ended
        self._nirc2_drain_pending_duplicates()

    def _on_nirc2_motion(self, event):
        """AUTOFIND OFF: live magnifier while picking the star — the
        MEASURED STAR panel follows the cursor with a crosshair, because
        the pointer itself hides faint cores (Eduardo 2026-07-23)."""
        if (self.n2_autofind.isChecked() or self._n2_image is None
                or self._n2_pick_locked
                or event.xdata is None or event.ydata is None
                or event.inaxes is None):
            return
        img = self._n2_image
        r = 24
        x = int(event.xdata)
        y = int(event.ydata)
        x0 = min(max(x - r, 0), max(img.shape[1] - 2 * r, 0))
        y0 = min(max(y - r, 0), max(img.shape[0] - 2 * r, 0))
        cut = img[y0:y0 + 2 * r, x0:x0 + 2 * r]
        self.n2_fig_star.clear()
        ax = self.n2_fig_star.add_subplot(111)
        disp, vmin, vmax = self._nirc2_scaled(cut)
        ax.imshow(disp, cmap="gray", origin="lower", vmin=vmin, vmax=vmax,
                  interpolation="nearest")
        ax.axvline(x - x0, color="yellow", lw=0.6)
        ax.axhline(y - y0, color="yellow", lw=0.6)
        ax.set_xticks([]); ax.set_yticks([])
        self.n2_cap_star.setText("PICK ZOOM")
        self.n2_canvas_star.draw_idle()

    def _on_nirc2_pick_sky(self):
        """Arm sky picking, or clear an existing picked value."""
        if self._n2_sky_override is not None:
            self._n2_sky_override = None
            self.n2_pick_sky.setChecked(False)
            self.n2_pick_sky.setText("Pick sky")
            self.n2_log.appendPlainText("sky override cleared — back to the "
                                        "annulus estimator")
            return
        if self.n2_pick_sky.isChecked():
            self.n2_log.appendPlainText(
                "pick sky: click an EMPTY patch in the image")

    def _nirc2_set_sky_from(self, event):
        """Armed Pick sky + canvas click: 41x41 px σ-clipped median."""
        img = self._n2_image
        r = 20
        x = int(event.xdata)
        y = int(event.ydata)
        x0 = min(max(x - r, 0), max(img.shape[1] - 2 * r - 1, 0))
        y0 = min(max(y - r, 0), max(img.shape[0] - 2 * r - 1, 0))
        val = engine.sigma_clipped_median(img[y0:y0 + 2 * r + 1,
                                              x0:x0 + 2 * r + 1])
        self._n2_sky_override = val
        self.n2_pick_sky.setChecked(False)
        self.n2_pick_sky.setText(f"Sky {val:.1f} (clear)")
        self.n2_log.appendPlainText(
            f"sky picked at ({x}, {y}): {val:.2f} ADU (41x41 σ-clipped "
            "median) — used instead of the annulus until cleared")

    def _on_nirc2_click(self, event):
        """AUTOFIND OFF: re-measure the displayed frame at the clicked pixel
        (replaces the IDL widget's 'CLICK ON THE STAR' cursor prompt).
        With Pick sky armed, the click sets the sky patch instead."""
        if (self.n2_pick_sky.isChecked() and self._n2_image is not None
                and event.xdata is not None and event.inaxes is not None):
            self._nirc2_set_sky_from(event)
            return
        if (self._n2_image is None or event.xdata is None
                or event.ydata is None or event.inaxes is None):
            return
        if self.n2_add_star.isChecked():
            result = self._nirc2_measure_at(event.xdata, event.ydata)
            verdict = self._nirc2_field_accept(result)
            if verdict is None:
                self._nirc2_add_to_field(result)
            else:
                self.n2_log.appendPlainText(f"field: not added — {verdict}")
            self._n2_pick_locked = True
            return
        if self.n2_autofind.isChecked():
            return
        result = self._nirc2_measure_at(event.xdata, event.ydata)
        self._n2_pick_locked = True    # zoom freezes on the measured star

    def _on_nirc2_map_ext_click(self, event):
        """Pop-out-only mirror of _on_nirc2_click's add-star branch: while
        'Add star by click' is armed AND the pop-out is showing the real
        image backdrop (show_image=True -- _nirc2_draw_map_into never draws
        one otherwise, so there's nothing meaningful to click), a click on
        the pop-out map measures that pixel and adds it to the field,
        exactly like clicking the Image sub-tab -- no tab switch needed.
        The click arrives in arcsec (the map's plot frame); this inverts
        the SAME ps/half-extent transform _nirc2_draw_map_into uses to
        place the markers, to recover the detector pixel _nirc2_measure_at
        expects."""
        if not self.n2_add_star.isChecked():
            return
        if not getattr(self, "_n2_map_ext_show_image", False):
            return
        if (self._n2_image is None or self._n2_params is None
                or event.xdata is None or event.ydata is None
                or event.inaxes is None):
            return
        ps = self._n2_params.plate_scale_mas / 1000.0
        h, w = self._n2_image.shape
        px = event.xdata / ps + w / 2.0
        py = event.ydata / ps + h / 2.0
        result = self._nirc2_measure_at(px, py)
        verdict = self._nirc2_field_accept(result)
        if verdict is None:
            self._nirc2_add_to_field(result)
        else:
            self.n2_log.appendPlainText(f"field: not added — {verdict}")
        self._n2_pick_locked = True

    def _nirc2_add_to_field(self, result):
        """Add a manually-clicked measurement to the field map, honouring
        D20 exactly as the auto-find tick loop does: a psf_clean-excluded
        star (cleaning would have removed almost all of its own aperture
        flux) goes to `_n2_field_dropped` instead of `_n2_field` -- same
        ×/reinsert mechanism as a field-consistency outlier, no second
        one. A field_consistent outlier, by contrast, is still ADDED with
        a note ("kept because you placed it") -- that override is about a
        statistical judgment call the user is entitled to overrule;
        psf_clean_excluded is not a judgment call, it is the fact that
        the reported number would mostly be the ePSF model, not the
        star."""
        if result.psf_clean_excluded:
            self._n2_field_dropped = (
                getattr(self, "_n2_field_dropped", None) or []) + [result]
            self._nirc2_draw_map()
            self.n2_log.appendPlainText(
                f"field: star at pos {result.x:.1f} {result.y:.1f} left "
                f"off the map — {100 * result.subtracted_frac:.1f}% of "
                "its aperture flux was neighbour light (reinsertable)")
            return
        self._n2_field.append(result)
        self._nirc2_draw_map()
        note = ""
        _keep, dropped = engine.field_consistent(self._n2_field)
        if any(r is result for r in dropped):
            note = ("  [field outlier — kept because you placed "
                    "it; check crowding]")
        self.n2_log.appendPlainText(
            f"field: star added ({len(self._n2_field)} on the "
            f"map){note}")

    def _nirc2_pick_psf_clean(self):
        """Whether a single-star pick/click should be PSF-fit cleaned.

        The checkbox alone is not enough (Eduardo 2026-08-07): cleaning is a
        MEASURE-FIELD feature, because the empirical PSF is a property of
        the field and takes seconds to build. A pick therefore cleans only
        when "Measure field" has ALREADY built that model for the frame on
        screen -- then it is free, and it is also the same model the rest of
        the map was measured against, which is the consistency argument
        _nirc2_measure_field_setup makes for building it once. Before that,
        picks measure exactly as the summit tool does."""
        return bool(self.n2_psf_clean.isChecked()
                    and getattr(self, "_n2_field_epsf", None) is not None)

    def _nirc2_measure_at(self, x, y):
        """Single-star measurement at (x, y) on the displayed frame with
        the current photometry settings; displays and returns the result.
        Cleans only against an already-built field model -- see
        _nirc2_pick_psf_clean; the model is passed in rather than rebuilt,
        so a click costs no more than it did before cleaning existed."""
        photrad, bgin, bgout, peakrad = self._nirc2_radii()
        clean = self._nirc2_pick_psf_clean()
        result = engine.measure_strehl(
            self._n2_image, params=self._n2_params, pos=(x, y),
            background_subtracted=self._n2_bg_used,
            photometry_radius_arcsec=photrad, bg_inner_arcsec=bgin,
            bg_outer_arcsec=bgout, peak_radius_arcsec=peakrad,
            dl_psf=self._n2_dl, robust_sky=self.n2_robust_sky.isChecked(),
            sky_override=self._n2_sky_override,
            auto_radius=self.n2_auto_rad.isChecked(),
            psf_clean=clean, epsf=self._n2_field_epsf if clean else None,
            star_catalog=self._n2_field_catalog if clean else None)
        if clean:
            self._nirc2_log_psf_clean(result)
        elif self.n2_psf_clean.isChecked():
            self.n2_log.appendPlainText(
                "  [psf-clean] skipped — no field model for this frame; "
                "run 'Measure field' first and later picks will reuse it")
        self._nirc2_display(result)
        return result

    def _nirc2_log_psf_clean(self, r):
        """One [psf-clean] log line per measurement -- the engine's own
        `psf_clean_note` already reads correctly for both a real
        subtraction ("[psf-clean] N neighbour(s) subtracted; X% of the
        aperture flux; residual Y%...") and every null/refusal outcome
        ("0 neighbours above the 0.1% contamination floor…", "cleaning
        refused: annulus contamination got WORSE…") -- printed VERBATIM,
        never re-worded (WP-3 handoff), just normalized to always carry
        exactly one leading "[psf-clean] " tag regardless of which of
        those two forms the engine produced."""
        note = r.psf_clean_note
        if note.startswith("[psf-clean] "):
            note = note[len("[psf-clean] "):]
        self.n2_log.appendPlainText(f"  [psf-clean] {note}")
        # D27: state which WAY the number is likely wrong, on every
        # measurement that was actually cleaned. The engine decides the
        # wording (UNDERESTIMATE below the validated envelope, an explicit
        # OVERESTIMATE warning above it); the GUI only has to not swallow
        # it. A residual error the observer knows the sign of is usable --
        # an unsigned one is not -- and erring low is only the "safe"
        # direction if the observer is told that is what happened.
        if r.psf_clean_bias:
            self.n2_log.appendPlainText(f"  [psf-clean] {r.psf_clean_bias}")
        # D25: warn (never refuse) when cleaning ran and landed above the
        # validated envelope -- the restriction is about confidence, not
        # correctness. Complements the bias line above: that one names the
        # direction, this one the magnitude and where it was measured.
        if r.cleaned and r.strehl > engine.PSF_FIT_SR_VALIDATED_MAX:
            self.n2_log.appendPlainText(f"  [psf-clean] {engine.PSF_FIT_SR_ENVELOPE_NOTE}")

    # ---- measured field map --------------------------------------------------
    def _on_nirc2_measure_field(self):
        """Auto-find the N brightest stars, then measure them one per
        timer tick so the UI stays live: the button counts progress, each
        star flashes in the MEASURED STAR panel as it is measured, and
        kept points pop onto the map as they land.  Negative/over-unity
        SR and saturated stars are rejected with a log line."""
        if (getattr(self, "_n2_field_queue", None)
                or getattr(self, "_n2_field_busy", False)):
            return                          # already running
        if self._n2_image is None or self._n2_params is None:
            self.n2_log.appendPlainText(
                "! measure a frame first — the field map works on the "
                "displayed image")
            return
        photrad_px = (self.n2_photrad.value() * 1000.0
                      / self._n2_params.plate_scale_mas)
        n_req = self.n2_nstars.value() or 30    # 0 = Auto: quality decides
        self._n2_field_auto = self.n2_nstars.value() == 0
        # Everything from here to the first timer tick runs on the GUI
        # thread and can take 10-20 s on a crowded field (the deep
        # catalogue alone measured 14 s on a Galactic-Centre frame), so
        # each stage announces itself BEFORE it starts and the event loop
        # is flushed so the line actually paints. `_n2_field_busy` is set
        # first and checked above: flushing the loop lets the user click
        # "Measure field" again, and re-entering this prologue would build
        # a second catalogue and clobber the queue.
        self._n2_field_busy = True
        self.n2_field_btn.setEnabled(False)
        QtWidgets.QApplication.setOverrideCursor(
            Qt.CursorShape.WaitCursor)
        try:
            self._nirc2_stage("finding stars…")
            # deeper candidate list so rejected stars don't consume map
            # slots; the detection floors are the natural "no more good
            # stars" stop
            positions = engine.find_stars(
                self._n2_image, n_stars=max(3 * n_req, n_req + 20),
                exclude_px=photrad_px)
            if not positions:
                self.n2_log.appendPlainText(
                    "field: no stars found above the detection floors")
                return
            self._nirc2_measure_field_setup(positions, n_req)
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
            self._n2_field_busy = False
            if not getattr(self, "_n2_field_queue", None):
                self.n2_field_btn.setEnabled(True)
                self.n2_field_btn.setText("Measure field")

    def _nirc2_stage(self, message):
        """Announce a blocking stage and force it to paint.

        These stages run on the GUI thread, so appending to the log is not
        enough -- without flushing the event loop the text sits in the
        widget unpainted until the work finishes, which is exactly the
        blind hang this exists to remove. Re-entrancy is guarded by
        `_n2_field_busy` in the caller.
        """
        self.n2_field_btn.setText(f"Measuring… ({message.rstrip('…')})")
        self.n2_log.appendPlainText(f"  field: {message}")
        QtWidgets.QApplication.processEvents()

    def _nirc2_measure_field_setup(self, positions, n_req):
        """The rest of the prologue, stage-announced. Split out so the
        cursor/busy-flag cleanup in the caller covers every exit path."""
        self._n2_field = []
        self._n2_field_dropped = []
        self._n2_ee_pairs = {}          # id(small-ap result) -> full-radius result
        self._n2_ee_done = set()        # ids already EE-decided
        self._nirc2_clear_selection()
        self._nirc2_set_ee_readout("")
        self._nirc2_draw_map()
        self._n2_field_epsf = None
        self._n2_field_catalog = None
        if self.n2_psf_clean.isChecked():
            # the ePSF and deep neighbour catalogue are properties of the
            # FIELD -- built ONCE here and shared across every tick, exactly
            # as measure_field does internally and for the same reason:
            # rebuilding per star is both slow and inconsistent (each star
            # cleaned against a slightly different model). Progress
            # messaging (this line) is the GUI's job per the WP-3 handoff;
            # the build itself is the engine's.
            work = engine.sigma_filter3(self._n2_image)
            self._nirc2_stage("building the deep neighbour catalogue…")
            self._n2_field_catalog = engine.deep_star_catalog(
                work, self._n2_params)
            n_cat = len(self._n2_field_catalog)
            if getattr(self._n2_field_catalog, "truncated", False):
                self.n2_log.appendPlainText(
                    f"  field: catalogue capped at {n_cat} stars (the "
                    "brightest); fainter detections are not modelled as "
                    "neighbours")
            self._nirc2_stage(
                f"building the field ePSF from {n_cat} catalogued stars…")
            self._n2_field_epsf = engine.build_epsf(
                work, self._n2_params, catalog=self._n2_field_catalog)
            ep = self._n2_field_epsf
            self.n2_log.appendPlainText(
                f"  [psf-clean] field ePSF: tag={ep.tag!r} "
                f"delta={ep.delta:.4f} converged={ep.converged} "
                f"phase_coverage={ep.phase_coverage:.0%}"
                + ("" if ep.usable else " -- cleaning will be skipped "
                                        "for every star this field"))
        self._n2_field_queue = list(positions)
        self._n2_field_target = n_req
        self._n2_field_tried = 0
        self._n2_field_poor = 0
        self.n2_field_btn.setEnabled(False)
        QtCore.QTimer.singleShot(0, self._nirc2_field_tick)

    def _nirc2_field_tick(self):
        queue = self._n2_field_queue
        kept = len(self._n2_field)
        if not queue or kept >= self._n2_field_target:
            self._nirc2_apply_ee()      # convention fix BEFORE the clip
            # field self-consistency (Eduardo: 0.61 next to 0.35 next to
            # 0.09 in one frame is not physics): gradient-aware MAD
            # filter in both metrics -- residuals about a robust plane,
            # so stars riding the anisoplanatic falloff stay (Eduardo
            # 2026-07-25: the median clip was discarding the well-
            # corrected minority on the asterism side).  Dropped stars
            # stay on the map as × markers -- click one to inspect and
            # reinsert it; dropped outliers hand their slots back to
            # the backfill when candidates remain
            keep, dropped = engine.field_consistent(self._n2_field)
            if dropped:
                s_med = float(np.median([r.strehl for r in keep])) \
                    if keep else float("nan")
                for r in dropped:
                    self.n2_log.appendPlainText(
                        f"  rejected as field outlier — SR {r.strehl:.2f} "
                        f"/ FWHM {r.fwhm_mas:.1f} mas off the local trend "
                        f"(field median SR {s_med:.2f}) — × on the map, "
                        f"click to inspect / reinsert")
                self._n2_field_dropped = (
                    getattr(self, "_n2_field_dropped", None) or []) + dropped
                self._n2_field = keep
                self._nirc2_clear_selection()   # indices shifted
                self._nirc2_draw_map()
                if queue and len(keep) < self._n2_field_target:
                    QtCore.QTimer.singleShot(10, self._nirc2_field_tick)
                    return
            self._n2_field_queue = None
            self.n2_field_btn.setEnabled(True)
            self.n2_field_btn.setText("Measure field")
            self.n2_cap_star.setText("MEASURED STAR")
            if self._n2_field_auto:
                summary = (f"field: kept {kept} quality star(s) — auto "
                           f"stop at SR noise ±{engine.SR_ERR_MAX} "
                           f"({self._n2_field_tried} candidate(s) tried)")
            else:
                summary = (f"field: kept {kept} of "
                           f"{self._n2_field_target} requested "
                           f"({self._n2_field_tried} candidate(s) tried)")
                if kept < self._n2_field_target:
                    summary += (" — field exhausted above the detection "
                                "floors")
            self.n2_log.appendPlainText(summary)
            self._nirc2_draw_map()
            if getattr(self, "_n2_field_st", None) is not None:
                self.n2_log.appendPlainText(
                    "field stats: "
                    + self._nirc2_field_stats_text(self._n2_field_st))
            return
        x, y = queue.pop(0)
        self._n2_field_tried += 1
        k = self._n2_field_tried
        self.n2_field_btn.setText(
            f"Measuring… {kept}/{self._n2_field_target} kept ({k} tried)")
        photrad, bgin, bgout, peakrad = self._nirc2_radii()
        r = engine.measure_strehl(
            self._n2_image, params=self._n2_params, pos=(x, y),
            background_subtracted=self._n2_bg_used,
            photometry_radius_arcsec=photrad, bg_inner_arcsec=bgin,
            bg_outer_arcsec=bgout, peak_radius_arcsec=peakrad,
            dl_psf=self._n2_dl, robust_sky=self.n2_robust_sky.isChecked(),
            sky_override=self._n2_sky_override,
            auto_radius=self.n2_auto_rad.isChecked(),
            psf_clean=self.n2_psf_clean.isChecked(),
            epsf=self._n2_field_epsf, star_catalog=self._n2_field_catalog)
        self._nirc2_flash_star(r, k)
        verdict = self._nirc2_field_accept(r)
        if verdict is None and self._n2_field_auto and r.sr_err > engine.SR_ERR_MAX:
            self._n2_field_poor = getattr(self, "_n2_field_poor", 0) + 1
            verdict = (f"quality gate — ±{r.sr_err:.3f} SR noise "
                       f"(limit ±{engine.SR_ERR_MAX})")
            if self._n2_field_poor >= 2:
                self._n2_field_queue = []   # fainter ones only get worse
        elif verdict is None:
            self._n2_field_poor = 0
        if verdict is None and self.n2_psf_clean.isChecked():
            self._nirc2_log_psf_clean(r)
        if verdict is None and r.psf_clean_excluded:
            # D20: cleaning was refused because it would have removed
            # almost all of the star's own aperture flux (neighbour light
            # dominated it) -- left OFF the map by default, same × /
            # reinsert mechanism as a field-consistency outlier, no
            # second mechanism.
            self._n2_field_dropped = (
                getattr(self, "_n2_field_dropped", None) or []) + [r]
            self.n2_log.appendPlainText(
                f"  star {k}: left off the map — "
                f"{100 * r.subtracted_frac:.1f}% of its aperture flux was "
                "neighbour light (reinsertable)")
            self._nirc2_draw_map()
        elif verdict is None:
            self._n2_field.append(r)
            # EE aperture correction: a small-aperture star gets a
            # companion FULL-radius measurement; clean pairs calibrate
            # the field's h at completion (Eduardo 2026-07-25)
            if (self.n2_ee_corr.isChecked()
                    and self.n2_auto_rad.isChecked()
                    and r.photrad_used_arcsec < photrad * 0.9):
                full = engine.measure_strehl(
                    self._n2_image, params=self._n2_params,
                    pos=(r.x, r.y),
                    background_subtracted=self._n2_bg_used,
                    photometry_radius_arcsec=photrad,
                    bg_inner_arcsec=bgin, bg_outer_arcsec=bgout,
                    peak_radius_arcsec=peakrad,
                    dl_psf=self._n2_dl,
                    robust_sky=self.n2_robust_sky.isChecked(),
                    sky_override=self._n2_sky_override,
                    auto_radius=False)
                if full.ok and 0 < full.strehl < 1:
                    self._n2_ee_pairs[id(r)] = full
            self._nirc2_draw_map()
            self.n2_log.appendPlainText(
                f"  star {k}: SR {r.strehl:.3f} ±{r.sr_err:.3f}  "
                f"FWHM {r.fwhm_mas:6.2f} mas  pos {r.x:6.1f} {r.y:6.1f}  "
                f"r={r.photrad_used_arcsec:.2f}\""
                + ("  [edge]" if r.edge else "")
                + ("  [crowded]" if r.crowded else ""))
        else:
            self.n2_log.appendPlainText(f"  star {k}: rejected — {verdict}")
        QtCore.QTimer.singleShot(10, self._nirc2_field_tick)

    def _nirc2_apply_ee(self):
        """EE aperture correction at field completion (Eduardo
        2026-07-25): stars whose full-radius companion measurement is
        CLEAN take that value directly (it IS the convention value);
        crowded ones get the growth-curve correction with the field's
        own fitted h. Runs before the consistency clip so the clip
        sees final-convention values; idempotent across backfill
        passes via the done-set."""
        import dataclasses
        pairs = getattr(self, "_n2_ee_pairs", None)
        if not self.n2_ee_corr.isChecked() or not pairs:
            return
        done = self._n2_ee_done
        todo = [i for i, r in enumerate(self._n2_field)
                if id(r) in pairs and id(r) not in done]
        if not todo:
            return
        # calibrators: the CLEANEST full-radius companions this field
        # offers. Strictly-clean first; in extreme crowding (the GC
        # case: ~92% contamination everywhere, zero pristine
        # companions) fall back to the least-crowded quartile with a
        # loose-calibration tag -- and ALWAYS log the outcome
        # (2026-07-25: the first cut silently did nothing on GC).
        cands = [(r, pairs[id(r)]) for r in self._n2_field
                 if id(r) in pairs and not pairs[id(r)].saturated]
        clean = [(r, f) for r, f in cands if not f.crowded]
        loose = False
        if len(clean) < 5 and len(cands) >= 5:
            cands.sort(key=lambda p: p[1].crowding)
            clean = cands[:max(5, len(cands) // 4)]
            loose = True
        calib = [(r.strehl, f.strehl) for r, f in clean]
        h = None
        if calib:
            try:
                h, _rms = engine.ee_calibrate_h(calib)
            except ValueError:      # too few pairs this field
                h = None
        n_full = n_corr = 0
        for i in todo:
            r = self._n2_field[i]
            f = pairs[id(r)]
            if not f.crowded and not f.saturated:
                new = f                    # clean full radius = truth
                n_full += 1
            elif h is not None:
                sr_c = float(engine.ee_correct(r.strehl, h))
                scale = sr_c / max(r.strehl, 1e-6)
                new = dataclasses.replace(r, strehl=sr_c,
                                          sr_err=r.sr_err * scale)
                n_corr += 1
            else:
                done.add(id(r))
                continue
            self._n2_field[i] = new
            done.add(id(r))
            done.add(id(new))
            pairs[id(new)] = f
        if h is not None:
            tag = (" — LOOSE calibration: no pristine companions, "
                   f"used the {len(calib)} least-crowded (min "
                   f"contamination {min(f.crowding for _, f in clean):.0%})"
                   if loose else f", {len(calib)} clean pairs")
            self.n2_log.appendPlainText(
                f"EE aperture correction: h={h:.2f}{tag} — "
                f"{n_full} star(s) -> full-radius value, "
                f"{n_corr} corrected")
            self._nirc2_set_ee_readout(
                f"EE h = {h:.2f}{' (loose calibration)' if loose else ''} "
                f"— {len(calib)} pairs, {n_full} full-radius + "
                f"{n_corr} corrected")
        else:
            self.n2_log.appendPlainText(
                "EE aperture correction: h UNCALIBRATED — not enough "
                "usable two-aperture pairs in this field; values left "
                "in the small-aperture convention (interpret "
                "accordingly)")
            self._nirc2_set_ee_readout(
                "EE h UNCALIBRATED — too few usable pairs; values left "
                "in the small-aperture convention")

    @staticmethod
    def _nirc2_field_accept(r):
        """None if the point belongs on the map, else the rejection reason
        (matter-of-fact physics: Eduardo 2026-07-23)."""
        if not r.ok:
            return r.error or "measurement failed"
        if r.saturated:                 # cause before symptom: saturation
            return "saturated"          # is why the SR went strange
        if r.unphysical:
            return f"unphysical SR ({r.strehl:+.3f})"
        if r.fwhm_mas <= 0.0:
            return f"invalid FWHM ({r.fwhm_mas:.2f} mas)"
        return None

    def _nirc2_flash_star(self, r, k):
        """Flash the star being measured in the MEASURED STAR panel."""
        if not r.ok or self._n2_image is None:
            return
        img = self._n2_image
        rad = 24
        xi = int(min(max(r.x, rad), img.shape[1] - rad - 1))
        yi = int(min(max(r.y, rad), img.shape[0] - rad - 1))
        cut = img[yi - rad:yi + rad, xi - rad:xi + rad]
        self.n2_fig_star.clear()
        ax = self.n2_fig_star.add_subplot(111)
        disp, vmin, vmax = self._nirc2_scaled(cut)
        ax.imshow(disp, cmap="gray", origin="lower", vmin=vmin, vmax=vmax,
                  interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        self.n2_cap_star.setText(
            f"CANDIDATE {k} — SR {r.strehl:.2f}")
        self.n2_canvas_star.draw_idle()

    def _nirc2_clear_selection(self):
        self._n2_sel_star = None
        self._n2_sel_dropped = None
        self.n2_reject_star.setEnabled(False)
        self.n2_reject_star.setText("Reject star")

    def _on_nirc2_field_clear(self):
        self._n2_field = []
        self._n2_field_dropped = []
        self._nirc2_clear_selection()
        self._nirc2_set_ee_readout("")
        self._nirc2_draw_map()

    def _on_nirc2_map_pick(self, event):
        """Click a star on the field map (embedded or popped out):
        select it, show its full measurement in the Results block and
        its cutout in the MEASURED STAR panel, and arm the button —
        'Reject star' for a kept point, 'Reinsert star' for a rejected
        × (the same mechanism in the opposite direction, Eduardo
        2026-07-25)."""
        pool = getattr(getattr(event, "artist", None), "_n2_pool", "kept")
        stars = (getattr(self, "_n2_field_dropped", None)
                 if pool == "dropped"
                 else getattr(self, "_n2_field", None)) or []
        ind = getattr(event, "ind", None)
        if ind is None or not stars:
            return
        i = int(np.atleast_1d(ind)[0])
        if not 0 <= i < len(stars):
            return
        r = stars[i]
        self._n2_sel_star = i if pool == "kept" else None
        self._n2_sel_dropped = i if pool == "dropped" else None
        self._nirc2_display(r)          # results block + aperture circles
        tag = "FIELD STAR" if pool == "kept" else "REJECTED STAR"
        self.n2_cap_star.setText(f"{tag} {i + 1} — SR {r.strehl:.3f} "
                                 f"±{r.sr_err:.3f}")
        self.n2_reject_star.setText(
            "Reject star" if pool == "kept" else "Reinsert star")
        self.n2_reject_star.setEnabled(True)
        self._nirc2_draw_map()          # selection ring

    def _on_nirc2_reject_star(self):
        """Move the selected star OUT of the fit (kept → ×: user
        judgment — a blend, an edge case, an artifact) or BACK INTO it
        (× → kept: the user overrides the auto-clip or an earlier
        rejection), then regenerate every field statistic."""
        field = getattr(self, "_n2_field", None) or []
        dropped = getattr(self, "_n2_field_dropped", None) or []
        i, j = getattr(self, "_n2_sel_star", None), \
            getattr(self, "_n2_sel_dropped", None)
        if j is not None and 0 <= j < len(dropped):
            r = dropped.pop(j)
            field.append(r)
            verb = "reinserted into the fit by user"
        elif i is not None and 0 <= i < len(field):
            r = field.pop(i)
            dropped.append(r)
            verb = "rejected by user — × on the map, reinsertable"
        else:
            return
        self._n2_field, self._n2_field_dropped = field, dropped
        self._nirc2_clear_selection()
        self.n2_log.appendPlainText(
            f"field: star at pos {r.x:.1f} {r.y:.1f} {verb} — "
            f"SR {r.strehl:.3f}, FWHM {r.fwhm_mas:.1f} mas"
            + ("  [edge]" if r.edge else "")
            + ("  [crowded]" if r.crowded else ""))
        self._nirc2_draw_map()
        if self._n2_field_st is not None:
            self.n2_log.appendPlainText(
                "field stats: "
                + self._nirc2_field_stats_text(self._n2_field_st))

    def _on_nirc2_map_popout(self):
        """The field map in its own resizable window (the ranking-dialog
        pattern: recreated per click, redrawn with the embedded map)."""
        if getattr(self, "_n2_map_dialog", None) is not None:
            self._n2_map_dialog.close()
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Measured field map")
        dlg.resize(900, 800)
        lay = QtWidgets.QVBoxLayout(dlg)
        fig = Figure(figsize=(8.5, 7.5), layout="constrained")
        canvas = FigureCanvasQTAgg(fig)
        canvas.mpl_connect("pick_event", self._on_nirc2_map_pick)
        canvas.mpl_connect("button_press_event", self._on_nirc2_map_ext_click)
        lay.addWidget(canvas, 1)
        img_toggle = QtWidgets.QCheckBox("Show image (hollow markers)")
        img_toggle.setToolTip(
            "Overlay the measurements on the actual frame instead of a "
            "plain background -- markers become hollow rings (colour "
            "still carries the SR/FWHM value) so the star underneath is "
            "visible. Pop-out only: the embedded map stays as it was.")
        img_toggle.setChecked(
            getattr(self, "_n2_map_ext_show_image", False))
        img_toggle.toggled.connect(self._on_nirc2_map_ext_image_toggle)
        lay.addWidget(img_toggle)
        stats = QtWidgets.QLabel("")
        stats.setWordWrap(True)
        stats.setStyleSheet("font-size: 11px;")
        lay.addWidget(stats)
        ee_lab = QtWidgets.QLabel(self.n2_ee_out.text())
        ee_lab.setWordWrap(True)
        ee_lab.setStyleSheet(
            "QLabel { font-weight:bold; font-size:11px; padding:3px 6px; "
            "border-radius:3px; background:#fff3cd; color:#664d03; }")
        # seed from the embedded label's CURRENT value -- _nirc2_draw_map()
        # only recomputes field-stats text on every call, not the EE
        # readout (that's pushed separately, from _nirc2_apply_ee), so a
        # freshly-opened dialog must start in sync itself
        ee_lab.setVisible(bool(self.n2_ee_out.text()))
        lay.addWidget(ee_lab)
        self._n2_map_ext = (fig, canvas)
        self._n2_map_ext_stats = stats
        self._n2_map_ext_ee = ee_lab
        self._nirc2_draw_map()

        def _detach(*_):
            self._n2_map_ext = None
            self._n2_map_ext_stats = None
            self._n2_map_ext_ee = None
        dlg.finished.connect(_detach)
        dlg.show()
        self._n2_map_dialog = dlg

    def _on_nirc2_map_ext_image_toggle(self, on):
        """Pop-out-only background-image mode (Eduardo 2026-07-26):
        remembered across repeated pop-outs, never affects the embedded
        map (too small for the image to help there)."""
        self._n2_map_ext_show_image = on
        self._nirc2_draw_map()

    def _on_nirc2_log_popout(self):
        """The measurement log in its own resizable window: a free-text tab
        (a SECOND VIEW onto the same QTextDocument, Qt keeps both in sync
        automatically as lines are appended -- unchanged from before) plus
        an "Image log" tab -- one sortable row per measurement, the
        structured counterpart of the free-text log (self._n2_csv_rows) --
        with an on-demand "Export CSV..." button (Eduardo: exporting should
        be optional, not automatic on every measurement). Same recreate-
        per-click pattern as the field-map pop-out."""
        if getattr(self, "_n2_log_dialog", None) is not None:
            self._n2_log_dialog.close()
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Measurement log")
        dlg.resize(1000, 600)
        lay = QtWidgets.QVBoxLayout(dlg)
        tabs = QtWidgets.QTabWidget()
        lay.addWidget(tabs, 1)

        text_view = QtWidgets.QPlainTextEdit()
        text_view.setReadOnly(True)
        text_view.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.NoWrap)
        text_view.setFont(self.n2_log.font())
        text_view.setDocument(self.n2_log.document())
        tabs.addTab(text_view, "Text log")

        image_log_tab = QtWidgets.QWidget()
        il_v = QtWidgets.QVBoxLayout(image_log_tab)
        table = QtWidgets.QTableWidget(len(self._n2_csv_rows),
                                       len(NIRC2_CSV_COLUMNS))
        table.setHorizontalHeaderLabels([lbl for _k, lbl in NIRC2_CSV_COLUMNS])
        table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self._nirc2_fill_csv_table(table)
        table.setSortingEnabled(True)
        table.resizeColumnsToContents()
        il_v.addWidget(table, 1)
        export_row = QtWidgets.QHBoxLayout()
        delete_btn = QtWidgets.QPushButton("Delete selected")
        delete_btn.setToolTip(
            "Remove the selected row(s) from the log (select rows, then "
            "click this -- no confirmation, but nothing is written to disk "
            "until you Export).")
        delete_btn.clicked.connect(self._on_nirc2_delete_csv_rows)
        export_row.addWidget(delete_btn)
        export_btn = QtWidgets.QPushButton("Export CSV…")
        export_btn.setToolTip(
            "Save every measurement logged this session to a CSV file "
            "(target coordinates, telescope pointing, filter, LBWFS "
            "seeing, DIMM/MASS seeing, and time, alongside the "
            "measured/predicted SR and FWHM).")
        export_btn.clicked.connect(self._on_nirc2_export_csv)
        export_row.addWidget(export_btn)
        export_row.addStretch(1)
        il_v.addLayout(export_row)
        tabs.addTab(image_log_tab, "Image log")
        self._n2_csv_table = table

        def _detach(*_):
            self._n2_log_dialog = None
            self._n2_csv_table = None
        dlg.finished.connect(_detach)
        dlg.show()
        self._n2_log_dialog = dlg

    # text columns sort correctly as plain strings (time_utc is ISO-8601,
    # so lexicographic == chronological); every other column WANTS numeric
    # sorting (SortableItem), but frame_number needs care: workers.py's
    # Nirc2MeasureWorker always emits the frame number as a str (str(no)
    # for a numbered sequence, or an arbitrary file-list label when
    # self.files overrides the numbering) -- feeding that string straight
    # into f"{v:.4g}" is exactly what crashed the app (Eduardo 2026-07-28:
    # ValueError raised from inside a Qt callback -> SIGABRT, not a caught
    # Python exception). See the numeric branch below for the fix: parse
    # first, sort by the parsed number, but display and fall back to the
    # original string so an actual filename label still renders instead of
    # erroring.
    _CSV_TEXT_COLS = {"time_utc", "target_name", "ao_mode", "guide_star",
                      "guide_mag_src", "ra", "dec", "filter"}

    # Nothing in this log is meaningful past 3 decimal places (Eduardo
    # 2026-07-28) -- SR to 0.001, FWHM/pixel/Az/El well past their real
    # precision by then. Applied to BOTH the table and the CSV export so
    # the file matches what's on screen.
    _CSV_DECIMALS = 3

    @classmethod
    def _csv_round(cls, v):
        """`v` rounded to _CSV_DECIMALS -- the value written to CSV (stays
        a number, so the file remains machine-readable)."""
        return round(float(v), cls._CSV_DECIMALS)

    @classmethod
    def _csv_text(cls, v):
        """`v` as display text: at most _CSV_DECIMALS places, with trailing
        zeros trimmed so columns don't read as falsely precise (0.42 not
        0.420, 137.34 not 137.340)."""
        s = f"{float(v):.{cls._CSV_DECIMALS}f}"
        return s.rstrip("0").rstrip(".") if "." in s else s

    def _nirc2_fill_csv_table(self, table):
        """(Re)populate `table` from self._n2_csv_rows -- called when the
        popout is built and after every new measurement while it's open.
        Column 0's item carries the row's REAL index into self._n2_csv_rows
        (Qt.ItemDataRole.UserRole, the same trick starlist_picker.py uses):
        once the table is sorted, the visual row no longer equals the list
        index, so _on_nirc2_delete_csv_rows must never delete by row
        number directly."""
        table.setSortingEnabled(False)
        table.setRowCount(len(self._n2_csv_rows))
        for i, row in enumerate(self._n2_csv_rows):
            for j, (key, _label) in enumerate(NIRC2_CSV_COLUMNS):
                v = row.get(key)
                if key in self._CSV_TEXT_COLS:
                    item = QtWidgets.QTableWidgetItem("—" if v is None
                                                      else str(v))
                elif v is None:
                    item = SortableItem("—", None)
                elif isinstance(v, str):
                    # frame_number in particular: workers.py always emits
                    # it as a str. Parse for a numeric SORT key when it
                    # looks like one (the common "26"/"0026" case), but
                    # keep the ORIGINAL string as both the display text and
                    # the fallback -- an arbitrary file-list label (not a
                    # number at all) must render, not crash.
                    try:
                        item = SortableItem(v, float(v))
                    except ValueError:
                        item = QtWidgets.QTableWidgetItem(v)
                else:
                    try:
                        item = SortableItem(self._csv_text(v), v)
                    except (TypeError, ValueError):
                        # a "numeric" column holding something unformattable
                        # (shouldn't happen, but this table must never fail
                        # to render over one bad cell) -- fall back to a
                        # plain text item rather than losing the whole row
                        item = QtWidgets.QTableWidgetItem(str(v))
                if j == 0:
                    item.setData(Qt.ItemDataRole.UserRole, i)
                table.setItem(i, j, item)
        table.setSortingEnabled(True)

    def _on_nirc2_delete_csv_rows(self):
        """Delete every selected row from self._n2_csv_rows (see
        _nirc2_fill_csv_table's UserRole note for why this can't just use
        the visual row number)."""
        table = self._n2_csv_table
        if table is None:
            return
        vis_rows = {idx.row() for idx in table.selectedIndexes()}
        if not vis_rows:
            self.status.setText(
                "Select one or more rows in the Image log first.")
            return
        list_idxs = sorted(
            {table.item(r, 0).data(Qt.ItemDataRole.UserRole)
             for r in vis_rows} - {None},
            reverse=True)
        for i in list_idxs:
            del self._n2_csv_rows[i]
        self._nirc2_fill_csv_table(table)
        self.status.setText(
            f"Deleted {len(list_idxs)} measurement row(s) from the log.")

    def _on_nirc2_export_csv(self):
        """On-demand CSV export (Eduardo: optional, not automatic) of every
        measurement logged this session, in NIRC2_CSV_COLUMNS' order."""
        if not self._n2_csv_rows:
            self.status.setText("No measurements logged yet to export.")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export measured-SR CSV", "", "CSV files (*.csv)")
        if not path:
            return
        import csv
        try:
            with open(path, "w", newline="", encoding="utf-8") as fh:
                w = csv.writer(fh)
                w.writerow([label for _k, label in NIRC2_CSV_COLUMNS])
                for row in self._n2_csv_rows:
                    out = []
                    for k, _label in NIRC2_CSV_COLUMNS:
                        v = row.get(k)
                        if v is None:
                            out.append("")
                        elif isinstance(v, (int, float)) and not isinstance(
                                v, bool):
                            # same 3-decimal cap the table shows, kept as a
                            # NUMBER so the file stays machine-readable
                            out.append(self._csv_round(v))
                        else:
                            out.append(v)
                    w.writerow(out)
            self.status.setText(
                f"Exported {len(self._n2_csv_rows)} measurements to {path}")
        except OSError as e:
            self.status.setText(f"CSV export failed: {e}")

    def _nirc2_set_ee_readout(self, text):
        """The EE aperture correction's h result, shown prominently next
        to the field map (embedded + popped-out) instead of only in the
        scrolling log."""
        self.n2_ee_out.setText(text)
        self.n2_ee_out.setVisible(bool(text))
        ext = getattr(self, "_n2_map_ext_ee", None)
        if ext is not None:
            ext.setText(text)
            ext.setVisible(bool(text))

    def _nirc2_field_statistics(self):
        """FieldStats of the current measured field (arcsec offsets from
        the frame centre, the map's own axes), or None when empty."""
        field = self._n2_field
        if not field or self._n2_image is None:
            return None
        p = field[0].params
        ps = p.plate_scale_mas / 1000.0
        h, w = self._n2_image.shape
        xs = [(r.x - w / 2.0) * ps for r in field]
        ys = [(r.y - h / 2.0) * ps for r in field]
        return engine.field_statistics(
            xs, ys, [r.strehl for r in field],
            sr_errs=[r.sr_err for r in field],
            wavelength_um=p.effwave_um)

    def _nirc2_field_stats_text(self, st):
        if st is None:
            return ""
        txt = (f"peak SR {st.peak_sr:.3f} at "
               f"({st.peak_dx_arcsec:+.1f}\u2033, "
               f"{st.peak_dy_arcsec:+.1f}\u2033) \u00b7 "
               f"mean {st.mean_sr:.3f} \u00b1 {st.scatter_sr:.3f} "
               f"(n={st.n})")
        if st.grad_sr_per_arcmin is not None:
            txt += (f" \u00b7 gradient {st.grad_sr_per_arcmin:.3f} "
                    f"SR/\u2032 downhill at {st.grad_pa_deg:+.0f}\u00b0 "
                    "(det, CCW of +x)")
        else:
            txt += (f" \u00b7 gradient: needs \u2265"
                    f"{engine.GRADIENT_MIN_STARS} non-collinear stars")
        if st.theta0_arcsec is not None:
            err = (f" \u00b1 {st.theta0_err_arcsec:.1f}"
                   if st.theta0_err_arcsec is not None else "")
            txt += (f" \u00b7 \u03b8\u2080(eff) "
                    f"{st.theta0_arcsec:.1f}{err}\u2033 at "
                    f"{self._n2_params.effwave_um:.2f} \u00b5m "
                    f"({st.theta0_500nm_arcsec:.1f}\u2033 at 500 nm)")
        else:
            txt += f" \u00b7 \u03b8\u2080: {st.theta0_note}"
        return txt

    def _nirc2_draw_map(self):
        """Measured field map: SR at each measured star, in arcsec offsets
        from the frame centre (detector orientation) — the observational
        twin of the Field-map tab's model map.  Renders the embedded map
        and, when open, the popped-out copy; every redraw refreshes the
        field-statistics readout too."""
        try:
            self._n2_field_st = self._nirc2_field_statistics()
        except Exception:
            self._n2_field_st = None
        stats_txt = self._nirc2_field_stats_text(self._n2_field_st)
        self.n2_field_stats.setText(stats_txt)
        ext_lab = getattr(self, "_n2_map_ext_stats", None)
        if ext_lab is not None:
            ext_lab.setText(stats_txt)
        # background-image mode is pop-out only (Eduardo 2026-07-26: the
        # embedded map is too small for it to help) -- never applied here
        self._nirc2_draw_map_into(self.n2_map_fig, self.n2_map_canvas,
                                  show_image=False)
        ext = getattr(self, "_n2_map_ext", None)
        if ext is not None:
            fig, canvas = ext
            self._nirc2_draw_map_into(
                fig, canvas,
                show_image=getattr(self, "_n2_map_ext_show_image", False))

    def _nirc2_draw_map_into(self, fig, canvas, show_image=False):
        """show_image (pop-out only, Eduardo 2026-07-26): draw the actual
        frame as a background and switch the kept-star markers to
        hollow rings (colour still carries the SR/FWHM value) so a
        marker can be checked against the real star underneath it.
        Default False reproduces the original plain-background,
        filled-marker map exactly -- the embedded map always uses it;
        the pop-out toggles it via its own checkbox."""
        fig.clear()
        ax = fig.add_subplot(111)
        field = self._n2_field
        dropped = getattr(self, "_n2_field_dropped", None) or []
        if not field and not dropped:
            ax.text(0.5, 0.5, "Measure field (auto) or arm 'Add star by "
                    "click'\nto build a measured SR map of this frame",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=9, color="#777")
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_anchor("C")
            canvas.draw_idle()
            return
        p = (field or dropped)[0].params
        ps = p.plate_scale_mas / 1000.0
        h, w = self._n2_image.shape
        half_x, half_y = w / 2.0 * ps, h / 2.0 * ps
        if show_image:
            disp, im_vmin, im_vmax = self._nirc2_scaled(self._n2_image)
            ax.imshow(disp, cmap="gray", origin="lower",
                      extent=(-half_x, half_x, -half_y, half_y),
                      vmin=im_vmin, vmax=im_vmax, aspect="equal",
                      zorder=0)
        text_kw = {"path_effects": _TEXT_HALO} if show_image else {}
        xs = [(r.x - w / 2.0) * ps for r in field]
        ys = [(r.y - h / 2.0) * ps for r in field]
        fwhm_mode = self.n2_map_metric.currentText().startswith("FWHM")
        if fwhm_mode:
            vals = [r.fwhm_mas for r in field]
            dvals = [r.fwhm_mas for r in dropped]
            cmap, label, fmt = "viridis_r", "measured FWHM (mas)", "{:.1f}"
            vmin, vmax = (min(vals), max(vals)) if vals else (0.0, 1.0)
        else:
            vals = [r.strehl for r in field]
            dvals = [r.strehl for r in dropped]
            cmap, label, fmt = "viridis", "measured SR", "{:.2f}"
            vmin = min(0.0, min(vals)) if vals else 0.0
            vmax = max(vals) if vals else 1.0
        if vmax <= vmin:
            vmax = vmin + 1e-6
        if field:
            if show_image:
                # hollow: the image shows through the ring, so the
                # marker can be checked against the real star underneath
                # it. The ring colour still carries the SR/FWHM value
                # (and feeds the colorbar) via the collection's own
                # cmap/norm -- only the FACE is overridden to none.
                sc = ax.scatter(xs, ys, c=vals, cmap=cmap, s=120,
                                vmin=vmin, vmax=vmax, linewidths=2.0,
                                zorder=3)
                sc.set_facecolor("none")
                sc.set_edgecolor(sc.to_rgba(vals))
            else:
                sc = ax.scatter(xs, ys, c=vals, cmap=cmap, s=90,
                                vmin=vmin, vmax=vmax,
                                edgecolors="black", linewidths=0.5,
                                zorder=3)
            sc.set_picker(6)        # click a point: select + inspect
            sc._n2_pool = "kept"
            for xv, yv, v in zip(xs, ys, vals):
                ax.annotate(fmt.format(v), xy=(xv, yv), xytext=(5, 5),
                            textcoords="offset points", fontsize=8,
                            zorder=3, **text_kw)
            fig.colorbar(sc, ax=ax, label=label, shrink=0.85)
        if dropped:
            # rejected stars stay inspectable: × markers, muted values;
            # click one -> Reinsert star (Eduardo 2026-07-25)
            dxs = [(r.x - w / 2.0) * ps for r in dropped]
            dys = [(r.y - h / 2.0) * ps for r in dropped]
            dsc = ax.scatter(dxs, dys, marker="x", s=70, c="#b04a4a",
                             linewidths=1.6, zorder=2)
            dsc.set_picker(6)
            dsc._n2_pool = "dropped"
            for xv, yv, v in zip(dxs, dys, dvals):
                ax.annotate(fmt.format(v), xy=(xv, yv), xytext=(5, 5),
                            textcoords="offset points", fontsize=7,
                            color="#b04a4a", alpha=0.85, zorder=2,
                            **text_kw)
        sel = getattr(self, "_n2_sel_star", None)
        dsel = getattr(self, "_n2_sel_dropped", None)
        ring = None
        if sel is not None and 0 <= sel < len(field):
            ring = (xs[sel], ys[sel])
        elif dsel is not None and 0 <= dsel < len(dropped):
            ring = ((dropped[dsel].x - w / 2.0) * ps,
                    (dropped[dsel].y - h / 2.0) * ps)
        if ring is not None:
            ax.scatter([ring[0]], [ring[1]], s=240, facecolors="none",
                       edgecolors="#d62728", linewidths=2.0, zorder=5)
            ax.annotate("selected", xy=ring, xytext=(0, -16),
                        textcoords="offset points", ha="center", fontsize=8,
                        color="#d62728", zorder=5, **text_kw)
        # (colorbar drawn with the kept scatter above; brighter = better
        # in BOTH metrics, FWHM colormap reversed)
        st = getattr(self, "_n2_field_st", None)
        if st is not None and not fwhm_mode:
            ax.scatter([st.peak_dx_arcsec], [st.peak_dy_arcsec], s=340,
                       facecolors="none", edgecolors="#d4a017",
                       linewidths=1.8, zorder=4, label="peak")
            ax.annotate("peak", xy=(st.peak_dx_arcsec, st.peak_dy_arcsec),
                        xytext=(0, 14), textcoords="offset points",
                        ha="center", fontsize=8, color="#a97c00",
                        fontweight="bold", zorder=4, **text_kw)
            if st.grad_sr_per_arcmin is not None:
                # the arrow is a FIELD property: the downhill direction of
                # the best-fit performance plane, anchored at the peak
                # star ("top of the hill"); its length is fixed for
                # visibility -- the steepness is the labelled number
                ang = np.radians(st.grad_pa_deg)
                span = 0.18 * min(w, h) * ps
                tip = (st.peak_dx_arcsec + span * np.cos(ang),
                       st.peak_dy_arcsec + span * np.sin(ang))
                ax.annotate(
                    "", xytext=(st.peak_dx_arcsec, st.peak_dy_arcsec),
                    xy=tip,
                    arrowprops=dict(arrowstyle="-|>", color="#d4a017",
                                    lw=1.6), zorder=4)
                ax.annotate(
                    "downhill\n"
                    f"{st.grad_sr_per_arcmin:.2f} SR/\u2032",
                    xy=tip, xytext=(8 * np.cos(ang), 8 * np.sin(ang)),
                    textcoords="offset points",
                    ha="left" if np.cos(ang) >= 0 else "right",
                    va="top" if np.sin(ang) < 0 else "bottom",
                    fontsize=8, color="#a97c00", fontweight="bold",
                    zorder=4, **text_kw)
        ax.set_xlim(-half_x, half_x)
        ax.set_ylim(-half_y, half_y)
        ax.set_aspect("equal")
        ax.set_anchor("C")
        ax.grid(alpha=0.2)
        ax.set_xlabel("Δx from frame centre (arcsec, detector)")
        ax.set_ylabel("Δy (arcsec, detector)")
        rej = f" (+{len(dropped)} rejected ×)" if dropped else ""
        ax.set_title(
            f"Image {self._n2_imno} — {len(field)} star(s){rej} — "
            f"{self._nirc2_mode_text(p)} — {p.effwave_um:.3f} µm",
            fontsize=9)
        canvas.draw_idle()

    # ---- measured vs predicted ----------------------------------------------
    def _nirc2_compare(self, result):
        """Measured-vs-predicted comparison for a measured frame against the
        currently loaded prediction (self.res). The frame's header supplies
        the parameters: UTC timestamp -> nearest predicted sample (within
        the match tolerance), LSPROP -> which predicted series (LGS vs
        NGS), EFFWAVE -> Marechal conversion of the predicted Strehl from
        the run's science wavelength.

        Returns None when no prediction is loaded / the frame carries no
        timestamp, else a dict: s_conv, fwhm (predicted FWHM in mas,
        RE-EVALUATED at the FRAME's own wavelength via fwhm_srtool_mas (the
        convention THIS tab measures in -- see the note at that call) -- same
        band conversion s_conv already gets, so a J-band frame loaded under
        a K-band run gets a J-band predicted FWHM, not the run's stale K-band
        one -- may be None), delta (measured - predicted), and text (the
        full sentence for the log and tooltips); an unmatchable frame
        returns the dict with only text set.

        NGS s_conv is RE-EVALUATED at the frame's own RESOLVED guide-star
        magnitude (_nirc2_resolve_guide_star), not read off the precomputed
        self.res.ngs_bright series: that series is a single Gompertz curve
        computed ONCE, at whatever ngs_bright the spinbox held when Run was
        clicked -- for a batch touching many different NGS targets (each
        with its OWN real guide-star magnitude, now resolved correctly
        into the Guide-mag log column), that one precomputed series can
        only ever be right for ONE of them. Eduardo 2026-07-28: "you
        correctly update the guide mag in the image log table [but] the
        predicted value is still coming off the default settings and
        does not change." Falls back to the precomputed series (Marechal
        band-shifted) when no real magnitude is resolved (still ASSUMED)
        or ngs_strehl's inputs aren't available, so behavior for a
        properly-loaded single-target run is unchanged."""
        import datetime as dt
        p = result.params
        if self.res is None or self.prep is None or p.utc is None:
            return None
        t_hst = p.utc - dt.timedelta(hours=engine.HST_TO_UTC_HOURS)

        is_ngs = p.lgs is False
        if is_ngs:
            label, times = "NGS (bright)", self.res.times
            series, fwhm_series = (self.res.ngs_bright,
                                   getattr(self.res, "fwhm_ngs_bright", None))
        elif p.aoopsmod == 3:
            # THIS FRAME's own decoded AO mode says LTAO -- use the LTAO
            # series regardless of whether the currently-loaded RUN was
            # itself configured for tomography. self.res.sr_ltao/fwhm_ltao
            # are computed unconditionally by compute_timeline (same loop
            # as sr_single, not gated on prep.tomography_on), so they're
            # always there to compare against. Previously this gated on
            # the run's OWN tomography_on/telescope config, so a frame
            # that was actually taken in LTAO mode still got compared
            # against the single-LGS series whenever the loaded run
            # wasn't itself set up for LTAO (Eduardo 2026-07-28: "if the
            # mode is LTAO the estimator changes from single laser to
            # LTAO").
            label, times = "LTAO", self.res.p_times
            series, fwhm_series = (self.res.sr_ltao,
                                   getattr(self.res, "fwhm_ltao", None))
        else:
            label = "single-LGS" if p.lgs else "single-LGS (LSPROP unknown)"
            times = self.res.p_times
            series, fwhm_series = (self.res.sr_single,
                                   getattr(self.res, "fwhm_single", None))
        if times is None or len(times) == 0 or series is None:
            return None
        miss = {"s_conv": None, "fwhm": None, "delta": None}

        offs = np.array([abs((t - t_hst).total_seconds()) for t in times])
        i = int(offs.argmin())
        if offs[i] > engine.DEF_MATCH_TOL:
            return miss | {"text": (
                f"no predicted sample within "
                f"{engine.DEF_MATCH_TOL / 60:.0f} min of the frame time "
                f"({t_hst:%H:%M} HST) — is the right night loaded?")}
        s_pred = float(series[i])
        if not np.isfinite(s_pred) or not 0.0 < s_pred < 1.0:
            return miss | {"text": (f"predicted {label} S at "
                                    f"{times[i]:%H:%M} HST is undefined")}

        lam_pred_nm = float(self.prep.lam_nm)
        lam_frame_nm = p.effwave_um * 1e3
        s_conv = s_pred ** ((lam_pred_nm / lam_frame_nm) ** 2)
        real_mag_used = None
        if is_ngs:
            try:
                eps_los_ngs = float(self.res.col_dimm[i] * self.res.col_zf[i])
                _gname, real_mag, src = self._nirc2_resolve_guide_star(p)
                tel = getattr(self.args_cached, "telescope", None)
                if (real_mag is not None and "ASSUMED" not in src
                        and tel is not None):
                    s_real = engine.ngs_strehl(
                        eps_los_ngs, real_mag, tel, lam_frame_nm,
                        seeing_law=self.args_cached.ngs_seeing_law,
                        ngs_s0=self.args_cached.ngs_s0,
                        ngs_a=self.args_cached.ngs_a,
                        ngs_m0=self.args_cached.ngs_m0,
                        ngs_w=self.args_cached.ngs_w,
                        k1_quadcell=self.args_cached.k1_quadcell_penalty)
                    if np.isfinite(s_real) and 0.0 < s_real < 1.0:
                        s_conv = float(s_real)
                        real_mag_used = real_mag
            except Exception:
                pass
        real_offset_used = None
        lgs_terms_override = None
        if not is_ngs:
            # LGS/LTAO needs BOTH the TT-star magnitude AND its OFFSET from
            # the target (Eduardo 2026-07-28: "the estimator needs to know
            # the magnitude and the offset") -- reading the run's precomputed
            # sr_single/sr_ltao series has exactly the NGS problem, twice
            # over. The offset doesn't need the star's IDENTITY at all: the
            # TSS-vs-pointing-origin odometer's SEPARATION is direction-free
            # and valid on its own (ttstar.py), so it's available for
            # essentially every real frame, independent of whether the
            # magnitude tiers above found a starlist/catalogue match.
            try:
                eps_tot_los = float(self.res.p_dimm_in[i] * self.res.p_zf[i])
                eps_fa_los = float(self.res.col_mass[i] * self.res.p_zf[i])
                tel = getattr(self.args_cached, "telescope", None)
                mode = "ltao" if label == "LTAO" else "single"
                header = getattr(self, "_n2_header", None)
                off = self._nirc2_tt_offset_evidence(p, header)
                _gname, real_mag, src = self._nirc2_resolve_guide_star(p)
                mag_resolved = real_mag is not None and "ASSUMED" not in src
                if tel is not None and (off is not None or mag_resolved):
                    mag_to_use = real_mag if mag_resolved else self.args_cached.tt_mag
                    offset_to_use = (off["sep_arcsec"] if off is not None
                                    else self.args_cached.tt_offset)
                    bw_factor = (2.0 ** (5.0 / 6.0)
                                if self.args_cached.legacy_budget else
                                engine.ltao_bw_factor(
                                    self.args_cached.ltao_bw_floor_frac))
                    _bkw = dict(
                        tt_mag=mag_to_use, tt_offset=offset_to_use,
                        lgs_offset=self.args_cached.lgs_offset,
                        legacy=self.args_cached.legacy_budget,
                        bw_factor=bw_factor,
                        v_ground=self.args_cached.wind_ground,
                        v_free=self.args_cached.wind_free,
                        tt_sensor=getattr(self.args_cached, "_tt_sensor_base",
                                         "strap"),
                        ltao_tt_theta0_gain=getattr(
                            self.args_cached, "ltao_tt_theta0_gain", None))
                    cn2_kw = {}
                    if mode == "ltao":
                        col_cn2 = getattr(self.res, "col_cn2", None)
                        if col_cn2 is not None and len(col_cn2):
                            cn2_kw["cn2_bins"] = col_cn2[i]
                    s_real = engine.lgs_strehl(eps_tot_los, eps_fa_los, tel,
                                               mode, lam_frame_nm, **_bkw,
                                               **cn2_kw)
                    if np.isfinite(s_real) and 0.0 < s_real < 1.0:
                        s_conv = float(s_real)
                        real_mag_used = mag_to_use if mag_resolved else None
                        real_offset_used = (offset_to_use if off is not None
                                            else None)
                        _terms = engine.lgs_budget_terms(
                            eps_tot_los, eps_fa_los, tel, mode, **_bkw,
                            **cn2_kw)
                        lgs_terms_override = (float(_terms["tt"]),
                                              float(_terms["fit"]))
            except Exception:
                pass
        delta = result.strehl - s_conv
        fwhm = dfwhm = None
        if fwhm_series is not None and np.isfinite(fwhm_series[i]):
            # re-evaluate psf_fwhm_mas AT THE FRAME'S OWN WAVELENGTH
            # (lam_frame_nm) instead of indexing fwhm_series (computed once
            # at the run's lam_pred_nm) -- otherwise a frame in a different
            # band than the loaded run gets a predicted FWHM that silently
            # disagrees with the band s_conv already correctly reports.
            if is_ngs:
                eps_los = float(self.res.col_dimm[i] * self.res.col_zf[i])
                tt_arr, fit_arr = self.res.tt_ngs_bright, self.res.fit_ngs
                tt_nm = float(tt_arr[i]) if tt_arr is not None else 0.0
                fit_nm = float(fit_arr[i]) if fit_arr is not None else None
            elif lgs_terms_override is not None:
                # keep the FWHM's tt/fit terms consistent with the
                # resolved-mag/offset s_conv just computed above, instead of
                # mixing it with the run's STALE col_terms (computed at the
                # spinbox defaults) -- same tt/fit source lgs_budget_terms
                # already returned when s_conv was recomputed, not a second
                # formula
                eps_los = float(self.res.p_dimm_in[i] * self.res.p_zf[i])
                tt_nm, fit_nm = lgs_terms_override
            else:
                eps_los = float(self.res.p_dimm_in[i] * self.res.p_zf[i])
                tt_arr = fit_arr = None
                col_terms = self.res.col_terms
                if col_terms is not None and len(col_terms):
                    tt_arr, fit_arr = col_terms[:, 7], col_terms[:, 0]
                tt_nm = float(tt_arr[i]) if tt_arr is not None else 0.0
                fit_nm = float(fit_arr[i]) if fit_arr is not None else None
            nact = engine.DM_ACTUATORS_ACROSS[self.args_cached.telescope]
            # CONVENTION (Eduardo 2026-08-07): predict with fwhm_srtool_mas,
            # not psf_fwhm_mas. This box is compared directly against the
            # MEASURED FWHM one line above it, so it must be the convention
            # THIS TAB measures in -- the tab's own find_fwhm.pro port, on
            # the model PSF rendered at THIS FRAME's plate scale, annulus
            # sky and all (see psf.fwhm_srtool_mas). The half-max
            # convention read ~1 mas low against 60 isolated-standard NIRC2
            # frames; this one reads ~0.4 mas low. Both are far smaller
            # than the several-mas predicted-vs-delivered gaps, which come
            # from the STREHL prediction, not from the FWHM convention --
            # this fixes the part that was ours to fix.
            fwhm_new = engine.fwhm_srtool_mas(
                s_conv, eps_los, lam_frame_nm, tt_nm, fit_nm=fit_nm,
                n_act=nact, plate_scale_mas=p.plate_scale_mas)
            if np.isfinite(fwhm_new):
                fwhm = float(fwhm_new)
                dfwhm = result.fwhm_mas - fwhm
        tel = getattr(self.args_cached, "telescope", "?")
        inst_tel = "K1" if p.camname == "osiris" else "K2"
        inst_name = "OSIRIS" if p.camname == "osiris" else "NIRC2"
        warn = (f"(prediction is {tel} — {inst_name} is on {inst_tel}!)  "
                if tel != inst_tel else "")
        if real_mag_used is not None or real_offset_used is not None:
            bits = []
            if real_mag_used is not None:
                bits.append(f"guide mag {real_mag_used:.2f}")
            if real_offset_used is not None:
                bits.append(f"TT offset {real_offset_used:.2f}\"")
            pred_text = (f"predicted {label} S re-evaluated at this "
                        f"frame's real {' and '.join(bits)} "
                        f"(not the loaded run's default): {s_conv:.3f} @ "
                        f"{lam_frame_nm:.0f} nm")
        else:
            pred_text = (f"predicted {label} S at {times[i]:%H:%M} HST: "
                        f"{s_pred:.3f} @ {lam_pred_nm:.0f} nm -> "
                        f"{s_conv:.3f} @ {lam_frame_nm:.0f} nm")
        return {
            "s_conv": s_conv, "fwhm": fwhm, "delta": delta, "dfwhm": dfwhm,
            "label": label, "when": f"{times[i]:%H:%M}",
            "text": (f"{warn}{pred_text};  measured - predicted = "
                     f"{delta:+.3f}")}

    def _nirc2_show_compare(self, cmp_res):
        """Fill the PREDICTED SR / FWHM / dSR boxes (empty when there is
        nothing to compare against); the full sentence goes to the boxes'
        tooltips and the log."""
        boxes = (self.n2_pred_sr, self.n2_pred_fwhm, self.n2_dsr,
                 self.n2_dfwhm)
        text = (cmp_res or {}).get("text", "")
        for b in boxes:
            b.setToolTip(text or "run the estimator to get a prediction")
        if cmp_res is None or cmp_res["s_conv"] is None:
            for b in boxes:
                b.setText("")
            return
        self.n2_pred_sr.setText(f"{cmp_res['s_conv']:.3f}")
        self.n2_pred_fwhm.setText(
            "—" if cmp_res["fwhm"] is None else f"{cmp_res['fwhm']:.2f}")
        self.n2_dsr.setText(f"{cmp_res['delta']:+.3f}")
        self.n2_dfwhm.setText(
            "—" if cmp_res.get("dfwhm") is None
            else f"{cmp_res['dfwhm']:+.2f}")

    # ---- Match SR tool: drive a conditions selector to the frame time -------
    def _nirc2_match_tool(self, which):
        """'Match SR tool' (summary-stats Period / field-map Conditions):
        set the selector to 'specific time' at the last NIRC2-measured
        frame's timestamp, entered in the current display zone (UTC mode
        aware), AND set the science wavelength override to the frame's
        EFFWAVE — so the stats/map quote the atmosphere of that exact
        frame at the same wavelength the SR tool measured (otherwise the
        summary LGS SR and the PREDICTED box differ by the Marechal
        band conversion). Normal signal wiring does the refreshes."""
        t = getattr(self, "_n2_frame_hst", None)
        if t is None:
            self.status.setText(
                "Measure a frame in the Measured SR tab first — no "
                "SR-tool timestamp to match.")
            return
        h_disp = (t.hour + 10) % 24 if self._utc() else t.hour
        qt = QtCore.QTime(h_disp, t.minute)
        cond, tw = ((self.stats_cond, self.stats_time) if which == "stats"
                    else (self.fm_cond, self.fm_time))
        cond.setCurrentText("specific time")
        tw.setTime(qt)
        if self._n2_params is not None:
            self.wl_enable.setChecked(True)
            self.wl_nm.setValue(self._n2_params.effwave_um * 1000.0)

    # ---- OBJECT -> estimator target -----------------------------------------
    def _nirc2_target_index(self, name):
        """Index of `name` in the loaded target list (starlist name rules:
        engine.same_star_name), or None."""
        if not name:
            return None
        for i, t in enumerate(self._targets):
            if engine.same_star_name(t.get("name", ""), name):
                return i
        return None

    def _on_nirc2_set_target(self):
        name = self.n2_object.text().strip()
        if not name:
            return
        idx = self._nirc2_target_index(name)
        if idx is not None:
            # setCurrentIndex alone is a no-op when already selected --
            # invoke the loader explicitly so coords/GS always apply
            self.target_select.setCurrentIndex(idx)
            self._on_target_selected(idx)
            self.n2_log.appendPlainText(
                f"target set from list: {self._targets[idx]['name']}")
        else:
            self.tname_edit.setText(name)
            ra, dec = self.n2_ra.text(), self.n2_dec.text()
            if ra and dec:
                self.ra_edit.setText(ra)
                self.dec_edit.setText(dec)
                self.n2_log.appendPlainText(
                    f"'{name}' not in the loaded target list — name and "
                    "header RA/Dec filled on the Target tab")
            else:
                self.n2_log.appendPlainText(
                    f"'{name}' not in the loaded target list — name filled "
                    "on the Target tab; resolve/enter coordinates there")

    # ---- display ------------------------------------------------------------
    def _nirc2_scaled(self, img):
        """Display scaling for the main view and the pick-zoom magnifier:
        (array, vmin, vmax) per the Stretch combo. Display-only — the
        measurement never sees this. 'IDL ±5σ' is the summit tool's fixed
        linear window; the others clip black at the 0.5th percentile and
        set white at the White-point percentile, with nonlinear transfer
        curves for hunting faint objects."""
        mode = self.n2_stretch.currentText()
        if mode == "IDL ±5σ":
            av, sd = float(img.mean()), float(img.std())
            return img, av - 5 * sd, av + 5 * sd
        lo = float(np.percentile(img, 0.5))
        hi = float(np.percentile(img, self.n2_white.value()))
        if hi <= lo:
            hi = lo + 1.0
        if mode == "linear %":
            return img, lo, hi
        pos = np.clip(img - lo, 0.0, None)
        if mode == "sqrt":
            return np.sqrt(pos), 0.0, float(np.sqrt(hi - lo))
        if mode == "log":
            return (np.log10(pos + 1.0), 0.0,
                    float(np.log10(hi - lo + 1.0)))
        knee = max((hi - lo) / 30.0, 1e-9)      # asinh soft knee
        return (np.arcsinh(pos / knee), 0.0,
                float(np.arcsinh((hi - lo) / knee)))

    def _on_nirc2_tt_star(self):
        """Estimate the TT star's position from the loaded frame's AO
        headers (TSS-vs-pointing-origin odometer, rotation-free
        separation), then its identity/magnitude by a catalogue RING
        match at that separation (direction is not derivable from one
        frame -- see engine.ttstar)."""
        hdr = getattr(self, "_n2_header", None)
        off = engine.tt_star_offset(hdr) if hdr is not None else None
        if off is None:
            self.n2_ttstar_out.setText(
                "TT star: no TSS/pointing-origin keywords in this frame")
            return
        if off["on_axis"]:
            txt = (f"TT star: ON-AXIS — TSS sits {off['sep_arcsec']:.2f}″ "
                   f"from the '{off['po_name']}' pointing origin")
        else:
            txt = (f"TT star: {off['sep_arcsec']:.1f}″ from the field "
                   f"centre (bench PA {off['bench_pa_deg']:+.0f}°; sky "
                   "direction not header-derivable — ring match)")
        self.n2_ttstar_out.setText(txt + " · querying catalogue…")
        self.n2_log.appendPlainText(
            f"TT star odometer: TSS−PO = ({off['dx_mm']:+.3f}, "
            f"{off['dy_mm']:+.3f}) mm → {off['sep_arcsec']:.2f}″")
        try:
            ra, dec = float(hdr["RA"]), float(hdr["DEC"])
        except (KeyError, TypeError, ValueError):
            self.n2_ttstar_out.setText(
                txt + " · no RA/DEC in header — catalogue lookup skipped")
            return
        if getattr(self, "_n2_tt_worker", None) is not None:
            return                              # one query in flight
        self._n2_tt_ring = (ra, dec, off["sep_arcsec"], txt)
        from ..workers import CatalogFetchWorker
        cat = (self.fm_catalog.currentText()
               if hasattr(self, "fm_catalog") else "GSC 2.4")
        w = CatalogFetchWorker(
            cat, ra, dec, off["sep_arcsec"] + 2.0 * engine.RING_TOL_ARCSEC,
            parent=self)
        w.done.connect(self._on_nirc2_tt_star_catalog)
        self._n2_tt_worker = w
        w.start()

    def _on_nirc2_tt_star_catalog(self, cat, stars, err):
        self._n2_tt_worker = None
        ra, dec, sep, txt = self._n2_tt_ring
        if err:
            self.n2_ttstar_out.setText(f"{txt} · {cat}: {err}")
            return
        cands = engine.tt_ring_match(stars, ra, dec, sep)
        if not cands:
            self.n2_ttstar_out.setText(
                f"{txt} · no {cat} star within ±"
                f"{engine.RING_TOL_ARCSEC:g}″ of that separation "
                f"({len(stars)} in field)")
            return
        best = cands[0]
        band, mag = engine.best_mag(best["star"])
        magtxt = f"{band}={mag:.1f}" if band else "no magnitude"
        # remember it: this is the most direct evidence of what the REAL
        # TT star was for this frame (delivered odometer separation +
        # catalogue photometry), so the image log's guide-magnitude
        # resolution prefers it over anything the estimator was merely
        # configured with (_nirc2_resolve_guide_star)
        if band and mag is not None:
            self._n2_tt_star_resolved = {
                "mag": float(mag), "band": band,
                "id": best["star"].get("id", "?"), "catalog": cat,
                "imno": self._n2_imno}
        self.n2_ttstar_out.setText(
            f"{txt} · likely {cat} {best['star']['id']}: {magtxt} "
            f"at {best['sep_arcsec']:.1f}″ (Δring "
            f"{best['dsep_arcsec']:+.1f}″; {len(cands)} candidate(s))")
        for c in cands[:3]:
            b, m = engine.best_mag(c["star"])
            self.n2_log.appendPlainText(
                f"  TT-star candidate {c['star']['id']}: "
                f"{(b + '=' + format(m, '.2f')) if b else 'no mag'} "
                f"at {c['sep_arcsec']:.2f}″ (Δring "
                f"{c['dsep_arcsec']:+.2f}″)")

    def _on_nirc2_stretch(self, *_):
        self.n2_white.setEnabled(self.n2_stretch.currentText() != "IDL ±5σ")
        if self._n2_image is not None and self._n2_last_draw is not None:
            self._nirc2_draw_main()

    def _nirc2_draw_main(self):
        """Redraw the main view from the stored frame + last draw state
        (title text, measured result for the aperture circles) with the
        current stretch — called by display, the pick view, and any
        stretch-control change."""
        title, result = self._n2_last_draw
        img = self._n2_image
        self.n2_fig.clear()
        ax = self.n2_fig.add_subplot(111)
        disp, vmin, vmax = self._nirc2_scaled(img)
        ax.imshow(disp, cmap="gray", origin="lower", vmin=vmin, vmax=vmax,
                  interpolation="nearest")
        if result is not None:
            ps = result.params.plate_scale_mas
            th = np.linspace(0, 2 * np.pi, 121)
            used = result.photrad_used_arcsec or self.n2_photrad.value()
            for r_as, style in ((used, "-"),
                                (self.n2_bgin.value(), "--"),
                                (self.n2_bgout.value(), "--")):
                r_px = r_as * 1000.0 / ps
                ax.plot(result.x + r_px * np.cos(th),
                        result.y + r_px * np.sin(th),
                        style, color="white", lw=0.8)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title, fontsize=9)
        self.n2_canvas.draw_idle()

    def _nirc2_display(self, result):
        img = self._n2_image
        if not result.ok:
            self.n2_log.appendPlainText(
                f"Image {self._n2_imno}: {result.error}")
            return

        ps = result.params.plate_scale_mas
        # main view: drawn via _nirc2_draw_main so stretch changes can
        # re-render this exact state later
        self._n2_last_draw = (
            f"Image {self._n2_imno} — "
            f"{self._nirc2_mode_text(result.params)}", result)
        self._nirc2_draw_main()
        # warnings live UNDER the cutouts (Eduardo 2026-07-23), symptom
        # first, then cause: out-of-(0,1] SR is impossible, and annulus
        # contamination is the usual culprit
        tags = []
        if result.saturated:
            tags.append(f"SATURATED (>{result.params.max_counts:.0f} "
                        "counts/coadd)")
        if result.unphysical:
            tags.append("UNPHYSICAL SR")
        if result.crowded:
            tags.append(f"CROWDED (~{100 * result.crowding:.0f}% "
                        "annulus contamination)")
        if result.edge:
            tags.append(f"EDGE ({100 * result.edge_clip:.0f}% of the "
                        "photometry footprint off-array; auto-radius "
                        "disabled)")
        if result.cleaned and result.strehl > engine.PSF_FIT_SR_VALIDATED_MAX:
            # D25: cleaning ran and the result is above the validated
            # envelope -- a confidence warning, not a correctness one (the
            # measurement is not refused). Full sentence goes to the log
            # (_nirc2_log_psf_clean); this is the at-a-glance tag.
            tags.append(f"PSF-CLEAN ABOVE VALIDATED SR (>{engine.PSF_FIT_SR_VALIDATED_MAX:.2f})")
        set_cue(self.n2_warn,
                "err" if (result.saturated or result.unphysical) else "warn")
        self.n2_warn.setText(" · ".join(tags))
        self.n2_canvas.draw_idle()

        # the tv2/tv3 pair: cube-root cutouts of the DL PSF and the star
        radius = int(np.ceil(self.n2_peakrad.value() * 1000.0 / ps))
        ctr = self._n2_dl.shape[0] // 2
        dlcut = self._n2_dl[ctr - 2 * radius:ctr + 2 * radius,
                            ctr - 2 * radius:ctr + 2 * radius]
        xi, yi = int(result.x), int(result.y)
        starcut = img[max(yi - 2 * radius + 1, 0):yi + 2 * radius + 1,
                      max(xi - 2 * radius + 1, 0):xi + 2 * radius + 1]
        for fig, cut in ((self.n2_fig_dl, dlcut), (self.n2_fig_star, starcut)):
            fig.clear()
            axz = fig.add_subplot(111)
            axz.imshow(np.clip(cut, 0, None) ** (1.0 / 3.0), cmap="gray",
                       origin="lower", interpolation="bicubic")
            axz.set_xticks([]); axz.set_yticks([])
        self.n2_cap_star.setText("MEASURED STAR")
        self.n2_canvas_dl.draw_idle()
        self.n2_canvas_star.draw_idle()

        self.n2_strehl_out.setText(f"{result.strehl:.3f}")
        self.n2_fwhm_out.setText(f"{result.fwhm_mas:.2f}")
        self.n2_wfe_out.setText(f"{result.wfe_nm:.1f}")

        # two aligned lines per measurement, echoing the readout rows
        # (Eduardo 2026-07-23); the full comparison sentence lives in the
        # predicted boxes' tooltip
        cmp_res = self._nirc2_compare(result)
        self._nirc2_show_compare(cmp_res)
        self.n2_log.appendPlainText(
            f"Image {self._n2_imno}  SR {result.strehl:.3f} "
            f"±{result.sr_err:.3f}  "
            f"FWHM {result.fwhm_mas:7.2f} mas  WFE {result.wfe_nm:6.1f} nm  "
            f"pos {result.x:6.1f} {result.y:6.1f}")
        if cmp_res is not None:
            if cmp_res["s_conv"] is None:
                self.n2_log.appendPlainText(f"predicted: {cmp_res['text']}")
            else:
                fp = cmp_res["fwhm"]
                fp_txt = f"{fp:7.2f}" if fp is not None else "      —"
                df_txt = (f"{result.fwhm_mas - fp:+7.2f}"
                          if fp is not None else "      —")
                self.n2_log.appendPlainText(
                    f"predicted   SR {cmp_res['s_conv']:.3f}  "
                    f"FWHM {fp_txt} mas  ΔSR {cmp_res['delta']:+.3f}  "
                    f"ΔFWHM {df_txt}")
        self._nirc2_add_csv_row(result, cmp_res)
        self._nirc2_warn_guide_mag_mismatch(result.params)

        # identity readouts + target auto-load: a loaded target list wins
        # (Eduardo 2026-07-23) -- selecting the entry also restores its
        # guide star; otherwise name+RA/Dec are shown for Set-as-target
        name = result.params.object_name
        self._nirc2_show_identity(result.params)
        idx = self._nirc2_target_index(name)
        if idx is not None and not engine.same_star_name(
                self.tname_edit.text(), name):
            self.target_select.setCurrentIndex(idx)
            self._on_target_selected(idx)
            self.n2_log.appendPlainText(
                f"target set from list: {self._targets[idx]['name']}")

    def _nirc2_seeing_at(self, t_hst):
        """Nearest-sample DIMM/MASS seeing (arcsec, 500 nm, zenith) to
        `t_hst`, each on its OWN timebase (times/p_times) -- independent of
        which predicted series (NGS/LGS/LTAO) a Strehl/FWHM comparison used,
        since DIMM/MASS are telescope- and mode-independent (same values
        summary_stats.py's DIMM/MASS rows read). (None, None) with no
        prediction loaded or nothing within the match tolerance on either
        timebase."""
        if self.res is None:
            return None, None

        def nearest(times, col):
            if times is None or len(times) == 0 or col is None or not len(col):
                return None
            offs = np.array([abs((t - t_hst).total_seconds()) for t in times])
            i = int(offs.argmin())
            if offs[i] > engine.DEF_MATCH_TOL:
                return None
            v = float(col[i])
            return v if np.isfinite(v) else None

        return (nearest(self.res.times, self.res.col_dimm),
                nearest(self.res.p_times, self.res.col_mass))

    def _nirc2_csv_row(self, result, cmp_res):
        """One structured row for the Measured-SR CSV log/Image-log table,
        built from `result`/`cmp_res` (already computed by the caller) plus
        Az/El/filter/airmass straight off Nirc2FrameParams (nirc2.py's
        opt_header_float additions) and DIMM/MASS via _nirc2_seeing_at.
        guide_star/guide_mag/guide_mag_src come from
        _nirc2_resolve_guide_star, which prefers real per-frame evidence
        (the TSS odometer, automatically, no manual click needed) over
        the TT-magnitude spinbox and always records WHICH it used, so an
        estimator default can never be silently logged as a measurement.
        ao_mode is the AOOPSMOD header code decoded to NGS/single-LGS/
        LTAO/GLAO (nirc2.decode_ao_ops_mode). pixel_x/y (the detector
        pixel the star was measured at) also double as the duplicate-
        measurement guard's position key (_nirc2_find_duplicate_rows) --
        useful in its own right for a FIELD measurement, where several
        rows share one frame number and RA/Dec isn't per-star. Never
        raises -- every field defaults to None (rendered '—' in the
        table / blank in the CSV) rather than losing the whole row over
        one missing header keyword."""
        p = result.params
        row = {k: None for k, _label in NIRC2_CSV_COLUMNS}
        row["time_utc"] = p.utc.isoformat(sep=" ") if p.utc else None
        row["frame_number"] = self._n2_imno
        row["target_name"] = p.object_name
        row["ao_mode"] = engine.decode_ao_ops_mode(p.aoopsmod)
        gname, gmag, gsrc = self._nirc2_resolve_guide_star(p)
        row["guide_star"] = gname
        row["guide_mag"] = gmag
        row["guide_mag_src"] = gsrc
        row["ra"] = p.ra
        row["dec"] = p.dec
        row["pixel_x"] = float(result.x)
        row["pixel_y"] = float(result.y)
        row["filter"] = p.filter_name
        row["az_deg"] = p.az_deg
        row["el_deg"] = p.el_deg
        row["airmass"] = p.airmass
        row["measured_sr"] = result.strehl
        # image_strehl.radial_profile_fwhm returns -1.0 px as a documented
        # FAILURE sentinel ("profile peak off-center or no half-max
        # crossing"), which measure_strehl then scales by the plate scale
        # -- so a failed fit arrives here as a negative FWHM (-9.95 mas on
        # OSIRIS). Log it as "no value" rather than a number: a sentinel
        # written into the CSV would silently poison any FWHM statistic
        # computed from it downstream. (The field-map path already applies
        # the same `fwhm_mas > 0` test when accepting stars.)
        row["measured_fwhm"] = (result.fwhm_mas if result.fwhm_mas > 0.0
                                else None)
        row["lbwfs_fwhm"] = p.lbwfs_fwhm
        if cmp_res is not None:
            row["predicted_sr"] = cmp_res.get("s_conv")
            row["delta_sr"] = cmp_res.get("delta")
            row["predicted_fwhm"] = cmp_res.get("fwhm")
            # dfwhm is (measured - predicted), so it inherits the sentinel
            # whenever the measured FWHM fit failed -- drop it too rather
            # than logging a difference against a non-measurement
            row["delta_fwhm"] = (cmp_res.get("dfwhm")
                                 if row["measured_fwhm"] is not None else None)
        if p.utc is not None:
            import datetime as _dt
            t_hst = p.utc - _dt.timedelta(hours=engine.HST_TO_UTC_HOURS)
            try:
                row["dimm_seeing"], row["mass_seeing"] = \
                    self._nirc2_seeing_at(t_hst)
            except Exception:
                pass
        return row

    @staticmethod
    def _nirc2_tt_offset_evidence(p, header):
        """The best available TT-star offset-from-field-centre evidence
        for this frame: TRICK's own ROI position (engine.trick_roi_offset)
        when TRICK is confirmed active for this frame (p.trick_active),
        else the TSS-vs-pointing-origin odometer (engine.tt_star_offset),
        which tracks the STRAP stage specifically and has no reason to
        mean anything while STRAP isn't the sensor in the loop. Returns
        the same dict shape either way (dx_*, dy_*, sep_arcsec, on_axis)
        -- callers only need sep_arcsec/on_axis, which both functions
        provide identically -- or None with no header/keywords at all.
        See ttstar.py's trick_roi_sky_offset for the one still-
        unconfirmed piece (compass direction) if a TRICK off-axis match
        ever looks wrong."""
        if not header:
            return None
        if p.trick_active is True:
            try:
                off = engine.trick_roi_offset(header)
            except Exception:
                off = None
            if off is not None:
                return off
        try:
            return engine.tt_star_offset(header)
        except Exception:
            return None

    @staticmethod
    def _nirc2_gs_cache_key(name):
        """Normalized key for self._n2_auto_gs_cache -- same fold as
        starlist.same_star_name (case/whitespace/underscore-insensitive),
        so a target seen with slightly different header whitespace still
        hits the cache."""
        return " ".join((name or "").replace("_", " ").lower().split())

    @staticmethod
    def _nirc2_best_catalog_match(stars, ra, dec):
        """The single catalogue star closest to (ra, dec) -- the query's
        OWN center, i.e. the target's reference position at the moment it
        was queried -- or None if the query returned nothing. Picking this
        ONCE, at fetch time, is what makes the auto-fetch cache immune to
        a routine small telescope offset: nothing re-matches it against a
        later frame's own (possibly dithered) RA/Dec (see
        _nirc2_catalog_selfmag)."""
        if not stars:
            return None
        from astropy.coordinates import SkyCoord
        import astropy.units as u
        try:
            center = engine.parse_radec(ra, dec)
        except Exception:
            return stars[0]
        best, best_sep = None, None
        for s in stars:
            sep = center.separation(
                SkyCoord(s["ra"] * u.deg, s["dec"] * u.deg)).arcsec
            if best_sep is None or sep < best_sep:
                best, best_sep = s, sep
        return best

    def _nirc2_catalog_selfmag(self, name, p, band):
        """Best-effort (mag, source_label) for `name`, checked in order:
          - self._n2_auto_gs_cache: this batch's per-target auto-fetch
            (_nirc2_prefetch_guide_stars) already resolved WHICH catalogue
            star is this target, once, at fetch time
            (_nirc2_best_catalog_match) -- a cache hit is trusted
            directly, with NO further per-frame position re-check. That
            re-check used to compare the catalogue star against THIS
            FRAME's own RA/Dec, which broke the moment a routine small
            telescope offset moved the header RA/Dec by more than the
            on-axis tolerance -- NGS has no "off-axis" case at all (the
            target IS always the guide star), so there was never a real
            identity ambiguity to resolve with a position gate here
            (Eduardo 2026-07-28: "I move by 1", the guide star gets back
            to the default" -- examples/failed_gs_selection shows RA/Dec
            drifting ~1.5" frame to frame while Pixel X/Y jump ~140 px:
            an intentional dither, not a different star);
          - self._catalog_stars: the Field-map's manually-queried,
            potentially WIDE-field cache that can hold many unrelated
            stars, so identity genuinely still needs a position gate here
            -- matched against the target's own SAVED (fixed) RA/Dec when
            one exists, so a routine dither doesn't defeat this source
            either; falls back to this frame's RA/Dec only when the
            target has no saved entry yet to anchor to.
        (None, None) if neither source has anything usable."""
        key = self._nirc2_gs_cache_key(name)
        auto_cache = getattr(self, "_n2_auto_gs_cache", None) or {}
        star = auto_cache.get(key)
        if star is not None:
            mag, kind, _lab = engine.estimate_sensing_mag(star["mags"], band)
            if mag is not None:
                return mag, f"auto-fetched “{star['id']}”, {band}, {kind}"

        if getattr(self, "_catalog_stars", None):
            ref_ra, ref_dec = p.ra, p.dec
            idx = self._nirc2_target_index(name)
            if idx is not None:
                t = self._targets[idx]
                ref_ra = t.get("ra") or p.ra
                ref_dec = t.get("dec") or p.dec
            try:
                from astropy.coordinates import SkyCoord
                import astropy.units as u
                tgt = engine.parse_radec(ref_ra, ref_dec)
                best, best_sep = None, None
                for s in self._catalog_stars:
                    sep = tgt.separation(
                        SkyCoord(s["ra"] * u.deg, s["dec"] * u.deg)).arcsec
                    if best_sep is None or sep < best_sep:
                        best, best_sep = s, sep
                if (best is not None
                        and best_sep <= engine.TT_ONAXIS_MAX_ARCSEC):
                    mag, kind, _lab = engine.estimate_sensing_mag(
                        best["mags"], band)
                    if mag is not None:
                        return mag, (f"{self._catalog_name} "
                                    f"“{best['id']}”, {band}, {kind}")
            except Exception:
                pass
        return None, None

    def _nirc2_pending_seq(self, path, files):
        """The exact (label, path) list Nirc2MeasureWorker is about to
        measure, built the same way it does internally (numbered n#### or
        an explicit files=), so the pre-flight guide-star scan looks at
        precisely the files the run will touch."""
        import os
        if files is not None:
            return list(files)
        im1, nim = self.n2_im1.value(), self.n2_nim.value()
        return [(str(no), os.path.join(path, f"n{no:04d}.fits"))
               for no in range(im1, im1 + nim)]

    def _nirc2_prefetch_guide_stars(self, seq, on_done):
        """Before a batch run touches any of these files, resolve every
        DISTINCT target's guide-star magnitude automatically, without you
        loading a starlist or catalogue by hand for each one (Eduardo
        2026-07-28: "I want to batch process a whole night ... I dont
        want to have to set the target or load the catalog with every
        change in target"). Starlist-first: read just the headers (fast,
        no pixel data) to find the distinct targets, skip any already
        covered by the loaded starlist or a previous auto-fetch this
        session, and auto-query the Field-map's selected catalogue ONCE
        per remaining target -- never once per frame, and never for a
        target the starlist already names. Sequential, not parallel, so
        as not to hammer Vizier with a burst of simultaneous queries.
        Calls on_done() once the queue (possibly empty) is drained."""
        from astropy.io import fits
        targets = {}   # cache key -> (name, ra, dec)
        for _label, fpath in seq:
            try:
                header = fits.getheader(fpath)
            except Exception:
                continue     # the real run will report this properly
            try:
                inst = engine.detect_instrument(header)
                p = (engine.osiris_frame_params(header) if inst == "osiris"
                    else engine.nirc2_frame_params(header))
            except Exception:
                continue
            name = (p.object_name or "").strip()
            if not name:
                continue
            key = self._nirc2_gs_cache_key(name)
            if key in targets:
                continue
            ra, dec = p.ra, p.dec
            if p.trick_active is True:
                # TRICK's own ROI can put the guide star OFF the target's
                # own position -- query where the star actually IS, not
                # the bare target position, or an off-axis TRICK target
                # would never resolve at all. See trick_roi_sky_offset's
                # orientation caveat (ttstar.py) if a TRICK off-axis
                # match ever looks wrong -- that's the one unconfirmed
                # piece this depends on.
                try:
                    troi = engine.trick_roi_offset(header)
                    if troi is not None and not troi["on_axis"]:
                        sky_off = engine.trick_roi_sky_offset(header)
                        if sky_off is not None:
                            import astropy.units as u
                            c0 = engine.parse_radec(p.ra, p.dec)
                            c1 = c0.spherical_offsets_by(
                                sky_off[0] * u.arcsec, sky_off[1] * u.arcsec)
                            ra = c1.ra.to_string(unit=u.hourangle, sep=":",
                                                 precision=4)
                            dec = c1.dec.to_string(unit=u.deg, sep=":",
                                                  precision=3, alwayssign=True)
                except Exception:
                    ra, dec = p.ra, p.dec
            targets[key] = (name, ra, dec)

        entries = getattr(self, "_starlist_entries", None) or []
        auto_cache = self._n2_auto_gs_cache
        todo = [(key, name, ra, dec)
               for key, (name, ra, dec) in targets.items()
               if key not in auto_cache
               and not any(engine.same_star_name(e["name"], name)
                          for e in entries)]
        if not todo:
            on_done()
            return

        self.n2_log.appendPlainText(
            f"resolving guide stars for {len(todo)} target(s) not in the "
            f"loaded starlist (one catalogue query each, cached)…")
        catalog = self.fm_catalog.currentText()

        def fetch_next():
            if not todo:
                self._n2_prefetch_worker = None
                on_done()
                return
            key, name, ra, dec = todo.pop(0)
            try:
                c = engine.parse_radec(ra, dec)
            except Exception:
                auto_cache[key] = None
                fetch_next()
                return
            worker = CatalogFetchWorker(
                catalog, float(c.ra.deg), float(c.dec.deg),
                self._N2_AUTO_GS_RADIUS_ARCSEC, self)

            def _done(_cat_name, stars, err, key=key, name=name, ra=ra, dec=dec):
                auto_cache[key] = (None if err else
                                  self._nirc2_best_catalog_match(stars, ra, dec))
                if err:
                    self.n2_log.appendPlainText(
                        f"! guide-star catalogue query failed for "
                        f"{name!r}: {err} -- its frames will show ASSUMED")
                fetch_next()

            worker.done.connect(_done)
            self._n2_prefetch_worker = worker
            worker.start()

        fetch_next()

    # query radius for the per-target auto-fetch: small and self-star-
    # focused (find just THIS target's own catalogue counterpart), not
    # the Field-map's wide field-of-regard radius which is meant to turn
    # up OTHER candidate stars nearby
    _N2_AUTO_GS_RADIUS_ARCSEC = 5.0

    def _nirc2_resolve_guide_star(self, p):
        """(name, magnitude, source) of the star that ACTUALLY sensed
        tip-tilt for this frame -- resolved from real evidence wherever
        possible, rather than reporting whatever the TT-magnitude spinbox
        happens to hold. name is None where a tier resolves only a
        magnitude with no identity distinct from the target's own (the
        spinbox fallback).

        The spinbox is only meaningful when the user has actually loaded
        this target into the estimator; on a frame measured from disk it is
        usually still the default, which silently mislabels the log AND
        means the predicted SR/FWHM being compared against was computed at
        the wrong TT magnitude (Eduardo 2026-07-28). Order, most direct
        evidence first:

          0. this frame's own TT-star-offset evidence
             (_nirc2_tt_offset_evidence: TRICK's own ROI position when
             TRICK is confirmed active for this frame, else the
             TSS-vs-pointing-origin odometer (AOTSX/AOTSY vs POXPOS/
             POYPOS -- engine.tt_star_offset) which tracks the STRAP
             stage specifically -- pure header math, no network,
             computed automatically for EVERY frame -- no manual "TT
             star" click needed, Eduardo 2026-07-28's "smarter way to
             find out what the guide star was"): on-axis means the TT
             star IS the science target; off-axis, the separation is
             cross-matched (engine.tt_ring_match -- the SAME ring-match
             the "TT star" button uses) against the LOADED STARLIST's
             own entries instead of a live catalogue query, so it needs
             no network round-trip and still names the star, not just a
             magnitude;
          1. a manual "TT star" catalogue ring match already run for this
             exact frame (self._n2_tt_star_resolved) -- kept as a
             fallback for when no starlist is loaded (tier 0's off-axis
             case needs one to name a star; this needs only the network);
          2. a starlist TT-star explicitly linked (target=) to this OBJECT;
          3. the target-list entry's stored tt_mag for this OBJECT (set when
             it was picked from a starlist / the field-map catalogue);
          4. NGS frames (LSPROP=no -> no laser): the AO guides on the
             science target itself, so the target's OWN magnitude applies
             -- the same rule the starlist picker already encodes for NGS;
          5. the spinbox, explicitly labelled ASSUMED so a default is never
             mistaken for a measurement.

        Note the headers carry STRAP flux (STAPDQMN, "STRAP Quad Mean APD
        counts", non-zero on laser frames) but NO counts-to-magnitude
        calibration exists anywhere in this repo, so it is deliberately
        NOT converted into a magnitude here -- that would be an invented
        zero point. See the log line in _nirc2_csv_row's caller."""
        # The sensing BAND depends on which loop actually senses tip-tilt,
        # NOT on whatever the (LGS-only) TT-sensor combo currently shows:
        # NGS always senses on R (same rule _apply_target_self_mags
        # already hard-codes for ngs_bright) regardless of a STRAP/TRICK
        # selection left over from a PREVIOUS, unrelated LGS frame -- the
        # live combo is UI state, not a property of THIS frame, exactly
        # the same trap as blindly trusting a spinbox value (Eduardo
        # 2026-07-28, second report: this bug's twin, same root cause).
        ngs_band = "R"
        # For LGS frames the same trap has a header-confirmable half: when
        # THIS frame's own DYYMASTR/DTSENSOR telemetry says TRICK was NOT
        # the sensor (p.trick_active is False), the band is R full stop --
        # no need to trust the live combo at all, which may be sitting on
        # a stale TRICK H/K selection from an earlier target (Eduardo
        # 2026-07-28: "when TRICK is in use we can tell this if DYYMASTR=1
        # and dtsensor=3"). Those two keywords only ever confirm STRAP;
        # they don't distinguish TRICK's H from K, so when the header DOES
        # confirm TRICK, the combo is still the only source for which of
        # the two -- unless it disagrees outright (shows a STRAP entry),
        # in which case neither source can be trusted and the caller
        # should be told the band is unresolved rather than silently
        # guessing R.
        combo_band = self._tt_sensor_band()
        if p.trick_active is False:
            lgs_band = "R"
        elif p.trick_active is True and combo_band == "R":
            lgs_band = None
        else:
            lgs_band = combo_band
        name = (p.object_name or "").strip()
        entries = getattr(self, "_starlist_entries", None) or []

        header = getattr(self, "_n2_header", None)
        off = self._nirc2_tt_offset_evidence(p, header)
        if off is not None:
            if off["on_axis"] and name:
                band = ngs_band if p.lgs is False else lgs_band
                for e in (entries if band is not None else ()):
                    if engine.same_star_name(e["name"], name):
                        mag, kind, _lab = engine.estimate_sensing_mag(
                            _starlist_entry_mags(e), band)
                        if mag is not None:
                            return (name, mag,
                                   f"TSS odometer: on-axis ({band}, {kind})")
                idx = self._nirc2_target_index(name)
                if idx is not None:
                    tm = self._targets[idx].get("tt_mag")
                    if tm is not None:
                        return (name, float(tm),
                               "TSS odometer: on-axis (target list entry)")
                # no starlist/target-list entry either -- same catalogue
                # auto-fetch fallback the NGS tier already uses (on-axis
                # means the guide star IS the target regardless of NGS
                # vs LGS, so the same "the star found at the target's own
                # position is unambiguously it" logic applies to both).
                # Without this an on-axis LGS target with no starlist
                # loaded fell straight to ASSUMED even after the odometer
                # correctly confirmed on-axis (2026-07-28 on-sky
                # session).
                if band is not None:
                    mag, src_tail = self._nirc2_catalog_selfmag(name, p, band)
                    if mag is not None:
                        return (name, mag, f"TSS odometer: on-axis ({src_tail})")
            elif not off["on_axis"]:
                ranked = []
                if entries:
                    try:
                        c = engine.parse_radec(p.ra, p.dec)
                        cand_stars = [{"id": e["name"], "ra": e["ra_deg"],
                                       "dec": e["dec_deg"],
                                       "mags": _starlist_entry_mags(e)}
                                     for e in entries]
                        ranked = engine.tt_ring_match(
                            cand_stars, float(c.ra.deg), float(c.dec.deg),
                            off["sep_arcsec"])
                    except Exception:
                        ranked = []
                if ranked:
                    star = ranked[0]["star"]
                    b, m = engine.best_mag(star)
                    if m is not None:
                        return (star["id"], m,
                               f"TSS odometer → starlist match ({b}, "
                               f"Δ{ranked[0]['dsep_arcsec']:+.1f}″ off ring)")
                # no starlist match (or none loaded) -- the batch
                # auto-fetch already queried at this off-axis position
                # for TRICK frames specifically (_nirc2_prefetch_guide_
                # stars uses the ROI's real sky offset as the query
                # centre, not the bare target position) and cached
                # whatever it found under the TARGET's name; magnitude
                # only, no identity claimed -- the matched star is NOT
                # the target itself here, unlike the on-axis case above
                if name:
                    band = ngs_band if p.lgs is False else lgs_band
                    if band is not None:
                        mag, src_tail = self._nirc2_catalog_selfmag(
                            name, p, band)
                        if mag is not None:
                            return (None, mag,
                                   f"TSS odometer: off-axis ({src_tail})")

        tt = getattr(self, "_n2_tt_star_resolved", None)
        if tt and tt.get("imno") == self._n2_imno:
            return (tt["id"], tt["mag"],
                   f"{tt['catalog']} ring match ({tt['band']})")

        if p.lgs is False and name:
            # NO LASER -> there is no separate tip-tilt star: the AO guides
            # on the science target itself, so its OWN magnitude is the
            # sensing magnitude. This must be checked BEFORE the starlist
            # target= link below, because a target=-linked entry is an LGS
            # tip-tilt star specifically -- crediting it on an NGS frame
            # reports the wrong star entirely. (Same rule the starlist
            # picker already encodes: "NGS: the guide star is ALWAYS the
            # target itself, no starlist exception exists for NGS".)
            for e in entries:
                if engine.same_star_name(e["name"], name):
                    mag, kind, _lab = engine.estimate_sensing_mag(
                        _starlist_entry_mags(e), ngs_band)
                    if mag is not None:
                        return (name, mag,
                               f"science target itself, NGS on-axis "
                               f"({ngs_band}, {kind})")
            # no starlist entry carries this target's OWN name/magnitude --
            # a catalogue star essentially AT the target's own position is
            # unambiguously the target itself (Eduardo 2026-07-28: the
            # ranking dialog already knew o Her's real R mag (~5.7-5.8)
            # from a catalogue while the log fell all the way to the bare
            # ngs_bright default (8) because only the starlist was
            # checked here). See _nirc2_catalog_selfmag for the two
            # catalogue sources this tries, in order.
            mag, src_tail = self._nirc2_catalog_selfmag(name, p, ngs_band)
            if mag is not None:
                return (name, mag,
                       f"science target itself, NGS on-axis ({src_tail})")

        if name and p.lgs is not False and lgs_band is not None:
            for e in entries:
                if e.get("target") and engine.same_star_name(e["target"], name):
                    mag, kind, _lab = engine.estimate_sensing_mag(
                        _starlist_entry_mags(e), lgs_band)
                    if mag is not None:
                        return (e["name"], mag,
                               f"starlist TT star ({lgs_band}, {kind})")

        idx = self._nirc2_target_index(name)
        if idx is not None:
            tm = self._targets[idx].get("tt_mag")
            if tm is not None:
                # the target-list entry records a tt_mag but NOT which
                # star it belongs to (on-axis, or an offset star picked
                # via some earlier flow) -- so a magnitude is known but
                # an identity genuinely isn't; don't claim the target's
                # own name here, that would overstate this tier's evidence
                return None, float(tm), "target list entry"

        # Last resort: whichever spinbox is the CORRECT control for this
        # frame's mode -- NGS senses on ngs_bright (R-band), never tt_mag
        # (that's the LGS/STRAP/TRICK tip-tilt sensing control, sat on a
        # completely different star; reading it for an NGS frame reports
        # a number with no relationship to this frame at all, which is
        # exactly the bug Eduardo hit: AO mode correctly read NGS while
        # Guide mag showed a leftover TT-mag value from an unrelated LGS
        # target). Still labelled ASSUMED: even the right spinbox may be
        # a stale default nobody has set for this particular target.
        if p.lgs is False:
            return None, float(self.ngs_bright.value()), \
                "estimator default — ASSUMED (NGS R spinbox)"
        if p.trick_active is True and combo_band == "R":
            # the header confirms TRICK was on but the live combo is
            # sitting on a STRAP entry -- neither source can be trusted
            # for H vs K, so say so rather than silently reporting R
            return None, float(self.tt_mag.value()), \
                ("estimator default — ASSUMED (TT-mag spinbox); band "
                 "unresolved: header confirms TRICK but TT-sensor combo "
                 "shows STRAP")
        return None, float(self.tt_mag.value()), \
            "estimator default — ASSUMED (TT-mag spinbox)"

    # how far the real TT magnitude may sit from the one the loaded
    # prediction was computed at before it's worth saying so out loud
    _GUIDE_MAG_WARN_TOL = 0.3

    def _nirc2_warn_guide_mag_mismatch(self, p):
        """Say so when the magnitude the prediction was computed at (the
        spinbox that actually feeds THIS frame's compared series -- see
        below) disagrees with the real one resolved for this frame -- the
        predicted SR/FWHM being compared against is then evaluated for the
        WRONG guide star, which is invisible otherwise (Eduardo
        2026-07-28). Warn-only by design: silently re-running the
        estimator underneath a measurement would change the comparison
        the user is reading. Once per frame."""
        if self.res is None or self._n2_imno == self._n2_guide_mag_warned:
            return
        _gname, real, src = self._nirc2_resolve_guide_star(p)
        if real is None or "ASSUMED" in src:
            return
        # the predicted series _nirc2_compare picks for this frame is
        # NGS-bright (ngs_bright) when there's no laser, else the LGS/LTAO
        # series (tt_mag) -- comparing "real" against the OTHER spinbox
        # would flag a mismatch (or miss one) for a control that was never
        # part of this frame's prediction at all (same bug as the
        # fallback tier above, Eduardo's second report)
        if p.lgs is False:
            used, ctrl = float(self.ngs_bright.value()), "NGS mag"
        else:
            used, ctrl = float(self.tt_mag.value()), "TT mag"
        if abs(real - used) < self._GUIDE_MAG_WARN_TOL:
            return
        self._n2_guide_mag_warned = self._n2_imno
        self.n2_log.appendPlainText(
            f"! {ctrl}: this frame's real guide star is "
            f"{real:.2f} ({src}), but the loaded prediction was computed "
            f"at {used:.2f} — the PREDICTED SR/FWHM above are for the "
            f"wrong guide star. Set {ctrl} to {real:.2f} and re-run to "
            f"compare like for like.")

    def _nirc2_find_duplicate_rows(self, row):
        """Existing self._n2_csv_rows indices that look like the same
        measurement as `row`: same frame number AND the star centroid
        within _DUP_POS_TOL_PX pixels of an already-logged one. Catches
        re-clicking an already-logged point on the field map to inspect it
        (_on_nirc2_map_pick calls _nirc2_display, which logs again) without
        false-positiving on a genuinely different star measured in the same
        frame."""
        fn = row.get("frame_number")
        x, y = row.get("pixel_x"), row.get("pixel_y")
        if fn is None or x is None or y is None:
            return []
        out = []
        for i, r in enumerate(self._n2_csv_rows):
            ox, oy = r.get("pixel_x"), r.get("pixel_y")
            if r.get("frame_number") != fn or ox is None or oy is None:
                continue
            if np.hypot(x - ox, y - oy) <= _DUP_POS_TOL_PX:
                out.append(i)
        return out

    def _nirc2_add_csv_row(self, result, cmp_res):
        """Append one measurement to self._n2_csv_rows, guarding against
        logging the same star on the same frame twice (see
        _nirc2_find_duplicate_rows): with a match, asks whether to Append
        (log both), Overwrite (replace the existing matching row(s)), or
        Don't add (skip this one).

        Does nothing at all when the "Append to log" checkbox is off.

        On a duplicate the run PAUSES and asks (Eduardo 2026-07-28), with
        the choice applicable to just this file or to every remaining
        duplicate in the run (_n2_dup_batch_policy, reset per run in
        _nirc2_start).

        The whole body is wrapped: this is auxiliary logging, called from
        the middle of the main measurement display path (_nirc2_display)
        -- an exception here must never be allowed to propagate out and
        take the rest of that method (or the app) down with it (Eduardo
        2026-07-28: it did). Caught and reported on the status line/text
        log instead."""
        try:
            if not self.n2_append_log.isChecked():
                return                     # logging switched off entirely
            row = self._nirc2_csv_row(result, cmp_res)
            dupes = self._nirc2_find_duplicate_rows(row)
            if dupes:
                policy = getattr(self, "_n2_dup_batch_policy", None)
                if policy is None:
                    # A modal dialog spins a NESTED Qt event loop, which
                    # would deliver frames the worker already finished ->
                    # straight back in here -> a second dialog stacked on
                    # this one. So: PAUSE the worker (it really stops --
                    # see Nirc2MeasureWorker.pause) and refuse to open a
                    # second box while one is up; anything that slips
                    # through in-flight is queued and handled after
                    # (_nirc2_drain_pending_duplicates).
                    if self._n2_dup_dialog_open:
                        self._n2_dup_pending.append(row)
                        return
                    policy = self._nirc2_ask_duplicate_paused(row, dupes)
                self._nirc2_apply_dup_policy(policy, row, dupes)
                # settle anything that landed in-flight behind the dialog
                self._nirc2_drain_pending_duplicates()
                return
            self._n2_csv_rows.append(row)
            if self._n2_csv_table is not None:
                self._nirc2_fill_csv_table(self._n2_csv_table)
        except Exception as ex:
            self.n2_log.appendPlainText(
                f"! image log: failed to record this measurement ({ex}) "
                "-- the measurement itself is unaffected")

    def _nirc2_ask_duplicate_paused(self, row, dupes):
        """Ask what to do about ONE duplicate frame, with the sequence
        PAUSED for the duration, then resume it.

        The pause is the point (Eduardo 2026-07-28): a modal dialog runs a
        nested Qt event loop, so without stopping the worker the run races
        on and its queued results pile up behind the box. Nirc2MeasureWorker
        .pause() makes the loop actually wait for the answer.

        Returns "append" | "overwrite" | "skip"; when the user ticks
        "apply to all remaining", the choice is also stored as this run's
        _n2_dup_batch_policy so nothing asks again."""
        worker = self._n2_worker
        running = worker is not None and worker.isRunning()
        if running:
            worker.pause()
        self._n2_dup_dialog_open = True
        try:
            box = QtWidgets.QMessageBox(self)
            box.setIcon(QtWidgets.QMessageBox.Icon.Question)
            box.setWindowTitle("Frame already in the log")
            box.setText(
                f"Frame {row.get('frame_number')} is already in the image "
                f"log ({len(dupes)} matching measurement(s) at this "
                "star's position).\n\nWhat should this measurement do?")
            if running:
                box.setInformativeText("The run is paused until you choose.")
            all_cb = QtWidgets.QCheckBox(
                "Apply to all remaining duplicates in this run")
            box.setCheckBox(all_cb)
            append_btn = box.addButton(
                "Append", QtWidgets.QMessageBox.ButtonRole.AcceptRole)
            overwrite_btn = box.addButton(
                "Overwrite", QtWidgets.QMessageBox.ButtonRole.DestructiveRole)
            skip_btn = box.addButton(
                "Don't add", QtWidgets.QMessageBox.ButtonRole.RejectRole)
            box.setDefaultButton(skip_btn)
            box.exec()
            clicked = box.clickedButton()
            policy = ("append" if clicked is append_btn else
                      "overwrite" if clicked is overwrite_btn else "skip")
            if all_cb.isChecked():
                self._n2_dup_batch_policy = policy
        finally:
            self._n2_dup_dialog_open = False
            if running:
                worker.resume()
        # NOTE: draining the in-flight queue is deliberately NOT done here.
        # This method is itself called FROM the drain loop, so draining here
        # too would make the two mutually recursive. The top-level caller
        # (_nirc2_add_csv_row) drains once, after its own decision is
        # applied.
        return policy

    def _nirc2_apply_dup_policy(self, policy, row, dupes):
        """Carry out a duplicate decision for one row."""
        if policy == "skip":
            self.status.setText(
                f"Frame {row.get('frame_number')}: duplicate not logged.")
            return
        if policy == "overwrite":
            for i in sorted(dupes, reverse=True):
                del self._n2_csv_rows[i]
        self._n2_csv_rows.append(row)
        if self._n2_csv_table is not None:
            self._nirc2_fill_csv_table(self._n2_csv_table)

    def _nirc2_drain_pending_duplicates(self):
        """Handle duplicates that arrived while a dialog was already open
        (frames the worker had finished before the pause took effect). With
        an "all remaining" policy set they resolve silently; otherwise each
        gets its own dialog -- one after another, never stacked.

        Guarded against re-entry: each dialog it opens spins a nested event
        loop that can deliver more frames, which land back in
        _nirc2_add_csv_row and call this again. The outermost call owns the
        queue and keeps looping until it is empty."""
        if getattr(self, "_n2_draining", False):
            return
        self._n2_draining = True
        try:
            self._nirc2_drain_loop()
        finally:
            self._n2_draining = False

    def _nirc2_drain_loop(self):
        while self._n2_dup_pending:
            row = self._n2_dup_pending.pop(0)
            dupes = self._nirc2_find_duplicate_rows(row)
            if not dupes:                  # earlier choice already cleared it
                self._n2_csv_rows.append(row)
                continue
            policy = getattr(self, "_n2_dup_batch_policy", None)
            if policy is None:
                policy = self._nirc2_ask_duplicate_paused(row, dupes)
            self._nirc2_apply_dup_policy(policy, row, dupes)
        if self._n2_csv_table is not None:
            self._nirc2_fill_csv_table(self._n2_csv_table)
