"""MainWindow: the shell (window/menu/bottom-bar construction), the
debounced recompute/render pipeline, config save/load, PNG/CSV export,
and the about/help dialogs. Composes the tab mixins (gui.tabs.*), which
own each tab's own UI construction and tab-specific logic; mixins share
`self`, so this split does not change any widget/state wiring.
"""
import os
import tempfile

# The engine uses astropy target geometry (airmass). On a box whose bundled
# IERS Earth-orientation table is >30 days stale and offline, astropy refuses
# to interpolate it and target mode would crash. astropy's own documented
# remedy; changes nothing numeric (verified against the regression refs). Must
# happen before the engine imports astropy.time/coordinates.
try:
    from astropy.utils.iers import conf as _iers_conf
    _iers_conf.auto_max_age = None
except Exception:
    pass

import numpy as np
from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar)
from matplotlib.figure import Figure
from qtcompat import QTimer, Qt, QtCore, QtWidgets

import keck_ao_estimator as engine

from .._version import APP_NAME, MAINTAINER, ORGANIZATION, __version__
from .about import DOC_BENCH_DIAGRAMS, DOC_TECH_NOTE, DOC_USER_MANUAL, _bundled_doc
from .constants import HIPS_SURVEYS, LOCAL_BACKDROP, NGS_LGS_ONLY_TERMS
from .tabs.data import DataTabMixin
from .tabs.fa_geometry import FaGeometryMixin
from .tabs.fieldmap_overlays import FieldMapOverlaysMixin
from .tabs.fieldmap_tab import FieldMapMixin
from .tabs.fieldmap_view import FieldMapViewMixin
from .tabs.lgs import LgsTabMixin
from .tabs.ngs import NgsTabMixin
from .tabs.nighttime import NighttimeModeMixin
from .tabs.nirc2_strehl import Nirc2StrehlTabMixin
from .tabs.prediction import PredictionTabMixin
from .tabs.starlist_picker import StarlistPickerMixin
from .tabs.summary_stats import SummaryStatsMixin
from .tabs.target import TargetTabMixin
from .tabs.wfe import WfeTabMixin
from .theme import apply_theme
from .widgets import _dspin, _shrinkable_label
from .workers import PrepareWorker


def _build_summary(args, prep, res, offsets=None):
    """Compose the status-bar line: night / mode / counts + budget provenance
    (§2 bottom bar), echoing what the CLI console summary conveys plus a
    MODIFIED BUDGET note when WFE sliders are off-default."""
    n_dimm = len(res.times)
    n_mass = len(res.p_times)
    tomo = "ON" if prep.tomography_on else "off"
    lgs_off = (engine.DEF_LGS_OFFSET[args.telescope]
               if args.lgs_offset is None else args.lgs_offset)
    parts = [
        f"Night {prep.night_date.date()}  {args.telescope}  tomography {tomo}"
        f"  ({n_dimm} DIMM, {n_mass} MASS)",
        f"budget: {'LEGACY' if args.legacy_budget else 'refined 2026-07'}"
        f"  TT R={args.tt_mag:g}@{args.tt_offset:g}\"  LGS offset {lgs_off:g}\""
        f"  NGS law {args.ngs_seeing_law}",
    ]
    if offsets:
        parts.append("MODIFIED BUDGET: "
                     + ", ".join(f"{k}={v:g}" for k, v in sorted(offsets.items())))
    return "  |  ".join(parts)


class MainWindow(DataTabMixin, FaGeometryMixin, TargetTabMixin,
                 StarlistPickerMixin,
                 SummaryStatsMixin, NgsTabMixin, LgsTabMixin,
                 WfeTabMixin, PredictionTabMixin, FieldMapMixin,
                 FieldMapOverlaysMixin, FieldMapViewMixin, NighttimeModeMixin,
                 Nirc2StrehlTabMixin,
                 QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        # The parser is the single source of truth for defaults (§3): build it
        # once, read a defaults namespace, and seed every widget from it.
        self.parser = engine.build_parser()
        self.defaults = self.parser.parse_args([])

        self.worker = None
        self.prep = None            # cached prepared night (expensive)
        self.args_cached = None     # args used for the current prep
        self.res = None             # last compute_timeline result
        self._targets = []          # night's targets: [{name, ra, dec}, ...]
        self.fig_terms = None       # last terms Figure (for the Phase-5 tab)
        self.last_offsets = {}      # WFE offsets the current res was computed under
        self._busy = False          # guard: drop re-entrant recomputes
        self._terms_dirty = True    # terms figure is stale (rendered lazily)
        self._fieldmap_dirty = True # field map is stale (rendered lazily)
        self._mass_note = ""
        self._tmpdir = tempfile.mkdtemp(prefix="ao_strehl_gui_")

        # theme (gui/theme.py): install the light cue-stylesheet up front so
        # the semantic label colors work before any dark switch; also captures
        # the platform-default style/palette that "light" restores.
        apply_theme(QtWidgets.QApplication.instance(), dark=False)
        self._dark_syncing = False  # guard: programmatic dark_action flips
        self._dark_auto = False     # True while Nighttime mode owns the theme

        # Live-update debounce (§4): coalesce rapid control changes into one
        # action 150 ms after the last change. Two kinds of change:
        #   'recompute' — compute-only controls (NGS mags, TT star, LGS offset,
        #                 WFE sliders): re-run compute_timeline on the cached
        #                 prepared night (milliseconds, GUI thread).
        #   'rerun'     — controls baked into prepare_night (telescope,
        #                 wavelength, zenith, tomography/legacy/LTAO-floor):
        #                 re-prepare off-thread, then compute.
        # 'rerun' wins if both are pending in one window.
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(150)
        self._debounce.timeout.connect(self._on_debounced)
        self._pending_mode = "recompute"
        # Field-map-specific debounce (§4 cont'd): the map is a whole grid
        # re-evaluation (tens-to-~130 ms, vs. the main panel's single-point
        # recompute), so redrawing it synchronously on EVERY input tick --
        # e.g. every step while a spinbox's up/down button is held -- blocks
        # the UI thread long enough per tick to feel choppy rather than live.
        # Same 150 ms coalescing window, kept as its own timer so the
        # (typically cheaper) main-panel recompute is never held up by it.
        self._fm_debounce = QTimer(self)
        self._fm_debounce.setSingleShot(True)
        self._fm_debounce.setInterval(150)
        self._fm_debounce.timeout.connect(self._render_field_map_if_visible)
        # Level-of-detail while scrubbing: the throttled redraws above keep up
        # with a held button by RATE, but each full-resolution grid is still
        # tens-to-~130 ms. While input is actively arriving we render a COARSE
        # grid (fast, blocky) so motion stays fluid, then this trailing timer
        # -- a true debounce, restarted on every input tick -- fires one
        # full-resolution redraw once the button is released and input goes
        # quiet. _render_field_map reads _fm_settle.isActive() to decide which.
        self._fm_settle = QTimer(self)
        self._fm_settle.setSingleShot(True)
        self._fm_settle.setInterval(220)
        self._fm_settle.timeout.connect(self._render_field_map_full)
        self._loading = False       # True while _apply_config bulk-sets widgets

        self.setWindowTitle(f"{APP_NAME} v{__version__}")
        # fit the screen: never open (or be forced) larger than the desktop
        scr = QtWidgets.QApplication.primaryScreen()
        avail = scr.availableSize() if scr is not None else None
        w = min(1500, avail.width() - 40) if avail else 1500
        h = min(950, avail.height() - 60) if avail else 950
        self.resize(max(w, 900), max(h, 600))
        self._build_ui()
        self._validate()          # set initial Run-enabled state

    # ---- UI construction ----------------------------------------------------
    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QHBoxLayout(central)

        # left: control dock (tabs)
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setMinimumWidth(430)
        self.tabs.setMaximumWidth(520)
        self.tabs.addTab(self._tab_data(), "Data")
        self.tabs.addTab(self._tab_target(), "Target")
        self.tabs.addTab(self._tab_ngs(), "NGS")
        self.tabs.addTab(self._tab_budget(), "LGS")
        self.tabs.addTab(self._tab_wfe(), "WFE sliders")
        self.tabs.addTab(self._tab_prediction(), "Prediction")
        root.addWidget(self.tabs)

        # center: two plot tabs (main timeline + error-terms figure), each a
        # matplotlib canvas + nav toolbar. _show_figure() rebinds the canvas to
        # each freshly engine-rendered Figure.
        self.plot_tabs = QtWidgets.QTabWidget()
        self._main_holder = self._make_canvas_tab(
            "Set inputs on the left, then press Run.")
        self._terms_holder = self._make_canvas_tab(
            "Error-terms figure appears here after a run with MASS data.")
        self.plot_tabs.addTab(self._main_holder["widget"], "Timeline")
        self.plot_tabs.addTab(self._build_field_map_tab(), "Field map")
        self.plot_tabs.addTab(self._terms_holder["widget"], "Error terms")
        self.plot_tabs.addTab(self._build_nirc2_tab(), "Measured SR")
        # terms + field map are expensive -> render only when their tab is shown
        self.plot_tabs.currentChanged.connect(self._on_plot_tab_changed)
        root.addWidget(self.plot_tabs, 1)

        # menu bar (config, presets, shortcuts) + bottom bar
        self._build_menu()
        self._build_bottom_bar()

    def _build_menu(self):
        """Menu bar with config save/load, presets, and keyboard shortcuts."""
        from qtcompat import QAction
        mb = self.menuBar()
        m_file = mb.addMenu("&File")

        def act(text, shortcut, slot):
            a = QAction(text, self)
            if shortcut:
                a.setShortcut(shortcut)
            a.triggered.connect(slot)
            return a

        m_file.addAction(act("&Run", "Ctrl+R", self.on_run))
        m_file.addSeparator()
        m_file.addAction(act("&Save config…", "Ctrl+S", self.on_save_config))
        m_file.addAction(act("&Load config…", "Ctrl+O", self.on_load_config))
        m_file.addSeparator()
        m_file.addAction(act("Export &PNG…", "Ctrl+P", self.on_export_png))
        m_file.addAction(act("Export &CSV…", "Ctrl+E", self.on_export_csv))
        m_file.addSeparator()
        m_file.addAction(act("&Quit", "Ctrl+Q", self.close))

        m_preset = mb.addMenu("&Presets")
        m_preset.addAction(act("Reference budget (reset WFE sliders)",
                               "Ctrl+0", self._preset_reference_budget))
        m_preset.addAction(act("Last night (fetch yesterday UT)", None,
                               self._preset_last_night))
        m_preset.addAction(act("Tonight's data (fetch current UT)", None,
                               self._preset_tonight))

        m_view = mb.addMenu("&View")
        self.dark_action = QAction("&Dark theme", self)
        self.dark_action.setCheckable(True)
        self.dark_action.setToolTip(
            "Dark Qt controls (plots keep their light print styling). "
            "Nighttime mode switches this on automatically; toggling it "
            "yourself takes ownership back.")
        self.dark_action.toggled.connect(self._on_dark_toggled)
        m_view.addAction(self.dark_action)

        m_help = mb.addMenu("&Help")
        m_help.addAction(act("&User Manual — KAON 1556 (PDF)", None,
                             lambda: self._open_doc(DOC_USER_MANUAL)))
        m_help.addAction(act("Technical Note — KAON 1542 (PDF)", None,
                             lambda: self._open_doc(DOC_TECH_NOTE)))
        m_help.addAction(act("AO Bench Block Diagrams — KAON 1488 (PDF)", None,
                             lambda: self._open_doc(DOC_BENCH_DIAGRAMS)))
        m_help.addSeparator()
        m_help.addAction(act(f"&About {APP_NAME}…", None,
                             lambda: self._show_about(at_startup=False)))

    def _build_bottom_bar(self):
        bar = QtWidgets.QWidget()
        h = QtWidgets.QHBoxLayout(bar)
        h.setContentsMargins(8, 4, 8, 4)
        self.run_btn = QtWidgets.QPushButton("Run")
        self.run_btn.setDefault(True)
        self.run_btn.clicked.connect(self.on_run)
        # performance metric: Strehl (default) / FWHM / both (--report)
        self.report_combo = QtWidgets.QComboBox()
        self.report_combo.addItems(["Strehl", "FWHM", "Strehl + FWHM"])
        self.report_combo.setToolTip(
            "What the main figure reports. FWHM comes from a per-sample "
            "Airy-core + Moffat-halo PSF model (core energy = Strehl); "
            "'Strehl + FWHM' overlays FWHM on a right-hand axis.")
        self.report_combo.currentTextChanged.connect(self._on_compute_changed)
        # which FWHM convention(s) the FWHM/both figure plots (halfmax default;
        # plotting both is busy). Only relevant when the report includes FWHM.
        self.fwhm_curves_combo = QtWidgets.QComboBox()
        self.fwhm_curves_combo.addItems(
            # "as the SR tool reads it" is FIRST, and therefore the default
            # (Eduardo 2026-08-07): it is the convention this app's own
            # Measured-SR tab reports in, so it is the one a predicted-vs-
            # delivered comparison should be reading.
            ["as the SR tool reads it", "half-max", "Gaussian-fit",
             "Gaussian-fit (+background)", "both curves", "all four"])
        self.fwhm_curves_combo.setToolTip(
            "FWHM figure: which convention to draw — half-max (core+halo "
            "model, half-max crossing; no confirmed real-tool analog), "
            "Gaussian-fit (no background — models the OSIRIS quicklook "
            "tool's rarely-used Strehl button, OSIRISSTREHL_QL2.pro), "
            "Gaussian-fit (+background) (models the OSIRIS quicklook tool's "
            "hand-drawn-box fit feature — a separate, independent tool from "
            "the AO Strehl tool), 'as the SR tool reads it' (THIS tool's "
            "own Measured-SR tab run on the same model PSF — the one "
            "convention directly comparable to a measured FWHM, and the "
            "closest to what the tab actually delivers), or combos. CSV "
            "always carries all four sets of columns.")
        self.fwhm_curves_combo.currentTextChanged.connect(self._on_compute_changed)
        self.report_combo.currentTextChanged.connect(self._sync_fwhm_curves_enable)
        # fit-domain radius shared by both Gaussian-fit conventions above —
        # the OSIRIS quicklook tool's hand-drawn-box fit feature has no fixed
        # box, so this is a real, explorable knob rather than a calibrated
        # default
        self.fwhm_box_mas = _dspin(5, 2000, 10, 300.0, 0, " mas")
        self.fwhm_box_mas.setToolTip(
            "Fit-domain radius (mas) for BOTH Gaussian-fit FWHM conventions. "
            "The OSIRIS quicklook tool's hand-drawn-box fit feature has its "
            "box drawn by hand with the mouse — there is no single correct "
            "value. Default 300 preserves the validated 20260701 numbers; "
            "OSIRISSTREHL_QL2.pro's own auto-sized box works out to ~30.7 "
            "mas at K band.")
        self.fwhm_box_mas.valueChanged.connect(self._on_compute_changed)
        self.spinner = QtWidgets.QProgressBar()
        self.spinner.setRange(0, 0)          # indeterminate "busy" style
        self.spinner.setMaximumWidth(140)
        self.spinner.setVisible(False)
        self.status = QtWidgets.QLabel("Ready.")
        self.status.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse)
        _shrinkable_label(self.status)
        self.export_png_btn = QtWidgets.QPushButton("Export PNG…")
        self.export_csv_btn = QtWidgets.QPushButton("Export CSV…")
        for b in (self.export_png_btn, self.export_csv_btn):
            b.setEnabled(False)              # enabled after a successful run
        self.export_png_btn.clicked.connect(self.on_export_png)
        self.export_csv_btn.clicked.connect(self.on_export_csv)

        h.addWidget(self.run_btn)
        h.addWidget(QtWidgets.QLabel("Report:"))
        h.addWidget(self.report_combo)
        h.addWidget(QtWidgets.QLabel("FWHM:"))
        h.addWidget(self.fwhm_curves_combo)
        h.addWidget(QtWidgets.QLabel("box:"))
        h.addWidget(self.fwhm_box_mas)
        self._sync_fwhm_curves_enable()
        h.addWidget(self.spinner)
        h.addWidget(self.status, 1)
        h.addWidget(self.export_png_btn)
        h.addWidget(self.export_csv_btn)

        dock = QtWidgets.QDockWidget()
        dock.setTitleBarWidget(QtWidgets.QWidget())   # hide title bar
        dock.setFeatures(QtWidgets.QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)
        dock.setWidget(bar)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, dock)
        self._busy = False

    # ---- tabs ---------------------------------------------------------------
    def _wrap(self, layout):
        c = QtWidgets.QWidget()
        c.setLayout(layout)
        # a wrapped row must sit in its form exactly like a bare widget would:
        # the default 9px QLayout margins both indented every wrapped row and
        # added 18px to its minimum width/height -- across a whole form that
        # width was part of what forced a horizontal scrollbar on narrow
        # panels (and needless height on short screens)
        layout.setContentsMargins(0, 0, 0, 0)
        return c

    def _scroll(self, w):
        s = QtWidgets.QScrollArea()
        s.setWidgetResizable(True)
        s.setWidget(w)
        return s

    def _file_picker(self, label, directory=False):
        edit = QtWidgets.QLineEdit()
        edit.setMinimumWidth(60)   # floor: paths elide, never widen the panel
        btn = QtWidgets.QPushButton("Browse…")
        btn.setMinimumWidth(60)

        def browse():
            if directory:
                p = QtWidgets.QFileDialog.getExistingDirectory(self, f"Choose {label}")
            else:
                p, _ = QtWidgets.QFileDialog.getOpenFileName(self, f"Choose {label}")
            if p:
                edit.setText(p)
        btn.clicked.connect(browse)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(edit, 1)
        row.addWidget(btn)
        return edit, self._wrap(row)

    def _on_compute_changed(self, *_):
        """A compute-only control moved (NGS mags, TT star, LGS offset): live
        recompute on the cached prepared night."""
        self._schedule("recompute")

    def _on_prep_changed(self, *_):
        """A control baked into prepare_night moved (telescope, wavelength,
        zenith, tomography/legacy/LTAO-floor): schedule a live re-prepare."""
        self._schedule("rerun")

    def _schedule(self, mode):
        """Arm a live update, throttled to roughly one per 150 ms rather than
        debounced to fire only once input goes quiet: holding a spinbox's
        up/down button (NGS/TT magnitude, a WFE slider, ...) fires
        valueChanged faster than 150 ms apart, so a plain reset-on-every-tick
        timer would never get a gap to fire and the timeline/field map would
        sit frozen until release. Only arming when the timer isn't already
        running gives one recompute+redraw (recompute_and_draw also redraws
        the field map if that tab is visible) roughly every 150 ms for the
        whole duration of a hold. Only fires once a night has been prepared
        (after the first manual Run) -- or, with NO night loaded, while the
        Prediction tab's scenario is enabled (the field map and terms tab
        are then fed straight from the live widgets via the surrogates, so
        control changes must re-render them; 2026-08-12) -- and never
        during a bulk config load. 'rerun' takes precedence over
        'recompute' within one window."""
        if self._loading or (self.prep is None and not self._pred_scenario_on()):
            return
        if mode == "rerun" or self._pending_mode == "rerun":
            self._pending_mode = "rerun"
        else:
            self._pending_mode = "recompute"
        if not self._debounce.isActive():
            self._debounce.start()
        # these controls (NGS/TT mag, WFE sliders) also redraw the field map via
        # recompute_and_draw -> arm the LOD settle timer so a held button there
        # gets the same coarse-while-scrubbing / full-on-release treatment.
        if self._field_map_visible():
            self._fm_settle.start()

    def _pred_scenario_on(self):
        box = getattr(self, "pred_enable", None)
        return box is not None and box.isChecked()

    def _refresh_prediction_views(self):
        """No night loaded, scenario on: the field map and the predicted
        terms figure read the live widgets through the _fm_args/_fm_prep
        surrogates, so a control change just needs fresh WFE offsets and a
        re-render of whichever view is showing. (With a night loaded the
        recompute path does the equivalent at lines ~610-620.)"""
        self.last_offsets = self.current_offsets()
        self._fieldmap_dirty = True
        self._terms_dirty = True
        self._render_field_map_if_visible()
        self._render_terms_if_visible()

    def _on_debounced(self):
        """Debounce fired: either recompute on the cached night, or (for a
        prep-affecting change) kick off a full re-prepare via the run path.
        With no night loaded (scenario-only mode) both flavors reduce to
        re-rendering the prediction views from the live widgets."""
        mode = self._pending_mode
        self._pending_mode = "recompute"
        if self.prep is None:
            self._refresh_prediction_views()
            return
        if mode == "rerun":
            if self.prep is not None and self.worker is None:
                self.on_run()                  # re-prepare + compute + draw
        else:
            self.recompute_and_draw()

    def _validate(self):
        errors = []
        # reset field styling
        for e in (getattr(self, "ra_edit", None), getattr(self, "dec_edit", None)):
            if e is not None:
                e.setStyleSheet("")
        if getattr(self, "target_enable", None) and self.target_enable.isChecked():
            if not self._radec_ok(self.ra_edit.text(), self.dec_edit.text()):
                self.ra_edit.setStyleSheet("border:1px solid red;")
                self.dec_edit.setStyleSheet("border:1px solid red;")
                errors.append("RA/Dec unparseable")
            bad_win = [self.windows_list.item(i).text()
                       for i in range(self.windows_list.count())
                       if not self._window_ok(self.windows_list.item(i).text())]
            if bad_win:
                errors.append(f"bad window(s): {', '.join(bad_win)}")
        if getattr(self, "mode_local", None) and self.mode_local.isChecked():
            for lbl, e in (("DIMM", self.dimm_edit),):   # DIMM is required
                if not e.text().strip():
                    errors.append(f"{lbl} file required")
        # offset entries in star-coordinate mode need parseable coordinates
        for lbl, entry in (("NGS offset", getattr(self, "ngs_offset", None)),
                           ("TT-star offset", getattr(self, "tt_offset", None)),
                           ("Target offset", getattr(self, "target_offset", None))):
            if entry is not None and not entry.ok():
                errors.append(f"{lbl}: star/target coordinates")
        ok = not errors
        if hasattr(self, "run_btn"):
            self.run_btn.setEnabled(ok and self.worker is None)
            if errors and self.worker is None:
                self.status.setText("Cannot run: " + "; ".join(errors))
        return ok

    def collect_args(self, out_path):
        """Build an args namespace exactly as the CLI would: start from parser
        defaults, then overlay widget values. This is the ONE place the widget
        -> flag mapping lives, so the CLI and GUI can never drift."""
        a = self.parser.parse_args([])        # fresh defaults every time

        # --- Data ---
        if self.mode_fetch.isChecked():
            a.fetch_date = self.fetch_date.date().toString("yyyyMMdd")
        else:
            a.fetch_date = None
            a.dimm = self.dimm_edit.text().strip() or a.dimm
            a.mass = self.mass_edit.text().strip() or a.mass
            a.masspro = self.masspro_edit.text().strip() or a.masspro
        a.telescope = "K1" if self.tel_k1.isChecked() else "K2"
        a.band = self.band_combo.currentText()
        a.wavelength = float(self.wl_nm.value()) if self.wl_enable.isChecked() else None
        a.wind_ground = float(self.wind_ground.value())
        a.wind_free = float(self.wind_free.value())
        a.report = {"Strehl": "strehl", "FWHM": "fwhm",
                    "Strehl + FWHM": "both"}[self.report_combo.currentText()]
        a.fwhm_curves = {"half-max": "halfmax", "Gaussian-fit": "gaussfit",
                         "both curves": "both",
                         "Gaussian-fit (+background)": "gaussfit-sky",
                         "as the SR tool reads it": "srtool",
                         "all four": "all"}[self.fwhm_curves_combo.currentText()]
        a.fwhm_box_mas = float(self.fwhm_box_mas.value())
        a.force = self.force_cb.isChecked()
        a.out = out_path

        # --- Target ---
        a.show_target = self.target_enable.isChecked()
        a.target_name = self.tname_edit.text().strip() or a.target_name
        # the target-offset control shifts the RA/Dec fields above (0 offset,
        # the default, leaves them unchanged); fall back to the raw fields if
        # they're empty/unparseable, exactly as before the offset existed.
        eff = self._effective_target_coords()
        if eff is not None:
            a.ra, a.dec = eff
        else:
            a.ra = self.ra_edit.text().strip() or a.ra
            a.dec = self.dec_edit.text().strip() or a.dec
        # always HST for the engine, whatever the display mode shows
        wins = self._windows_hst()
        a.window = wins or None
        a.zenith_angle = (float(self.za_spin.value())
                          if self.za_enable.isChecked() else 0.0)

        # --- NGS ---
        a.ngs_bright = float(self.ngs_bright.value())
        a.ngs_faint = float(self.ngs_faint.value())
        a.ngs_offset = float(self.ngs_offset.value())
        a.assumed_theta0 = float(self.assumed_theta0.value())
        a.ngs_seeing_law = self.seeing_law.currentText()
        a.ngs_s0 = float(self.ngs_s0.value())
        a.ngs_a = float(self.ngs_a.value())
        a.ngs_m0 = float(self.ngs_m0.value())
        a.ngs_w = float(self.ngs_w.value())
        a.k1_quadcell_penalty = float(self.k1_quadcell.value())

        # --- Budget ---
        a.tt_sensor = self._TT_SENSOR_MAP[self.tt_sensor.currentText()][0]
        a.tt_mag = float(self.tt_mag.value())
        a.tt_offset = float(self.tt_offset.value())
        a.lgs_offset = (float(self.lgs_offset.value())
                        if self.lgs_offset_enable.isChecked() else None)
        a.ltao_bw_floor_frac = float(self.ltao_floor.value())
        a.ltao_tt_theta0_gain = float(self.ltao_tt_gain.value())
        a.legacy_budget = self.legacy_cb.isChecked()
        tomo = self.tomo_combo.currentText()
        a.tomography = (None if tomo.startswith("auto")
                        else (tomo == "on"))
        return a

    # ---- run ----------------------------------------------------------------
    def on_run(self):
        """Collect args, then prepare the night off-thread. On success the GUI
        thread computes + renders (see _on_prepared)."""
        if self.worker is not None:
            return
        if not self._validate():
            return
        outdir = self.outdir_edit.text().strip() or self._tmpdir
        os.makedirs(outdir, exist_ok=True)
        out_path = os.path.join(outdir, "ao_strehl_gui_run.png")
        try:
            args = self.collect_args(out_path)
        except Exception as e:
            self._on_failed(f"collect_args failed: {e}", "")
            return
        self.args_cached = args

        self.run_btn.setEnabled(False)
        self.spinner.setVisible(True)
        self.status.setText("Preparing night (fetch / parse / target geometry)…")
        self.worker = PrepareWorker(args, self)
        self.worker.prepared.connect(self._on_prepared)
        self.worker.failed.connect(self._on_failed)
        self.worker.finished.connect(self._on_worker_cleanup)
        self.worker.start()

    def _on_prepared(self, prep, log):
        """Prepared night is ready (GUI thread). Cache it, then compute + render
        + draw + save. prep is reused by later slider recomputes (Phase 3)."""
        self.prep = prep
        self.prep_log = log
        # surface an NGS-only night from prep contents (§4)
        if len(getattr(prep, "mass_see", [])) == 0 or len(getattr(prep, "profiles", [])) == 0:
            self._mass_note = "MASS missing: NGS-only night"
        else:
            self._mass_note = ""
        # record a nighttime-mode pull's timestamp BEFORE the redraw below, so
        # a field map showing "time of last pull" picks up the fresh value on
        # this same redraw rather than one run behind
        if getattr(self, "nighttime_enable", None) is not None \
                and self.nighttime_enable.isChecked():
            self._on_nighttime_pull_done()
        self.recompute_and_draw()

    def recompute_and_draw(self, offsets=None):
        """The live path (GUI thread). Deliberately does ONLY what is needed to
        put the visible plot on screen:

          compute_timeline  ~31 ms
          render_main       ~71 ms
          canvas redraw     ~17 ms

        Everything else that used to live here is now deferred:
          * the terms figure (~570 ms) renders lazily, only when its tab shows;
          * savefig of main+terms (~1.5 s) and the CSV happen only on Export.
        Reuses the cached prepared night; re-entrant calls are dropped."""
        if self.prep is None or self._busy:
            return
        # Re-collect args from the widgets so compute-only edits (NGS mags, TT
        # star, LGS offset) take effect on this recompute. Prep-affecting fields
        # are unchanged here -- those go through the re-prepare (rerun) path --
        # so the freshly collected args stay consistent with the cached prep.
        self._busy = True
        try:
            try:
                args = self.collect_args(self.args_cached.out)
            except Exception as e:
                self._on_failed(f"collect_args failed: {e}", "")
                return
            self.args_cached = args
            if offsets is None:
                offsets = self.current_offsets()
            try:
                with engine.budget_overrides(**offsets):
                    res = engine.compute_timeline(args, self.prep)
                    fig_main = self._render_active_main(args, res)
                    if offsets:
                        self._decorate_main(fig_main, res, offsets, args)
                if self._utc():
                    # display-only relabel: ticks/annotations to UTC. The
                    # figure DATA stays HST -- CLI/harness untouched.
                    engine.apply_utc_display(fig_main)
            except Exception as e:
                self._on_failed(f"compute/render failed: {e}", "")
                return

            self.res = res
            self.last_offsets = dict(offsets)
            self._sync_assumed_theta0()
            self._terms_dirty = True          # invalidate the (lazy) terms tab
            self._fieldmap_dirty = True        # and the (lazy) field map
            self._render_field_map_if_visible()
            self._update_m_readout(args, res)
            self._update_lgs_profile_plot()
            self._refresh_summary_stats()
            self._update_fa_advisory()
            self._show_figure(self._main_holder, fig_main)
            self._render_terms_if_visible()
            self.setWindowTitle(
                f"{APP_NAME} v{__version__} — {self.prep.night_date.date()} "
                f"{args.telescope} "
                f"({'tomography' if self.prep.tomography_on else 'single'})"
                + ("  · MODIFIED BUDGET" if offsets else ""))
            msg = _build_summary(args, self.prep, res, offsets)
            if self._mass_note:
                msg = self._mass_note + "  |  " + msg
            self.status.setText(msg)
            self._resolve_pending_guide_star()  # may append to the status set above
            self.export_png_btn.setEnabled(True)
            self.export_csv_btn.setEnabled(True)
        finally:
            self._busy = False

    def _sync_fwhm_curves_enable(self, *_):
        _on = self.report_combo.currentText() != "Strehl"
        self.fwhm_curves_combo.setEnabled(_on)
        self.fwhm_box_mas.setEnabled(_on)

    def _render_active_main(self, args, res):
        """Build the figure the current Report mode calls for. Mirrors the
        CLI's main(): fwhm REPLACES the Strehl figure; both overlays a
        right-hand FWHM axis; strehl (default) is the unmodified figure."""
        if args.report == "fwhm":
            return engine.render_fwhm_figure(args, self.prep, res)
        # window_label_margin: keep the observing-window label out of the data
        fig = engine.render_main_figure(args, self.prep, res,
                                        window_label_margin=True)
        if args.report == "both":
            engine.overlay_fwhm_on_main(fig, args, self.prep, res)
        return fig

    def _decorate_main(self, fig_main, res, offsets, args):
        """MODIFIED BUDGET indicator + provenance footer + the projected-NGS
        overlay. Applied to any main figure we display OR export (§0.3/§5.3).
        The NGS projection is a STREHL-axis overlay, so it is skipped in fwhm
        mode (where axes[0] is in mas); the MODIFIED BUDGET box still shows."""
        self._annotate_modified(fig_main, offsets)
        if args.report != "fwhm":
            delta_var = self._ngs_delta_var(offsets, args)
            self._overlay_ngs_projection(fig_main, res, delta_var)

    # ---- lazy terms figure --------------------------------------------------
    def _terms_tab_visible(self):
        return self.plot_tabs.currentIndex() == 2

    def _on_plot_tab_changed(self, _idx):
        self._render_terms_if_visible()
        self._render_field_map_if_visible()

    # ---- field map ----------------------------------------------------------
    def _render_terms_if_visible(self):
        """Render the (expensive, ~570 ms) terms figure only when its tab is
        actually showing and the cached one is stale. Under the Prediction
        tab's scenario the tab instead shows the SNAPSHOT term breakdown
        (render_predicted_terms_figure) -- which also works with no run
        loaded, via the same _fm_args/_fm_prep surrogates the field map
        uses (2026-08-12)."""
        if not (self._terms_dirty and self._terms_tab_visible()):
            return
        if getattr(self, "pred_enable", None) is not None \
                and self.pred_enable.isChecked():
            self._render_predicted_terms()
            return
        if self.prep is None or self.res is None:
            # restore the placeholder (a predicted-terms figure may be
            # showing from before the scenario was toggled off)
            self._show_placeholder(
                self._terms_holder,
                "Error-terms figure appears here after a run with MASS "
                "data — or enable the Prediction tab's scenario.")
            self._terms_dirty = False
            return
        args, offsets = self.args_cached, self.last_offsets
        try:
            with engine.budget_overrides(**offsets):
                fig_terms = engine.render_terms_figure(args, self.prep, self.res)
                if fig_terms is not None and offsets:
                    self._annotate_modified(fig_terms, offsets, terms=True)
            if fig_terms is not None and self._utc():
                engine.apply_utc_display(fig_terms)
        except Exception as e:
            self._on_failed(f"terms render failed: {e}", "")
            return
        self.fig_terms = fig_terms
        self._terms_dirty = False
        if fig_terms is not None:
            self._show_figure(self._terms_holder, fig_terms)
            self.plot_tabs.setTabEnabled(2, True)
        else:
            self._show_placeholder(self._terms_holder,
                                   "No error-terms figure (no MASS profiles).")

    def _render_predicted_terms(self):
        """The Error-terms tab under a predicted scenario: single-snapshot
        term bars at the Prediction tab's conditions, for single-beacon and
        LTAO, honoring the WFE-slider overrides like every renderer."""
        args = self._fm_args()
        if args is None:
            self._show_placeholder(self._terms_holder,
                                   "Prediction scenario: fix the control "
                                   "inputs (a field could not be parsed).")
            return
        prep = self._fm_prep()
        snap = self._pred_snapshot()
        offsets = self.last_offsets
        try:
            with engine.budget_overrides(**offsets):
                fig = engine.render_predicted_terms_figure(
                    args, snap, prep._ltao_bw_fac, prep.lam_nm,
                    prep.lam_label)
                if offsets:
                    self._annotate_modified(fig, offsets, terms=True)
        except Exception as e:
            self._on_failed(f"predicted terms render failed: {e}", "")
            return
        self.fig_terms = fig
        self._terms_dirty = False
        self._show_figure(self._terms_holder, fig)
        self.plot_tabs.setTabEnabled(2, True)

    def _on_failed(self, msg, log):
        self.status.setText("ERROR: " + msg.replace("\n", " ")[:200])
        # a failed FIRST nighttime pull (typically: MKWC has no DIMM file for
        # tonight yet) disarms nighttime mode rather than re-hammering the
        # archive every 5 minutes -- see _on_nighttime_pull_failed
        if getattr(self, "nighttime_enable", None) is not None \
                and self.nighttime_enable.isChecked():
            self._on_nighttime_pull_failed()
        box = QtWidgets.QMessageBox(self)
        box.setIcon(QtWidgets.QMessageBox.Icon.Warning)
        box.setWindowTitle("Run failed")
        box.setText(msg)
        if log:
            box.setDetailedText(log)
        box.exec()

    def _on_worker_cleanup(self):
        self.worker = None
        self.spinner.setVisible(False)
        self._validate()

    # ---- canvas helpers -----------------------------------------------------
    def _make_canvas_tab(self, placeholder):
        """Build a plot tab: a matplotlib canvas + nav toolbar in a VBox,
        starting on a placeholder figure. Returns a holder dict whose canvas/
        navbar _show_figure() later rebinds."""
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        fig = Figure(figsize=(8, 9))
        ax = fig.add_subplot(111)
        ax.axis("off")
        ax.text(0.5, 0.5, placeholder, ha="center", va="center",
                fontsize=13, color="#555")
        canvas = FigureCanvas(fig)
        canvas.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding,
                             QtWidgets.QSizePolicy.Policy.Expanding)
        nav = NavigationToolbar(canvas, self)
        lay.addWidget(nav)
        lay.addWidget(canvas, 1)
        return {"widget": w, "layout": lay, "canvas": canvas, "navbar": nav}

    def _show_figure(self, holder, fig):
        """Point a holder's existing canvas at a freshly rendered Figure.

        Rebuilding FigureCanvas + NavigationToolbar per redraw cost ~52 ms and
        churned widgets (losing toolbar state); reassigning the figure on the
        SAME canvas costs ~17 ms and keeps the toolbar bound."""
        canvas, navbar = holder["canvas"], holder["navbar"]
        old_fig = canvas.figure

        # matplotlib >= 3.6 keeps the event CallbackRegistry on the FIGURE:
        # `canvas.callbacks` is a property returning `figure._canvas_callbacks`.
        # So assigning a new figure swaps in a fresh, EMPTY registry and orphans
        # every connection made against the old one -- including the navigation
        # toolbar's zoom/pan handlers, which silently stop firing. Carry the
        # registry across so zoom, pan and picking keep working after a redraw.
        cb = getattr(old_fig, "_canvas_callbacks", None)
        if cb is not None:
            fig._canvas_callbacks = cb

        canvas.figure = fig
        fig.set_canvas(canvas)

        # Raster at the widget's real pixel size. The engine builds figures at
        # print size (13 x ~12.9 in -> 1.68 Mpx); rasterizing that costs ~248 ms
        # while the canvas can only show ~0.8 Mpx. Matching the widget halves
        # the draw. Export re-renders a fresh figure at full size / 150 dpi, so
        # this only affects what is on screen.
        dpi = fig.get_dpi()
        w_px, h_px = canvas.width(), canvas.height()
        if w_px > 10 and h_px > 10:
            fig.set_size_inches(w_px / dpi, h_px / dpi, forward=False)

        # the nav stack still holds views of the OLD axes -> reset it, else
        # Home/Back/Forward restore limits from a figure that no longer exists.
        navbar.update()
        canvas.draw_idle()          # coalesce paints through the event loop

    @staticmethod
    def _sr_to_nm(sr, lam_nm):
        """Maréchal inverse: Strehl -> equivalent wavefront error (nm RMS).
        SR = exp(-(2π σ/λ)^2)  =>  σ = (λ/2π)·sqrt(-ln SR)."""
        sr = np.clip(np.asarray(sr, float), 1e-6, 1.0)
        return (lam_nm / (2.0 * np.pi)) * np.sqrt(-np.log(sr))

    @staticmethod
    def _nm_to_sr(nm, lam_nm):
        """Maréchal: wavefront error (nm RMS) -> Strehl."""
        return np.exp(-(2.0 * np.pi * np.asarray(nm, float) / lam_nm) ** 2)

    def _ngs_delta_var(self, offsets, args):
        """Signed change in NGS wavefront VARIANCE (nm²) implied by the
        off-default sliders.

        Each included term is ALREADY PRESENT inside the empirical NGS Strehl
        (the Gompertz fit is on-sky delivered performance, which already
        contains the DM fitting, bandwidth, measurement, static ... errors). So
        moving a slider from its reference value a to b SWAPS that term:

            sigma'^2 = sigma_ngs^2 - a^2 + b^2      -> contributes (b^2 - a^2)

        NOT sigma_ngs^2 + (b-a)^2, which bolts the change on as an extra
        independent error and badly under-predicts it (K2 fitting 60 -> 141 nm
        is a 127.6 nm quadrature swap, not 81 nm). The signed form also means
        LOWERING a slider correctly IMPROVES the projected NGS.

        Included terms: everything shared with NGS -- i.e. all but the
        LGS-specific ones (focal aniso / Na focus / LTAO tomography), the
        inactive telescope's DM fitting, and angular aniso unless the NGS is
        off-axis (ngs_offset > 0).
        """
        excluded = set(NGS_LGS_ONLY_TERMS)
        # only the ACTIVE telescope's per-telescope terms (DM fitting + static
        # telescope aberration) are relevant; the other telescope's are inert
        if args.telescope == "K1":
            excluded.update(("FITTING_ERR_K2", "STATIC_TEL_K2"))
        else:
            excluded.update(("FITTING_ERR_K1", "STATIC_TEL_K1"))
        if float(args.ngs_offset or 0.0) <= 0.0:
            excluded.add("ANG_REF")
        return float(sum(v ** 2 - engine.BUDGET_DEFAULTS[k] ** 2
                         for k, v in offsets.items() if k not in excluded))

    def _overlay_ngs_projection(self, fig, res, delta_var):
        """Project the swapped budget terms onto the NGS curve: turn each NGS
        Strehl into nm RMS (Maréchal), apply the signed variance change, turn it
        back into Strehl, and overlay it on the NGS panel. An explicit what-if,
        labelled with the effective quadrature nm (⊕ worse / ⊖ better)."""
        if delta_var == 0.0:
            return
        ax = fig.axes[0]                          # NGS/Strehl panel
        lam = self.prep.lam_nm
        sig = self._sr_to_nm(res.ngs_bright, lam)
        # clamp: a large reduction cannot drive the variance below zero
        sig_new = np.sqrt(np.maximum(sig ** 2 + delta_var, 0.0))
        proj = self._nm_to_sr(sig_new, lam)
        eff = float(np.sqrt(abs(delta_var)))
        sign = "⊕" if delta_var > 0 else "⊖"
        ax.plot(res.times, proj, ":", color="#C0392B", lw=1.8, zorder=6,
                label=f"projected NGS ({sign}{eff:.0f} nm quad)")
        ax.text(0.008, 0.86,
                f"projected NGS = Maréchal(NGS Strehl) with budget terms "
                f"swapped ({sign}{eff:.0f} nm in quadrature)",
                transform=ax.transAxes, color="#C0392B", fontsize=8.5,
                ha="left", va="top", zorder=1000)

    def _annotate_modified(self, fig, offsets, terms=False):
        """Draw the red MODIFIED BUDGET indicator (§5.3) top-left of the Strehl
        panel, plus the machine-readable budget_overrides= provenance as a small
        gray footer (§5.4), so a what-if figure/export can never be mistaken for
        the reference budget."""
        names = ", ".join(f"{k}={v:g}" for k, v in sorted(offsets.items()))
        txt = "MODIFIED BUDGET  (" + names + ")"
        footer = ("budget_overrides="
                  + ",".join(f"{k}={v:g}" for k, v in sorted(offsets.items())))
        if terms:
            fig.text(0.055, 0.965, txt, color="#C0392B", fontweight="bold",
                     fontsize=9, ha="left", va="top", zorder=1000)
        else:
            ax0 = fig.axes[0]
            ax0.text(0.008, 0.965, txt, transform=ax0.transAxes,
                     color="#C0392B", fontweight="bold", fontsize=10,
                     ha="left", va="top", zorder=1000,
                     bbox=dict(boxstyle="round,pad=0.25", fc="white",
                               ec="#C0392B", lw=1.0, alpha=0.9))
        # gray provenance footer (bottom of the figure), on both display+export
        fig.text(0.5, 0.004, footer, color="#777777", fontsize=7.5,
                 ha="center", va="bottom", zorder=1000)

    def _show_placeholder(self, holder, text):
        """Draw a plain message figure on a holder's canvas."""
        ph = Figure(figsize=(8, 9))
        ax = ph.add_subplot(111)
        ax.axis("off")
        ax.text(0.5, 0.5, text, ha="center", va="center",
                fontsize=13, color="#555")
        self._show_figure(holder, ph)

    # ---- exports: render/save on demand, under the active overrides ---------
    def on_export_png(self):
        """Export a PNG of the plot CURRENTLY SHOWING. On the Timeline tab
        that is the main figure re-rendered at export time, under the active
        WFE overrides (so it carries the MODIFIED BUDGET indicator, the NGS
        projection, and the provenance footer -- savefig at 150 dpi is
        ~0.7 s, exactly why it no longer sits in the live recompute path).
        On the Field map / Error terms tabs it is that tab's on-screen
        figure, saved as-is -- exporting the Timeline while looking at the
        field map was a genuine surprise (Eduardo, 2026-07-21)."""
        if self.res is None:
            return
        tab = self.plot_tabs.currentIndex()
        if tab in (1, 2):                     # Field map / Error terms
            holder, label, default = (
                (self._fm_holder, "field map", "ao_field_map.png") if tab == 1
                else (self._terms_holder, "error terms", "ao_terms.png"))
            dest, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, "Export PNG", default, "PNG (*.png)")
            if not dest:
                return
            try:
                holder["canvas"].figure.savefig(dest, dpi=150,
                                                bbox_inches="tight")
                self.status.setText(f"Exported {label} PNG -> {dest}")
            except Exception as e:
                self._on_failed(f"PNG export failed: {e}", "")
            return
        dest, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export PNG", "ao_strehl.png", "PNG (*.png)")
        if not dest:
            return
        args, offsets = self.args_cached, self.last_offsets
        self.status.setText("Exporting PNG…")
        QtWidgets.QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            with engine.budget_overrides(**offsets):
                fig = self._render_active_main(args, self.res)
                if offsets:
                    self._decorate_main(fig, self.res, offsets, args)
                if self._utc():
                    engine.apply_utc_display(fig)   # export follows display
                fig.savefig(dest, dpi=150, bbox_inches="tight")
            self.status.setText(f"Exported PNG -> {dest}")
        except Exception as e:
            self._on_failed(f"PNG export failed: {e}", "")
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

    def on_export_csv(self):
        """Write the CSV at export time, under the active WFE overrides so the
        engine's provenance line records them (§5.4)."""
        if self.res is None:
            return
        dest, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export CSV", "ao_strehl.csv", "CSV (*.csv)")
        if not dest:
            return
        args, offsets = self.args_cached, self.last_offsets
        try:
            with engine.budget_overrides(**offsets):
                engine.write_csv_table(args, self.prep, self.res, dest)
            self.status.setText(f"Exported CSV -> {dest}")
        except Exception as e:
            self._on_failed(f"CSV export failed: {e}", "")

    # ---- config save / load (JSON of all widget states) --------------------
    def _collect_config(self):
        wins = self._windows_hst()      # configs store HST canonically
        return {
            "mode": "fetch" if self.mode_fetch.isChecked() else "local",
            "fetch_date": self.fetch_date.date().toString("yyyyMMdd"),
            "dimm": self.dimm_edit.text(), "mass": self.mass_edit.text(),
            "masspro": self.masspro_edit.text(),
            "telescope": "K1" if self.tel_k1.isChecked() else "K2",
            "band": self.band_combo.currentText(),
            "wl_enable": self.wl_enable.isChecked(), "wl_nm": self.wl_nm.value(),
            "wind_ground": self.wind_ground.value(),
            "wind_free": self.wind_free.value(),
            "report": self.report_combo.currentText(),
            "fwhm_curves": self.fwhm_curves_combo.currentText(),
            "fwhm_box_mas": self.fwhm_box_mas.value(),
            "force": self.force_cb.isChecked(), "outdir": self.outdir_edit.text(),
            "target_enable": self.target_enable.isChecked(),
            "target_name": self.tname_edit.text(), "ra": self.ra_edit.text(),
            "dec": self.dec_edit.text(),
            "pm_ra": self.pmra_spin.value(), "pm_dec": self.pmdec_spin.value(),
            "windows": wins,
            "utc_times": self.utc_cb.isChecked(),
            "target_offset_cfg": self.target_offset.get_config(),
            "targets": list(self._targets),
            "za_enable": self.za_enable.isChecked(), "za": self.za_spin.value(),
            "ngs_bright": self.ngs_bright.value(), "ngs_faint": self.ngs_faint.value(),
            "ngs_offset": self.ngs_offset.value(),        # resolved (back-compat)
            "ngs_offset_cfg": self.ngs_offset.get_config(),
            "assumed_theta0": self.assumed_theta0.value(),
            "seeing_law": self.seeing_law.currentText(),
            "ngs_s0": self.ngs_s0.value(), "ngs_a": self.ngs_a.value(),
            "ngs_m0": self.ngs_m0.value(), "ngs_w": self.ngs_w.value(),
            "ngs_fit_tel": self._ngs_fit_tel,
            "k1_quadcell": self.k1_quadcell.value(),
            "tt_sensor": self.tt_sensor.currentText(),
            "tt_mag": self.tt_mag.value(), "tt_offset": self.tt_offset.value(),
            "tt_offset_cfg": self.tt_offset.get_config(),
            "laser_pa": self.laser_pa.value(),
            "stats_cond": self.stats_cond.currentText(),
            "stats_time": self.stats_time.time().toString("HH:mm"),
            "fm_mode": self.fm_mode.currentText(),
            "fm_metric": self.fm_metric.currentText(),
            "fm_cond": self.fm_cond.currentText(),
            "fm_osiris_mode": self.fm_osiris_mode.currentText(),
            "fm_osiris_scale": self.fm_osiris_scale.currentText(),
            "fm_osiris_lenslet": self.fm_osiris_lenslet.currentText(),
            "fm_nirc2_fov": self.fm_nirc2_fov.currentText(),
            "fm_for": self.fm_for.isChecked(),
            "fm_tss": self.fm_tss.isChecked(),
            "fm_pa": self.fm_pa.value(),
            "fm_bg_flip_x": self.fm_bg_flip_x.isChecked(),
            "fm_bg_flip_y": self.fm_bg_flip_y.isChecked(),
            "fm_fg_flip_x": self.fm_fg_flip_x.isChecked(),
            "fm_fg_flip_y": self.fm_fg_flip_y.isChecked(),
            "fm_img_pa": self.fm_img_pa.value(),
            "fm_survey": self.fm_sky.currentText(),
            "fm_frame_path": self._sky_local_path or "",
            "fm_backdrop_path": self._sky_bg_local_path or "",
            "fm_markers": [{"name": m["name"], "x": m["x"], "y": m["y"]}
                           for m in getattr(self, "_fm_markers", [])],
            "fm_catalog": self.fm_catalog.currentText(),
            "dark_theme": self.dark_action.isChecked(),
            "nirc2": {
                "path": self.n2_path.text(),
                "im1": self.n2_im1.value(), "nim": self.n2_nim.value(),
                "bg1": self.n2_bg1.value(), "nbg": self.n2_nbg.value(),
                "autofind": self.n2_autofind.isChecked(),
                "robust_sky": self.n2_robust_sky.isChecked(),
                "auto_radius": self.n2_auto_rad.isChecked(),
                "psf_clean": self.n2_psf_clean.isChecked(),
                "stretch": self.n2_stretch.currentText(),
                "white": self.n2_white.value(),
                "nstars": self.n2_nstars.value(),
                "map_metric": self.n2_map_metric.currentText(),
                "photrad": self.n2_photrad.value(),
                "bgin": self.n2_bgin.value(), "bgout": self.n2_bgout.value(),
                "peakrad": self.n2_peakrad.value(),
            },
            "lgs_offset_enable": self.lgs_offset_enable.isChecked(),
            "lgs_offset": self.lgs_offset.value(),
            "laser_fix_to_base": self.laser_fix_to_base.isChecked(),
            "ltao_floor": self.ltao_floor.value(),
            "ltao_tt_gain": self.ltao_tt_gain.value(),
            "legacy": self.legacy_cb.isChecked(),
            "tomo": self.tomo_combo.currentText(),
            "wfe": {name: r["spin"].value() for name, r in self.wfe_rows.items()},
            "prediction": {
                "enabled": self.pred_enable.isChecked(),
                "dimm": self.pred_dimm.value(), "mass": self.pred_mass.value(),
                "theta0": self.pred_theta0.value(),
                "theta0_auto": self.pred_theta0_auto.isChecked(),
                "za": self.pred_za.value(),
            },
        }

    def _apply_config(self, c):
        """Restore widget states from a config dict. Signals are blocked during
        the bulk apply so we recompute at most once at the end."""
        widgets = [self.fetch_date, self.dimm_edit, self.mass_edit,
                   self.masspro_edit, self.band_combo, self.wl_nm, self.wl_enable,
                   self.wind_ground, self.wind_free, self.report_combo,
                   self.fwhm_curves_combo, self.fwhm_box_mas,
                   self.ngs_bright, self.ngs_faint,
                   self.assumed_theta0, self.seeing_law,
                   self.ngs_s0, self.ngs_a, self.ngs_m0, self.ngs_w,
                   self.k1_quadcell, self.tt_sensor, self.tt_mag, self.laser_pa,
                   self.lgs_offset, self.ltao_floor, self.ltao_tt_gain,
                   self.tomo_combo, self.windows_list, self.za_spin,
                   self.za_enable, self.pred_enable, self.pred_theta0_auto,
                   self.fm_osiris_mode, self.fm_osiris_scale,
                   self.fm_osiris_lenslet, self.fm_nirc2_fov, self.fm_for,
                   self.fm_tss,
                   self.fm_pa, self.fm_bg_flip_x, self.fm_bg_flip_y,
                   self.fm_fg_flip_x, self.fm_fg_flip_y, self.fm_img_pa,
                   self.pmra_spin, self.pmdec_spin,
                   self.stats_cond, self.stats_time] + \
                  [r["spin"] for r in self.wfe_rows.values()] + \
                  [r["spin"] for r in self._pred_rows.values()]
        self._loading = True        # suppress live-update scheduling during apply
        for w in widgets:
            w.blockSignals(True)
        try:
            (self.mode_fetch if c.get("mode") == "fetch"
             else self.mode_local).setChecked(True)
            self.fetch_date.setDate(QtCore.QDate.fromString(
                c.get("fetch_date", "20260525"), "yyyyMMdd"))
            self.dimm_edit.setText(c.get("dimm", ""))
            self.mass_edit.setText(c.get("mass", ""))
            self.masspro_edit.setText(c.get("masspro", ""))
            (self.tel_k1 if c.get("telescope", "K2") == "K1"
             else self.tel_k2).setChecked(True)
            self.band_combo.setCurrentText(c.get("band", "K"))
            self.wl_enable.setChecked(c.get("wl_enable", False))
            self.wl_nm.setValue(c.get("wl_nm", engine.LAMBDA_K_NM))
            self.wl_nm.setEnabled(self.wl_enable.isChecked())
            self.wind_ground.setValue(c.get("wind_ground", self.defaults.wind_ground))
            self.wind_free.setValue(c.get("wind_free", self.defaults.wind_free))
            self.report_combo.setCurrentText(c.get("report", "Strehl"))
            self.fwhm_curves_combo.setCurrentText(c.get("fwhm_curves", "half-max"))
            self.fwhm_box_mas.setValue(c.get("fwhm_box_mas", 300.0))
            self.force_cb.setChecked(c.get("force", True))
            self.outdir_edit.setText(c.get("outdir", self._tmpdir))
            self.target_enable.setChecked(c.get("target_enable", False))
            self.tname_edit.setText(c.get("target_name", ""))
            self.ra_edit.setText(c.get("ra", ""))
            self.dec_edit.setText(c.get("dec", ""))
            self.pmra_spin.setValue(c.get("pm_ra", 0.0))
            self.pmdec_spin.setValue(c.get("pm_dec", 0.0))
            if "target_offset_cfg" in c:
                self.target_offset.set_config(c["target_offset_cfg"])
            self._targets = [dict(t) for t in c.get("targets", [])]
            self._refresh_target_combo()
            # windows arrive in HST (the config canon): drop to HST display
            # mode SILENTLY first, load them, and re-apply the saved UTC
            # preference at the end (unblocked, so the toggle converts the
            # freshly-loaded texts for display)
            self.utc_cb.blockSignals(True)
            self.utc_cb.setChecked(False)
            self.utc_cb.blockSignals(False)
            self.windows_row_label.setText("Windows (HST):")
            self.windows_list.clear()
            for wtxt in c.get("windows", []):
                self._add_window_item(wtxt)
            self.utc_cb.setChecked(c.get("utc_times", False))
            self.za_enable.setChecked(c.get("za_enable", False))
            self.za_spin.setValue(c.get("za", 0.0))
            self.za_spin.setEnabled(self.za_enable.isChecked())
            self.ngs_bright.setValue(c.get("ngs_bright", self.defaults.ngs_bright))
            self.ngs_faint.setValue(c.get("ngs_faint", self.defaults.ngs_faint))
            if "ngs_offset_cfg" in c:
                self.ngs_offset.set_config(c["ngs_offset_cfg"])
            else:
                self.ngs_offset.setValue(c.get("ngs_offset", 0.0))
            self.assumed_theta0.setValue(c.get("assumed_theta0", 15.0))
            self.seeing_law.setCurrentText(c.get("seeing_law", "kolmogorov"))
            # Gompertz fit: seed from the (already-set) telescope's default,
            # then overlay any saved values. Pin _ngs_fit_tel so the end-of-
            # apply _sync does not repopulate over these.
            _tel = "K1" if self.tel_k1.isChecked() else "K2"
            _par = engine.NGS_PARAMS[_tel]
            self._ngs_fit_tel = c.get("ngs_fit_tel", _tel)
            self.ngs_s0.setValue(c.get("ngs_s0", _par["S0"]))
            self.ngs_a.setValue(c.get("ngs_a", _par["A"]))
            self.ngs_m0.setValue(c.get("ngs_m0", _par["m0"]))
            self.ngs_w.setValue(c.get("ngs_w", _par["w"]))
            self.k1_quadcell.setValue(c.get("k1_quadcell",
                                            self.defaults.k1_quadcell_penalty))
            self.laser_pa.setValue(
                c.get("laser_pa", engine.DEF_LASER_PA_DEG))
            self.stats_cond.setCurrentText(c.get("stats_cond", "observing window"))
            self.stats_time.setTime(QtCore.QTime.fromString(
                c.get("stats_time", "20:00"), "HH:mm"))
            self.stats_time.setEnabled(self.stats_cond.currentText() == "specific time")
            self.fm_mode.setCurrentText(c.get("fm_mode", self.fm_mode.currentText()))
            self.fm_metric.setCurrentText(c.get("fm_metric", "Strehl"))
            self.fm_cond.setCurrentText(c.get("fm_cond", "observing window"))
            self.fm_osiris_mode.setCurrentText(
                c.get("fm_osiris_mode", self.fm_osiris_mode.currentText()))
            self.fm_osiris_scale.setCurrentText(
                c.get("fm_osiris_scale", self.fm_osiris_scale.currentText()))
            self.fm_osiris_lenslet.setCurrentText(
                c.get("fm_osiris_lenslet", self.fm_osiris_lenslet.currentText()))
            self.fm_nirc2_fov.setCurrentText(
                c.get("fm_nirc2_fov", self.fm_nirc2_fov.currentText()))
            self.fm_for.setChecked(c.get("fm_for", False))
            self.fm_tss.setChecked(c.get("fm_tss", False))
            self.fm_pa.setValue(c.get("fm_pa", 0.0))
            self.fm_bg_flip_x.setChecked(c.get("fm_bg_flip_x", False))
            self.fm_bg_flip_y.setChecked(c.get("fm_bg_flip_y", False))
            self.fm_fg_flip_x.setChecked(c.get("fm_fg_flip_x", False))
            self.fm_fg_flip_y.setChecked(c.get("fm_fg_flip_y", False))
            self.fm_img_pa.setValue(c.get("fm_img_pa", 0.0))
            self.fm_catalog.setCurrentText(
                c.get("fm_catalog", self.fm_catalog.currentText()))
            # dark theme is a static preference (unlike nighttime mode, which
            # is an action and stays out of configs). The action is NOT in the
            # blocked-widgets list, so setChecked applies the theme right here;
            # a config's value counts as a user choice (_dark_auto stays off).
            self.dark_action.setChecked(c.get("dark_theme", False))
            n2 = c.get("nirc2", {})
            self.n2_path.setText(n2.get("path", ""))
            self.n2_im1.setValue(int(n2.get("im1", 1)))
            self.n2_nim.setValue(int(n2.get("nim", 1)))
            self.n2_bg1.setValue(int(n2.get("bg1", 2)))
            self.n2_nbg.setValue(int(n2.get("nbg", 0)))
            self.n2_autofind.setChecked(bool(n2.get("autofind", True)))
            self.n2_robust_sky.setChecked(bool(n2.get("robust_sky", False)))
            self.n2_auto_rad.setChecked(bool(n2.get("auto_radius", False)))
            self.n2_psf_clean.setChecked(bool(n2.get("psf_clean", False)))
            self.n2_stretch.setCurrentText(n2.get("stretch", "IDL ±5σ"))
            self.n2_white.setValue(n2.get("white", 99.5))
            self.n2_nstars.setValue(int(n2.get("nstars", 5)))
            self.n2_map_metric.setCurrentText(n2.get("map_metric", "SR"))
            self.n2_photrad.setValue(n2.get(
                "photrad", engine.NIRC2_PHOTOMETRY_RADIUS_ARCSEC))
            self.n2_bgin.setValue(n2.get(
                "bgin", engine.NIRC2_BG_INNER_RADIUS_ARCSEC))
            self.n2_bgout.setValue(n2.get(
                "bgout", engine.NIRC2_BG_OUTER_RADIUS_ARCSEC))
            self.n2_peakrad.setValue(n2.get(
                "peakrad", engine.NIRC2_PEAK_RADIUS_ARCSEC))
            self._fm_markers = [{"name": m.get("name", f"T{i+1}"),
                                 "x": float(m.get("x", 0.0)),
                                 "y": float(m.get("y", 0.0)), "val": None}
                                for i, m in enumerate(c.get("fm_markers", []))]
            self._pending_frame_path = c.get("fm_frame_path", "")
            self._pending_survey = c.get("fm_survey", "off")
            self._pending_backdrop_path = c.get("fm_backdrop_path", "")
            _tts = c.get("tt_sensor", "STRAP (R)")
            if self.tt_sensor.findText(_tts) < 0:
                _tts = "STRAP (R)"   # e.g. the retired "STRAP legacy (R)"
            self.tt_sensor.setCurrentText(_tts)
            self.tt_mag.setValue(c.get("tt_mag", self.defaults.tt_mag))
            if "tt_offset_cfg" in c:
                self.tt_offset.set_config(c["tt_offset_cfg"])
            else:
                self.tt_offset.setValue(c.get("tt_offset", self.defaults.tt_offset))
            self.lgs_offset_enable.setChecked(c.get("lgs_offset_enable", False))
            self.lgs_offset.setValue(c.get("lgs_offset", 0.0))
            self.lgs_offset.setEnabled(self.lgs_offset_enable.isChecked())
            # fires naturally (not signal-blocked): by this point RA/Dec,
            # target offset, and laser_pa/lgs_offset above are all already
            # restored, so re-anchoring here reads the correct final state
            self.laser_fix_to_base.setChecked(c.get("laser_fix_to_base", False))
            self.ltao_floor.setValue(c.get("ltao_floor",
                                           self.defaults.ltao_bw_floor_frac))
            self.ltao_tt_gain.setValue(c.get(
                "ltao_tt_gain", self.defaults.ltao_tt_theta0_gain))
            self.legacy_cb.setChecked(c.get("legacy", False))
            self.tomo_combo.setCurrentText(c.get("tomo", "auto (per telescope)"))
            for name, val in c.get("wfe", {}).items():
                if name in self.wfe_rows:
                    self.wfe_rows[name]["spin"].setValue(val)
                    self.wfe_rows[name]["slider"].setValue(
                        int(round(val * self.wfe_rows[name]["scale"])))
            pc = c.get("prediction", {})
            for key, dflt in (("dimm", engine.REF_TOTAL),
                              ("mass", engine.REF_FREEATM),
                              ("theta0", None), ("za", 0.0)):
                r = self._pred_rows[key]
                val = pc.get(key, dflt)
                if val is None:                    # theta0: profile-derived
                    val = engine.synthetic_field_snapshot(
                        pc.get("dimm", engine.REF_TOTAL),
                        pc.get("mass", engine.REF_FREEATM))["theta0_k_zenith"]
                r["spin"].setValue(val)
                r["slider"].setValue(int(round(val * r["scale"])))
            self.pred_theta0_auto.setChecked(pc.get("theta0_auto", True))
            self.pred_enable.setChecked(pc.get("enabled", False))
        finally:
            for w in widgets:
                w.blockSignals(False)
            self._loading = False
        self._on_data_mode()
        self._on_target_toggle()
        self._sync_ngs_fit_fields()
        self._update_wfe_summary()
        self._sync_pred_ui()
        self._update_pred_profile_plot()
        self._sync_fm_fov_controls()
        # restore the sky layers: re-load the frame FITS (defines the centre),
        # then the survey (centred on it). Done after the bulk apply so neither
        # fires during it.
        self._sky_fg_img = None; self._sky_local_path = None; self._sky_center = None
        self._sky_bg_img = None; self._sky_key = None; self._sky_bg_local_path = None
        fp = getattr(self, "_pending_frame_path", "")
        if fp and os.path.exists(fp):
            self._load_local_sky(fp)
        survey = getattr(self, "_pending_survey", "off")
        self.fm_sky.blockSignals(True)
        self.fm_sky.setCurrentText(survey)
        self.fm_sky.blockSignals(False)
        bp = getattr(self, "_pending_backdrop_path", "")
        if survey == LOCAL_BACKDROP and bp and os.path.exists(bp):
            self._load_bg_local(bp)              # local wide-FITS backdrop
        elif survey in HIPS_SURVEYS:
            self._load_sky(force=True)           # online survey backdrop
        self._update_sky_status()
        self._validate()
        if self.prep is not None:
            self.recompute_and_draw()

    def on_save_config(self):
        dest, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save config", "ao_strehl_config.json", "JSON (*.json)")
        if not dest:
            return
        import json
        with open(dest, "w") as fh:
            json.dump(self._collect_config(), fh, indent=2)
        self.status.setText(f"Saved config -> {dest}")

    def on_load_config(self):
        src, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load config", "", "JSON (*.json)")
        if not src:
            return
        import json
        try:
            with open(src) as fh:
                c = json.load(fh)
            self._apply_config(c)
            self.status.setText(f"Loaded config <- {src}")
        except Exception as e:
            self._on_failed(f"config load failed: {e}", "")

    # ---- theme --------------------------------------------------------------
    def _on_dark_toggled(self, checked):
        """View ▸ Dark theme. A HUMAN toggle (not one _dark_syncing wraps)
        takes ownership: Nighttime mode then stops auto-managing the theme
        for the rest of the session (see nighttime.py)."""
        apply_theme(QtWidgets.QApplication.instance(), dark=checked)
        if not self._dark_syncing:
            self._dark_auto = False

    def _sync_dark(self, checked):
        """Programmatically flip the Dark-theme action (Nighttime mode's
        auto-switch) without it reading as a user override."""
        self._dark_syncing = True
        try:
            self.dark_action.setChecked(checked)
        finally:
            self._dark_syncing = False

    # ---- presets ------------------------------------------------------------
    def _preset_reference_budget(self):
        """Reset every WFE slider to its reference value (clears MODIFIED
        BUDGET)."""
        self._reset_all_wfe()

    def _preset_last_night(self):
        """Switch to fetch mode for the most recent completed night. MKWC files
        are stamped by the morning/UT date, so use *UT* yesterday, not the local
        civil date: e.g. an HST evening that is already the next UT day loads
        that UT date minus one (evening of 2026-07-09 HST = 2026-07-10 UT ->
        20260709), which the local civil date would get wrong."""
        self.mode_fetch.setChecked(True)
        ut_yesterday = QtCore.QDateTime.currentDateTimeUtc().date().addDays(-1)
        self.fetch_date.setDate(ut_yesterday)
        self._on_data_mode()
        self.status.setText(
            f"Preset: fetch {ut_yesterday.toString('yyyyMMdd')} "
            f"(UT yesterday) — press Run")

    def _preset_tonight(self):
        """Switch to fetch mode for the CURRENT UT date -- tonight's night,
        while it is still in progress (or about to start). Same MKWC
        stamping convention as _preset_last_night, one day later: no
        adjustment needed since "today" in UT already IS the label an
        evening HST session's data will be filed under."""
        self.mode_fetch.setChecked(True)
        ut_today = QtCore.QDateTime.currentDateTimeUtc().date()
        self.fetch_date.setDate(ut_today)
        self._on_data_mode()
        self.status.setText(
            f"Preset: fetch {ut_today.toString('yyyyMMdd')} "
            f"(UT today) — press Run")


    # ---- Help / documentation -----------------------------------------------
    def _open_doc(self, filename):
        """Open a bundled documentation PDF in the system viewer."""
        from qtcompat import QtGui, QtCore
        path = _bundled_doc(filename)
        if not path:
            QtWidgets.QMessageBox.warning(
                self, "Document not found",
                f"Could not locate the bundled document:\n{filename}")
            return
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(path))

    # ---- About / credits ----------------------------------------------------
    def _about_html(self):
        """Credits, acknowledgments, brief instructions and limitations, shown
        in the About dialog (at startup and from Help ▸ About)."""
        return f"""
        <h2 style="margin-bottom:2px">{APP_NAME}</h2>
        <div style="color:#666">version {__version__}</div>
        <p>Developed by the <b>{ORGANIZATION}</b>. Distributed under the
        project's existing licence (GNU General Public License v3.0) — see the
        LICENSE file. This tool is provided as-is, with no warranty.</p>
        <p><b>Questions and comments</b> should be directed to the
        <b>W. M. Keck Observatory</b>, where <b>{MAINTAINER}</b> is the current
        maintainer.</p>

        <h3>Getting started</h3>
        <ul>
          <li><b>Atmosphere:</b> fetch a night's MASS/DIMM seeing from the
              Mauna Kea Weather Center, or load local <tt>.dat</tt> profiles.</li>
          <li>Pick the <b>telescope</b> (K1/K2), enter the <b>target</b>, and set
              the guide-star / tip-tilt configuration, then press <b>Run</b> for
              the Strehl (or FWHM) timeline.</li>
          <li>The <b>Field map</b> tab maps performance across the patrol field;
              overlay a DSS/2MASS survey or a local FITS as a backdrop, and
              inscribe an OSIRIS/NIRC2 science frame at its true size.</li>
          <li>The <b>WFE sliders</b> tab explores the error budget; any modified
              budget is clearly flagged on the plot and in exports.</li>
        </ul>
        <p>A full <b>user manual</b> and the <b>technical note (KAON 1542)</b>
        are available from the <b>Help</b> menu.</p>

        <h3>Limitations</h3>
        <ul>
          <li>This is a <b>semi-analytical error-budget estimator</b> calibrated
              to specific Keck on-sky data — not an end-to-end AO simulation.</li>
          <li>Strehl uses the extended <b>Maréchal approximation</b>; it is most
              reliable at moderate-to-high Strehl and less so in poor
              conditions / very low Strehl.</li>
          <li><b>NGS</b> performance comes from empirical fits anchored at
              K-band; extrapolation to other wavelengths/magnitudes is
              approximate.</li>
          <li>The NGS Gompertz fit is calibrated to <b>optimized</b> on-sky
              data, typically taken after a Fast &amp; Furious (F&amp;F) run —
              worth about 10 K-band Strehl points. To estimate performance
              <b>without</b> F&amp;F, lower the fit ceiling S₀ by 0.1 (NGS
              tab).</li>
          <li>Tip-tilt (STRAP/TRICK) and static/calibration terms derive partly
              from the KAON 1542 budget sheet; some allocations are not yet
              fully on-sky validated.</li>
          <li>Atmospheric scaling assumes reference seeing profiles; results
              depend on the quality of the MASS/DIMM inputs.</li>
          <li>Sky overlays are placed from each frame's WCS (with
              instrument-specific parity handling, e.g. GSAOI) — always verify
              orientation against known sources.</li>
          <li>Outputs are estimates for planning; verify against delivered
              on-sky performance.</li>
        </ul>

        <h3>Acknowledgments</h3>
        <p>The baseline analytical AO error budget used for the LGS modes was
        developed by <b>Richard G. Dekany</b> (California Institute of
        Technology).</p>
        <p>The <b>Measured SR</b> tab's photometry was ported from the IDL
        Strehl-ratio tool originally written for Keck by
        <b>Marcos van Dam</b>.</p>
        <p>The W. M. Keck Observatory is operated as a scientific partnership
        among the California Institute of Technology, the University of
        California, and the National Aeronautics and Space Administration. The
        Observatory was made possible by the generous financial support of the
        W. M. Keck Foundation.</p>
        <p>This tool was partially funded by the HAKA project, an NSF Major
        Research Instrumentation Program award <b>AST-2320038</b>.</p>
        """

    def _show_about(self, at_startup=False):
        """Modal About dialog. At startup it is skipped if the user has ticked
        'don't show again' (persisted via QSettings); the checkbox is only
        offered in the startup presentation."""
        from qtcompat import QtCore as _QtCore
        settings = _QtCore.QSettings("WMKO", "KeckAOPerformanceEstimator")
        if at_startup and settings.value("hide_about_at_startup", False, type=bool):
            return
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle(f"About {APP_NAME}")
        dlg.resize(560, 640)
        lay = QtWidgets.QVBoxLayout(dlg)
        view = QtWidgets.QTextBrowser()
        view.setOpenExternalLinks(True)
        view.setHtml(self._about_html())
        lay.addWidget(view, 1)
        row = QtWidgets.QHBoxLayout()
        if at_startup:
            chk = QtWidgets.QCheckBox("Don't show this at startup")
            row.addWidget(chk)
        row.addStretch(1)
        ok = QtWidgets.QPushButton("Close")
        ok.clicked.connect(dlg.accept)
        row.addWidget(ok)
        lay.addLayout(row)
        dlg.exec()
        if at_startup:
            settings.setValue("hide_about_at_startup", chk.isChecked())

