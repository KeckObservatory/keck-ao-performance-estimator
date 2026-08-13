"""Data tab: input-file source (fetch-by-date vs local files), telescope,
science wavelength, wind speeds (manual, or fetched from the GFS analysis),
and output-directory controls.
"""
import numpy as np

from qtcompat import QtCore, QtWidgets

import keck_ao_estimator as engine

from ..theme import set_cue
from ..widgets import _dspin
from ..workers import GfsWindsWorker


class DataTabMixin:
    def _tab_data(self):
        w = QtWidgets.QWidget()
        f = QtWidgets.QFormLayout(w)

        # fetch-by-date vs local files
        self.mode_fetch = QtWidgets.QRadioButton("Fetch by date (MKWC)")
        self.mode_local = QtWidgets.QRadioButton("Local files")
        # floors, not natural text widths: this row otherwise sets the whole
        # tab's minimum (radio min hint = full text width; see 631045c)
        self.mode_fetch.setMinimumWidth(110)
        self.mode_local.setMinimumWidth(80)
        self.mode_local.setChecked(True)
        self.mode_fetch.toggled.connect(self._on_data_mode)
        mode_row = QtWidgets.QHBoxLayout()
        mode_row.addWidget(self.mode_fetch)
        mode_row.addWidget(self.mode_local)
        f.addRow("Source:", self._wrap(mode_row))

        self.fetch_date = QtWidgets.QDateEdit()
        self.fetch_date.setDisplayFormat("yyyyMMdd")
        self.fetch_date.setCalendarPopup(True)
        self.fetch_date.setDate(QtCore.QDate(2026, 5, 25))
        f.addRow("Fetch date (UT):", self.fetch_date)

        # UTC mode: every DISPLAYED and ENTERED wall time switches between
        # HST (the tool's native/internal timebase -- engine math never
        # changes) and UTC (+10 h, purely at the display/input boundary):
        # plot time axes and annotations, the observing-windows list, the
        # field-map/summary "specific time" fields, and every status label.
        self.utc_cb = QtWidgets.QCheckBox("show / enter times in UTC")
        self.utc_cb.setMinimumWidth(110)   # floor, not text width (631045c)
        self.utc_cb.setToolTip(
            "Display and enter ALL wall times in UTC instead of HST "
            "(UTC = HST + 10 h): plot axes and window annotations, the "
            "Target tab's observing windows, the field-map and summary "
            "'specific time' fields, and the status readouts. Internally "
            "everything stays HST — this is a display/entry convention, "
            "not a recompute.")
        self.utc_cb.toggled.connect(self._on_utc_toggled)
        f.addRow("Times:", self.utc_cb)

        self._init_nighttime_mode(f)          # NighttimeModeMixin

        self.dimm_edit, dimm_row = self._file_picker("DIMM .dat")
        self.mass_edit, mass_row = self._file_picker("MASS .dat")
        self.masspro_edit, masspro_row = self._file_picker("MASSPRO .dat")
        f.addRow("DIMM file:", dimm_row)
        f.addRow("MASS file:", mass_row)
        f.addRow("MASSPRO file:", masspro_row)

        # telescope radios
        self.tel_k1 = QtWidgets.QRadioButton("K1")
        self.tel_k2 = QtWidgets.QRadioButton("K2")
        (self.tel_k1 if self.defaults.telescope == "K1"
         else self.tel_k2).setChecked(True)
        tel_row = QtWidgets.QHBoxLayout()
        tel_row.addWidget(self.tel_k1)
        tel_row.addWidget(self.tel_k2)
        f.addRow("Telescope:", self._wrap(tel_row))

        # science wavelength: band combo + optional explicit nm
        self.band_combo = QtWidgets.QComboBox()
        self.band_combo.addItems(list(engine.PHOTOMETRIC_BANDS.keys()))
        self.band_combo.setCurrentText("K")
        f.addRow("Science band:", self.band_combo)
        self.wl_enable = QtWidgets.QCheckBox("override with exact nm")
        self.wl_enable.setMinimumWidth(110)   # floor, not text width (631045c)
        self.wl_nm = _dspin(300, 30000, 10, engine.LAMBDA_K_NM, 0, " nm")
        self.wl_nm.setMinimumWidth(70)   # floor (631045c)
        self.wl_nm.setEnabled(False)
        self.wl_enable.toggled.connect(self.wl_nm.setEnabled)
        wl_row = QtWidgets.QHBoxLayout()
        wl_row.addWidget(self.wl_enable)
        wl_row.addWidget(self.wl_nm)
        f.addRow("Wavelength:", self._wrap(wl_row))

        # wind speeds (m/s) for the wind-weighted bandwidth term. We can't read
        # the true profile, so let the user supply a night estimate (e.g. GFS).
        self.wind_ground = _dspin(0, 100, 1, self.defaults.wind_ground, 1, " m/s")
        self.wind_free = _dspin(0, 150, 1, self.defaults.wind_free, 1, " m/s")
        self.wind_ground.setToolTip("boundary-layer (ground) wind; default 8 m/s")
        self.wind_free.setToolTip("free-atmosphere (jet) wind; default 25 m/s")
        # own label widgets so they can carry width floors -- these are the
        # widest labels on the tab and the form minimum is
        # max(label col) + max(field col) (631045c)
        for _txt, _w in (("Wind ground (BL):", self.wind_ground),
                         ("Wind free-atm (jet):", self.wind_free)):
            _lbl = QtWidgets.QLabel(_txt)
            _lbl.setMinimumWidth(130)
            f.addRow(_lbl, _w)

        # real winds instead of the 8/25 guesses: one bounded GFS fetch
        # (engine.night_winds -- hard 10 s timeout, no retries, cached per UT
        # date) collapsed to the two representative speeds bw_wind_scale
        # takes. Fills the two spinboxes above; a failed fetch changes
        # nothing and says why.
        self.winds_fetch_btn = QtWidgets.QPushButton("Fetch winds (GFS)")
        self.winds_fetch_btn.setMinimumWidth(90)   # floor (631045c)
        self.winds_fetch_btn.setToolTip(
            "Fetch the night's GFS wind profile over the summit (Open-Meteo "
            "analysis, validated against the Hilo radiosonde) and set the "
            "two wind fields above: ground = summit-level wind, free-atm = "
            "Cn²-weighted wind over the six MASS layer altitudes (the "
            "night's own median profile when data is loaded). Bounded: one "
            "request, 10 s timeout, no retries.")
        self.winds_fetch_btn.clicked.connect(self._fetch_gfs_winds)
        # status on its OWN full-width wrapped row, zero width floor -- NOT
        # inline after the button, where it only gets the leftover width and
        # visibly clips (the same lesson the nighttime status taught twice:
        # wrap, don't clip; guarded in gui_phase27 like gui_phase24 does)
        self.winds_status = QtWidgets.QLabel("")
        self.winds_status.setWordWrap(True)
        self.winds_status.setMinimumWidth(0)
        set_cue(self.winds_status, "secondary")
        # the live Figure-3 remake (FaGeometryMixin): pierce points, monitor
        # circles/cone, wind vectors, lead/lag -- display-only dialog
        self.fa_geometry_btn = QtWidgets.QPushButton("FA geometry…")
        self.fa_geometry_btn.setMinimumWidth(90)   # floor (631045c)
        self.fa_geometry_btn.setToolTip(
            "Plan + side view of the monitor's and the beam's pierce points "
            "at the MASS layer altitudes, with each layer's GFS wind vector "
            "and the resulting Keck-vs-monitor event lead/lag — the live "
            "version of the 2026-07-21 FA-event note's Figure 3.")
        self.fa_geometry_btn.clicked.connect(self._show_fa_geometry)
        winds_row = QtWidgets.QHBoxLayout()
        winds_row.addWidget(self.winds_fetch_btn)
        winds_row.addWidget(self.fa_geometry_btn)
        winds_row.addStretch(1)
        f.addRow("", self._wrap(winds_row))
        f.addRow("", self.winds_status)
        self._winds_worker = None
        self._gfs_winds_result = None    # last successful night_winds() dict
        self._fa_geo_dialog = None

        # FA timing advisory (fa_advisory.py; display-only, deliberately
        # changes NOTHING about the timeline or the estimate): how to READ
        # the MASS feed -- trailing-window range + event-in-progress cue,
        # and, once winds are fetched, the per-layer Keck-vs-monitor
        # lead/lag the 2026-07-21 FA event demonstrated. Own wrapped
        # full-width row, zero width floor (the nighttime-status lesson).
        self.fa_advisory = QtWidgets.QLabel("—")
        self.fa_advisory.setWordWrap(True)
        self.fa_advisory.setMinimumWidth(0)
        set_cue(self.fa_advisory, "secondary")
        self.fa_advisory.setToolTip(
            "The MASS/DIMM and the telescope pierce a moving turbulent "
            "layer kilometers apart, so a discrete FA event can reach the "
            "beam tens of minutes before or after the monitor records it "
            "(2026-07-21: Keck led by 22-35 min). This advisory summarizes "
            "the trailing FA window and, with GFS winds fetched, the "
            "per-layer expected lead (+ = Keck first; ± = the monitor's "
            "own unknown pierce point). It never alters the estimate.")
        f.addRow("FA advisory:", self.fa_advisory)

        self.force_cb = QtWidgets.QCheckBox("force-overwrite outputs")
        self.force_cb.setMinimumWidth(110)   # floor, not text width (631045c)
        self.force_cb.setChecked(True)   # GUI writes to a temp dir; safe default
        f.addRow("", self.force_cb)

        self.outdir_edit, outdir_row = self._file_picker("output dir", directory=True)
        self.outdir_edit.setText(self._tmpdir)
        f.addRow("Output dir:", outdir_row)

        self._init_summary_stats(f)           # SummaryStatsMixin

        # telescope / band / wavelength feed prepare_night -> live re-prepare.
        # (Data-source, file paths and fetch date stay manual-Run: they can be
        # mid-edit and, for fetch, may hit the network.)
        self.tel_k1.toggled.connect(self._on_prep_changed)
        self.band_combo.currentTextChanged.connect(self._on_prep_changed)
        self.wl_enable.toggled.connect(self._on_prep_changed)
        self.wl_nm.valueChanged.connect(self._on_prep_changed)
        # winds feed the bandwidth term in compute_timeline -> compute-only
        self.wind_ground.valueChanged.connect(self._on_compute_changed)
        self.wind_free.valueChanged.connect(self._on_compute_changed)

        self._on_data_mode()
        return self._scroll(w)

    def _on_data_mode(self):
        fetch = self.mode_fetch.isChecked()
        self.fetch_date.setEnabled(fetch)
        for e in (self.dimm_edit, self.mass_edit, self.masspro_edit):
            e.setEnabled(not fetch)
        self._validate()

    # ---- UTC display/entry mode -----------------------------------------
    def _utc(self):
        return self.utc_cb.isChecked()

    def _fmt_hm(self, dt, seconds=False):
        """A wall-clock label for the HST datetime `dt`, honouring UTC
        mode. All internal datetimes stay HST; only labels convert."""
        fmt = "%H:%M:%S" if seconds else "%H:%M"
        if self._utc():
            from datetime import timedelta
            return f"{dt + timedelta(hours=10):{fmt}} UTC"
        return f"{dt:{fmt}} HST"

    def _tz_text(self, s):
        """Convert an engine-generated 'HH:MM ... HST' display string to
        UTC when UTC mode is on (engine.shift_hst_text), else pass it
        through untouched."""
        return engine.shift_hst_text(s) if self._utc() else s

    @staticmethod
    def _shift_hhmm_range(txt, hours):
        """'HH:MM-HH:MM' shifted by `hours` (mod 24). Unparseable text is
        returned unchanged -- validation happens elsewhere."""
        try:
            a, b = txt.strip().split("-", 1)

            def sh(t):
                h, m = (int(x) for x in t.strip().split(":"))
                return f"{int((h + hours) % 24):02d}:{m:02d}"
            return f"{sh(a)}-{sh(b)}"
        except (ValueError, AttributeError):
            return txt

    def _windows_hst(self):
        """The observing-windows list as HST 'HH:MM-HH:MM' strings -- the
        ONLY form the engine and saved configs ever see. In UTC mode the
        list displays (and the user edits) UT, so convert back here."""
        wins = [self.windows_list.item(i).text()
                for i in range(self.windows_list.count())]
        if self._utc():
            wins = [self._shift_hhmm_range(w, -10) for w in wins]
        return wins

    def _on_utc_toggled(self, on):
        """Switch every displayed/entered wall time between HST and UTC.
        The observing-windows LIST TEXT is converted in place (the list is
        an entry surface, not just a display); collect_args converts back
        to HST for the engine. Statuses/figures refresh via the normal
        recompute path (skipped while a config load is mid-apply)."""
        if getattr(self, "windows_row_label", None) is None:
            return                       # Target tab not built yet
        self.windows_row_label.setText("Windows (UT):" if on
                                       else "Windows (HST):")
        shift = 10 if on else -10
        for i in range(self.windows_list.count()):
            it = self.windows_list.item(i)
            it.setText(self._shift_hhmm_range(it.text(), shift))
        if getattr(self, "_loading", False):
            return
        self._update_nighttime_status()
        self._update_fa_advisory()
        if self.prep is not None:
            self.recompute_and_draw()

    # ---- GFS winds fetch -----------------------------------------------
    def _gfs_winds_ymd(self):
        """The UT date stamp to fetch winds for: the PREPARED night's own
        stamp when a night is loaded (matches whatever data is on screen,
        local files included), else the fetch-date field. None (with the
        reason shown) when neither pins a date."""
        if self.prep is not None and getattr(self.prep, "ut_stamp", None):
            return self.prep.ut_stamp
        if self.mode_fetch.isChecked():
            return self.fetch_date.date().toString("yyyyMMdd")
        return None

    def _fetch_gfs_winds(self):
        if self._winds_worker is not None:
            return                        # one in flight already
        ymd = self._gfs_winds_ymd()
        if ymd is None:
            self.winds_status.setText(
                "no night to fetch winds for — Run first, or pick a "
                "fetch date")
            return
        # weight the free-atm collapse by the night's OWN median profile
        # when one is loaded; else the measured MK median shape
        fa_weights = None
        profiles = getattr(self.prep, "profiles", None) if self.prep else None
        if profiles:
            fa_weights = np.median(
                np.array([p[3] for p in profiles], dtype=float), axis=0)
        self.winds_fetch_btn.setEnabled(False)
        self.winds_status.setText(f"fetching GFS winds for {ymd}…")
        w = GfsWindsWorker(ymd, engine.DEF_CACHE_DIR, fa_weights, self)
        w.done.connect(lambda res, err, ymd=ymd, own=fa_weights is not None:
                       self._on_gfs_winds_done(res, err, ymd, own))
        self._winds_worker = w
        w.start()

    def _on_gfs_winds_done(self, result, err, ymd, own_profile):
        self._winds_worker = None
        self.winds_fetch_btn.setEnabled(True)
        if result is None:
            # a failed fetch changes NOTHING -- the spinboxes keep whatever
            # winds they had (and the advisory keeps its last good winds),
            # and the reason is shown right here
            self.winds_status.setText(f"⚠ {err} — winds unchanged")
            return
        self._gfs_winds_result = result
        # setValue on these fires their valueChanged -> _on_compute_changed,
        # so the estimate re-runs with the real winds automatically
        self.wind_ground.setValue(result["v_ground"])
        self.wind_free.setValue(result["v_free"])
        src = ("night-median Cn² weights" if own_profile
               else "MK median profile weights")
        self.winds_status.setText(
            f"GFS {ymd}: ground {result['v_ground']:g}, free-atm "
            f"{result['v_free']:g} m/s (Cn²-wtd, {result['n_hours']} h "
            f"{result['hours']})")
        self.winds_status.setToolTip(
            "per-bin medians (km above summit → m/s): "
            + ", ".join(f"{h:g}→{v:g}" for h, v in result["per_bin"])
            + f"; free-atm collapse: {src}")
        # the recompute the setValue calls above scheduled will refresh the
        # advisory too, but only if the values actually CHANGED -- refresh
        # explicitly so a re-fetch landing identical winds still updates it
        self._update_fa_advisory()

    # ---- FA timing advisory (display-only) ------------------------------
    def _update_fa_advisory(self):
        """Refresh the FA-advisory label. Reads res/winds, alters NOTHING
        else -- by design this never feeds back into the estimate or the
        timeline. Reference time: the Nighttime-mode last pull when one
        exists (the 'now' of a live night), else the newest MASS sample."""
        if not hasattr(self, "fa_advisory"):
            return
        if self.res is None:
            self.fa_advisory.setText("—")
            return
        if len(self.res.p_times) == 0:
            self.fa_advisory.setText("no MASS data this night")
            return
        t_ref = getattr(self, "_nighttime_last_pull", None) \
            or self.res.p_times[-1]
        stats = engine.trailing_fa_stats(self.res.p_times, self.res.col_mass,
                                         t_ref)
        ref_lbl = self._fmt_hm(t_ref)
        if stats is None:
            txt = f"no MASS samples in the 40 min before {ref_lbl}"
        elif stats["volatile"]:
            txt = (f"⚠ FA event in progress at {ref_lbl}: "
                   f"{stats['lo']:.2f}–{stats['hi']:.2f}″ over 40 min "
                   f"(median {stats['med']:.2f}″) — beam vs monitor timing "
                   f"can differ by 10–40 min")
        else:
            txt = (f"FA steady at {ref_lbl}: {stats['lo']:.2f}–"
                   f"{stats['hi']:.2f}″ (median {stats['med']:.2f}″) "
                   f"last 40 min")
        w = self._gfs_winds_result
        if w and w.get("bins_full") and self.target_enable.isChecked():
            eff = self._effective_target_coords()
            if eff is not None:
                try:
                    _am, elv, azv = engine.compute_airmass_curve(
                        eff[0], eff[1], [t_ref])
                    el, az = float(elv[0]), float(azv[0])
                except Exception:
                    el, az = -90.0, 0.0
                if el > 5.0:
                    # upper bins only: below 4 km the pierce-point
                    # separations are too small to matter
                    lead = engine.event_lead_lag(
                        az, el, [(h, v, d) for h, v, d in w["bins_full"]
                                 if h >= 4000.0])
                    if lead:
                        parts = " · ".join(
                            f"{h:g} km {c:+.0f}±{r:.0f}" for h, c, r in lead)
                        txt += (f".  Event lead vs monitor (az {az:.0f}° "
                                f"el {el:.0f}°): {parts} min (+ = Keck first)")
        self.fa_advisory.setText(txt)

