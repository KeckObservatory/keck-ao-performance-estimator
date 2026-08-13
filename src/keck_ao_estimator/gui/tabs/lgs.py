"""LGS tab: the tip-tilt sensor (STRAP/TRICK), TT-star magnitude/offset,
LGS/asterism offset, LTAO bandwidth-floor fraction, and the wind-weighted-
bandwidth preview plot.
"""
import numpy as np
import astropy.units as u
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from qtcompat import QtWidgets

import keck_ao_estimator as engine

from ..widgets import OffsetEntry, _dspin


class LgsTabMixin:
    def _tab_budget(self):
        w = QtWidgets.QWidget()
        f = QtWidgets.QFormLayout(w)
        # tip-tilt sensor: STRAP (R quadcell) or the K1 IR TRICK sensor (H/K).
        # K1-only; on K2 it is greyed to STRAP. TRICK swaps the science band
        # (dichroic) and reinterprets the TT-mag as the H/K guide magnitude.
        self.tt_sensor = QtWidgets.QComboBox()
        self.tt_sensor.addItems(["STRAP (R)", "TRICK (H)", "TRICK (K)"])
        self.tt_sensor.setToolTip(
            "Tip-tilt sensor. STRAP = R-band quadcell with the refined "
            "measurement row (recalibrated 2026-07 to paired on-sky "
            "STRAP/TRICK data). TRICK = the K1 "
            "IR sensor in H or K (holds tip-tilt to a much fainter guide "
            "star); the TRICK/OSIRIS dichroic ties the science band to the "
            "OTHER of H/K. TRICK is K1-only.")
        self.tt_mag = _dspin(0, 25, 0.1, self.defaults.tt_mag, 1, " mag")
        self.tt_mag_label = QtWidgets.QLabel("TT-star R mag:")
        self.tt_offset = OffsetEntry(self.defaults.tt_offset,
                                     self._effective_science_coords,
                                     fixable=True)
        # short checkbox text: the old "override (else per-telescope default)"
        # made this the widest field in the form and (with the label column)
        # pushed the tab's minimum width past the panel -> horizontal
        # scrollbar; the parenthetical lives in the tooltip instead
        self.lgs_offset_enable = QtWidgets.QCheckBox("override")
        self.lgs_offset_enable.setToolTip(
            "Override the LGS offset; unchecked uses the per-telescope "
            "default.")
        self.lgs_offset = _dspin(0, 120, 0.5, 0.0, 1, '"')
        self.lgs_offset.setEnabled(False)
        self.lgs_offset_enable.toggled.connect(self.lgs_offset.setEnabled)
        lgs_row = QtWidgets.QHBoxLayout()
        lgs_row.addWidget(self.lgs_offset_enable)
        lgs_row.addWidget(self.lgs_offset)
        # laser position angle for the field map: the laser sits radially at
        # the LGS-offset magnitude. PA (N->E) defaults to the K1 pointing-
        # offset campaign's measured direction (engine.DEF_LASER_PA_DEG =
        # 254.8 deg = 4.8" W, 1.3" S in the standard north-up/east-left
        # configuration; it used to be a placeholder 225 deg = radial SW,
        # from when the direction was genuinely unknown). Does not affect the
        # science-direction estimate -- that term is radial -- only where the
        # laser is drawn / evaluated on the field map. Meaningless on K2,
        # whose laser is not offset at all (DEF_LGS_OFFSET["K2"] = 0").
        self.laser_pa = _dspin(0, 360, 5, engine.DEF_LASER_PA_DEG, 1, "°")
        self.laser_pa.setToolTip(
            "Laser position angle (deg, North->East) for the field map. "
            f"Default {engine.DEF_LASER_PA_DEG:g}° = the K1 campaign "
            "direction (4.8\" W, 1.3\" S of the pointing origin, north-up/"
            "east-left). A bench stage-alignment property, so it is stable "
            "night to night. Not used by the budget, which is radial.")
        # "fix to base": anchor the laser's ABSOLUTE sky position (from the
        # current LGS-offset magnitude + laser PA) so it doesn't silently
        # follow a target-offset exploration -- same idea as the TT/NGS
        # offsets' fix_to_base, but hand-rolled since lgs_offset/laser_pa are
        # a plain magnitude+PA pair, not an OffsetEntry (angular anisoplanatism
        # is radially symmetric, so only the CLI/engine-facing magnitude, not
        # a direction, is normally needed).
        self._laser_anchor = None
        self._laser_refreshing = False
        self.laser_fix_to_base = QtWidgets.QCheckBox("fix to base position")
        self.laser_fix_to_base.setToolTip(
            "Keep the laser's ABSOLUTE sky position fixed as the Target tab's "
            "offset changes, instead of silently following it. Recomputes both "
            "the LGS-offset magnitude (engine-facing) and the laser PA.")
        self.ltao_floor = _dspin(0, 1, 0.05, self.defaults.ltao_bw_floor_frac, 2)
        self.ltao_tt_gain = _dspin(1.0, 3.0, 0.05,
                                   self.defaults.ltao_tt_theta0_gain, 2, "×")
        self.ltao_tt_gain.setToolTip(
            "Effective tilt-θ₀ gain from tomography, applied as "
            "(1/gain)^(5/6) on the TT-star tilt-anisoplanatism rows AND the "
            "angular-aniso charge at the laser/asterism-centre offset, LTAO "
            "mode ONLY — single-beacon and the legacy budget are untouched. "
            "Default 1.0 = disabled (KAON 1303 §5.5): with a SINGLE TT star "
            "the LGS-tomography null modes — field-varying tilt from "
            "focus/astigmatism aloft — cannot be estimated, so those errors "
            "are charged in full, same as single-conjugate; the tomographic "
            "reduction only appears with ≥2 TT stars (descoped). Raise "
            "above 1.0 only for a multi-star mode or a contrary on-sky "
            "ladder.")
        self.legacy_cb = QtWidgets.QCheckBox("legacy budget")
        self.legacy_cb.setChecked(bool(self.defaults.legacy_budget))
        self.tomo_combo = QtWidgets.QComboBox()
        self.tomo_combo.addItems(["auto (per telescope)", "on", "off"])
        f.addRow("TT sensor:", self.tt_sensor)
        f.addRow(self.tt_mag_label, self.tt_mag)
        f.addRow("TT-star offset:", self.tt_offset)
        f.addRow("LGS offset:", self._wrap(lgs_row))
        f.addRow("Laser PA (field map):", self.laser_pa)
        f.addRow("", self.laser_fix_to_base)
        f.addRow("LTAO bw floor frac:", self.ltao_floor)
        f.addRow("LTAO TT θ₀ gain:", self.ltao_tt_gain)
        f.addRow("Tomography:", self.tomo_combo)
        f.addRow("", self.legacy_cb)

        # layer-mismatch m readout (updated after each run) -- how far the
        # night's aloft Cn2 profile sits from the K1 reconstructor's static
        # prior; drives the LTAO tomography penalty (0 = matches prior, 1 = no
        # tomographic benefit).
        self.m_label = QtWidgets.QLabel("run to compute")
        self.m_label.setWordWrap(True)
        self.m_label.setToolTip(
            "Layer-mismatch m = total-variation distance between the night's "
            "normalized aloft MASS profile and the reconstructor's static "
            "prior. The LTAO altitude term ramps alt² = tomo² + m²·(cone²−tomo²), "
            "so m=0 gives full tomographic gain, m=1 none.")
        self.m_label.setStyleSheet("QLabel { color:#333; background:#f4f0e8; "
                                   "padding:5px; border:1px solid #ddd; }")
        f.addRow("Layer mismatch m:", self.m_label)

        # the night's REAL Cn² profile, for the field-map conditions (whole
        # night / observing-window mean, or the exact profile at a set time)
        self.lgs_prof_fig = Figure(figsize=(3.4, 3.0))
        self.lgs_prof_canvas = FigureCanvas(self.lgs_prof_fig)
        self.lgs_prof_canvas.setMinimumHeight(230)
        self.lgs_prof_canvas.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding)
        f.addRow(self.lgs_prof_canvas)
        _pa = self.lgs_prof_fig.add_subplot(111); _pa.axis("off")
        _pa.text(0.5, 0.5, "Run to see the night's Cn² profile",
                 ha="center", va="center", fontsize=9, color="#777")

        # TT star + LGS offset are compute-only -> live recompute; the LTAO
        # TT theta0 gain enters lgs_strehl the same way (and moves the LTAO
        # field map, so it marks that stale too)
        for sp in (self.tt_mag, self.lgs_offset, self.ltao_tt_gain):
            sp.valueChanged.connect(self._on_compute_changed)
        self.ltao_tt_gain.valueChanged.connect(self._on_fieldmap_input_changed)
        # a manual tt_mag edit means it's no longer just a catalogue star's
        # band swapped through Maréchal -- stop treating it as one
        self.tt_mag.valueChanged.connect(self._on_tt_mag_user_edit)
        self.lgs_offset_enable.toggled.connect(self._on_compute_changed)
        self.tt_offset.changed.connect(self._on_compute_changed)
        self.tt_offset.changed.connect(self._validate)
        self.tt_offset.changed.connect(self._on_fieldmap_input_changed)
        self.tt_offset.pos_changed.connect(self._on_fieldmap_input_changed)
        self.laser_pa.valueChanged.connect(self._on_fieldmap_input_changed)
        # LGS offset also moves the laser on the field map (previously only
        # laser PA did -- the magnitude was live on the main panel but never
        # marked the map stale, so it never visibly updated there)
        self.lgs_offset.valueChanged.connect(self._on_fieldmap_input_changed)
        self.lgs_offset_enable.toggled.connect(self._on_fieldmap_input_changed)
        # laser "fix to base": re-anchor on a genuine edit of magnitude/PA/
        # override-toggle while fixed; the checkbox itself sets or clears the
        # anchor outright
        for sig in (self.lgs_offset.valueChanged, self.laser_pa.valueChanged,
                   self.lgs_offset_enable.toggled):
            sig.connect(self._laser_on_edit)
        self.laser_fix_to_base.toggled.connect(self._laser_on_fix_toggled)
        # legacy / LTAO-floor / tomography feed prepare_night -> live re-prepare
        self.legacy_cb.toggled.connect(self._on_prep_changed)
        self.ltao_floor.valueChanged.connect(self._on_prep_changed)
        self.tomo_combo.currentTextChanged.connect(self._on_prep_changed)
        # the TT sensor swaps the science band -> re-prepare + relabel
        self.tt_sensor.currentTextChanged.connect(self._on_tt_sensor_changed)
        self.tel_k1.toggled.connect(self._sync_tt_sensor_for_tel)
        self._sync_tt_sensor_for_tel()
        return self._scroll(w)

    # map the sensor combo <-> engine value; the science-band complement (K1
    # dichroic) each TRICK mode forces
    _TT_SENSOR_MAP = {"STRAP (R)":        ("strap", "R", None),
                      "TRICK (H)":        ("trick-h", "H", "K"),
                      "TRICK (K)":        ("trick-k", "K", "H")}

    def _on_tt_sensor_changed(self, *_):
        """Relabel the TT-mag field to the sensing band, and (TRICK on K1) swap
        the science band to the dichroic complement and lock it."""
        _base, wfs, sci = self._TT_SENSOR_MAP[self.tt_sensor.currentText()]
        self.tt_mag_label.setText(f"TT-star {wfs} mag:")
        if sci is not None and not self.wl_enable.isChecked():
            self.band_combo.blockSignals(True)
            self.band_combo.setCurrentText(sci)
            self.band_combo.blockSignals(False)
        # TRICK pins the science band (dichroic); STRAP frees it
        self.band_combo.setEnabled(sci is None or self.wl_enable.isChecked())
        # only on an actual sensing-BAND change (R<->H<->K) -- a star's
        # R/H/K magnitudes are never the same, so the old number must not
        # silently carry over
        if wfs != getattr(self, "_tt_wfs_band", None) and not self._loading:
            self._sync_tt_mag_for_band(wfs)
            # a star's R/H/K magnitudes differ, so any existing guide-star
            # ranking (computed for the OLD sensing band) is now wrong
            self._invalidate_gs_ranking()
        self._tt_wfs_band = wfs
        if not self._loading:
            self._on_prep_changed()

    def _sync_tt_mag_for_band(self, band):
        """Re-derive the TT-star magnitude in the sensor's new working band
        from the catalogue star currently backing the TT selection (set by
        FieldMapOverlaysMixin._fm_select_star), rather than silently leaving
        the old band's number in place. Warns instead when there's no
        tracked catalogue star, or it has no derivable magnitude in the new
        band -- a real star's R/H/K magnitudes are never numerically equal,
        so a stale value must be flagged, not left looking valid."""
        status = getattr(self, "fm_catalog_status", None)
        if status is None or not hasattr(self, "_tt_star_ref"):
            return                      # field-map tab not built yet (startup)
        star = self._tt_star_ref
        if star is None:
            status.setText(
                f"TT-star switched to {band} band -- no catalogue star "
                f"selected, so the magnitude was NOT updated; a star's R/H/K "
                f"magnitudes differ, please check/re-enter it")
            return
        mag, kind, label = engine.estimate_sensing_mag(star["mags"], band)
        if mag is None:
            status.setText(
                f"TT-star switched to {band} band -- "
                f"{self._catalog_name or 'the catalogue'} star "
                f"“{star['id']}” has no derivable {band} magnitude; "
                f"the old value was NOT updated, please check/re-enter it")
            return
        self._tt_mag_auto = True
        self.tt_mag.setValue(mag)
        self._tt_mag_auto = False
        if kind == "exact":
            status.setText(f"TT-star magnitude updated for {band}: {mag:.1f}")
        elif kind == "est":
            status.setText(f"TT-star magnitude updated: {band}≈{mag:.1f} "
                           f"(estimated from {label})")
        else:
            status.setText(f"TT-star magnitude updated: {band}≈{mag:.1f} "
                           f"(rough — {label})")

    def _on_tt_mag_user_edit(self, *_):
        """A genuine user edit of the TT magnitude (not our own re-derivation
        or a catalogue selection, both of which set _tt_mag_auto) means it's
        no longer tied to a catalogue star -- stop tracking one, so a later
        sensor switch warns instead of overwriting the user's own number."""
        if not getattr(self, "_tt_mag_auto", False) and not self._loading:
            self._tt_star_ref = None

    def _sync_tt_sensor_for_tel(self, *_):
        """TRICK is a K1 instrument: on K2 grey out the TRICK entries (and fall
        back to STRAP if one was selected); the STRAP refined/legacy choice
        stays available on both telescopes."""
        k1 = self.tel_k1.isChecked()
        if not k1 and self.tt_sensor.currentText().startswith("TRICK"):
            self.tt_sensor.setCurrentText("STRAP (R)")
        model = self.tt_sensor.model()
        for i in range(self.tt_sensor.count()):
            it = model.item(i)
            if it.text().startswith("TRICK"):
                it.setEnabled(k1)
        self._on_tt_sensor_changed()

    def _laser_absolute(self, target_coord):
        """Absolute SkyCoord for the CURRENT LGS-offset magnitude + laser PA,
        applied to target_coord. Same (x=West+, y=North+) plot-frame
        convention as _laser_xy()/_mark_ref, converted to spherical_offsets_by's
        (d_lon=East+, d_lat=North+): d_lon = -x = r*sin(pa), d_lat = y = r*cos(pa)."""
        tel = "K1" if self.tel_k1.isChecked() else "K2"
        r = (self.lgs_offset.value() if self.lgs_offset_enable.isChecked()
             else engine.DEF_LGS_OFFSET[tel])
        pa = np.radians(self.laser_pa.value())
        return target_coord.spherical_offsets_by(
            r * np.sin(pa) * u.arcsec, r * np.cos(pa) * u.arcsec)

    def _laser_on_fix_toggled(self, checked):
        if not checked:
            self._laser_anchor = None
            return
        eff = self._effective_target_coords()
        if eff is None:
            self._laser_anchor = None
            self.laser_fix_to_base.blockSignals(True)
            self.laser_fix_to_base.setChecked(False)
            self.laser_fix_to_base.blockSignals(False)
            return
        try:
            self._laser_anchor = self._laser_absolute(engine.parse_radec(*eff))
        except Exception:
            self._laser_anchor = None

    def _laser_on_edit(self, *_):
        """A genuine user edit of magnitude/PA/override-toggle while fixed
        re-anchors to the newly-entered absolute position (mirrors
        OffsetEntry._recompute's re-anchor-on-edit; the _laser_refreshing
        guard keeps _laser_refresh_from_base's own writes from re-deriving
        the anchor from their own output)."""
        if (getattr(self, "laser_fix_to_base", None) is not None
                and self.laser_fix_to_base.isChecked()
                and not self._laser_refreshing):
            eff = self._effective_target_coords()
            if eff is not None:
                try:
                    self._laser_anchor = self._laser_absolute(
                        engine.parse_radec(*eff))
                except Exception:
                    pass

    def _laser_refresh_from_base(self):
        """Re-derive the LGS-offset magnitude + laser PA from the CURRENT
        effective target so a fixed (anchored) laser position is preserved.
        Call whenever the target/target-offset may have moved. No-op unless
        laser_fix_to_base is checked or the anchor/current target is
        unavailable."""
        if not (getattr(self, "laser_fix_to_base", None) is not None
                and self.laser_fix_to_base.isChecked()):
            return
        if self._laser_anchor is None:
            return
        eff = self._effective_target_coords()
        if eff is None:
            return
        try:
            target = engine.parse_radec(*eff)
            dlon, dlat = target.spherical_offsets_to(self._laser_anchor)
        except Exception:
            return
        r_new = float(np.hypot(dlon.arcsec, dlat.arcsec))
        pa_new = float(np.degrees(np.arctan2(dlon.arcsec, dlat.arcsec))) % 360.0
        self._laser_refreshing = True
        try:
            for wdg in (self.lgs_offset_enable, self.lgs_offset, self.laser_pa):
                wdg.blockSignals(True)
            self.lgs_offset_enable.setChecked(True)
            self.lgs_offset.setEnabled(True)
            self.lgs_offset.setValue(r_new)
            self.laser_pa.setValue(pa_new)
            for wdg in (self.lgs_offset_enable, self.lgs_offset, self.laser_pa):
                wdg.blockSignals(False)
        finally:
            self._laser_refreshing = False

    # Row grouping for the WFE sliders tab. Every ADJUSTABLE_BUDGET_PARAMS name
    # appears exactly once. LGS-only terms (NGS_LGS_ONLY_TERMS) are grouped by
    # that fact first; of the remaining (common, NGS-projecting) terms, the
    # seeing-independent ones (WFE_SCALING == "fixed") get their own group.
    # (title, tooltip, names) -- the parentheticals live in the tooltip: a
    # QGroupBox's minimum width includes its full unwrappable title, and a
    # long title alone forced a horizontal scrollbar on the panel (631045c)
    WFE_GROUPS = [
        ("Fixed terms", "", ["HOMEAS", "MARGIN"]),
        ("Static / calibration terms",
         "broken out; project onto NGS too",
         ["STATIC_TEL_K1", "STATIC_TEL_K2", "STATIC_CALIB", "STATIC_DM",
          "STATIC_INST", "STATIC_REG"]),
        ("Common terms",
         "project onto NGS too",
         ["FITTING_ERR_K1", "FITTING_ERR_K2", "BW_REF", "SCINT_REF", "ANG_REF"]),
        ("LGS-only terms",
         "do not project onto NGS",
         ["FA_REF", "NAFOC", "TOMO_ERR"]),
    ]

